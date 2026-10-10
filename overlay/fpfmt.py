# SPDX-License-Identifier: Apache-2.0
"""Bit-exact numpy models of the small float formats used here, plus a dependency-free safetensors reader/writer.

numpy only (no torch, no ml_dtypes), so the converters and the CPU tests run on the Mac with
    uv run --no-project --with numpy python ...

Formats
  bf16      1-8-7, round-to-nearest-even from fp32 (what torch .to(bfloat16) and Triton .to(tl.bfloat16) do)
  e4m3fn    1-4-3, bias 7, no inf, max 448, subnormal quantum 2**-9 (torch.float8_e4m3fn, OCP FP8 E4M3)
  e2m1      1-2-1, magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} (OCP MX FP4, NVFP4 element)
  e8m0      biased power-of-two exponent byte (OCP MX shared scale); 0 decodes to 0.0 here, see decode_e8m0
All rounding is round-to-nearest, ties-to-even, which is what the torch casts use.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

# ------------------------------------------------------------------------------------------------
# bf16
# ------------------------------------------------------------------------------------------------


def bf16_bits(x) -> np.ndarray:
    """fp32 -> bf16 bit pattern (uint16), RNE; NaN stays NaN."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    r = np.where(nan, (b >> 16) | 0x40, r)
    return r.astype(np.uint16)


def bf16_from_bits(u) -> np.ndarray:
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(x) -> np.ndarray:
    """fp32 -> nearest bf16 value, returned as fp32 (exact)."""
    return bf16_from_bits(bf16_bits(x))


# ------------------------------------------------------------------------------------------------
# e4m3fn
# ------------------------------------------------------------------------------------------------
E4M3_MAX = 448.0


def e4m3_round(x) -> np.ndarray:
    """Nearest e4m3fn value (as float64), RNE. |x| > 448 after rounding raises (torch would give NaN)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    _, e = np.frexp(ax)  # ax = m * 2**e, m in [0.5, 1)
    E = np.maximum(e - 1, -6)  # normal exponent, or the subnormal floor
    q = np.exp2(E - 3.0)
    r = np.rint(ax / q) * q
    if np.any(r > E4M3_MAX):
        raise ValueError("e4m3 overflow (value above 448 after rounding)")
    return np.copysign(r, x)


def e4m3_encode(v) -> np.ndarray:
    """e4m3-representable values -> bytes (uint8). Call e4m3_round first."""
    v = np.asarray(v, dtype=np.float64)
    s = (np.signbit(v) & (v != 0)).astype(np.uint8) << 7
    a = np.abs(v)
    out = np.zeros(a.shape, dtype=np.uint8)
    sub = (a > 0) & (a < 2.0 ** -6)
    out[sub] = np.rint(a[sub] / 2.0 ** -9).astype(np.uint8)
    nrm = a >= 2.0 ** -6
    _, e = np.frexp(a[nrm])
    E = e - 1
    mant = np.rint(a[nrm] / np.exp2(E - 3.0)).astype(np.int64) - 8
    out[nrm] = ((E + 7).astype(np.uint8) << 3) | mant.astype(np.uint8)
    return out | s


def e4m3_decode(b) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8)
    s = np.where(b & 0x80, -1.0, 1.0)
    ex = (b >> 3) & 0xF
    m = (b & 7).astype(np.float64)
    val = np.where(ex == 0, m * 2.0 ** -9, (1 + m / 8) * np.exp2(ex.astype(np.float64) - 7))
    val = np.where((ex == 15) & ((b & 7) == 7), np.nan, val)
    return s * val


# ------------------------------------------------------------------------------------------------
# e2m1 (FP4)
# ------------------------------------------------------------------------------------------------
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
_E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_encode(x) -> np.ndarray:
    """Nearest e2m1 code (uint8, sign in bit 3), ties to the even code, saturating at 6."""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), E2M1_MAX)
    code = np.zeros(a.shape, dtype=np.int64)
    for i, b in enumerate(_E2M1_MID):
        # boundary between code i and i + 1; on a tie take the even code
        code += (a > b) if i % 2 == 0 else (a >= b)
    sign = (np.signbit(x) & (code != 0)).astype(np.int64)
    return (code | (sign << 3)).astype(np.uint8)


def e2m1_decode(c) -> np.ndarray:
    c = np.asarray(c, dtype=np.uint8)
    return np.where(c & 8, -1.0, 1.0) * E2M1[(c & 7).astype(np.int64)]


def pack_nibbles(codes) -> np.ndarray:
    """[..., 2n] uint8 codes -> [..., n] bytes; element 2i in the low nibble (ModelOpt / compressed-tensors)."""
    c = np.asarray(codes, dtype=np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_nibbles(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


# ------------------------------------------------------------------------------------------------
# e8m0
# ------------------------------------------------------------------------------------------------


def decode_e8m0(b) -> np.ndarray:
    """Biased exponent byte -> 2**(b - 127) as float64. Byte 0 decodes to 0.0 here: that is what the kernels in
    this directory build (fp32 bits b << 23), and the table builders treat such blocks as unsafe."""
    b = np.asarray(b, dtype=np.int64)
    return np.where(b == 0, 0.0, np.exp2(b.astype(np.float64) - 127.0))


def round_up_f32(x) -> np.ndarray:
    """float64 -> the smallest fp32 >= x (so a stored norm never under-states the float64 value)."""
    x = np.asarray(x, dtype=np.float64)
    f = x.astype(np.float32)
    bump = f.astype(np.float64) < x
    f[bump] = np.nextafter(f[bump], np.float32(np.inf))
    return f


# ------------------------------------------------------------------------------------------------
# safetensors (numpy only)
# ------------------------------------------------------------------------------------------------
_ST_DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "F64": (np.float64, 8),
          "U8": (np.uint8, 1), "I8": (np.int8, 1), "I16": (np.int16, 2), "I32": (np.int32, 4),
          "I64": (np.int64, 8), "F8_E4M3": (np.uint8, 1), "BOOL": (np.bool_, 1)}


class SafeTensors:
    """Lazy reader. get(name) returns a numpy view (BF16/F8 as their raw uint16/uint8 bits);
    get_f32(name) upcasts BF16/F16/F32 to float32. Rows can be sliced without reading the whole tensor."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.meta = self.header.pop("__metadata__", None)

    def keys(self):
        return list(self.header)

    def info(self, name):
        h = self.header[name]
        return h["dtype"], tuple(h["shape"])

    def get(self, name, rows=None):
        h = self.header[name]
        dt, _ = _ST_DT[h["dtype"]]
        s, _ = h["data_offsets"]
        shape = tuple(h["shape"])
        mm = np.memmap(self.path, dtype=dt, mode="r", offset=self.base + s, shape=shape if shape else (1,))
        if not shape:
            return np.array(mm[0])
        return np.array(mm[rows[0]:rows[1]] if rows is not None else mm)

    def get_f32(self, name, rows=None):
        dtype, _ = self.info(name)
        a = self.get(name, rows)
        if dtype == "BF16":
            return bf16_from_bits(a)
        if dtype in ("F16", "F32", "F64"):
            return a.astype(np.float32)
        raise TypeError(f"{name}: {dtype} is not a float tensor")


def save_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> None:
    """tensors: name -> (st_dtype, numpy array holding the raw element bits / values)."""
    header, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        npdt, size = _ST_DT[dt]
        a = np.asarray(arr).astype(npdt, copy=False)  # (ascontiguousarray would turn a 0-d scalar into [1])
        if not a.flags.c_contiguous:
            a = a.copy(order="C")
        raw = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)

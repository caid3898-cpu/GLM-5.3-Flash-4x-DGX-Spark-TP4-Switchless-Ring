# SPDX-License-Identifier: Apache-2.0
"""4-bit drafter pieces for DFlash2 on the Tony v11 image (vLLM 0.1.dev20051+g487ecf187). All default OFF.

GLM_DS_DRAFT_NVFP4=1          the drafter checkpoint was re-encoded by drafter_nvfp4.py (ModelOpt W4A16_NVFP4:
                              packed e2m1 + fp8 scale per 16 + fp32 global scale, W4A16 Marlin). Installs the
                              context-KV fix: DFlashQwen3Model._build_context_kv_buffers concatenates
                              qkv_proj.weight[q_size:] of every layer into one BF16 GEMM weight at the end of
                              load_weights, before process_weights_after_loading; with NVFP4 that tensor is packed
                              uint8. The k/v rows are dequantized here (code x fp8 scale x max(global scale over
                              q/k/v), the value Marlin uses) and rounded once to BF16. Same role as
                              GLM_LV_DRAFT_FP8_KV for the block-FP8 drafter; the two hooks compose.
GLM_DS_DRAFT_HEAD_FP4=0|1|check
                              DFlash2 borrows the target's vocab-parallel lm_head (BF16 38720 x 4096 per rank) for
                              its top-16 candidates (compute_candidates). 1: an NVFP4 copy (e2m1, e4m3 scale per 16
                              weights, power-of-two global scale, 89.2 MB instead of 158.8 MB for the fp8 twin)
                              and a Triton kernel produce the draft logits. Drafts only: the target verifies every
                              draft, so committed tokens are unchanged; acceptance can move. With
                              GLM_DS_DRAFT_HEAD_FP8=1 as well, this wrapper sits outside the fp8 one: mode 0 (per
                              call, glm_ab) falls back to the fp8 head, so the two can be A/B-ed in one boot.
                              check: proposals from the stock / fp8 head, the NVFP4 head computed alongside, and
                              device counters (top-1 agreement, stock top-1 inside the NVFP4 top-16, identical
                              top-16 sets) printed by rank 0 every GLM_DS_DRAFT_HEAD_LOG_EVERY proposals.
GLM_DS_DRAFT_HEAD_MAX_M=64    more draft rows than this keep the previous head
GLM_DS_DRAFT_HEAD_DUMP=DIR    save compute_candidates inputs (draft hidden rows, bf16) for gpu_test_drafter_fp4.py;
                              only eager calls reach Python, so boot the capture run with --enforce-eager
GLM_DS_DRAFT_HEAD_DUMP_ROWS=20000
The same compute_candidates wrapper serves an MXINT8 copy that glm_cert_head.py hands over with
GLM_CERT_HEAD_DRAFT=1 (one 1-byte copy of the head for target screening and drafting).

Credits: NVFP4 / ModelOpt checkpoint conventions (NVIDIA TensorRT Model Optimizer); Marlin mixed-precision GEMM
(Frantar, Castro, Chen, Hoefler, Alistarh, IST-DASLab; FP4 Marlin in vLLM); OCP Microscaling Formats v1.0 (e2m1,
e8m0). The draft-head wrapper follows our glm_ds_draft.py (GLM_DS_DRAFT_HEAD_FP8). incoai DFlash2 is CC BY-NC-ND:
any re-encoded copy stays local.
"""
from __future__ import annotations

import math
import os
import sys

_OFF = ("", "0", "off", "false", "no")
NVFP4_KV = os.environ.get("GLM_DS_DRAFT_NVFP4", "0").strip().lower() not in _OFF
HEAD_MODE = os.environ.get("GLM_DS_DRAFT_HEAD_FP4", "0").strip().lower()
HEAD_MODE = "0" if HEAD_MODE in _OFF else ("check" if HEAD_MODE == "check" else "1")
MAX_M = min(64, int(os.environ.get("GLM_DS_DRAFT_HEAD_MAX_M", "64") or 64))
LOG_EVERY = int(os.environ.get("GLM_DS_DRAFT_HEAD_LOG_EVERY", "2000"))
DUMP = os.environ.get("GLM_DS_DRAFT_HEAD_DUMP", "").strip()
DUMP_ROWS = int(os.environ.get("GLM_DS_DRAFT_HEAD_DUMP_ROWS", "20000"))
SHARE_FROM_CERT = os.environ.get("GLM_CERT_HEAD_DRAFT", "0").strip().lower() not in _OFF
FP8_HEAD = os.environ.get("GLM_DS_DRAFT_HEAD_FP8", "0").strip().lower() not in _OFF

E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _log(msg):
    print(f"glm-ds-draft-fp4: {msg}", file=sys.stderr, flush=True)


def _head_mode() -> str:
    """Per call: the in-boot A/B harness (glm_ab.py, TEST ONLY, with GLM_AB_DRAFT_SETS=1 so every variant gets its
    own drafter graphs) can switch it; otherwise the value read at import."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and "GLM_DS_DRAFT_HEAD_FP4" in getattr(ab, "KNOWN", {}):
        return str(ab.norm_value("GLM_DS_DRAFT_HEAD_FP4", ab.env("GLM_DS_DRAFT_HEAD_FP4")))
    return HEAD_MODE


# ------------------------------------------------------------------------------------------
# NVFP4 copy of a BF16 matrix (torch, on the device) and its reference dequantization
# ------------------------------------------------------------------------------------------
def e2m1_encode_t(v):
    """torch: nearest e2m1 code, ties to the even code, saturating (same rule as common/fpfmt.py)."""
    import torch
    a = v.abs().clamp(max=6.0)
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75).to(torch.uint8) + (a > 1.25).to(torch.uint8)
            + (a >= 1.75).to(torch.uint8) + (a > 2.5).to(torch.uint8) + (a >= 3.5).to(torch.uint8)
            + (a > 5.0).to(torch.uint8))
    sign = ((v < 0) & (code != 0)).to(torch.uint8) << 3
    return code | sign


def e2m1_decode_t(c):
    import torch
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=c.device)
    mag = lut[(c & 7).long()]
    return torch.where((c & 8) != 0, -mag, mag)


def make_nvfp4(w, chunk_rows: int = 2048):
    """bf16 [N, K] -> (packed uint8 [N, K/2] (element 2i low nibble), e4m3 scale [N, K/16], global scale float).
    Global scale is a power of two >= amax / (6 * 448), so code x scale x global is exact in bf16. Per block the
    scale is the nearest e4m3 to amax_b / 6 / global or the next e4m3 up, whichever reconstructs the block with
    less squared error (the next-up candidate avoids clipping the block maximum)."""
    import torch
    N, K = w.shape
    assert K % 16 == 0
    amax = float(w.abs().max().float())
    s2 = 2.0 ** math.ceil(math.log2(max(amax, 1e-30) / (6.0 * 448.0)))
    packed = torch.empty((N, K // 2), dtype=torch.uint8, device=w.device)
    scale = torch.empty((N, K // 16), dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, N, chunk_rows):
        r1 = min(N, r0 + chunk_rows)
        wf = w[r0:r1].float().view(r1 - r0, K // 16, 16)
        s_raw = (wf.abs().amax(-1) / 6.0 / s2).clamp(max=448.0)
        s_a = s_raw.to(torch.float8_e4m3fn)
        bits = s_a.view(torch.uint8)
        s_b = torch.where(bits < 0x7E, bits + 1, bits).view(torch.float8_e4m3fn)  # next e4m3 up (0x7E = 448)
        best_err = best_s = best_c = None
        for s8 in (s_a, s_b):
            sf = s8.float() * s2
            den = torch.where(sf > 0, sf, torch.ones_like(sf))
            c = e2m1_encode_t(wf / den[..., None])
            c = torch.where((sf > 0)[..., None], c, torch.zeros_like(c))
            err = (e2m1_decode_t(c) * sf[..., None] - wf).pow(2).sum(-1)
            if best_err is None:
                best_err, best_s, best_c = err, s8, c
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s8.view(torch.uint8), best_s.view(torch.uint8)).view(torch.float8_e4m3fn)
                best_c = torch.where(take[..., None], c, best_c)
        codes = best_c.view(r1 - r0, K)
        packed[r0:r1] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scale[r0:r1] = best_s
    return packed, scale, s2


def dequant_nvfp4(packed, scale, s2):
    """Reference: [N, K] fp32 = code x e4m3 scale x global (the Marlin W4A16 semantics)."""
    import torch
    N, Kh = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(N, Kh * 2)
    sc = scale.float().repeat_interleave(16, dim=1)
    return e2m1_decode_t(codes) * sc * float(s2)


# ------------------------------------------------------------------------------------------
# kernels: bf16 logits from an NVFP4 or MXINT8 copy
# ------------------------------------------------------------------------------------------
_KER = {}
_TILES = ((16, 16, 32, 256, 4, 3), (32, 32, 32, 128, 4, 3), (64, 64, 64, 128, 4, 3))  # max M, BM, BN, BK, warps, stages


def _kernels():
    if _KER:
        return _KER
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1(c):
        m = c & 7
        bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 126 << 23, 0))
        bits = bits | ((c & 8) << 28)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _head_nvfp4_kernel(X, P, S, Y, M, N, s2, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1) * BM + tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        nok = n < N
        mok = m < M
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            kb = k0 // 2 + tl.arange(0, BK // 2)  # byte index = element pair
            p = tl.load(P + n64[None, :] * (K // 2) + kb[:, None], mask=nok[None, :], other=0).to(tl.int32)
            sc = tl.load(S + n64[None, :] * (K // 16) + kb[:, None] // 8, mask=nok[None, :], other=0.0)
            sc = sc.to(tl.float32) * s2
            wlo = (_e2m1(p & 15) * sc).to(tl.bfloat16)   # element 2 * kb
            whi = (_e2m1(p >> 4) * sc).to(tl.bfloat16)   # element 2 * kb + 1
            xe = tl.load(X + m[:, None] * K + 2 * kb[None, :], mask=mok[:, None], other=0.0)
            xo = tl.load(X + m[:, None] * K + 2 * kb[None, :] + 1, mask=mok[:, None], other=0.0)
            acc = tl.dot(xe, wlo, acc)
            acc = tl.dot(xo, whi, acc)
        tl.store(Y + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=mok[:, None] & nok[None, :])

    @triton.jit
    def _head_mxint8_kernel(X, Q, E, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                            BK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1) * BM + tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        nok = n < N
        mok = m < M
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            w = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0)
            eb = tl.load(E + n64[None, :] * (K // 32) + k[:, None] // 32, mask=nok[None, :], other=127)
            sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            wd = (w.to(tl.float32) * sc).to(tl.bfloat16)
            x = tl.load(X + m[:, None] * K + k[None, :], mask=mok[:, None], other=0.0)
            acc = tl.dot(x, wd, acc)
        tl.store(Y + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=mok[:, None] & nok[None, :])

    _KER.update(triton=triton, nvfp4=_head_nvfp4_kernel, mxint8=_head_mxint8_kernel)
    return _KER


def head_logits(x, twin):
    """x [M, K] bf16 (M <= 64) -> bf16 local logits [M, N] from twin = ("nvfp4", packed, scale, s2) or
    ("mxint8", q, e)."""
    import torch
    k = _kernels()
    x = x.contiguous()
    M, K = x.shape
    fmt = twin[0]
    N = twin[1].shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    _, bm, bn, bk, warps, stages = next(t for t in _TILES if M <= t[0])
    grid = (k["triton"].cdiv(N, bn), k["triton"].cdiv(M, bm))
    if fmt == "nvfp4":
        k["nvfp4"][grid](x, twin[1], twin[2], y, M, N, float(twin[3]), K=K, BM=bm, BN=bn, BK=bk,
                         num_warps=warps, num_stages=stages)
    else:
        k["mxint8"][grid](x, twin[1], twin[2], y, M, N, K=K, BM=bm, BN=bn, BK=bk, num_warps=warps,
                          num_stages=stages)
    return y


class _TwinHeadMethod:
    def __init__(self, twin):
        self.twin = twin

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise RuntimeError("glm-ds-draft-fp4: unexpected embedding bias")
        return head_logits(x, self.twin)


class TwinHeadProxy:
    """Stands in for the vocab-parallel lm_head inside LogitsProcessor.get_top_k_tokens / _apply_head (they read
    .quant_method.apply, .shard_indices and .tp_size only)."""
    __slots__ = ("quant_method", "shard_indices", "tp_size", "_head")

    def __init__(self, head, twin):
        self.quant_method = _TwinHeadMethod(twin)
        self.shard_indices = head.shard_indices
        self.tp_size = head.tp_size
        self._head = head


# ------------------------------------------------------------------------------------------
# compute_candidates wrapper, twin build, check counters
# ------------------------------------------------------------------------------------------
class _Stats:
    dev = None       # int64 [4] on the device: rows, top-1 agree, stock top-1 in the NVFP4 top-k, identical top-k sets
    calls = 0
    dumped = 0


def _rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        return 0


def _dump(h):
    import torch
    if not DUMP or _Stats.dumped >= DUMP_ROWS or torch.cuda.is_current_stream_capturing():
        return
    os.makedirs(DUMP, exist_ok=True)
    _Stats.dumped += h.shape[0]
    torch.save(h.detach().to("cpu"), os.path.join(DUMP, f"draft-rank{_rank()}-{_Stats.dumped:08d}.pt"))


def install_dflash2_model(mod) -> None:
    import torch
    cls = mod.DFlash2Qwen3ForCausalLM
    if getattr(cls, "_glm_ds_fp4", False):
        return
    if FP8_HEAD and not getattr(cls, "_glm_ds_head_fp8", False):
        # the fp8 wrapper must be the inner one, or it would shadow this one whenever its twin exists
        raise RuntimeError("glm-ds-draft-fp4: register after glm_ds_hooks (GLM_DS_DRAFT_HEAD_FP8 wrapper not yet "
                           "installed); see the sitecustomize block in overlay-integration.patch")
    orig = cls.compute_candidates

    def compute_candidates(self, hidden_states):
        twin = getattr(self, "_glm_head_twin", None)
        m = hidden_states.shape[0]
        lp = self.candidate_logits_processor
        if DUMP:
            _dump(hidden_states)
        if (twin is None or not 0 < m <= MAX_M or hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2
                or getattr(lp, "head_dtype", None) not in (None, torch.bfloat16)):
            return orig(self, hidden_states)
        top_k = self.model.candidate_selector.top_k
        mode = _head_mode() if twin[0] == "nvfp4" else "1"
        if mode == "0":
            return orig(self, hidden_states)
        if mode == "check":
            ids, vals = orig(self, hidden_states)
            qids, _ = lp.get_top_k_tokens(TwinHeadProxy(self.lm_head, twin), hidden_states, top_k)
            if _Stats.dev is None:
                _Stats.dev = torch.zeros(4, dtype=torch.int64, device=hidden_states.device)
            r = ids.shape[0]
            upd = torch.stack([
                ids.new_full((), r),
                (ids[:, 0] == qids[:, 0]).sum(),
                (qids == ids[:, :1]).any(-1).sum(),
                (ids.sort(-1).values == qids.sort(-1).values).all(-1).sum()]).to(torch.int64)
            _Stats.dev.add_(upd)
            return ids, vals
        return lp.get_top_k_tokens(TwinHeadProxy(self.lm_head, twin), hidden_states, top_k)

    cls.compute_candidates = compute_candidates
    cls._glm_ds_fp4 = True
    _log(f"compute_candidates wrapper armed (head mode {HEAD_MODE}, rows <= {MAX_M})")


def install_dflash_speculator(mod) -> None:
    import torch
    cls = mod.DFlashSpeculator
    if getattr(cls, "_glm_ds_fp4", False):
        return
    orig_load, orig_propose = cls.load_draft_model, cls.propose

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig_load(self, target_model, target_attn_layer_names)
        if HEAD_MODE != "0" and hasattr(model, "compute_candidates"):
            w = getattr(getattr(model, "lm_head", None), "weight", None)
            if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % 256:
                _log("no bf16 lm_head on the drafter; NVFP4 head off")
            else:
                p, s, s2 = make_nvfp4(w.data)
                model._glm_head_twin = ("nvfp4", p, s, s2)
                torch.cuda.empty_cache()
                ref = w.data[:2048].float()
                rel = float((dequant_nvfp4(p[:2048], s[:2048], s2) - ref).norm() / ref.norm())
                _log(f"NVFP4 copy of the draft head {tuple(w.shape)}: {(p.numel() + s.numel()) / 1e6:.1f} MB, "
                     f"global scale 2^{int(math.log2(s2))}, rel. error {rel * 100:.2f} % (first 2048 rows)")
        return model

    def propose(self, *a, **kw):
        out = orig_propose(self, *a, **kw)
        if HEAD_MODE == "check" and _Stats.dev is not None and LOG_EVERY:
            _Stats.calls += 1
            if _Stats.calls % LOG_EVERY == 0 and _rank() == 0:
                r, a1, rec, same = _Stats.dev.tolist()
                _log(f"check after {_Stats.calls} proposals: rows {r}, top-1 agree {a1 / max(r, 1) * 100:.2f} %, "
                     f"stock top-1 in NVFP4 top-k {rec / max(r, 1) * 100:.2f} %, identical top-k sets "
                     f"{same / max(r, 1) * 100:.2f} %")
        return out

    cls.load_draft_model = load_draft_model
    cls.propose = propose
    cls._glm_ds_fp4 = True


def install_nvfp4_kv(mod) -> None:
    import torch
    cls = mod.DFlashQwen3Model
    if getattr(cls._build_context_kv_buffers, "_glm_ds_fp4", False):
        return
    orig = cls._build_context_kv_buffers

    def _build_context_kv_buffers(self, layers_attn, has_bias):
        w0 = layers_attn[0].qkv_proj.weight
        if w0.dtype != torch.uint8:
            return orig(self, layers_attn, has_bias)
        dt = self.hidden_norm.weight.dtype
        self._hidden_norm_weight = self.hidden_norm.weight.data
        kv = []
        for a in layers_attn:
            p = a.qkv_proj
            sc = getattr(p, "weight_scale", None)
            s2 = getattr(p, "weight_scale_2", None)
            if sc is None or s2 is None or sc.dtype != torch.float8_e4m3fn:
                raise RuntimeError("glm-ds-draft-fp4: drafter qkv_proj is uint8 without NVFP4 scales; refusing to "
                                   "build the context-KV buffer from packed bytes")
            rows = p.weight.data[a.q_size:]
            kv.append(dequant_nvfp4(rows, sc.data[a.q_size:], float(s2.data.float().max())).to(dt))
        self._fused_kv_weight = torch.cat(kv, dim=0).contiguous()
        self._fused_kv_bias = (torch.cat([a.qkv_proj.bias[a.q_size:] for a in layers_attn], dim=0)
                               if has_bias else None)
        self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in layers_attn], dim=0).contiguous()
        _log(f"draft context-KV buffer built from dequantized NVFP4 k/v rows {tuple(self._fused_kv_weight.shape)}")

    _build_context_kv_buffers._glm_ds_fp4 = True
    cls._build_context_kv_buffers = _build_context_kv_buffers
    _log("NVFP4 drafter context-KV fix armed")


EXPECTED = {  # Tony v11 image, same scheme as glm_ds_hooks.py
    "vllm.model_executor.models.qwen3_dflash:DFlashQwen3Model._build_context_kv_buffers": "fa82fc75c0f2bdcb",
    "vllm.model_executor.models.qwen3_dflash:DFlashQwen3ForCausalLM.load_weights": "faf77b70f34b2f73",
    "vllm.model_executor.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM.compute_candidates": "c3ab78b7f9645cd0",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.load_draft_model": "f80a17509f09ccb8",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.propose": "432c21dae8db3abe",
    "vllm.model_executor.layers.quantization.modelopt:ModelOptNvFp4W4A16LinearMethod.create_weights": "e17c19459cd47de3",
    "vllm.model_executor.layers.quantization.modelopt:ModelOptNvFp4W4A16LinearMethod.process_weights_after_loading":
        "a0043d21707527c8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4:prepare_fp4_layer_for_marlin": "6c30f3852a14f114",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor.get_top_k_tokens": "d5c541467fd52166",
}


def check_sources() -> None:
    import glm_ds_hooks as h
    got = h.hashes_at(h.vllm_root(), EXPECTED)
    bad = {k: (v, got[k]) for k, v in EXPECTED.items() if got[k] != v}
    if bad and os.environ.get("GLM_DS_ALLOW_DRIFT", "0") != "1":
        lines = "\n".join(f"  {k}: expected {e}, image {g}" for k, (e, g) in sorted(bad.items()))
        raise RuntimeError(f"glm-ds-draft-fp4: engine sources differ from the qualified image:\n{lines}")


def register() -> dict:
    """From sitecustomize (after glm_ds_hooks.register). Returns what was armed."""
    want = {"kv": NVFP4_KV, "head": HEAD_MODE != "0", "share": SHARE_FROM_CERT}
    if not any(want.values()):
        return want
    check_sources()
    import glm_ds_hooks
    if want["kv"]:
        glm_ds_hooks.after_import("vllm.model_executor.models.qwen3_dflash", install_nvfp4_kv)
    if want["head"] or want["share"]:
        glm_ds_hooks.after_import("vllm.model_executor.models.qwen3_dflash2", install_dflash2_model)
    if want["head"]:
        glm_ds_hooks.after_import("vllm.v1.worker.gpu.spec_decode.dflash.speculator", install_dflash_speculator)
    return want

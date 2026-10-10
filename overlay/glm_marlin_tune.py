# SPDX-License-Identifier: Apache-2.0
"""GLM_MARLIN_TUNE=<tune.json>: per-shape Marlin launch configs for GLM-5.3-Flash decode, plus two wrapper cleanups.
Default off (unset / 0 / off: nothing is imported or patched). Same weights; only the fp32 summation order of a GEMM
can change (tile / grid choice), which the sweep gates per config and a later KLD run gates end to end.

What it changes (each part only when tune.json asks for it):
  1. Routed MoE (vLLM fused_marlin_moe, NVFP4 Marlin): `vllm._custom_ops.moe_wna16_marlin_gemm` is wrapped; a call
     whose (size_n, size_k) is a tuned GEMM (gate_up 1024 x 4096 / down 4096 x 512) and whose token count
     (topk_weights.shape[0]) falls in a tuned bucket gets thread_k / thread_n / blocks_per_sm (the op's own
     override arguments), or is routed to the JIT Marlin op when the entry carries a "jit" config (stages / exact
     shared memory / grid). Calls that already pass an override are left alone.
  2. MoE workspace: `marlin_make_workspace_new` inside marlin_moe (fused_marlin_moe allocates and zero-fills a lock
     buffer on every call because MarlinExperts.apply never passes one) returns one persistent per-device buffer.
     Marlin leaves every lock at zero when it finishes, exactly as the dense layers' persistent workspaces rely on.
  3. Dense W8A16 (MarlinMxfp8LinearKernel / MarlinFP8ScaledMMLinearKernel.apply_weights): a tuned (fmt, N, K, M)
     runs the JIT build of vLLM's Marlin template with the tuned tile / stages / grid / smem; with "unpad":
     "kernel" a padded layer (KDA in_proj, N 6416 padded to 6464) writes its [M, 6416] output directly (no slice
     copy). "unpad": "view" instead returns the strided view of stock's padded output (zero-copy; exact for the
     GEMM, but downstream kernels then see a 6464 row stride: gate that end to end before trusting it).

Graph safety: every decision depends only on static shapes and the JSON, so a captured graph bakes one choice.
Buffers (MoE workspace, per-layer dense workspaces) are allocated and zeroed on eager calls only (the profile run
reaches every layer before capture); a call that would need one during capture falls back to stock and says so.
Rank invariance: TP ranks see identical shapes and M, and read the same JSON; GLM_MARLIN_TUNE_SHA=<sha256> makes
every rank refuse a file whose content differs (the file is per node). A JIT entry whose library cannot be loaded
fails the boot (no silent per-rank divergence).

In-boot A/B (overlay/glm_ab.py armed; added for the 2026-09-28 speed screen): GLM_MARLIN_TUNE=<path> stays the
install gate, and the per-variant switch is GLM_MARLIN_TUNE_ON=0|1 (bool, read per call through glm_ab, default off
while armed; ignored when the harness is off). The persistent MoE workspace is allocated on eager calls whatever the
variant, so a tuned variant's capture finds it even though the profile run happens under variant 0.

Knobs: GLM_MARLIN_TUNE=<path>; GLM_MARLIN_TUNE_SHA=<sha256 of the file>; GLM_MARLIN_TUNE_CHECK=1 (TEST ONLY: every
eager tuned call also runs stock and compares; raises on rel L2 > 5e-3 or a finite/non-finite mismatch).

Credits: Marlin (Elias Frantar, IST-DASLab), its vLLM port and MoE / FP8 / MXFP8 extensions (Neural Magic, vLLM
contributors), stream-K scheduling from vllm#24722.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

TARGET_OPS = "vllm._custom_ops"
TARGET_MOE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
TARGET_MX = "vllm.model_executor.kernels.linear.mxfp8.marlin"
TARGET_FP8 = "vllm.model_executor.kernels.linear.scaled_mm.marlin"
_OFF = ("", "0", "off", "false", "no")
GATE_REL_L2 = 5e-3
MOE_GEMMS = {(1024, 4096): "gate_up", (4096, 512): "down"}

S = {"table": None, "jit": None, "moe_ws": {}, "logged": set(), "applied": {}, "fallback": {}, "checked": 0,
     "installed": set(), "sha": None}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-marlin-tune: {msg}\n")


def _log_once(key, msg):
    if key not in S["logged"]:
        S["logged"].add(key)
        _log(msg)


def _count(kind, key):
    d = S[kind]
    d[key] = d.get(key, 0) + 1


# ----------------------------------------------------------------------------------------------- table (pure)
class TableError(ValueError):
    pass


def bucket(m: int, tuned_ms):
    """Smallest tuned M >= m; None above the largest (prefill and large batches stay stock)."""
    for hi in sorted(tuned_ms):
        if m <= hi:
            return hi
    return None


class Table:
    """Parsed tune.json. Schema (version 1):
    {"version": 1,
     "moe": {"persistent_workspace": bool,
             "entries": [{"gemm": "gate_up"|"down", "m": 4, "thread_k": 128, "thread_n": 128, "blocks_per_sm": 1}
                         | {"gemm": ..., "m": ..., "jit": {"thread_k", "thread_n", "threads", "stages", "grid",
                                                         "smem"}}]},
     "dense": {"unpad": "off"|"view"|"kernel", "view_max_m": 32,
               "entries": [{"fmt": "mxfp8"|"fp8blk", "size_n": 6464, "size_k": 4096, "m": 4,
                            "jit": {"m_variant", "thread_k", "thread_n", "threads", "stages", "grid", "smem"}}]},
     "jit": {"so": "/path/lib.so", "manifest": "/path/manifest.json"}}   # needed by any "jit" entry
    "m_grid": the tuned token counts, default [4, 8, 16, 32]. A call with M tokens uses the smallest grid value
    >= M; if that grid point has no entry for the GEMM / layer, the call stays stock (an entry never leaks into
    another M range)."""

    def __init__(self, d: dict):
        if d.get("version") != 1:
            raise TableError("tune.json: version must be 1")
        self.m_grid = sorted(int(x) for x in d.get("m_grid", (4, 8, 16, 32)))
        if not self.m_grid or self.m_grid[0] <= 0:
            raise TableError("tune.json: bad m_grid")
        moe = d.get("moe") or {}
        dense = d.get("dense") or {}
        self.persistent_workspace = bool(moe.get("persistent_workspace", False))
        self.unpad = dense.get("unpad", "off")
        if self.unpad not in ("off", "view", "kernel"):
            raise TableError(f"tune.json: dense.unpad {self.unpad!r}")
        self.view_max_m = int(dense.get("view_max_m", 32))
        self.jit_paths = d.get("jit")
        self.moe = {}
        for e in moe.get("entries", []):
            if e.get("gemm") not in ("gate_up", "down") or int(e.get("m", 0)) <= 0:
                raise TableError(f"tune.json: bad moe entry {e}")
            if "jit" in e:
                j = e["jit"]
                for k in ("thread_k", "thread_n", "threads", "stages", "grid", "smem"):
                    int(j[k])
            else:
                for k in ("thread_k", "thread_n", "blocks_per_sm"):
                    int(e[k])
            if int(e["m"]) not in self.m_grid:
                raise TableError(f"tune.json: moe entry m={e['m']} not in m_grid {self.m_grid}")
            self.moe.setdefault(e["gemm"], {})[int(e["m"])] = e
        self.dense = {}
        for e in dense.get("entries", []):
            if e.get("fmt") not in ("mxfp8", "fp8blk") or "jit" not in e:
                raise TableError(f"tune.json: bad dense entry {e}")
            j = e["jit"]
            if j.get("m_variant") not in ("m8", "m16", "m32", "m48", "m64"):
                raise TableError(f"tune.json: bad m_variant in {e}")
            cap = {"m8": 8, "m16": 16, "m32": 32, "m48": 48, "m64": 64}[j["m_variant"]]
            if int(e["m"]) > cap:
                raise TableError(f"tune.json: entry m={e['m']} exceeds its kernel's m tile ({j['m_variant']})")
            if int(e["m"]) not in self.m_grid:
                raise TableError(f"tune.json: dense entry m={e['m']} not in m_grid {self.m_grid}")
            self.dense.setdefault((e["fmt"], int(e["size_n"]), int(e["size_k"])), {})[int(e["m"])] = e
        self.needs_jit = any("jit" in e for g in self.moe.values() for e in g.values()) or bool(self.dense)
        if self.needs_jit and not self.jit_paths:
            raise TableError("tune.json: jit entries need a top-level \"jit\": {\"so\", \"manifest\"}")

    def moe_entry(self, size_n: int, size_k: int, m_tokens: int):
        gemm = MOE_GEMMS.get((size_n, size_k))
        if gemm is None or gemm not in self.moe:
            return None
        b = bucket(m_tokens, self.m_grid)
        return None if b is None else self.moe[gemm].get(b)

    def dense_entry(self, fmt: str, size_n: int, size_k: int, m: int):
        per = self.dense.get((fmt, size_n, size_k))
        if not per:
            return None
        b = bucket(m, self.m_grid)
        return None if b is None else per.get(b)


def load_table(path: str, expect_sha: str | None = None) -> Table:
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    if expect_sha and expect_sha.strip().lower() != sha:
        raise TableError(f"tune.json sha256 {sha} != GLM_MARLIN_TUNE_SHA {expect_sha} (files differ across nodes?)")
    t = Table(json.loads(raw))
    S["sha"] = sha
    return t


# ----------------------------------------------------------------------------------------------- runtime helpers
def _capturing() -> bool:
    import torch
    return torch.cuda.is_current_stream_capturing()


def _jit():
    if S["jit"] is None:
        t = S["table"]
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (here, os.path.join(here, "..", "jit"), os.path.join(here, "..")):
            if os.path.exists(os.path.join(p, "marlin_jit.py")) and p not in sys.path:
                sys.path.insert(0, p)
        try:
            import marlin_jit
        except ImportError:
            marlin_jit = None
        if marlin_jit is not None:
            S["jit"] = marlin_jit.load_prebuilt(t.jit_paths["so"], t.jit_paths["manifest"])
        else:  # minimal loader: the ops and the manifest are all the overlay needs
            import torch
            torch.ops.load_library(t.jit_paths["so"])
            with open(t.jit_paths["manifest"]) as f:
                S["jit"] = _MiniJit(json.load(f), torch.ops.glm_marlin_tune)
        _log(f"JIT Marlin library loaded: {t.jit_paths['so']} (manifest {S['jit'].manifest.get('hash')})")
    return S["jit"]


class _MiniJit:
    def __init__(self, manifest, ops):
        self.manifest, self.ops = manifest, ops
        self._dense = {(d["fmt"], d["m_variant"], d["thread_k"], d["thread_n"], d["threads"], d["stages"]): d["id"]
                       for d in manifest["dense"]}
        self._moe = {(d["thread_k"], d["thread_n"], d["threads"], d["stages"]): d["id"] for d in manifest["moe"]}


def _dense_kid(jit, fmt, j):
    k = (fmt, j["m_variant"], int(j["thread_k"]), int(j["thread_n"]), int(j["threads"]), int(j["stages"]))
    kid = jit._dense.get(k)
    if kid is None:
        raise RuntimeError(f"glm-marlin-tune: dense JIT instance {k} is not in the loaded library")
    return kid


def _moe_kid(jit, j):
    k = (int(j["thread_k"]), int(j["thread_n"]), int(j["threads"]), int(j["stages"]))
    kid = jit._moe.get(k)
    if kid is None:
        raise RuntimeError(f"glm-marlin-tune: MoE JIT instance {k} is not in the loaded library")
    return kid


def _new_workspace(device, n):
    import torch
    return torch.zeros(n, dtype=torch.int32, device=device)


def _sms(device):
    import torch
    return torch.cuda.get_device_properties(device).multi_processor_count


def _compare(tuned, stock, what):
    import torch
    a, b = tuned.reshape(-1), stock.reshape(-1)
    fin_a, fin_b = torch.isfinite(a.float()), torch.isfinite(b.float())
    if bool((fin_a != fin_b).any()):
        raise RuntimeError(f"glm-marlin-tune CHECK: finite/non-finite mismatch in {what}")
    eq = float((a.view(torch.int16) == b.view(torch.int16)).float().mean())
    rel = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    S["checked"] += 1
    if S["checked"] <= 3 or S["checked"] % 5000 == 0:
        _log(f"CHECK {what}: bit-equal {eq:.4f}, rel L2 {rel:.2e} ({S['checked']} checked)")
    if rel > GATE_REL_L2:
        raise RuntimeError(f"glm-marlin-tune CHECK: {what} rel L2 {rel:.3e} > {GATE_REL_L2}")


def _ab_on() -> bool:
    """Per-call A/B gate: True unless overlay/glm_ab.py is armed and the current variant has GLM_MARLIN_TUNE_ON off."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.truthy(ab.env("GLM_MARLIN_TUNE_ON"))
    return True


def _table():
    return S["table"] if _ab_on() else None


def _check_on() -> bool:
    return os.environ.get("GLM_MARLIN_TUNE_CHECK", "0").strip().lower() not in _OFF


# ----------------------------------------------------------------------------------------------- MoE hooks
def make_moe_gemm_wrapper(orig):
    def moe_wna16_marlin_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, g_idx,
                              perm, workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
                              moe_block_size, top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k, is_k_full,
                              use_atomic_add, use_fp32_reduce, is_zp_float, thread_k=-1, thread_n=-1,
                              blocks_per_sm=-1):
        args = (input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, g_idx, perm, workspace,
                sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k,
                mul_topk_weights, b_q_type, size_m, size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce,
                is_zp_float)
        t = _table()
        if t is None or thread_k != -1 or thread_n != -1 or blocks_per_sm != -1:
            return orig(*args, thread_k=thread_k, thread_n=thread_n, blocks_per_sm=blocks_per_sm)
        e = t.moe_entry(size_n, size_k, int(topk_weights.shape[0]))
        if e is None or str(getattr(b_scales, "dtype", "")) != "torch.float8_e4m3fn":
            return orig(*args)
        key = (e["gemm"], int(topk_weights.shape[0]))
        if "jit" in e:
            out = _moe_jit_call(e, args)
            if out is None:
                return orig(*args)
        else:
            out = orig(*args, thread_k=int(e["thread_k"]), thread_n=int(e["thread_n"]),
                       blocks_per_sm=int(e["blocks_per_sm"]))
        _count("applied", key)
        _log_once(("moe",) + key, f"MoE {e['gemm']} M={key[1]}: " +
                  (f"JIT {e['jit']}" if "jit" in e else
                   f"thread_k={e['thread_k']} thread_n={e['thread_n']} blocks_per_sm={e['blocks_per_sm']}"))
        if _check_on() and not _capturing():
            ref = orig(*((args[0], None) + args[2:]))
            _compare(out, ref, f"moe {e['gemm']} M={key[1]}")
        return out
    moe_wna16_marlin_gemm._glm_marlin_tune = True
    return moe_wna16_marlin_gemm


def _moe_jit_call(e, args):
    (a, c, w, bias, s, a_s, gs, zp, g_idx, perm, _ws, sorted_ids, eids, ntp, topk_w, block_m, top_k, mul, _bt,
     size_m, size_n, size_k, is_k_full, atomic, fp32, zp_float) = args
    if (bias is not None or a_s is not None or zp is not None or (g_idx is not None and g_idx.numel()) or
            (perm is not None and perm.numel()) or gs is None or int(block_m) != 8 or atomic or not fp32 or
            zp_float or str(a.dtype) != "torch.bfloat16"):
        _count("fallback", ("moe-jit-unsupported", e["gemm"]))
        return None
    ws = _moe_ws(a.device, jit=True)
    if ws is None:
        _count("fallback", ("moe-jit-ws-in-capture", e["gemm"]))
        _log_once(("moe-ws-capture",), "MoE JIT workspace not allocated before capture: stock for that graph")
        return None
    jit = _jit()
    j = e["jit"]
    return jit.ops.moe_gemm(a, c, w, s, gs, ws, sorted_ids, eids, ntp, topk_w, int(block_m), int(top_k), bool(mul),
                            int(size_m), int(size_n), int(size_k), _moe_kid(jit, j), int(j["grid"]), int(j["smem"]))


def _moe_ws(device, jit=False):
    """Persistent MoE lock buffer (4 * SMs + 8 ints, kernel leaves it zeroed); None if first needed in capture."""
    key = (str(device), jit)
    ws = S["moe_ws"].get(key)
    if ws is None:
        if _capturing():
            return None
        ws = _new_workspace(device, 4 * _sms(device) + 8)
        S["moe_ws"][key] = ws
    return ws


def install_ops(mod) -> None:
    f = getattr(mod, "moe_wna16_marlin_gemm", None)
    if f is None or getattr(f, "_glm_marlin_tune", False):
        return
    mod.moe_wna16_marlin_gemm = make_moe_gemm_wrapper(f)
    S["installed"].add(TARGET_OPS)
    _log("moe_wna16_marlin_gemm wrapped (tuned thread_k / thread_n / blocks_per_sm or JIT per GEMM and M)")


def install_moe(mod) -> None:
    orig = getattr(mod, "marlin_make_workspace_new", None)
    if orig is None or getattr(orig, "_glm_marlin_tune", False):
        return

    def marlin_make_workspace_new(device, max_blocks_per_sm=1, existing=None):
        t_ = S["table"]
        if t_ is None or not t_.persistent_workspace or existing is not None or max_blocks_per_sm != 4:
            return orig(device, max_blocks_per_sm, existing) if existing is not None else \
                orig(device, max_blocks_per_sm)
        if not _ab_on():
            if not _capturing():
                _moe_ws(device)          # allocate now: a tuned variant captured later must find it
            return orig(device, max_blocks_per_sm)
        ws = _moe_ws(device)
        if ws is None:
            _count("fallback", ("moe-ws-in-capture",))
            _log_once(("moe-ws-capture0",), "persistent MoE workspace first requested inside capture: stock alloc")
            return orig(device, max_blocks_per_sm)
        _count("applied", ("moe-persistent-ws",))
        return ws

    marlin_make_workspace_new._glm_marlin_tune = True
    mod.marlin_make_workspace_new = marlin_make_workspace_new
    S["installed"].add(TARGET_MOE)
    t = S["table"]
    _log("fused_marlin_moe workspace hook installed (persistent lock buffer per device: "
         f"{'on' if t is not None and t.persistent_workspace else 'off in this table'})")


# ----------------------------------------------------------------------------------------------- dense hooks
def _layer_ws(layer, device):
    ws = layer.__dict__.get("_glm_mt_ws")
    if ws is None:
        if _capturing():
            return None
        ws = _new_workspace(device, 4 * _sms(device) + 8)
        layer.__dict__["_glm_mt_ws"] = ws
    return ws


def make_dense_wrapper(orig, fmt: str, scales_of):
    def apply_weights(self, layer, x, bias=None, *rest, **kw):
        t = _table()
        if t is None or rest or kw:
            return orig(self, layer, x, bias, *rest, **kw)
        w = layer.weight
        if w.dim() != 2 or bias is not None:
            return orig(self, layer, x, bias)
        size_k, size_n = w.size(0) * 16, w.size(1) * 4 // 16      # Marlin fp8 layout: [K/16, 4N] int32
        n_log = int(layer.output_size_per_partition)
        k_log = int(layer.input_size_per_partition)
        per = t.dense.get((fmt, size_n, size_k))
        if per is not None and not _capturing():
            _layer_ws(layer, w.device)                          # allocate on the eager profile / warm-up runs
        if k_log != size_k or x.shape[-1] != size_k:
            return orig(self, layer, x, bias)
        x2 = x.reshape(-1, size_k)
        m = int(x2.shape[0])
        e = t.dense_entry(fmt, size_n, size_k, m)
        padded = n_log != size_n
        if e is not None:
            j = e["jit"]
            ws = _layer_ws(layer, w.device)
            if ws is None:
                _count("fallback", ("dense-ws-in-capture", fmt, size_n, size_k))
                _log_once(("dense-ws-capture", fmt, size_n, size_k),
                          f"dense {fmt} {size_n}x{size_k}: workspace missing at capture, stock for that graph")
                return orig(self, layer, x, bias)
            jit = _jit()
            n_out = n_log if (padded and t.unpad == "kernel") else 0
            out = jit.ops.dense_gemm(x2, w, scales_of(self, layer), ws, size_n, size_k, n_out, _dense_kid(jit, fmt, j),
                                     int(j["grid"]), int(j["smem"]))
            if padded and n_out == 0:
                out = out[:, :n_log] if (t.unpad == "view" and m <= t.view_max_m) else out[:, :n_log].contiguous()
            key = (fmt, size_n, size_k, m)
            _count("applied", key)
            _log_once(("dense",) + key, f"dense {fmt} {size_n}x{size_k} M={m}: JIT {j} n_out={n_out}")
            if _check_on() and not _capturing():
                _compare(out, orig(self, layer, x, bias).reshape(-1, n_log), f"dense {fmt} {size_n}x{size_k} M={m}")
            return out.reshape(*x.shape[:-1], n_log)
        if padded and t.unpad == "view" and m <= t.view_max_m:
            from vllm import _custom_ops as ops
            from vllm.scalar_type import scalar_types
            out = ops.marlin_gemm(a=x2, c=None, b_q_weight=w, b_bias=None, b_scales=scales_of(self, layer),
                                  a_scales=None, global_scale=None, b_zeros=None, g_idx=None, perm=None,
                                  workspace=layer.workspace, b_q_type=scalar_types.float8_e4m3fn, size_m=m,
                                  size_n=size_n, size_k=size_k, use_atomic_add=False, use_fp32_reduce=True)
            _count("applied", ("view", fmt, size_n, size_k, m))
            _log_once(("view", fmt, size_n, size_k), f"dense {fmt} {size_n}x{size_k}: strided view, no slice copy")
            return out[:, :n_log].reshape(*x.shape[:-1], n_log)
        return orig(self, layer, x, bias)

    apply_weights._glm_marlin_tune = True
    return apply_weights


def install_mx(mod) -> None:
    cls = getattr(mod, "MarlinMxfp8LinearKernel", None)
    if cls is None or getattr(cls.apply_weights, "_glm_marlin_tune", False):
        return
    cls.apply_weights = make_dense_wrapper(cls.apply_weights, "mxfp8", lambda self, layer: layer.weight_scale)
    S["installed"].add(TARGET_MX)
    _log("MarlinMxfp8LinearKernel.apply_weights wrapped")


def install_fp8(mod) -> None:
    cls = getattr(mod, "MarlinFP8ScaledMMLinearKernel", None)
    if cls is None or getattr(cls.apply_weights, "_glm_marlin_tune", False):
        return

    def scales(self, layer):
        return layer.weight_scale_inv if getattr(self, "block_quant", False) else layer.weight_scale

    orig = cls.apply_weights

    def fp8_only_block(self, layer, x, bias=None, *rest, **kw):
        # per-channel / per-tensor FP8 keeps stock; the table only describes block-128 (BF16 scale) layers
        if not getattr(self, "block_quant", False) or getattr(self, "marlin_input_dtype", None) is not None:
            return orig(self, layer, x, bias, *rest, **kw)
        return wrapped(self, layer, x, bias, *rest, **kw)

    wrapped = make_dense_wrapper(orig, "fp8blk", scales)
    fp8_only_block._glm_marlin_tune = True
    cls.apply_weights = fp8_only_block
    S["installed"].add(TARGET_FP8)
    _log("MarlinFP8ScaledMMLinearKernel.apply_weights wrapped (block-FP8 layers)")


HOOKS = {TARGET_OPS: install_ops, TARGET_MOE: install_moe, TARGET_MX: install_mx, TARGET_FP8: install_fp8}


def set_table(table) -> None:
    """Switch the active table in-process (benchmarks); None = every wrapper passes through to stock."""
    S["table"] = table


def install_now() -> None:
    """Patch the targets that are already imported (benchmarks, tests)."""
    for name, fn in HOOKS.items():
        if name in sys.modules:
            fn(sys.modules[name])


def stats() -> dict:
    return {"applied": dict(S["applied"]), "fallback": dict(S["fallback"]), "checked": S["checked"],
            "installed": sorted(S["installed"]), "sha": S["sha"]}


def register() -> None:
    path = os.environ.get("GLM_MARLIN_TUNE", "").strip()
    if path.lower() in _OFF:
        return
    import importlib.abc
    import importlib.util

    S["table"] = load_table(path, os.environ.get("GLM_MARLIN_TUNE_SHA"))
    t = S["table"]
    _log(f"armed: {path} sha256 {S['sha'][:16]}; moe entries {sum(len(v) for v in t.moe.values())}, "
         f"persistent workspace {t.persistent_workspace}; dense entries "
         f"{sum(len(v) for v in t.dense.values())}, unpad {t.unpad}; jit {bool(t.needs_jit)}")
    if t.needs_jit:
        for k in ("so", "manifest"):
            if not os.path.exists(t.jit_paths[k]):
                raise RuntimeError(f"glm-marlin-tune: jit {k} missing: {t.jit_paths[k]}")
    install_now()

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path_, target=None):
            if name not in HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec, _fn=HOOKS[name]):
                _orig(module)
                _fn(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

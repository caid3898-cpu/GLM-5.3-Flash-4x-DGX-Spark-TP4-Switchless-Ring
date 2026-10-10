# SPDX-License-Identifier: Apache-2.0
"""GLM_PF3_*: prefill routed-MoE kernels from diagnostics/glm-prefill3-20260928 (GLM-5.3-Flash TP4, GB10). Default OFF.

Three independent switches, all eager-prefill only (never inside a CUDA-graph capture), all read per call through
overlay/glm_ab.py when the in-boot A/B harness is armed:

  GLM_PF3_SUMADD=1  EXACT. The routed moe_sum and the MoERunner's `shared_output + fused_output` become one kernel
                    with the literal stock rounding tree:
                      r = bf16(((0 + x0) + x1) + ... + x7)   fp32, slot order (moe_sum_vec_kernel)
                      y = bf16(float(shared) + float(r))     (aten::add, bf16, alpha 1)
                    (__float2bfloat16 as c10 uses in device code). MarlinExperts.moe_sum writes y into the routed
                    output when the runner's shared-expert output is already computed (NO_OVERLAP order, prefill),
                    routed_scaling_factor == 1, no routed output transform, not reduced before the transform; the
                    runner's _unpack then hands back (None, y) so the add is not done twice (pointer-checked).
                    Saves one [M, 4096] write + read per MoE layer (measured 0.43-0.47 ms/layer at 5760 rows).
  GLM_PF3_DOWN=1    SAME-MATH. The down GEMM (NVFP4 Marlin, [8M, 512] x [512, 4096], bm 64) runs vLLM's own MoE
                    Marlin template instantiated as tile 64x128 / 128 threads / 3 stages / 2 CTAs per SM (grid 96)
                    instead of stock 64x256 / 256 / 4 / grid 48. Same operands, same bf16 rounding points (global
                    scale, top-k weight); the stream-K split of the fp32 K sum differs on a few tiles (bit-equal on
                    real layers at 5760 rows, 1 ulp on 1e-7 of values at 3968). Stock itself is not run-to-run
                    bitwise (moe_align orders tokens inside a block with atomics).
  GLM_PF3_ACT=1     SAME-MATH. gate_up + silu_and_mul_with_clamp in one Marlin kernel (tile 64x128 / 3 stages /
                    grid 96): each thread_n tile reads gate[j*64..] and up[j*64..] straight from the served w13
                    (column remap in the B / scale addressing, no weight copy), and its epilogue applies vLLM's
                    act_and_mul math to the finished bf16 tile (same two bf16 roundings, same expf / IEEE div;
                    exhaustively bit-identical on all 65536 gate patterns). Only the fp32 split-K grouping of a
                    few tiles differs from stock (tile ids move).

Other knobs: GLM_PF3_MIN_M (default 1024 tokens), GLM_PF3_CHECK=N (the first N eligible calls per process also run
the stock path on the same routing and log bitwise stats; the sum-add fast path arms only after one exact check with
the pointer handshake confirmed), GLM_PF3_BUILD_DIR (default /cache/glm_pf3).

Credits: Marlin (Elias Frantar, IST-DASLab) and vLLM's MoE / NVFP4 / stream-K port (Neural Magic, vLLM
contributors); the fused-epilogue / finalize ideas come from Matt Mastracci's moe_prefill and SP_MOE_FUSED (ideas
only, no code).
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import threading

_OFF = ("", "0", "off", "false", "no")
KEYS = ("GLM_PF3_SUMADD", "GLM_PF3_DOWN", "GLM_PF3_ACT")
MARLIN_MOD = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
RUNNER_MOD = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
OPTIN = 101376
GRID, SMEM96 = 96, OPTIN // 2 - 1024
TILE = dict(tmb=4, tk=64, tn=128, th=128, stages=3)          # both the down and the act kernel
S = {"ext": None, "ids": None, "failed": None, "act_p": {}, "check_left": None, "stats": {},
     "sumadd_armed": False, "logged": set()}
_TLS = threading.local()
_LOCK = threading.Lock()


def _log(msg):
    sys.stderr.write(f"glm-pf3: {msg}\n")
    sys.stderr.flush()


def _once(key, msg):
    if key not in S["logged"]:
        S["logged"].add(key)
        _log(msg)


def env(name, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def on(name):
    return str(env(name, "0")).strip().lower() not in _OFF


def installed():
    return any(os.environ.get(k, "0").strip().lower() not in _OFF for k in KEYS)


def _int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _capturing():
    import torch
    return torch.cuda.is_current_stream_capturing()


def _check_take():
    """Budget of stock-comparison runs for the act / down path (GLM_PF3_CHECK)."""
    if S["check_left"] is None:
        S["check_left"] = _int("GLM_PF3_CHECK", 0)
    if S["check_left"] > 0:
        S["check_left"] -= 1
        return True
    return False


# ------------------------------------------------------------------------------------------------------------------
# extension (lazy, first eligible call; cached under the build dir)
# ------------------------------------------------------------------------------------------------------------------
def ext():
    if S["ext"] is not None or S["failed"] is not None:
        return S["ext"]
    with _LOCK:
        if S["ext"] is not None or S["failed"] is not None:
            return S["ext"]
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import glm_pf3_jit as PJ
            root = os.environ.get("GLM_PF3_BUILD_DIR", "").strip() or (
                "/cache/glm_pf3" if os.path.isdir("/cache") and os.access("/cache", os.W_OK)
                else os.path.expanduser("~/.cache/glm_pf3"))
            inst = [dict(TILE, act=False), dict(TILE, act=True)]
            e, tag, secs = PJ.build(inst, root, jobs=_int("GLM_PF3_JOBS", 4))
            S["ids"] = {a: e.kid(a, TILE["tmb"], TILE["tk"], TILE["tn"], TILE["th"], TILE["stages"])[0]
                        for a in (False, True)}
            S["ext"] = e.ops
            _log(f"extension {tag} ready in {secs:.1f} s (down kid {S['ids'][False]}, act kid {S['ids'][True]})")
        except Exception as exc:  # noqa: BLE001
            S["failed"] = repr(exc)
            _log(f"extension build/load FAILED, every GLM_PF3 path falls back to stock: {exc!r}")
    return S["ext"]


def _bitstats(a, b):
    import torch
    ai, bi = a.reshape(-1).view(torch.int16), b.reshape(-1).view(torch.int16)
    eq = (ai == bi)
    n = eq.numel()
    ne = int(n - int(eq.sum()))
    d = (a.float() - b.float()).reshape(-1)
    rel = float(d.norm() / b.float().norm().clamp_min(1e-30))
    return f"exact={ne == 0} differing={ne}/{n} rel_l2={rel:.2e}"


# ------------------------------------------------------------------------------------------------------------------
# _fused_marlin_moe: act + down kernels
# ------------------------------------------------------------------------------------------------------------------
def _eligible(b, M):
    import torch
    try:
        from vllm.scalar_type import scalar_types
        hs = b["hidden_states"]
        if hs.dtype != torch.bfloat16 or hs.dim() != 2 or not hs.is_contiguous():
            return "input"
        if M < _int("GLM_PF3_MIN_M", 1024) or _capturing():
            return "size/capture"
        if b.get("block_size_m") != 16 * TILE["tmb"]:
            return f"block_size_m {b.get('block_size_m')}"
        qt = b.get("quant_type")
        if getattr(qt, "id", qt) != scalar_types.float4_e2m1f.id:
            return "quant type"
        for k in ("bias1", "bias2", "w1_zeros", "w2_zeros", "g_idx1", "g_idx2", "sort_indices1", "sort_indices2",
                  "expert_map", "input_global_scale1", "input_global_scale2"):
            if b.get(k) is not None:
                return k
        if b.get("input_dtype") is not None or not b.get("is_k_full", True) or b.get("apply_router_weight_on_input"):
            return "input dtype / k_full / router weight on input"
        if b.get("global_scale1") is None or b.get("global_scale2") is None:
            return "global scale"
    except Exception as exc:  # noqa: BLE001
        return f"probe error {exc!r}"
    return None


def _act_params(b, device):
    """[limit, alpha, beta] as stock silu_and_mul_with_clamp gets them, or None if the activation is not that."""
    act = b.get("activation")
    cfg = b.get("activation_config")
    name = getattr(act, "name", str(act))
    if name != "SILU" or cfg is None or getattr(cfg, "clamp_limit", None) is None:
        return None
    alpha, beta = getattr(cfg, "alpha", None), getattr(cfg, "beta", None)
    alpha = 1.0 if alpha is None else float(alpha)
    beta = 0.0 if beta is None else float(beta)
    if (alpha, beta) != (1.0, 0.0):
        return None
    key = (str(device), float(cfg.clamp_limit))
    if key not in S["act_p"]:
        import torch
        S["act_p"][key] = torch.tensor([float(cfg.clamp_limit), 1.0, 0.0], dtype=torch.float32, device=device)
    return S["act_p"][key]


def make_fused_wrapper(orig):
    sig = inspect.signature(orig)

    def _fused_marlin_moe(*args, **kwargs):
        want_act, want_down = on("GLM_PF3_ACT"), on("GLM_PF3_DOWN")
        if not (want_act or want_down):
            return orig(*args, **kwargs)
        try:
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            b = b.arguments
        except TypeError:
            return orig(*args, **kwargs)
        hs = b["hidden_states"]
        M = hs.shape[0]
        why = _eligible(b, M)
        act_p = _act_params(b, hs.device) if want_act and why is None else None
        if why is not None or ext() is None:
            _once(f"fb:{why}", f"fused_marlin_moe stays stock ({why or S['failed']})")
            return orig(*args, **kwargs)
        if want_act and act_p is None:
            _once("fb:act", "activation is not silu_and_mul_with_clamp(alpha 1, beta 0): act stays stock")
            want_act = False
        check = _check_take()
        out = _run(b, M, want_act, want_down, act_p)
        if check:
            import torch
            b2 = dict(b)
            for k in ("intermediate_cache13", "intermediate_cache2"):
                if b.get(k) is not None:
                    b2[k] = torch.empty_like(b[k])
            b2["output"] = None
            ref = orig(**b2)
            _log(f"check fused_marlin_moe M={M} act={want_act} down={want_down}: {_bitstats(out, ref)}")
        return out
    _fused_marlin_moe._glm_pf3 = True
    return _fused_marlin_moe


def _run(b, M, want_act, want_down, act_p):
    """The stock _fused_marlin_moe body for bf16 x NVFP4 (no bias / zp / act order), with the chosen kernels."""
    import torch
    from vllm import _custom_ops as ops
    hs, w1, w2 = b["hidden_states"], b["w1"], b["w2"]
    K = hs.shape[1]
    N = w2.shape[1] * 16                        # w2 served [E, N/16, 2K]
    topk = b["num_topk"]
    bm = b["block_size_m"]
    sid, eid, ntp = b["sorted_token_ids"], b["expert_ids"], b["num_tokens_post_padded"]
    tw, ws = b["topk_weights"], b.get("workspace")
    if ws is None:      # stock: marlin_make_workspace_new(device, 4) per call; one zeroed buffer per device here
        key = f"ws:{hs.device}"
        if key not in S:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
            S[key] = marlin_make_workspace_new(hs.device, 4)
        ws = S[key]
    c13, c2 = b.get("intermediate_cache13"), b.get("intermediate_cache2")
    if c13 is None:     # as stock allocates them when the caller passes none
        c13 = torch.empty(M * topk * max(2 * N, K), device=hs.device, dtype=hs.dtype)
    if c2 is None:
        c2 = torch.empty(M * topk * N, device=hs.device, dtype=hs.dtype)
    # the engine hands 2-D workspaces ([M*topk, max(N, K)]): resize by elements, as stock _resize_cache does
    c13f, c2f = c13.flatten(), c2.flatten()
    cache1 = c13f[:M * topk * 2 * N].view(M * topk, 2 * N)
    cache3 = c13f[:M * topk * K].view(M * topk, K)
    cache2 = c2f[:M * topk * N].view(M * topk, N)
    e = S["ext"]
    if want_act:
        e.moe_gemm_act(hs, cache2, w1, b["w1_scale"], b["global_scale1"], ws, sid, eid, ntp, tw, bm, topk, False, M,
                       2 * N, K, S["ids"][True], GRID, SMEM96, act_p)
    else:
        ops.moe_wna16_marlin_gemm(hs, cache1, w1, None, b["w1_scale"], None, b["global_scale1"], None, None, None, ws,
                                  sid, eid, ntp, tw, moe_block_size=bm, top_k=topk, mul_topk_weights=False,
                                  b_q_type=b["quant_type"], size_m=M, size_n=2 * N, size_k=K, is_k_full=True,
                                  use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        fn = b.get("activation_func")
        if fn is None:
            from vllm.model_executor.layers.fused_moe.activation import (ApplyMoEActivationConfig,
                                                                          apply_moe_activation)
            cfg = b.get("activation_config") or ApplyMoEActivationConfig()
            apply_moe_activation(b["activation"], cache2, cache1, activation_config=cfg, topk_ids=b.get("topk_ids"),
                                 expert_map=b.get("expert_map"))
        else:
            fn(b["activation"], cache2, cache1, topk_ids=b.get("topk_ids"), expert_map=b.get("expert_map"))
    out = b.get("output")
    if out is None:
        out = cache3
    if want_down:
        e.moe_gemm(cache2, out, w2, b["w2_scale"], b["global_scale2"], ws, sid, eid, ntp, tw, bm, 1, True, M * topk,
                   K, N, S["ids"][False], GRID, SMEM96, None)
    else:
        ops.moe_wna16_marlin_gemm(cache2, out, w2, None, b["w2_scale"], None, b["global_scale2"], None, None, None,
                                  ws, sid, eid, ntp, tw, moe_block_size=bm, top_k=1, mul_topk_weights=True,
                                  b_q_type=b["quant_type"], size_m=M * topk, size_n=K, size_k=N, is_k_full=True,
                                  use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    return out


# ------------------------------------------------------------------------------------------------------------------
# moe_sum + shared add
# ------------------------------------------------------------------------------------------------------------------
def _shared_peek(runner):
    se = getattr(runner, "_shared_experts", None)
    if se is None:
        return None
    try:
        buf = se._output[se._output_idx]
    except Exception:  # noqa: BLE001
        return None
    return buf


def make_moe_sum_wrapper(orig):
    def moe_sum(self, input, output, topk_ids, expert_map):
        if not on("GLM_PF3_SUMADD") or S.get("sumadd_disabled"):
            return orig(self, input, output, topk_ids, expert_map)
        import torch
        runner = getattr(_TLS, "runner", None)
        why = None
        if runner is None:
            why = "no runner context"
        elif expert_map is not None or _capturing():
            why = "expert map / capture"
        elif input.dim() != 3 or input.shape[1] != 8 or input.dtype != torch.bfloat16 or not input.is_contiguous():
            why = "input layout"
        elif input.shape[0] < _int("GLM_PF3_MIN_M", 1024):
            why = "size"
        elif float(getattr(runner, "routed_scaling_factor", 1.0)) != 1.0:
            why = "routed_scaling_factor != 1"
        elif getattr(runner, "routed_output_transform", None) is not None or getattr(runner, "_fused_output_is_reduced", False):
            why = "transform / reduced routed output"
        shared = _shared_peek(runner) if why is None else None
        if why is None and (shared is None or shared.shape != output.shape or shared.dtype != torch.bfloat16
                            or not shared.is_contiguous() or not output.is_contiguous()):
            why = "shared output not ready"
        if why is not None or ext() is None:
            _once(f"sa:{why}", f"moe_sum stays stock ({why or S['failed']})")
            return orig(self, input, output, topk_ids, expert_map)
        if not S["sumadd_armed"]:
            # verification call: stock result stays in `output`; the fused one is compared at _unpack after the
            # engine's own add, and the pointer handshake is confirmed
            orig(self, input, output, topk_ids, expert_map)
            y = torch.empty_like(output)
            S["ext"].sum8_add(input, shared, y)
            _TLS.verify = (output.data_ptr(), y, shared)
            return None
        S["ext"].sum8_add(input, shared, output)
        _TLS.pending = output.data_ptr()
        return None
    moe_sum._glm_pf3 = True
    return moe_sum


def make_unpack_wrapper(orig):
    def _unpack(result):
        shared, fused = orig(result)
        pend = getattr(_TLS, "pending", None)
        ver = getattr(_TLS, "verify", None)
        if pend is not None:
            _TLS.pending = None
            if shared is None or fused.data_ptr() != pend:
                raise RuntimeError("glm-pf3: sum-add pointer handshake broken after verification (would double-add)")
            return None, fused
        if ver is not None:
            _TLS.verify = None
            ptr, y, sh = ver
            ok_ptr = shared is not None and fused.data_ptr() == ptr and shared.data_ptr() == sh.data_ptr()
            ref = shared + fused if shared is not None else None
            st = _bitstats(y, ref) if ref is not None else "no shared output at _unpack"
            _log(f"check sum8_add M={fused.shape[0]}: pointer handshake {'ok' if ok_ptr else 'MISMATCH'}; {st}")
            if ok_ptr and st.startswith("exact=True"):
                S["sa_verified"] = S.get("sa_verified", 0) + 1
                if S["sa_verified"] >= max(1, _int("GLM_PF3_CHECK", 0)):
                    S["sumadd_armed"] = True
                    _log(f"sum8_add fast path ARMED after {S['sa_verified']} exact verification(s)")
            else:
                S["sumadd_disabled"] = True
                _log("sum8_add DISABLED for this process: stays on the stock path")
        return shared, fused
    _unpack._glm_pf3 = True
    return _unpack


def make_apply_quant_wrapper(orig):
    def _apply_quant_method(self, *args, **kwargs):
        prev = getattr(_TLS, "runner", None)
        _TLS.runner = self
        try:
            return orig(self, *args, **kwargs)
        finally:
            _TLS.runner = prev
    _apply_quant_method._glm_pf3 = True
    return _apply_quant_method


# ------------------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------------------
def install_marlin(mod):
    if not getattr(mod._fused_marlin_moe, "_glm_pf3", False):
        mod._fused_marlin_moe = make_fused_wrapper(mod._fused_marlin_moe)
    cls = getattr(mod, "MarlinExperts", None)
    if cls is not None and not getattr(cls.moe_sum, "_glm_pf3", False):
        cls.moe_sum = make_moe_sum_wrapper(cls.moe_sum)
    _log("marlin_moe hooks installed (_fused_marlin_moe, MarlinExperts.moe_sum)")


def install_runner(mod):
    if not getattr(mod._unpack, "_glm_pf3", False):
        mod._unpack = make_unpack_wrapper(mod._unpack)
    cls = mod.MoERunner
    if not getattr(cls._apply_quant_method, "_glm_pf3", False):
        cls._apply_quant_method = make_apply_quant_wrapper(cls._apply_quant_method)
    _log("moe_runner hooks installed (_unpack, MoERunner._apply_quant_method)")


HOOKS = {MARLIN_MOD: install_marlin, RUNNER_MOD: install_runner}


def register():
    if not installed():
        return
    for name, fn in HOOKS.items():
        if name in sys.modules:
            fn(sys.modules[name])
    pending = {n: f for n, f in HOOKS.items() if n not in sys.modules}
    if not pending:
        return

    class _Finder:
        def find_spec(self, name, path, target=None):
            if name not in pending:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                if len(pending) > 1:
                    sys.meta_path.insert(0, self)
            fn = pending.pop(name)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module)
                _fn(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

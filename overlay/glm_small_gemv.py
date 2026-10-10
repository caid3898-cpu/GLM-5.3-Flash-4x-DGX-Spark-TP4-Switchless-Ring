# SPDX-License-Identifier: Apache-2.0
"""GLM_GATE_GEMV / GLM_ROUTER_GEMV: split-K Triton kernels for the two skinny fp32-output GEMMs of a decode step.

Tony v11 image (vLLM 0.1.dev20051+g487ecf187), SM121 (GB10). Both products are [M, 4096] bf16 activations times a
bf16 weight stored [N, 4096] (K contiguous), accumulated in fp32:

  gate    Indexer.forward (vllm.models.glm5next.nvidia.attention = overlay/glm5next_attention.py:354, image :336)
          `weights = torch.mm(hidden_states.float(), self._wp_fp32)`, _wp_fp32 = wk_weights_proj.weight[128:].t()
          .float() [4096, 32]. cuBLAS runs a bf16->fp32 copy (2.3 us) + gemmSN_NN grid [1,1,1] (66.8 us), 11 MLA
          layers, serial on the main stream. The fast path reads the bf16 rows wk_weights_proj.weight[128:] [32, 4096]
          directly (the same values: _wp_fp32 is their exact fp32 upcast) and writes fp32.
  router  GateLinear.forward (fused_moe/router/gate_linear.py:179), called from MoERunner._forward_impl
          (fused_moe/runner/moe_runner.py:863; the model's own call at glm5next_model.py:266 is the one
          GLM_ROUTER_DEDUP skips). out_dtype = float32 (moe_router_dtype), weight bf16 [288, 4096]. SM121 is not
          SM90/SM10x, so GateLinear takes tier 6: bf16 F.linear (cutlass wmma + cublasLt splitKreduce, bf16 out) and
          `.to(float32)` (a copy kernel). The logits are therefore bf16-rounded values in an fp32 tensor.

Kernel: pass 1 splits K into fixed K_CHUNK slices, one CTA per (N tile, K slice, M tile), and writes fp32 partials
[SPLIT, M, N]; pass 2 sums the partials in slice order 0, 1, ..., SPLIT-1 and writes [M, N] fp32. No atomics, no
autotune, tile shapes fixed per kernel and independent of M: a row's result does not depend on the other rows or on
M (batch invariant) and is bit-reproducible. Every bf16 x bf16 product is exact in fp32, so the result differs from
cuBLAS only by summation order (gate) or, for the router, by which fp32 sum is rounded to bf16. Numerics-changing:
gate with a KLD/qeval run, not bitwise.

  GLM_GATE_GEMV=0|1        indexer head gate (FMA path, CUDA cores, like the fp32 gemmSN it replaces)
  GLM_ROUTER_GEMV=0|1|fp32 router logits (tl.dot path, like the wmma kernel it replaces). 1: rounded to bf16 then
                           widened, the same value set stock tier 6 returns; fp32: the fp32 accumulator itself (the
                           GateLinear tier 5 / GLM_ROUTER_FP32OUT semantics; changes routing numerics further)
  GLM_SMALL_GEMV_GRAPH_ONLY=1   use the kernels only while a CUDA graph is captured (eager small steps stay stock);
                           eager calls that the capture would route to a kernel prewarm it once (prewarm()), so no
                           Triton compile / module load happens inside a capture
  GLM_SMALL_GEMV_GATE_CFG / GLM_SMALL_GEMV_ROUTER_CFG   tile overrides, e.g. "k_chunk=512,num_warps=8"; read once
                           at import, identical on every rank (they are part of the numerics)
Fallback to the stock op for M > 32, M == 0, non-bf16 input or weight, a non-unit inner stride, an unexpected
weight shape, or K not a multiple of K_CHUNK. The switches are read per call through overlay/glm_ab.py when it is
armed (keys GLM_GATE_GEMV / GLM_ROUTER_GEMV, kind "raw"), so both variants can share one boot.

CUDA graphs: no host sync; the partial and output buffers come from torch.empty on the current stream (the graph
pool inside a capture), sized by M only, so a captured shape always replays the same allocation. The first call
per specialisation compiles in the eager warm-up that precedes each capture.

Order with other GateLinear wrappers: a call flagged by glm_router_dedup (`_glm_skip_once`) is passed through, so
the placeholder still skips the GEMM whichever wrapper is outermost. With GLM_ROUTER_FP32OUT also on, the wrapper
registered later wins for M <= 32 (sitecustomize registers this module after glm_kda_stash).
"""
from __future__ import annotations

import __future__
import ast
import collections
import functools
import inspect
import os
import sys
import textwrap

TARGET_ATTN = "vllm.models.glm5next.nvidia.attention"
TARGET_GATE = "vllm.model_executor.layers.fused_moe.router.gate_linear"
_OFF = ("", "0", "off", "false", "no")
MAX_M = 32
# The gate kernel beats the stock gemv up to M = 16 (M=4: 68 -> 10.4 us cold; M=16: 67 -> 13.3 us) but loses at
# M = 32 (10.1 vs 16.9 us: cuBLAS leaves its single-CTA gemv there). GB10 GPU test, 2026-09-28.
GATE_MAX_M = 16

GATE_STMT = "weights = torch.mm(hidden_states.float(), self._wp_fp32)"
GATE_CALL = "weights = _glm_small_gemv_gate(self, hidden_states)"

Cfg = collections.namedtuple("Cfg", "block_m block_n block_k k_chunk num_warps num_stages use_dot")
# gate: N = 32, CUDA-core FMA over a [BLOCK_M, BLOCK_N, BLOCK_K] product (64 fp32 regs/thread at 4 warps)
GATE_CFG = Cfg(block_m=8, block_n=32, block_k=32, k_chunk=128, num_warps=4, num_stages=2, use_dot=False)
# router: N = 288 -> 9 N tiles x 16 K slices = 144 CTAs; one 32-row M tile covers every M <= 32
ROUTER_CFG = Cfg(block_m=32, block_n=32, block_k=64, k_chunk=256, num_warps=4, num_stages=3, use_dot=True)
REDUCE_BLOCK = 256

S = {"gate_fast": 0, "gate_stock": 0, "router_fast": 0, "router_stock": 0, "logged": set()}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-small-gemv: {msg}\n")


def _once(key: str, msg: str) -> None:
    if key not in S["logged"]:
        S["logged"].add(key)
        _log(msg)


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _mode(name: str) -> str:
    raw = str(env(name, "0")).strip().lower()
    return "0" if raw in _OFF else raw


def parse_cfg(text: str | None, base: Cfg) -> Cfg:
    """'key=value,...' over a base Cfg; validates the shape rules the kernels rely on."""
    cfg = base
    for item in (text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in Cfg._fields:
            raise ValueError(f"unknown tile key {key!r}; known {Cfg._fields}")
        value = value.strip().lower()
        cfg = cfg._replace(**{key: value in ("1", "true", "on") if key == "use_dot" else int(value)})
    for key in ("block_m", "block_n", "block_k", "num_warps"):
        v = getattr(cfg, key)
        if v < 1 or v & (v - 1):
            raise ValueError(f"{key}={v} must be a power of two")
    if cfg.k_chunk % cfg.block_k:
        raise ValueError(f"k_chunk={cfg.k_chunk} must be a multiple of block_k={cfg.block_k}")
    if cfg.use_dot and min(cfg.block_m, cfg.block_n, cfg.block_k) < 16:
        raise ValueError("use_dot needs block_m, block_n, block_k >= 16")
    return cfg


GATE_CFG = parse_cfg(os.environ.get("GLM_SMALL_GEMV_GATE_CFG"), GATE_CFG)
ROUTER_CFG = parse_cfg(os.environ.get("GLM_SMALL_GEMV_ROUTER_CFG"), ROUTER_CFG)


def plan(M: int, N: int, K: int, cfg: Cfg):
    """(pass-1 grid, split, pass-2 grid) for out[M, N] = x[M, K] @ w[N, K]^T; None when K does not tile."""
    if K % cfg.k_chunk:
        return None
    split = K // cfg.k_chunk
    grid1 = (-(-N // cfg.block_n), split, -(-M // cfg.block_m))
    grid2 = (-(-(M * N) // REDUCE_BLOCK),)
    return grid1, split, grid2


# -- kernels (built on first use: the module must import without torch/triton) -----------------------------------

@functools.lru_cache(maxsize=None)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["M"])
    def _splitk_partial(x_ptr, w_ptr, p_ptr, M, N, stride_xm, stride_wn,
                        K_CHUNK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, USE_DOT: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_m = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        x_rows = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
        w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k0 = pid_s * K_CHUNK
        for kk in range(0, K_CHUNK, BLOCK_K):
            offs_k = k0 + kk + tl.arange(0, BLOCK_K)
            x = tl.load(x_rows + offs_k[None, :], mask=mask_m[:, None], other=0.0)
            w = tl.load(w_rows + offs_k[None, :], mask=mask_n[:, None], other=0.0)
            if USE_DOT:
                acc = tl.dot(x, tl.trans(w), acc=acc, out_dtype=tl.float32)
            else:
                prod = x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
                acc += tl.sum(prod, axis=2)
        p = p_ptr + (pid_s * M + offs_m[:, None]).to(tl.int64) * N + offs_n[None, :]
        tl.store(p, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit(do_not_specialize=["MN"])
    def _splitk_reduce(p_ptr, o_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr, ROUND_BF16: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < MN
        acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
        for s in tl.static_range(1, SPLIT):  # fixed order: ((p0 + p1) + p2) + ...
            acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
        if ROUND_BF16:
            acc = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        tl.store(o_ptr + offs, acc, mask=mask)

    return _splitk_partial, _splitk_reduce


def skinny_gemm(x, w, cfg: Cfg, round_bf16: bool = False):
    """fp32 out[M, N] = x[M, K] @ w[N, K]^T with the fixed split-K order. Caller checks eligible() first."""
    import torch
    partial_k, reduce_k = _kernels()
    M, K = x.shape
    N = w.shape[0]
    grid1, split, grid2 = plan(M, N, K, cfg)
    part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    partial_k[grid1](x, w, part, M, N, x.stride(0), w.stride(0),
                     K_CHUNK=cfg.k_chunk, BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                     USE_DOT=cfg.use_dot, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    reduce_k[grid2](part, out, M * N, SPLIT=split, BLOCK=REDUCE_BLOCK, ROUND_BF16=round_bf16, num_warps=4)
    return out


def eligible(x, w, cfg: Cfg) -> str | None:
    """None when the fast path applies, else the reason for the stock path."""
    import torch
    if x.dim() != 2 or w.dim() != 2:
        return "rank"
    M, K = x.shape
    if not 1 <= M <= MAX_M:
        return "M"
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return "dtype"
    if w.shape[1] != K or K % cfg.k_chunk:
        return "K"
    if x.stride(1) != 1 or w.stride(1) != 1:
        return "stride"
    if not x.is_cuda or x.device != w.device:
        return "device"
    return None


def _scope_ok() -> bool:
    if str(env("GLM_SMALL_GEMV_GRAPH_ONLY", "0")).strip().lower() in _OFF:
        return True
    import torch
    return torch.cuda.is_current_stream_capturing()


_PREWARMED: set = set()


def prewarm(x, w, cfg: Cfg, round_bf16: bool = False) -> None:
    """GLM_SMALL_GEMV_GRAPH_ONLY=1: eager calls take the stock op, so without this the first launch of each Triton
    specialisation (JIT compile + module load) would happen INSIDE the CUDA graph capture. Called on the eager path
    (the warm-up runs vLLM does before capturing each shape) for every call the fast path would take while
    capturing: runs the kernels once on the same inputs, discards the result. The key covers everything Triton
    specialises on here (constexprs via cfg / round flag, N, K, and 16-divisibility of strides and pointers; M is
    do_not_specialize), so a later capture launches an already-loaded kernel."""
    key = (cfg, round_bf16, w.shape[0], x.shape[1], x.stride(0) % 16 == 0, w.stride(0) % 16 == 0,
           x.data_ptr() % 16 == 0, w.data_ptr() % 16 == 0, str(x.device))
    if key in _PREWARMED:
        return
    skinny_gemm(x, w, cfg, round_bf16=round_bf16)
    _PREWARMED.add(key)
    _once(f"prewarm_{len(_PREWARMED)}", f"prewarmed split-K kernels before capture (GRAPH_ONLY): {cfg}, "
                                        f"N={w.shape[0]} K={x.shape[1]} round_bf16={round_bf16}")


# -- gate (DSA indexer head weights) ---------------------------------------------------------------------------

def _gate_rows(layer):
    """wk_weights_proj.weight[head_dim:] (the gate rows), cached per layer and re-derived if the weight moved."""
    wg = layer.__dict__.get("_glm_gate_rows")
    w = layer.wk_weights_proj.weight
    if wg is None or wg.data_ptr() != w.data_ptr() + layer.head_dim * w.stride(0) * w.element_size():
        wg = w.data[layer.head_dim:] if w.dim() == 2 and w.shape[0] == layer.head_dim + layer.n_head else None
        layer.__dict__["_glm_gate_rows"] = wg
    return wg


def gate_weights(layer, hidden_states):
    """Replacement for the GATE_STMT line of Indexer.forward. `layer._wp_fp32` is already built by then."""
    import torch
    gate_on = _mode("GLM_GATE_GEMV") != "0"
    if gate_on and not _scope_ok():  # GRAPH_ONLY and eager: stock result, but compile what the capture will run
        wg = _gate_rows(layer)
        if wg is not None and eligible(hidden_states, wg, GATE_CFG) is None and hidden_states.shape[0] <= GATE_MAX_M:
            prewarm(hidden_states, wg, GATE_CFG)
    if gate_on and _scope_ok():
        wg = _gate_rows(layer)
        why = "weight shape" if wg is None else eligible(hidden_states, wg, GATE_CFG)
        if why is None and hidden_states.shape[0] > GATE_MAX_M:
            why = "M"
        if why is None:
            S["gate_fast"] += 1
            _once("gate_fast", f"gate: split-K kernel, M={hidden_states.shape[0]} N={wg.shape[0]} "
                               f"K={wg.shape[1]} {GATE_CFG}")
            return skinny_gemm(hidden_states, wg, GATE_CFG)
        _once("gate_stock_" + why, f"gate: stock torch.mm ({why}; first case M={hidden_states.shape[0]})")
    S["gate_stock"] += 1
    return torch.mm(hidden_states.float(), layer._wp_fp32)


def rewrite_indexer_forward(src: str) -> str:
    """Swap the one GATE_STMT line for GATE_CALL, keeping every other line (and the line count) unchanged."""
    lines = src.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == GATE_STMT]
    if len(hits) != 1:
        raise LookupError(f"expected the head-gate line exactly once in Indexer.forward, found {len(hits)}")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + GATE_CALL + ("\n" if lines[i].endswith("\n") else "")
    return "".join(lines)


def install_indexer(mod) -> None:
    cls = mod.Indexer
    if getattr(cls, "_glm_small_gemv", False):
        return
    fn = cls.forward
    try:
        if hasattr(fn, "__wrapped__"):
            raise LookupError("Indexer.forward is already wrapped by another hook; rewriting it would drop that hook")
        if fn.__code__.co_freevars:
            raise LookupError(f"Indexer.forward is a closure ({fn.__code__.co_freevars}); cannot re-create it")
        lines, first = inspect.getsourcelines(fn)
        new_src = rewrite_indexer_forward(textwrap.dedent("".join(lines)))
        tree = ast.parse(new_src)
        ast.increment_lineno(tree, first - 1)
        flags = __future__.annotations.compiler_flag if getattr(mod, "annotations", None) is \
            __future__.annotations else 0
        code = compile(tree, inspect.getsourcefile(fn) or "<glm_small_gemv>", "exec", flags=flags, dont_inherit=True)
    except (LookupError, OSError, SyntaxError, TypeError) as exc:
        _log(f"gate: NOT installed, Indexer.forward kept stock: {exc}")
        return
    mod._glm_small_gemv_gate = gate_weights
    ns: dict = {}
    exec(code, mod.__dict__, ns)  # globals = the attention module, as for the original def
    new_fn = functools.update_wrapper(ns[fn.__name__], fn)
    cls.forward = new_fn
    cls._glm_small_gemv = True
    _log(f"gate: Indexer.forward head gate -> split-K kernel when GLM_GATE_GEMV is on ({mod.__name__}:{first})")


# -- router (MoE gate logits) ----------------------------------------------------------------------------------

def router_ok(layer, x) -> str | None:
    import torch
    if layer.out_dtype != torch.float32:
        return "out_dtype"
    if getattr(layer, "bias", None) is not None:
        return "bias"
    return eligible(x, layer.weight, ROUTER_CFG)


def install_gate_linear(mod) -> None:
    cls = mod.GateLinear
    if getattr(cls, "_glm_small_gemv", False):
        return
    cls._glm_small_gemv = True
    orig = cls.forward

    def forward(self, x, *args, **kwargs):
        # a glm_router_dedup placeholder call passes straight through to the wrapped chain
        if not args and not kwargs and not self.__dict__.get("_glm_skip_once"):
            mode = _mode("GLM_ROUTER_GEMV")
            if mode != "0" and not _scope_ok() and router_ok(self, x) is None:
                prewarm(x, self.weight, ROUTER_CFG, round_bf16=mode != "fp32")  # GRAPH_ONLY, eager: compile only
            if mode != "0" and _scope_ok():
                why = router_ok(self, x)
                if why is None:
                    S["router_fast"] += 1
                    _once("router_fast", f"router: split-K kernel ({'bf16-rounded' if mode != 'fp32' else 'fp32'}"
                                         f" logits), M={x.shape[0]} E={self.weight.shape[0]} {ROUTER_CFG}")
                    return skinny_gemm(x, self.weight, ROUTER_CFG, round_bf16=mode != "fp32"), None
                _once("router_stock_" + why, f"router: stock GateLinear ({why}; first case M={x.shape[0]})")
        S["router_stock"] += 1
        return orig(self, x, *args, **kwargs)

    functools.update_wrapper(forward, orig)
    cls.forward = forward
    _log("router: GateLinear logits -> split-K kernel when GLM_ROUTER_GEMV is on")


HOOKS = {}


def register() -> None:
    import importlib.abc
    import importlib.util

    if os.environ.get("GLM_GATE_GEMV", "0").strip().lower() not in _OFF:
        HOOKS[TARGET_ATTN] = install_indexer
    if os.environ.get("GLM_ROUTER_GEMV", "0").strip().lower() not in _OFF:
        HOOKS[TARGET_GATE] = install_gate_linear
        if os.environ.get("GLM_ROUTER_FP32OUT", "0").strip().lower() not in _OFF:
            _log("GLM_ROUTER_FP32OUT is also on: the GateLinear wrapper registered last decides M <= 32 calls")
    if not HOOKS:
        return
    for name, fn in HOOKS.items():
        if name in sys.modules:
            fn(sys.modules[name])

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
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

"""CPU check for overlay/glm_prefill_shard.py (GLM_PREFILL_SHARD), four TP ranks emulated by threads.

The stock side is the image's own code: Glm5NextModel.forward and Glm5NextDecoderLayer.forward are
lifted from the extracted image source (GLM_IMAGE_SRC, default: the 2026-09-18 extraction in the
sparks workspace) and run on a toy model with the engine's structure:
  * residual [M, 4, D] with row-local mHC ops (torch reference math) whose pre-norm GEMM goes
    through fake tilelang modules with the engine's dispatch: a >= 1024-row variant and a generic
    variant that sum in different orders, and an optional DeepGEMM-like split-K keyed on
    compute_num_split(cdiv(rows, 64)), so a gate keyed on the owned quarter changes bits;
  * a row-mixing "attention" (causal running mean) and an MLP/MoE whose output projections are
    TP partials reduced unless reduce_results / skip_final_all_reduce say otherwise;
  * DFlash2 aux capture on two layers, a dense first layer, the last layer's hc_post + contract.
The fake process group sums all-reduce in rank order (bf16 rounding per hop) and reduce-scatter in
a rotated order, like NCCL's ring, unless rs_order="same".

Checks, for chunks of 4096, 2304 (the real chunk at BATCHED_TOKENS=4096), 2052 (odd quarters):
  * EXACT == stock bit for bit, on every rank (with the logical-row kernel keying);
  * shard with an RS that sums like the AR == stock bit for bit (the plumbing is exact);
  * shard with a ring-order RS != stock, only through summation order, and small;
  * comm behaves the same way;
  * without the keying, EXACT is NOT bit-exact at 2304 (the quarter falls under the 1024-row gate),
    and CHECK catches it, marks the row count bad and keeps the stock result for that chunk;
  * collective counts: rs = 2L, ag = 2L + aux (the first attention needs no gather);
  * validation refuses the unqualified layouts; deferral attributes are restored on exceptions.

  python3 tests/test_glm_prefill_shard.py            (needs torch)
"""
from __future__ import annotations

import os
import sys
import threading
import types
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
os.environ.setdefault("GLM_PREFILL_SHARD", "1")

import importlib.util  # noqa: E402

import torch  # noqa: E402

import glm_prefill_hooks as hooks  # noqa: E402
import glm_prefill_shard as ps  # noqa: E402

# Root holding vllm/models/glm5next/...: the vLLM installed in the image by default, GLM_IMAGE_SRC for a source tree.
IMAGE_SRC = os.environ.get("GLM_IMAGE_SRC") or os.path.dirname(
    os.path.dirname(importlib.util.find_spec("vllm").origin))
MODEL_PY = os.path.join(IMAGE_SRC, "vllm/models/glm5next/nvidia/model.py")

W = 4
D = 64
N = 4          # hc streams
BF = torch.bfloat16
torch.manual_seed(0)
torch.set_num_threads(1)


# ------------------------------------------------------------------------------------------
# fake TP group over threads
# ------------------------------------------------------------------------------------------
class ThreadGroup:
    def __init__(self, rs_order="ring"):
        self.world_size = W
        self._bar = threading.Barrier(W)
        self._slots = [None] * W
        self._tl = threading.local()
        self.rs_order = rs_order

    @property
    def rank_in_group(self):
        return self._tl.rank

    def bind(self, r):
        self._tl.rank = r

    def _exchange(self, t):
        self._slots[self.rank_in_group] = t.clone()
        self._bar.wait()
        vals = list(self._slots)
        self._bar.wait()
        return vals

    def all_reduce(self, x):
        vals = self._exchange(x)
        acc = vals[0].clone()
        for v in vals[1:]:
            acc = acc + v                       # rounds to the dtype at every hop
        return acc

    def all_gather(self, x, dim=0):
        assert dim == 0
        return torch.cat(self._exchange(x.contiguous()), 0)

    def reduce_scatter(self, x, dim=0):
        assert dim == 0
        vals = self._exchange(x.contiguous())
        r, s = self.rank_in_group, x.shape[0] // W
        parts = [v[r * s:(r + 1) * s] for v in vals]
        order = list(range(W)) if self.rs_order == "same" else [(r + 1 + k) % W for k in range(W)]
        acc = parts[order[0]].clone()
        for o in order[1:]:
            acc = acc + parts[o]
        return acc


def run_ranks(group, fn):
    out, errs = [None] * W, []

    def body(r):
        group.bind(r)
        try:
            out[r] = fn(r)
        except BaseException as exc:  # noqa: BLE001
            errs.append((r, exc))
            group._bar.abort()

    ts = [threading.Thread(target=body, args=(r,)) for r in range(W)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0][1]
    return out


# ------------------------------------------------------------------------------------------
# fake tilelang modules: the engine's pre-norm GEMM dispatch, with order-distinct variants
# ------------------------------------------------------------------------------------------
def _cdiv(a, b):
    return -(-a // b)


tk = types.ModuleType("fake_tilelang_kernels")
tl = types.ModuleType("fake_tilelang")
USE_DG = {"on": False}
KEYING = {"on": True}


def _gemm(x, fn, n_splits, variant):
    """[T, K] bf16 x [Nout, K] f32 -> ([n_splits, T, Nout], [n_splits, T]); row-local."""
    xf = x.float()
    T, K = xf.shape
    ks = K // n_splits
    outs, sqs = [], []
    for s in range(n_splits):
        xs, fs = xf[:, s * ks:(s + 1) * ks], fn[:, s * ks:(s + 1) * ks]
        prod = xs[:, None, :] * fs[None]
        if variant == "block_m":
            o = prod.sum(-1)
            q = (xs * xs).sum(-1)
        else:  # generic: blocked pairwise order -> different rounding
            o = prod.view(T, fn.shape[0], -1, 8).sum(-1).sum(-1)
            q = (xs * xs).view(T, -1, 8).sum(-1).sum(-1)
        outs.append(o)
        sqs.append(q)
    return torch.stack(outs), torch.stack(sqs)


def _block_m(x, fn, out, sqrsum, hidden, hc, n_out, n_thr, tile_n, block_m):
    o, q = _gemm(x, fn, 1, "block_m")
    out.copy_(o)
    sqrsum.copy_(q)


def _generic(x, fn, out, sqrsum, hidden, hc, n_out, n_thr, tile_n, n_splits):
    o, q = _gemm(x, fn, n_splits, "generic")
    out.copy_(o)
    sqrsum.copy_(q)


def _compute_num_split(block_k, k, grid_size):
    split_k = 48 // grid_size
    if k is not None:
        split_k = min(split_k, _cdiv(k, block_k) // 4)
    return max(split_k, 1)


tk.hc_prenorm_gemm_block_m_tilelang = _block_m
tk.hc_prenorm_gemm_tilelang = _generic
tk.compute_num_split = _compute_num_split


def _stock_prenorm(x, fn, out, sqrsum, hidden_size, hc_mult, tile_n=12, n_thr=512, n_splits=1):
    # the engine's gates (kernels/mhc/tilelang.py, vLLM 487ecf187), keyed on x.shape[0]
    default = tile_n == 12 and n_thr == 512
    if n_splits == 1 and default and x.shape[0] >= 1024:
        return tk.hc_prenorm_gemm_block_m_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, 2)
    if n_splits == 1 and default and x.shape[0] < 128 and x.shape[1] % 1024 == 0:
        return tk.hc_prenorm_gemm_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], 1024, 4, n_splits)
    return tk.hc_prenorm_gemm_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, n_splits)


def reset_kernels(keying: bool):
    tl._tilelang_hc_prenorm_gemm = _stock_prenorm
    tk.compute_num_split = _compute_num_split
    if keying:
        ps.install_kernel_keying(tl, tk)


def _pre_from_residual(res, fn, scale, base, norm_w, norm_eps):
    """mhc pre (torch reference math) with the GEMM through tl / tk module attributes."""
    T = res.shape[0]
    K = N * D
    x2 = res.reshape(T, K)
    if USE_DG["on"]:
        # engine: compute_num_split(64, 16384, cdiv(T, 64)); block_k 16 here so the toy K=256 can split
        n_splits = tk.compute_num_split(16, K, _cdiv(T, 64))
        o, q = _gemm(x2, fn, n_splits, "block_m")
    else:
        n_splits = 1
        o = torch.empty(1, T, fn.shape[0])
        q = torch.empty(1, T)
        tl._tilelang_hc_prenorm_gemm(x2, fn, o, q, D, N)
    mixes = torch.zeros(T, fn.shape[0])
    sq = torch.zeros(T)
    for s in range(o.shape[0]):
        mixes = mixes + o[s]
        sq = sq + q[s]
    mixes = mixes * torch.rsqrt(sq[:, None] / K + 1e-6)
    pre = torch.sigmoid(mixes[:, :N] * scale[0] + base[:N]) + 1e-6
    post = torch.sigmoid(mixes[:, N:2 * N] * scale[1] + base[N:2 * N]) * 2.0
    comb = torch.softmax(mixes[:, 2 * N:].view(T, N, N) * scale[2] + base[2 * N:].view(1, N, N), -1)
    for _ in range(3):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    li = (pre.unsqueeze(-1) * res.float()).sum(1)
    li = li * torch.rsqrt(li.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_w.float()
    return post.view(T, N, 1), comb, li.to(BF)


def _post(x, res, post, comb):
    mixed = torch.einsum("tij,tih->tjh", comb.float(), res.float())
    return (mixed + post.float() * x.unsqueeze(-2).float()).to(BF)


# ------------------------------------------------------------------------------------------
# toy engine with the image's structure
# ------------------------------------------------------------------------------------------
class OProj:
    """RowParallelLinear stand-in: per-rank weight slice, reduce unless reduce_results is off."""

    def __init__(self, group, seed):
        g = torch.Generator().manual_seed(seed)
        self.w = [torch.randn(D, D, generator=g) * 0.05 for _ in range(W)]
        self.group, self.reduce_results, self.tp_size, self.bias = group, True, W, None

    def __call__(self, x):
        r = self.group.rank_in_group
        out = (x.float() @ self.w[r]).to(BF)
        if self.reduce_results:
            out = self.group.all_reduce(out)
        return out, None


class Attn:
    def __init__(self, group, seed, kda, idx):
        self.o_proj = OProj(group, seed)
        self.prefix = f"model.layers.{idx}.self_attn"
        if not kda:
            self.mla_attn = NS(o_proj=self.o_proj)

    def __call__(self, hidden_states, positions):
        h = hidden_states.float()
        core = (torch.cumsum(h, 0) / (positions.float()[:, None] + 1.0)).to(BF)
        return self.o_proj(core)[0]


class Runner:
    def __init__(self, group, seed):
        self.moe_config = NS(skip_final_all_reduce=False, is_sequence_parallel=False, tp_size=W,
                             hidden_dim=D, moe_parallel_config=NS(use_all2all_kernels=False), dp_size=1)
        self._fused_output_is_reduced = False
        self.routed_output_transform = None
        self.proj = OProj(group, seed)
        self.group = group

    def __call__(self, hidden_states, router_logits):
        gate = torch.sigmoid(router_logits.float()[:, :1])
        self.proj.reduce_results = False
        part = (self.proj(hidden_states)[0].float() * gate).to(BF)
        if not self.moe_config.skip_final_all_reduce:
            part = self.group.all_reduce(part)
        return part


class MoE:
    def __init__(self, group, seed):
        self.experts = Runner(group, seed)
        self.gate_w = torch.randn(D, 8, generator=torch.Generator().manual_seed(seed + 7))
        self.is_sequence_parallel = False

    def __call__(self, hidden_states, already_sequence_parallel=False):
        logits = hidden_states.float() @ self.gate_w
        return self.experts(hidden_states=hidden_states, router_logits=logits)


class MLP:
    def __init__(self, group, seed):
        self.down_proj = OProj(group, seed)

    def __call__(self, x):
        return self.down_proj(torch.nn.functional.silu(x.float()).to(BF))[0]


class Norm:
    def __init__(self):
        self.weight = NS(data=torch.linspace(0.5, 1.5, D).to(BF))
        self.variance_epsilon = 1e-6

    def __call__(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight.data.float()).to(BF)


class Layer:
    def __init__(self, group, idx, L):
        g = torch.Generator().manual_seed(100 + idx)
        self.layer_idx, self.num_hidden_layers, self.n = idx, L, N
        self.mhc, self.is_mtp_layer, self.is_sequence_parallel = True, False, False
        self.layer_kind = "kda" if idx % 4 != 3 else "mla"
        self.self_attn = Attn(group, 10 + idx, self.layer_kind == "kda", idx)
        self._mlp_is_moe = idx > 0
        self.mlp = MoE(group, 20 + idx) if self._mlp_is_moe else MLP(group, 20 + idx)
        self.hidden_size = D
        mix = (2 + N) * N
        self.hc_attn_fn = torch.randn(mix, N * D, generator=g) * 0.05
        self.hc_attn_base = torch.randn(mix, generator=g) * 0.1
        self.hc_attn_scale = torch.rand(3, generator=g) + 0.5
        self.hc_ffn_fn = torch.randn(mix, N * D, generator=g) * 0.05
        self.hc_ffn_base = torch.randn(mix, generator=g) * 0.1
        self.hc_ffn_scale = torch.rand(3, generator=g) + 0.5
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # the engine's op signatures (image model.py), toy math
    def hc_pre(self, x, hc_fn, hc_scale, hc_base, norm_weight=None, norm_eps=0.0):
        return _pre_from_residual(x, hc_fn, hc_scale, hc_base, norm_weight, norm_eps)

    def hc_post(self, x, residual, post, comb):
        return _post(x, residual, post, comb)

    def hc_fused_post_pre(self, x, residual, post, comb, hc_fn, hc_scale, hc_base, norm_weight=None, norm_eps=0.0):
        res = _post(x, residual, post, comb)
        p, c, li = _pre_from_residual(res, hc_fn, hc_scale, hc_base, norm_weight, norm_eps)
        return res, p, c, li


class Model:
    def __init__(self, group, L=6, aux=(2, 4)):
        g = torch.Generator().manual_seed(1)
        self.embed = torch.randn(512, D, generator=g).to(BF)
        self.layers = [Layer(group, i, L) for i in range(L)]
        self.start_layer, self.end_layer = 0, L
        self._active_layers = self.layers
        self.aux_hidden_state_layers = aux
        self.norm = Norm()
        self.is_sequence_parallel = False

    def embed_input_ids(self, ids):
        return self.embed[ids]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ------------------------------------------------------------------------------------------
# the image's stock forwards, lifted from the extracted source
# ------------------------------------------------------------------------------------------
def _hc_expand(x, n):
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def _hc_contract(x, n):
    return x.mean(dim=1)


def load_stock():
    g = {
        "torch": torch,
        "get_pp_group": lambda: NS(is_first_rank=True, is_last_rank=True),
        "hc_expand": _hc_expand,
        "hc_contract": _hc_contract,
        "sp_shard": None, "sp_all_gather": None, "sp_reduce_scatter": None,
        "IntermediateTensors": dict,
    }
    ns = {}
    for qual in ("Glm5NextModel.forward", "Glm5NextDecoderLayer.forward"):
        exec(compile(hooks.func_source(MODEL_PY, qual), MODEL_PY, "exec",
                     flags=__import__("__future__").annotations.compiler_flag, dont_inherit=True), g, ns)
        (Model if qual.startswith("Glm5NextModel") else Layer).forward = ns["forward"]


OPS = NS(hc_expand=_hc_expand, hc_contract=_hc_contract)


def run(M, mode, exact=False, rs_order="ring", keying=True, dg=False, check=False, L=6, pad=False):
    group = ThreadGroup(rs_order)
    USE_DG["on"] = dg
    reset_kernels(keying)
    models = [Model(group, L) for _ in range(W)]   # one per rank, as in the engine (same weights)
    ids = torch.randint(0, 512, (M,), generator=torch.Generator().manual_seed(M))
    pos = torch.arange(M)
    rt = ps._Runtime()

    def body(r):
        model = models[r]
        if mode == "stock":
            out = model.forward(ids, pos, None)
            return out, None
        own = ps.Ownership(ps.Coll(group), M, mode, exact, pad=pad)
        checker = ps._check_fused(own, rt) if check else None
        out = ps.model_forward(model, own, ids, pos, None, OPS, checker)
        return out, own.counts

    res = run_ranks(group, body)
    return res, rt


def eq(a, b):
    ha, aa = a if isinstance(a, tuple) else (a, [])
    hb, ab = b if isinstance(b, tuple) else (b, [])
    return torch.equal(ha, hb) and len(aa) == len(ab) and all(torch.equal(x, y) for x, y in zip(aa, ab))


def maxdiff(a, b):
    return float((a[0].float() - b[0].float()).abs().max())


# ------------------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------------------
def test_stock_is_rank_consistent_and_row_local():
    (res, _) = run(2304, "stock")
    for r in range(1, W):
        assert eq(res[0][0], res[r][0]), "stock differs across ranks"


def test_exact_is_bit_identical():
    for M in (4096, 2304, 2052):
        stock, _ = run(M, "stock")
        for dg in (False, True):
            shard, _ = run(M, "shard", exact=True, dg=dg)
            ref = run(M, "stock", dg=dg)[0] if dg else stock
            for r in range(W):
                assert eq(shard[r][0], ref[r][0]), f"EXACT differs from stock at M={M} dg={dg} rank={r}"


def test_shard_plumbing_exact_with_same_order_rs():
    for M in (4096, 2304, 2052):
        stock, _ = run(M, "stock")
        shard, _ = run(M, "shard", rs_order="same")
        comm, _ = run(M, "comm", rs_order="same")
        for r in range(W):
            assert eq(shard[r][0], stock[r][0]), f"shard (same-order RS) differs at M={M}"
            assert eq(comm[r][0], stock[r][0]), f"comm (same-order RS) differs at M={M}"


def test_ring_rs_changes_only_rounding():
    M = 2304
    stock, _ = run(M, "stock")
    shard, _ = run(M, "shard")
    comm, _ = run(M, "comm")
    assert not eq(shard[0][0], stock[0][0]), "expected reduction-order differences with a ring RS"
    for r in range(1, W):
        assert eq(shard[r][0], shard[0][0]), "sharded result differs across ranks"
    d = maxdiff(shard[0][0], stock[0][0])
    assert 0 < d < 0.25, d
    assert eq(shard[0][0], comm[0][0]), "shard and comm use the same reduce-scatters on the same partials"


def test_keying_is_needed_and_check_catches_it():
    M = 2304   # full >= 1024 -> block_m; quarter 576 -> generic
    stock, _ = run(M, "stock")
    nokey, _ = run(M, "shard", exact=True, keying=False)
    assert not eq(nokey[0][0], stock[0][0]), "a gate keyed on the quarter should change bits"
    checked, rt = run(M, "shard", exact=True, keying=False, check=True)
    assert M in rt.bad_rows, "CHECK should mark the row count"
    # CHECK keeps the stock result for the checked op; later layers still run the unkeyed kernel,
    # so only the first sharded op is corrected; the row count then runs stock in the engine.
    ok, rt2 = run(M, "shard", exact=True, keying=True, check=True)
    assert not rt2.bad_rows and eq(ok[0][0], stock[0][0])
    # DeepGEMM-like split-K: stock M=2304 -> 1 split, quarter -> 5 splits
    stock_dg, _ = run(M, "stock", dg=True)
    nokey_dg, _ = run(M, "shard", exact=True, keying=False, dg=True)
    key_dg, _ = run(M, "shard", exact=True, keying=True, dg=True)
    assert not eq(nokey_dg[0][0], stock_dg[0][0])
    assert eq(key_dg[0][0], stock_dg[0][0])


def test_collective_counts():
    L, aux = 6, 2
    res, _ = run(2304, "shard", L=L)
    counts = res[0][1]
    assert counts["rs"] == 2 * L and counts["ag"] == 2 * L + aux and counts["aux"] == aux, counts
    res, _ = run(2304, "shard", exact=True, L=L)
    assert res[0][1]["ar"] == 2 * L and res[0][1]["rs"] == 0


def test_validation_and_restore():
    group = ThreadGroup()
    m = Model(group)
    assert ps.validate_model(m) == "model.layers.0.self_attn"
    m.layers[3].self_attn.mla_attn.o_proj = OProj(group, 99)
    try:
        ps.validate_model(m)
        raise AssertionError("MLA wrapper with a foreign o_proj must be refused")
    except RuntimeError:
        pass
    m = Model(group)
    m.layers[2].mlp.experts.moe_config.hidden_dim = D + 64
    try:
        ps.validate_model(m)
        raise AssertionError("padded MoE hidden dim must be refused")
    except RuntimeError:
        pass
    lin = OProj(group, 1)
    try:
        with ps._defer_linear(lin):
            assert lin.reduce_results is False
            raise ValueError
    except ValueError:
        pass
    assert lin.reduce_results is True
    run_ = Runner(group, 1)
    with ps._defer_moe(run_):
        assert run_.moe_config.skip_final_all_reduce is True
    assert run_.moe_config.skip_final_all_reduce is False


def test_metadata_gate():
    md = lambda **k: {"kda": NS(**{**dict(num_prefills=1, num_prefill_tokens=2304, num_decodes=0,  # noqa: E731
                                              num_spec_decodes=0), **k})}
    assert ps.pure_or_mixed_prefill(md(), "kda", 2304)
    assert not ps.pure_or_mixed_prefill(md(num_decodes=2), "kda", 2304)
    assert not ps.pure_or_mixed_prefill(md(num_prefills=0, num_prefill_tokens=0), "kda", 2304)
    assert not ps.pure_or_mixed_prefill(None, "kda", 2304)
    assert not ps.pure_or_mixed_prefill({"other": 1}, "kda", 2304)
    ps.MIXED = True
    try:
        assert ps.pure_or_mixed_prefill(md(num_decodes=2, num_prefill_tokens=2296), "kda", 2304)
    finally:
        ps.MIXED = False


def test_pad_rows_not_divisible_by_tp():
    """GLM_PREFILL_SHARD_PAD: row counts that are not a multiple of 4 (2049..2051, 2303, 4097)."""
    L, aux = 6, 2
    for M in (2049, 2050, 2051, 2303, 4097):
        try:
            run(M, "shard", L=L)
            raise AssertionError("an unpadded ownership of a non-multiple row count must be refused")
        except RuntimeError as exc:
            assert "not divisible" in str(exc), exc
        stock, _ = run(M, "stock", L=L)
        same, _ = run(M, "shard", rs_order="same", pad=True, L=L)
        exact, _ = run(M, "shard", exact=True, pad=True, L=L)
        ring, _ = run(M, "shard", pad=True, L=L)
        comm, _ = run(M, "comm", rs_order="same", pad=True, L=L)
        for r in range(W):
            h = same[r][0][0] if isinstance(same[r][0], tuple) else same[r][0]
            assert h.shape[0] == M, h.shape
            assert eq(same[r][0], stock[r][0]), f"padded shard (same-order RS) differs from stock at M={M} rank={r}"
            assert eq(exact[r][0], stock[r][0]), f"padded EXACT differs from stock at M={M} rank={r}"
            assert eq(comm[r][0], stock[r][0]), f"padded comm (same-order RS) differs at M={M} rank={r}"
            assert eq(ring[r][0], ring[0][0]), "padded ring shard differs across ranks"
        d = maxdiff(ring[0][0], stock[0][0])
        assert d < 0.25, d
        c = same[0][1]
        assert c["rs"] == 2 * L and c["ag"] == 2 * L + aux and c["padcopy"] == 2 * L, c
        assert exact[0][1]["ar"] == 2 * L and exact[0][1]["padcopy"] == 0, exact[0][1]
    # divisible row counts never copy
    res, _ = run(2304, "shard", pad=True, L=L)
    assert res[0][1]["padcopy"] == 0, res[0][1]


def test_pad_ownership_geometry():
    group = ThreadGroup()
    for M, want_real in ((2049, [513, 513, 513, 510]), (2050, [513, 513, 513, 511]), (2051, [513, 513, 513, 512]),
                         (2052, [513, 513, 513, 513])):
        reals = []
        for r in range(W):
            group.bind(r)
            own = ps.Ownership(ps.Coll(group), M, "shard", False, pad=True)
            reals.append(own.real)
            t = torch.arange(M, dtype=torch.float32)[:, None].repeat(1, 3)
            loc = own.local(t)
            assert loc.shape[0] == own.q
            assert torch.equal(loc[: own.real, 0], torch.arange(own.lo, own.lo + own.real, dtype=torch.float32))
            assert float(loc[own.real:].abs().sum()) == 0.0
        assert reals == want_real, (M, reals)


def test_per_call_switch_through_glm_ab():
    fake = types.ModuleType("glm_ab")
    fake.ACTIVE = True
    fake.KNOWN = {"GLM_PREFILL_SHARD": "raw", "GLM_PREFILL_SHARD_PAD": "bool"}
    fake.truthy = lambda v: v is not None and str(v).strip().lower() not in ("", "0", "off", "false", "no")
    cur = {"GLM_PREFILL_SHARD": "0", "GLM_PREFILL_SHARD_PAD": "0"}
    fake.env = lambda name, default=None: cur.get(name, default)
    sys.modules["glm_ab"] = fake
    try:
        assert ps.call_mode() == "off" and ps.call_pad() is False
        cur.update(GLM_PREFILL_SHARD="1", GLM_PREFILL_SHARD_PAD="1")
        assert ps.call_mode() == "shard" and ps.call_pad() is True
        cur.update(GLM_PREFILL_SHARD="comm")
        assert ps.call_mode() == "comm"
        fake.ACTIVE = False                       # harness off: the install-time values
        assert ps.call_mode() == ps.MODE and ps.call_pad() == ps.PAD
    finally:
        del sys.modules["glm_ab"]
    assert ps.call_mode() == ps.MODE


def main():
    load_stock()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()

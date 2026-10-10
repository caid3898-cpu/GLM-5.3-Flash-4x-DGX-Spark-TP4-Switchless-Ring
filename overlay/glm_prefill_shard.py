# SPDX-License-Identifier: Apache-2.0
"""mHC prefill sharding for GLM-5.3-Flash at TP4 (GLM_PREFILL_SHARD=comm | 1).

Credits
  * Jacopo Nardiello (jnardiello), E03 "FP8 mHC prefill sharding" in
    https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless
    (commit f320cd0, scripts/node/experiments/e03/source.patch): per-call TP
    reduction deferral, eager-prefill row ownership, reduce-scatter / all-gather around local
    mHC, DFlash2 aux gathers. Measured there: cold prefill +9-10 %, C2 +5 %.
  * FujitsuPolycom / SparkRing, the Apache-2.0 mHC prefill package his port is adapted from:
    https://github.com/FujitsuPolycom/sparkring/tree/61f277bd0c97fbff892668e12ea04a330a45fa01/runtime/glm53-spark-mtp3-mesh/performance/mhc-prefill
  * The mode layout (comm / shard / exact), the logical-row kernel keying and the drift guard
    follow our DeepSeek-V4.1 adapter (knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4,
    adapter/prefill_sp.py).
This file is an independent re-implementation for the tonyd2wild v11 image (vLLM 487ecf187,
tilelang mHC kernels, no B12X mHC); no code is copied from the packages above.

What it does
  Stock: every TP rank runs the mHC work (hc_post / hc_pre / Sinkhorn / RMSNorm over the
  [M, 4, 4096] residual) on all M rows of a prefill chunk, and every sublayer ends with an
  all-reduce of [M, 4096]. Here rank r owns rows [r*M/4, (r+1)*M/4) of the residual state:

    layer 0 hc_pre  : full rows (the embedding is already full on every rank), then keep the
                      owned rows of residual / post / comb
    attention       : all-gather x -> stock attention on all M rows, o_proj without its
                      all-reduce -> reduce-scatter
    hc post+pre     : stock fused op on the owned rows; every row-count-keyed kernel choice is
                      keyed on the FULL chunk (logical_rows)
    MLP / MoE       : all-gather x -> stock MLP/MoE on all M rows, final all-reduce deferred ->
                      reduce-scatter
    DFlash2 aux     : hc_post + contract on the owned rows -> all-gather
    exit            : all-gather the contracted rows, stock final norm

Modes
  GLM_PREFILL_SHARD=0      off (default): nothing is installed beyond the class patch, which
                           falls straight through to the stock forward.
  GLM_PREFILL_SHARD=comm   transport only: nothing is sharded; each sublayer's all-reduce becomes
                           reduce-scatter + all-gather. Isolates the collective cost.
  GLM_PREFILL_SHARD=1      sharded mHC (the port of E03).
  GLM_PREFILL_SHARD_EXACT=1  (with =1) every reduce-scatter is the stock all-reduce + keep the
                           owned rows. Bit-identical to stock by construction (see RESULTS.md),
                           checked by GLM_PREFILL_SHARD_CHECK and the greedy-hash fleet test.
  GLM_PREFILL_SHARD_MIN_ROWS  default 2048; smaller chunks or chunks not divisible by the TP size
                           run the stock forward.
  GLM_PREFILL_SHARD_MIXED=1  also shard chunks that carry decode / verify rows (default: pure
                           prefill only, as E03).
  GLM_PREFILL_SHARD_CHECK=1  on the first sharded chunk of every row count, run the stock fused
                           hc op on the gathered full rows next to the sharded one, compare bit
                           for bit, agree over TP; on any difference that row count runs stock
                           from then on (and this chunk keeps the stock result).
  GLM_PREFILL_SHARD_PAD=1  (2026-09-28, default off) also shard chunks whose row count is not a
                           multiple of the TP size (the last chunk of almost every prompt, and a
                           whole short prompt 3 times in 4). Ownership runs over ceil(M/4)*4 rows:
                           the last rank's owned state carries 1-3 zero rows, every reduce-scatter
                           gets a zero-padded copy of the partial (one [M, 4096] copy per sublayer,
                           only on such chunks), every all-gather is narrowed back to the M real
                           rows (a view). Attention and MLP/MoE still see exactly the M real rows.
                           EXACT stays bit-identical (stock all-reduce on the unpadded partial);
                           the RS mode has the RS mode's numerics.

In-boot A/B (overlay/glm_ab.py armed): GLM_PREFILL_SHARD (raw: 0|1|comm) and GLM_PREFILL_SHARD_PAD
(bool) are switchable keys (registered in sitecustomize). The union of the variants installs the
adapter; every prefill forward reads the runtime variant's values, so one boot can time stock,
comm and shard chunks side by side. EXACT / MIN_ROWS / MIXED / CHECK stay static.

Scope: eager forwards only (never under CUDA-graph capture or torch.compile), TP4, PP1, DP1,
no expert parallel, no sequence-parallel MoE, every layer an mHC layer. Every decision depends on
shape, forward mode, attention metadata and static config, so it is identical on every rank; the
flag itself is agreed over TP once at model construction.

Drift guard: sha256[:16] of the engine sources this relies on are checked at install; any
difference raises (refuses to boot) unless GLM_PREFILL_SHARD=0.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

try:  # the overlay directory is on PYTHONPATH in the container; tests import it directly
    import glm_prefill_hooks as _hooks
except ImportError:  # pragma: no cover
    from . import glm_prefill_hooks as _hooks  # type: ignore

logger = logging.getLogger("vllm.glm_prefill_shard")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "on", "true", "yes")


def parse_mode(raw) -> str:
    raw = "" if raw is None else str(raw).strip().lower()
    return "comm" if raw == "comm" else ("shard" if raw in ("1", "on", "true", "shard") else "off")


_RAW = os.environ.get("GLM_PREFILL_SHARD", "0").strip().lower()
MODE = parse_mode(_RAW)
EXACT = _flag("GLM_PREFILL_SHARD_EXACT")
MIN_ROWS = int(os.environ.get("GLM_PREFILL_SHARD_MIN_ROWS", "2048"))
MIXED = _flag("GLM_PREFILL_SHARD_MIXED")
CHECK = _flag("GLM_PREFILL_SHARD_CHECK")
PAD = _flag("GLM_PREFILL_SHARD_PAD")
TP = 4


def _ab():
    import sys
    ab = sys.modules.get("glm_ab")
    return ab if ab is not None and getattr(ab, "ACTIVE", False) else None


def call_mode() -> str:
    """Mode of this forward: the install MODE, or the runtime variant's value when glm_ab is armed and knows
    the key (rank-invariant: the switch is a collective at the same step on every rank)."""
    ab = _ab()
    if ab is None or "GLM_PREFILL_SHARD" not in ab.KNOWN:
        return MODE
    return parse_mode(ab.env("GLM_PREFILL_SHARD", "0"))


def call_pad() -> bool:
    ab = _ab()
    if ab is None or "GLM_PREFILL_SHARD_PAD" not in ab.KNOWN:
        return PAD
    return ab.truthy(ab.env("GLM_PREFILL_SHARD_PAD", "0"))
_LOG_FIRST = 8

# Engine sources of ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6... (vLLM 0.1.dev20051+
# g487ecf187 + tonyd2wild's DFlash2 patches), sha256[:16] of glm_prefill_hooks.func_source().
# [image] = hashed from files extracted from the pinned image on 2026-09-18
# (diagnostics/glm53-boot-sharded-20260918/evidence); [git] = from vLLM 487ecf187, identical to the
# image wherever both were compared; must be confirmed on a node (fleet step 0 prints them).
EXPECTED = {
    # [image] vllm/models/glm5next/nvidia/model.py
    "vllm.models.glm5next.nvidia.model:Glm5NextModel.forward": "6bf4d80050edcb64",
    "vllm.models.glm5next.nvidia.model:Glm5NextDecoderLayer.forward": "39c77ae8d0ed88be",
    "vllm.models.glm5next.nvidia.model:Glm5NextDecoderLayer.hc_pre": "998d1eb30c425120",
    "vllm.models.glm5next.nvidia.model:Glm5NextDecoderLayer.hc_post": "28d25abf781d29fb",
    "vllm.models.glm5next.nvidia.model:Glm5NextDecoderLayer.hc_fused_post_pre": "b4e1296cd1b43989",
    "vllm.models.glm5next.nvidia.model:Glm5NextMoE.forward": "39a23efafbcd293e",
    "vllm.models.glm5next.nvidia.model:Glm5NextMLP.forward": "42d88f10393524ea",
    # [image] attention output projections and the MoE reduction points
    "vllm.models.glm5next.nvidia.kda:Glm5NextLinearAttention.forward": "30746ac56387a7ef",
    "vllm.models.glm5next.nvidia.attention:Glm5NextMLAAttention.forward": "ed65780308660e28",
    "vllm.model_executor.layers.linear:RowParallelLinear.forward": "6f282baaa162ded9",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner.forward": "e16f2c9d63014e68",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner._maybe_reduce_final_output": "6a699287ced46d4a",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner._maybe_reduce_shared_expert_output": "f5b2e4ade8a3d36a",
    # [git] MLA wrapper, mHC ops and kernel dispatch
    "vllm.model_executor.layers.mla:MultiHeadLatentAttentionWrapper.forward": "4a3b9e6f74c94a46",
    "vllm.model_executor.layers.mhc:hc_expand": "ffc2debff734ea2f",
    "vllm.model_executor.layers.mhc:hc_contract": "9437034b4e7185d4",
    "vllm.model_executor.kernels.mhc.tilelang:_tilelang_hc_prenorm_gemm": "0d3f0297c112129e",
    "vllm.model_executor.kernels.mhc.tilelang:mhc_pre_tilelang": "810cd9ae96448146",
    "vllm.model_executor.kernels.mhc.tilelang:mhc_fused_post_pre_tilelang": "10af4f003b243e07",
    "vllm.model_executor.kernels.mhc.tilelang:mhc_post_tilelang": "10ddd1549050c62a",
    "vllm.model_executor.kernels.mhc.tilelang_kernels:compute_num_split": "5c35a41b87481cc0",
}
# Regenerate with: python3 tests/test_glm_prefill_hooks.py --print-hashes <extracted image root>


# ------------------------------------------------------------------------------------------
# logical row count (kernel choices keyed on the full chunk, not the owned quarter)
# ------------------------------------------------------------------------------------------
_state = threading.local()


def logical_rows() -> int | None:
    """Full chunk row count while a sharded mHC op runs, else None."""
    return getattr(_state, "rows", None)


@contextmanager
def _logical(rows: int | None):
    prev = getattr(_state, "rows", None)
    _state.rows = rows
    try:
        yield
    finally:
        _state.rows = prev


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def make_prenorm_dispatch(orig, kernels_module):
    """Wrap tilelang._tilelang_hc_prenorm_gemm so its row-count gates read logical_rows().

    Stock dispatch (vLLM 487ecf187): n_splits == 1 and default config and rows >= 1024 ->
    block_m kernel; rows < 128 and K % 1024 == 0 -> 1024-thread kernel; else the generic one.
    Every variant computes each row independently, so the same variant gives the same bits on
    a quarter of the rows as on all of them.
    """

    def dispatch(x, fn, out, sqrsum, hidden_size, hc_mult, tile_n=12, n_thr=512, n_splits=1):
        rows = logical_rows()
        if rows is None or rows == x.shape[0]:
            return orig(x, fn, out, sqrsum, hidden_size, hc_mult, tile_n, n_thr, n_splits)
        assert out.shape[0] == n_splits and sqrsum.shape[0] == n_splits
        assert x.shape[1] == hc_mult * hidden_size
        default = tile_n == 12 and n_thr == 512
        k = kernels_module
        if n_splits == 1 and default and rows >= 1024:
            return k.hc_prenorm_gemm_block_m_tilelang(
                x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, 2)
        if n_splits == 1 and default and rows < 128 and x.shape[1] % 1024 == 0:
            return k.hc_prenorm_gemm_tilelang(
                x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], 1024, 4, n_splits)
        return k.hc_prenorm_gemm_tilelang(
            x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, n_splits)

    dispatch.__glm_prefill_shard__ = True
    return dispatch


def make_num_split(orig):
    """Wrap tilelang_kernels.compute_num_split (split-K of the DeepGEMM tf32 pre-norm GEMM).

    mhc_pre_tilelang / mhc_fused_post_pre_tilelang call it with grid = cdiv(rows, 64); under a
    sharded op the grid is recomputed from the logical rows, so the split (and the fp32
    summation of the split partials) is the one the full chunk would use.
    """

    def compute_num_split(block_k, k, grid_size):
        rows = logical_rows()
        if rows is not None:
            grid_size = _cdiv(rows, 64)
        return orig(block_k, k, grid_size)

    compute_num_split.__glm_prefill_shard__ = True
    return compute_num_split


def install_kernel_keying(tl_module, tk_module) -> None:
    """Idempotent. tl = vllm.model_executor.kernels.mhc.tilelang, tk = ...tilelang_kernels."""
    if not getattr(tl_module._tilelang_hc_prenorm_gemm, "__glm_prefill_shard__", False):
        tl_module._tilelang_hc_prenorm_gemm = make_prenorm_dispatch(
            tl_module._tilelang_hc_prenorm_gemm, tk_module)
    if not getattr(tk_module.compute_num_split, "__glm_prefill_shard__", False):
        tk_module.compute_num_split = make_num_split(tk_module.compute_num_split)


# ------------------------------------------------------------------------------------------
# collectives (the TP GroupCoordinator; tests pass a thread group with the same three methods)
# ------------------------------------------------------------------------------------------
class Coll:
    def __init__(self, group):
        self.g = group
        self.rank = group.rank_in_group
        self.world = group.world_size

    def all_reduce(self, x):
        return self.g.all_reduce(x)

    def all_gather(self, x):
        return self.g.all_gather(x, 0)

    def reduce_scatter(self, x):
        return self.g.reduce_scatter(x, 0)

    def agree(self, ok: bool, like) -> bool:
        """True on every rank iff ok on every rank (sum of ones over TP)."""
        import torch
        t = torch.full((1,), 1.0 if ok else 0.0, dtype=torch.float32, device=like.device)
        return int(self.all_reduce(t).item()) == self.world


class Ownership:
    """Row ownership for one sharded forward. mode: 'shard' or 'comm'."""

    def __init__(self, coll: Coll, rows: int, mode: str, exact: bool, pad: bool = False):
        self.c = coll
        self.rows = rows
        self.mode = mode
        self.exact = exact and mode == "shard"
        w = coll.world
        if rows % w and not pad:
            raise RuntimeError(f"glm-prefill-shard: {rows} rows not divisible by {w} and padding is off")
        self.q = -(-rows // w)                       # owned rows per rank (ceil)
        self.padded = self.q * w
        self.pad = self.padded - rows                # zero rows appended at the end (0 unless pad)
        self.lo = coll.rank * self.q
        self.real = max(0, min(self.q, rows - self.lo))   # real rows among this rank's owned rows
        self.counts = {"rs": 0, "ag": 0, "ar": 0, "aux": 0, "padcopy": 0}

    def _pad_to(self, t):
        """[rows, ...] -> [padded, ...] with zero rows at the end (a copy; only when pad > 0)."""
        if not self.pad:
            return t
        out = t.new_empty((self.padded, *t.shape[1:]))
        out[: self.rows].copy_(t)
        out[self.rows:].zero_()
        self.counts["padcopy"] += 1
        return out

    def local(self, t):
        """Full-row tensor -> this rank's rows (a contiguous view; the short last owner gets zero rows)."""
        if self.mode == "comm":
            return t
        if t.shape[0] != self.rows:
            raise RuntimeError(f"glm-prefill-shard: row mismatch {t.shape[0]} != {self.rows}")
        if self.real == self.q:
            return t.narrow(0, self.lo, self.q)
        import torch
        part = t.narrow(0, min(self.lo, self.rows), self.real)
        return torch.cat([part, t.new_zeros((self.q - self.real, *t.shape[1:]))], 0)

    def _full(self, t):
        """Padded full rows -> the real rows (a contiguous prefix view)."""
        return t if not self.pad else t.narrow(0, 0, self.rows)

    def gather(self, t, aux: bool = False):
        """Owned rows -> full rows (attention / MLP input, aux capture, exit)."""
        if self.mode == "comm":
            return t
        if t.shape[0] != self.q:
            raise RuntimeError(f"glm-prefill-shard: gather of {t.shape[0]} rows, owner has {self.q}")
        self.counts["ag"] += 1
        if aux:
            self.counts["aux"] += 1
        return self._full(self.c.all_gather(t.contiguous()))

    def reduce(self, partial):
        """Full-row TP partial -> reduced rows (owned rows, or full rows in comm mode)."""
        if partial.shape[0] != self.rows:
            raise RuntimeError(f"glm-prefill-shard: partial has {partial.shape[0]} rows, chunk {self.rows}")
        partial = partial.contiguous()
        if self.mode == "comm":
            self.counts["rs"] += 1
            self.counts["ag"] += 1
            return self._full(self.c.all_gather(self.c.reduce_scatter(self._pad_to(partial))))
        if self.exact:
            self.counts["ar"] += 1
            return self.local(self.c.all_reduce(partial))
        self.counts["rs"] += 1
        return self.c.reduce_scatter(self._pad_to(partial))

    def rows_ctx(self):
        """Kernel choices keyed on the full chunk (the real row count) while an owned-row mHC op runs."""
        return _logical(self.rows if self.mode == "shard" else None)


# ------------------------------------------------------------------------------------------
# reduction deferral (attributes the engine reads at call time; restored in finally)
# ------------------------------------------------------------------------------------------
@contextmanager
def _defer_linear(linear):
    prev = linear.reduce_results
    linear.reduce_results = False
    try:
        yield
    finally:
        linear.reduce_results = prev


@contextmanager
def _defer_moe(runner):
    cfg = runner.moe_config
    prev = cfg.skip_final_all_reduce
    cfg.skip_final_all_reduce = True
    try:
        yield
    finally:
        cfg.skip_final_all_reduce = prev


# ------------------------------------------------------------------------------------------
# the sharded layer and model loop (copies of the image's mHC branch / Glm5NextModel.forward,
# guarded by EXPECTED; differences are marked "shard:")
# ------------------------------------------------------------------------------------------
def layer_forward(layer, positions, x, residual, post, comb, own, ops, checker=None):
    first = post is None
    if first:
        if layer.layer_idx == 0:
            x = ops.hc_expand(x, layer.n)
        residual = x
        post, comb, x = layer.hc_pre(
            x,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            norm_weight=layer.input_layernorm.weight.data,
            norm_eps=layer.input_layernorm.variance_epsilon,
        )
        # shard: the first pre ran on all rows (identical to stock); keep the owned state and
        # hand attention the full x without a gather.
        residual, post, comb = own.local(residual), own.local(post), own.local(comb)
    else:
        args = (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        kw = dict(norm_weight=layer.input_layernorm.weight.data,
                  norm_eps=layer.input_layernorm.variance_epsilon)
        if checker is not None:
            residual, post, comb, x = checker(layer, x, residual, post, comb, args, kw)
        else:
            with own.rows_ctx():
                residual, post, comb, x = layer.hc_fused_post_pre(x, residual, post, comb, *args, **kw)
        x = own.gather(x)

    with _defer_linear(layer.self_attn.o_proj):
        x = layer.self_attn(hidden_states=x, positions=positions)
    x = own.reduce(x)

    with own.rows_ctx():
        residual, post, comb, x = layer.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            norm_weight=layer.post_attention_layernorm.weight.data,
            norm_eps=layer.post_attention_layernorm.variance_epsilon,
        )

    x = own.gather(x)
    if layer._mlp_is_moe:
        with _defer_moe(layer.mlp.experts):
            x = layer.mlp(x, already_sequence_parallel=False)
    else:
        with _defer_linear(layer.mlp.down_proj):
            x = layer.mlp(x)
    x = own.reduce(x)

    if layer.layer_idx == layer.num_hidden_layers - 1:
        with own.rows_ctx():
            x = layer.hc_post(x, residual, post, comb)
        x = ops.hc_contract(x, layer.n)
        return x, None, None, None
    return x, residual, post, comb


def model_forward(model, own, input_ids, positions, inputs_embeds, ops, checker=None):
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = model.embed_input_ids(input_ids)
    residual = post = comb = None
    aux_hidden_states = []
    for idx, layer in enumerate(model._active_layers, start=model.start_layer):
        hidden_states, residual, post, comb = layer_forward(
            layer, positions, hidden_states, residual, post, comb, own, ops,
            checker if idx == model.start_layer + 1 else None)
        if idx + 1 in model.aux_hidden_state_layers:
            if post is not None:
                with own.rows_ctx():
                    aux_recon = layer.hc_post(hidden_states, residual, post, comb)
                aux_hidden_state = ops.hc_contract(aux_recon, layer.n)
            else:
                aux_hidden_state = hidden_states
            # shard: aux states are consumed at full-sequence granularity.
            aux_hidden_states.append(own.gather(aux_hidden_state, aux=True))
    hidden_states = own.gather(hidden_states)
    hidden_states = model.norm(hidden_states)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# ------------------------------------------------------------------------------------------
# eligibility, validation, install
# ------------------------------------------------------------------------------------------
class _Runtime:
    def __init__(self):
        self.validated = None       # None = not yet, True/False after the first candidate chunk
        self.error = None
        self.bad_rows: set[int] = set()
        self.checked_rows: set[int] = set()
        self.reports = 0
        self.kda_name = ""


def validate_model(model) -> str:
    """Raise if the loaded model is not the qualified TP4 layout; return a KDA layer name."""
    kda_name = None
    if model.start_layer != 0 or getattr(model, "is_sequence_parallel", False):
        raise RuntimeError("needs PP1 and no sequence-parallel MoE")
    for layer in model._active_layers:
        if not layer.mhc or layer.is_mtp_layer:
            raise RuntimeError(f"layer {layer.layer_idx} is not an mHC base layer")
        o_proj = layer.self_attn.o_proj
        if not o_proj.reduce_results or o_proj.tp_size != TP or getattr(o_proj, "bias", None) is not None:
            raise RuntimeError(f"layer {layer.layer_idx}: unqualified o_proj")
        mla = getattr(layer.self_attn, "mla_attn", None)
        if mla is not None and getattr(mla, "o_proj", None) is not o_proj:
            raise RuntimeError(f"layer {layer.layer_idx}: MLA wrapper does not use the layer's o_proj")
        if layer.layer_kind == "kda" and kda_name is None:
            kda_name = layer.self_attn.prefix
        if layer._mlp_is_moe:
            r = layer.mlp.experts
            cfg = r.moe_config
            # hidden_dim == hidden_size: no kernel padding, so the deferred partial has the
            # stock all-reduce's exact shape (NCCL chunking, hence summation order, is by size)
            if (cfg.skip_final_all_reduce or cfg.is_sequence_parallel or cfg.tp_size != TP
                    or cfg.hidden_dim != layer.hidden_size
                    or cfg.moe_parallel_config.use_all2all_kernels or getattr(cfg, "dp_size", 1) != 1
                    or r._fused_output_is_reduced or r.routed_output_transform is not None
                    or getattr(layer.mlp, "is_sequence_parallel", False)):
                raise RuntimeError(f"layer {layer.layer_idx}: MoE output is not one deferred TP partial")
        else:
            d = layer.mlp.down_proj
            if not d.reduce_results or d.tp_size != TP or getattr(d, "bias", None) is not None:
                raise RuntimeError(f"layer {layer.layer_idx}: unqualified dense down_proj")
    if kda_name is None:
        raise RuntimeError("no KDA layer to read prefill metadata from")
    return kda_name


def pure_or_mixed_prefill(metadata, kda_name: str, rows: int) -> bool:
    if not isinstance(metadata, dict):
        return False
    md = metadata.get(kda_name)
    if md is None:
        return False
    try:
        n_pf, n_pf_tok = int(md.num_prefills), int(md.num_prefill_tokens)
        n_dec, n_spec = int(md.num_decodes), int(md.num_spec_decodes)
    except (AttributeError, TypeError, ValueError):
        return False
    if n_pf <= 0 or n_pf_tok <= 0 or n_pf_tok > rows:
        return False
    if MIXED:
        return True
    return n_dec == 0 and n_spec == 0


def candidate_context(positions, intermediate_tensors, rt: _Runtime):
    """Rank-invariant gates (shape, mode, graph state). Returns the forward context or None."""
    import torch
    rows = positions.shape[0]
    if MODE == "off" or call_mode() == "off" or rt.validated is False or rows in rt.bad_rows:
        return None
    if intermediate_tensors is not None or rows < MIN_ROWS or (rows % TP and not call_pad()):
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    if ctx.cudagraph_runtime_mode.name != "NONE" or getattr(ctx, "ubatch_slices", None) is not None:
        return None
    return ctx


def first_candidate(model, rt: _Runtime, coll: Coll, like) -> None:
    """Validate once, at the first candidate forward, which every rank reaches together (the
    gates above are rank-invariant); agree over TP so a local failure can never leave the ranks
    on different reduction paths."""
    try:
        rt.kda_name = validate_model(model)
    except (AttributeError, RuntimeError) as exc:
        rt.kda_name, rt.error = "", str(exc)
    rt.validated = coll.agree(bool(rt.kda_name), like)
    if not rt.validated:
        logger.error("glm-prefill-shard: disabled, model not qualified: %s",
                     rt.error or "another TP rank failed validation")


def _check_fused(own: Ownership, rt: _Runtime):
    """GLM_PREFILL_SHARD_CHECK: first sharded fused hc op of each row count, stock vs sharded."""
    import torch

    def checker(layer, x, residual, post, comb, args, kw):
        with own.rows_ctx():
            out_s = layer.hc_fused_post_pre(x, residual, post, comb, *args, **kw)
        full = [own.c.all_gather(t.contiguous()) for t in (x, residual, post, comb)]
        out_f = layer.hc_fused_post_pre(*full, *args, **kw)
        mine = [own.local(t) for t in out_f]
        same = all(torch.equal(a, b) for a, b in zip(out_s, mine))
        agreed = own.c.agree(same, x)
        diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(out_s, mine))
        logger.warning("GLM_PREFILL_SHARD_CHECK rows=%d rank=%d local_equal=%s agreed=%s max_abs=%g",
                       own.rows, own.c.rank, same, agreed, diff)
        if not agreed:
            rt.bad_rows.add(own.rows)
            return tuple(t.contiguous() for t in mine)   # keep the stock result for this chunk
        return out_s

    return checker


def install(model_module, coll_factory=None) -> None:
    """Patch Glm5NextModel (forward + an __init__ vote). Called on import of the model module."""
    if MODE == "off":
        return
    if MODE == "comm" and EXACT:
        raise RuntimeError("GLM_PREFILL_SHARD_EXACT applies to GLM_PREFILL_SHARD=1 only")
    _hooks.check_sources(EXPECTED, "glm-prefill-shard")
    import vllm.model_executor.layers.mhc as mhc_layer

    cls = model_module.Glm5NextModel
    if getattr(cls, "__glm_prefill_shard__", False):
        return
    orig_forward, orig_init = cls.forward, cls.__init__
    ops = type("Ops", (), {"hc_expand": staticmethod(mhc_layer.hc_expand),
                           "hc_contract": staticmethod(mhc_layer.hc_contract)})

    def _coll():
        if coll_factory is not None:
            return coll_factory()
        from vllm.distributed import get_tp_group
        return Coll(get_tp_group())

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        import torch
        from vllm.distributed import get_tp_group
        g = get_tp_group()
        mine = (MODE, EXACT, MIN_ROWS, MIXED, CHECK, PAD)
        votes = [None] * g.world_size
        torch.distributed.all_gather_object(votes, mine, group=g.cpu_group)
        if any(v != mine for v in votes):
            raise RuntimeError(f"glm-prefill-shard: TP ranks disagree on GLM_PREFILL_SHARD*: {votes}")
        if g.world_size != TP:
            raise RuntimeError(f"glm-prefill-shard: needs TP{TP}, got {g.world_size}")
        self._glm_ps_rt = _Runtime()
        logger.warning("GLM_PREFILL_SHARD_READY mode=%s exact=%d min_rows=%d mixed=%d check=%d pad=%d",
                       MODE, int(EXACT), MIN_ROWS, int(MIXED), int(CHECK), int(PAD))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        rt = getattr(self, "_glm_ps_rt", None)
        ctx = None if rt is None else candidate_context(positions, intermediate_tensors, rt)
        if ctx is not None and rt.validated is None:
            first_candidate(self, rt, _coll(), positions)
            if rt.validated:
                import vllm.model_executor.kernels.mhc.tilelang as tl
                import vllm.model_executor.kernels.mhc.tilelang_kernels as tk
                install_kernel_keying(tl, tk)
            else:
                ctx = None
        rows = positions.shape[0]
        if ctx is None or not pure_or_mixed_prefill(ctx.attn_metadata, rt.kda_name, rows):
            return orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        mode = call_mode()
        own = Ownership(_coll(), rows, mode, EXACT, pad=call_pad())
        checker = None
        if CHECK and mode == "shard" and rows not in rt.checked_rows:
            rt.checked_rows.add(rows)
            checker = _check_fused(own, rt)
        out = model_forward(self, own, input_ids, positions, inputs_embeds, ops, checker)
        if rt.reports < _LOG_FIRST:
            rt.reports += 1
            logger.warning("GLM_PREFILL_SHARD rank=%d rows=%d owner_rows=%d pad=%d mode=%s exact=%d %s",
                           own.c.rank, rows, own.q, own.pad, mode, int(own.exact), own.counts)
        return out

    cls.__init__ = __init__
    cls.forward = forward
    cls.__glm_prefill_shard__ = True
    logger.info("glm-prefill-shard: installed (mode=%s exact=%d)", MODE, int(EXACT))

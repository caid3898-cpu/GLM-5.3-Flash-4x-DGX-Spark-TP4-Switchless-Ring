"""Adaptive verify cut (confidence-based dead-row remap) for GLM-5.3-Flash on vLLM 487ecf187 (V2 runner).

Origin: a port of our DeepSeek-V4.1 stack's ``DSV41_VERIFY_CAP=conf:0.1`` (ds41 adapter ``verify_cap.py``,
2026-09-23: prose +5..8 % at c1, code flat; fixed caps lost there) together with its router remap. On DS the
confidence came from DSpark's trained confidence head; this file supports that drafter (``method="dspark"``)
and the incoai DFlash2 drafter that production runs (``method="dflash"`` with a candidate selector), which has
no confidence head, so the confidence is taken from the drafter's own selector distribution (below).

Enable with ``GLM_VERIFY_CUT=conf:<T>``. Unset = the module does nothing. ``conf:0`` arms every hook and cuts
nothing. ``GLM_VERIFY_CUT_FILE=/cache/<name>.json`` (opt-in, head only) makes TP rank 0 re-read
``{"threshold": T}`` every 64 draft steps, so thresholds can be A/B'd inside one boot.

Why not upstream's ``enable_adaptive_verification``: it trims verify rows on device, and SSM backends (the GLM
KDA layers) return False from ``supports_device_cpu_query_lens_mismatch``. This keeps the verify rectangle the
scheduler planned (KDA, indexer, MLA and graph shapes unchanged) and makes the dead rows free in the MoE:

  step N-1, after drafting: live length L[slot] = number of leading drafts whose running product of per-position
      confidences stays >= T. Written per persistent request slot, then broadcast from TP rank 0 (pynccl) so every
      rank cuts identically (rank-invariance rule).
  step N, in model_state.prepare_inputs: one Triton kernel writes ROW_SRC (static buffer, graph safe): rows after
      anchor+L of a request -> the anchor row; graph padding rows -> row 0; everything else identity. It also
      writes DEAD (the rows whose draft token must be rejected).
  inside every Glm5NextMoE.forward: moe_input = moe_input[ROW_SRC], so a dead row routes to exactly the anchor's
      experts and adds no expert to the per-layer union (no extra expert bytes). The gather is on the MoE input,
      not on router_logits, because the image's MoE runner recomputes the logits from its own gate (a logit gather
      was a silent no-op; measured 2026-09-26). Captured in the FULL decode graphs; in eager steps it runs only
      when the batch has drafts or padding.
  step N, rejection: dead drafts are replaced by -1, which the rejection kernel treats as "placeholder, stop
      verifying" (greedy: reject and emit the target argmax; sampled: resample from the target directly).
      glm_target_argmax (GLM_TARGET_VOCAB_ARGMAX) reads S.dead and does the same on its greedy fast path.

DFlash2 confidence: the drafter's selector walk stores, per draft position s, the scores of its top-k candidates
conditioned on the candidate chosen at s-1 (``DFlash2Speculator._selector_scores``). conf[s] = max softmax of
those scores. It depends on drafts < s only, never on the draft at s, so the cut is a stopping rule; production
drafts greedily (draft_sample_method "greedy"), so drafts and cuts are deterministic functions of the context
and the output distribution is the target's, exactly as with the adaptive draft-length scheduler.

Exactness: rows are causal in attention, KDA, the indexer and mHC, and independent in the MoE, so live rows
never read dead rows; dead rows' KV/state slots are overwritten like any rejected draft's.

Hooks (monkeypatches installed by overlay/sitecustomize.py when GLM_VERIFY_CUT is set):
  vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator.propose
  vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator._sample_sequential / .propose
  vllm.models.glm5next.nvidia.model.Glm5NextMoE.forward
  vllm.v1.worker.gpu.model_runner.GPUModelRunner.__init__ / .load_model (wraps model_state.prepare_inputs)
  vllm.v1.worker.gpu.spec_decode.rejection_sampler.RejectionSampler.__call__
Not for method="mtp" (its MoE sees drafter-shaped batches); sequence-parallel MoE raises.
"""
from __future__ import annotations

import os
import sys

_SPEC = os.environ.get("GLM_VERIFY_CUT", "").strip()
ENABLED = _SPEC.startswith("conf:")
THRESHOLD = float(_SPEC.split(":", 1)[1]) if ENABLED else 0.0
BROADCAST = os.environ.get("GLM_VERIFY_CUT_BCAST", "1") != "0"
LOG_EVERY = int(os.environ.get("GLM_VERIFY_CUT_LOG_EVERY", "2000"))
CONTROL = os.environ.get("GLM_VERIFY_CUT_FILE", "").strip()
AUDIT = os.environ.get("GLM_VERIFY_CUT_AUDIT", "1") != "0"   # count rank disagreements before the broadcast
ROWS_BLOCK = 256


class _State:
    row_src = None      # int64 [max_tokens]
    dead_u8 = None      # uint8 [max_tokens]
    dead = None         # bool view of dead_u8 (read by glm_target_argmax)
    live = None         # int32 [max_reqs]: live draft count per persistent request slot
    hist = None         # int64 [64]: histogram of live lengths (after the cut, before min with scheduled k)
    dead_count = None   # int64 [1]: dead rows so far
    max_tokens = 0
    max_reqs = 0
    gather = False      # eager steps: run the router gather only when this batch has drafts or padding
    threshold = THRESHOLD
    ctl_mtime = None
    propose_calls = 0
    steps = 0
    draft_rows = 0
    rank0 = True
    local = None        # int32 [max_reqs]: this rank's own live lengths (audit)
    audit = None        # int64 [1]: disagreeing slots seen so far on this rank (always 0 on rank 0)


S = _State()
_K: dict = {}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-verify-cut: {msg}\n")


def _alloc(max_tokens: int, max_reqs: int, device) -> None:
    import torch

    if S.row_src is not None:
        return
    S.max_tokens, S.max_reqs = max_tokens, max_reqs
    S.row_src = torch.arange(max_tokens, dtype=torch.int64, device=device)
    S.dead_u8 = torch.zeros(max_tokens, dtype=torch.uint8, device=device)
    S.dead = S.dead_u8.view(torch.bool)
    S.live = torch.full((max_reqs,), 1 << 20, dtype=torch.int32, device=device)
    S.hist = torch.zeros(64, dtype=torch.int64, device=device)
    S.dead_count = torch.zeros(1, dtype=torch.int64, device=device)
    S.audit = torch.zeros(1, dtype=torch.int64, device=device)
    _log(f"armed: threshold={THRESHOLD} max_tokens={max_tokens} max_reqs={max_reqs} bcast={BROADCAST} "
         f"control={CONTROL or '-'}")


# --------------------------------------------------------------------------- Triton kernels (built lazily)
def kernels() -> dict:
    if _K:
        return _K
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _vc_live_kernel(scores_ptr, stride_r, stride_s, idx_ptr, live_ptr, hist_ptr, thr,
                        K: tl.constexpr, TOPK: tl.constexpr, BLOCK_K: tl.constexpr):
        # one program per request row of the draft batch: conf[s] = max softmax over the selector's realized
        # candidate scores at position s; L = #leading positions whose running product stays >= thr.
        r = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        m = offs < TOPK
        cum = tl.full((), 1.0, tl.float32)
        alive = tl.full((), 1, tl.int32)
        length = tl.full((), 0, tl.int32)
        for s in tl.static_range(K):
            x = tl.load(scores_ptr + r * stride_r + s * stride_s + offs, mask=m, other=float("-inf"))
            x = x.to(tl.float32)
            mx = tl.max(x, axis=0)
            den = tl.sum(tl.where(m, tl.exp(x - mx), 0.0), axis=0)
            p = tl.where(mx > float("-inf"), 1.0 / den, 0.0)
            cum = cum * p
            alive = alive * (cum >= thr).to(tl.int32)
            length += alive
        slot = tl.load(idx_ptr + r)
        tl.store(live_ptr + slot, length)
        tl.atomic_add(hist_ptr + length, 1)

    @triton.jit
    def _vc_rows_kernel(row_src_ptr, dead_ptr, qsl_ptr, cu_ptr, idx_ptr, live_ptr, count_ptr,
                        n_req, n_tok, n_pad, BLOCK: tl.constexpr):
        # rows qs+1+L .. qs+nd of each request are dead (router -> row qs, draft rejected);
        # rows n_tok .. n_pad-1 are graph padding (router -> row 0); the rest is identity.
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        valid = rows < n_pad
        src = tl.where(rows < n_tok, rows, 0)
        dead = rows < 0
        for r in range(n_req):
            qs = tl.load(qsl_ptr + r)
            qe = tl.load(qsl_ptr + r + 1)
            nd = tl.load(cu_ptr + r + 1) - tl.load(cu_ptr + r) - 1
            lv = tl.minimum(tl.load(live_ptr + tl.load(idx_ptr + r)), nd)
            j = rows - qs
            d = (rows < qe) & (j >= 1) & (j <= nd) & (j > lv)
            src = tl.where(d, qs, src)
            dead = dead | d
        tl.store(row_src_ptr + rows, src.to(tl.int64), mask=valid)
        tl.store(dead_ptr + rows, dead.to(tl.uint8), mask=valid)
        tl.atomic_add(count_ptr, tl.sum((dead & valid).to(tl.int64), axis=0))

    _K["live"] = _vc_live_kernel
    _K["rows"] = _vc_rows_kernel
    _K["next_pow2"] = triton.next_power_of_2
    _K["cdiv"] = triton.cdiv
    return _K


def launch_live(scores, idx_mapping, live, hist, n: int, threshold: float) -> None:
    """scores [>=n, K, TOPK] fp32 (row = draft-batch request order) -> live[idx_mapping[r]] = L_r."""
    k = kernels()
    topk = int(scores.shape[2])
    k["live"][(n,)](scores, scores.stride(0), scores.stride(1), idx_mapping, live, hist, float(threshold),
                    K=int(scores.shape[1]), TOPK=topk, BLOCK_K=k["next_pow2"](topk), num_warps=1)


def launch_rows(row_src, dead_u8, qsl, cu, idx_mapping, live, count, n_req: int, n_tok: int, n_pad: int) -> None:
    k = kernels()
    k["rows"][(k["cdiv"](n_pad, ROWS_BLOCK),)](row_src, dead_u8, qsl, cu, idx_mapping, live, count,
                                               n_req, n_tok, n_pad, BLOCK=ROWS_BLOCK, num_warps=4)


# --------------------------------------------------------------------------- pure references (tests)
def live_lengths(conf, threshold: float):
    """[R, K] per-position confidences -> [R] number of leading drafts with cumprod >= threshold."""
    import torch

    return (conf.float().cumprod(dim=1) >= threshold).sum(dim=1).to(torch.int32)


def selector_conf(scores):
    """[R, K, TOPK] realized selector scores -> [R, K] max softmax probability."""
    return scores.float().softmax(dim=-1).amax(dim=-1)


def rows_reference(qsl, cu, idx_mapping, live, n_req: int, n_tok: int, n_pad: int):
    """Python reference of _vc_rows_kernel -> (row_src list, dead list)."""
    src = [r if r < n_tok else 0 for r in range(n_pad)]
    dead = [False] * n_pad
    for r in range(n_req):
        qs, qe = int(qsl[r]), int(qsl[r + 1])
        nd = int(cu[r + 1]) - int(cu[r]) - 1
        lv = min(int(live[int(idx_mapping[r])]), nd)
        for j in range(lv + 1, nd + 1):
            if qs + j < qe:
                src[qs + j] = qs
                dead[qs + j] = True
    return src, dead


# --------------------------------------------------------------------------- helpers
def _reload_threshold() -> None:
    import json

    try:
        st = os.stat(CONTROL)
    except OSError:
        return
    if st.st_mtime == S.ctl_mtime:
        return
    S.ctl_mtime = st.st_mtime
    try:
        with open(CONTROL) as f:
            t = float(json.load(f)["threshold"])
        if 0.0 <= t <= 1.0 and t != S.threshold:
            _log(f"threshold {S.threshold} -> {t} (from {CONTROL}) at propose call {S.propose_calls}")
            S.threshold = t
    except Exception as exc:  # noqa: BLE001
        _log(f"bad control file {CONTROL}: {exc!r}")


def _sync_live() -> None:
    """Rank 0's live lengths win (pynccl broadcast on the already-initialized TP communicator; the torch
    device-group broadcast lazily creates a new NCCL communicator and failed at boot on 2026-09-25)."""
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if not BROADCAST or tp.world_size == 1:
        return
    if AUDIT:
        S.local = S.live.clone() if S.local is None else S.local.copy_(S.live)
    tp.device_communicator.broadcast(S.live, 0)  # raises if there is no pynccl communicator
    if AUDIT:
        # audit soak for dropping the broadcast later: slots where this rank's own cut differed from rank 0's
        S.audit.add_((S.local != S.live).sum())


def _after_propose() -> None:
    S.propose_calls += 1
    if AUDIT and BROADCAST and not S.rank0 and LOG_EVERY and S.propose_calls % LOG_EVERY == 0:
        _log(f"audit at propose {S.propose_calls}: slots where this rank's cut differed from rank 0: "
             f"{int(S.audit.item())}")
    if S.rank0 and LOG_EVERY and S.propose_calls % LOG_EVERY == 0:
        h = S.hist.tolist()
        last = max((i for i, v in enumerate(h) if v), default=0)
        _log(f"propose {S.propose_calls}: T={S.threshold} dead rows {int(S.dead_count.item())} of "
             f"{S.draft_rows} scheduled drafts; live-length hist {h[:last + 1]}")


def _before_propose() -> None:
    if S.rank0 and CONTROL and BROADCAST and S.propose_calls % 64 == 0:
        _reload_threshold()


# --------------------------------------------------------------------------- hooks
def _patch_dflash2(mod) -> None:
    import torch

    cls = mod.DFlash2Speculator
    orig_propose = cls.propose

    def propose(self, input_batch, *args, **kwargs):
        out = orig_propose(self, input_batch, *args, **kwargs)
        if S.live is None or kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing():
            return out
        n = int(input_batch.num_reqs)
        if n == 0:
            return out
        _before_propose()
        launch_live(self._selector_scores, input_batch.idx_mapping, S.live, S.hist, n, S.threshold)
        _sync_live()
        _after_propose()
        return out

    cls.propose = propose
    _log("DFlash2 speculator hooked (selector max-softmax confidence)")


def _patch_dspark(mod) -> None:
    import torch

    cls = mod.DSparkSpeculator
    orig_seq = cls._sample_sequential
    orig_propose = cls.propose

    def _sample_sequential(self, num_reqs, head_hidden):
        # Compute the confidence head inside the (captured) draft step without telling the model runner,
        # which would otherwise create upstream's AdaptiveVerificationManager (SSM-incompatible).
        prev = self.enable_adaptive_verification
        self.enable_adaptive_verification = True
        try:
            return orig_seq(self, num_reqs, head_hidden)
        finally:
            self.enable_adaptive_verification = prev

    def propose(self, input_batch, *args, **kwargs):
        out = orig_propose(self, input_batch, *args, **kwargs)
        if S.live is None or kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing():
            return out
        n = int(input_batch.num_reqs)
        if n == 0:
            return out
        _before_propose()
        lens = live_lengths(self.draft_token_confidence_probs[:n], S.threshold)
        S.live[input_batch.idx_mapping] = lens
        _sync_live()
        _after_propose()
        return out

    cls._sample_sequential = _sample_sequential
    cls.propose = propose
    _log("DSpark speculator hooked (confidence head)")


def _patch_moe(mod) -> None:
    import torch

    cls = mod.Glm5NextMoE

    def forward(self, hidden_states, already_sequence_parallel: bool = False):
        num_tokens, hidden_dim = hidden_states.shape
        if self.is_sequence_parallel and not already_sequence_parallel:
            raise RuntimeError("glm-verify-cut: sequence-parallel MoE is not supported; unset GLM_VERIFY_CUT")
        # Gather the MoE INPUT rows, not the router logits: in this image the MoE runner owns the gate and
        # recomputes router_logits = gate(hidden_states) inside _forward_impl (fused_moe/runner/moe_runner.py
        # 858-863), so a gather on the logits passed in is discarded. With the input gathered, a dead row is
        # the anchor row for routing and experts alike; live rows are untouched (identity entries).
        if (S.row_src is not None and num_tokens <= S.max_tokens
                and (S.gather or torch.cuda.is_current_stream_capturing())):
            hidden_states = hidden_states.index_select(0, S.row_src[:num_tokens])
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return final_hidden_states.view(num_tokens, hidden_dim)

    cls.forward = forward
    _log("Glm5NextMoE.forward hooked (MoE-input row gather)")


def _patch_runner(mod) -> None:
    cls = mod.GPUModelRunner
    orig_init = cls.__init__
    orig_load = cls.load_model

    def __init__(self, vllm_config, device, *a, **kw):
        orig_init(self, vllm_config, device, *a, **kw)
        spec = vllm_config.speculative_config
        if spec is None or spec.method not in ("dspark", "dflash"):
            _log(f"speculative method is {getattr(spec, 'method', None)!r}: staying off")
            return
        try:
            from vllm.distributed.parallel_state import get_tp_group
            S.rank0 = get_tp_group().rank_in_group == 0
        except Exception:  # noqa: BLE001
            S.rank0 = True
        _alloc(int(self.max_num_tokens), int(self.max_num_reqs), device)

    def load_model(self, *args, **kwargs):
        out = orig_load(self, *args, **kwargs)
        if S.row_src is None:
            return out
        ms = self.model_state
        orig_prep = ms.prepare_inputs

        def prepare_inputs(input_batch, req_states):
            res = orig_prep(input_batch, req_states)
            n_tok = int(input_batch.num_tokens)
            n_pad = int(input_batch.num_tokens_after_padding)
            n_req = int(input_batch.num_reqs)
            if n_pad > S.max_tokens or n_req == 0:
                S.gather = False
                return res
            nd = int(input_batch.num_draft_tokens)
            # always rewrite ROW_SRC for this batch: a FULL graph replay gathers through it unconditionally
            launch_rows(S.row_src, S.dead_u8, input_batch.query_start_loc, input_batch.cu_num_logits,
                        input_batch.idx_mapping, S.live, S.dead_count,
                        n_req if nd > 0 else 0, n_tok, n_pad)
            S.gather = nd > 0 or n_pad > n_tok
            S.draft_rows += nd
            return res

        ms.prepare_inputs = prepare_inputs
        _log("model_state.prepare_inputs hooked (Triton row builder)")
        return out

    cls.__init__ = __init__
    cls.load_model = load_model


def _patch_rejection(mod) -> None:
    cls = mod.RejectionSampler
    orig_call = cls.__call__

    def __call__(self, logits, input_batch, draft_logits=None):
        n = int(input_batch.num_tokens)
        if S.dead is None or input_batch.num_draft_tokens == 0 or n > S.max_tokens:
            return orig_call(self, logits, input_batch, draft_logits)
        ids = input_batch.input_ids
        saved = ids[:n].clone()
        ids[:n].masked_fill_(S.dead[:n], -1)
        try:
            return orig_call(self, logits, input_batch, draft_logits)
        finally:
            ids[:n].copy_(saved)

    cls.__call__ = __call__
    _log("RejectionSampler.__call__ hooked (-1 for dead drafts)")


HOOKS = {
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": _patch_dflash2,
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator": _patch_dspark,
    "vllm.models.glm5next.nvidia.model": _patch_moe,
    "vllm.v1.worker.gpu.model_runner": _patch_runner,
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler": _patch_rejection,
}

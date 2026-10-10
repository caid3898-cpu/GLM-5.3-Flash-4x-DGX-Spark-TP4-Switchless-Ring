# SPDX-License-Identifier: Apache-2.0
"""Context-lookup hybrid drafter, worker side (diagnostics/glm-lookup-20260928). Default OFF.

Tony v11 image, vLLM 0.1.dev20051+g487ecf187, V2 model runner (vllm/v1/worker/gpu). Pairs with the scheduler
half, overlay/glm_lookup_draft.py (LookupScheduler), which puts a plan on scheduler_output.glm_lookup.

  GLM_LOOKUP_DRAFT=1          install (shadow / 0: nothing is patched on the workers)
  GLM_LOOKUP_CHECK=1          TEST ONLY: after every lookup kernel, recompute the rows on the host (torch
                              reference) and count mismatches; synchronises the stream
  GLM_LOOKUP_CORRUPT=1        TEST ONLY: lookup ids are XOR 1 (always rejected; forced-reject test)
  GLM_LOOKUP_FILL=1           TEST ONLY: a row whose check fails gets the last committed token in all k slots
                              (with GLM_LOOKUP_FORCE=alt-skip / alt-override: identical drafts in both arms)
  GLM_LOOKUP_TRACE=/path      TEST ONLY: TP rank 0 appends one JSON line per lookup step (plan seq, rows,
                              skip, draft ids); synchronises the stream
  GLM_LOOKUP_WORKER_LOG_EVERY (2000) per-rank counters; identical lines across ranks = the ranks agreed
  GLM_LOOKUP_ALLOW_DRIFT=1    patch even if the engine sources differ from the qualified image

Hooks
  GPUModelRunner.execute_model   remembers scheduler_output.glm_lookup for the sample_tokens call that follows
                                 (the V2 worker alternates execute_model / sample_tokens per step)
  GPUModelRunner.sample_tokens   exposes (plan, req_states) to the speculator for the propose() it runs
  DFlashSpeculator.propose       (DFlash2Speculator inherits it)
      * plan rows are matched to batch rows by req_id; no row -> stock propose, untouched;
      * skip step (plan "skip", every batch row a skip row, no prefill row): the stock propose runs with the
        draft forward suppressed: the target hidden states are still copied, prepare_dflash_inputs still
        runs and precompute_and_store_context_kv still writes the context K/V of every target token of
        this step, and the draft attention metadata is still rebuilt. Only _generate_draft / the FULL
        graph replay is skipped (query forward, candidate head, selector walk);
      * then one Triton kernel writes the lookup ids into speculator.draft_tokens for the valid rows.

Why the skip keeps DFlash2 consistent: the drafter's persistent state is its K/V cache. Context K/V is written
by precompute for every target position of each step (all verified positions, rejected ones included, which
the next step overwrites). The draft forward writes K/V only at its query positions last_valid+1 ..
last_valid+1+N, and the next step's precompute rewrites those positions before any query reads them as
context: the next verify batch starts at last_valid+1. With greedy drafts (draft_logits None) and no adaptive
verification there is no other state: _selector_scores / draft logits are read only in probabilistic mode.
The install refuses probabilistic drafts, adaptive verification and GLM_VERIFY_CUT.

Kernel (per batch row b with k > 0):
  slot = idx_mapping[b]; L = total_len[slot] (committed length, this step's tokens included);
  delta = L - lhost; ok = 0 <= delta <= max_delta and src < lhost <= L
         and tokens[lhost + i] == tokens[src + i] for i < delta       (the copy continued while in flight)
  draft[j] = tokens[src + delta + (j mod P)], P = lhost - src, j < k  (periodic; else j, while < L)
Rows that fail keep what is in draft_tokens: the DFlash2 ids on a normal step, the previous ids on a skip
step (then verified and rejected like any bad draft). Every input is identical on all TP ranks (plan from the
scheduler, committed tokens from the rank-identical sampler), so every rank writes the same ids.

Credit: the V2 runner's persistent all_token_ids / total_len buffers (vLLM contributors) make the device-side
lookup sync-free; the lookup itself follows vLLM's n-gram proposer and STRML's mlx-serve #523 (see the
scheduler module).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

_OFF = ("", "0", "off", "false", "no")
MODE_SKIP = 1   # = glm_lookup_draft.MODE_SKIP (kept local: this module must not need the scheduler module)

TARGET_RUNNER = "vllm.v1.worker.gpu.model_runner"
TARGET_SPEC = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"

# function text sha256[:16] in the qualified image (same scheme as glm_ds_hooks)
EXPECTED = {
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.propose": "432c21dae8db3abe",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator._generate_draft": "22817856ec5178eb",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._generate_draft": "2550bc0960238491",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample_tokens": "840ba66305b71ec7",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.execute_model": "707374f1a381cc3f",
    "vllm.v1.worker.gpu.input_batch:_post_update_kernel": "a69ddd095da1e265",
    "vllm.v1.worker.gpu.input_batch:_combine_sampled_and_draft_tokens_kernel": "04bacd2d01b317dc",
}


def _on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in _OFF


def enabled() -> bool:
    v = os.environ.get("GLM_LOOKUP_DRAFT", "0").strip().lower()
    return v not in _OFF and v != "shadow"


CHECK = _on("GLM_LOOKUP_CHECK")
CORRUPT = _on("GLM_LOOKUP_CORRUPT")
FILL = _on("GLM_LOOKUP_FILL")
TRACE = os.environ.get("GLM_LOOKUP_TRACE", "").strip()
LOG_EVERY = int(os.environ.get("GLM_LOOKUP_WORKER_LOG_EVERY", "2000") or 0)


class _S:
    rank = None
    disabled = None      # reason string once disabled
    c = {"proposes": 0, "lookup_steps": 0, "rows": 0, "skip_steps": 0, "check_rows": 0, "check_mismatch": 0,
         "valid_rows": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-lookup[{_rank()}]: {msg}\n")


def _rank() -> int:
    if _S.rank is None:
        try:
            import torch.distributed as dist
            _S.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:  # noqa: BLE001
            _S.rank = 0
    return _S.rank


# ------------------------------------------------------------------------------------------------------------
# reference (pure Python on lists; the CPU tests use it directly, check mode feeds it host copies)
# ------------------------------------------------------------------------------------------------------------
def lookup_rows_reference(token_rows, total_len, idx_mapping, plan_rows, draft_rows, periodic: bool,
                          max_delta: int, xor: int = 0, fill: bool = False):
    """token_rows[slot] -> list of tokens; plan_rows[b] = (src, lhost, k); draft_rows[b] = list (copied).
    Returns (new_draft_rows, status) with status[b] = 0 (row not written) or 1 + ids written."""
    out, status = [], []
    for b, (src, lhost, k) in enumerate(plan_rows):
        row = list(draft_rows[b])
        st = 0
        if k > 0:
            slot = idx_mapping[b]
            L = total_len[slot]
            toks = token_rows[slot]
            delta = L - lhost
            ok = 0 <= delta <= max_delta and 0 <= src < lhost <= L
            if ok:
                ok = all(toks[lhost + i] == toks[src + i] for i in range(delta))
            if ok:
                period = max(lhost - src, 1)
                written = 0
                for j in range(min(k, len(row))):
                    pos = src + delta + ((j % period) if periodic else j)
                    if pos < L:
                        row[j] = toks[pos] ^ xor
                        written += 1
                st = 1 + written
            elif fill and L > 0:
                for j in range(min(k, len(row))):
                    row[j] = toks[L - 1]
        out.append(row)
        status.append(st)
    return out, status


# ------------------------------------------------------------------------------------------------------------
# Triton kernel
# ------------------------------------------------------------------------------------------------------------
_KERNEL = {}


def _kernel():
    if "k" in _KERNEL:
        return _KERNEL["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _lookup_draft_kernel(tok_ptr, tok_stride, total_len_ptr, idx_map_ptr, plan_ptr, n_rows,
                             draft_ptr, draft_stride, status_ptr, n_steps, periodic, xor, max_delta, fill,
                             DELTA_BLOCK: tl.constexpr, KBLOCK: tl.constexpr):
        b = tl.program_id(0)
        src = tl.load(plan_ptr + b)
        lhost = tl.load(plan_ptr + n_rows + b)
        k = tl.minimum(tl.load(plan_ptr + 2 * n_rows + b), n_steps)
        slot = tl.load(idx_map_ptr + b).to(tl.int64)
        L = tl.load(total_len_ptr + slot)
        delta = L - lhost
        ok = (k > 0) & (delta >= 0) & (delta <= max_delta) & (src >= 0) & (src < lhost) & (lhost <= L)
        base = tok_ptr + slot * tok_stride
        offs = tl.arange(0, DELTA_BLOCK)
        m = (offs < delta) & ok
        a = tl.load(base + lhost + offs, mask=m, other=0)
        c = tl.load(base + src + offs, mask=m, other=0)
        bad = tl.sum(tl.where(m & (a != c), 1, 0), axis=0)
        ok = ok & (bad == 0)
        period = tl.maximum(lhost - src, 1)
        j = tl.arange(0, KBLOCK)
        off = tl.where(periodic != 0, j % period, j)
        pos = src + delta + off
        valid = ok & (j < k) & (pos < L)
        t = tl.load(base + pos, mask=valid, other=0)
        t = t ^ xor
        tl.store(draft_ptr + b * draft_stride + j, t.to(tl.int64), mask=valid)
        # TEST ONLY fill: failed rows get tokens[L-1] in every slot (L >= 1 whenever a plan exists)
        do_fill = (fill != 0) & (k > 0) & (ok == 0) & (L > 0)
        last = tl.load(base + tl.maximum(L - 1, 0), mask=do_fill, other=0)
        tl.store(draft_ptr + b * draft_stride + j, (last + j * 0).to(tl.int64), mask=do_fill & (j < k))
        written = tl.sum(valid.to(tl.int32), axis=0)
        tl.store(status_ptr + b, tl.where(ok, 1 + written, 0))

    _KERNEL["k"] = _lookup_draft_kernel
    return _lookup_draft_kernel


def launch(tokens, total_len, idx_mapping, plan_gpu, draft, status, n_rows: int, periodic: bool,
           max_delta: int, xor: int = 0, fill: bool = False) -> None:
    import triton
    _kernel()[(n_rows,)](
        tokens, tokens.stride(0), total_len, idx_mapping, plan_gpu, n_rows,
        draft, draft.stride(0), status, draft.shape[1], int(bool(periodic)), int(xor), int(max_delta), int(bool(fill)),
        DELTA_BLOCK=triton.next_power_of_2(max(int(max_delta), 1)),
        KBLOCK=triton.next_power_of_2(max(int(draft.shape[1]), 1)),
        num_warps=1,
    )


# ------------------------------------------------------------------------------------------------------------
# propose wrapper
# ------------------------------------------------------------------------------------------------------------
def _noop(*args, **kwargs):
    return None


@contextlib.contextmanager
def no_draft_forward(spec):
    """Suppress only the draft forward (eager _generate_draft and the FULL graph replay) for one propose."""
    mgr = getattr(spec, "query_cudagraph_manager", None)
    saved_gen = spec.__dict__.get("_generate_draft")
    saved_run = mgr.__dict__.get("run_fullgraph") if mgr is not None else None
    spec._generate_draft = _noop
    if mgr is not None:
        mgr.run_fullgraph = _noop
    try:
        yield
    finally:
        if saved_gen is None:
            spec.__dict__.pop("_generate_draft", None)
        else:
            spec._generate_draft = saved_gen
        if mgr is not None:
            if saved_run is None:
                mgr.__dict__.pop("run_fullgraph", None)
            else:
                mgr.run_fullgraph = saved_run


def plan_rows_for_batch(plan: dict, req_ids, num_reqs: int, n_steps: int):
    """(rows [(src, lhost, k)] in batch order, n_rows, n_skip_rows) for the batch; k = 0 for non-plan rows."""
    rows = plan.get("rows") or {}
    out, n, n_skip = [], 0, 0
    for b in range(num_reqs):
        r = rows.get(req_ids[b])
        if r is None:
            out.append((0, 0, 0))
            continue
        mode, src, lhost, k = r
        out.append((int(src), int(lhost), max(0, min(int(k), n_steps))))
        n += 1
        n_skip += int(mode == MODE_SKIP)
    return out, n, n_skip


def _trace(plan, req_ids, rows, skip, draft) -> None:
    if _rank() != 0:
        return
    rec = {"seq": plan.get("seq"), "skip": bool(skip),
           "rows": {req_ids[b]: [list(rows[b]), draft[b]] for b in range(len(rows)) if rows[b][2] > 0}}
    with open(TRACE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _make_propose(orig):
    def propose(self, input_batch, *args, **kwargs):
        ctx = self.__dict__.get("_glm_lk_ctx")
        if ctx is None or kwargs.get("dummy_run") or kwargs.get("is_profile") or _S.disabled:
            return orig(self, input_batch, *args, **kwargs)
        plan, req_states = ctx
        num_reqs = int(input_batch.num_reqs)
        n_steps = int(self.num_speculative_steps)
        rows, n_rows, n_skip = plan_rows_for_batch(plan, input_batch.req_ids, num_reqs, n_steps)
        _S.c["proposes"] += 1
        if n_rows == 0:
            return orig(self, input_batch, *args, **kwargs)
        skip = bool(plan.get("skip")) and n_skip == num_reqs and not bool(getattr(input_batch, "has_prefill", True))
        if skip:
            with no_draft_forward(self):
                orig(self, input_batch, *args, **kwargs)
        else:
            orig(self, input_batch, *args, **kwargs)
        import numpy as np
        import torch
        from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

        arr = np.asarray(rows, dtype=np.int32).T.copy()          # [3, num_reqs]: src, lhost, k
        plan_gpu = async_copy_to_gpu(arr, device=self.device)
        status = self.__dict__.get("_glm_lk_status")
        if status is None:
            status = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
            self._glm_lk_status = status
        draft = self.draft_tokens
        tokens = req_states.all_token_ids.gpu
        total_len = req_states.total_len.gpu
        periodic = bool(plan.get("periodic", True))
        max_delta = int(plan.get("max_delta", 64))
        xor = 1 if CORRUPT else 0
        before = draft[:num_reqs].clone() if CHECK else None
        launch(tokens, total_len, input_batch.idx_mapping, plan_gpu, draft, status, num_reqs, periodic,
               max_delta, xor, FILL)
        _S.c["lookup_steps"] += 1
        _S.c["rows"] += n_rows
        _S.c["skip_steps"] += int(skip)
        if CHECK:
            _check(tokens, total_len, input_batch.idx_mapping, rows, before, draft, status, num_reqs, periodic,
                   max_delta, xor)
        if TRACE:
            _trace(plan, input_batch.req_ids, rows, skip, draft[:num_reqs].tolist())
        if LOG_EVERY and _S.c["lookup_steps"] % LOG_EVERY == 0:
            _log(f"counters {_S.c}")
        return self.draft_tokens[:num_reqs]

    propose._glm_lookup = True
    return propose


def _check(tokens, total_len, idx_mapping, rows, before, draft, status, num_reqs, periodic, max_delta, xor):
    idx = idx_mapping[:num_reqs].tolist()
    tl_host = total_len.tolist()
    token_rows = {}
    for b, slot in enumerate(idx):
        if rows[b][2] > 0 and slot not in token_rows:
            token_rows[slot] = tokens[slot, :tl_host[slot]].tolist()
    ref, ref_status = lookup_rows_reference(token_rows, tl_host, idx, rows, before.tolist(), periodic, max_delta,
                                            xor, FILL)
    got = draft[:num_reqs].tolist()
    got_status = status[:num_reqs].tolist()
    for b in range(num_reqs):
        if rows[b][2] <= 0:
            continue
        _S.c["check_rows"] += 1
        _S.c["valid_rows"] += int(got_status[b] > 0)
        if got[b] != ref[b] or got_status[b] != ref_status[b]:
            _S.c["check_mismatch"] += 1
            if _S.c["check_mismatch"] <= 8:
                _log(f"CHECK MISMATCH row {b} plan {rows[b]} got {got[b]}/{got_status[b]} ref {ref[b]}/{ref_status[b]}")


# ------------------------------------------------------------------------------------------------------------
# runner hooks
# ------------------------------------------------------------------------------------------------------------
def _validate(runner) -> str | None:
    spec = getattr(runner, "speculator", None)
    if spec is None or not hasattr(spec, "parallel_drafting_token_id"):
        return "speculator is not DFlash / DFlash2"
    if getattr(spec, "draft_logits", None) is not None:
        return "probabilistic drafts (draft_logits set): a lookup token would be checked against DFlash2's q"
    if getattr(runner, "adaptive_verification", None) is not None:
        return "adaptive verification reads DFlash2 confidences"
    if os.environ.get("GLM_VERIFY_CUT", "").strip():
        return "GLM_VERIFY_CUT reads DFlash2 selector scores"
    if getattr(runner, "use_pp", False):
        return "pipeline parallel"
    return None


def install_runner(mod) -> None:
    cls = mod.GPUModelRunner
    if getattr(cls, "_glm_lookup", False):
        return
    orig_exec = cls.execute_model
    orig_sample = cls.sample_tokens

    def execute_model(self, scheduler_output, *args, **kwargs):
        plan = None
        if not kwargs.get("dummy_run", False) and scheduler_output is not None:
            plan = getattr(scheduler_output, "glm_lookup", None)
        self._glm_lk_next = plan
        return orig_exec(self, scheduler_output, *args, **kwargs)

    def sample_tokens(self, *args, **kwargs):
        plan = self.__dict__.pop("_glm_lk_next", None)
        spec = getattr(self, "speculator", None)
        if plan is None or spec is None:
            return orig_sample(self, *args, **kwargs)
        if _S.disabled is None:
            reason = _validate(self)
            _S.disabled = reason or ""
            if reason:
                _log(f"DISABLED ({reason}); stock drafting")
        if _S.disabled or plan.get("v") != 1:
            return orig_sample(self, *args, **kwargs)
        spec._glm_lk_ctx = (plan, self.req_states)
        try:
            return orig_sample(self, *args, **kwargs)
        finally:
            spec._glm_lk_ctx = None

    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    cls._glm_lookup = True
    _log("runner hooks armed")


def install_speculator(mod) -> None:
    cls = mod.DFlashSpeculator
    if getattr(cls.propose, "_glm_lookup", False):
        return
    cls.propose = _make_propose(cls.propose)
    _log("DFlash propose wrapped (lookup rows, forward skip)")


def check_sources(root: str | None = None) -> None:
    import glm_ds_hooks as h   # same overlay directory; import has no side effects
    got = h.hashes_at(root or h.vllm_root(), EXPECTED)
    bad = {k: (v, got[k]) for k, v in EXPECTED.items() if got[k] != v}
    if bad and not _on("GLM_LOOKUP_ALLOW_DRIFT"):
        lines = "\n".join(f"  {k}: expected {e}, image {g}" for k, (e, g) in sorted(bad.items()))
        raise RuntimeError("glm-lookup: engine sources differ from the qualified image; refusing to patch "
                           f"(unset GLM_LOOKUP_DRAFT or set GLM_LOOKUP_ALLOW_DRIFT=1):\n{lines}")


def register() -> None:
    if not enabled():
        return
    check_sources()
    import glm_ds_hooks as h
    h.after_import(TARGET_RUNNER, install_runner)
    h.after_import(TARGET_SPEC, install_speculator)

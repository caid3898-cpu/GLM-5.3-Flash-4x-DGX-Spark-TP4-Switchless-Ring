# SPDX-License-Identifier: Apache-2.0
"""glm_kv_lens_exact: sync-free EXACT kv_lens for FlashInferMLASparseSM90.

Removes the one blocking D2H copy per metadata build
(``positions[:num_rows].cpu()`` at flashinfer_mla_sparse_sm90.py:348 of the
deployed file, sha256 4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136)
from the DSA sparse-attention metadata path when async scheduling is on.

Mechanism (V2 model runner, vllm/v1/worker/gpu/):
  * The device-side ``req_states.num_computed_tokens`` is authoritative and is
    advanced ON DEVICE each step by ``post_update`` (input_batch.py:_post_update_kernel)
    with ``computed_delta = query_len - num_rejected``. Positions are built from
    it (input_batch.py:prepare_pos_seq_lens), so ``positions[:num_rows] + 1`` is
    the exact per-row context — but reading it costs the D2H sync.
  * The host mirror ``num_computed_tokens_np`` is only refreshed from the
    scheduler at update_requests, so under async scheduling it lags by the
    still-unprocessed step(s) — it is the "optimistic" upper bound.
  * The missing piece, step k-1's per-request ``num_sampled``, is already
    copied to the host WITHOUT any extra sync: ``AsyncOutput`` (async_utils.py)
    launches a non-blocking D2H of ``sampled_token_ids``/``num_sampled_tokens``
    during sample_tokens(k-1) and guards it with ``copy_event``. In the steady
    pipelined state that copy is complete by the time the next build runs
    (the engine core awaits step k-1's output before scheduling step k+1, and
    the copy is stream-ordered before the drafter forward the D2H used to wait
    behind). The event is checked with ``query()`` (non-blocking); if it has
    not fired we FALL BACK to the current device path.

So per build: exact per-request verified length
    value(i) = tracker(i) + delta(k-1, i),   delta = query_len - num_rejected
    (= num_sampled for verify rows, = query_len for prefill/no-draft rows),
then the same lens arithmetic as the deployed sync-free branch:
    lens = ctx if ctx <= index_topk else index_topk + ctx % index_kpool.
On any step where the mirror is incomplete (event not fired, unknown request,
CUDA-graph capture, warmup), the deployed device path runs verbatim and the
tracker is re-anchored from the exact ``positions`` the copy returns — so one
uncertain step costs one sync, never a wrong length.

Safety properties:
  * Default OFF. With the flag unset, register() does nothing: byte-identical
    behavior (the wrapper is never installed).
  * Fallback path is a verbatim copy of the deployed body; when it takes the
    positions branch it also re-anchors the tracker from the exact ctx.
  * Both paths compute the SAME exact lens, so per-rank eligibility races
    (event fired on rank A, not on rank B) cannot diverge TP results.
  * Never touches the device except through the existing deployed code paths.
  * Composes with GLM_KDA_STASH_NOCOPY / kpool / stash flags: it only wraps
    ``FlashInferMLASparseSM90Builder._kv_lens_host`` and observes runner state.

Deployed-semantics constraints the mirror depends on (each one forces the
fallback path when violated, checked per step from the runner):
  * ``InputBatch.num_computed_tokens_np`` is the ROW-indexed copy
    ``req_states.num_computed_tokens_np[idx_mapping_np]`` (model_runner.py:1350,
    passed at :1387); the added-request bootstrap indexes it by row, never by
    state index (the two orderings are independent).
  * No PCP partitioning (``runner.pcp_manager is None``): PCP would
    repartition the captured batch between the snapshot and the kernels.
  * No adaptive-verification draft reallocation (``runner.adaptive_verification
    is None``): it mutates cu_num_logits/query_start_loc on the GPU side only.
  * The draft model must not use this SM90 sparse-MLA builder (it does not:
    /draft is sliding-window); otherwise an in-``propose`` build could see a
    record for the same serial before it exists.

Enable: GLM_KVLENS_EXACT=1 env, or the marker file ``glm_kvlens_exact.enabled``
next to this module (env "0" wins over the marker). Debug (compares host vs
device lens every build and logs mismatches; reintroduces the sync): env
GLM_KVLENS_EXACT_DEBUG=1 or marker ``glm_kvlens_exact.debug``.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.abc
import importlib.util
import os
import sys
import threading
import time

import numpy as np
import torch

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENABLED_MARKER = os.path.join(_MODULE_DIR, "glm_kvlens_exact.enabled")
_DEBUG_MARKER = os.path.join(_MODULE_DIR, "glm_kvlens_exact.debug")
_STATUS_EVERY = 2000

_TRUE = ("1", "on", "true", "yes")
_FALSE = ("0", "off", "false", "no", "")


def _flag(env_name: str, marker: str) -> bool:
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip().lower() in _TRUE:
        return True
    if raw is not None and raw.strip().lower() not in _FALSE:
        return True
    if raw is not None and raw.strip().lower() in _FALSE:
        return False  # explicit env off wins over the marker file
    return os.path.exists(marker)


ENABLED = _flag("GLM_KVLENS_EXACT", _ENABLED_MARKER)
DEBUG = _flag("GLM_KVLENS_EXACT_DEBUG", _DEBUG_MARKER) if ENABLED else False


@dataclasses.dataclass
class TrackerEntry:
    value: int          # device num_computed_tokens for the request as of `serial`
    serial: int         # build serial that produced this value


@dataclasses.dataclass
class StepRecord:
    """Host-visible facts about one scheduled step (worker side)."""

    serial: int
    req_ids: list                       # row -> req_id (real rows only)
    idx_mapping_np: np.ndarray          # row -> req_state index (may hold -1)
    qsl_np: np.ndarray                  # query_start_loc host copy [num_reqs+1]
    cu_logits_np: np.ndarray            # cu_num_logits host copy [num_reqs+1]
    is_prefilling_np: np.ndarray        # per-row bool
    num_sampled_np: np.ndarray | None   # from AsyncOutput (rows of this step)
    copy_event: object | None           # AsyncOutput.copy_event (query())

    def delta_for(self, row: int) -> int:
        """computed_delta the device post_update applies for this row."""
        if int(self.idx_mapping_np[row]) < 0:
            return 0  # post_update skips filtered rows
        qlen = int(self.qsl_np[row + 1] - self.qsl_np[row])
        if self.is_prefilling_np[row]:
            return qlen  # num_rejected forced to 0 for chunked prefill rows
        if self.num_sampled_np is None:
            return 0  # counts never materialized: treat as no info
        nlogits = int(self.cu_logits_np[row + 1] - self.cu_logits_np[row])
        num_sampled = int(self.num_sampled_np[row])
        num_rejected = nlogits - num_sampled
        return qlen - num_rejected

    def row_of(self, req_id: str) -> int | None:
        # rows are few (<= max_num_seqs); linear scan is fine at build rate
        try:
            return self.req_ids.index(req_id)
        except ValueError:
            return None

    def fired(self) -> bool:
        if self.copy_event is None:
            return self.num_sampled_np is not None
        try:
            return bool(self.copy_event.query())
        except Exception:
            return False


class Registry:
    """Per-TP-worker host-side exact verified-length registry."""

    def __init__(self, debug: bool = DEBUG) -> None:
        self.lock = threading.Lock()
        self.debug = debug
        self.serial = 0
        self.current_ib = None              # current step InputBatch (duck-typed)
        self.current_ib_serial = -1
        self.added_now: set[str] = set()    # req_ids whose device nct was reset this step
        self.records: collections.deque = collections.deque(maxlen=8)
        self.tracker: dict[str, TrackerEntry] = {}
        self.desync = False                 # runner config the mirror cannot model
        self._desync_warned = False
        # counters
        self.builds = 0
        self.host_steps = 0
        self.device_steps = 0
        self.mismatches = 0
        self._last_status = time.monotonic()

    # -- hooks called from the wrapped runner methods -----------------------
    def note_serial(self) -> int:
        with self.lock:
            self.serial += 1
            self.added_now = set()
            return self.serial

    def note_current_batch(self, input_batch) -> None:
        with self.lock:
            self.current_ib = input_batch
            self.current_ib_serial = self.serial

    def note_added(self, req_id: str) -> None:
        with self.lock:
            self.added_now.add(req_id)

    def note_desync(self, desync: bool) -> None:
        """Latch an unsupported runner configuration (PCP partitioning,
        adaptive-verification draft reallocation): both mutate the batch or
        the logits layout between the InputBatch snapshot and the device
        kernels, so the host mirror cannot track deltas. Force the fallback
        path for every step while active (degrades to today's behavior)."""
        with self.lock:
            self.desync = bool(desync)
            if self.desync and not self._desync_warned:
                self._desync_warned = True
                sys.stderr.write(
                    "glm-kvlens-exact: PCP/adaptive-verification active; "
                    "forcing the deployed device path (host mirror off)\n"
                )

    def note_sample(self, input_batch, async_output) -> None:
        """Capture one step's host-visible sampled counts."""
        try:
            req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
            rec = StepRecord(
                serial=self.serial,
                req_ids=req_ids,
                idx_mapping_np=np.asarray(input_batch.idx_mapping_np),
                qsl_np=np.asarray(input_batch.query_start_loc_np),
                cu_logits_np=np.asarray(input_batch.cu_num_logits_np),
                is_prefilling_np=np.asarray(input_batch.is_prefilling_np),
                num_sampled_np=(
                    np.asarray(async_output.num_sampled_tokens_np)
                    if async_output is not None
                    and getattr(async_output, "num_sampled_tokens_np", None)
                    is not None
                    else None
                ),
                copy_event=getattr(async_output, "copy_event", None),
            )
        except Exception as exc:  # never break serving on diagnostics
            sys.stderr.write(f"glm-kvlens-exact: record failed: {exc!r}\n")
            return
        with self.lock:
            self.records.append(rec)
            self._prune_locked()

    def _prune_locked(self) -> None:
        if len(self.tracker) <= 4096:
            return
        cutoff = self.serial - 1024
        stale = [k for k, v in self.tracker.items() if v.serial < cutoff]
        for k in stale:
            del self.tracker[k]

    def _record_for(self, serial: int) -> StepRecord | None:
        for rec in self.records:
            if rec.serial == serial:
                return rec
        return None

    def _unfold_value_locked(self, req_id: str) -> int | None:
        """Device num_computed for req_id at the current build, or None.

        A record with serial s is created during sample_tokens(s), i.e.
        between build(s) and build(s+1); its post_update delta is already
        enqueued on the device at build(s+1). So a tracker entry committed at
        build(s) must be unfolded with records of serial >= s. Every serial in
        [entry.serial, self.serial - 1] must have a record (a scheduled step
        always produces one before the next build); a gap means we cannot
        account for the deltas and must fall back.
        """
        entry = self.tracker.get(req_id)
        if entry is None:
            return None
        for expected in range(entry.serial, self.serial):
            if self._record_for(expected) is None:
                return None  # unaccounted step: cannot mirror safely
        value = entry.value
        for rec in self.records:
            if rec.serial < entry.serial:
                continue
            row = rec.row_of(req_id)
            if row is None:
                continue  # not scheduled in that step: delta 0
            if rec.num_sampled_np is None:
                return None  # counts missing: cannot mirror safely
            if not rec.fired():
                return None  # acceptance unknown: cannot mirror
            value += rec.delta_for(row)
        return value

    # -- the decision -------------------------------------------------------
    def host_values(self, input_batch) -> list[int] | None:
        """Exact per-row device num_computed for the current build, or None.

        `input_batch` is the current step's InputBatch (duck-typed); rows are
        its real rows. Returns None when any row's value is unknown — the
        caller must then take the deployed device path.
        """
        with self.lock:
            if (
                self.desync
                or self.current_ib_serial != self.serial
                or input_batch is None
            ):
                return None
            try:
                req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
                nct_np = np.asarray(input_batch.num_computed_tokens_np)
            except Exception:
                return None
            if len(req_ids) == 0:
                return []
            values: list[int] = []
            for row, req_id in enumerate(req_ids):
                if req_id in self.added_now:
                    # add_request reset the device value from the scheduler
                    # count this step: the host mirror is exact for it.
                    # InputBatch.num_computed_tokens_np is the ROW-INDEXED
                    # copy req_states.num_computed_tokens_np[idx_mapping_np]
                    # (model_runner.py:1350, passed to InputBatch at :1387);
                    # row order comes from sort_batch_req_ids while state
                    # indices come from free_indices.pop() — independent
                    # orderings. Index by row, never by idx_mapping.
                    if row >= nct_np.shape[0]:
                        return None
                    values.append(int(nct_np[row]))
                    continue
                value = self._unfold_value_locked(req_id)
                if value is None:
                    return None
                values.append(value)
            return values

    def commit(self, input_batch, values: list[int]) -> None:
        with self.lock:
            try:
                req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
            except Exception:
                return
            for row, req_id in enumerate(req_ids):
                if row < len(values):
                    self.tracker[req_id] = TrackerEntry(int(values[row]), self.serial)
            self.host_steps += 1

    def invalidate(self, input_batch) -> None:
        """Forget tracker entries for the current batch (unsafe branch)."""
        with self.lock:
            try:
                req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
            except Exception:
                return
            for req_id in req_ids:
                self.tracker.pop(req_id, None)

    def heal(self, input_batch, ctx, qsl_cpu) -> None:
        """Re-anchor the tracker from the exact device ctx (fallback path).

        ctx[row] = positions[row] + 1, so the request's device num_computed is
        ctx[first_row_of_request] - 1. Only valid for branches that derive ctx
        from device positions (or from an exact host bound).
        """
        with self.lock:
            try:
                req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
                qsl = qsl_cpu.to(torch.int64)
                ctx64 = ctx.to(torch.int64)
                for row, req_id in enumerate(req_ids):
                    first = int(qsl[row])
                    self.tracker[req_id] = TrackerEntry(
                        int(ctx64[first]) - 1, self.serial
                    )
                self.device_steps += 1
            except Exception as exc:
                sys.stderr.write(f"glm-kvlens-exact: heal failed: {exc!r}\n")

    def note_mismatch(self, detail: str) -> None:
        self.mismatches += 1
        sys.stderr.write(
            f"GLM_KVLENS_EXACT_MISMATCH #{self.mismatches}: {detail}\n"
        )

    def maybe_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (self.builds % _STATUS_EVERY or now - self._last_status < 60):
            return
        self._last_status = now
        sys.stderr.write(
            "glm-kvlens-exact: builds=%d host=%d device=%d mismatches=%d "
            "tracked=%d\n"
            % (
                self.builds,
                self.host_steps,
                self.device_steps,
                self.mismatches,
                len(self.tracker),
            )
        )


REGISTRY = Registry()


# --------------------------------------------------------------------------
# Pure lens arithmetic (no vllm imports) — mirrors the deployed sync-free
# branch exactly. Exposed for the CPU test-suite.
# --------------------------------------------------------------------------

def lens_from_values(
    values: list[int],
    qsl_cpu: torch.Tensor,
    cam_num_reqs: int,
    index_topk: int,
    index_kpool: int,
) -> tuple[int, torch.Tensor]:
    """kv_lens from exact per-request verified lengths.

    Same arithmetic as the deployed _kv_lens_host sync-free branch with
    first_pos = value (seq_lens - q_len == value because positions are
    contiguous per request).
    """
    qsl = qsl_cpu[: cam_num_reqs + 1]
    num_rows = int(qsl[-1])
    if num_rows == 0:
        return 0, torch.zeros(0, dtype=torch.int32)
    q_lens = qsl[1:] - qsl[:-1]
    v = torch.zeros(cam_num_reqs, dtype=torch.int64)
    real = min(len(values), cam_num_reqs)
    if real:
        v[:real] = torch.tensor(values[:real], dtype=torch.int64)
    first_pos = v
    req_of_row = torch.repeat_interleave(
        torch.arange(cam_num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
    )
    rows = torch.arange(num_rows, dtype=torch.int32)
    ctx = (
        first_pos[req_of_row]
        + rows.to(torch.int64)
        - qsl.to(torch.int64)[req_of_row]
        + 1
    )
    topk = int(index_topk)
    kpool = max(int(index_kpool), 1)
    lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
    return num_rows, lens.to(torch.int32)


# --------------------------------------------------------------------------
# Verbatim fallback: the deployed _kv_lens_host body, returning the branch tag
# and the raw ctx so the registry can re-anchor. Diffed against the deployed
# file (sha256 4449ea25…8f136); behavior-identical.
# --------------------------------------------------------------------------

def _original_body_with_ctx(builder, cam):
    num_reqs = cam.num_reqs
    qsl = cam.query_start_loc_cpu[: num_reqs + 1]
    num_rows = int(qsl[-1])
    if num_rows == 0:
        return 0, None, torch.zeros(0, dtype=torch.int32), "empty"
    sl_host = getattr(cam, "seq_lens_cpu_upper_bound", None)
    positions = getattr(cam, "positions", None)
    if not getattr(builder, "_async_scheduling", False) and sl_host is not None:
        seq_lens = sl_host[:num_reqs].to(torch.int32)
        q_lens = qsl[1:] - qsl[:-1]
        first_pos = seq_lens - q_lens
        req_of_row = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
        )
        rows = torch.arange(num_rows, dtype=torch.int32)
        ctx = (
            first_pos.to(torch.int64)[req_of_row]
            + rows.to(torch.int64)
            - qsl.to(torch.int64)[req_of_row]
            + 1
        )
        branch = "upper_bound"
    elif positions is not None and num_rows <= positions.shape[0]:
        ctx = positions[:num_rows].cpu().to(torch.int64) + 1
        branch = "positions"
    else:
        seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int32)
        q_lens = qsl[1:] - qsl[:-1]
        first_pos = seq_lens - q_lens
        req_of_row = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
        )
        rows = torch.arange(num_rows, dtype=torch.int32)
        ctx = (
            first_pos.to(torch.int64)[req_of_row]
            + rows.to(torch.int64)
            - qsl.to(torch.int64)[req_of_row]
            + 1
        )
        branch = "optimistic_seq_lens"
    topk = builder._index_topk
    kpool = max(builder._index_kpool, 1)
    lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
    return num_rows, ctx, lens.to(torch.int32), branch


# --------------------------------------------------------------------------
# In-container glue: the patched _kv_lens_host.
# --------------------------------------------------------------------------

def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


_ORIG_KV_LENS_HOST = None
_LAST_CAM = None
_LAST_RESULT = None


def _patched_kv_lens_host(self, cam):
    global _LAST_CAM, _LAST_RESULT
    reg = REGISTRY
    reg.builds += 1

    with reg.lock:
        if _LAST_CAM is cam and _LAST_RESULT is not None:
            reg.builds -= 1  # same build re-entered; do not double-count
            return _LAST_RESULT

    input_batch = reg.current_ib
    eligible = False
    values = None
    if (
        input_batch is not None
        and reg.current_ib_serial == reg.serial
        and not reg.desync
        and not _capturing()
    ):
        values = reg.host_values(input_batch)
        eligible = values is not None

    if eligible:
        num_rows, lens = lens_from_values(
            values,
            cam.query_start_loc_cpu,
            cam.num_reqs,
            self._index_topk,
            self._index_kpool,
        )
        if DEBUG:
            # No-traffic verification: also run the deployed device path and
            # compare. Keeps the device result (expected identical).
            d_num_rows, _ctx, d_lens, _branch = _original_body_with_ctx(self, cam)
            if d_num_rows != num_rows or not torch.equal(d_lens, lens):
                reg.note_mismatch(
                    f"num_rows {d_num_rows} vs {num_rows}; "
                    f"lens diff at {torch.nonzero(d_lens != lens)[:8].tolist()}"
                )
            lens = d_lens
            reg.heal(input_batch, _ctx, cam.query_start_loc_cpu)
        else:
            reg.commit(input_batch, values)
        result = (num_rows, lens)
        reg.maybe_status()
    else:
        num_rows, ctx, lens, branch = _original_body_with_ctx(self, cam)
        if input_batch is not None and reg.current_ib_serial == reg.serial:
            if branch in ("positions", "upper_bound"):
                reg.heal(input_batch, ctx, cam.query_start_loc_cpu)
            elif branch == "optimistic_seq_lens":
                # ctx derived from the optimistic bound: do NOT anchor.
                reg.invalidate(input_batch)
        result = (num_rows, lens)
        reg.maybe_status()

    with reg.lock:
        _LAST_CAM = cam
        _LAST_RESULT = result
    return result


# --------------------------------------------------------------------------
# Import-hook installation (same pattern as the overlay sitecustomize).
# --------------------------------------------------------------------------

class _KVLensHook(importlib.abc.MetaPathFinder):
    def __init__(self, targets: dict[str, object]) -> None:
        self._targets = targets

    def find_spec(self, name, path, target=None):
        patcher = self._targets.get(name)
        if patcher is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def patched_exec(module, _patcher=patcher, _exec=exec_module):
            _exec(module)
            try:
                _patcher(module)
            except Exception as exc:
                sys.stderr.write(
                    f"glm-kvlens-exact: patch for {name} failed: {exc!r}\n"
                )

        loader.exec_module = patched_exec
        return spec


def _patch_mla_module(mod) -> None:
    global _ORIG_KV_LENS_HOST
    cls = mod.FlashInferMLASparseSM90Builder
    _ORIG_KV_LENS_HOST = cls._kv_lens_host
    cls._kv_lens_host = _patched_kv_lens_host
    sys.stderr.write(
        "glm-kvlens-exact: _kv_lens_host wrapped "
        f"(enabled={ENABLED}, debug={DEBUG})\n"
    )


def _patch_runner_module(mod) -> None:
    runner_cls = mod.GPUModelRunner
    orig_execute = runner_cls.execute_model

    def execute_model(self, *args, **kwargs):
        try:
            so = args[0] if args else kwargs.get("scheduler_output")
            dummy = kwargs.get("dummy_run", False)
            if not dummy and len(args) > 2 and isinstance(args[2], bool):
                dummy = args[2]
            if (
                so is not None
                and int(getattr(so, "total_num_scheduled_tokens", 0) or 0) > 0
                and not dummy
            ):
                REGISTRY.note_serial()
            # Configurations the mirror cannot model (see Registry.note_desync):
            # PCP repartitions the captured batch (model_runner.py:1407);
            # adaptive verification reallocates drafts on the GPU side only,
            # desyncing host cu_num_logits (model_runner.py:1264-1278).
            REGISTRY.note_desync(
                getattr(self, "pcp_manager", None) is not None
                or getattr(self, "adaptive_verification", None) is not None
            )
        except Exception:
            pass
        return orig_execute(self, *args, **kwargs)

    runner_cls.execute_model = execute_model
    orig_sample = runner_cls.sample_tokens

    def sample_tokens(self, *args, **kwargs):
        state = getattr(self, "execute_model_state", None)
        input_batch = getattr(state, "input_batch", None) if state else None
        result = orig_sample(self, *args, **kwargs)
        try:
            if (
                input_batch is not None
                and result is not None
                and hasattr(result, "copy_event")
            ):
                REGISTRY.note_sample(input_batch, result)
        except Exception:
            pass
        return result

    runner_cls.sample_tokens = sample_tokens
    sys.stderr.write("glm-kvlens-exact: runner hooks installed\n")


def _patch_states_module(mod) -> None:
    state_cls = mod.RequestState
    orig_add = state_cls.add_request

    def add_request(self, req_id, *args, **kwargs):
        try:
            REGISTRY.note_added(req_id)
        except Exception:
            pass
        return orig_add(self, req_id, *args, **kwargs)

    state_cls.add_request = add_request


def _patch_input_batch_module(mod) -> None:
    orig_cls = mod.InputBatch

    class _KVLensInputBatch(orig_cls):
        # Subclass, NOT a function replacement: classmethod
        # InputBatch.make_dummy (input_batch.py:116) and isinstance checks
        # must keep working on the dummy/profiling/capture paths
        # (model_runner.py:766, :1598; cudagraph_utils.py:755;
        # spec_decode/dflash/cudagraph.py:34). A dummy batch notes
        # current_ib with a stale serial, which host_values rejects.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                REGISTRY.note_current_batch(self)
            except Exception:
                pass

    mod.InputBatch = _KVLensInputBatch


def status() -> dict:
    reg = REGISTRY
    return {
        "enabled": ENABLED,
        "debug": DEBUG,
        "builds": reg.builds,
        "host_steps": reg.host_steps,
        "device_steps": reg.device_steps,
        "mismatches": reg.mismatches,
        "tracked": len(reg.tracker),
        "installed": _ORIG_KV_LENS_HOST is not None,
    }


def register() -> None:
    """Idempotent. No-op unless ENABLED (default OFF)."""
    if not ENABLED:
        return
    targets = {
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90": _patch_mla_module,
        "vllm.v1.worker.gpu.model_runner": _patch_runner_module,
        "vllm.v1.worker.gpu.states": _patch_states_module,
        "vllm.v1.worker.gpu.input_batch": _patch_input_batch_module,
    }
    if not any(isinstance(f, _KVLensHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _KVLensHook(targets))
    sys.stderr.write(
        f"glm-kvlens-exact: registered (enabled={ENABLED}, debug={DEBUG})\n"
    )

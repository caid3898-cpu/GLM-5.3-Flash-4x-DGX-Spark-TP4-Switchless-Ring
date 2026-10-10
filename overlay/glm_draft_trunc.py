# SPDX-License-Identifier: Apache-2.0
"""GLM_DRAFT_TRUNC: choose the REAL verify shape from the current draft. GLM-5.3-Flash,
vLLM 487ecf187 V2 runner, async scheduling, DFlash2, LeversScheduler. SPEED FIRST.

Plain speculative decoding with a shorter draft: the target verifies the first L drafts of this step (M = L + 1
rows per request, the family's ordinary FULL graph), the standard sampler accepts / resamples, nothing is remapped
and no -1 sentinel exists. Exact at T = 0 and T > 0 (rejection sampling over a truncated draft is valid: q is just
shorter). Only the shape changes.

Mechanism (variant-switchable in-boot through glm_ab, key GLM_DRAFT_TRUNC, kind raw):
  engine core   while the variant is active every request is scheduled with engine_k (7) placeholders (the
                SpecProbeScheduler's default_k path), so the adaptive-k EMA plays no role and sees no false zeros;
                the core follows glm_ab switches (EngineCore.collective_rpc wrapper records the variant).
  worker        after DFlash2's propose: rank 0's realized selector scores [n, 7, 16] are broadcast (one tiny
                collective, so every rank decides from identical inputs), reduced on device to the per-depth
                confidence log max softmax [n, 7], copied to a pinned buffer (non-blocking) with an event.
                At the start of the next GPUModelRunner.execute_model: wait for that event (the draft end: this is
                the late-decision cost, the early plan's overlap of the host prepare with the drafter is lost for
                this step), then per request E(L) = 1 + sum_{s<L} prod_{i<=s} g[i][bin(conf_i)] (table fitted on
                the current drafter, capture 1633) and
                  c = 1:  L* = argmax_{1<=L<=scheduled} E(L) - lam * T1(L + 1)
                  c > 1:  one uniform L* for the batch (FULL graphs need a uniform query length):
                          argmax_L sum_r E_r(L) - lam * Tn(n, n (L + 1))
                lam = running predicted throughput (EMA 0.05 of E / T; per batch size bucket). The scheduler
                output is truncated in place (scheduled_spec_decode_tokens[r] = spec[:L*], num_scheduled_tokens,
                total_num_scheduled_tokens) BEFORE the runner reads it; the engine core rolls back the unused
                placeholders like any rejection. Rank-invariant: identical broadcast inputs, identical host math.
                Skipped (no truncation) for: dummy / profile runs, steps with structured-output requests, requests
                without a confidence record (new), c > 1 batches where any request lacks one.
Cost table: GLM_DRAFT_TRUNC_COST (JSON {"c1": {"2": ms, ..., "8": ms}, "row_ms": ms per extra row beyond 8}),
default = the bavprobe-1629 model; replace with the frontier probe's measured real shapes.
Table: GLM_DRAFT_TRUNC_TABLE (default /overlay/overlay/glm_bav_table_seg.json: "edges" + "g" [7][bins]).
Needs verify graph families for every M the policy can choose (SPEC_TABLE listing k 1..7, CAPTURE_SIZES with 2..8 and
the c > 1 products); a missing family falls to a PIECEWISE / eager step (slow, still exact).
Log: every GLM_DRAFT_TRUNC_LOG_EVERY (2000) truncated steps, rank 0: chosen-L histogram, lam, wait ms.

v2 (follow-up to A/B 2209; every knob off = the 2209 behaviour exactly):
  GLM_DRAFT_TRUNC_GATE=<thr>   (glm_ab raw key; 0 = off) skip the late decision on steps where it can only keep the
                full shape: if the policy's decision for the PREVIOUS draft was kmax with a margin >= thr (tokens,
                obj(kmax) - max_{L<kmax} obj(L)), this step keeps the scheduled k without waiting for the draft
                (no truncation: the early plan and the host run-ahead stay intact, ~1.7 ms/step saved on those steps).
                The skipped draft's confidences are scored post hoc at the next step (its event fired long before),
                so the gate always sees the policy's decision for the previous draft. Rank-invariant: every input is
                the broadcast confidence, scored in the same order on every rank; timing never enters a decision.
  GLM_DRAFT_TRUNC_EARLYSKIP=1  (glm_ab raw key) steps the policy would skip anyway (structured output, a request
                without a confidence record, mixed scheduled k) no longer wait for the draft first.
  GLM_DRAFT_TRUNC_PERIODS=1    (rank 0, logging only) GPU period draft-end -> next draft-end per (variant, batch
                size, verify rows, mode w=waited / g=gated / s=skipped-no-wait / x=skipped-after-wait), from timing
                events read lagged with query(); the same shape waited vs gated is the exposed late-decision cost,
                and the per-shape medians are the live cost table (c1 and c4) for GLM_DRAFT_TRUNC_COST.
Confidences go through a ring of 3 pinned slots (the gate keeps one draft unscored for one step).

v3 (DEVSELECT): with GLM_DEVSELECT installed (overlay/glm_devselect.py), c1 steps with 7 scheduled
drafts skip the late decision entirely: the 8-row step is prepared as usual and glm_devselect replays a parent graph
whose selector picks the verify shape on the device. Every GLM_DEVSELECT knob off = the v2 behaviour exactly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

_OFF = ("", "0", "off", "false", "no")
TABLE = os.environ.get("GLM_DRAFT_TRUNC_TABLE", "/overlay/overlay/glm_bav_table_seg.json")
COST = os.environ.get("GLM_DRAFT_TRUNC_COST", "")
LOG_EVERY = int(os.environ.get("GLM_DRAFT_TRUNC_LOG_EVERY", "2000") or 0)
PERIODS = str(os.environ.get("GLM_DRAFT_TRUNC_PERIODS", "0")).strip().lower() not in ("", "0", "off", "false", "no")
RING = 3
T1_DEFAULT = {2: 30.0, 3: 32.4, 4: 35.7, 5: 38.8, 6: 42.0, 7: 44.9, 8: 47.9}
ROW_MS_DEFAULT = 1.2
ALPHA = 0.05


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-draft-trunc: {msg}\n")
    sys.stderr.flush()


def _knob(name: str, default: str = "0") -> str:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return str(ab.env(name, default))
    return str(os.environ.get(name, default))


def _gate_thr() -> float:
    v = _knob("GLM_DRAFT_TRUNC_GATE", "0").strip().lower()
    if v in _OFF:
        return 0.0
    try:
        return max(float(v), 0.0)
    except ValueError:
        return 0.0


def _variant() -> int:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        try:
            return int(ab.current())
        except Exception:  # noqa: BLE001
            return -1
    return 0


def active() -> bool:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        v = ab.env("GLM_DRAFT_TRUNC", "0")
    else:
        v = os.environ.get("GLM_DRAFT_TRUNC", "0")
    return str(v).strip().lower() not in _OFF


# --------------------------------------------------------------------------- policy (pure, tested on CPU)
class Policy:
    def __init__(self, table: dict, cost: dict | None = None):
        self.edges = [float(x) for x in table["edges"]]
        self.g = [[float(v) for v in row] for row in table["g"]]
        # bins below the lowest populated one keep the fit's 0.5 prior (no data: log max softmax over 16 candidates
        # is >= -2.77, so they are rarely hit); extend the lowest populated value downwards instead
        for row in self.g:
            first = next((b for b, v in enumerate(row) if v != 0.5), None)
            if first is not None:
                for b in range(first):
                    row[b] = row[first]
        cost = cost or {}
        self.t1 = {int(k): float(v) for k, v in (cost.get("c1") or T1_DEFAULT).items()}
        for m in range(2, 9):
            self.t1.setdefault(m, T1_DEFAULT[m])
        self.row_ms = float(cost.get("row_ms", ROW_MS_DEFAULT))
        self.lam: dict = {}

    def cost(self, n: int, q: int) -> float:
        """Step ms for n requests of q rows each (uniform)."""
        tot = n * q
        return self.t1[min(max(tot, 2), 8)] + max(0, tot - 8) * self.row_ms

    def expect(self, conf):
        """conf [<=7] log max softmax per depth -> E[0..len]"""
        E, cum = [1.0], 1.0
        for s, lq in enumerate(conf):
            b = sum(1 for e in self.edges if lq >= e)
            cum *= self.g[s][b]
            E.append(E[-1] + cum)
        return E

    def choose(self, confs, kmax: int) -> int:
        """confs: per request [7] confidences; kmax: scheduled drafts (uniform). -> L* (1..kmax), uniform."""
        n = len(confs)
        Es = [self.expect(c[:kmax]) for c in confs]
        lam = self.lam.get(n, 2.2 / self.cost(1, 4) * (n ** 0.7))
        best, bL, vals = -1e30, kmax, []
        for L in range(1, kmax + 1):
            v = sum(E[L] for E in Es) - lam * self.cost(n, L + 1)
            vals.append(v)
            if v > best:
                best, bL = v, L
        self.lam[n] = (1 - ALPHA) * lam + ALPHA * sum(E[bL] for E in Es) / self.cost(n, bL + 1)
        # how clearly the full shape won (tokens; < 0 when it lost): the gate's input
        self.last_margin = vals[-1] - max(vals[:-1]) if kmax > 1 else float("inf")
        self.last_kmax = kmax
        return bL


# --------------------------------------------------------------------------- worker
class _W:
    policy = None
    rank0 = True
    pend = None          # (event, n, req_ids tuple, ring slot) of the latest draft
    pin = None           # 2209 name kept for the tests; the ring below is what propose writes
    ring = None          # RING pinned [>=64, 7] fp32 slots
    slot_i = 0
    unscored = None      # (event, n, req_ids, slot, kmax) of a gated draft, scored at the next step
    last = None          # (L, margin, kmax) the policy's decision for the previous draft, or None
    cur_meta = None      # (variant, n, rows, mode) of the step being executed (periods)
    timeline = None      # deque of (event, meta) (periods, rank 0)
    per = {}             # (variant, n, rows, mode) -> [ms]
    st = {}              # variant -> counters
    pols = {}            # variant -> Policy (own lam per arm; the template is `policy`)
    v_last = None


def _stats(v: int) -> dict:
    d = _W.st.get(v)
    if d is None:
        d = _W.st[v] = {"hist": [0] * 9, "steps": 0, "wait_ms": 0.0, "waited": 0, "skipped": 0, "gated": 0,
                        "posthoc": 0, "early": 0}
    return d


def _load_policy():
    t = json.load(open(TABLE))
    c = json.load(open(COST)) if COST else None
    return Policy(t, c)


def _patch_propose(mod) -> None:
    import torch

    cls = mod.DFlash2Speculator
    orig = cls.propose

    def propose(self, input_batch, *args, **kwargs):
        out = orig(self, input_batch, *args, **kwargs)
        _W.pend = None
        if kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing() or not active() or _W.policy is None:
            return out
        n = int(input_batch.num_reqs)
        if n == 0:
            return out
        sc = self._selector_scores[:n].contiguous()
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        if tp.world_size > 1:
            tp.device_communicator.broadcast(sc, 0)            # identical decision inputs on every rank
        conf = torch.log_softmax(sc.float(), dim=-1).amax(dim=-1)   # [n, 7]
        if _W.ring is None or _W.ring[0].shape[0] < n or _W.ring[0].shape[1] != conf.shape[1]:
            if _W.unscored is not None:        # never reallocate under an unread slot
                _W.unscored = None
                _W.last = None
            _W.ring = [torch.empty(max(64, n), conf.shape[1], dtype=torch.float32, pin_memory=True)
                       for _ in range(RING)]
            _W.pin = _W.ring[0]
        slot = _W.slot_i % RING
        _W.slot_i += 1
        ds = sys.modules.get("glm_devselect")
        if ds is not None:
            ds.note_propose(conf, n)
        _W.ring[slot][:n].copy_(conf, non_blocking=True)
        timing = PERIODS and _W.rank0
        ev = torch.cuda.Event(enable_timing=timing)
        ev.record()
        _W.pend = (ev, n, tuple(input_batch.req_ids[:n]), slot)
        if timing:
            _note_period(ev)
        return out

    cls.propose = propose
    _log("DFlash2Speculator.propose hooked (score broadcast + confidence D2H)")


def _note_period(ev) -> None:
    """rank 0 logging only: the GPU period between consecutive draft ends belongs to the step whose draft ended."""
    import collections
    if _W.timeline is None:
        _W.timeline = collections.deque()
    _W.timeline.append((ev, _W.cur_meta))
    _W.cur_meta = None
    while len(_W.timeline) >= 2:
        (e0, _), (e1, meta) = _W.timeline[0], _W.timeline[1]
        try:
            if not e1.query():
                break
            ms = e0.elapsed_time(e1)
        except Exception:  # noqa: BLE001
            ms = None
        _W.timeline.popleft()
        if meta is not None and ms is not None and ms < 250.0:        # idle gaps between requests dropped
            _W.per.setdefault(meta, []).append(ms)
    while len(_W.timeline) > 16:
        _W.timeline.popleft()


def _period_report(v: int) -> str:
    rows = []
    for key in sorted(k for k in _W.per if k[0] == v):
        xs = sorted(_W.per.pop(key))
        if len(xs) >= 5:
            rows.append(f"{key[1]}x{key[2]}{key[3]}: {xs[len(xs) // 2]:.2f} [{xs[len(xs) // 10]:.2f},"
                        f"{xs[(9 * len(xs)) // 10]:.2f}] n{len(xs)}")
    return "; ".join(rows) if rows else "none"


def eligible(scheduler_output, rids) -> int | None:
    """The uniform scheduled k when truncate() would act on this step (same skips), else None. Pure host."""
    spec = scheduler_output.scheduled_spec_decode_tokens
    if not spec or getattr(scheduler_output, "has_structured_output_requests", False):
        return None
    have = set(rids)
    live = [r for r, x in spec.items() if x]
    if not live or any(r not in have for r in live):
        return None
    ks = {len(spec[r]) for r in live}
    return ks.pop() if len(ks) == 1 else None


def _choose(policy: Policy, confs, kmax: int) -> int:
    """Policy.choose; a c1 decision shares its lam with glm_devselect's device lam (parity fix 2026-09-29)."""
    ds = sys.modules.get("glm_devselect")
    sync = ds is not None and len(confs) == 1 and hasattr(ds, "lam_pull")
    if sync:
        ds.lam_pull(policy)
    L = policy.choose(confs, kmax)
    if sync:
        ds.lam_push(policy)
    return L


def truncate(scheduler_output, confs_by_req: dict, policy: Policy):
    """Pure host step (tested on CPU): -> chosen L or None (untouched)."""
    spec = scheduler_output.scheduled_spec_decode_tokens
    if not spec or getattr(scheduler_output, "has_structured_output_requests", False):
        return None
    rids = [r for r, s in spec.items() if s]
    if not rids or any(r not in confs_by_req for r in rids):
        return None
    ks = {len(spec[r]) for r in rids}
    if len(ks) != 1:
        return None                                  # mixed scheduled k: keep the scheduler's shape
    kmax = ks.pop()
    L = _choose(policy, [confs_by_req[r] for r in rids], kmax)
    if L >= kmax:
        return L
    cut = 0
    for r in rids:
        d = len(spec[r]) - L
        spec[r] = spec[r][:L]
        scheduler_output.num_scheduled_tokens[r] -= d
        cut += d
    scheduler_output.total_num_scheduled_tokens -= cut
    return L


def _patch_runner(mod) -> None:
    cls = mod.GPUModelRunner
    o_init, o_exec = cls.__init__, cls.execute_model

    def __init__(self, vllm_config, device, *a, **kw):
        o_init(self, vllm_config, device, *a, **kw)
        try:
            _W.policy = _load_policy()
            from vllm.distributed.parallel_state import get_tp_group
            _W.rank0 = get_tp_group().rank_in_group == 0
            _log(f"armed: table {TABLE} ({len(_W.policy.g)}x{len(_W.policy.edges) + 1}), T1 {_W.policy.t1}, "
                 f"row_ms {_W.policy.row_ms}, rank0={_W.rank0}")
        except Exception as exc:  # noqa: BLE001
            _W.policy = None
            _log(f"policy load failed ({exc!r}): truncation off")

    def execute_model(self, scheduler_output, *args, **kwargs):
        ds = sys.modules.get("glm_devselect")
        if ds is not None:
            ds._D.go = False
        if kwargs.get("dummy_run") or not active() or _W.policy is None:
            return o_exec(self, scheduler_output, *args, **kwargs)
        v = _variant()
        st = _stats(v)
        pol = _W.pols.get(v)
        if pol is None:
            import copy
            pol = _W.pols[v] = copy.deepcopy(_W.policy)
        if v != _W.v_last:                  # glm_ab switches only with no request in flight, on every rank alike
            _W.v_last, _W.unscored, _W.last = v, None, None
        # 1. a draft gated last step is scored now (its event fired a step ago: no wait on the critical path);
        #    the same inputs in the same order on every rank, so the gate below stays rank-invariant
        if _W.unscored is not None:
            ev, n, idx, slot, kmax = _W.unscored
            _W.unscored = None
            ev.synchronize()
            arr = _W.ring[slot][:n].tolist()
            L = _choose(pol, [arr[i] for i in idx], kmax)     # the live requests, in truncate()'s order
            _W.last = (L, pol.last_margin, kmax)
            st["posthoc"] += 1
        pend, _W.pend = _W.pend, None
        mode = None
        go = False
        if ds is not None:
            go = ds.eligible(self, scheduler_output, pend)
            ds.begin_step(self, scheduler_output, go)
        if go:                                   # device-side selection: no wait, no host truncation
            _W.unscored, _W.last = None, None
            st["dev"] = st.get("dev", 0) + 1
            st["steps"] += 1
            mode = "d"
        elif pend is not None:
            ev, n, rids, slot = pend
            kmax = eligible(scheduler_output, rids)
            thr = _gate_thr()
            last = _W.last
            if kmax is None and _knob("GLM_DRAFT_TRUNC_EARLYSKIP", "0").strip().lower() not in _OFF:
                st["skipped"] += 1; st["early"] += 1; _W.last = None; mode = "s"
            elif (kmax is not None and thr > 0.0 and last is not None and last[0] == last[2] == kmax
                  and last[1] >= thr):
                pos = {r: i for i, r in enumerate(rids)}
                live = [pos[r] for r, x in scheduler_output.scheduled_spec_decode_tokens.items() if x]
                _W.unscored = (ev, n, live, slot, kmax)        # keep the full shape, decide nothing now
                st["gated"] += 1; mode = "g"
            else:
                t0 = time.perf_counter()
                ev.synchronize()                               # the draft end: the late decision
                st["wait_ms"] += (time.perf_counter() - t0) * 1e3
                st["waited"] += 1
                arr = _W.ring[slot][:n].tolist()
                L = truncate(scheduler_output, dict(zip(rids, arr)), pol)
                if L is None:
                    st["skipped"] += 1; _W.last = None; mode = "x"
                else:
                    st["hist"][min(L, 8)] += 1
                    _W.last = (L, pol.last_margin, pol.last_kmax)
                    mode = "w"
            st["steps"] += 1
            if _W.rank0 and LOG_EVERY and st["steps"] % LOG_EVERY == 0:
                w = max(st["waited"], 1)
                _log(f"v{v} steps {st['steps']}: L hist {st['hist'][1:8]} lam {{{', '.join(f'{k}: {x:.4f}' for k, x in sorted(pol.lam.items()))}}} "
                     f"mean wait {st['wait_ms'] / w:.3f} ms over {st['waited']} waited, skipped {st['skipped']} "
                     f"(early {st['early']}) gated {st['gated']} posthoc {st['posthoc']} dev {st.get('dev', 0)}")
                if PERIODS:
                    _log(f"v{v} periods ms (n x rows mode: median [p10,p90] count): {_period_report(v)}")
                st["wait_ms"], st["waited"] = 0.0, 0
        if PERIODS and _W.rank0:
            _W.cur_meta = (v, len(scheduler_output.num_scheduled_tokens),
                           int(scheduler_output.total_num_scheduled_tokens), mode or "-")
        return o_exec(self, scheduler_output, *args, **kwargs)

    cls.__init__, cls.execute_model = __init__, execute_model
    _log("GPUModelRunner.execute_model hooked (in-place truncation of the scheduled drafts)")


# --------------------------------------------------------------------------- engine core
def _patch_core(mod) -> None:
    cls = mod.EngineCore
    orig = cls.collective_rpc

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        res = orig(self, method, timeout, args, kwargs)
        if method == "glm_ab_switch":
            try:
                ok = res and all(isinstance(r, dict) and "variant" in r and not r.get("error") for r in res)
                ab = sys.modules.get("glm_ab")
                if ok and ab is not None and getattr(ab, "ACTIVE", False):
                    ab.set_runtime(int(res[0]["variant"]))
                    _log(f"engine core follows glm_ab variant {int(res[0]['variant'])} (truncation "
                         f"{'on' if active() else 'off'})")
            except Exception as exc:  # noqa: BLE001
                _log(f"core variant tracking failed: {exc!r}")
        return res

    cls.collective_rpc = collective_rpc
    _log("EngineCore.collective_rpc wrapped (glm_ab variant tracking in the scheduler process)")


def _patch_sched(mod) -> None:
    cls = getattr(mod, "SpecProbeScheduler", None)
    if cls is None:
        return

    def _get(self):
        if active():
            return int(getattr(self, "_ak_engine_k", 0) or 0) or self.__dict__.get("_pdk")
        return self.__dict__.get("_pdk")

    def _set(self, v):
        self.__dict__["_pdk"] = v

    cls._probe_default_k = property(_get, _set)
    _log("SpecProbeScheduler: default_k = engine_k while the truncation variant is active")


TARGETS = {"vllm.v1.worker.gpu.model_runner": _patch_runner,
           "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": _patch_propose,
           "vllm.v1.engine.core": _patch_core,
           "spec_probe_scheduler": _patch_sched}


def register() -> None:
    import importlib.abc
    import importlib.util

    if str(os.environ.get("GLM_DRAFT_TRUNC", "0")).strip().lower() in _OFF:
        return
    if str(os.environ.get("GLM_DEVSELECT", "0")).strip().lower() not in _OFF:
        import glm_devselect
        glm_devselect.register()
    fns = dict(TARGETS)
    for name in [n for n in fns if n in sys.modules]:
        fns.pop(name)(sys.modules[name])
    pending = fns

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in pending:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module
            fn = pending.pop(name)

            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module)
                _fn(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _log(f"registered ({len(pending)} pending module hooks)")

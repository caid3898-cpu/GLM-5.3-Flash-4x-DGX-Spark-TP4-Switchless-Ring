# SPDX-License-Identifier: Apache-2.0
"""Context-lookup hybrid drafter for GLM-5.3-Flash + DFlash2, scheduler side (diagnostics/glm-lookup-20260928).

Tony v11 image, vLLM 0.1.dev20051+g487ecf187, V2 model runner, async scheduling, FULL_DECODE_ONLY graphs.
Default OFF. The worker half (draft write, DFlash2 forward skip) is overlay/glm_lookup_hooks.py.

  GLM_LOOKUP_DRAFT=0        off (default): nothing is imported or patched
  GLM_LOOKUP_DRAFT=shadow   index + counterfactual accounting only; no plan reaches the workers, drafts unchanged
  GLM_LOOKUP_DRAFT=1        plans are attached to the scheduler output; needs the worker hooks (same variable)
  --scheduler-cls glm_lookup_draft.LookupScheduler   (LeversScheduler + this; profile SCHEDULER_CLS=...)
  levers control file (SPEC_PROBE_CONTROL) keys "lookup": off|shadow|on and "lookup_skip": 0|1 switch it in-boot
  GLM_LOOKUP_SCHED_TRACE=/cache/lk.jsonl   one JSON line per resolved step (lookup vs DFlash2 on the same positions)

What it does, per decode request, every step (all in the single engine-core process, so every TP rank
receives the same plan; nothing here reads a device value or a rank-local clock):

  1. Keeps an index of the request's COMMITTED tokens (prompt + accepted output, i.e. request.all_token_ids;
     rejected drafts never reach that list). Incremental: each appended token inserts one n-gram key per
     n in GLM_LOOKUP_N (default 2..4) into a dict; the dict is frozen into sorted arrays every
     GLM_LOOKUP_HOT_TOKENS tokens (LSM tiers, merged geometrically), so memory is ~12 bytes per n-gram and
     lookups are a dict probe plus O(log) searches. Big prompt slices are indexed vectorised (numpy).
     Reset when the request's token list is shorter than the index or disagrees at a probe position.
  2. Looks up the last n committed tokens, longest n first, latest earlier occurrence wins; measures how far
     back the two sites agree ("agreement", capped at 64) and classes the match by it (<8, <32, >=32).
  3. Decides between the neural draft (DFlash2 at the policy k) and a lookup draft of K in the captured
     verify families (3/5/7), in two lookup flavours:
       skip      the DFlash2 forward is not run for this step (only its context-KV ingest), draft = lookup;
                 only when EVERY request in the batch is a skip row;
       override  DFlash2 runs as usual, then lookup rows get their ids replaced on device.
     Gate: expected tokens per step over step cost. Lookup acceptance per match class is measured on EVERY
     step (used or not) by replaying the lookup against the committed tokens that followed (shadow), so the
     lookup never needs exploration; neural acceptance comes only from steps that used DFlash2 ids.
  4. Hands the plan to the workers as scheduler_output.glm_lookup:
       {"v": 1, "seq": s, "skip": bool, "periodic": bool, "max_delta": int, "rows": {req_id: (mode, src, lhost, K)}}
     and sizes that request's placeholders to K (so the next step verifies K lookup tokens).

Async lag: the plan is made from committed tokens up to length lhost; when the workers draft, the device
history is longer (the in-flight steps). The worker checks, on device, that the tokens committed since
lhost equal the tokens after src (the copy continued), and then drafts tokens[src + delta + (j mod P)],
delta = total_len - lhost, P = lhost - src. The same check is replayed here on the committed tokens once
they are known, which attributes each step to lookup / neural / junk (skip with a failed check) exactly.

Lossless: drafts are greedy (draft_sample_method "greedy", no draft_logits), so a lookup draft goes through
the same rejection path as a DFlash2 draft: temperature 0 commits the target argmax chain, sampled requests
keep the target distribution. Accounting check (counted, never silent): on a lookup step the accepted count
must equal the longest common prefix of the lookup draft and the committed tokens, capped at K.

Credits (ideas; no code copied): STRML's prompt-lookup-in-the-MTP-round for ddalcu/mlx-serve PR #523 (MIT):
latest-occurrence index over committed tokens, agreement ("suffix") strength, separate lookup vs drafter
tokens-per-cost gate, lookup rounds kept out of the drafter's statistics. vLLM's n-gram proposer (prompt
lookup, longest n in [min, max]) and SuffixDecoding (Oliaro et al., Snowflake/CMU; Arctic Inference) for
the proposer shape; Prompt Lookup Decoding (Apoorv Saxena). Scheduler base: jnardiello's AdaptiveKScheduler
and our spec_probe / levers schedulers.
"""
from __future__ import annotations

import bisect
import logging
import os
import time
from array import array
from collections import deque
from dataclasses import dataclass, field

try:  # the image has numpy; the pure-Python fallback keeps the CPU tests runnable anywhere
    import numpy as np
except Exception:  # noqa: BLE001
    np = None
if os.environ.get("GLM_LOOKUP_NO_NUMPY", "0") == "1":
    np = None

MODE_NEURAL, MODE_SKIP, MODE_OVERRIDE = 0, 1, 2
MODE_NAMES = {MODE_NEURAL: "neural", MODE_SKIP: "skip", MODE_OVERRIDE: "override"}
SRC_NONE, SRC_NEURAL, SRC_LOOKUP, SRC_JUNK = "none", "neural", "lookup", "junk"
CLASS_EDGES = (8, 32)          # agreement classes: [min_agree, 8), [8, 32), [32, inf)
N_CLASSES = len(CLASS_EDGES) + 1
PAYLOAD_ATTR = "glm_lookup"
PAYLOAD_VERSION = 1

_P = 0x9E3779B97F4A7C15        # odd 64-bit multiplier (golden-ratio constant)
_M64 = (1 << 64) - 1
_OFF = ("", "0", "off", "false", "no")

logger = logging.getLogger("vllm.glm_lookup")


# ------------------------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------------------------
def _env(env, name, default):
    raw = env.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class LookupConfig:
    mode: str = "off"                 # off | shadow | on
    n_min: int = 2
    n_max: int = 4
    min_agree: int = 2                # agreement below this is not a match at all
    max_agree: int = 64
    max_batch: int = 1                # plans only when the step has <= this many requests (v1: c1)
    skip: bool = False                # allow skipping the DFlash2 forward (v1.1; opt-in until the state A/B passes)
    skip_streak: bool = True          # ... only right after a lookup step that landed all its drafts
    periodic: bool = True             # draft tokens[src + delta + (j mod P)] (repetition continues)
    max_delta: int = 64               # device check window (tokens committed since the plan)
    margin: float = 0.05              # lookup must beat neural by this factor (hysteresis both ways)
    alpha: float = 0.15               # EMA weight (adaptive-k uses 0.15; 0.3 as in mlx-serve lost ~1-2 % on prose in sim)
    min_obs: int = 2                  # shadow evaluations of a class before it may be used
    force: str = ""                   # "" | neural | skip | override | alt-skip | alt-override  (TEST ONLY)
    hot_tokens: int = 4096
    bulk_min: int = 2048
    ingest_budget: int = 32768        # tokens indexed per schedule() call, all requests together
    cost: str = "v3:35.9,v7:45.8,draft:3.1"   # ms; 2026-09-28 accept_probe c1 (prose k3 39.0, code k7 48.9)
    learn_cost: bool = False
    log_every: int = 2000

    def __post_init__(self):
        if self.mode not in ("off", "shadow", "on"):
            raise ValueError(f"glm-lookup: mode {self.mode!r}")
        if not 1 <= self.n_min <= self.n_max <= 8:
            raise ValueError(f"glm-lookup: need 1 <= n_min <= n_max <= 8, got {self.n_min}..{self.n_max}")
        if self.force not in ("", "neural", "skip", "override", "alt-skip", "alt-override"):
            raise ValueError(f"glm-lookup: force {self.force!r}")
        if not 1 <= self.max_delta <= 256:
            raise ValueError("glm-lookup: max_delta must be in 1..256")
        if self.max_batch < 1 or self.hot_tokens < 16 or self.ingest_budget < 1:
            raise ValueError("glm-lookup: bad max_batch / hot_tokens / ingest_budget")

    @classmethod
    def from_env(cls, environ=None) -> "LookupConfig":
        env = os.environ if environ is None else environ
        raw = _env(env, "GLM_LOOKUP_DRAFT", "0").lower()
        mode = "off" if raw in _OFF else ("shadow" if raw == "shadow" else "on")
        n = _env(env, "GLM_LOOKUP_N", "2:4").replace("..", ":").split(":")
        return cls(
            mode=mode,
            n_min=int(n[0]), n_max=int(n[-1]),
            min_agree=int(_env(env, "GLM_LOOKUP_MIN_AGREE", "2")),
            max_batch=int(_env(env, "GLM_LOOKUP_MAX_BATCH", "1")),
            skip=_env(env, "GLM_LOOKUP_SKIP", "0").lower() not in _OFF,
            skip_streak=_env(env, "GLM_LOOKUP_SKIP_STREAK", "1").lower() not in _OFF,
            periodic=_env(env, "GLM_LOOKUP_PERIODIC", "1").lower() not in _OFF,
            max_delta=int(_env(env, "GLM_LOOKUP_MAX_DELTA", "64")),
            margin=float(_env(env, "GLM_LOOKUP_MARGIN", "0.05")),
            alpha=float(_env(env, "GLM_LOOKUP_ALPHA", "0.15")),
            min_obs=int(_env(env, "GLM_LOOKUP_MIN_OBS", "2")),
            force=_env(env, "GLM_LOOKUP_FORCE", "").lower(),
            hot_tokens=int(_env(env, "GLM_LOOKUP_HOT_TOKENS", "4096")),
            ingest_budget=int(_env(env, "GLM_LOOKUP_INGEST_BUDGET", "32768")),
            cost=_env(env, "GLM_LOOKUP_COST", cls.cost),
            learn_cost=_env(env, "GLM_LOOKUP_COST_LEARN", "0").lower() not in _OFF,
            log_every=int(_env(env, "GLM_LOOKUP_LOG_EVERY", "2000")),
        )


# ------------------------------------------------------------------------------------------------------------
# n-gram keys: fold from the LAST token backwards, so the keys of the 2-, 3- and 4-gram ending at the same
# position come out of one pass. 64-bit, identical in the scalar and the numpy path. A collision can only
# make a lookup miss (every candidate is verified against the tokens), never propose a wrong match silently.
# ------------------------------------------------------------------------------------------------------------
def gram_key(gram) -> int:
    h = 0
    for t in reversed(gram):
        h = (h * _P + int(t) + 1) & _M64
    return h


class _Tier:
    """Sorted unique n-gram keys -> latest end position (the index of the first continuation token)."""

    __slots__ = ("keys", "pos")

    def __init__(self, keys, pos):
        self.keys = keys
        self.pos = pos

    def __len__(self):
        return len(self.keys)

    @classmethod
    def from_pairs(cls, keys, pos) -> "_Tier":
        if np is not None:
            k = np.asarray(keys, dtype=np.uint64)
            p = np.asarray(pos, dtype=np.int64)
            order = np.lexsort((p, k))
            k, p = k[order], p[order]
            last = np.ones(len(k), dtype=bool)
            if len(k) > 1:
                last[:-1] = k[1:] != k[:-1]
            return cls(k[last], p[last])
        best: dict = {}
        for kk, pp in zip(keys, pos):
            if pp > best.get(kk, -1):
                best[kk] = pp
        items = sorted(best.items())
        return cls(array("Q", (k for k, _ in items)), array("q", (p for _, p in items)))

    @classmethod
    def from_dict(cls, d: dict) -> "_Tier":
        return cls.from_pairs(list(d.keys()), list(d.values()))

    def merged(self, other: "_Tier") -> "_Tier":
        if np is not None:
            return _Tier.from_pairs(np.concatenate([self.keys, other.keys]),
                                    np.concatenate([self.pos, other.pos]))
        return _Tier.from_pairs(list(self.keys) + list(other.keys), list(self.pos) + list(other.pos))

    def get(self, key: int):
        keys = self.keys
        if np is not None and isinstance(keys, np.ndarray):
            i = int(np.searchsorted(keys, np.uint64(key)))
            if i < len(keys) and int(keys[i]) == key:
                return int(self.pos[i])
            return None
        i = bisect.bisect_left(keys, key)
        if i < len(keys) and keys[i] == key:
            return self.pos[i]
        return None


@dataclass(frozen=True)
class Match:
    src: int        # index of the first continuation token (tokens[src-n:src] == last n tokens)
    n: int          # matched n-gram length
    agree: int      # how far back the two sites agree (>= n, capped)
    avail: int      # length - src: distinct continuation tokens before the current end (the period P)
    length: int     # committed length when matched (lhost)


def agreement_class(agree: int) -> int:
    for i, edge in enumerate(CLASS_EDGES):
        if agree < edge:
            return i
    return len(CLASS_EDGES)


class ContextIndex:
    """Per-request index of committed tokens. Append-only; see the module docstring."""

    def __init__(self, ns=(2, 3, 4), hot_tokens: int = 4096, bulk_min: int = 2048):
        self.ns = tuple(sorted(set(int(n) for n in ns)))
        self.nmax = self.ns[-1]
        self.hot_limit = int(hot_tokens)
        self.bulk_min = int(bulk_min)
        self.reset()

    # -- state ------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.tokens = array("i")
        self.hot = {n: {} for n in self.ns}
        self.hot_count = 0
        self.tiers = {n: [] for n in self.ns}
        self.resets = getattr(self, "resets", -1) + 1

    def __len__(self) -> int:
        return len(self.tokens)

    def memory_bytes(self) -> int:
        tier = sum(len(t) * 16 for ts in self.tiers.values() for t in ts)
        hot = sum(len(d) for d in self.hot.values()) * 100   # rough CPython dict + int cost
        return len(self.tokens) * 4 + tier + hot

    # -- ingest ------------------------------------------------------------------------------------------------
    def extend(self, new) -> None:
        if len(new) == 0:
            return
        if np is not None and len(new) >= self.bulk_min:
            self._freeze()
            self._bulk(new)
            return
        for t in new:
            self._push(int(t))

    def _push(self, t: int) -> None:
        toks = self.tokens
        i = len(toks)
        toks.append(t)
        h = 0
        hot = self.hot
        for d in range(1, self.nmax + 1):
            j = i - d
            if j < 0:
                break
            h = (h * _P + toks[j] + 1) & _M64
            tbl = hot.get(d)
            if tbl is not None:
                tbl[h] = i
        self.hot_count += 1
        if self.hot_count >= self.hot_limit:
            self._freeze()

    def _bulk(self, new) -> None:
        i0 = len(self.tokens)
        self.tokens.extend(new)
        i1 = len(self.tokens)
        tok = np.frombuffer(self.tokens, dtype=np.int32)
        ends = np.arange(i0, i1, dtype=np.int64)
        h = np.zeros(i1 - i0, dtype=np.uint64)
        p = np.uint64(_P)
        for d in range(1, self.nmax + 1):
            idx = ends - d
            t = tok[np.maximum(idx, 0)].astype(np.uint64) + np.uint64(1)
            h = h * p + t
            if d in self.hot:
                keep = idx >= 0
                if keep.any():
                    self._push_tier(d, _Tier.from_pairs(h[keep], ends[keep]))
        del tok

    def _freeze(self) -> None:
        for n in self.ns:
            if self.hot[n]:
                self._push_tier(n, _Tier.from_dict(self.hot[n]))
                self.hot[n] = {}
        self.hot_count = 0

    def _push_tier(self, n: int, tier: _Tier) -> None:
        ts = self.tiers[n]
        ts.append(tier)
        while len(ts) >= 2 and len(ts[-2]) <= 2 * len(ts[-1]):
            b = ts.pop()
            a = ts.pop()
            ts.append(a.merged(b))

    def sync(self, src, budget: int) -> tuple[bool, int]:
        """Catch up with the committed token list `src` (prompt + output). Returns (caught_up, tokens_added)."""
        n_src = len(src)
        n = len(self.tokens)
        if n > n_src or (n > 0 and (src[n - 1] != self.tokens[n - 1] or src[0] != self.tokens[0]
                                    or src[n // 2] != self.tokens[n // 2])):
            self.reset()   # request reuse, truncation or a rewritten history: start over
            n = 0
        take = min(n_src - n, max(0, int(budget)))
        if take > 0:
            self.extend(src[n:n + take])
        return len(self.tokens) == n_src, take

    # -- query -------------------------------------------------------------------------------------------------
    def get(self, n: int, key: int):
        e = self.hot[n].get(key)
        if e is not None:
            return e
        for tier in reversed(self.tiers[n]):   # newer tiers hold larger positions
            e = tier.get(key)
            if e is not None:
                return e
        return None

    def match(self, n_min: int, n_max: int, max_agree: int = 64) -> Match | None:
        toks = self.tokens
        length = len(toks)
        for n in range(min(n_max, self.nmax), n_min - 1, -1):
            if n not in self.hot or length < n + 1:
                continue
            e = self.get(n, gram_key(toks[length - n:length]))
            if e is None or e >= length:
                continue
            if toks[e - n:e] != toks[length - n:length]:   # 64-bit collision: treat as no match at this n
                continue
            agree = n
            while agree < max_agree and e - 1 - agree >= 0 and toks[e - 1 - agree] == toks[length - 1 - agree]:
                agree += 1
            return Match(src=e, n=n, agree=agree, avail=length - e, length=length)
        return None


# ------------------------------------------------------------------------------------------------------------
# device logic, replayed on the host (must stay identical to glm_lookup_hooks.lookup_draft_reference)
# ------------------------------------------------------------------------------------------------------------
def replay_check(tokens, l_dev: int, src: int, lhost: int, max_delta: int) -> bool:
    """The worker's validity check: the tokens committed since the plan continued the copy."""
    delta = l_dev - lhost
    if delta < 0 or delta > max_delta or src < 0 or src >= lhost or lhost > len(tokens) or l_dev > len(tokens):
        return False
    for i in range(delta):
        if tokens[lhost + i] != tokens[src + i]:
            return False
    return True


def lookup_draft(tokens, l_dev: int, src: int, lhost: int, k: int, periodic: bool = True):
    """Draft the worker writes for a valid row (None where it keeps the previous id: non-periodic tail)."""
    delta = l_dev - lhost
    period = lhost - src
    out = []
    for j in range(k):
        off = (j % period) if periodic else j
        pos = src + delta + off
        out.append(tokens[pos] if pos < l_dev else None)
    return out


def common_prefix(draft, tokens, start: int) -> int:
    """Leading draft ids equal to tokens[start:]; stops at the first mismatch, a hole or the end."""
    n = 0
    for j, t in enumerate(draft):
        if t is None or start + j >= len(tokens) or tokens[start + j] != t:
            break
        n += 1
    return n


# ------------------------------------------------------------------------------------------------------------
# cost model (ms per step at c1)
# ------------------------------------------------------------------------------------------------------------
class CostModel:
    """Step ms: verify(k) interpolated from the table, + draft when DFlash2 runs. Optionally learned."""

    def __init__(self, spec: str, learn: bool = False, alpha: float = 0.1, min_samples: int = 8):
        pts, draft = {}, 3.1
        for part in spec.split(","):
            name, _, val = part.strip().partition(":")
            if name == "draft":
                draft = float(val)
            elif name.startswith("v"):
                pts[int(name[1:])] = float(val)
        if len(pts) < 1:
            raise ValueError(f"glm-lookup: cost table {spec!r} needs at least one vK:ms point")
        self.points = dict(sorted(pts.items()))
        self.draft = draft
        self.learn = learn
        self.alpha = alpha
        self.min_samples = min_samples
        self.learned: dict = {}   # (k, drafted) -> [ms, n]
        self._seen: set = set()

    def verify(self, k: int) -> float:
        ks = list(self.points)
        if len(ks) == 1:
            return self.points[ks[0]]
        if k <= ks[0]:
            lo, hi = ks[0], ks[1]
        elif k >= ks[-1]:
            lo, hi = ks[-2], ks[-1]
        else:
            hi = next(x for x in ks if x >= k)
            lo = ks[ks.index(hi) - 1]
            if hi == k:
                return self.points[k]
        slope = (self.points[hi] - self.points[lo]) / (hi - lo)
        return self.points[lo] + slope * (k - lo)

    def step(self, k: int, drafted: bool) -> float:
        cell = self.learned.get((k, drafted))
        if self.learn and cell is not None and cell[1] >= self.min_samples:
            return cell[0]
        return self.verify(k) + (self.draft if drafted else 0.0)

    def observe(self, k: int, drafted: bool, ms: float) -> None:
        """Fold one c1 step period. The first sample per shape is dropped (compile / first replay)."""
        key = (k, drafted)
        if key not in self._seen:
            self._seen.add(key)
            return
        if not (ms > 0.0) or ms > 10 * self.step(k, drafted):
            return
        cell = self.learned.setdefault(key, [ms, 0])
        if cell[1] >= self.min_samples and ms > 2.5 * cell[0]:
            return   # a stall (prefill of another request, GC), not a price
        cell[0] = ms if cell[1] == 0 else self.alpha * ms + (1 - self.alpha) * cell[0]
        cell[1] += 1


# ------------------------------------------------------------------------------------------------------------
# per-request statistics and the gate
# ------------------------------------------------------------------------------------------------------------
NEURAL_PRIOR = (0.62, 0.40, 0.28, 0.20, 0.15, 0.12, 0.10)   # P(accepted >= j): ~2.3 tokens/step at k=3


@dataclass
class Shadow:
    src: int
    lhost: int
    cls: int
    agree: int
    n: int


@dataclass
class PlanRec:
    mode: int
    k: int
    shadow: Shadow | None


@dataclass
class Pending:
    p: int                  # committed length when the drafts were made (device total_len)
    shadow: Shadow
    ok: bool
    source: str
    k: int
    accepted: int


@dataclass
class ReqState:
    idx: ContextIndex
    kmax: int
    p_lookup: list = field(default_factory=list)   # [class][j] P(lcp >= j+1 | check ok)
    f_lookup: list = field(default_factory=list)   # [class] P(check failed)
    n_lookup: list = field(default_factory=list)   # [class] shadow evaluations
    p_neural: list = field(default_factory=list)   # [j] P(accepted >= j+1), neural steps only
    plans: dict = field(default_factory=dict)      # seq -> PlanRec
    pending: deque = field(default_factory=deque)
    cur: tuple | None = None                       # (seq, PlanRec, ok, source, pre_len) between classify/observe
    prev_lookup: bool = False
    last_shadow: Shadow | None = None              # the source followed at the previous plan (sticky pointer)
    streak: bool = False                           # the last lookup step landed every draft (skip needs it)

    def __post_init__(self):
        k = self.kmax
        self.p_lookup = [[0.0] * k for _ in range(N_CLASSES)]
        self.f_lookup = [0.0] * N_CLASSES
        self.n_lookup = [0] * N_CLASSES
        self.p_neural = [NEURAL_PRIOR[min(j, len(NEURAL_PRIOR) - 1)] for j in range(k)]


def expected_tokens(p, k: int) -> float:
    return 1.0 + sum(p[:max(0, k)])


def choose(st: ReqState, cls: int, k_policy: int, cands, cost: CostModel, cfg: LookupConfig,
           allow_skip: bool) -> tuple[int, int]:
    """(mode, K) maximising expected tokens per ms; neural at the policy k unless lookup clearly wins."""
    if cfg.force == "neural":
        return MODE_NEURAL, k_policy
    if cfg.force in ("skip", "override"):
        if cfg.force == "skip" and allow_skip:
            return MODE_SKIP, max(cands)
        return MODE_OVERRIDE, max(cands)
    k_policy = max(1, k_policy)
    e_neural = expected_tokens(st.p_neural, k_policy)
    rate_neural = e_neural / cost.step(k_policy, drafted=True)
    if st.n_lookup[cls] < cfg.min_obs:
        return MODE_NEURAL, k_policy
    pl, f = st.p_lookup[cls], st.f_lookup[cls]
    best = (0.0, MODE_NEURAL, k_policy)
    for k in cands:
        el = (1.0 - f) * expected_tokens(pl, k)
        if allow_skip:
            r = (el + f * 1.0) / cost.step(k, drafted=False)
            if r > best[0]:
                best = (r, MODE_SKIP, k)
        r = (el + f * expected_tokens(st.p_neural, k)) / cost.step(k, drafted=True)
        if r > best[0]:
            best = (r, MODE_OVERRIDE, k)
    rate_lookup, mode, k = best
    need = rate_neural * ((1.0 - cfg.margin) if st.prev_lookup else (1.0 + cfg.margin))
    if mode != MODE_NEURAL and rate_lookup > need:
        return mode, k
    return MODE_NEURAL, k_policy


def _ema_vec(vec, value: int, upto: int, alpha: float) -> None:
    for j in range(min(upto, len(vec))):
        vec[j] = alpha * (1.0 if value >= j + 1 else 0.0) + (1.0 - alpha) * vec[j]


@dataclass
class PlanEntry:
    rid: str
    tokens: object          # committed tokens (request.all_token_ids or any sequence)
    eligible: bool          # decode row this schedule, not structured output, placeholders assigned
    k_policy: int           # placeholder length the adaptive-k policy chose
    fixed_k: bool = False   # the request pinned its verify length (vllm_xargs.spec_k): lookup keeps it


@dataclass
class StepPlan:
    seq: int
    rows: dict              # rid -> (mode, src, lhost, K)
    skip: bool
    k_override: dict        # rid -> K (placeholder length to set)

    def payload(self, cfg: LookupConfig) -> dict:
        return {"v": PAYLOAD_VERSION, "seq": self.seq, "skip": self.skip, "periodic": cfg.periodic,
                "max_delta": cfg.max_delta, "rows": dict(self.rows)}


class LookupPlanner:
    """Everything the scheduler hook needs, free of vLLM (the CPU tests and the simulator drive it directly)."""

    def __init__(self, cfg: LookupConfig, engine_k: int, families=(3, 5, 7)):
        self.cfg = cfg
        self.engine_k = int(engine_k)
        fam = sorted({int(k) for k in families if 1 <= int(k) <= self.engine_k})
        self.families = fam or [self.engine_k]
        self.cost = CostModel(cfg.cost, learn=cfg.learn_cost)
        self.states: dict[str, ReqState] = {}
        self.c = {"plans": 0, "rows_skip": 0, "rows_override": 0, "steps_skip": 0, "no_match": 0,
                  "not_indexed": 0, "sticky": 0, "evals": 0, "check_fail": 0, "acct_ok": 0, "acct_mismatch": 0,
                  "resets": 0, "ingested": 0}
        self.by_source = {s: [0, 0] for s in (SRC_NEURAL, SRC_LOOKUP, SRC_JUNK)}   # steps, tokens
        self.mismatch_log: list = []
        self.trace = None   # callable(dict) per resolved evaluation (GLM_LOOKUP_SCHED_TRACE)

    # -- state -------------------------------------------------------------------------------------------------
    def state(self, rid: str) -> ReqState:
        st = self.states.get(rid)
        if st is None:
            ns = range(self.cfg.n_min, self.cfg.n_max + 1)
            st = ReqState(ContextIndex(ns, self.cfg.hot_tokens, self.cfg.bulk_min), self.engine_k)
            self.states[rid] = st
        return st

    def evict(self, rid: str) -> None:
        self.states.pop(rid, None)

    # -- planning ----------------------------------------------------------------------------------------------
    def plan(self, seq: int, entries, n_batch: int) -> StepPlan:
        cfg = self.cfg
        rows, k_over = {}, {}
        can_plan = cfg.mode == "on" and n_batch <= cfg.max_batch
        budget = cfg.ingest_budget
        for e in entries:
            st = self.state(e.rid)
            before = st.idx.resets
            caught, used = st.idx.sync(e.tokens, budget)
            budget -= used
            self.c["ingested"] += used
            self.c["resets"] += st.idx.resets - before
            shadow = None
            if caught:
                m = st.idx.match(cfg.n_min, cfg.n_max, cfg.max_agree)
                if m is not None and m.agree >= cfg.min_agree:
                    shadow = Shadow(m.src, m.length, agreement_class(m.agree), m.agree, m.n)
                sticky = self._sticky(st)
                if sticky is not None and (shadow is None or sticky.agree > shadow.agree):
                    shadow = sticky
                    self.c["sticky"] += 1
                if shadow is None:
                    self.c["no_match"] += 1
                st.last_shadow = shadow
            else:
                self.c["not_indexed"] += 1
            mode, k = MODE_NEURAL, int(e.k_policy)
            if cfg.force.startswith("alt-") and can_plan and e.eligible and e.k_policy > 0:
                # TEST ONLY (DFlash2 state A/B): every even step is a lookup row at the policy k, odd steps are
                # neural. With GLM_LOOKUP_FILL=1 on the workers both arms draft the same ids on even steps.
                if seq % 2 == 0:
                    mode = MODE_SKIP if (cfg.force == "alt-skip" and n_batch == len(entries)) else MODE_OVERRIDE
                    src, lh = (shadow.src, shadow.lhost) if shadow else (max(len(st.idx) - 1, 0), len(st.idx))
                    st.plans[seq] = PlanRec(mode, k, shadow)
                    rows[e.rid] = (mode, src, lh, k)
                    continue
                st.plans[seq] = PlanRec(mode, k, shadow)
                continue
            if can_plan and e.eligible and shadow is not None and e.k_policy > 0:
                cands = (self.families if n_batch == 1 and not e.fixed_k
                         else [k for k in self.families if k == e.k_policy] or [int(e.k_policy)])
                allow_skip = cfg.skip and n_batch == len(entries) and (st.streak or not cfg.skip_streak
                                                                       or cfg.force == "skip")
                mode, k = choose(st, shadow.cls, int(e.k_policy), cands, self.cost, cfg, allow_skip)
            st.plans[seq] = PlanRec(mode, k, shadow)
            if len(st.plans) > 8:   # plans whose output never came (preemption, stale outputs)
                for s in sorted(st.plans)[:-8]:
                    del st.plans[s]
            if mode != MODE_NEURAL:
                rows[e.rid] = (mode, shadow.src, shadow.lhost, k)
                if k != e.k_policy:
                    k_over[e.rid] = k
        skip = bool(rows) and len(rows) == n_batch and all(r[0] == MODE_SKIP for r in rows.values())
        if not skip:   # the forward runs for the batch: every skip row becomes an override row
            for rid, r in list(rows.items()):
                if r[0] == MODE_SKIP:
                    rows[rid] = (MODE_OVERRIDE,) + r[1:]
                    self.states[rid].plans[seq].mode = MODE_OVERRIDE
        for rid, r in rows.items():
            self.c["rows_skip" if r[0] == MODE_SKIP else "rows_override"] += 1
        if skip:
            self.c["steps_skip"] += 1
        self.c["plans"] += 1
        for e in entries:
            self.states[e.rid].prev_lookup = e.rid in rows
        return StepPlan(seq, rows, skip, k_over)

    def _sticky(self, st: ReqState) -> Shadow | None:
        """Keep following the previous source while the copy continues. The index returns the LATEST earlier
        occurrence of the suffix, which inside a long echo is often a repeat of a common n-gram in the part
        already echoed (short agreement) rather than the source being copied; the previous source advanced by
        the tokens committed since agrees back at least as far as it did, plus those tokens."""
        s0 = st.last_shadow
        if s0 is None:
            return None
        toks = st.idx.tokens
        L = len(toks)
        d = L - s0.lhost
        if d < 0 or d > 4 * self.cfg.max_delta or toks[s0.lhost:L] != toks[s0.src:s0.src + d]:
            return None
        agree = min(self.cfg.max_agree, s0.agree + d)
        return Shadow(s0.src + d, L, agreement_class(agree), agree, min(self.cfg.n_max, agree))

    # -- attribution -------------------------------------------------------------------------------------------
    def classify(self, seq: int, rid: str, pre_len: int) -> str:
        """Which draft the step `seq` verified for `rid` (the plan of rid's previous schedule). Call before
        the new tokens are appended; pre_len = committed length before them (= device total_len at drafting)."""
        st = self.states.get(rid)
        if st is None:
            return SRC_NONE
        older = [s for s in st.plans if s < seq]
        if not older:
            st.cur = None
            return SRC_NONE
        rec = st.plans[max(older)]
        for s in older:
            del st.plans[s]
        ok = False
        if rec.shadow is not None and len(st.idx) >= pre_len:
            ok = replay_check(st.idx.tokens, pre_len, rec.shadow.src, rec.shadow.lhost, self.cfg.max_delta)
        if rec.mode != MODE_NEURAL and ok:
            source = SRC_LOOKUP
        elif rec.mode == MODE_SKIP:
            source = SRC_JUNK
        else:
            source = SRC_NEURAL
        st.cur = (seq, rec, ok, source, pre_len)
        return source

    def observe(self, seq: int, rid: str, k_sched: int, accepted: int, tokens_after) -> None:
        st = self.states.get(rid)
        if st is None or st.cur is None or st.cur[0] != seq:
            return
        _, rec, ok, source, pre_len = st.cur
        st.cur = None
        caught, used = st.idx.sync(tokens_after, self.cfg.ingest_budget)
        self.c["ingested"] += used
        if k_sched <= 0:
            return
        self.by_source[source][0] += 1
        self.by_source[source][1] += accepted + 1
        if source == SRC_NEURAL:
            _ema_vec(st.p_neural, accepted, k_sched, self.cfg.alpha)
        if source in (SRC_LOOKUP, SRC_JUNK):
            st.streak = source == SRC_LOOKUP and accepted >= k_sched
        if rec.shadow is not None:
            st.pending.append(Pending(pre_len, rec.shadow, ok, source, k_sched, accepted))
        self._resolve(st, final=False)

    def _resolve(self, st: ReqState, final: bool) -> None:
        toks = st.idx.tokens
        a = self.cfg.alpha
        while st.pending:
            pd = st.pending[0]
            if not final and len(toks) < pd.p + self.engine_k:
                break
            st.pending.popleft()
            if len(toks) < pd.p:
                continue   # history rewritten under us (reset); drop
            cls = pd.shadow.cls
            self.c["evals"] += 1
            st.f_lookup[cls] = a * (0.0 if pd.ok else 1.0) + (1 - a) * st.f_lookup[cls]
            if not pd.ok:
                self.c["check_fail"] += 1
                st.n_lookup[cls] += 1
                if self.trace is not None:
                    self._emit(pd, None)
                continue
            draft = lookup_draft(toks, pd.p, pd.shadow.src, pd.shadow.lhost, self.engine_k, self.cfg.periodic)
            lcp = common_prefix(draft, toks, pd.p)
            if not final or len(toks) >= pd.p + self.engine_k:
                _ema_vec(st.p_lookup[cls], lcp, self.engine_k, a)
                st.n_lookup[cls] += 1
            if pd.source == SRC_LOOKUP:
                self._account(pd, draft, lcp)
            if self.trace is not None:
                self._emit(pd, lcp)

    def _emit(self, pd: Pending, lcp) -> None:
        """One record per resolved step: what was verified (source, k, accepted) and what the lookup would have
        landed (lcp, uncensored up to engine_k; None = the in-flight check failed). On neural steps this pairs
        DFlash2's acceptance with the lookup's on the same positions: the number the gate is built on."""
        sh = pd.shadow
        self.trace({"p": pd.p, "cls": sh.cls, "agree": sh.agree, "n": sh.n, "dist": sh.lhost - sh.src,
                    "ok": pd.ok, "lcp": lcp, "source": pd.source, "k": pd.k, "accepted": pd.accepted})

    def _account(self, pd: Pending, draft, lcp: int) -> None:
        k = pd.k
        holes = next((j for j, t in enumerate(draft) if t is None), len(draft))
        firm = min(k, holes)
        if lcp < firm:
            ok = pd.accepted == lcp
        else:   # past a hole the worker kept an old id, which may match by chance
            ok = firm <= pd.accepted <= k
        if ok:
            self.c["acct_ok"] += 1
        else:
            self.c["acct_mismatch"] += 1
            if len(self.mismatch_log) < 32:
                self.mismatch_log.append((pd.p, pd.shadow.src, pd.shadow.lhost, k, pd.accepted, lcp))

    def finish(self, rid: str) -> None:
        st = self.states.get(rid)
        if st is not None:
            self._resolve(st, final=True)
        self.evict(rid)

    def summary(self) -> str:
        bs = " ".join(f"{s}={v[0]}/{v[1]}" for s, v in self.by_source.items())
        c = self.c
        return (f"plans={c['plans']} skip_steps={c['steps_skip']} rows skip/override={c['rows_skip']}/"
                f"{c['rows_override']} steps/tokens {bs} evals={c['evals']} check_fail={c['check_fail']} "
                f"acct ok/mismatch={c['acct_ok']}/{c['acct_mismatch']} no_match={c['no_match']} "
                f"not_indexed={c['not_indexed']} ingested={c['ingested']} resets={c['resets']} "
                f"tracked={len(self.states)}")


def families_from_table(table, engine_k: int):
    """k values of num_speculative_tokens_per_batch_size ([[min_bs, max_bs, k], ...]); graphs exist for each."""
    ks = set()
    for entry in table or ():
        try:
            ks.add(int(entry[-1]))
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(k for k in ks if 1 <= k <= engine_k) or [engine_k]


# ------------------------------------------------------------------------------------------------------------
# vLLM scheduler subclass (only inside the image)
# ------------------------------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised only inside the vLLM image
    from vllm.logger import init_logger as _init_logger
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _AsyncCheck  # noqa: F401

    logger = _init_logger("vllm.glm_lookup")
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001
    _HAVE_VLLM = False


if _HAVE_VLLM:  # pragma: no cover
    from types import SimpleNamespace

    from glm_levers_sched import LeversScheduler

    try:
        from spec_probe_scheduler import request_override as _request_override
    except Exception:  # noqa: BLE001
        def _request_override(request):
            return None

    class LookupScheduler(LeversScheduler):  # type: ignore[misc]
        """LeversScheduler + context-lookup plans. Inert (stock LeversScheduler) with GLM_LOOKUP_DRAFT=0."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._lk_failed = False
            self._lk_seq = 0
            self._lk_last_t = None
            self._lk_exclude: set = set()
            cfg = LookupConfig.from_env()
            spec = getattr(self.vllm_config, "speculative_config", None)
            draft_method = getattr(spec, "draft_sample_method", "greedy")
            if cfg.mode == "on" and draft_method != "greedy":
                logger.error("glm-lookup: draft_sample_method=%s (lookup drafts need greedy drafts: the "
                             "rejection sampler would read DFlash2's q for a lookup token); shadow only", draft_method)
                cfg = LookupConfig(**{**cfg.__dict__, "mode": "shadow"})
            engine_k = int(getattr(self, "_ak_engine_k", 0) or getattr(self, "num_spec_tokens", 0) or 0)
            fams = families_from_table(getattr(spec, "num_speculative_tokens_per_batch_size", None), engine_k)
            self._lk_cfg = cfg
            self._lk_on_ok = cfg.mode == "on"          # worker hooks installed and greedy drafts
            self._lk_engine = (max(engine_k, 1), fams)
            self._lk = LookupPlanner(cfg, max(engine_k, 1), fams) if cfg.mode != "off" and engine_k > 0 else None
            self._lk_trace_f = None
            path = os.environ.get("GLM_LOOKUP_SCHED_TRACE", "").strip()
            if path and self._lk is not None:
                self._lk_trace_f = open(path, "a", buffering=1 << 16)
                import json as _json
                self._lk.trace = lambda rec: self._lk_trace_f.write(_json.dumps(rec) + "\n")
            logger.info("glm-lookup: LookupScheduler mode=%s engine_k=%s families=%s cfg=%s", cfg.mode, engine_k,
                        fams, cfg)
            pending = getattr(self, "_lk_pending_ctl", None)
            if pending:
                self._lk_apply(pending)

        # -- live switch through the levers control file (in-boot A/B, scheduler only) -------------------------
        def _lv_apply(self, ctl: dict) -> None:
            super()._lv_apply(ctl)
            if "lookup" in ctl or "lookup_skip" in ctl:
                if not hasattr(self, "_lk_cfg"):   # called from LeversScheduler.__init__ (GLM_LV_POLICY)
                    self._lk_pending_ctl = ctl
                else:
                    self._lk_apply(ctl)

        def _lk_apply(self, ctl: dict) -> None:
            """{"lookup": "off"|"shadow"|"on", "lookup_skip": 0|1}. "on" needs GLM_LOOKUP_DRAFT on at boot (the
            worker hooks); otherwise it is downgraded to shadow, because a plan nobody executes would still
            resize the placeholders."""
            from dataclasses import replace as _replace
            mode = str(ctl.get("lookup", self._lk_cfg.mode))
            if mode not in ("off", "shadow", "on"):
                logger.error("glm-lookup: control key lookup=%r ignored", mode)
                return
            if mode == "on" and not self._lk_on_ok:
                logger.error("glm-lookup: lookup=on needs GLM_LOOKUP_DRAFT=1 at boot (worker hooks) and greedy "
                             "drafts; using shadow")
                mode = "shadow"
            skip = bool(int(ctl.get("lookup_skip", int(self._lk_cfg.skip))))
            self._lk_cfg = _replace(self._lk_cfg, mode=mode, skip=skip)
            if self._lk is None and mode != "off":
                self._lk = LookupPlanner(self._lk_cfg, *self._lk_engine)
            if self._lk is not None:
                self._lk.cfg = self._lk_cfg
                if mode == "off":
                    self._lk = None
            logger.info("glm-lookup: now mode=%s skip=%s", mode, skip)

        # -- plan (runs after the adaptive-k policy sized the placeholders) ------------------------------------
        def _update_after_schedule(self, scheduler_output) -> None:
            super()._update_after_schedule(scheduler_output)
            self._lk_seq += 1
            setattr(scheduler_output, "_glm_lk_seq", self._lk_seq)
            if self._lk is None or self._lk_failed:
                return
            try:
                self._lk_plan(scheduler_output)
            except Exception:  # noqa: BLE001 - never let the drafter kill the engine core
                self._lk_failed = True
                logger.exception("glm-lookup: planning failed; stock behaviour from now on")

        def _lk_plan(self, so) -> None:
            reqs = self.requests
            entries = []
            n_batch = len(so.num_scheduled_tokens)
            for rid in so.num_scheduled_tokens:
                r = reqs.get(rid)
                if r is None or r.is_finished() or getattr(r, "is_prefill_chunk", False) or not r.spec_token_ids:
                    continue
                eligible = not getattr(r, "use_structured_output", False)
                entries.append(PlanEntry(rid, r.all_token_ids, eligible, len(r.spec_token_ids),
                                         fixed_k=_request_override(r) is not None))
            if not entries:
                return
            plan = self._lk.plan(self._lk_seq, entries, n_batch)
            if self._lk_cfg.mode != "on" or not plan.rows:
                return
            for rid, k in plan.k_override.items():
                reqs[rid].spec_token_ids = self._ak_placeholders[min(k, self._ak_engine_k)]
            setattr(so, PAYLOAD_ATTR, plan.payload(self._lk_cfg))

        # -- attribution ---------------------------------------------------------------------------------------
        def update_from_output(self, scheduler_output, model_runner_output):
            info = []
            if self._lk is not None and not self._lk_failed:
                try:
                    info = self._lk_classify(scheduler_output, model_runner_output)
                except Exception:  # noqa: BLE001
                    self._lk_failed = True
                    logger.exception("glm-lookup: classify failed; stock behaviour from now on")
            out = super().update_from_output(scheduler_output, model_runner_output)
            self._lk_exclude = set()
            if info:
                try:
                    self._lk_observe(scheduler_output, info)
                except Exception:  # noqa: BLE001
                    self._lk_failed = True
                    logger.exception("glm-lookup: observe failed; stock behaviour from now on")
            return out

        def _lk_classify(self, so, mro):
            seq = getattr(so, "_glm_lk_seq", None)
            sched_spec = so.scheduled_spec_decode_tokens or {}
            if seq is None or not sched_spec:
                return []
            sampled = mro.sampled_token_ids
            index = mro.req_id_to_index
            info = []
            for rid, spec in sched_spec.items():
                r = self.requests.get(rid)
                i = index.get(rid)
                if r is None or r.is_finished() or i is None or getattr(r, "num_stale_output_tokens", 0) > 0:
                    continue
                gen = sampled[i] if sampled else []
                if not gen:
                    continue
                src = self._lk.classify(seq, rid, len(r.all_token_ids))
                if src in (SRC_LOOKUP, SRC_JUNK):
                    self._lk_exclude.add(rid)   # lookup steps never feed the adaptive-k EMA
                info.append((rid, r, len(spec), max(len(gen) - 1, 0), src))
            return info

        def _ak_observe(self, scheduler_output, model_runner_output) -> None:
            if not self._lk_exclude:
                return super()._ak_observe(scheduler_output, model_runner_output)
            kept = {rid: t for rid, t in scheduler_output.scheduled_spec_decode_tokens.items()
                    if rid not in self._lk_exclude}
            return super()._ak_observe(SimpleNamespace(scheduled_spec_decode_tokens=kept), model_runner_output)

        def _lk_observe(self, so, info) -> None:
            seq = so._glm_lk_seq
            now = time.monotonic()
            if self._lk_cfg.learn_cost and self._lk_last_t is not None and len(so.num_scheduled_tokens) == 1:
                # This output's period = verify of k drafts (planned one step earlier) + the drafting done at the
                # end of this step, which skipped DFlash2 only if this step's own payload said so.
                k = info[0][2]
                drafted = not (getattr(so, PAYLOAD_ATTR, None) or {}).get("skip", False)
                self._lk.cost.observe(k, drafted, (now - self._lk_last_t) * 1000.0)
            self._lk_last_t = now
            for rid, r, k, acc, src in info:
                if r.is_finished():
                    self._lk.finish(rid)
                    continue
                self._lk.observe(seq, rid, k, acc, r.all_token_ids)
            every = self._lk_cfg.log_every
            if every and seq % every == 0:
                logger.info("glm-lookup: %s", self._lk.summary())
                if self._lk.mismatch_log:
                    logger.error("glm-lookup: ACCOUNTING MISMATCH (p, src, lhost, k, accepted, lcp) %s",
                                 self._lk.mismatch_log[:8])

        def _free_request(self, request, delay_free_blocks: bool = False):
            if self._lk is not None:
                try:
                    self._lk.finish(request.request_id)
                except Exception:  # noqa: BLE001
                    pass
            return super()._free_request(request, delay_free_blocks)

else:

    class LookupScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("LookupScheduler needs vLLM")

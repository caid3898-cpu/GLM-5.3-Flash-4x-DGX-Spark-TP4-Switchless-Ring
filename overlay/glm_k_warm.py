# SPDX-License-Identifier: Apache-2.0
"""GLM_LV_WARM=<w0>: bias-corrected warm start for the adaptive-k EMA of glm_levers_sched.LvPolicy. Exact.

Speculation length never changes committed tokens (greedy output is the target's argmax; sampled output keeps the
target distribution), so this only moves which verify-row count a step pays for.

Problem (glm-steplevel-20260928, trace c1 prose): every new request starts at ema = seed = 1.0 (k_hi = 7) and,
with alpha 0.15 and down 0.42, needs ~6-12 observations (plus 1-2 steps of async observation lag) to fall to
k_lo = 3. The 09-28 c1 prose trace ran its first 14 decode steps at 8 rows (51.8 ms) before settling at 4 rows
(39.4 ms); the c4 trace ran its first 8 steps at 32 rows (97.5 ms) instead of 16 (78.3 ms). Prose gains ~0.2
tokens per step from positions 4-7 (accept_probe 09-28: unconditional acceptance 0.665/0.403/0.204/~0.01), which
does not pay for +12.4 ms (c1) / +19.2 ms (c4).

Fix: treat the seed as w0 pseudo-observations and average until the EMA's own memory takes over:
    alpha_eff(n) = max(alpha, 1 / (n + 1 + w0))      n = observations already folded in for this request
With w0 = 1 and seed 1.0 a prose request reaches 0.33 (< down) after two zero observations; a code/JSON request
(P(position 3) ~0.62-0.80) stays at k_hi. After n >= 1/alpha - 1 - w0 observations (~5 at alpha 0.15) the update
is exactly the deployed EMA, so steady-state behaviour is unchanged.

Knobs (read per observation):
  GLM_LV_WARM=0|<w0>   0/unset: stock observe (byte-identical path). <w0> > 0: prior weight of the seed (1 is the
                       intended value; 0.5 converges faster, 3 slower).
The same effect without code, for a first live probe: set "seed": 0.5 in the levers policy file (the EMA then
needs two zero observations to cross 0.42; convergence afterwards is the stock alpha).

Credits: jnardiello (AdaptiveKScheduler, the EMA policy this wraps); bias correction of an exponential moving
average is the standard warm-up used by Adam (Kingma & Ba, 2015).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

TARGET = "glm_levers_sched"
_OFF = ("", "0", "off", "false", "no")


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-k-warm: {msg}\n")


def prior_weight() -> float:
    raw = (os.environ.get("GLM_LV_WARM") or "").strip().lower()
    if raw in _OFF:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0 if raw in ("on", "true", "yes") else 0.0


def alpha_eff(alpha: float, n: int, w0: float) -> float:
    """Pure: effective update weight of observation n+1 (n already folded in)."""
    if w0 <= 0:
        return alpha
    return max(alpha, 1.0 / (n + 1.0 + w0))


def signal_of(policy, signal_mode: str, num_accepted: int, num_draft: int):
    """The value the stock observe would fold in, or None when it would not update (pure given the policy)."""
    if signal_mode == "cond":
        k = policy.cfg.k_lo
        if num_draft < k or num_accepted < k - 1:
            return None
        return 1.0 if num_accepted >= k else 0.0
    if num_draft <= 0:
        return None
    return policy.signal(num_accepted, num_draft)


def patch_policy(cls, mode_getter) -> None:
    if getattr(cls, "_glm_k_warm", False):
        return
    cls._glm_k_warm = True
    orig_observe, orig_evict = cls.observe, cls.evict

    def observe(self, req_id, num_accepted, num_draft):
        w0 = prior_weight()
        if w0 <= 0:
            return orig_observe(self, req_id, num_accepted, num_draft)
        st = self._get(req_id)
        x = signal_of(self, mode_getter(), num_accepted, num_draft)
        if x is None:
            return st.ema
        counts = self.__dict__.setdefault("_glm_warm_n", {})
        n = counts.get(req_id, 0)
        a = alpha_eff(self.cfg.alpha, n, w0)
        st.ema = a * x + (1.0 - a) * st.ema
        counts[req_id] = n + 1
        self.counters["observations"] += 1
        return st.ema

    def evict(self, req_id):
        self.__dict__.get("_glm_warm_n", {}).pop(req_id, None)
        return orig_evict(self, req_id)

    cls.observe, cls.evict = observe, evict


def _patch_module(mod) -> None:
    patch_policy(mod.LvPolicy, lambda: mod.LV.get("signal", "pos"))
    _log(f"LvPolicy.observe wrapped (GLM_LV_WARM={os.environ.get('GLM_LV_WARM')})")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            _patch_module(module)

        spec.loader.exec_module = exec_module
        return spec


def register() -> None:
    """Idempotent. No-op unless GLM_LV_WARM is a positive number (default off)."""
    if prior_weight() <= 0:
        return
    if TARGET in sys.modules:
        _patch_module(sys.modules[TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

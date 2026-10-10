#!/usr/bin/env python3
"""CPU tests for overlay/glm_gumbel_coupled.py (Gumbel-coupled drafting; llama.cpp-lab PR #26 by Jim Routh).

  uv run --no-project --with torch --with numpy python tests/test_glm_gumbel_coupled.py

1. noise: g(seed, pos, token) is deterministic, independent of the token id's integer width, different per
   position / seed, and Gumbel(0, 1) distributed (mean 0.5772, variance pi^2/6).
2. exactness, sequence level: a toy autoregressive target (logits a hash of the prefix, temperature 0.7, top-p 0.9 with
   vLLM's apply_top_k_top_p_pytorch rule) is decoded (a) without speculation, one gumbel draw per position, and
   (b) with coupled speculation under five drafters: coupled (the draft uses the SAME noise), independent-noise,
   argmax, adversarial (proposes the target's second choice), and -1 padding; with 1..5 drafts per step. Every
   sequence of (b) must equal (a) token for token, for every seed: the committed stream is a function of (seed,
   logits) only.
3. exactness, distribution level: the first two tokens over 20000 seeds vs the exact target probabilities (G-test).
4. acceptance: the coupled drafter accepts more than the argmax drafter at T > 0 and the same at T = 0.
5. the refused combination: a noise-sampled draft verified by the STOCK one-hot residual rule with the same noise
   is biased (G-test fails); the same rule with a deterministic (argmax) draft is exact.
6. compare(): the check-mode comparison flags every kind of corruption and passes a correct step.
7. walk_torch: T = 0 is the plain argmax walk; the draft nucleus always keeps the best candidate.
8. hooks (fake vLLM modules, torch gumbel_sample on the numpy Philox noise): RejectionSampler._verify is wrapped;
   mode 0 calls the stock rule; mode 1 returns the coupled result; check mode compares with Sampler.sample and counts
   0 mismatches, and counts a planted corruption; the glm_ab guard refuses shared drafter graphs.
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
import glm_gumbel_coupled as gc  # noqa: E402
import glm_target_argmax as ta  # noqa: E402

V = 12
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ---- noise ------------------------------------------------------------------------------------------
_NCACHE = {}


def noise(seed, pos, tokens):
    t = np.asarray(tokens, dtype=np.int64)
    key = (seed, pos)
    if key not in _NCACHE:
        _NCACHE[key] = gc.gumbel_noise_np(seed, pos, np.arange(V))
    return torch.from_numpy(_NCACHE[key][t].astype(np.float64))


def test_noise():
    a = gc.gumbel_noise_np(1234, 77, np.arange(64))
    b = gc.gumbel_noise_np(1234, 77, np.arange(64, dtype=np.int32).astype(np.int64))
    c = gc.gumbel_noise_np(1234, 78, np.arange(64))
    d = gc.gumbel_noise_np(-1234, 77, np.arange(64))
    check(np.array_equal(a, b), "noise: same values for int32 / int64 token ids")
    check(not np.array_equal(a, c) and not np.array_equal(a, d), "noise: differs per position and per seed")
    big = gc.gumbel_noise_np(987654321, 5, np.arange(200000)).astype(np.float64)
    check(abs(big.mean() - 0.5772) < 0.01 and abs(big.var() - math.pi ** 2 / 6) < 0.03,
          f"noise: Gumbel(0,1) moments (mean {big.mean():.4f}, var {big.var():.4f})")


# ---- toy target -------------------------------------------------------------------------------------
TEMP, TOP_P = 0.7, 0.9


def target_logits(prefix):
    h = 1469598103934665603
    for t in prefix:
        h = ((h ^ (t + 1)) * 1099511628211) & ((1 << 63) - 1)
    rng = np.random.default_rng(h % (2 ** 32))
    return torch.from_numpy(rng.normal(0, 1.6, V))


def processed(logits, temp=TEMP, top_p=TOP_P):
    """Sampler.apply_sampling_params (temperature) + apply_top_k_top_p_pytorch (top-p rule)."""
    x = logits.clone().double()
    if temp == 0.0:
        return x
    x = x / temp
    srt, idx = x.sort(descending=False)
    cs = torch.cumsum(srt.softmax(-1), -1)
    m = cs <= 1 - top_p
    m[-1] = False
    srt = srt.masked_fill(m, -float("inf"))
    return x.scatter(0, idx, srt)


def draw(prefix, seed, pos, temp=TEMP):
    p = processed(target_logits(prefix), temp)
    if temp == 0.0:
        return int(torch.argmax(p))
    return int(torch.argmax(p + noise(seed, pos, np.arange(V))))


def decode_plain(seed, prompt, n, temp=TEMP):
    seq = list(prompt)
    while len(seq) < len(prompt) + n:
        seq.append(draw(seq, seed, len(seq) - 1, temp))       # the row whose input sits at len-1 predicts len
    return seq


def drafter(kind, seq, seed, k, temp=TEMP, rng=None):
    """k draft tokens for positions len(seq) .. len(seq)+k-1 (a fake 'lattice': the target's logits + a bias)."""
    out = []
    ctx = list(seq)
    for j in range(k):
        pos_key = len(ctx) - 1                                   # key of the verify row for this draft
        q = target_logits(ctx) + torch.from_numpy(np.random.default_rng(len(ctx) * 7 + seed % 97).normal(0, 0.8, V))
        if kind == "coupled":
            t = int(torch.argmax(q / temp + noise(seed, pos_key, np.arange(V)))) if temp else int(torch.argmax(q))
        elif kind == "indep":
            t = int(torch.argmax(q / max(temp, 1e-6) + torch.from_numpy(rng.gumbel(size=V))))
        elif kind == "argmax":
            t = int(torch.argmax(q))
        elif kind == "adversarial":
            p = processed(target_logits(ctx), temp) + (noise(seed, pos_key, np.arange(V)) if temp else 0)
            t = int(torch.argsort(p, descending=True)[1])
        elif kind == "pad":
            t = -1
        else:
            raise ValueError(kind)
        out.append(t)
        ctx.append(max(t, 0))
    return out


def decode_coupled(seed, prompt, n, kind, kmax, temp=TEMP, rng=None, stats=None):
    """One request through the coupled verify: rows = [bonus] + drafts; t_i = draw at each row; greedy_verify."""
    seq = list(prompt)
    while len(seq) < len(prompt) + n:
        k = 1 + (len(seq) % kmax)
        d = drafter(kind, seq, seed, k, temp, rng)
        # verify rows: row i input = seq[-1] (i = 0) or d[i-1]; its logits need the prefix up to that input
        target = []
        ctx = list(seq)
        for i in range(k + 1):
            if i > 0:
                ctx = ctx + [max(d[i - 1], 0)]
            target.append(draw(ctx, seed, len(ctx) - 1, temp))
        draft_col = torch.tensor([seq[-1]] + d, dtype=torch.int64)       # input ids at the logits rows
        cu = torch.tensor([0, k + 1], dtype=torch.int64)
        sampled, ns = ta.greedy_verify_torch(torch.tensor(target, dtype=torch.int64), draft_col, cu, 1, kmax + 1)
        nsv = int(ns[0])
        res = gc.compare(torch.tensor([draw(seq + [max(x, 0) for x in d[:i]], seed, len(seq) - 1 + i, temp)
                                       for i in range(k + 1)], dtype=torch.int64),
                         torch.tensor(target, dtype=torch.int64), draft_col, cu, sampled, ns,
                         torch.tensor([temp > 0] * (k + 1)))
        if stats is not None:
            stats["bad"] += res["bad"]
            stats["acc"] += nsv - 1
            stats["drafts"] += k
        seq.extend(int(x) for x in sampled[0, :nsv])
    return seq[:len(prompt) + n]


def test_sequences():
    rng = np.random.default_rng(0)
    bad_seq, n_seq, bad_check = 0, 0, 0
    for seed in [1, 2, 3, -5, 2 ** 40 + 7, 123456789]:
        for prompt in ([3], [1, 4, 1, 5]):
            ref = decode_plain(seed, prompt, 24)
            for kind in ("coupled", "indep", "argmax", "adversarial", "pad"):
                for kmax in (1, 3, 5):
                    st = {"bad": 0, "acc": 0, "drafts": 0}
                    got = decode_coupled(seed, prompt, 24, kind, kmax, rng=rng, stats=st)
                    n_seq += 1
                    bad_seq += got != ref
                    bad_check += st["bad"]
    check(bad_seq == 0, f"sequences: coupled speculation == plain sampling for every drafter / k / seed "
                        f"({n_seq - bad_seq}/{n_seq} identical)")
    check(bad_check == 0, f"compare(): 0 mismatches across all steps of all runs ({bad_check})")


def exact_probs(prefix, temp=TEMP):
    return torch.softmax(processed(target_logits(prefix), temp), -1).numpy()


def gtest(counts, probs):
    counts = np.asarray(counts, float)
    probs = np.asarray(probs, float)
    n = counts.sum()
    m = probs > 0
    if (counts[~m] > 0).any():
        return float("inf"), 1
    e = n * probs[m]
    c = counts[m]
    g = 2 * np.sum(np.where(c > 0, c * np.log(np.maximum(c, 1e-300) / e), 0.0))
    return g, int(m.sum()) - 1


def chi2_crit(dof):          # 99.9 % quantile, Wilson-Hilferty
    z = 3.09
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def test_distribution():
    prompt = [2, 7]
    N = 20000
    joint = {}
    for seed in range(N):
        s = decode_coupled(seed, prompt, 2, "coupled", 3)
        joint[(s[2], s[3])] = joint.get((s[2], s[3]), 0) + 1
    p1 = exact_probs(prompt)
    probs, counts = [], []
    for a in range(V):
        if p1[a] == 0:
            continue
        p2 = exact_probs(prompt + [a])
        for b in range(V):
            if p2[b] > 0:
                probs.append(p1[a] * p2[b])
                counts.append(joint.get((a, b), 0))
    extra = sum(v for (a, b), v in joint.items() if p1[a] == 0 or exact_probs(prompt + [a])[b] == 0)
    g, dof = gtest(counts, probs)
    check(extra == 0 and g < chi2_crit(dof),
          f"distribution: first two tokens over {N} seeds match the target (G {g:.1f} < {chi2_crit(dof):.1f}, dof {dof}, "
          f"out-of-nucleus draws {extra})")


def test_acceptance():
    for temp in (TEMP, 0.0):
        acc = {}
        for kind in ("coupled", "argmax"):
            st = {"bad": 0, "acc": 0, "drafts": 0}
            for seed in range(60):
                decode_coupled(seed, [5], 40, kind, 5, temp=temp, stats=st)
            acc[kind] = st["acc"] / st["drafts"]
        if temp:
            check(acc["coupled"] > acc["argmax"] + 0.03,
                  f"acceptance at T={temp}: coupled {acc['coupled']:.3f} > argmax {acc['argmax']:.3f}")
        else:
            check(abs(acc["coupled"] - acc["argmax"]) < 1e-12,
                  f"acceptance at T=0: coupled {acc['coupled']:.3f} == argmax {acc['argmax']:.3f}")


def test_refused_combination():
    """Stock one-hot residual with the same noise: x ~ coupled draft; accept if u < p(x); else resample
    argmax(log p + g) over v != x with the SAME g. Biased for a noise-sampled x, exact for a deterministic x."""
    prompt = [9]
    p = exact_probs(prompt)
    lp = np.log(np.maximum(p, 1e-300))
    lp[p == 0] = -np.inf
    q_logits = (target_logits(prompt) + torch.from_numpy(np.random.default_rng(3).normal(0, 1.5, V))).numpy()
    rng = np.random.default_rng(11)
    N = 40000
    res = {}
    for kind in ("sampled", "argmax"):
        counts = np.zeros(V)
        for s in range(N):
            g = gc.gumbel_noise_np(s, 0, np.arange(V)).astype(np.float64)
            x = int(np.argmax(q_logits / TEMP + g)) if kind == "sampled" else int(np.argmax(q_logits))
            if rng.random() < p[x]:          # one-hot q: accept with probability min(1, p(x) / 1)
                y = x
            else:
                r = lp.copy()
                r[x] = -np.inf
                y = int(np.argmax(r + g))
            counts[y] += 1
        res[kind] = gtest(counts, p)
    # the image's probabilistic DFlash2 path: full q (softmax of the draft scores), Leviathan ratio test, residual
    # max(p - q, 0) resampled with the SAME keyed noise the draft was drawn with (dflash2 walk + _resample_kernel)
    q = np.exp(q_logits / TEMP - np.max(q_logits / TEMP)); q /= q.sum()
    lr = np.log(np.maximum(p - q, 0) + 1e-300); lr[p - q <= 0] = -np.inf
    counts = np.zeros(V)
    for s in range(N):
        g = gc.gumbel_noise_np(s, 0, np.arange(V)).astype(np.float64)
        x = int(np.argmax(np.log(q) + g))
        y = x if rng.random() < min(1.0, p[x] / q[x]) else int(np.argmax(lr + g))
        counts[y] += 1
    gq, dofq = gtest(counts, p)
    print(f"info image path (draft_sample_method=probabilistic: full q, residual resampled with the draft's own noise):"
          f" G {gq:.1f} vs 99.9 % critical {chi2_crit(dofq):.1f} (consistent with the target in this example; the"
          f" bias needs a MIS-specified q, i.e. a sampled draft verified as one-hot)")
    gs, dofs = res["sampled"]
    ga, dofa = res["argmax"]
    check(gs > chi2_crit(dofs), f"refused combination is biased: noise-sampled draft + stock one-hot residual with "
                                f"the same noise, G {gs:.0f} > {chi2_crit(dofs):.1f}")
    check(ga < chi2_crit(dofa), f"control: deterministic draft + stock one-hot residual is exact, G {ga:.1f} < "
                                f"{chi2_crit(dofa):.1f}")


def test_compare():
    # two requests: r0 rows 0..3 (3 drafts), r1 rows 4..5 (1 draft)
    ref = torch.tensor([5, 6, 7, 8, 1, 2])
    draft = torch.tensor([0, 5, 6, 9, 3, 1])         # r0: d1=5 ok, d2=6 ok, d3=9 != 7 -> n=3; r1: d1=1 == ref[4]
    cu = torch.tensor([0, 4, 6])
    sampled = torch.tensor([[5, 6, 7, 0], [1, 2, 0, 0]])
    ns = torch.tensor([3, 2])
    temp = torch.ones(6, dtype=torch.bool)
    ok = gc.compare(ref, ref, draft, cu, sampled, ns, temp)
    check(ok["bad"] == 0 and ok["sampled_drafts"] == 4 and ok["sampled_accepted"] == 3,
          f"compare(): a correct step passes ({ok})")
    cases = {
        "wrong committed token": (sampled.clone().index_put_((torch.tensor(0), torch.tensor(2)), torch.tensor(9)), ns),
        "accepted past a mismatch": (torch.tensor([[5, 6, 9, 8], [1, 2, 0, 0]]), torch.tensor([4, 2])),
        "rejected a matching draft": (torch.tensor([[5, 6, 0, 0], [1, 2, 0, 0]]), torch.tensor([2, 2])),
        "num_sampled out of range": (sampled, torch.tensor([3, 0])),
    }
    for name, (s, n) in cases.items():
        r = gc.compare(ref, ref, draft, cu, s, n, temp)
        check(r["bad"] > 0, f"compare(): flags '{name}' ({r['bad']})")


def test_walk():
    rng = np.random.default_rng(5)
    K, C = 7, 16
    scores = torch.from_numpy(rng.normal(0, 2, (K, C, C))).float()
    cands = torch.from_numpy(rng.choice(1000, (K, C), replace=True)).long()
    pos = torch.arange(100, 100 + K)
    greedy = []
    prev = 0
    for j in range(K):
        i = int(torch.argmax(scores[j, prev])); greedy.append(int(cands[j, i])); prev = i
    nz = lambda s, p, t: torch.from_numpy(gc.gumbel_noise_np(s, p, t.numpy()).astype(np.float64)).float()  # noqa
    check(gc.walk_torch(scores, cands, pos, 0.0, 7, top_p=0.9, noise=nz) == greedy, "walk: T=0 is the argmax walk")
    keep = gc.draft_topp_keep(torch.tensor([3.0, 1.0, 0.5, -2.0]), 1.0, 0.01)
    check(bool(keep[0]) and int(keep.sum()) == 1, "draft nucleus: a tiny top-p keeps exactly the best candidate")
    keep = gc.draft_topp_keep(torch.tensor([3.0, 1.0, 0.5, -2.0]), 1.0, 1.0)
    check(bool(keep.all()), "draft nucleus: top-p 1 keeps every candidate")


def test_hooks():
    import types
    import glm_ab
    torch.manual_seed(0)
    Vh = 50

    def gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature, use_fp64=False, **kw):
        out = []
        for i in range(logits.shape[0]):
            r = int(expanded_idx_mapping[i])
            x = logits[i].double()
            if float(temperature[r]) != 0.0:
                x = x + torch.from_numpy(gc.gumbel_noise_np(int(seed[r]), int(pos[i]), np.arange(Vh)).astype(np.float64))
            out.append(int(torch.argmax(x)))
        return torch.tensor(out, dtype=torch.int64)

    class Arr:
        def __init__(self, t):
            self.gpu, self.np = t, t.numpy()

    class States:
        def __init__(self, n):
            self.max_num_reqs = n
            self.temperature = Arr(torch.tensor([1.0, 0.0, 0.7]))
            self.seeds = Arr(torch.tensor([11, 22, -33], dtype=torch.int64))
            self.top_p = Arr(torch.tensor([0.9, 1.0, 0.8]))

    class Sampler:
        def __init__(self, max_num_reqs, vocab_size, device, *a, **k):
            self.sampling_states = States(max_num_reqs)
            self.use_fp64_gumbel = False
            self.use_flashinfer = True

        def add_request(self, req_idx, prompt_len, sp):
            pass

        def apply_sampling_params(self, logits, eidx, idx, idx_np, pos, input_ids, elp, skip_top_k_top_p=False):
            rows = []
            for i in range(logits.shape[0]):
                r = int(eidx[i])
                t = float(self.sampling_states.temperature.np[r])
                rows.append(processed(logits[i], t, 1.0 if skip_top_k_top_p else float(self.sampling_states.top_p.np[r])))
            return torch.stack(rows).float()

        def sample(self, logits, eidx, idx, idx_np, pos, input_ids, elp, return_logprobs=False):
            assert not self.use_flashinfer, "check mode must force the gumbel path"
            p = self.apply_sampling_params(logits, eidx, idx, idx_np, pos, input_ids, elp, skip_top_k_top_p=True)
            rows = [processed(p[i].double(), 1.0, float(self.sampling_states.top_p.np[int(eidx[i])]))
                    for i in range(p.shape[0])]
            p = torch.stack(rows).float()
            return gumbel_sample(p, eidx, self.sampling_states.temperature.gpu, self.sampling_states.seeds.gpu, pos,
                                 False), p

    class RejectionSampler:
        def __init__(self, sampler):
            self.sampler, self.num_speculative_steps = sampler, 4
            self.synthetic_conditional_rates, self.use_block_verification = None, False

        def _verify(self, *a):
            return "stock"

    fake = {"vllm": types.ModuleType("vllm"), "vllm.v1.worker.gpu.sample.gumbel": types.ModuleType("g")}
    fake["vllm.v1.worker.gpu.sample.gumbel"].gumbel_sample = gumbel_sample
    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)
    try:
        smod = types.SimpleNamespace(Sampler=Sampler, __name__=gc.SAMPLER)
        rmod = types.SimpleNamespace(RejectionSampler=RejectionSampler, __name__=gc.RS)
        gc.install_sampler(smod)
        gc.install_rejection(rmod)
        rs = RejectionSampler(Sampler(3, Vh, "cpu"))
        # 3 requests: slots 0 (T=1, p .9), 1 (greedy), 2 (T=.7, p .8); 4 / 2 / 3 rows
        cu = torch.tensor([0, 4, 6, 9])
        eidx = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
        pos = torch.tensor([10, 11, 12, 13, 5, 6, 40, 41, 42])
        logits = torch.randn(9, Vh) * 2
        ref, _ = None, None
        # drafts: make the first two drafts of request 0 equal the coupled draws (so something is accepted)
        draft = torch.randint(0, Vh, (9,))
        os.environ[gc.KEY] = "1"
        gc._MODE = "1"
        tgt = gumbel_sample(rs.sampler.apply_sampling_params(logits, eidx, None, None, pos, draft, None), eidx,
                            rs.sampler.sampling_states.temperature.gpu, rs.sampler.sampling_states.seeds.gpu, pos, False)
        draft[1], draft[2] = tgt[0], tgt[1]
        draft[5] = -1
        args = (logits, None, draft, pos, cu, torch.arange(3), np.arange(3), eidx, None)
        gc._MODE = "0"
        check(rs._verify(*args) == "stock", "hooks: mode 0 runs the stock rule")
        gc._MODE = "1"
        proc, sampled, ns = rs._verify(*args)
        check(int(ns[0]) >= 3 and int(sampled[0, 0]) == int(tgt[0]) and int(ns[1]) == 1,
              f"hooks: mode 1 accepts the matching drafts and stops at -1 (num_sampled {ns.tolist()})")
        gc._MODE = "check"
        before = gc.Stats.mismatch
        rs._verify(*args)
        check(gc.Stats.checked >= 1 and gc.Stats.mismatch == before and rs.sampler.use_flashinfer,
              f"hooks: check mode, 0 mismatches vs Sampler.sample, FlashInfer flag restored ({gc.stats()})")
        orig_cv = gc.coupled_verify

        def corrupt(*a):
            p, s_, n_, t_ = orig_cv(*a)
            s_ = s_.clone(); s_[0, 0] = (s_[0, 0] + 1) % Vh
            return p, s_, n_, t_
        gc.coupled_verify = corrupt
        rs._verify(*args)
        gc.coupled_verify = orig_cv
        check(gc.Stats.mismatch == before + 1, "hooks: check mode counts a planted corruption")
        # glm_ab guard: variants differ on the switch, drafter graphs shared -> refuse
        env = {"GLM_AB_VARIANTS": "2", "GLM_AB_V0": "GLM_GUMBEL_COUPLED=0", "GLM_AB_V1": "GLM_GUMBEL_COUPLED=1"}
        glm_ab.KNOWN.setdefault(gc.KEY, "mode")
        sys.modules["glm_ab"] = glm_ab
        glm_ab.configure(dict(env))
        try:
            gc._ab_guard()
            check(False, "hooks: glm_ab guard refuses shared drafter graphs")
        except RuntimeError:
            check(True, "hooks: glm_ab guard refuses shared drafter graphs")
        env["GLM_AB_DRAFT_SETS"] = "1"
        glm_ab.configure(dict(env))
        gc._ab_guard()
        glm_ab.set_runtime(1)
        m1 = gc._mode()
        glm_ab.set_runtime(0)
        check(m1 == "1" and gc._mode() == "0", "hooks: glm_ab switches the mode per variant")
        glm_ab.configure({})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        gc._MODE = "0"
        os.environ.pop(gc.KEY, None)


if __name__ == "__main__":
    test_noise()
    test_compare()
    test_walk()
    test_hooks()
    test_sequences()
    test_acceptance()
    test_refused_combination()
    test_distribution()
    print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)

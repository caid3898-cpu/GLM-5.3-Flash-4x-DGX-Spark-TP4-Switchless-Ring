"""CPU test for overlay/glm_mamba_align_fix.py (issue #2): no torch, no vLLM install.

Runs the image's own ``Scheduler._mamba_block_aligned_split`` (vendored verbatim in tests/fixtures, sha-pinned)
and the patched one through a prefill simulation with this recipe's geometry: attention/hash block 1152 (the
DFlash2 drafter group sets cache_config.block_size), KDA state block 2304 (--block-size), DFlash counts as Eagle,
6912 batched tokens minus 7 lookahead = 6905-token chunk budget.

Model of align mode (vLLM V2 mamba_hybrid): each chunk writes only its final state, into slot (end - 1) // 2304;
a slot is a real (publishable) block only if some chunk ended in it. Every published full block i must hold the
state after exactly (i + 1) * 2304 tokens. The repaired coordinator can hit up to floor((P - 1) / 2304) blocks,
so with the fix that last full block must also be materialized.

  python3 -B tests/test_glm_mamba_align_fix.py
"""
import os
import sys
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "fixtures"), os.path.join(HERE, "..", "overlay")]

import glm_mamba_align_fix as G  # noqa: E402
import vllm_487ecf187_mamba_split as V  # noqa: E402

ATTN_BS, MAMBA_BS, BUDGET = 1152, 2304, 6912 - 7
STOCK = V.Scheduler._mamba_block_aligned_split
FIXED = G.build(STOCK, vars(V))


def fake_sched(budget):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=ATTN_BS, mamba_block_size=MAMBA_BS),
        use_eagle=True, max_num_scheduled_tokens=budget, hash_block_size=ATTN_BS, mamba_partial_cache_hit=False,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0))


def prefill(fn, P, start=0, budget=BUDGET):
    s = fake_sched(budget)
    req = SimpleNamespace(num_computed_tokens=start, num_prompt_tokens=P, num_tokens=P, shared_prefix_boundary=0)
    ends = []
    for _ in range(10000):
        if req.num_computed_tokens >= P:
            return ends
        n = fn(s, req, min(budget, P - req.num_computed_tokens))
        if n <= 0:
            raise AssertionError(f"stall at {req.num_computed_tokens} (P={P})")
        req.num_computed_tokens += n
        ends.append(req.num_computed_tokens)
    raise AssertionError("no progress")


def violations(P, ends, need_last=False):
    """(bad published blocks, missing last-hit block)"""
    last = {}
    for e in ends:
        last[(e - 1) // MAMBA_BS] = e
    bad = [(i, e) for i, e in sorted(last.items()) if (i + 1) * MAMBA_BS <= P and e != (i + 1) * MAMBA_BS]
    h = (P - 1) // MAMBA_BS          # blocks the repaired coordinator can hit
    missing = need_last and h > 0 and last.get(h - 1) != h * MAMBA_BS
    return bad, missing


# 31 prompt lengths: the issue's (99447 / 97596 / 102866 / 106073) plus every residue class that matters
# (0, 1, 1151, 1152, 1153, 2303 mod 2304) at small, medium and ~100k lengths, and a few warm-start lengths.
CASES = [99447, 97596, 102866, 106073, 5000, 7000, 13824, 20000, 262143]
CASES += [MAMBA_BS * k + r for k in (3, 20, 43) for r in (0, 1, 1151, 1152, 1153, 2303)]
CASES += [MAMBA_BS * 60 + 700, MAMBA_BS * 61 + 1500, MAMBA_BS * 90 + 1152, MAMBA_BS * 100 + 1]
assert len(CASES) == 31


class MambaAlignFix(unittest.TestCase):
    def test_issue_prompt_endpoints(self):
        stock = prefill(STOCK, 99447)
        fixed = prefill(FIXED, 99447)
        self.assertEqual(stock[-2:], [97920, 99447])         # slot 42 keeps S(97920), hashed as 99072
        self.assertEqual(violations(99447, stock)[0][-1], (42, 97920))
        self.assertEqual(fixed[-3:], [96768, 99072, 99447])  # materialized: 99072 is now a real checkpoint
        self.assertEqual(violations(99447, fixed, need_last=True), ([], False))

    def test_stock_is_wrong_on_most_lengths(self):
        failing = [P for P in CASES if violations(P, prefill(STOCK, P))[0]]
        self.assertGreaterEqual(len(failing), 15, failing)

    def test_fixed_all_cases_cold(self):
        for P in CASES:
            with self.subTest(P=P):
                self.assertEqual(violations(P, prefill(FIXED, P), need_last=True), ([], False))

    def test_fixed_warm_starts_and_budgets(self):
        for P in CASES:
            for budget in (BUDGET, 6912, 4096, 8192 - 7, 2048):
                h = (P - 1) // MAMBA_BS
                for start in {0, max(h - 1, 0) * MAMBA_BS, max(h - 5, 0) * MAMBA_BS}:
                    if start >= P:
                        continue
                    with self.subTest(P=P, budget=budget, start=start):
                        ends = prefill(FIXED, P, start=start, budget=budget)
                        self.assertEqual(violations(P, ends, need_last=True), ([], False))

    def test_chunk_size_side_effect(self):
        # 6905-token budget: stock aligns chunks to 1152 (5760), fixed to 2304 (4608); 6919 restores 6912 chunks.
        self.assertEqual(prefill(STOCK, 50000)[0], 5760)
        self.assertEqual(prefill(FIXED, 50000)[0], 4608)
        self.assertEqual(prefill(FIXED, 50000, budget=6919)[0], 6912)

    def test_hash_pin_refuses_other_source(self):
        def other(self, request, num_new_tokens):
            return num_new_tokens
        with self.assertRaises(RuntimeError):
            G.build(other, {})

    def test_switch(self):
        os.environ["GLM_MAMBA_ALIGN_FIX"] = "0"
        try:
            self.assertFalse(G.enabled())
        finally:
            del os.environ["GLM_MAMBA_ALIGN_FIX"]
        self.assertTrue(G.enabled())


if __name__ == "__main__":
    unittest.main(verbosity=1)

"""CPU-only integer lifecycle proof; does not import Torch/vLLM or touch GPU."""
import ast
import json
import math
from pathlib import Path
import struct
import unittest

HERE = Path(__file__).resolve().parents[1] / "overlay"
MARKER = 0x7FC51A5E
# Exercise the installed slow wrapper expression without importing Torch or CUDA.
_tree = ast.parse((HERE / "glm_kda_stash.py").read_text())
_assign = next(n for n in ast.walk(_tree) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "to"
               and isinstance(n.value.func.value, ast.Compare)
               and "lookahead" in ast.unparse(n.value.func.value))
_FULL_EXPR = compile(ast.Expression(_assign.value.func.value), "stash_full_flag", "eval")


def full(first, length, block, max_query, patched):
    if patched:
        return eval(_FULL_EXPR, {"__builtins__": {}}, {
            "first": first, "last": first + length - 1,
            "lookahead": max_query - 1, "bs": block,
        })
    return (first + length) // block != first // block


def migrated_marker(first, length, accepted, next_query, block, max_query, patched):
    src = (first + length - 1) // block
    after = first + accepted
    dst = (after + next_query - 1) // block
    raw_stash = accepted > 1 and not full(first, length, block, max_query, patched)
    return src != dst and raw_stash


class BoundaryTests(unittest.TestCase):
    def test_observed_boundary_timeline(self):
        # Previous verify 55290..55293 accepts 3; next verify 55293..55296
        # moves from column23 to24 before forward and copies previous slot2.
        args = (55290, 4, 3, 4, 2304, 8)
        self.assertTrue(migrated_marker(*args, False))
        self.assertFalse(migrated_marker(*args, True))
        self.assertTrue(math.isnan(struct.unpack('<f', struct.pack('<I', MARKER))[0]))

    def test_exhaustive_supported_near_boundaries(self):
        checked = unsafe_old = 0
        for block in (16, 32, 2304):
            for max_query in (2, 4, 6, 8):
                for boundary in (block, block * 3, block * 24):
                    for first in range(boundary - 2 * max_query, boundary + max_query):
                        for length in range(1, max_query + 1):
                            for accepted in range(1, length + 1):
                                for next_query in range(1, max_query + 1):
                                    args = (first, length, accepted, next_query, block, max_query)
                                    old = migrated_marker(*args, False)
                                    new = migrated_marker(*args, True)
                                    self.assertFalse(new, args)
                                    unsafe_old += old
                                    checked += 1
        self.assertGreater(unsafe_old, 0)
        print(json.dumps({'timelines': checked, 'old_unsafe': unsafe_old, 'new_unsafe': 0}))

    def test_preserves_existing_full_flag(self):
        for first in range(0, 128):
            for length in range(1, 9):
                if full(first, length, 32, 8, False):
                    self.assertTrue(full(first, length, 32, 8, True))

    def test_interior_keeps_stash(self):
        self.assertFalse(full(50000, 8, 2304, 8, True))

    def test_source_wiring(self):
        for name in ('glm_kda_stash.py', 'glm_kda_stash_fast.py'):
            text = (HERE / name).read_text()
            ast.parse(text)
            self.assertIn('md.spec_state_indices_tensor.shape[-1] - 1', text)
            self.assertIn('last + 1 + lookahead', text)
        fast = (HERE / 'glm_kda_stash_fast.py').read_text()
        self.assertIn('n, hi, bs, lookahead,', fast)
        self.assertIn('spec_query_start_loc, n, bs, lookahead)', fast)
        self.assertIn('other=0).to(tl.int64) + 1 + lookahead', fast)


if __name__ == '__main__':
    unittest.main(verbosity=2)

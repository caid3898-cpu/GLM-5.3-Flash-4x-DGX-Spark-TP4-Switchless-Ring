#!/usr/bin/env python3
"""CPU tests for overlay/glm_mhc_bf16w.py: the source patch against the image's real tilelang_kernels.py, and the
dispatcher routing with fake tensors. Standard library only (the bitwise GPU check is gpu/gpu_test.py)."""
from __future__ import annotations

import ast
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
import glm_mhc_bf16w as b  # noqa: E402

def _image_tk():
    """tilelang_kernels.py of the image's vLLM: GLM_IMAGE_TK, else GLM_IMAGE_SRC/vllm/..., else the installed vLLM."""
    if os.environ.get("GLM_IMAGE_TK"):
        return os.environ["GLM_IMAGE_TK"]
    rel = "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    if os.environ.get("GLM_IMAGE_SRC"):
        return os.path.join(os.environ["GLM_IMAGE_SRC"], rel)
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    return os.path.join(os.path.dirname(os.path.dirname(spec.origin)), rel) if spec else None


TK = _image_tk()


class TestPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TK or not os.path.exists(TK):
            raise unittest.SkipTest("no image vLLM source: set GLM_IMAGE_SRC / GLM_IMAGE_TK or run inside the image")
        cls.mod_src = open(TK).read()
        cls.src = b.extract_kernel_source(cls.mod_src)

    def test_extract_bounds(self):
        self.assertTrue(self.src.startswith("@tilelang.jit("))
        self.assertIn("def mhc_fused_tilelang(", self.src)
        self.assertNotIn("def mhc_post_tilelang(", self.src)
        self.assertEqual(self.src.count("@tilelang.jit("), 1)

    def test_patch_is_minimal(self):
        for cast in b.CASTS:
            out = b.patch_source(self.src, cast)
            ast.parse(out)
            self.assertIn("def mhc_fused_bf16w_tilelang(", out)
            self.assertIn(b.DECL_NEW, out)
            a, c = self.src.splitlines(), out.splitlines()
            self.assertEqual(len(a), len(c))
            diff = [(x, y) for x, y in zip(a, c) if x != y]
            self.assertEqual(len(diff), 3, diff)  # def name, weight_t dtype, the one FMA line
            fma = [y for x, y in diff if "acc[n] +=" in y][0]
            self.assertIn(cast.format("weight_t[i_nt * tile_n + n, j, h_idx]"), fma)
            self.assertTrue(fma.rstrip().endswith("* new_r[j]"))  # operand order unchanged: w * r + acc

    def test_other_fp32_inputs_untouched(self):
        out = b.patch_source(self.src)
        for decl in ("comb_mix: T.Tensor((m, hc, hc), T.float32)", "post_mix: T.Tensor((m, hc), T.float32)",
                     "yp_out: T.Tensor((split_k, m, n_out), T.float32)"):
            self.assertIn(decl, out)

    def test_drift_fails_closed(self):
        with self.assertRaises(RuntimeError):
            b.patch_source(self.src.replace(b.FMA_OLD, "acc[n] += new_r[j] * weight_t[i_nt * tile_n + n, j, h_idx]"))
        with self.assertRaises(RuntimeError):
            b.patch_source(self.src + "\n" + b.DECL_OLD + "\n")

    def test_wrapper_looks_kernel_up_at_call_time(self):
        """tilelang.py imports mhc_fused_tilelang inside the wrapper body, so replacing the module attribute
        reroutes every later call (the property the dispatcher relies on)."""
        wrapper = open(os.path.join(os.path.dirname(TK), "tilelang.py")).read()
        body = wrapper[wrapper.index("def mhc_fused_post_pre_tilelang("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertIn("from vllm.model_executor.kernels.mhc.tilelang_kernels import (", body)
        self.assertIn("mhc_fused_tilelang,", body)
        self.assertIn("fn.view(hc_mult3, hc_mult, hidden_size)", body)
        self.assertIn("use_small_fma = num_tokens <= 16", body)


class FT:
    """Fake tensor for the dispatcher."""

    def __init__(self, ptr, shape, dtype="torch.float32", contiguous=True):
        self.ptr, self.shape, self.dtype, self._c = ptr, tuple(shape), dtype, contiguous

    def data_ptr(self):
        return self.ptr

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def is_contiguous(self):
        return self._c

    def view(self, shape):
        return FT(self.ptr, shape, self.dtype)


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.calls = []
        b.S.update(ok=True, stock=lambda *a, **k: self.calls.append(("stock", a[4])),
                   kernel=lambda *a, **k: self.calls.append(("bf16", a[4])), calls_bf16=0, calls_stock=0)
        self.fn = FT(0x1000, (24, 4, 4096))
        self.tw = FT(0x9000, (24, 16384), dtype="torch.bfloat16")
        b.S["twins"] = {0x1000: self.tw}
        os.environ["GLM_MHC_BF16W"] = "1"

    def tearDown(self):
        os.environ.pop("GLM_MHC_BF16W", None)
        b.S.update(ok=False, twins={})

    def args(self, w):
        return ("comb", "res", "post", "x", w, "yp", "rp", "rout", 4, 4096, 24)

    def test_routes_twin(self):
        b.dispatch(*self.args(self.fn), tile_n=2, n_splits=8)
        kind, w = self.calls[-1]
        self.assertEqual(kind, "bf16")
        self.assertEqual((w.ptr, w.shape, w.dtype), (0x9000, (24, 4, 4096), "torch.bfloat16"))

    def test_unknown_or_off_goes_stock(self):
        b.dispatch(*self.args(FT(0x2000, (24, 4, 4096))))
        os.environ["GLM_MHC_BF16W"] = "0"
        b.dispatch(*self.args(self.fn))
        os.environ["GLM_MHC_BF16W"] = "1"
        b.S["ok"] = False
        b.dispatch(*self.args(self.fn))
        self.assertEqual([k for k, _ in self.calls], ["stock", "stock", "stock"])

    def test_non_fp32_weight_goes_stock(self):
        b.dispatch(*self.args(FT(0x1000, (24, 4, 4096), dtype="torch.bfloat16")))
        self.assertEqual(self.calls[-1][0], "stock")




class TestKernelModule(unittest.TestCase):
    def test_free_names_are_imported(self):
        if not TK or not os.path.exists(TK):
            self.skipTest("no image vLLM source")
        import builtins
        src = open(TK).read()
        text, _ = b.kernel_module_text(src, b.CASTS[0])
        tree = ast.parse(text)
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)][0]
        loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        stores = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        args = {a.arg for a in fn.args.args}
        imported = set()
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                imported |= {a.asname or a.name for a in n.names}
        free = loads - stores - args - set(dir(builtins)) - imported
        self.assertEqual(free, set(), free)
        # every imported name exists at module level in the stock kernel file
        stock = ast.parse(src)
        top = set()
        for n in stock.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                top |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.Assign):
                top |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                top.add(n.target.id)
            elif isinstance(n, ast.If):  # the guarded `import tilelang` / `import tilelang.language as T`
                for m in ast.walk(n):
                    if isinstance(m, (ast.Import, ast.ImportFrom)):
                        top |= {(a.asname or a.name).split(".")[0] for a in m.names}
        self.assertTrue({"ENABLE_PDL", "T", "pass_configs", "tilelang", "math"} <= top, top)


if __name__ == "__main__":
    unittest.main(verbosity=2)

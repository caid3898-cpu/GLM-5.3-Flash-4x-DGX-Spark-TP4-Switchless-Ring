"""CPU tests of overlay/glm_marlin_tune.py with fake tensors / ops (no torch): table schema, M buckets, the MoE
override injection, the JIT routing, capture-time fallbacks, the dense wrapper and the pass-through cases."""
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "overlay"))
import glm_marlin_tune as T  # noqa: E402


class Dt:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"


class FT:
    """Minimal tensor stand-in: shape, dtype, reshape, slicing, size()."""

    def __init__(self, shape, dtype="bfloat16", tag=None):
        self.shape, self.dtype, self.device, self.tag = tuple(shape), Dt(dtype), "cuda:0", tag

    def reshape(self, *shape):
        shape = list(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
        total = 1
        for s in self.shape:
            total *= s
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape[shape.index(-1)] = total // known
        return FT(shape, self.dtype.name, self.tag)

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def dim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        rows, cols = idx
        n = cols.stop if cols.stop is not None else self.shape[1]
        return FT((self.shape[0], n), self.dtype.name, (self.tag, "slice"))

    def contiguous(self):
        return FT(self.shape, self.dtype.name, (self.tag, "contig"))


def table(**kw):
    d = {"version": 1, "m_grid": [4, 8, 16, 32],
         "moe": {"persistent_workspace": True, "entries": [
             {"gemm": "gate_up", "m": 4, "thread_k": 128, "thread_n": 128, "blocks_per_sm": 1},
             {"gemm": "down", "m": 16, "jit": {"thread_k": 64, "thread_n": 128, "threads": 128, "stages": 3,
                                               "grid": 192, "smem": 30000}}]},
         "dense": {"unpad": "kernel", "entries": [
             {"fmt": "mxfp8", "size_n": 6464, "size_k": 4096, "m": 4,
              "jit": {"m_variant": "m8", "thread_k": 256, "thread_n": 64, "threads": 256, "stages": 4, "grid": 96,
                      "smem": 60000}}]},
         "jit": {"so": "/x/lib.so", "manifest": "/x/manifest.json"}}
    d.update(kw)
    return d


class FakeOps:
    def __init__(self):
        self.calls = []

    def moe_gemm(self, *a):
        self.calls.append(("moe", a))
        return FT((a[13] * a[11], a[14]), tag="jit-moe")

    def dense_gemm(self, *a):
        self.calls.append(("dense", a))
        x, n_out, size_n = a[0], a[6], a[4]
        return FT((x.shape[0], n_out if n_out else size_n), tag="jit-dense")


class FakeJit:
    def __init__(self):
        self.ops = FakeOps()
        self._dense = {("mxfp8", "m8", 256, 64, 256, 4): 7}
        self._moe = {(64, 128, 128, 3): 2}
        self.manifest = {"hash": "fake"}


class Base(unittest.TestCase):
    def setUp(self):
        self.saved = dict(T.S)
        T.S.update(table=None, jit=None, moe_ws={}, logged=set(), applied={}, fallback={}, checked=0)
        self.capturing = False
        self._cap, self._nw, self._sms = T._capturing, T._new_workspace, T._sms
        T._capturing = lambda: self.capturing
        T._new_workspace = lambda device, n: FT((n,), "int32", tag="ws")
        T._sms = lambda device: 48

    def tearDown(self):
        T.S.clear()
        T.S.update(self.saved)
        T._capturing, T._new_workspace, T._sms = self._cap, self._nw, self._sms


def moe_args(M, gemm, scales_dtype="float8_e4m3fn"):
    size_n, size_k, top_k, size_m = (1024, 4096, 8, M) if gemm == "gate_up" else (4096, 512, 1, M * 8)
    pos = (FT((size_m, size_k)), FT((size_m * top_k, size_n)), FT((288, 1, 1), "int32"), None,
           FT((288, 1, size_n), scales_dtype), None, FT((288,), "float32"), None, None, None, FT((192,), "int32"),
           FT((100,), "int32"), FT((20,), "int32"), FT((1,), "int32"), FT((M, 8), "float32"))
    kw = dict(moe_block_size=8, top_k=top_k, mul_topk_weights=gemm == "down", b_q_type="fp4", size_m=size_m,
              size_n=size_n, size_k=size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
              is_zp_float=False)
    return pos, kw


class TableTests(unittest.TestCase):
    def test_parse_and_lookup(self):
        t = T.Table(table())
        self.assertEqual(t.moe_entry(1024, 4096, 4)["thread_k"], 128)
        self.assertEqual(t.moe_entry(1024, 4096, 1)["thread_k"], 128)      # (0, 4] uses the M=4 entry
        self.assertIsNone(t.moe_entry(1024, 4096, 5))                     # (4, 8]: no gate_up entry -> stock
        self.assertIsNone(t.moe_entry(1024, 4096, 33))                    # above the grid -> stock
        self.assertIn("jit", t.moe_entry(4096, 512, 12))                  # (8, 16] down -> JIT
        self.assertIsNone(t.moe_entry(512, 4096, 4))                      # not a tuned GEMM
        self.assertEqual(t.dense_entry("mxfp8", 6464, 4096, 3)["jit"]["grid"], 96)
        self.assertIsNone(t.dense_entry("mxfp8", 6464, 4096, 6))
        self.assertIsNone(t.dense_entry("fp8blk", 6464, 4096, 4))
        self.assertTrue(t.needs_jit and t.persistent_workspace and t.unpad == "kernel")

    def test_schema_errors(self):
        for bad in (dict(version=2), dict(dense={"unpad": "maybe"}),
                    dict(moe={"entries": [{"gemm": "up", "m": 4, "thread_k": 1, "thread_n": 1, "blocks_per_sm": 1}]}),
                    dict(moe={"entries": [{"gemm": "down", "m": 6, "thread_k": 64, "thread_n": 128, "blocks_per_sm": 1}]}),
                    dict(dense={"entries": [{"fmt": "mxfp8", "size_n": 1, "size_k": 1, "m": 16,
                                             "jit": {"m_variant": "m8", "thread_k": 1, "thread_n": 1, "threads": 1,
                                                     "stages": 1, "grid": 1, "smem": 1}}]}),
                    dict(jit=None)):
            with self.assertRaises((T.TableError, KeyError, ValueError, TypeError)):
                T.Table(table(**bad))

    def test_sha_guard(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(table(), f)
        with open(p, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertIsInstance(T.load_table(p, sha), T.Table)
        with self.assertRaises(T.TableError):
            T.load_table(p, "0" * 64)
        os.remove(p)

    def test_register_off_is_inert(self):
        os.environ.pop("GLM_MARLIN_TUNE", None)
        n = len(sys.meta_path)
        T.register()
        self.assertEqual(len(sys.meta_path), n)
        self.assertNotIn("torch", sys.modules)


class MoeWrapper(Base):
    def test_injects_exposed_override(self):
        seen = []

        def orig(*a, thread_k=-1, thread_n=-1, blocks_per_sm=-1):
            seen.append((thread_k, thread_n, blocks_per_sm))
            return a[1]

        w = T.make_moe_gemm_wrapper(orig)
        pos, kw = moe_args(3, "gate_up")
        T.S["table"] = None
        w(*pos, **kw)
        T.S["table"] = T.Table(table())
        w(*pos, **kw)
        w(*pos, **kw, thread_k=64, thread_n=128, blocks_per_sm=2)              # explicit override wins
        pos8, kw8 = moe_args(8, "gate_up")
        w(*pos8, **kw8)                                                           # no entry at 8 -> stock
        posx, kwx = moe_args(4, "gate_up", scales_dtype="float8_e8m0fnu")        # MXFP4 layer -> stock
        w(*posx, **kwx)
        self.assertEqual(seen, [(-1, -1, -1), (128, 128, 1), (64, 128, 2), (-1, -1, -1), (-1, -1, -1)])
        self.assertEqual(T.S["applied"], {("gate_up", 3): 1})

    def test_jit_route_and_capture_fallback(self):
        seen = []

        def orig(*a, **k):
            seen.append(k)
            return "stock"

        T.S["table"] = T.Table(table())
        T.S["jit"] = FakeJit()
        w = T.make_moe_gemm_wrapper(orig)
        pos, kw = moe_args(16, "down")
        self.capturing = True                       # first need inside capture: stock, no allocation
        self.assertEqual(w(*pos, **kw), "stock")
        self.assertEqual(T.S["fallback"], {("moe-jit-ws-in-capture", "down"): 1})
        self.capturing = False
        out = w(*pos, **kw)
        self.assertEqual(out.tag, "jit-moe")
        kind, a = T.S["jit"].ops.calls[-1]
        self.assertEqual(a[16:], (2, 192, 30000))  # kernel id, grid, smem
        self.assertEqual(a[5].tag, "ws")            # the persistent JIT workspace, not the caller's
        self.assertEqual(a[5].shape, (4 * 48 + 8,))
        self.capturing = True                       # later captures reuse the eager-allocated buffer
        self.assertEqual(w(*pos, **kw).tag, "jit-moe")

    def test_persistent_workspace(self):
        mod = types.SimpleNamespace(marlin_make_workspace_new=lambda device, bps=1, existing=None: FT((48 * bps,), "int32", tag="fresh"))
        T.install_moe(mod)
        f = mod.marlin_make_workspace_new
        self.assertEqual(f("cuda:0", 4).tag, "fresh")          # no table
        T.S["table"] = T.Table(table())
        self.capturing = True
        self.assertEqual(f("cuda:0", 4).tag, "fresh")          # never allocate inside capture
        self.capturing = False
        a = f("cuda:0", 4)
        self.assertEqual(a.tag, "ws")
        self.assertIs(f("cuda:0", 4), a)                        # one buffer per device
        self.assertEqual(f("cuda:0", 1).tag, "fresh")          # dense-style callers untouched
        t = table()
        t["moe"]["persistent_workspace"] = False
        T.S["table"] = T.Table(t)
        self.assertEqual(f("cuda:0", 4).tag, "fresh")


class DenseWrapper(Base):
    def layer(self):
        return types.SimpleNamespace(weight=FT((4096 // 16, 4 * 6464), "int32"), weight_scale=FT((128, 6464), "e8m0"),
                                     output_size_per_partition=6416, input_size_per_partition=4096,
                                     workspace=FT((48,), "int32"), __dict__={})

    def test_jit_unpadded_and_stock_elsewhere(self):
        seen = []

        def orig(self_, layer, x, bias=None):
            seen.append(x.shape)
            return FT((x.shape[0], 6416), tag="stock")

        T.S["table"] = T.Table(table())
        T.S["jit"] = FakeJit()
        w = T.make_dense_wrapper(orig, "mxfp8", lambda self_, layer: layer.weight_scale)
        layer = types.SimpleNamespace(weight=FT((256, 4 * 6464), "int32"), weight_scale=FT((128, 6464)),
                                      output_size_per_partition=6416, input_size_per_partition=4096)
        out = w(None, layer, FT((3, 4096)))
        self.assertEqual(out.shape, (3, 6416))
        kind, a = T.S["jit"].ops.calls[-1]
        self.assertEqual((a[4], a[5], a[6], a[7], a[8], a[9]), (6464, 4096, 6416, 7, 96, 60000))
        self.assertEqual(w(None, layer, FT((12, 4096))).tag, "stock")          # no (8, 16] entry
        self.assertEqual(w(None, layer, FT((4, 4096)), FT((6416,))).tag, "stock")   # bias -> stock
        self.assertEqual(seen, [(12, 4096), (4, 4096)])

    def test_capture_without_workspace_falls_back(self):
        T.S["table"] = T.Table(table())
        T.S["jit"] = FakeJit()
        w = T.make_dense_wrapper(lambda s, l, x, b=None: FT((x.shape[0], 6416), tag="stock"), "mxfp8",
                                 lambda s, l: l.weight_scale)
        layer = types.SimpleNamespace(weight=FT((256, 4 * 6464), "int32"), weight_scale=FT((128, 6464)),
                                      output_size_per_partition=6416, input_size_per_partition=4096)
        self.capturing = True
        self.assertEqual(w(None, layer, FT((4, 4096))).tag, "stock")
        self.capturing = False
        w(None, layer, FT((200, 4096)))                     # an eager prefill-sized call allocates the buffer
        self.capturing = True
        self.assertEqual(w(None, layer, FT((4, 4096))).tag, "jit-dense")

    def test_view_mode_unpadded_shape(self):
        t = table()
        t["dense"] = {"unpad": "view", "entries": t["dense"]["entries"]}
        T.S["table"] = T.Table(t)
        T.S["jit"] = FakeJit()
        w = T.make_dense_wrapper(lambda s, l, x, b=None: None, "mxfp8", lambda s, l: l.weight_scale)
        layer = types.SimpleNamespace(weight=FT((256, 4 * 6464), "int32"), weight_scale=FT((128, 6464)),
                                      output_size_per_partition=6416, input_size_per_partition=4096)
        out = w(None, layer, FT((4, 4096)))
        kind, a = T.S["jit"].ops.calls[-1]
        self.assertEqual(a[6], 0)                                        # padded write, then a view
        self.assertEqual(out.shape, (4, 6416))
        self.assertEqual(out.tag, ("jit-dense", "slice"))

    def test_fp8_per_channel_passthrough(self):
        calls = []

        class K:
            def apply_weights(self, layer, x, bias=None):
                calls.append("orig")
                return "orig"

        mod = types.SimpleNamespace(MarlinFP8ScaledMMLinearKernel=K)
        T.install_fp8(mod)
        T.S["table"] = T.Table(table())
        k = K()
        k.block_quant = False
        self.assertEqual(k.apply_weights(None, FT((4, 4096))), "orig")
        self.assertTrue(getattr(K.apply_weights, "_glm_marlin_tune", False))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""CPU tests for overlay/glm_l2_prefetch_mla.py (windows M, B-MLA, D). Standard library only. Uses the real
overlay/glm_l2_prefetch.py of the GLM repo for take()/module_tensors()/_plan_b()/_state; _table/_fork/_capturing
are faked (no torch needed)."""
from __future__ import annotations

import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
sys.path.insert(1, os.environ.get("GLM_REPO_OVERLAY", os.path.expanduser("~/Projects/glm53-flash-4x-spark/overlay")))

import glm_l2_prefetch as base  # noqa: E402
import glm_l2_prefetch_mla as m  # noqa: E402

MiB = 1 << 20
KNOBS = ("GLM_L2_PREFETCH", "GLM_L2_PREFETCH_AR", "GLM_L2_PREFETCH_MLA", "GLM_L2_PREFETCH_MLA_AR",
         "GLM_L2_PREFETCH_DRAFT", "GLM_L2_PREFETCH_MLA_MB", "GLM_L2_PREFETCH_DRAFT_MB", "GLM_L2_PREFETCH_MAXTOK",
         "GLM_L2_PREFETCH_DRAFT_MAXTOK", "GLM_GATE_GEMV", "GLM_L2_PREFETCH_AR_MB")


class T:
    """Fake CUDA tensor / parameter (1-D bytes unless shape/stride given)."""
    _next = 1 << 30

    def __init__(self, nbytes, shape=None, stride=None, es=1, cuda=True, ptr=None):
        if ptr is None:
            T._next = (T._next + 255) // 256 * 256
            ptr = T._next
            T._next += nbytes + 4096
        self.ptr, self.es, self.is_cuda = ptr, es, cuda
        self.shape = tuple(shape) if shape else (nbytes // es,)
        self._stride = tuple(stride) if stride else (1,) * len(self.shape) if len(self.shape) == 1 else None
        if self._stride is None:  # row-major
            st, acc = [], 1
            for s in reversed(self.shape):
                st.append(acc)
                acc *= s
            self._stride = tuple(reversed(st))

    def data_ptr(self):
        return self.ptr

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def element_size(self):
        return self.es

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._stride == T(0, self.shape, None, self.es, ptr=0)._stride

    def size(self, d):
        return self.shape[d]

    @property
    def data(self):
        return self


class Mod:
    """Fake nn.Module: named_parameters(recurse=False), modules(), instance-level forward override."""

    def __init__(self, params=None, children=None, **attrs):
        self._p = dict(params or {})
        self._c = list(children or [])
        for k, v in self._p.items():
            setattr(self, k, v)
        for k, v in attrs.items():
            setattr(self, k, v)

    def named_parameters(self, recurse=False):
        return iter(self._p.items())

    def modules(self):
        yield self
        for c in self._c:
            yield from c.modules()


def setup_env(**extra):
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update({"GLM_L2_PREFETCH": "1", "GLM_L2_PREFETCH_AR": "1", "GLM_L2_PREFETCH_MLA": "1",
                       "GLM_L2_PREFETCH_MLA_AR": "1", "GLM_L2_PREFETCH_DRAFT": "1"})
    os.environ.update(extra)


def fake_base():
    base._table = lambda segs: ("T", len(segs), sum(s[1] for s in segs), tuple(segs))
    forks = []
    base._fork = lambda item: forks.append(item)
    joins = []
    base.join_all = lambda: joins.append(1)
    base._state["depth"] = 0
    base._state["armed"] = None
    m._capturing = lambda: False
    m._state["draft_depth"] = 0
    return forks, joins


def mla_attn(ar_log=None):
    """Glm5NextMLAAttention stand-in: indexer, inner MLA layer with W_UK_T/W_UV, FP8 o_proj whose forward
    runs one (fake) custom all-reduce that consumes whatever is armed."""
    idx = Mod(wk_weights_proj=Mod({"weight": T(160 * 4096 * 2)}),
              index_kpool_compress_gate=T(128 * 4096 * 2, shape=(128, 4096), es=2))
    idx.__dict__["_wp_fp32"] = T(4096 * 32 * 4, shape=(4096, 32), es=4)
    wuk = T(4 * MiB, shape=(16, 256, 512), es=2)
    wuv_store = T(4 * MiB, shape=(512, 16, 256), es=2)
    wuv = T(0, shape=(16, 512, 256), stride=(256, 16 * 256, 1), es=2, ptr=wuv_store.ptr)  # transpose(0,1) view
    inner = Mod(W_UK_T=wuk, W_UV=wuv)
    wrapper = Mod(children=[inner])

    def o_forward(x):
        if ar_log is not None:
            ar_log.append(base._state["armed"])
            base._state["armed"] = None
        return ("o-out", None)

    o_proj = Mod({"weight": T(16 * MiB), "weight_scale": T(256 * 1024)})
    o_proj.forward = o_forward
    attn = Mod(children=[wrapper], indexer=idx, o_proj=o_proj, mla_attn=wrapper)
    return attn


def decoder_layer(attn):
    mlp = Mod(gate=Mod({"weight": T(288 * 4096 * 2)}),
              shared_experts=Mod(gate_up_proj=Mod({"weight": T(4 * MiB), "weight_scale": T(64 * 1024)})))
    return Mod(self_attn=attn, mlp=mlp, hc_ffn_fn=T(24 * 16384 * 4, shape=(24, 16384), es=4))


class Hidden:
    def __init__(self, n):
        self.n = n

    def size(self, d):
        return self.n


class TestSpan(unittest.TestCase):
    def test_contiguous_and_views(self):
        t = T(4 * MiB, shape=(16, 256, 512), es=2)
        self.assertEqual(m.tensor_span(t), (t.ptr, 4 * MiB))
        v = T(0, shape=(16, 512, 256), stride=(256, 16 * 256, 1), es=2, ptr=t.ptr)  # a transpose(0, 1) view
        self.assertEqual(m.tensor_span(v), (t.ptr, 4 * MiB))

    def test_sparse_view_refused(self):
        v = T(0, shape=(16, 128), stride=(4096, 1), es=2, ptr=1 << 32)  # 16 rows of a 4096-wide matrix
        self.assertIsNone(m.tensor_span(v))
        self.assertIsNone(m.tensor_span(None))
        self.assertIsNone(m.tensor_span(T(64, cuda=False)))


class TestQueueM(unittest.TestCase):
    def setUp(self):
        setup_env()
        fake_base()

    def test_order_stock_gate(self):
        a = mla_attn()
        q = m.queue_m(a, gate_stock=True)
        sizes = [n for _, n in q]
        self.assertEqual(sizes, [160 * 4096 * 2, 4096 * 32 * 4, 128 * 4096 * 2, 4 * MiB, 4 * MiB, 256 * 1024, 16 * MiB])
        self.assertEqual(q[0][0], a.indexer.wk_weights_proj.weight.ptr)

    def test_gemv_gate_skips_fp32_copy(self):
        q = m.queue_m(mla_attn(), gate_stock=False)
        self.assertNotIn(4096 * 32 * 4, [n for _, n in q])

    def test_budget_and_cache(self):
        a = mla_attn()
        p = m.plan_m(a)
        self.assertEqual(p[2], 16 * MiB)
        os.environ["GLM_L2_PREFETCH_MLA_MB"] = "8"
        p8 = m.plan_m(a)
        self.assertEqual(p8[2], 8 * MiB)
        os.environ["GLM_L2_PREFETCH_MLA_MB"] = "16"
        self.assertIs(m.plan_m(a), p)
        os.environ["GLM_GATE_GEMV"] = "1"
        self.assertIsNot(m.plan_m(a), p)  # the gate path is part of the cache key


class TestWindowM(unittest.TestCase):
    def setUp(self):
        setup_env()
        self.forks, _ = fake_base()
        calls = self.calls = []

        class Indexer:
            def forward(self, hidden_states, qr, positions, rotary_emb):
                calls.append(hidden_states.n)
                return "topk"

        self.mod = types.SimpleNamespace(Indexer=Indexer)
        m.install_attn(self.mod)
        self.attn = mla_attn()
        self.idx = self.mod.Indexer()
        self.idx.__dict__["_glm_l2m_attn"] = self.attn
        base._state["depth"] = 1

    def tearDown(self):
        base._state["depth"] = 0

    def test_first_eager_call_builds_after_then_forks(self):
        self.assertEqual(self.idx.forward(Hidden(4), None, None, None), "topk")
        self.assertEqual(self.forks, [])
        self.assertIn("_glm_l2_m", self.attn.__dict__)
        self.idx.forward(Hidden(4), None, None, None)
        self.assertEqual(len(self.forks), 1)
        self.assertEqual(self.forks[0][2], 16 * MiB)

    def test_capture_without_plan_prefetches_nothing(self):
        m._capturing = lambda: True
        self.idx.forward(Hidden(4), None, None, None)
        self.assertEqual(self.forks, [])
        self.assertNotIn("_glm_l2_m", self.attn.__dict__)

    def test_gates(self):
        self.idx.forward(Hidden(4), None, None, None)  # build
        self.idx.forward(Hidden(64), None, None, None)  # above MAXTOK 32
        os.environ["GLM_L2_PREFETCH_MLA"] = "0"
        self.idx.forward(Hidden(4), None, None, None)
        os.environ["GLM_L2_PREFETCH_MLA"] = "1"
        base._state["depth"] = 0  # outside the target forward
        self.idx.forward(Hidden(4), None, None, None)
        self.assertEqual(self.forks, [])
        self.assertEqual(self.calls, [4, 64, 4, 4])  # the stock forward always runs


class TestWindowBMLA(unittest.TestCase):
    def setUp(self):
        setup_env()
        fake_base()
        self.ar = []
        self.attn = mla_attn(self.ar)
        self.layer = decoder_layer(self.attn)
        model = Mod(layers=[self.layer])
        self.assertEqual(m.link_model(model), 1)
        base._state["depth"] = 1

    def tearDown(self):
        base._state["depth"] = 0

    def test_arm_consumed_by_o_proj_all_reduce(self):
        self.attn.o_proj.forward(Hidden(4))
        armed = self.ar[-1]
        self.assertIsNotNone(armed)
        self.assertEqual(armed[3][0], (self.layer.hc_ffn_fn.ptr, 24 * 16384 * 4))  # hc_ffn_fn first
        self.assertEqual(armed[2], 4 * MiB)
        self.assertIsNone(base._state["armed"])

    def test_indexer_linked(self):
        self.assertIs(self.attn.indexer.__dict__["_glm_l2m_attn"], self.attn)

    def test_not_consumed_does_not_leak(self):
        self.attn.o_proj.__dict__.pop("_glm_l2_bm")
        a2 = mla_attn(None)  # its o_proj runs no all-reduce
        m._wrap_o_proj_b(a2)
        a2.__dict__["_glm_l2_layer"] = self.layer
        a2.o_proj.forward(Hidden(4))
        self.assertIsNone(base._state["armed"])

    def test_off_and_cap(self):
        os.environ["GLM_L2_PREFETCH_MLA_AR"] = "0"
        self.attn.o_proj.forward(Hidden(4))
        os.environ["GLM_L2_PREFETCH_MLA_AR"] = "1"
        self.attn.o_proj.forward(Hidden(33))
        self.assertEqual(self.ar, [None, None])

    def test_kda_layer_not_touched(self):
        kda = Mod(o_proj=Mod(), in_proj_qkvbfg_a=Mod())
        self.assertFalse(m.is_mla(kda))


class TestWindowD(unittest.TestCase):
    def setUp(self):
        setup_env()
        self.forks, self.joins = fake_base()
        self.ar = []
        ar = self.ar

        def row_parallel():
            lin = Mod()

            def fwd(x):
                ar.append(base._state["armed"])
                base._state["armed"] = None
                return ("y", None)
            lin.forward = fwd
            return lin

        self.layers = []
        for _ in range(3):
            conv = lambda: Mod(kernel_projection=Mod({"weight": T(8 * MiB, shape=(1024, 4096), es=2)}))  # noqa: E731
            self.layers.append(Mod(self_attn=Mod(o_proj=row_parallel()), mlp=Mod(down_proj=row_parallel()),
                                   attention_conv=conv(), mlp_conv=conv()))
        layers = self.layers

        class DFlashQwen3Model:
            def __init__(self):
                self.layers = layers

            def forward(self, input_ids, positions, input_embeds=None):
                for lay in self.layers:
                    lay.self_attn.o_proj.forward(input_ids)
                    lay.mlp.down_proj.forward(input_ids)
                return "h"

        self.mod = types.SimpleNamespace(DFlashQwen3Model=DFlashQwen3Model)
        m.install_draft(self.mod)
        self.model = DFlashQwen3Model()

    def test_arms_next_conv_projection(self):
        self.model.forward(Hidden(8), None)   # eager: builds and arms
        L = self.layers
        want = [L[0].mlp_conv, L[1].attention_conv, L[1].mlp_conv, L[2].attention_conv, L[2].mlp_conv, None]
        for got, conv in zip(self.ar, want):
            if conv is None:
                self.assertIsNone(got)   # last layer's down_proj: nothing after it
            else:
                self.assertEqual(got[3][0][0], conv.kernel_projection.weight.ptr)
                self.assertEqual(got[2], 8 * MiB)
        self.assertEqual(len(self.joins), 1)
        self.assertEqual(m._state["draft_depth"], 0)

    def test_budget_and_cap(self):
        os.environ["GLM_L2_PREFETCH_DRAFT_MB"] = "2"
        self.model.forward(Hidden(8), None)
        self.assertEqual(self.ar[0][2], 2 * MiB)
        self.ar.clear()
        self.model.forward(Hidden(65), None)   # above GLM_L2_PREFETCH_DRAFT_MAXTOK 64
        self.assertTrue(all(a is None for a in self.ar))

    def test_off(self):
        os.environ["GLM_L2_PREFETCH_DRAFT"] = "0"
        self.model.forward(Hidden(8), None)
        self.assertTrue(all(a is None for a in self.ar))

    def test_target_windows_ignore_drafter(self):
        # window D never fires inside the target forward depth and target windows never fire in the drafter
        base._state["depth"] = 0
        self.assertFalse(m._on_m(4))


class TestRegister(unittest.TestCase):
    def test_default_off(self):
        for k in KNOBS:
            os.environ.pop(k, None)
        self.assertFalse(m.installed())
        os.environ["GLM_L2_PREFETCH"] = "1"
        self.assertFalse(m.installed())
        os.environ["GLM_L2_PREFETCH_MLA"] = "1"
        self.assertTrue(m.installed())
        self.assertNotIn(m.TARGET_DRAFT, m.active_hooks())
        os.environ["GLM_L2_PREFETCH_DRAFT"] = "1"
        self.assertIn(m.TARGET_DRAFT, m.active_hooks())
        for k in KNOBS:
            os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)

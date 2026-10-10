"""CPU checks for overlay/glm_sparse_mla_prefill.py (dispatch, fallback, glm_ab switch, install, drift guard).

Triton does not run here: the kernel is injected as a fake that records its arguments.

  <python with torch> test_glm_sparse_mla_prefill.py
"""
from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace as NS

import torch

import importlib.util  # noqa: E402

REPO = os.environ.get("GLM_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Root holding vllm/...: GLM_IMAGE_SRC (an extracted image source tree), else the vLLM installed here (the image).
_spec = None if os.environ.get("GLM_IMAGE_SRC") else importlib.util.find_spec("vllm")
IMAGE_SRC = os.environ.get("GLM_IMAGE_SRC") or (os.path.dirname(os.path.dirname(_spec.origin)) if _spec else None)
sys.path.insert(0, os.path.join(REPO, "overlay"))

import glm_prefill_hooks  # noqa: E402
import glm_sparse_mla_prefill as tm  # noqa: E402

H, D, PAGE, W = 16, 512, 2304, 2176


def _reset():
    tm._state.update(disabled=False, calls=0, stock=0, checks=0, reasons={}, installed=False)
    sys.modules.pop("glm_ab", None)


def make_case(T=8, prefills=1, kvd="fp8_e4m3", rope=0, heads=H, dcp=1, block=PAGE, topk_dtype=torch.int32):
    impl = NS(qk_rope_head_dim=rope, num_heads=heads, kv_lora_rank=D, dcp_world_size=dcp, kv_cache_dtype=kvd,
              scale=0.0625, topk_indices_buffer=torch.full((32, W), -1, dtype=topk_dtype))
    base = torch.randn(heads, T, D).to(torch.bfloat16)
    q_nope = base.transpose(0, 1)                    # (B, N, L) view of (N, B, L), as forward_impl passes it
    q_pe = torch.empty(T, heads, rope, dtype=torch.bfloat16)
    kv = torch.zeros(3, PAGE, D, dtype=torch.uint8 if kvd.startswith("fp8") else torch.bfloat16)
    md = NS(num_prefills=prefills, num_decode_tokens=0, block_size=block,
            block_table=torch.zeros(2, 4, dtype=torch.int32), req_id_per_token=torch.zeros(32, dtype=torch.int32))
    layer = NS(_k_scale_float=1.0)
    return impl, (q_nope, q_pe), kv, md, layer


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, q, kv, topk, rid, bt, sm_scale, kv_scale, cfg):
        self.calls.append(NS(q=q, kv=kv, topk=topk, rid=rid, bt=bt, sm_scale=sm_scale, kv_scale=kv_scale, cfg=cfg))
        if self.fail:
            raise RuntimeError("boom")
        return torch.full((q.shape[0], q.shape[1], q.shape[2]), 7.0, dtype=torch.bfloat16)


def stock():
    calls = []

    def prev(self, q, kv, md, layer):
        calls.append(q)
        return torch.zeros(q[0].shape, dtype=torch.bfloat16), None
    prev.calls = calls
    return prev


def test_register_inert():
    _reset()
    n = len(sys.meta_path)
    assert tm.register({}) is False and tm.register({"GLM_TRITON_MLA_PREFILL": "0"}) is False
    assert len(sys.meta_path) == n


def test_bad_cfg_is_fatal():
    _reset()
    for bad in ("32,4", "48,4,2", "32,3,2", "32,4,9", "32,4,2,3", "32,4,2,1,2", "32,4,2,1,1,1"):
        try:
            tm.register({"GLM_TRITON_MLA_PREFILL": "1", "GLM_TRITON_MLA_CFG": bad})
        except (RuntimeError, ValueError):
            continue
        raise AssertionError(bad)


def test_takes_prefill_step():
    _reset()
    prev, rec = stock(), Recorder()
    fwd = tm.make_forward_mqa(prev, rec, env={"GLM_TRITON_MLA_CFG": "64,8,3"})
    impl, q, kv, md, layer = make_case(T=8)
    layer._k_scale_float = 0.5
    out, lse = fwd(impl, q, kv, md, layer)
    assert lse is None and not prev.calls and len(rec.calls) == 1
    assert float(out[0, 0, 0]) == 7.0 and out.shape == (8, H, D)
    c = rec.calls[0]
    assert c.q is q[0] and c.kv.dtype == torch.float8_e4m3fn and c.kv.shape == kv.shape
    assert c.topk.shape == (8, W) and c.rid.shape == (8,) and c.bt is md.block_table
    assert c.sm_scale == 0.0625 and c.kv_scale == 0.5 and c.cfg == (64, 8, 3, 1, 0)
    assert tm.status()["triton_calls"] == 1


def test_bf16_cache_no_scale():
    _reset()
    prev, rec = stock(), Recorder()
    fwd = tm.make_forward_mqa(prev, rec, env={})
    impl, q, kv, md, layer = make_case(kvd="auto")
    layer._k_scale_float = 0.5
    fwd(impl, q, kv, md, layer)
    assert rec.calls[0].kv.dtype == torch.bfloat16 and rec.calls[0].kv_scale == 1.0 and rec.calls[0].cfg is None


def test_declines():
    cases = {
        "no-prefill": dict(prefills=0),
        "rope": dict(rope=64),
        "shape": dict(heads=8),
        "dcp": dict(dcp=2),
        "block-size": dict(block=64),
        "topk-layout": dict(topk_dtype=torch.int64),
        "kv-dtype": dict(kvd="fp8_e5m2"),
    }
    for why, kw in cases.items():
        _reset()
        prev, rec = stock(), Recorder()
        fwd = tm.make_forward_mqa(prev, rec, env={})
        impl, q, kv, md, layer = make_case(**kw)
        fwd(impl, q, kv, md, layer)
        assert not rec.calls and len(prev.calls) == 1, why
        assert tm.status()["reasons"] == {why: 1}, (why, tm.status())
    # q not split, too few rows, non-bf16 q
    _reset()
    prev, rec = stock(), Recorder()
    impl, q, kv, md, layer = make_case(T=4)
    tm.make_forward_mqa(prev, rec, env={"GLM_TRITON_MLA_MIN_ROWS": "5"})(impl, q, kv, md, layer)
    assert tm.status()["reasons"] == {"rows": 1}
    _reset()
    impl, q, kv, md, layer = make_case()
    try:
        tm.make_forward_mqa(prev, rec, env={})(impl, torch.cat([q[0], q[0]], -1), kv, md, layer)
    except Exception:  # noqa: BLE001  (the fake stock does not take an unsplit q)
        pass
    assert tm.status()["reasons"] == {"q-not-split": 1}
    _reset()
    impl, q, kv, md, layer = make_case()
    tm.make_forward_mqa(prev, rec, env={})(impl, (q[0].half(), q[1]), kv, md, layer)
    assert tm.status()["reasons"] == {"q-layout": 1} and not rec.calls


def test_capture_declines():
    _reset()
    prev, rec = stock(), Recorder()
    fwd = tm.make_forward_mqa(prev, rec, env={})
    orig = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    try:
        fwd(*make_case())
    finally:
        torch.cuda.is_current_stream_capturing = orig
    assert not rec.calls and tm.status()["reasons"] == {"capture": 1}


def test_ab_switch():
    _reset()
    ab = types.ModuleType("glm_ab")
    ab.ACTIVE, ab.KNOWN, ab.value = True, {"GLM_TRITON_MLA_PREFILL": "bool"}, "0"
    ab.env = lambda name, default=None: ab.value if name == "GLM_TRITON_MLA_PREFILL" else default
    sys.modules["glm_ab"] = ab
    try:
        prev, rec = stock(), Recorder()
        fwd = tm.make_forward_mqa(prev, rec, env={})
        fwd(*make_case())
        assert not rec.calls and len(prev.calls) == 1
        ab.value = "1"
        fwd(*make_case())
        assert len(rec.calls) == 1 and len(prev.calls) == 1
        ab.ACTIVE = False          # harness present but not armed: the install gate rules
        ab.value = "0"
        fwd(*make_case())
        assert len(rec.calls) == 2
    finally:
        sys.modules.pop("glm_ab", None)


def test_failure_falls_back_and_disables():
    _reset()
    prev, rec = stock(), Recorder(fail=True)
    fwd = tm.make_forward_mqa(prev, rec, env={})
    out, _ = fwd(*make_case())
    assert float(out.abs().max()) == 0.0 and len(prev.calls) == 1 and tm._state["disabled"]
    fwd(*make_case())
    assert len(rec.calls) == 1 and len(prev.calls) == 2 and tm.status()["reasons"] == {"off": 1}


def test_install_chains_and_is_idempotent():
    _reset()
    prev = stock()
    mod = types.ModuleType("fake_backend")
    mod.FlashInferMLASparseSM90Impl = type("FlashInferMLASparseSM90Impl", (), {"forward_mqa": prev})
    tm.install(mod, env={"GLM_TRITON_MLA_ALLOW_DRIFT": "1"})
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    assert wrapped._glm_tmla_prev is prev
    tm.install(mod, env={"GLM_TRITON_MLA_ALLOW_DRIFT": "1"})
    assert mod.FlashInferMLASparseSM90Impl.forward_mqa is wrapped
    # a decode-only step goes to the stock binding it chained
    impl, q, kv, md, layer = make_case(prefills=0)
    wrapped(impl, q, kv, md, layer)
    assert len(prev.calls) == 1


def test_expected_hashes_match_image():
    if IMAGE_SRC is None:
        print("  (skipped: no vLLM source; set GLM_IMAGE_SRC or run inside the image)")
        return
    got = glm_prefill_hooks.hashes_at(IMAGE_SRC, tm.EXPECTED)
    assert got == tm.EXPECTED, got


def test_register_hooks_target():
    _reset()
    assert tm.register({"GLM_TRITON_MLA_PREFILL": "1"}) is True
    finder = glm_prefill_hooks._FINDER
    assert finder is not None and tm.TARGET in finder.hooks
    finder.hooks.pop(tm.TARGET)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)

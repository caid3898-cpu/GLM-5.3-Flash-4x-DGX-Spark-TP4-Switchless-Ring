"""CPU check for overlay/glm_kda_conv_split.py (GLM_KDA_CONV_SPLIT). Needs torch.

A fake kda module carries a per-channel causal conv with the stock conv's contract (x [dim, T] channel-last in,
empty_like(x) out, conv_states [N, dim, width-1] updated in place) and a fake chunk_kda that records the layout it
receives and returns a function of q / k / v. The caller code is the image's prefill path
(conv -> transpose -> split -> reshape -> chunk_kda). Checks: outputs and conv states bit-equal to stock; the chunk
sees contiguous q / k / v; a later wrapper on the chunk name is re-wrapped under us; an unexpected consumer gets a
filled carrier; the glm_ab switch and the install gate.

  python3 tests/test_glm_kda_conv_split.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
os.environ["GLM_KDA_CONV_SPLIT"] = "1"

import torch  # noqa: E402

import glm_kda_conv_split as cs  # noqa: E402

H, D = 4, 8
P = H * D
W = 4
torch.manual_seed(0)


def stock_conv(x, weight, bias=None, activation="silu", conv_states=None, has_initial_state=None,
               cache_indices=None, query_start_loc=None, metadata=None):
    out = torch.empty_like(x)                      # keeps x's channel-last strides (as in the image)
    dim, T = x.shape
    xs = x.float()
    for s in range(query_start_loc.numel() - 1):
        a, b = int(query_start_loc[s]), int(query_start_loc[s + 1])
        ci = int(cache_indices[s])
        init = conv_states[ci].float() if bool(has_initial_state[s]) else torch.zeros(dim, W - 1)
        seq = torch.cat([init, xs[:, a:b]], dim=1)            # [dim, W-1+L]
        y = sum(seq[:, j:j + (b - a)] * weight[:, j:j + 1].float() for j in range(W))
        if bias is not None:
            y = y + bias[:, None].float()
        out[:, a:b] = torch.nn.functional.silu(y).to(out.dtype)
        conv_states[ci] = seq[:, -(W - 1):].to(conv_states.dtype)
    return out


SEEN = []


def fake_chunk(q=None, k=None, v=None, **kw):
    SEEN.append(tuple(t.is_contiguous() for t in (q, k, v)))
    return (q.float() * 2 + k.float() - v.float()).to(q.dtype), None


def make_mod():
    m = types.ModuleType("fake_kda")
    m.causal_conv1d_fn = stock_conv
    m.chunk_kda_with_fused_gate = fake_chunk
    return m


def caller(mod, proj, weight, bias, conv_states, hist, csl, idx):
    """The image's prefill path in Glm5NextLinearAttention._forward (merged conv, then split views)."""
    qkv = proj[:, : 3 * P]
    qkv = mod.causal_conv1d_fn(qkv.transpose(0, 1), weight, bias, activation="silu", conv_states=conv_states,
                               has_initial_state=hist, cache_indices=idx, query_start_loc=csl,
                               metadata=None).transpose(0, 1)
    q, k, v = qkv.split(P, dim=-1)
    r = lambda t: t.reshape(1, -1, H, D)  # noqa: E731
    return mod.chunk_kda_with_fused_gate(q=r(q), k=r(k), v=r(v), beta=None)[0]


def inputs(T=37, bias=True):
    torch.manual_seed(T)
    proj = torch.randn(T, 3 * P + 20).to(torch.bfloat16)      # merged in_proj output (extra beta/f/g columns)
    weight = torch.randn(3 * P, W).to(torch.bfloat16)
    b = torch.randn(3 * P).to(torch.bfloat16) if bias else None
    states = torch.randn(3, 3 * P, W - 1).to(torch.bfloat16)
    csl = torch.tensor([0, 20, T])
    hist = torch.tensor([True, False])
    idx = torch.tensor([2, 0])
    return proj, weight, b, states, hist, csl, idx


def run(split, bias=True):
    SEEN.clear()
    mod = make_mod()
    if split:
        cs.S.update(pending=None, taken=0, substituted=0, filled=0)
        cs.install(mod)
    proj, weight, b, states, hist, csl, idx = inputs(bias=bias)
    out = caller(mod, proj, weight, b, states, hist, csl, idx)
    return out, states, list(SEEN), mod


def test_bit_equal_and_contiguous():
    for bias in (True, False):
        o0, s0, seen0, _ = run(False, bias)
        o1, s1, seen1, _ = run(True, bias)
        assert torch.equal(o0, o1) and torch.equal(s0, s1), "split conv differs from stock"
        assert seen0 == [(False, False, False)], seen0
        assert seen1 == [(True, True, True)], seen1
        assert cs.S["substituted"] == 1 and cs.S["filled"] == 0, cs.S


def test_outer_wrappers():
    o0, _, _, _ = run(False)
    for aware in (False, True):
        mod = make_mod()
        cs.S.update(pending=None, substituted=0, filled=0)
        cs.install(mod)
        later = []
        inner = mod.chunk_kda_with_fused_gate

        def other(*a, **kw):      # an overlay that wrapped the name after us and reads q itself
            if aware:
                kw["q"], kw["k"], kw["v"] = cs.take(kw["q"], kw["k"], kw["v"])
            later.append(kw["q"].is_contiguous())
            return inner(*a, **kw)
        other.__glm_conv_split_aware__ = aware
        mod.chunk_kda_with_fused_gate = other
        SEEN.clear()
        proj, weight, b, states, hist, csl, idx = inputs()
        out = caller(mod, proj, weight, b, states, hist, csl, idx)
        assert torch.equal(out, o0)
        assert later == [aware], (aware, later)   # aware: dense q; not aware: split not used (stock views)
        assert cs.S["filled"] == 0


def test_unexpected_consumer_gets_filled_carrier():
    mod = make_mod()
    cs.S.update(pending=None, substituted=0, filled=0)
    cs.install(mod)
    proj, weight, b, states, hist, csl, idx = inputs()
    ref_states = states.clone()
    qkv = mod.causal_conv1d_fn(proj[:, : 3 * P].transpose(0, 1), weight, b, activation="silu",
                               conv_states=states, has_initial_state=hist, cache_indices=idx,
                               query_start_loc=csl, metadata=None).transpose(0, 1)
    SEEN.clear()
    # chunk called with something else (e.g. a copy): the carrier must be made real first
    mod.chunk_kda_with_fused_gate(q=qkv[:, :P].clone().reshape(1, -1, H, D), k=qkv[:, P:2 * P].reshape(1, -1, H, D),
                                  v=qkv[:, 2 * P:].reshape(1, -1, H, D), beta=None)
    ref = stock_conv(proj[:, : 3 * P].transpose(0, 1), weight, b, conv_states=ref_states, has_initial_state=hist,
                     cache_indices=idx, query_start_loc=csl).transpose(0, 1)
    assert cs.S["filled"] == 1 and torch.equal(qkv, ref)


def test_switch_and_gates():
    fake = types.ModuleType("glm_ab")
    fake.ACTIVE, fake.KNOWN = True, {"GLM_KDA_CONV_SPLIT": "bool"}
    fake.truthy = lambda v: v is not None and str(v).strip().lower() not in cs._OFF
    cur = {"GLM_KDA_CONV_SPLIT": "0"}
    fake.env = lambda n, d=None: cur.get(n, d)
    sys.modules["glm_ab"] = fake
    try:
        _, _, seen, _ = run(True)
        assert seen == [(False, False, False)], seen   # variant off: stock layout
        cur["GLM_KDA_CONV_SPLIT"] = "1"
        _, _, seen, _ = run(True)
        assert seen == [(True, True, True)], seen
    finally:
        del sys.modules["glm_ab"]
    # a non-channel-last or odd-shaped call stays stock
    x = torch.randn(3 * P, 5)
    assert cs.eligible(x, torch.randn(3 * P, W), None, torch.zeros(1, 3 * P, 3)) == 0
    assert cs.eligible(x.t().contiguous().t(), torch.randn(3 * P + 1, W), None, torch.zeros(1, 3 * P, 3)) == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()

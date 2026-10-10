"""CPU check for overlay/glm_flashkda.py (GLM_FLASHKDA_PREFILL): dispatch, fallback and chaining, with fakes.
Needs torch (no GPU, no build).

A fake kda module carries a recording `chunk_kda_with_fused_gate` (the stock path) and `_cast_sigmoid`; the kernel
call (glm_flashkda.flashkda_call) is replaced by a recorder that returns the stock contract. Checks: a call is
taken only when ready and eligible; every ineligible shape / flag / dtype goes to prev with the caller's arguments;
the raw-beta tap is used when it matches and the logit fallback otherwise; the glm_ab per-call switch; chaining
(prev saved at install; a later wrapper stays outermost); the GLM_KDA_CONV_SPLIT take() protocol; prepare()'s
fail-closed path and TP agreement without a group; idempotent install.

  python3 tests/test_glm_flashkda.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
os.environ["GLM_FLASHKDA_PREFILL"] = "1"

import torch  # noqa: E402

import glm_flashkda as fk  # noqa: E402

fk.REQUIRE_CUDA = False
H, D = 4, 128
torch.manual_seed(0)
PREV_CALLS = []
FK_CALLS = []


def stock_chunk(*args, **kw):
    PREV_CALLS.append((args, kw))
    q = kw["q"] if "q" in kw else args[0]
    N = 1 if kw.get("cu_seqlens") is None else kw["cu_seqlens"].numel() - 1
    return torch.zeros_like(q), torch.zeros(N, q.shape[2], D, D)


def fake_fk_call(q, k, v, raw_g, beta_raw, A_log, g_bias, scale=None, initial_state=None,
                 output_final_state=True, cu_seqlens=None, lower_bound=-5.0):
    FK_CALLS.append(dict(q=q, k=k, v=v, g=raw_g, beta=beta_raw, h0=initial_state, cu=cu_seqlens, lb=lower_bound,
                         ofs=output_final_state))
    B, T, Hh, _ = q.shape
    N = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    return torch.ones(B, T, Hh, D, dtype=torch.bfloat16), (torch.ones(N, Hh, D, D) if output_final_state else None)


fk.flashkda_call = fake_fk_call


def make_mod():
    m = types.ModuleType("fake_kda")
    m.chunk_kda_with_fused_gate = stock_chunk
    m._cast_sigmoid = lambda x: x.float().sigmoid()
    return m


def caller(mod, T=64, lens=(20, 44), h=H, dtype=torch.bfloat16, **over):
    """The image's prefill call: q/k/v split views of one qkv buffer, beta through _cast_sigmoid."""
    qkv = torch.randn(T, 3 * h * D).to(dtype)
    q, k, v = (x.reshape(1, T, h, D) for x in qkv.split(h * D, dim=-1))
    beta_raw = torch.randn(1, T, h).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    kw = dict(q=q, k=k, v=v, raw_g=torch.randn(1, T, h, D).to(torch.bfloat16),
              beta=mod._cast_sigmoid(beta_raw.squeeze(0)).unsqueeze(0),
              A_log=torch.randn(1, 1, h, 1), g_bias=torch.randn(h * D),
              initial_state=torch.randn(len(lens), h, D, D), output_final_state=True, use_qk_l2norm_in_kernel=True,
              cu_seqlens=cu, safe_gate=True, lower_bound=-5.0)
    kw.update(over)
    return kw, beta_raw


def call(mod, kw, *args):
    PREV_CALLS.clear()
    FK_CALLS.clear()
    return mod.chunk_kda_with_fused_gate(*args, **kw)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def main():
    mod = make_mod()
    fk.install(mod)
    check(mod.chunk_kda_with_fused_gate is fk.dispatch and fk.S["prev"] is stock_chunk, "install wraps, prev saved")
    fk.install(mod)
    check(fk.S["prev"] is stock_chunk, "install is idempotent")
    check(getattr(fk.dispatch, "__glm_conv_split_aware__", False), "dispatch declares conv-split awareness")

    # not ready (self-test not run) -> stock
    fk.S["ready"] = False
    kw, _ = caller(mod)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS and fk.S["reasons"].get("not_ready"), "not ready -> prev")

    fk.S["ready"] = True
    kw, beta_raw = caller(mod)
    o, ht = call(mod, kw)
    c = FK_CALLS[0] if FK_CALLS else None
    check(c is not None and not PREV_CALLS, "eligible call taken")
    check(c["beta"].dtype == torch.bfloat16 and torch.equal(c["beta"], beta_raw), "raw beta from the tap (exact)")
    check(c["q"] is kw["q"] and c["cu"] is kw["cu_seqlens"] and c["h0"] is kw["initial_state"], "operands passed through")
    check(o.dtype == torch.bfloat16 and tuple(o.shape) == tuple(kw["q"].shape) and tuple(ht.shape) == (2, H, D, D),
          "return contract (o bf16 [B,T,H,D], state [N,H,D,D])")
    check(fk.S["tap"] is None, "tap consumed")

    # tap mismatch -> logit fallback (exact for moderate logits)
    kw, beta_raw = caller(mod)
    fk.S["tap"] = None
    n0 = fk.S["logit_beta"]
    call(mod, kw)
    check(fk.S["logit_beta"] == n0 + 1 and torch.equal(FK_CALLS[0]["beta"], beta_raw), "logit fallback recovers bf16 beta")

    # ineligible calls -> prev, arguments unchanged
    bad = {
        "no_l2norm": dict(use_qk_l2norm_in_kernel=False),
        "gate": dict(safe_gate=False),
        "gate_lb": dict(lower_bound=-7.0),
        "gate_none": dict(lower_bound=None),
        "extra_kwargs": dict(chunk_indices=None),
        "initial_state": dict(initial_state=torch.randn(3, H, D, D)),
        "cu_seqlens": dict(cu_seqlens=torch.tensor([0, 64], dtype=torch.float32)),
    }
    for name, over in bad.items():
        kw, _ = caller(mod, **over)
        call(mod, kw)
        check(len(PREV_CALLS) == 1 and not FK_CALLS and PREV_CALLS[0][1] is not None, f"ineligible ({name}) -> prev")
    kw, _ = caller(mod, dtype=torch.float16)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "ineligible (fp16 q/k/v) -> prev")
    kw, _ = caller(mod)
    kw["q"] = kw["q"][..., :64]
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "ineligible (head dim != 128) -> prev")
    kw, _ = caller(mod)
    kw["A_log"] = kw["A_log"].to(torch.bfloat16)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "ineligible (bf16 A_log) -> prev")
    kw, _ = caller(mod)
    kw2 = {k: v for k, v in kw.items() if k != "q"}
    call(mod, kw2, kw["q"])
    check(len(PREV_CALLS) == 1 and PREV_CALLS[0][0][0] is kw["q"] and not FK_CALLS, "positional call -> prev unchanged")
    kw, _ = caller(mod, cu_seqlens=None, initial_state=torch.randn(1, H, D, D))
    kw["q"] = kw["q"].expand(2, -1, -1, -1)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "ineligible (batch 2) -> prev")
    os.environ["GLM_FLASHKDA_MIN_T"] = "128"
    kw, _ = caller(mod)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "T below GLM_FLASHKDA_MIN_T -> prev")
    del os.environ["GLM_FLASHKDA_MIN_T"]
    kw, _ = caller(mod, initial_state=None, cu_seqlens=None, lens=(64,))
    call(mod, kw)
    check(len(FK_CALLS) == 1 and FK_CALLS[0]["h0"] is None, "no initial state, no cu_seqlens -> taken")
    kw, _ = caller(mod, output_final_state=False)
    o, ht = call(mod, kw)
    check(len(FK_CALLS) == 1 and ht is None, "output_final_state=False -> (o, None)")

    # glm_ab armed: the variant's value decides per call
    ab = types.ModuleType("glm_ab")
    ab.ACTIVE, ab.KNOWN = True, {"GLM_FLASHKDA_PREFILL": "bool"}
    ab.VAL = "0"
    ab.env = lambda name, default=None: ab.VAL if name == "GLM_FLASHKDA_PREFILL" else default
    ab.truthy = lambda v: v is not None and str(v).strip().lower() not in ("", "0", "off", "false", "no")
    sys.modules["glm_ab"] = ab
    try:
        kw, _ = caller(mod)
        call(mod, kw)
        check(len(PREV_CALLS) == 1 and not FK_CALLS and fk.S["tap"] is None, "glm_ab variant 0 -> prev (no tap)")
        ab.VAL = "1"
        kw, beta_raw = caller(mod)
        call(mod, kw)
        check(len(FK_CALLS) == 1 and torch.equal(FK_CALLS[0]["beta"], beta_raw), "glm_ab variant 1 -> FlashKDA")
    finally:
        del sys.modules["glm_ab"]

    # chaining: an earlier wrapper is our prev; a later wrapper stays outermost and still reaches us
    mod2 = make_mod()
    earlier = []

    def earlier_wrapper(*a, **k):
        earlier.append(1)
        return stock_chunk(*a, **k)
    mod2.chunk_kda_with_fused_gate = earlier_wrapper
    saved = dict(fk.S)
    fk.S["prev"] = None
    fk.install(mod2)
    check(fk.S["prev"] is earlier_wrapper, "prev = the earlier wrapper")
    later = []
    inner = mod2.chunk_kda_with_fused_gate

    def later_wrapper(*a, **k):
        later.append(1)
        return inner(*a, **k)
    mod2.chunk_kda_with_fused_gate = later_wrapper
    kw, _ = caller(mod2, safe_gate=False)
    call(mod2, kw)
    check(later and earlier and len(PREV_CALLS) == 1, "ineligible: later -> dispatch -> earlier -> stock")
    later.clear()
    earlier.clear()
    kw, _ = caller(mod2)
    call(mod2, kw)
    check(later and not earlier and len(FK_CALLS) == 1, "eligible: later -> dispatch -> FlashKDA")
    fk.S.update(saved)

    # GLM_KDA_CONV_SPLIT protocol: take() substitutes the dense q/k/v before anything reads them
    cs = types.ModuleType("glm_kda_conv_split")
    dense = {}

    def take(q, k, v):
        d = tuple(t.contiguous().clone() for t in (q, k, v))
        dense["d"] = d
        return d
    cs.take = take
    sys.modules["glm_kda_conv_split"] = cs
    try:
        kw, _ = caller(mod)
        call(mod, kw)
        check(FK_CALLS and FK_CALLS[0]["q"] is dense["d"][0] and FK_CALLS[0]["v"] is dense["d"][2],
              "conv-split take(): FlashKDA gets the dense tensors")
        kw, _ = caller(mod, safe_gate=False)
        call(mod, kw)
        check(PREV_CALLS and PREV_CALLS[0][1]["k"] is dense["d"][1], "conv-split take(): prev gets them too")
    finally:
        del sys.modules["glm_kda_conv_split"]

    # prepare(): no KDA layers -> no collective, unchanged; a failing build/test -> OFF (fail closed)
    fk.S["ready"] = True
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    check(fk.prepare(model) is True and fk.S["ready"], "prepare without KDA layers leaves the flag alone")

    class Glm5NextLinearAttention(torch.nn.Module):
        pass
    model = torch.nn.Sequential(Glm5NextLinearAttention())
    orig_load = fk.load_ext

    def boom():
        raise RuntimeError("nvcc failed")
    fk.load_ext = boom
    try:
        check(fk.prepare(model, prev=stock_chunk) is False and not fk.S["ready"] and "nvcc" in fk.S["detail"],
              "build failure -> OFF, boot continues")
    finally:
        fk.load_ext = orig_load
    orig_st = fk.self_test
    fk.load_ext = lambda: None
    fk.self_test = lambda layer, prev: (True, "fake pass")
    try:
        check(fk.prepare(model, prev=stock_chunk) is True and fk.S["ready"], "self-test pass -> ON")
        fk.self_test = lambda layer, prev: (False, "rel_o=1e0")
        check(fk.prepare(model, prev=stock_chunk) is False and not fk.S["ready"], "self-test fail -> OFF")
    finally:
        fk.load_ext, fk.self_test = orig_load, orig_st
    check(fk.agree(True) and not fk.agree(False), "agree() without a TP group is local")
    kw, _ = caller(mod)
    call(mod, kw)
    check(len(PREV_CALLS) == 1 and not FK_CALLS, "after a failed self-test every call is stock")
    print("PASS")


if __name__ == "__main__":
    main()

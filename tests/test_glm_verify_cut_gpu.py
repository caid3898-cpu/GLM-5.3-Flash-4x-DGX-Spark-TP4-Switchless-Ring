"""GPU unit test for overlay/glm_verify_cut.py (run inside the vLLM image, GPU required, fleet stopped).

docker run --rm --gpus all --memory 8g --entrypoint python3 -e PYTHONPATH=/overlay/overlay -e GLM_VERIFY_CUT=conf:0.1 \
    -v ~/glm53-flash-4x-spark:/overlay:ro <image> /overlay/tests/test_glm_verify_cut_gpu.py
Checks the two Triton kernels against the pure references and that the sitecustomize hooks land on the image's
classes. Adaptive verify cut = port of the ds41 DSV41_VERIFY_CAP=conf:0.1 dead-row remap.
"""
import random

import torch

import glm_verify_cut as g

dev = torch.device("cuda")
torch.manual_seed(0)
random.seed(0)

# ---- live kernel: [R, K, TOPK] selector scores -> live[idx_mapping[r]]
R, K, TOPK, MAXR = 13, 7, 16, 32
for scale in (0.5, 2.0, 6.0):
    scores = (torch.randn(MAXR, K, TOPK, device=dev) * scale).float()
    idx = torch.tensor(random.sample(range(MAXR), R), dtype=torch.int32, device=dev)
    for thr in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        live = torch.full((MAXR,), 999, dtype=torch.int32, device=dev)
        hist = torch.zeros(64, dtype=torch.int64, device=dev)
        g.launch_live(scores, idx, live, hist, R, thr)
        ref = g.live_lengths(g.selector_conf(scores[:R]), thr)
        got = live[idx.long()]
        assert torch.equal(got.cpu(), ref.cpu()), (scale, thr, got.tolist(), ref.tolist())
        untouched = torch.ones(MAXR, dtype=torch.bool)
        untouched[idx.long().cpu()] = False
        assert (live.cpu()[untouched] == 999).all()
        assert int(hist.sum()) == R
        if thr == 0.0:
            assert (ref == K).all()
print("live kernel OK")

# ---- rows kernel: mixed batch (decode k=3/5/7, a prefill chunk, a k=0 decode) + graph padding
for trial in range(200):
    n_req = random.randint(1, 12)
    nds = [random.choice([0, 3, 5, 7]) for _ in range(n_req)]
    qlens = [nd + 1 if nd > 0 or random.random() < 0.5 else random.randint(2, 40) for nd in nds]
    qsl = [0]
    for q in qlens:
        qsl.append(qsl[-1] + q)
    cu = [0]
    for nd in nds:
        cu.append(cu[-1] + nd + 1)
    n_tok = qsl[-1]
    n_pad = n_tok + random.choice([0, 0, 1, 3, 17])
    maxr = 32
    idx = random.sample(range(maxr), n_req)
    live_h = [random.randint(0, 8) for _ in range(maxr)]
    row_src = torch.full((600,), -7, dtype=torch.int64, device=dev)
    dead = torch.full((600,), 9, dtype=torch.uint8, device=dev)
    count = torch.zeros(1, dtype=torch.int64, device=dev)
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
    g.launch_rows(row_src, dead, t(qsl), t(cu), t(idx), t(live_h), count, n_req, n_tok, n_pad)
    ref_src, ref_dead = g.rows_reference(qsl, cu, idx, live_h, n_req, n_tok, n_pad)
    assert row_src[:n_pad].tolist() == ref_src, (trial, row_src[:n_pad].tolist(), ref_src)
    assert [bool(x) for x in dead[:n_pad].tolist()] == ref_dead, trial
    assert int(count) == sum(ref_dead)
    assert (row_src[n_pad:] == -7).all() and (dead[n_pad:] == 9).all()
    # n_req=0 path (no drafts): identity + padding only
    g.launch_rows(row_src, dead, t(qsl), t(cu), t(idx), t(live_h), count, 0, n_tok, n_pad)
    assert row_src[:n_pad].tolist() == [r if r < n_tok else 0 for r in range(n_pad)]
    assert not dead[:n_pad].any()
print("rows kernel OK")

# ---- the row gather reproduces the anchor row on dead rows (applied to the MoE input)
logits = torch.randn(40, 288, device=dev)
g._alloc(64, 8, dev)
g.launch_rows(g.S.row_src, g.S.dead_u8, t([0, 8, 12]), t([0, 8, 12]), t([0, 1]),
              torch.tensor([2, 7] + [0] * 6, dtype=torch.int32, device=dev), g.S.dead_count, 2, 12, 16)
got = logits[:16].index_select(0, g.S.row_src[:16])
assert torch.equal(got[3:8], logits[0:1].expand(5, -1)) and torch.equal(got[:3], logits[:3])
assert torch.equal(got[8:12], logits[8:12]) and torch.equal(got[12:16], logits[0:1].expand(4, -1))
assert g.S.dead[:16].nonzero().flatten().tolist() == [3, 4, 5, 6, 7]
print("gather OK")

# ---- hooks land on the image's classes (sitecustomize with GLM_VERIFY_CUT set)
import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as d2  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rs  # noqa: E402
import vllm.v1.worker.gpu.model_runner as mr  # noqa: E402

assert "_patch_dflash2" in d2.DFlash2Speculator.propose.__qualname__, d2.DFlash2Speculator.propose.__qualname__
assert "_patch_rejection" in rs.RejectionSampler.__call__.__qualname__
assert "_patch_runner" in mr.GPUModelRunner.load_model.__qualname__
print("hooks OK: dflash2 propose, rejection sampler, model runner")
try:
    import vllm.models.glm5next.nvidia.model as gm  # noqa: E402
    assert "_patch_moe" in gm.Glm5NextMoE.forward.__qualname__
    print("hooks OK: Glm5NextMoE.forward")
except ImportError as exc:
    print("glm5next model import skipped:", exc)
print("ALL OK")

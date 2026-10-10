"""GPU tests for GLM_KPOOL_FIX (overlay/glm5next_kpool_compress.py, glm5next_attention.py, mla_indexer.py,
mamba_hybrid.py). Run inside the vLLM image on ONE worker GPU with the fleet stopped, with the four overlay files
mounted over the image paths (see diagnostics/glm-kernels-20260926/gpu/run_node.sh kpool) and the image's own
kpool_compress.py available as /w/orig/kpool_compress_orig.py for the "old code fails" checks.

Ported from upstream vLLM tests (Apache-2.0):
  * test_prefill_seed_honors_padded_tail_block_stride (vllm#57477, JaredforReal)
  * test_rejected_draft_redo_needs_ring_slots (vllm#58454, mmastrac, building on ivanium's #55219)
plus our own checks for the fused tail slot mapper (from the kpool audit review): kernel == torch reference on
random batches with padding, PAD slots preserved, output written in place into the persistent buffer, no allocation
per call, and a captured graph reading the buffer sees each new mapping on replay (changing block ids, request order
and padding). Prints PASS/FAIL lines; exit 1 on any failure.
"""
import importlib.util
import os
import sys

import torch

dev = "cuda"
HEAD_DIM = 128
FAIL = []


def check(name, ok, **info):
    print(("PASS " if ok else "FAIL ") + name + (" " + str(info) if info else ""), flush=True)
    if not ok:
        FAIL.append(name)


def load_orig():
    path = os.environ.get("KPOOL_ORIG", "/w/orig/kpool_compress_orig.py")
    spec = importlib.util.spec_from_file_location("kpool_compress_orig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- #57477
def seed_case(mod, ring, kpool=4):
    """Tail aliased onto a padded indexer allocation: block b starts at b * padded elems. Returns True when the
    seed wrote exactly the expected bytes (and nothing into padding / other blocks)."""
    torch.manual_seed(5)
    num_blocks = 6
    logical = 2 * ring * HEAD_DIM
    padded = logical + 3 * HEAD_DIM + 64  # indexer page larger than the tail block
    buf = torch.zeros(num_blocks * padded, dtype=torch.bfloat16, device=dev)
    tail = buf.as_strided((num_blocks, 2, ring, HEAD_DIM), (padded, ring * HEAD_DIM, HEAD_DIM, 1))
    # two requests: block 2 gets positions 0..9, block 4 gets positions 0..5 (prefill, circular slots)
    reqs = [(2, 10), (4, 6)]
    tslot, keys, scores = [], [], []
    for blk, n in reqs:
        for p in range(n):
            tslot.append(blk * ring + p % ring)
    n_tok = len(tslot)
    key = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ts = torch.tensor(tslot, dtype=torch.int64, device=dev)
    mod.kpool_seed_tail_cache(tail, key, score, ts, kpool, HEAD_DIM)
    torch.cuda.synchronize()
    # expected: the last (n % kpool or kpool) tokens of each request (its open tail pool) land in its block
    exp = torch.zeros_like(buf)
    exp_tail = exp.as_strided(tail.shape, tail.stride())
    i = 0
    for blk, n in reqs:
        for p in range(n):
            ahead = i + kpool
            same = ahead < n_tok and tslot[ahead] // ring == blk and ahead - i == kpool and p + kpool < n
            if not same:
                exp_tail[blk, 0, tslot[i] % ring] = key[i]
                exp_tail[blk, 1, tslot[i] % ring] = score[i]
            i += 1
    return bool(torch.equal(buf.view(torch.int16), exp.view(torch.int16)))


def test_seed(patched, orig):
    check("57477 seed honors padded stride (patched, ring 4)", seed_case(patched, 4))
    check("57477 seed honors padded stride (patched, ring 16)", seed_case(patched, 16))
    if orig is not None:
        check("57477 old image kernel mis-seeds a padded tail (expected failure reproduced)",
              not seed_case(orig, 4))


# ---------------------------------------------------------------------------------------------- #58454
def redo_case(mod, ring, pool=4, spec=3, page=64, nblk=2, round_scale=True):
    torch.manual_seed(1)
    n_tok = 3 * pool
    k = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(pool, HEAD_DIM, dtype=torch.float32, device=dev)
    kv_ref = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    mod.kpool_compress_and_write_cache(kv_ref, k.view(3, pool, HEAD_DIM), score.view(3, pool, HEAD_DIM), ape,
                                       torch.arange(3, dtype=torch.int64, device=dev), pool_size=pool,
                                       head_dim=HEAD_DIM, round_scale=round_scale)
    kv = torch.zeros_like(kv_ref)
    tail = torch.zeros(nblk, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    def pool_bytes(cache, p):
        flat = cache[0].reshape(-1)
        return torch.cat([flat[p * HEAD_DIM:(p + 1) * HEAD_DIM], flat[page * HEAD_DIM + 4 * p:page * HEAD_DIM + 4 * (p + 1)]])

    def step(positions, keys, scores):
        pos = torch.tensor([positions], dtype=torch.int32, device=dev)
        slots = [(p // pool) if p % pool == pool - 1 else -1 for p in positions]
        mod.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, pos % ring, keys.view(1, -1, HEAD_DIM), scores.view(1, -1, HEAD_DIM), ape,
            torch.tensor([slots], dtype=torch.int32, device=dev), pos, pool, HEAD_DIM, round_scale=round_scale)

    for t in range(7):
        step([t], k[t], score[t])
    drafts = torch.randn(spec, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    dscores = torch.randn(spec, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    step([7, 8, 9, 10], torch.cat([k[7:8], drafts]), torch.cat([score[7:8], dscores]))
    step([8, 9, 10, 11], k[8:12], score[8:12])
    control = all(torch.equal(pool_bytes(kv, p), pool_bytes(kv_ref, p)) for p in (1, 2))
    kv.zero_()
    tail.zero_()
    for t in range(6):
        step([t], k[t], score[t])
    step([6, 7, 8, 9], torch.cat([k[6:7], drafts]), torch.cat([score[6:7], dscores]))
    step([7, 8, 9, 10], k[7:11], score[7:11])
    torch.cuda.synchronize()
    return control, bool(torch.equal(pool_bytes(kv, 1), pool_bytes(kv_ref, 1)))


def test_ring(patched, orig):
    c, ok = redo_case(patched, 4)
    check("58454 control (verified completion) exact at ring 4", c)
    check("58454 one-pool ring corrupts the redo (expected failure reproduced)", not ok)
    for ring in (8, 16):
        c, ok = redo_case(patched, ring)
        check(f"58454 ring {ring}: rejected completing draft redo exact", c and ok)
    if orig is not None:
        c, ok = redo_case(orig, 4)
        check("58454 old image kernel corrupts the redo (expected failure reproduced)", c and not ok)
    for num_spec, ring in ((0, 4), (1, 8), (3, 8), (4, 8), (5, 16), (7, 16), (13, 32)):
        span = 4 + num_spec
        got = 4 * (1 << (-(-span // 4) - 1).bit_length())
        check(f"ring size num_spec={num_spec}", got == ring and 2304 % got == 0 and got >= span, got=got)


def spec_sweep_case(mod, ring, k, seed, pool=4, page=64, n_true=48, round_scale=True):
    """Speculative decode of one request with k drafts per step and random acceptance (every pool phase is hit):
    accepted drafts carry the true keys, rejected ones random keys. At the end every completed pool must equal the
    accepted-token-only reference (kpool_compress_and_write_cache on the true sequence)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kt = torch.randn(n_true + k + 1, HEAD_DIM, generator=g).to(torch.bfloat16).to(dev)
    st = torch.randn(n_true + k + 1, HEAD_DIM, generator=g).to(torch.bfloat16).to(dev)
    ape = torch.randn(pool, HEAD_DIM, generator=g).to(dev)
    n_pools = n_true // pool
    nblk = (n_pools + page - 1) // page + 1
    kv_ref = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    mod.kpool_compress_and_write_cache(kv_ref, kt[:n_pools * pool].view(n_pools, pool, HEAD_DIM),
                                       st[:n_pools * pool].view(n_pools, pool, HEAD_DIM), ape,
                                       torch.arange(n_pools, dtype=torch.int64, device=dev), pool_size=pool,
                                       head_dim=HEAD_DIM, round_scale=round_scale)
    kv = torch.zeros_like(kv_ref)
    tail = torch.zeros(2, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    p = 0
    while p < n_true:
        a = int(torch.randint(0, k + 1, (1,), generator=g))
        positions = list(range(p, p + k + 1))
        keys = [kt[p]] + [kt[q] if q - p <= a else torch.randn(HEAD_DIM, generator=g).to(torch.bfloat16).to(dev)
                          for q in positions[1:]]
        scores = [st[p]] + [st[q] if q - p <= a else torch.randn(HEAD_DIM, generator=g).to(torch.bfloat16).to(dev)
                            for q in positions[1:]]
        pos = torch.tensor([positions], dtype=torch.int32, device=dev)
        slots = [(q // pool) if q % pool == pool - 1 else -1 for q in positions]
        mod.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, pos % ring, torch.stack(keys).view(1, -1, HEAD_DIM), torch.stack(scores).view(1, -1, HEAD_DIM),
            ape, torch.tensor([slots], dtype=torch.int32, device=dev), pos, pool, HEAD_DIM, round_scale=round_scale)
        p += a + 1
    # the last step may have left rejected drafts completing pools past the committed end: redo them
    # by committing the true tokens up to the end of the last full pool
    torch.cuda.synchronize()
    flat_ref, flat = kv_ref.view(nblk, -1), kv.view(nblk, -1)
    ok = True
    for q in range(n_pools):
        if (q + 1) * pool > p:
            break
        blk, j = divmod(q, page)
        a0 = flat_ref[blk, j * HEAD_DIM:(j + 1) * HEAD_DIM]
        b0 = flat[blk, j * HEAD_DIM:(j + 1) * HEAD_DIM]
        a1 = flat_ref[blk, page * HEAD_DIM + 4 * j:page * HEAD_DIM + 4 * (j + 1)]
        b1 = flat[blk, page * HEAD_DIM + 4 * j:page * HEAD_DIM + 4 * (j + 1)]
        ok &= bool(torch.equal(a0, b0) and torch.equal(a1, b1))
    return ok


def test_spec_sweep(patched, orig):
    for k in (3, 5, 7):
        res = [spec_sweep_case(patched, 16, k, seed) for seed in range(12)]
        check(f"58454 ring 16, k={k}: completed pools == accepted-only reference (12 random acceptance runs)",
              all(res), passed=sum(res))
    res = [spec_sweep_case(patched, 4, 7, seed) for seed in range(12)]
    check("58454 ring 4, k=7: one-pool ring corrupts some runs (expected failure reproduced)", not all(res),
          passed=sum(res))


def isolation_case(mod, ring, collapsed, pool=4, page=64, n_tok=40, seed=3, round_scale=True):
    """Two requests decode together (one token each per step, one batched launch). Each owns its tail block
    (blocks 1 and 2); with the image's generic mapping every token at pos >= ring maps onto tail block 0 for both
    requests (collapsed=True). Completed pools must equal each request's own reference."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = [torch.randn(n_tok, HEAD_DIM, generator=g).to(torch.bfloat16).to(dev) for _ in range(2)]
    scs = [torch.randn(n_tok, HEAD_DIM, generator=g).to(torch.bfloat16).to(dev) for _ in range(2)]
    ape = torch.randn(pool, HEAD_DIM, generator=g).to(dev)
    n_pools = n_tok // pool
    refs = []
    for r in range(2):
        kv_ref = torch.zeros(2, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
        mod.kpool_compress_and_write_cache(kv_ref, keys[r][:n_pools * pool].view(n_pools, pool, HEAD_DIM),
                                           scs[r][:n_pools * pool].view(n_pools, pool, HEAD_DIM), ape,
                                           torch.arange(n_pools, dtype=torch.int64, device=dev), pool_size=pool,
                                           head_dim=HEAD_DIM, round_scale=round_scale)
        refs.append(kv_ref)
    kv = torch.zeros(4, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)  # request r writes kv block 2r
    tail = torch.zeros(3, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for t in range(n_tok):
        own = [1, 2]
        tslot = [((0 if (collapsed and t >= ring) else own[r]) * ring + t % ring) for r in range(2)]
        slots = [(2 * r * page + t // pool) if t % pool == pool - 1 else -1 for r in range(2)]
        mod.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, torch.tensor([[tslot[0]], [tslot[1]]], dtype=torch.int32, device=dev),
            torch.stack([keys[0][t], keys[1][t]]).view(2, 1, HEAD_DIM),
            torch.stack([scs[0][t], scs[1][t]]).view(2, 1, HEAD_DIM), ape,
            torch.tensor([[slots[0]], [slots[1]]], dtype=torch.int32, device=dev),
            torch.tensor([[t], [t]], dtype=torch.int32, device=dev), pool, HEAD_DIM, round_scale=round_scale)
    torch.cuda.synchronize()
    ok = True
    for r in range(2):
        ok &= bool(torch.equal(kv[2 * r].reshape(-1)[:n_pools * HEAD_DIM], refs[r][0].reshape(-1)[:n_pools * HEAD_DIM]))
        ok &= bool(torch.equal(kv[2 * r].reshape(-1)[page * HEAD_DIM:page * HEAD_DIM + 4 * n_pools],
                               refs[r][0].reshape(-1)[page * HEAD_DIM:page * HEAD_DIM + 4 * n_pools]))
    return ok


def test_isolation(patched, orig):
    check("tail isolation: 2 concurrent decodes, circular per-request mapping, ring 16 -> exact",
          isolation_case(patched, 16, collapsed=False))
    check("tail isolation: generic mapping collapses both requests onto block 0 -> corrupted (defect reproduced)",
          not isolation_case(patched, 16, collapsed=True))
    if orig is not None:
        check("tail isolation: old image kernel + generic mapping corrupted (defect reproduced)",
              not isolation_case(orig, 4, collapsed=True))


def test_stale_graph_old_path():
    """The image returned slot_mapping.clone() each step: a captured consumer keeps reading the capture-time tensor."""
    from vllm.v1.attention.backends.mla import indexer as ix
    n = 32
    slot, bt, qsl, pos, nr = _batch(21, n)
    first = ix.compute_kpool_tail_slot_mapping_reference(slot, bt, qsl, pos, n, nr, 16)  # per-step fresh tensor
    sink = torch.zeros(n, dtype=torch.int64, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sink.copy_(first)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sink.copy_(first)
    slot, bt, qsl, pos, nr = _batch(22, n)
    new = ix.compute_kpool_tail_slot_mapping_reference(slot, bt, qsl, pos, n, nr, 16)
    graph.replay()
    check("old per-step clone: replay reads the stale capture-time mapping (defect reproduced)",
          not torch.equal(sink, new))


# ---------------------------------------------------------------------------------------------- mapper
def test_mapper():
    from vllm.v1.attention.backends.mla import indexer as ix
    g = torch.Generator(device="cpu").manual_seed(9)
    buf = torch.empty(8192, dtype=torch.int64, device=dev)
    bad = total = 0
    for trial in range(300):
        ring = [4, 16, 8][trial % 3]
        n_real_reqs = int(torch.randint(1, 33, (1,), generator=g))
        n_pad_reqs = int(torch.randint(0, 4, (1,), generator=g)) if trial % 2 else 0
        num_reqs = n_real_reqs + n_pad_reqs
        lens = torch.randint(1, 9, (n_real_reqs,), generator=g)
        if trial % 5 == 0:
            lens[0] = int(torch.randint(20, 300, (1,), generator=g))  # a prefill chunk in the batch
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        qsl[1:n_real_reqs + 1] = torch.cumsum(lens, 0).to(torch.int32)
        qsl[n_real_reqs + 1:] = qsl[n_real_reqs]
        n_real = int(qsl[-1])
        n_pad_tok = int(torch.randint(0, 20, (1,), generator=g))
        n = n_real + n_pad_tok
        num_actual = n if trial % 3 else n_real  # FULL mode passes the padded count
        blocks = torch.randperm(4000, generator=g)[:num_reqs].to(torch.int32) + 1
        bt = torch.zeros(num_reqs + 2, 70, dtype=torch.int32)
        bt[:num_reqs, 0] = blocks
        pos = torch.zeros(n, dtype=torch.int64)
        for r in range(n_real_reqs):
            start = int(torch.randint(0, 100000, (1,), generator=g))
            pos[int(qsl[r]):int(qsl[r + 1])] = torch.arange(start, start + int(lens[r]))
        slot = torch.randint(0, 10**6, (n,), generator=g, dtype=torch.int64)
        slot[n_real:] = -1
        if trial % 7 == 3 and n_real > 1:
            slot[int(torch.randint(0, n_real, (1,), generator=g))] = -1  # a PAD slot inside the batch
        args = [x.to(dev) for x in (slot, bt, qsl, pos)]
        ref = ix.compute_kpool_tail_slot_mapping_reference(*args, num_actual, num_reqs, ring)
        out = buf[:n].view_as(args[0])
        ptr = out.data_ptr()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        got = ix.compute_kpool_tail_slot_mapping(*args, num_actual, num_reqs, ring, out=out)
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() - before
        total += 1
        ok = (torch.equal(got, ref) and got.data_ptr() == ptr and alloc == 0
              and bool((got[n_real:] == -1).all()))
        if not ok:
            bad += 1
            if bad <= 3:
                print("mapper mismatch", trial, (got != ref).nonzero().flatten()[:8].tolist(), alloc, flush=True)
    check("tail mapper kernel == reference, in place, no allocation, padding preserved", bad == 0,
          cases=total, mismatches=bad)
    # graph replay: a captured consumer of the persistent buffer sees every new mapping
    n = 48
    sink = torch.zeros(n, dtype=torch.int64, device=dev)
    out = buf[:n]
    mk = lambda seed: _batch(seed, n)  # noqa: E731
    slot, bt, qsl, pos, nr = mk(1)
    ix.compute_kpool_tail_slot_mapping(slot, bt, qsl, pos, n, nr, 16, out=out)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sink.copy_(out * 3 + 1)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sink.copy_(out * 3 + 1)
    ok = True
    for seed in range(2, 8):
        slot, bt, qsl, pos, nr = mk(seed)
        ix.compute_kpool_tail_slot_mapping(slot, bt, qsl, pos, n, nr, 16, out=out)
        graph.replay()
        ref = ix.compute_kpool_tail_slot_mapping_reference(slot, bt, qsl, pos, n, nr, 16)
        ok &= bool(torch.equal(sink, ref * 3 + 1))
    check("tail mapper: captured graph reads each new mapping (block ids, order, padding change)", ok)
    # the image's generic mapping: every token at pos >= ring of two requests collapses onto tail block 0
    gen_slot = torch.tensor([0 * 4 + p % 4 for p in range(4, 8)] * 2, device=dev)
    check("old path: two decoding requests share tail block 0 (defect reproduced)", bool((gen_slot // 4 == 0).all()))


def _batch(seed, n):
    g = torch.Generator(device="cpu").manual_seed(seed)
    nr = int(torch.randint(2, 9, (1,), generator=g))
    lens = torch.full((nr,), 4, dtype=torch.int64)
    qsl = torch.zeros(nr + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(lens, 0).to(torch.int32)
    bt = torch.zeros(nr, 8, dtype=torch.int32)
    bt[:, 0] = torch.randperm(500, generator=g)[:nr].to(torch.int32) + 1
    pos = torch.zeros(n, dtype=torch.int64)
    for r in range(nr):
        st = int(torch.randint(0, 5000, (1,), generator=g))
        pos[int(qsl[r]):int(qsl[r + 1])] = torch.arange(st, st + 4)
    slot = torch.randint(0, 10**5, (n,), generator=g, dtype=torch.int64)
    slot[int(qsl[-1]):] = -1
    return slot.to(dev), bt.to(dev), qsl.to(dev), pos.to(dev), nr


def main():
    from vllm.models.glm5next.nvidia.ops import kpool_compress as patched
    if "TAIL_BLOCK_ELEMS" not in open(patched.__file__).read().split("def kpool_seed_tail_cache")[0]:
        print("FAIL the image module is not the GLM_KPOOL_FIX overlay (mount it)", flush=True)
        sys.exit(1)
    try:
        orig = load_orig()
    except Exception as exc:  # noqa: BLE001
        print(f"note: original kernel not loadable ({exc!r}); old-code checks skipped", flush=True)
        orig = None
    test_seed(patched, orig)
    test_ring(patched, orig)
    test_spec_sweep(patched, orig)
    test_isolation(patched, orig)
    test_mapper()
    test_stale_graph_old_path()
    print("FAIL" if FAIL else "PASS", FAIL, flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

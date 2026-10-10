"""Raw-byte GPU qualification against the unmodified installed stock wrappers.

Exit 77 = unavailable dependencies/CUDA, never PASS. Full reports require all
M=1..256, two seeds, graphs, both dispatch modes and the mandatory edge suite.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def guard(t, offset, guards):
    import torch
    backing = torch.full((t.numel() + offset + 37,), 83, dtype=t.dtype, device=t.device)
    view = backing[offset:offset + t.numel()].view(t.shape)
    view.copy_(t)
    guards.append((backing, offset, t.numel()))
    return view


def check_guards(guards):
    import torch
    for backing, offset, size in guards:
        if not (torch.all(backing[:offset] == 83).item()
                and torch.all(backing[offset + size:] == 83).item()):
            raise AssertionError('Destination/input guard overwritten')


def edge(d, name):
    import torch
    r, x, c, p = (d[k] for k in ('r', 'x', 'c', 'p'))
    if name == 'signed_zero':
        r.zero_(); x.zero_(); c.zero_(); p.zero_()
        for t in (r, x, c, p):
            t.reshape(-1)[::2] = -0.0
    elif name == 'bf16_subnormal':
        values = torch.tensor([1, -32767, 127, -32641], dtype=torch.int16,
                              device='cuda').view(torch.bfloat16)
        r.zero_(); x.zero_()
        r.reshape(-1)[:4].copy_(values)
        x.reshape(-1)[:4].copy_(values)
    elif name == 'fp32_subnormal':
        values = torch.tensor([1, -2147483647, 8388607, -2139095041],
                              dtype=torch.int32, device='cuda').view(torch.float32)
        c.zero_(); p.zero_()
        c.reshape(-1)[:4].copy_(values)
        p.reshape(-1)[:4].copy_(values)
    elif name == 'overflow':
        r.fill_(torch.finfo(torch.bfloat16).max)
        x.fill_(torch.finfo(torch.bfloat16).max)
        c.fill_(1e20); p.fill_(1e20)
    elif name == 'infinity':
        r.reshape(-1)[:3] = torch.tensor([float('inf'), -float('inf'), 0.], device='cuda')
        x.reshape(-1)[:3] = torch.tensor([0., float('inf'), -float('inf')], device='cuda')
        c.zero_(); p.fill_(1.)
    elif name == 'nan_payload':
        r.reshape(-1)[:4].copy_(torch.tensor([0x7fc1, 0x7fff, -63, 0x7f81],
                               dtype=torch.int16, device='cuda').view(torch.bfloat16))
        c.reshape(-1)[:4].copy_(torch.tensor([0x7fc00001, 0x7fffffff, -4194303, 0x7f800001],
                               dtype=torch.int32, device='cuda').view(torch.float32))
    elif name == 'cancellation':
        r[:, 1] = -r[:, 0]
        r[:, 2, ::7] *= 32
        x[:, ::31] = 0
    else:
        raise ValueError(name)


def low_level(d, fp, packed, offset=0):
    import torch
    import mhc_fused as f
    from mhc_validation import compare, allocation_count
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    m = d['r'].shape[0]
    nt, tile = (1024, 4) if m < 128 else (512, 12)
    guards = []
    d = {k: guard(v, offset, guards) if k in ('r', 'x', 'c', 'p') else v
         for k, v in d.items()}
    fp = guard(fp, offset, guards)
    packed = guard(packed, offset, guards)
    snapshots = [t.clone() for t in (d['r'], d['x'], d['p'], d['c'], fp, packed)]
    ref = (torch.empty((1, m, 24), device='cuda'), torch.empty((1, m), device='cuda'))
    tk.hc_prenorm_gemm_tilelang(d['r'].view(m, -1), fp, *ref, 4096, 4, 24, nt, tile, 1)
    for w in (fp, packed):
        dst = tuple(guard(torch.empty_like(t), offset, guards) for t in ref)
        call = lambda: f.project(d['r'].view(m, -1), w, *dst, nt, tile)
        call()  # compilation/warmup excluded from allocation assertion
        before = allocation_count()
        call()
        if allocation_count() != before:
            raise AssertionError('project allocated tensor storage')
        compare(ref, dst, f'project M={m} offset={offset} dtype={w.dtype}')
    if m <= 16:
        splits, tile = (8, 2) if m < 8 else (4, 3)
        ref = (torch.empty((splits, m, 24), device='cuda'),
               torch.empty((splits, m), device='cuda'), torch.empty_like(d['r']))
        tk.mhc_fused_tilelang(d['c'], d['r'], d['p'].view(m, 4), d['x'],
                             fp.view(24, 4, 4096), *ref, 4, 4096, 24, 256, 256, tile, splits)
        for w in (fp, packed):
            dst = tuple(guard(torch.empty_like(t), offset, guards) for t in ref)
            call = lambda: f.post_project(d['c'], d['r'], d['p'], d['x'], w,
                                           *dst, tile, splits)
            call()
            before = allocation_count()
            call()
            if allocation_count() != before:
                raise AssertionError('post_project allocated tensor storage')
            compare(ref, dst, f'fused M={m} offset={offset} dtype={w.dtype}')
    compare(tuple(snapshots), (d['r'], d['x'], d['p'], d['c'], fp, packed), 'inputs unchanged')
    check_guards(guards)


def contracts(fp):
    import torch
    import mhc_fused as f
    from mhc_validation import inputs, compare
    d = inputs(1)
    x = d['r'].view(1, -1)
    out = torch.empty((1, 1, 24), device='cuda')
    sq = torch.empty((1, 1), device='cuda')
    project = lambda **kw: f.project(**(dict(x=x, weight=fp, out=out, sqrsum=sq,
                                            n_thr=1024, tile_n=4) | kw))
    for kw in (dict(out=out.to(torch.bfloat16)), dict(out=out.cpu()),
               dict(out=torch.empty((1, 1, 48), device='cuda')[..., ::2]),
               dict(sqrsum=sq.to(torch.bfloat16)), dict(x=x[:, ::2]),
               dict(x=x[:0]), dict(weight=fp[:1]), dict(tile_n=5),
               dict(n_thr=512, tile_n=12), dict(out=fp.view(-1)[:24].view(1, 1, 24)),
               dict(sqrsum=out.view(-1)[:1].view(1, 1))):
        rejects(lambda kw=kw: project(**kw))
    if torch.cuda.device_count() > 1:
        rejects(lambda: project(out=out.to('cuda:1')))
    po = torch.empty((8, 1, 24), device='cuda')
    ps = torch.empty((8, 1), device='cuda')
    rr = torch.empty_like(d['r'])
    post = lambda **kw: f.post_project(**(dict(comb=d['c'], residual=d['r'], post=d['p'],
                     x=d['x'], weight=fp, out=po, sqrsum=ps, residual_out=rr,
                     tile_n=2, splits=8) | kw))
    for kw in (dict(residual_out=rr[:, :, :1]), dict(residual_out=rr.float()),
               dict(residual_out=d['r']), dict(comb=d['c'].bfloat16()),
               dict(post=d['p'].bfloat16()), dict(x=d['x'].float()),
               dict(tile_n=5), dict(splits=3), dict(splits=4, tile_n=3),
               dict(residual=d['r'][:0]), dict(out=po.transpose(0, 1)),
               dict(sqrsum=po.reshape(-1)[:8].view(8, 1))):
        rejects(lambda kw=kw: post(**kw))
    bad = fp.clone(); bad[0, 0] = 1.0001
    rejects(lambda: f.pack_weight(bad))
    compare(fp, f.pack_weight(fp).float(), 'weight round trip')
    # Pack signed zeros and BF16 subnormals without losing their bits.
    special = fp.clone()
    special.view(-1)[:4].copy_(torch.tensor([0, -32768, 1, -32767],
                             device='cuda', dtype=torch.int16).view(torch.bfloat16).float())
    compare(special, f.pack_weight(special).float(), 'special weight bits')


def installed_metadata(model):
    import torch
    import mhc_fused as f
    weights = f._STATE['weights']
    w = model.hc_attn_fn
    for bad in (w.view(-1)[:16], w.view(torch.bfloat16), w.t()):
        if f._lookup_weight(weights, bad) is not None:
            raise AssertionError('Malformed view selected packed twin')
    if f._lookup_weight(weights, w.view(24, 4, 4096)) is None:
        raise AssertionError('Valid full weight view rejected')


def padded(weight, m, graph):
    import torch
    from mhc_validation import inputs, dispatch, compare, invoke, capture, mutate
    d = inputs(m)
    active = m - 1
    for k in ('r', 'x', 'p', 'c'):
        d[k][active:].zero_()
    with dispatch(False):
        clean = invoke(d, weight)
    with dispatch(True):
        compare(clean, invoke(d, weight), f'padded clean M={m}')
    for k in ('r', 'x', 'p', 'c'):
        d[k][active:].fill_(float('nan'))
    with dispatch(False):
        poisoned = invoke(d, weight)
    with dispatch(True):
        got = invoke(d, weight)
        compare(poisoned, got, f'padded poison M={m}')
        if graph:
            g, gout = capture(lambda: invoke(d, weight))
            mutate(d)
            with dispatch(False):
                changed = invoke(d, weight)
            g.replay()
            compare(changed, gout, f'padded replay M={m}')
    # Compare only active rows, always at the SAME physical M/dispatch.
    for aa, bb in zip(clean, poisoned, strict=True):
        for a, b in zip(aa, bb, strict=True):
            compare(a[:active], b[:active], f'inactive row isolation M={m}')


def run(args):
    import torch
    import mhc_fused as f
    import mhc_validation as v
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    start, end = map(int, args.tokens.split(':'))
    if not 1 <= start <= end <= 256:
        raise ValueError('tokens must be a range within 1:256')
    tokens = list(range(start, end + 1))
    seeds = list(dict.fromkeys(map(int, args.seeds.split(','))))
    if not seeds:
        raise ValueError('At least one seed required')
    report = dict(schema=v.SCHEMA, status='FAIL', native_deepgemm=bool(is_deep_gemm_supported()))
    # Compute provenance once, outside capture. No monkeypatched stock oracle.
    report['fingerprint'] = v.fingerprint()
    for seed in seeds:
        torch.manual_seed(seed)
        fp = (torch.load(args.weight, weights_only=True, map_location='cuda') if args.weight
              else torch.randn((24, 16384), device='cuda', dtype=torch.bfloat16).float())
        packed = f.pack_weight(fp)
        contracts(fp)
        for m in tokens:
            low_level(v.inputs(m), fp, packed)
        for m in v.BOUNDARIES:
            for offset in (1, 3, 32):
                low_level(v.inputs(m), fp, packed, offset)
            for name in v.EDGES:
                d = v.inputs(m)
                edge(d, name)
                low_level(d, fp, packed, 1)
        model = torch.nn.Module()
        model.register_parameter('hc_attn_fn', torch.nn.Parameter(fp, requires_grad=False))
        # Environment gate and missing-report gate must leave stock untouched.
        old = tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang
        os.environ.pop('GLM_MHC_FUSED', None)
        if f.install(model) is not False:
            raise AssertionError('Install is not default-off')
        os.environ['GLM_MHC_FUSED'] = '1'
        os.environ.pop('GLM_MHC_QUALIFICATION', None)
        rejects(lambda: f.install(model), RuntimeError)
        if old != (tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang):
            raise AssertionError('Failed install modified dispatch')
        try:
            if not f._install_for_qualification(model):
                raise AssertionError('Test bootstrap did not install')
            installed_metadata(model)
            for mode in ('tilelang', 'native'):
                with v.branch(mode) as dg:
                    print(f'branch={mode} native_deepgemm={dg}', flush=True)
                    for m in tokens:
                        for norm_kind in ('none', 'bf16', 'bf16_strided', 'fp32_strided'):
                            d = v.inputs(m, norm_kind)
                            with v.dispatch(False):
                                try:
                                    ref = v.invoke(d, model.hc_attn_fn)
                                except Exception as exc:
                                    raise RuntimeError('Unmodified stock wrapper failed; do not '
                                                       'normalize n_splits/split_k in the oracle') from exc
                            with v.dispatch(True):
                                v.compare(ref, v.invoke(d, model.hc_attn_fn), f'{mode} M={m} {norm_kind}')
                                if args.graph:
                                    g, gout = v.capture(lambda: v.invoke(d, model.hc_attn_fn))
                                    v.replay_without_allocation(g)
                                    v.compare(ref, gout, 'initial graph')
                                    for _ in range(2):
                                        v.mutate(d)
                                        with v.dispatch(False):
                                            changed = v.invoke(d, model.hc_attn_fn)
                                        g.replay()
                                        v.compare(changed, gout, 'changed graph')
                                    del g, gout
                        print(f'PASS seed={seed} branch={mode} M={m}', flush=True)
                    for m in (7, 8, 16, 17, 32, 127, 128, 256):
                        padded(model.hc_attn_fn, m, args.graph)
                    # Exceptional wrapper/epilogue coverage is separate from finite tests.
                    for m in v.BOUNDARIES:
                        for name in v.EDGES:
                            d = v.inputs(m)
                            edge(d, name)
                            with v.dispatch(False):
                                ref = v.invoke(d, model.hc_attn_fn)
                            with v.dispatch(True):
                                v.compare(ref, v.invoke(d, model.hc_attn_fn), f'{name} wrapper M={m}')
            pristine = model.hc_attn_fn.detach().clone()
            with torch.no_grad():
                model.hc_attn_fn.add_(1)
            rejects(lambda: f._lookup_weight(f._STATE['weights'], model.hc_attn_fn), RuntimeError)
            rejects(lambda: f._install_for_qualification(model), RuntimeError)
            with torch.no_grad():
                model.hc_attn_fn.copy_(pristine)
        finally:
            v.uninstall_test()
    report['coverage'] = dict(tokens=tokens, seeds=seeds, graph=args.graph,
                              edges=list(v.EDGES), offsets=[1, 3, 32],
                              branches=['tilelang', 'native'], unmodified_stock=True)
    full = tokens == list(range(1, 257)) and len(seeds) >= 2 and args.graph
    report['status'] = 'PASS' if full else 'PARTIAL'
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    if full and args.report:
        # Exercise the public report gate, not only the internal test bootstrap.
        # The fresh install repacks the restored original parameter.
        os.environ['GLM_MHC_QUALIFICATION'] = str(Path(args.report).resolve())
        try:
            if not f.install(model) or not f.install(model):
                raise AssertionError('Public qualified install failed')
        finally:
            v.uninstall_test()
    print(f"{report['status']}: all requested RAW-BYTE tests passed; full qualification={full}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens', default='1:256')
    ap.add_argument('--seeds', default='51,819')
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--weight')
    ap.add_argument('--report', default='qualification.json')
    args = ap.parse_args()
    # Remove stale success BEFORE imports or tests can fail.
    if args.report:
        Path(args.report).write_text(json.dumps(dict(status='RUNNING')) + '\n')
    try:
        import torch
        import mhc_fused  # noqa: F401
        from vllm.model_executor.kernels.mhc import tilelang_kernels  # noqa: F401
    except ImportError as exc:
        print(f'UNTESTED: {exc}')
        if args.report:
            Path(args.report).write_text(json.dumps(dict(status='UNTESTED', reason=str(exc))) + '\n')
        return 77
    if not torch.cuda.is_available():
        print('UNTESTED: CUDA unavailable')
        if args.report:
            Path(args.report).write_text(json.dumps(dict(status='UNTESTED', reason='CUDA unavailable')) + '\n')
        return 77
    saved = {k: os.environ.get(k) for k in ('GLM_MHC_FUSED', 'GLM_MHC_QUALIFICATION')}
    try:
        return run(args)
    except Exception as exc:
        if args.report:
            Path(args.report).write_text(json.dumps(dict(status='FAIL', reason=repr(exc))) + '\n')
        raise
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    sys.exit(main())

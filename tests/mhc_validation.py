"""Shared qualification/benchmark utilities; never used inside a captured kernel."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

SCHEMA = 1
BOUNDARIES = (1, 4, 7, 8, 16, 17, 32, 127, 128, 256)
EDGES = ('signed_zero', 'bf16_subnormal', 'fp32_subnormal', 'overflow',
         'infinity', 'nan_payload', 'cancellation')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint():
    """Pin more than a Python kernel file: compiler/header/binary wheel contents.

    Hash all installed compiler distribution files except pyc/metadata. Include
    editable package trees too; refuse to qualify without the reduction header.
    This is evidence of provenance, not a proof of generated-code equivalence.
    """
    import torch
    import triton
    from vllm.model_executor.kernels.mhc import tilelang as mt, tilelang_kernels as tk
    import tilelang
    from vllm.utils import deep_gemm as dg
    root = Path(__file__).resolve().parent
    sources = {name: sha256(root / name) for name in
               ('mhc_fused.py', 'mhc_validation.py', 'test_mhc_fused.py', 'bench_mhc_fused.py')}
    sources.update({name: sha256(mod.__file__) for name, mod in
                    (('stock_wrapper', mt), ('stock_kernels', tk), ('deep_gemm_wrapper', dg))})
    compilers = {}
    for name, mod in (('tilelang', tilelang), ('triton', triton)):
        dist = importlib.metadata.distribution(name)
        files = set()  # populated from both wheel RECORD and package tree
        files.update(p.resolve() for p in Path(mod.__file__).resolve().parent.rglob('*')
                     if p.is_file())
        files.update(Path(dist.locate_file(p)).resolve() for p in (dist.files or ())
                     if Path(dist.locate_file(p)).is_file())
        files = sorted(p for p in files if p.suffix != '.pyc'
                       and '__pycache__' not in p.parts
                       and not any(part.endswith('.dist-info') for part in p.parts))
        if name == 'tilelang' and not any(p.name == 'reduce.h' for p in files):
            raise RuntimeError('Cannot fingerprint the installed TileLang reduction header')
        digest = hashlib.sha256()
        for p in files:
            # Absolute roots intentionally pin the serving installation as well.
            digest.update(str(p).encode())
            digest.update(sha256(p).encode())
        compilers[name] = dict(version=dist.version, files=len(files), sha256=digest.hexdigest())
    driver = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(schema=SCHEMA, sources=sources, compilers=compilers,
                torch=torch.__version__, cuda=torch.version.cuda, driver=driver,
                gpu=dict(name=prop.name, capability=list(torch.cuda.get_device_capability()),
                         sms=prop.multi_processor_count,
                         max_threads_per_multi_processor=prop.max_threads_per_multi_processor,
                         total_memory=prop.total_memory,
                         l2_cache_size=getattr(prop, 'L2_cache_size', None)),
                pass_configs=repr(tk.pass_configs), pdl=bool(tk.ENABLE_PDL),
                compiler_env={k: v for k, v in sorted(os.environ.items())
                              if k.startswith(('TRITON_', 'TILELANG_', 'TVM_', 'CUDA_'))
                              and k not in ('CUDA_VISIBLE_DEVICES',)})


def require_qualification(path):
    if not path:
        raise RuntimeError('GLM_MHC_QUALIFICATION must name a full PASS report; run RUN_ON_GPU.md')
    report = json.loads(Path(path).read_text())
    coverage = report.get('coverage', {})
    if (report.get('status') != 'PASS' or report.get('schema') != SCHEMA
            or coverage.get('tokens') != list(range(1, 257))
            or len(coverage.get('seeds', [])) < 2
            or coverage.get('graph') is not True
            or coverage.get('edges') != list(EDGES)
            or coverage.get('offsets') != [1, 3, 32]
            or coverage.get('branches') != ['tilelang', 'native']
            or coverage.get('unmodified_stock') is not True):
        raise RuntimeError('Incomplete or failed mHC qualification report')
    if report.get('fingerprint') != fingerprint():
        raise RuntimeError('mHC qualification provenance mismatch; rerun the full suite')
    return report


def compare(a, b, label='output'):
    import torch
    import mhc_fused as f
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not f.bit_equal(a, b):
            raise AssertionError(f'{label}: RAW-BYTE MISMATCH')
    else:
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f'{label}: output structure mismatch')
        for i, (aa, bb) in enumerate(zip(a, b, strict=True)):
            compare(aa, bb, f'{label}/{i}')


@contextlib.contextmanager
def branch(mode):
    # Force only the capability predicate, equally for stock and candidate.
    # Native exercises actual DeepGEMM when supported; never force unsupported DG.
    from vllm.utils import deep_gemm as dg
    old = dg.is_deep_gemm_supported
    if mode == 'tilelang':
        dg.is_deep_gemm_supported = lambda: False
    try:
        yield bool(dg.is_deep_gemm_supported())
    finally:
        dg.is_deep_gemm_supported = old


@contextlib.contextmanager
def dispatch(candidate):
    import mhc_fused as f
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    state = f._STATE
    if state is None:
        raise RuntimeError('Install before selecting dispatch')
    saved = tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang
    tk.hc_prenorm_gemm_tilelang = state['installed_pre' if candidate else 'pre']
    tk.mhc_fused_tilelang = state['installed_fused' if candidate else 'fused']
    try:
        yield
    finally:
        tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang = saved


def uninstall_test():
    import mhc_fused as f
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    if f._STATE is not None:
        tk.hc_prenorm_gemm_tilelang = f._STATE['pre']
        tk.mhc_fused_tilelang = f._STATE['fused']
        f._STATE = None


def inputs(m, norm_kind='bf16'):
    import torch
    kw = dict(device='cuda')
    r = torch.randn((m, 4, 4096), dtype=torch.bfloat16, **kw)
    x = torch.randn((m, 4096), dtype=torch.bfloat16, **kw)
    p = torch.rand((m, 4, 1), **kw)
    c = torch.rand((m, 4, 4), **kw) * .1
    scale = torch.randn(3, **kw) * .01
    base = torch.randn(24, **kw) * .1
    norm = None
    if norm_kind != 'none':
        dtype = torch.float32 if norm_kind == 'fp32_strided' else torch.bfloat16
        n = 8192 if norm_kind.endswith('strided') else 4096
        norm = torch.randn(n, dtype=dtype, **kw)
        if norm_kind.endswith('strided'):
            norm = norm[::2]
            if norm.is_contiguous():
                raise AssertionError('norm test must actually be strided')
    return dict(r=r, x=x, p=p, c=c, scale=scale, base=base, norm=norm)


def invoke(d, weight, whole=True):
    from vllm.model_executor.kernels.mhc import tilelang as mt
    common = dict(fn=weight, hc_scale=d['scale'], hc_base=d['base'], rms_eps=1e-6,
                  hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.,
                  sinkhorn_repeat=20, norm_weight=d['norm'], norm_eps=1e-6)
    fused = mt.mhc_fused_post_pre_tilelang(d['x'], d['r'], d['p'], d['c'], **common)
    return (mt.mhc_pre_tilelang(d['r'], **common), fused) if whole else fused


def mutate(d):
    d['r'].neg_()
    d['x'].mul_(.5)
    d['p'].mul_(.75)
    d['c'].neg_()
    d['scale'].mul_(.5)
    d['base'].add_(.125)
    if d['norm'] is not None:
        d['norm'].neg_()


def capture(call, warmup=3):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def allocation_count():
    import torch
    return torch.cuda.memory_stats()['allocation.all.allocated']


def replay_without_allocation(graph):
    import torch
    before = allocation_count()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if allocation_count() != before:
        raise AssertionError('Graph replay allocated a PyTorch CUDA tensor')

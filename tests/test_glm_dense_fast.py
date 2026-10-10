"""CPU checks for overlay/glm_dense_fast.py (GLM_DENSE_FAST / GLM_DENSE_FAST_PREFILL).

  python3 tests/test_glm_dense_fast.py                                        # pure-Python part only
  uv run --no-project --with torch python tests/test_glm_dense_fast.py        # + the torch (CPU) part

No GPU, no vLLM: the engine classes are small fakes with the call shapes of vLLM 487ecf187
(MarlinMxfp8LinearKernel / MarlinFP8ScaledMMLinearKernel.apply_weights(self, layer, x, bias), BaseModelLoader
.load_model(self, vllm_config, model_config, prefix)). The kernels themselves are checked on GB10 by
tests/test_glm_dense_fast_gpu.py (parity vs stock Marlin on real weights, CUDA-graph replay, timing).

Pure Python: the shipped table (schema, entry order, the speedup rules it was cut with, regenerated from the
source measurements when they are present), switch parsing, M buckets, workspace sizing, nvcc flag form, the fatal
config errors at register(), and the sitecustomize / glm_ab wiring (raw switch kinds, install union, hashed source).
torch (CPU): layer classification on fake Marlin buffers, the unpack index formulas vs an independent decoder of
the Marlin 8-bit layout, prepare() (self-test verdicts, build failure, exceptions, MIN agreement and the table-hash
check over a fake TP group, layer tagging), per-call dispatch by M (buckets, gaps, prefill threshold, capture guard,
bias / dtype fallbacks) including per-variant switching through glm_ab, the kernel-class wrappers and the loader
hook (target tagged, drafter left stock).
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.join(HERE, "..", "overlay")
sys.path.insert(0, OVERLAY)
# Optional: the directory holding the sweep measurements the table was cut from (not shipped).
DIAG = os.environ.get("GLM_SWEEP_DIR", "/nonexistent")
SRC_BEST = os.path.join(DIAG, "glm-cuda-kernels-20260928/remote/best_configs.json")
SRC_RES = os.path.join(DIAG, "glm-cuda-kernels-20260928/remote/results_20260928_122508.jsonl")
SRC_PRE = os.path.join(DIAG, "glm-prefill-kernels-20260928/runs/r1/dense.jsonl")

RESULTS = []
ENV_KEYS = ("GLM_DENSE_FAST", "GLM_DENSE_FAST_PREFILL", "GLM_DENSE_FAST_DECODE_MAX_M", "GLM_DENSE_FAST_PREFILL_MIN_M",
            "GLM_DENSE_FAST_TABLE", "GLM_DENSE_FAST_DRAFT")


def fresh():
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    sys.modules.pop("glm_ab", None)
    import glm_dense_fast as m
    importlib.reload(m)
    sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "_glm_dense_fast", False)]
    return m


def case(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception:  # noqa: BLE001
            import traceback
            RESULTS.append((fn.__name__, "FAIL", traceback.format_exc()))
        finally:
            for k in ENV_KEYS:
                os.environ.pop(k, None)
            sys.modules.pop("glm_ab", None)
    run.__name__ = fn.__name__
    return run


def raises(fn, text):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        assert text in str(exc), f"expected {text!r} in {exc!r}"
        return
    raise AssertionError(f"no exception (expected {text!r})")


# ---- pure Python ------------------------------------------------------------------------------------------------

@case
def table_schema_and_rules():
    m = fresh()
    t = m.table()
    assert set(t["shapes"]) == set(m.SHAPES), sorted(t["shapes"])
    for name, e in t["shapes"].items():
        K, PN, mx, calls = m.SHAPES[name]
        assert (e["K"], e["PN"], e["fmt"], e["calls_per_step"]) == (K, PN, "mx" if mx else "blk", calls), name
        for b, d in e["decode"].items():
            assert min(d["speedup"].values()) >= t["rules"]["min_decode_speedup"], (name, b, d["speedup"])
            lo, hi = m.BUCKET_RANGE[b]
            assert all(lo <= int(M) <= hi for M in d["speedup"]), (name, b)
            assert d["cfg"].get("ldg", 0) == 0
    dec = {n: sorted(e["decode"]) for n, e in t["shapes"].items()}
    # the two measured losers / non-winners of the 2026-09-28 sweep stay on stock at 17 <= M <= 32
    assert dec["shared_gate_up"] == ["1", "2"] and dec["mla_qkv_a"] == ["1", "2"], dec
    assert all(dec[n] == ["1", "2", "4"] for n in ("kda_in_proj", "kda_o", "mla_o", "mla_q_b", "shared_down")), dec
    pre = {n: e["prefill"]["min_m"] for n, e in t["shapes"].items() if "prefill" in e}
    assert pre == {"kda_in_proj": 1024, "kda_o": 4096}, pre
    ents = m.entries(t)
    assert len(ents) == sum(len(v) for v in dec.values()) + len(pre) == 21, ents
    assert ents == sorted(ents, key=lambda k: (k[1], k[0] != "dec", k[2] or "")), ents
    assert m.load_table()["_hash"] == t["_hash"] and len(t["_hash"]) == 16


@case
def table_regenerates_from_sources():
    m = fresh()
    if not all(os.path.exists(p) for p in (SRC_BEST, SRC_RES, SRC_PRE)):
        print("  (skipped: sweep sources not on this machine)")
        return
    out = subprocess.run([sys.executable, os.path.join(OVERLAY, "glm_dense_fast.py"), "--make-table", SRC_BEST,
                          SRC_RES, SRC_PRE], capture_output=True, text=True, check=True).stdout
    with open(m.DEFAULT_TABLE) as f:
        shipped = json.load(f)
    regen = json.loads(out)
    # 'sources' is a digest of the input files; best_configs.json was rewritten (same entries) after the table was cut.
    diff = sorted(k for k in set(regen) | set(shipped) if regen.get(k) != shipped.get(k))
    assert diff in ([], ["sources"]), f"shipped table differs from --make-table over the source measurements: {diff}"


@case
def switch_parsing_and_buckets():
    m = fresh()
    names = tuple(m.table()["shapes"])
    assert m.parse_mode("0") == frozenset() and m.parse_mode("") == frozenset() and m.parse_mode(None) == frozenset()
    assert m.parse_mode("1") is None and m.parse_mode("ON") is None and m.parse_mode("all") is None
    assert m.parse_mode("kda_in_proj, kda_o", names) == frozenset({"kda_in_proj", "kda_o"})
    raises(lambda: m.parse_mode("kda_in_proj,kda_bogus", names), "kda_bogus")
    assert m.mode_on("kda_o") and not m.mode_on("off")
    assert [m.bucket(M) for M in (1, 8, 9, 16, 17, 32, 33)] == ["1", "1", "2", "2", "4", "4", None]
    assert m.test_Ms("1", 32) == [1, 8] and m.test_Ms("2", 32) == [9, 16] and m.test_Ms("4", 32) == [17, 32]
    assert m.test_Ms("4", 24) == [17, 24] and m.test_Ms("4", 16) == [] and m.test_Ms("2", 9) == [9]
    assert m.ws_need({"gs": 1, "tpw": 1}, 6464, 1) == (0, 0)
    assert m.ws_need({"gs": 4, "tpw": 1}, 1024, 2) == (4 * 8 * 2 * 1024, 16)
    assert m.ws_need({"gs": 3, "tpw": 2}, 4096, 4) == (3 * 8 * 4 * 4096, 32)
    os.environ["GLM_DENSE_FAST"] = "kda_o"
    assert m.shape_on("GLM_DENSE_FAST", "kda_o") and not m.shape_on("GLM_DENSE_FAST", "mla_o")
    assert not m.shape_on("GLM_DENSE_FAST_PREFILL", "kda_o")


@case
def nvcc_flag_form():
    m = fresh()
    cu, cxx = m.build_flags(["/a/include", "/b/include"])
    assert "-idirafter" not in cu, cu                       # nvcc rejects a bare -idirafter
    assert cu.count("-Xcompiler") == 2 and "-idirafter,/a/include" in cu and "-idirafter,/b/include" in cu
    assert cxx[-4:] == ["-idirafter", "/a/include", "-idirafter", "/b/include"]
    os.environ["GLM_DENSE_FAST_BUILD_DIR"] = "/tmp/x"
    try:
        assert m.build_root() == "/tmp/x"
    finally:
        del os.environ["GLM_DENSE_FAST_BUILD_DIR"]
    for f in m.CSRC_FILES:
        assert os.path.exists(os.path.join(m.CSRC, f)), f


def _run_py(code, env):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GLM_", "PYTHON"))}
    e.update(env)
    e["PYTHONPATH"] = OVERLAY
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)


@case
def register_fatal_on_config_error():
    r = _run_py("import glm_dense_fast as m; m.register(); print('alive')", {"GLM_DENSE_FAST": "kda_in_proj,nope"})
    assert r.returncode == 1 and "refusing to start" in r.stdout + r.stderr and "alive" not in r.stdout, r
    r = _run_py("import glm_dense_fast as m; m.register(); print('alive')", {"GLM_DENSE_FAST": "1",
                                                                             "GLM_DENSE_FAST_DECODE_MAX_M": "x"})
    assert r.returncode == 1 and "refusing to start" in r.stdout + r.stderr, r
    r = _run_py("import glm_dense_fast as m, sys; m.register(); "
                "print('alive', any(getattr(f, '_glm_dense_fast', False) for f in sys.meta_path))",
                {"GLM_DENSE_FAST": "0", "GLM_DENSE_FAST_PREFILL": "off"})
    assert r.returncode == 0 and "alive False" in r.stdout, r
    r = _run_py("import glm_dense_fast as m, sys; m.register(); "
                "print('alive', any(getattr(f, '_glm_dense_fast', False) for f in sys.meta_path))",
                {"GLM_DENSE_FAST": "kda_in_proj", "GLM_DENSE_FAST_PREFILL": "kda_in_proj"})
    assert r.returncode == 0 and "alive True" in r.stdout and "armed" in r.stderr, r


@case
def sitecustomize_and_glm_ab_wiring():
    code = ("import sys, os, glm_ab; print(json.dumps(dict(known={k: glm_ab.KNOWN.get(k) for k in "
            "('GLM_DENSE_FAST', 'GLM_DENSE_FAST_PREFILL')}, env=[os.environ.get('GLM_DENSE_FAST'), "
            "os.environ.get('GLM_DENSE_FAST_PREFILL')], hashed='glm_dense_fast' in glm_ab.HASHED_SOURCES, "
            "finder=any(getattr(f, '_glm_dense_fast', False) for f in sys.meta_path), "
            "v=[glm_ab.env_for(i, 'GLM_DENSE_FAST') for i in range(glm_ab.N)])))")
    env = {"GLM_AB_VARIANTS": "2", "GLM_AB_V0": "GLM_DENSE_FAST=0+GLM_DENSE_FAST_PREFILL=0",
           "GLM_AB_V1": "GLM_DENSE_FAST=kda_in_proj,kda_o+GLM_DENSE_FAST_PREFILL=1", "GLM_AB_QUIET": "1"}
    r = _run_py("import json; " + code, env)
    assert r.returncode == 0, r
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["known"] == {"GLM_DENSE_FAST": "raw", "GLM_DENSE_FAST_PREFILL": "raw"}, got
    assert got["env"] == ["1", "1"] and got["hashed"] and got["finder"], got   # union armed the install gate
    assert got["v"] == ["0", "kda_in_proj,kda_o"], got
    # a variant spec with an unknown shape name is fatal at startup
    env["GLM_AB_V1"] = "GLM_DENSE_FAST=kda_in_proj,bogus"
    r = _run_py("print('alive')", env)
    assert r.returncode == 1 and "refusing to start" in r.stdout + r.stderr and "alive" not in r.stdout, r
    # nothing armed: the module is not even imported
    r = _run_py("import sys; print('mod', 'glm_dense_fast' in sys.modules)", {})
    assert r.returncode == 0 and "mod False" in r.stdout, r


# ---- torch (CPU) ------------------------------------------------------------------------------------------------

def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class FakeLinear:
    """Marlin-prepared 8-bit linear as vLLM leaves it: weight int32 [K/16, 4*PN], MX weight_scale e8m0 [K/32, PN]
    or block weight_scale_inv bf16 [K/128, PN]."""

    def __init__(self, K, N, PN, mx, seed=0, bias=None, scale_dtype=None):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-2 ** 31, 2 ** 31 - 1, (K // 16, 4 * PN), generator=g, dtype=torch.int64) \
            .to(torch.int32)
        if mx:
            self.weight_scale = torch.randint(117, 125, (K // 32, PN), generator=g).to(torch.uint8) \
                .view(torch.float8_e8m0fnu)
        else:
            self.weight_scale_inv = ((torch.rand((K // 128, PN), generator=g) * 0.048 + 0.002) * 2.0 ** 120) \
                .to(scale_dtype or torch.bfloat16)
        self.output_size_per_partition = N
        self.input_size_per_partition = K
        self.bias = bias
        self.workspace = None


class FakeModel:
    def __init__(self, mods):
        self.mods = mods

    def named_modules(self):
        return list(self.mods.items())


def model_all_shapes(m, per_shape=2):
    mods = {}
    for name, (K, PN, mx, _) in m.SHAPES.items():
        N = 6416 if name == "kda_in_proj" else PN
        for i in range(per_shape):
            mods[f"layers.{i}.{name}"] = FakeLinear(K, N, PN, mx, seed=hash((name, i)) & 0xFFFF)
    mods["layers.0.other"] = FakeLinear(4096, 3000, 3008, True)          # not a table shape
    mods["layers.0.biased"] = FakeLinear(2048, 4096, 4096, True, bias=object())
    return FakeModel(mods)


def unpack_torch(weight, scales, mx):
    """Torch port of unmarlin8_kernel's index arithmetic (same expressions, whole matrix at once)."""
    import torch
    kt_, words = weight.shape
    K, PN = kt_ * 16, words // 4
    flat = weight.reshape(-1).to(torch.int64)
    k = torch.arange(K)
    n = torch.arange(PN)
    kt, hk, tr, kb = k // 16, (k // 8) % 2, (k // 2) % 4, k % 2
    nt, w, hn, tc = n // 64, (n // 16) % 4, (n // 8) % 2, n % 8
    woff = (kt * 4 * PN + tr * 8)[:, None] + (nt * 256 + tc * 32 + w * 2 + hn)[None, :]
    byte = (flat[woff] >> (8 * (hk + 2 * kb))[:, None]) & 255
    e, mm = (byte >> 3) & 15, (byte & 7).double()
    mag = torch.where(e == 0, mm * 2.0 ** -9, (8 + mm) * torch.exp2(e.double() - 10))
    val = torch.where((byte & 128) != 0, -mag, mag)
    sflat = scales.reshape(-1)
    if mx:
        col = nt * 64 + 8 * tc + 4 * (w // 2) + 2 * hn + (w % 2)
        sb = sflat.view(torch.uint8)[((kt // 2) * PN)[:, None] + col[None, :]].to(torch.int64)
        sv = torch.where(sb == 0, torch.zeros_like(sb, dtype=torch.float64), torch.exp2(sb.double() - 127))
    else:
        col = nt * 64 + 8 * tc + 2 * w + hn
        sv = sflat[((kt // 8) * PN)[:, None] + col[None, :]].double() * 2.0 ** -120
    return (val * sv).float().to(torch.bfloat16)


def decode_reference(weight, scales, mx):
    """Independent decoder of the Marlin 8-bit layout (bench_gpu.py dense_reference of the glm-cuda-kernels study:
    byte position in a k16 row = ((nt*256 + lane*8 + j*2 + side)*4 + q), lane = tc*4 + tr)."""
    import torch
    kt_, words = weight.shape
    K, PN = kt_ * 16, words // 4
    b = torch.arange(256, dtype=torch.int64)
    e, mm = (b >> 3) & 15, (b & 7).double()
    lut = torch.where((b & 128) != 0, -1.0, 1.0) * torch.where(e == 0, mm * 2.0 ** -9, (8 + mm) * torch.exp2(e.double() - 10))
    by = weight.contiguous().view(torch.uint8).reshape(K // 16, 4 * PN * 4).to(torch.int64)
    nt, tc, tr, j, side, q = torch.meshgrid(*[torch.arange(v) for v in (PN // 64, 8, 4, 4, 2, 4)], indexing="ij")
    pos = ((nt * 256 + (tc * 4 + tr) * 8 + j * 2 + side) * 4 + q).reshape(-1)
    koff = (2 * tr + torch.tensor([0, 8, 1, 9])[q]).reshape(-1)
    n = (nt * 64 + 16 * j + 8 * side + tc).reshape(-1)
    W = torch.zeros((K, PN), dtype=torch.float64)
    for kt in range(K // 16):
        W[kt * 16 + koff, n] = lut[by[kt, pos]]
    nn = torch.arange(PN)
    c, r = nn // 64, nn % 64
    tc_, w_, hn_ = r % 8, r // 16, (r % 16) // 8
    if mx:
        col = 64 * c + 8 * tc_ + 4 * (w_ // 2) + 2 * hn_ + (w_ % 2)
        sb = scales.view(torch.uint8).reshape(K // 32, PN).to(torch.int64)[:, col]
        sv = torch.where(sb == 0, torch.zeros_like(sb, dtype=torch.float64), torch.exp2(sb.double() - 127))
        return (W * sv.repeat_interleave(32, 0)).float().to(torch.bfloat16)
    col = 64 * c + 8 * tc_ + 2 * w_ + hn_
    sv = scales.reshape(K // 128, PN)[:, col].double() * 2.0 ** -120
    return (W * sv.repeat_interleave(128, 0)).float().to(torch.bfloat16)


@case
def t_classify_fake_marlin_layers():
    m = fresh()
    import torch
    t = m.table()
    model = model_all_shapes(m)
    got = m.scan(model, t, require_cuda=False)
    names = sorted({tag.name for _, _, tag in got})
    assert names == sorted(m.SHAPES), names
    assert len(got) == 2 * len(m.SHAPES)
    for mname, mod, tag in got:
        assert tag.sattr == ("weight_scale" if tag.mx else "weight_scale_inv")
        assert tag.N == mod.output_size_per_partition and tag.K == mod.input_size_per_partition
    assert m.classify(model.mods["layers.0.other"], t, False) is None
    assert m.classify(model.mods["layers.0.biased"], t, False) is None
    assert m.classify(FakeLinear(1536, 4096, 4096, False, scale_dtype=torch.float16), t, False) is None
    odd = FakeLinear(4096, 6416, 6464, True)
    odd.input_size_per_partition = 4000                               # K padded by Marlin -> stock
    assert m.classify(odd, t, False) is None
    assert m.classify(FakeLinear(4096, 6416, 6464, True), t, True) is None   # CPU tensors: not served


@case
def t_unpack_formulas_match_independent_decoder():
    import torch
    for K, N, PN, mx in ((256, 192, 192, True), (256, 128, 128, False), (512, 320, 320, True)):
        lay = FakeLinear(K, N, PN, mx, seed=K + PN)
        s = lay.weight_scale if mx else lay.weight_scale_inv
        a = unpack_torch(lay.weight, s, mx)
        b = decode_reference(lay.weight, s, mx)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (K, PN, mx)


class Patched:
    """Swap the GPU entry points of the module for CPU fakes: decode/prefill = exact reference (x @ W) unless
    told to break a shape, stock = the same product computed separately."""

    def __init__(self, m, broken=(), raise_on=(), build_fails=False):
        import torch
        self.m, self.broken, self.raise_on, self.build_fails = m, set(broken), set(raise_on), build_fails
        self.calls = []
        self.W = {}

        def dense(x2, weight, scales, N, mx):
            key = (weight.data_ptr(), N)
            if key not in self.W:
                self.W[key] = unpack_torch(weight, scales, mx)[:, :N].float()
            return (x2.float() @ self.W[key]).to(torch.bfloat16)

        def dec(x2, weight, scales, N, mx, cfg):
            self.calls.append(("dec", x2.shape[0], N, dict(cfg)))
            if ("dec", N) in self.raise_on:
                raise RuntimeError("boom")
            y = dense(x2, weight, scales, N, mx)
            return y * 1.5 if ("dec", N) in self.broken else y

        def pre(x2, weight, scales, N, mx):
            self.calls.append(("pre", x2.shape[0], N, None))
            y = dense(x2, weight, scales, N, mx)
            return y * 1.5 if ("pre", N) in self.broken else y

        m.decode_gemm, m.prefill_gemm = dec, pre
        self.stock_call = lambda layer, mx: (lambda x: dense(x, layer.weight, layer.weight_scale if mx
                                                             else layer.weight_scale_inv,
                                                             layer.output_size_per_partition, mx))
        self.build = (lambda: (_ for _ in ()).throw(RuntimeError("nvcc: fatal"))) if build_fails else (lambda: None)


def small_model(m, per_shape=2):
    """Same shape keys as the table but a patched table with small K/PN so CPU products stay cheap."""
    return model_all_shapes(m, per_shape)


def _prep(m, P, model, **kw):
    return m.prepare(m.scan(model, m.table(), require_cuda=False), stock_call=P.stock_call, build=P.build, **kw)


def _small_table(m, tmpdir):
    """Write a table with the real structure but small shapes; returns (path, shape dims)."""
    import copy
    t = json.load(open(m.DEFAULT_TABLE))
    # (K, PN, N); every N distinct so the fakes can break one shape by N
    small = {"kda_in_proj": (256, 256, 200), "kda_o": (128, 128, 128), "mla_qkv_a": (256, 384, 384),
             "mla_q_b": (128, 320, 320), "mla_o": (256, 192, 192), "shared_gate_up": (256, 64, 64),
             "shared_down": (128, 448, 448)}
    t2 = copy.deepcopy(t)
    for name, (K, PN, N) in small.items():
        t2["shapes"][name]["K"], t2["shapes"][name]["PN"] = K, PN
    p = os.path.join(tmpdir, "table.json")
    json.dump(t2, open(p, "w"))
    return p, small


def _small_model(small, per_shape=2, mxmap=None):
    mods = {}
    import glm_dense_fast as m
    for name, (K, PN, N) in small.items():
        mx = m.SHAPES[name][2]
        for i in range(per_shape):
            mods[f"layers.{i}.{name}"] = FakeLinear(K, N, PN, mx, seed=(K * 7 + PN + i))
    return FakeModel(mods)


def _setup(tmp, env):
    m = fresh()
    p, small = _small_table(m, tmp)
    os.environ["GLM_DENSE_FAST_TABLE"] = p
    os.environ.update(env)
    m._STATE["table"] = None
    return m, small


def _xs(K, M, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((M, K), generator=g) * 0.5).to(torch.bfloat16)


@case
def t_prepare_tags_agreed_entries():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1", "GLM_DENSE_FAST_PREFILL": "1"})
        P = Patched(m)
        model = _small_model(small)
        rep = _prep(m, P, model)
        assert rep["off"] == [], rep
        assert rep["layers"] == {n: 2 for n in small}, rep
        tags = {n: model.mods[f"layers.0.{n}"].__dict__[m.TAG] for n in small}
        assert tags["kda_in_proj"] is model.mods["layers.1.kda_in_proj"].__dict__[m.TAG]   # shared per shape
        assert sorted(tags["shared_gate_up"].dec) == ["1", "2"] and tags["shared_gate_up"].dec_max_m == 16
        assert sorted(tags["kda_o"].dec) == ["1", "2", "4"] and tags["kda_o"].dec_max_m == 32
        assert tags["kda_in_proj"].pre_min_m == 1024 and tags["kda_o"].pre_min_m == 4096
        assert tags["mla_o"].pre_min_m == 1 << 62
        # the self-test ran each decode bucket at its two ends and each prefill entry at min_m, on layer 0 only
        ran = sorted({(c[0], c[1], c[2]) for c in P.calls})
        assert ("dec", 1, 200) in ran and ("dec", 8, 200) in ran and ("dec", 17, 200) in ran and ("dec", 32, 200) in ran
        assert ("pre", 1024, 200) in ran and ("pre", 4096, 128) in ran
        assert not any(c[0] == "dec" and c[2] == 64 and c[1] > 16 for c in P.calls)   # shared_gate_up bucket 4 absent
        # a second prepare (e.g. the drafter with GLM_DENSE_FAST_DRAFT=1) without prefill shapes keeps the scratch
        assert m._STATE["scratch"] and m._STATE["prepared_models"] == 1
        extra = FakeModel({"d.mla_o": FakeLinear(256, 192, 192, False, seed=99)})
        rep2 = m.prepare(m.scan(extra, m.table(), False), stock_call=P.stock_call, build=P.build)
        assert rep2["layers"] == {"mla_o": 1} and m._STATE["scratch"] and m._STATE["prepared_models"] == 2


@case
def t_prepare_failures_turn_entries_off():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1", "GLM_DENSE_FAST_PREFILL": "1"})
        # kda_o decode kernel returns wrong numbers, mla_o decode raises, kda_in_proj prefill wrong
        P = Patched(m, broken={("dec", 128), ("pre", 200)}, raise_on={("dec", 192)})
        model = _small_model(small)
        rep = _prep(m, P, model)
        off = sorted(rep["off"])
        assert ("dec", "kda_o", "1") in off and ("dec", "mla_o", "1") in off and ("pre", "kda_in_proj", None) in off
        t_o = model.mods["layers.0.kda_o"].__dict__[m.TAG]
        assert t_o.dec == {} and t_o.pre_min_m == 4096           # decode off, its prefill still on
        assert "mla_o" not in rep["layers"] and m.TAG not in model.mods["layers.0.mla_o"].__dict__
        t_i = model.mods["layers.0.kda_in_proj"].__dict__[m.TAG]
        assert sorted(t_i.dec) == ["1", "2", "4"] and t_i.pre_min_m == 1 << 62
        # the mla_q_b block path still passed (same fake code path as mla_o, different N)
        assert sorted(model.mods["layers.0.mla_q_b"].__dict__[m.TAG].dec) == ["1", "2", "4"]
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1", "GLM_DENSE_FAST_PREFILL": "1"})
        P = Patched(m, build_fails=True)
        model = _small_model(small)
        rep = _prep(m, P, model)
        assert all(k[0] == "dec" for k in rep["off"]) and len(rep["off"]) == 19, rep["off"]
        assert set(rep["layers"]) == {"kda_in_proj", "kda_o"}   # prefill-only tags
        assert all(model.mods[f"layers.0.{n}"].__dict__[m.TAG].dec == {} for n in ("kda_in_proj", "kda_o"))
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1", "GLM_DENSE_FAST_PREFILL": "1"})
        P = Patched(m, broken={("pre", 200), ("pre", 128)})                  # every prefill entry fails
        rep = _prep(m, P, _small_model(small))
        assert ("pre", "kda_o", None) in rep["off"] and not m._STATE["scratch"]   # scratch released


@case
def t_prepare_respects_switches_and_caps():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "kda_in_proj", "GLM_DENSE_FAST_PREFILL": "0",
                                "GLM_DENSE_FAST_DECODE_MAX_M": "12", "GLM_DENSE_FAST_PREFILL_MIN_M": "3000"})
        P = Patched(m)
        model = _small_model(small)
        rep = _prep(m, P, model)
        assert rep["layers"] == {"kda_in_proj": 2}, rep
        t = model.mods["layers.0.kda_in_proj"].__dict__[m.TAG]
        assert sorted(t.dec) == ["1", "2"] and t.dec_max_m == 12 and t.pre_min_m == 1 << 62
        assert max(c[1] for c in P.calls) == 12 and {c[0] for c in P.calls} == {"dec"}
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST_PREFILL": "kda_in_proj", "GLM_DENSE_FAST_PREFILL_MIN_M": "3000"})
        P = Patched(m)
        model = _small_model(small)
        rep = _prep(m, P, model)
        t = model.mods["layers.0.kda_in_proj"].__dict__[m.TAG]
        assert rep["layers"] == {"kda_in_proj": 2} and t.dec == {} and t.pre_min_m == 3000
        assert P.calls == [("pre", 3000, 200, None)] * 2   # the self-test runs it twice (determinism)


@case
def t_rank_agreement_over_fake_group():
    import torch
    import torch.distributed as dist
    m = fresh()
    peers = {"flags": None, "hash": None}
    orig = dist.all_reduce

    def fake_all_reduce(t, op=None, group=None):
        other = t.clone()
        if peers["flags"] is not None:
            other[2:] = torch.tensor(peers["flags"])
        if peers["hash"] is not None:
            other[0], other[1] = peers["hash"], -peers["hash"]
        if op == dist.ReduceOp.MIN:
            t.copy_(torch.minimum(t, other))
        else:
            t.copy_(torch.maximum(t, other))
    dist.all_reduce = fake_all_reduce
    try:
        h = "00000000000000ff"
        peers["flags"] = [1, 0, 1, 1]
        assert m.agree([1, 1, 1, 0], h, group=object(), world=2) == [1, 0, 1, 0]
        peers["hash"] = 0xfe
        assert m.agree([1, 1, 1, 1], h, group=object(), world=2) == [0, 0, 0, 0]   # table differs -> all off
        assert m.agree([1, 0], h, group=None, world=1) == [1, 0]
    finally:
        dist.all_reduce = orig


@case
def t_dispatch_by_M_and_variant():
    import tempfile
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1", "GLM_DENSE_FAST_PREFILL": "1"})
        P = Patched(m)
        model = _small_model(small)
        _prep(m, P, model)
        lay = model.mods["layers.1.shared_gate_up"]        # K 256, N 64, buckets 1, 2
        io = model.mods["layers.1.kda_in_proj"]             # K 256, N 200, buckets 1, 2, 4, prefill >= 1024
        P.calls.clear()
        for M in range(1, 33):
            y = m.fast_apply(lay, _xs(256, M), None)
            assert (y is None) == (M > 16), M
            if y is not None:
                assert y.shape == (M, 64)
            assert m.fast_apply(io, _xs(256, M), None) is not None
        assert {c[3]["ks"] for c in P.calls if c[2] == 200 and c[1] == 32} == {4}          # bucket-4 cfg of the table
        assert m.fast_apply(io, _xs(256, 33), None) is None and m.fast_apply(io, _xs(256, 1023), None) is None
        y = m.fast_apply(io, _xs(256, 1024).view(2, 512, 256), None)                        # 3-D input
        assert y is not None and y.shape == (2, 512, 200)
        assert m.fast_apply(io, _xs(256, 4), object()) is None                              # bias
        assert m.fast_apply(io, _xs(256, 4).half(), None) is None                           # fp16
        assert m.fast_apply(io, _xs(128, 4), None) is None                                  # wrong K
        cap = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: True

        class CudaLike(torch.Tensor):   # CPU tensor that reports is_cuda (only the capture guard reads it)
            @property
            def is_cuda(self):
                return True
        try:
            x = _xs(256, 2048).as_subclass(CudaLike)
            assert m.fast_apply(io, x, None) is None                                        # never in a capture
            assert m.fast_apply(io, _xs(256, 8), None) is not None                          # decode is capture-safe
        finally:
            torch.cuda.is_current_stream_capturing = cap
        assert m.fast_apply(FakeLinear(256, 200, 200, True), _xs(256, 4), None) is None     # untagged
        # per-call switch (no harness): off -> stock even though tagged
        os.environ["GLM_DENSE_FAST"] = "kda_o"
        assert m.fast_apply(io, _xs(256, 4), None) is None
        os.environ["GLM_DENSE_FAST_PREFILL"] = "0"
        assert m.fast_apply(io, _xs(256, 2048), None) is None
        # through glm_ab: the variant being captured / run decides
        ab = types.SimpleNamespace(ACTIVE=True, cur=0,
                                   specs=[{"GLM_DENSE_FAST": "0", "GLM_DENSE_FAST_PREFILL": "0"},
                                          {"GLM_DENSE_FAST": "kda_in_proj", "GLM_DENSE_FAST_PREFILL": "1"}])
        ab.env = lambda name, default=None: ab.specs[ab.cur].get(name, default)
        sys.modules["glm_ab"] = ab
        assert m.fast_apply(io, _xs(256, 4), None) is None and m.fast_apply(io, _xs(256, 2048), None) is None
        ab.cur = 1
        assert m.fast_apply(io, _xs(256, 4), None) is not None and m.fast_apply(io, _xs(256, 2048), None) is not None
        assert m.fast_apply(lay, _xs(256, 4), None) is None                                 # shape not in variant


@case
def t_kernel_class_wrappers_and_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m, small = _setup(tmp, {"GLM_DENSE_FAST": "1"})
        P = Patched(m)

        class MarlinMxfp8LinearKernel:
            def apply_weights(self, layer, x, bias=None):
                return "stock-mx"

        class MarlinFP8ScaledMMLinearKernel:
            def __init__(self, block_quant, marlin_input_dtype=None):
                self.block_quant, self.marlin_input_dtype = block_quant, marlin_input_dtype

            def apply_weights(self, layer, x, bias=None):
                return "stock-fp8"

        class BaseModelLoader:
            def __init__(self, model):
                self.model = model

            def load_model(self, vllm_config, model_config, prefix=""):
                return self.model

        mx_mod = types.SimpleNamespace(MarlinMxfp8LinearKernel=MarlinMxfp8LinearKernel)
        fp8_mod = types.SimpleNamespace(MarlinFP8ScaledMMLinearKernel=MarlinFP8ScaledMMLinearKernel)
        ld_mod = types.SimpleNamespace(BaseModelLoader=BaseModelLoader)
        m.install_mxfp8(mx_mod)
        m.install_fp8(fp8_mod)
        m.install_loader(ld_mod)
        m.install_mxfp8(mx_mod)                                           # idempotent
        assert MarlinMxfp8LinearKernel.apply_weights.__wrapped__.__name__ == "apply_weights"

        orig_scan = m.scan
        m.scan = lambda model, tbl, require_cuda=True: orig_scan(model, tbl, False)
        orig_prepare = m.prepare
        m.prepare = lambda cands, **kw: orig_prepare(cands, stock_call=P.stock_call, build=P.build, **kw)
        target = _small_model(small, per_shape=1)
        mc = object()
        vc = types.SimpleNamespace(model_config=mc)
        assert BaseModelLoader(target).load_model(vc, mc) is target
        assert m.TAG in target.mods["layers.0.kda_in_proj"].__dict__
        draft = _small_model(small, per_shape=1)
        dc = types.SimpleNamespace(model="/draft")
        vc.model_config = types.SimpleNamespace(model="/model")
        BaseModelLoader(draft).load_model(vc, dc)
        assert not any(m.TAG in mod.__dict__ for mod in draft.mods.values())         # drafter: stock
        os.environ["GLM_DENSE_FAST_DRAFT"] = "1"
        BaseModelLoader(draft).load_model(vc, dc)
        assert m.TAG in draft.mods["layers.0.mla_o"].__dict__
        m.scan, m.prepare = orig_scan, orig_prepare

        io, blk = target.mods["layers.0.kda_in_proj"], target.mods["layers.0.mla_o"]
        k_mx = MarlinMxfp8LinearKernel()
        assert k_mx.apply_weights(io, _xs(256, 4)) is not None and not isinstance(k_mx.apply_weights(io, _xs(256, 4)), str)
        assert k_mx.apply_weights(io, _xs(256, 64)) == "stock-mx"
        assert k_mx.apply_weights(FakeLinear(256, 200, 200, True), _xs(256, 4)) == "stock-mx"
        assert not isinstance(MarlinFP8ScaledMMLinearKernel(True).apply_weights(blk, _xs(256, 4)), str)
        assert MarlinFP8ScaledMMLinearKernel(False).apply_weights(blk, _xs(256, 4)) == "stock-fp8"   # per-tensor
        import torch
        assert MarlinFP8ScaledMMLinearKernel(True, torch.float8_e4m3fn).apply_weights(blk, _xs(256, 4)) == "stock-fp8"


@case
def t_close_to_tolerances():
    import torch
    m = fresh()
    g = torch.Generator().manual_seed(0)
    b = (torch.randn(64, 512, generator=g) * 3).to(torch.bfloat16)
    assert m.close_to(b.clone(), b)["ok"] and m.close_to(b.clone(), b)["eq"] == 1.0
    # one-ulp flips on 1 % of elements: pass
    a = b.clone().view(torch.int16)
    idx = torch.randperm(a.numel(), generator=g)[: a.numel() // 100]
    a.view(-1)[idx] += 1
    r = m.close_to(a.view(torch.bfloat16), b)
    assert r["ok"] and r["eq"] < 1.0, r
    # a 1 % scale error: fail
    assert not m.close_to((b.float() * 1.01).to(torch.bfloat16), b)["ok"]
    nan = b.clone()
    nan[0, 0] = float("nan")
    assert not m.close_to(nan, b)["ok"]


def main():
    tests = [table_schema_and_rules, table_regenerates_from_sources, switch_parsing_and_buckets, nvcc_flag_form,
             register_fatal_on_config_error, sitecustomize_and_glm_ab_wiring]
    torch_tests = [t_classify_fake_marlin_layers, t_unpack_formulas_match_independent_decoder,
                   t_prepare_tags_agreed_entries, t_prepare_failures_turn_entries_off,
                   t_prepare_respects_switches_and_caps, t_rank_agreement_over_fake_group,
                   t_dispatch_by_M_and_variant, t_kernel_class_wrappers_and_loader, t_close_to_tolerances]
    for t in tests:
        t()
    if _have_torch():
        for t in torch_tests:
            t()
    else:
        print(f"(torch not importable: {len(torch_tests)} torch checks skipped; "
              "uv run --no-project --with torch python tests/test_glm_dense_fast.py)")
    bad = 0
    for name, status, tb in RESULTS:
        print(f"[{status}] {name}")
        if status != "PASS":
            bad += 1
            print(tb)
    print(f"{len(RESULTS) - bad}/{len(RESULTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

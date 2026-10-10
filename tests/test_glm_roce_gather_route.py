#!/usr/bin/env python3
"""CPU test of overlay/glm_roce_gather_route.py (GLM_ROCE_AG_DIM0_NCCL) against the repo's real RoCE shim
(roce/glm_roce/adapter.py + install.py). No GPU, no process group: the adapter is built without its constructor and
its b12x runtime is a stand-in with the runtime's own eligibility rule (dim 0 or last, 0 < bytes <= max_gather).

  1. byte-size parsing, including the values that must be refused
  2. the rule: dim-0 shards above the threshold -> NCCL; last-dim, 1-D and small dim-0 gathers -> unchanged
  3. install through the post-import finder (glm_roce.adapter imported AFTER register), idempotent re-install
  4. the real CudaCommunicator.all_gather wrapper (install.patch_cuda_communicator) routes by the patched decision:
     prefill-shard rows -> stock (NCCL) all_gather, logits shard (last dim, 9.9 MB) -> RoCE, decode rows -> RoCE
  5. switch off through glm_ab (in-boot A/B semantics) restores the adapter's own decision exactly
  6. rank invariance: four adapters with ranks 0..3 decide identically for every shape
  7. a bad GLM_ROCE_AG_DIM0_NCCL_ABOVE fails register(); switch off = no install at all
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "overlay"), os.path.join(REPO, "roce")]

import torch  # noqa: E402

FAIL = []


def check(cond, what):
    print(("[PASS] " if cond else "[FAIL] ") + what)
    if not cond:
        FAIL.append(what)


# in-boot A/B stand-in (glm_ab.env per variant); inactive until a case switches it on
ab = types.ModuleType("glm_ab")
ab.ACTIVE = False
ab.values = {}
ab.env = lambda name, default=None: ab.values.get(name, default)
sys.modules["glm_ab"] = ab

os.environ["GLM_ROCE_AG_DIM0_NCCL"] = "1"
os.environ.pop("GLM_ROCE_AG_DIM0_NCCL_ABOVE", None)
import glm_roce_gather_route as G  # noqa: E402

# 1. parsing
for v, want in (("4MiB", 4 << 20), ("4M", 4 << 20), ("4mb", 4 << 20), ("4096KiB", 4 << 20), ("4194304", 4 << 20),
                ("512k", 512 << 10), ("1GiB", 1 << 30), ("64b", 64)):
    check(G.parse_bytes(v) == want, f"parse_bytes({v!r}) == {want}")
for bad in ("", "0", "-4MiB", "4ib", "four", "4TiB", "4 MiBs"):
    try:
        G.parse_bytes(bad)
        check(False, f"parse_bytes({bad!r}) refused")
    except ValueError:
        check(True, f"parse_bytes({bad!r}) refused")

# 2 + 3. install via the finder, then the real adapter module
assert "glm_roce.adapter" not in sys.modules
G.register()
import glm_roce.adapter as A  # noqa: E402
import glm_roce.install as I  # noqa: E402

check(getattr(A.GlmRoceAllReduce.should_all_gather, "_glm_gather_route", False), "finder installed the wrapper at import")
w = A.GlmRoceAllReduce.should_all_gather
G._install(A)
check(A.GlmRoceAllReduce.should_all_gather is w, "re-install is a no-op (no double wrap)")
check(G.above() == 4 << 20, "default threshold 4 MiB")

MAX_GATHER = A.read_limits({})[1]       # the served default, 16 MiB


class FakeRuntime:
    """b12x RoceAllReduce.should_all_gather / all_gather semantics without CUDA."""

    def __init__(self):
        self.calls = []

    def should_all_gather(self, inp, dim=-1):
        if inp.dim() == 0:
            return False
        d = dim + inp.dim() if dim < 0 else dim
        if d not in (0, inp.dim() - 1):
            return False
        n = inp.numel() * inp.element_size()
        return 0 < n <= MAX_GATHER

    def all_gather(self, inp, dim=-1):
        self.calls.append(("roce", tuple(inp.shape), dim))
        shape = list(inp.shape)
        shape[dim] *= 4
        return ("roce", tuple(shape))


def make_adapter(rank=0):
    a = A.GlmRoceAllReduce.__new__(A.GlmRoceAllReduce)
    a.disabled, a.rank, a.world_size = False, rank, 4
    a.max_size, a.max_gather = 4 << 20, MAX_GATHER
    a._runtime, a._announced, a._announced_gather, a._health_logged = FakeRuntime(), False, False, False
    return a


bf = torch.bfloat16
H = 4096
CASES = [  # (label, shape, dtype, dim, expected with the switch on, expected with the switch off)
    ("prefill shard 5760 rows [1440,4096] dim 0 (11.8 MB)", (1440, H), bf, 0, False, True),
    ("prefill shard 6912 rows [1728,4096] dim 0 (14.2 MB)", (1728, H), bf, 0, False, True),
    ("prefill shard 2880 rows [720,4096] dim 0 (5.9 MB)", (720, H), bf, 0, False, True),
    ("shard of exactly 4 MiB [512,4096] dim 0 stays RoCE", (512, H), bf, 0, True, True),
    ("one row above 4 MiB [513,4096] dim 0", (513, H), bf, 0, False, True),
    ("decode verify rows [32,4096] dim 0", (32, H), bf, 0, True, True),
    ("fp32 [300,4096] dim 0 (4.9 MB)", (300, H), torch.float32, 0, False, True),
    ("logits shard [128,38720] last dim (9.9 MB)", (128, 38720), bf, 1, True, True),
    ("logits shard [128,38720] dim -1", (128, 38720), bf, -1, True, True),
    ("argmax pairs [128,4] fp32 dim -1", (128, 4), torch.float32, -1, True, True),
    ("1-D [3_000_000] bf16 dim 0 (dim 0 is the last dim): adapter decides", (3_000_000,), bf, 0, True, True),
    ("3-D [1440,4,1024] dim 0", (1440, 4, 1024), bf, 0, False, True),
    ("3-D [1440,4,1024] middle dim: runtime refuses anyway", (1440, 4, 1024), bf, 1, False, False),
    ("above GLM_ROCE_GATHER_MAX_SIZE [2200,4096] dim 0 (18 MB): NCCL either way", (2200, H), bf, 0, False, False),
]

a0 = make_adapter(0)
for label, shape, dt, dim, on, off in CASES:
    x = torch.empty(shape, dtype=dt)
    got = a0.should_all_gather(x, dim)
    check(got == on, f"switch on : {label}: RoCE={got} (want {on})")

# 4. the real CudaCommunicator wrapper
cc = types.ModuleType("fake_cuda_communicator")


class CudaCommunicator:
    def __init__(self, cpu_group=None, device=None, device_group=None, unique_name="tp:0"):
        self.cpu_group, self.device, self.device_group, self.unique_name = cpu_group, device, device_group, unique_name
        self.world_size = 4
        self.nccl_calls = []

    def all_reduce(self, input_):
        return ("nccl-ar",)

    def all_gather(self, input_, dim=-1):
        self.nccl_calls.append((tuple(input_.shape), dim))
        return ("nccl", tuple(input_.shape), dim)

    def destroy(self):
        pass


cc.CudaCommunicator = CudaCommunicator
shared = {"adapter": None}


def factory(**kw):
    shared["adapter"] = make_adapter(0)
    return shared["adapter"]


I.patch_cuda_communicator(cc, adapter_factory=factory)
comm = cc.CudaCommunicator(unique_name="tp:0")
r = comm.all_gather(torch.empty((1440, H), dtype=bf), 0)
check(r[0] == "nccl", f"communicator: prefill shard rows -> stock NCCL all_gather ({r[0]})")
r = comm.all_gather(torch.empty((1440, H), dtype=bf), -2)
check(r[0] == "nccl", f"communicator: prefill shard rows as dim -2 -> stock NCCL ({r[0]})")
r = comm.all_gather(torch.empty((128, 38720), dtype=bf), -1)
check(r[0] == "roce", f"communicator: logits shard (last dim) -> RoCE ({r[0]})")
r = comm.all_gather(torch.empty((32, H), dtype=bf), 0)
check(r[0] == "roce", f"communicator: decode rows dim 0 -> RoCE ({r[0]})")
check(G.stats()["routed"] >= 3 and G.stats()["kept"] >= 1, f"counters {G.stats()}")

# 5. switch off through glm_ab: exactly the adapter's own decision
ab.ACTIVE, ab.values = True, {"GLM_ROCE_AG_DIM0_NCCL": "0"}
for label, shape, dt, dim, on, off in CASES:
    x = torch.empty(shape, dtype=dt)
    got = a0.should_all_gather(x, dim)
    own = A.GlmRoceAllReduce.should_all_gather.__wrapped__(a0, x, dim)
    check(got == off and got == own, f"switch off: {label}: RoCE={got} (want {off}, adapter alone {own})")
r = comm.all_gather(torch.empty((1440, H), dtype=bf), 0)
check(r[0] == "roce", f"communicator, switch off: prefill shard rows -> RoCE ({r[0]})")
ab.values = {"GLM_ROCE_AG_DIM0_NCCL": "1"}
r = comm.all_gather(torch.empty((1440, H), dtype=bf), 0)
check(r[0] == "nccl", f"communicator, switch back on: prefill shard rows -> NCCL ({r[0]})")
ab.ACTIVE, ab.values = False, {}

# 6. rank invariance
ranks = [make_adapter(r) for r in range(4)]
same = all(len({a.should_all_gather(torch.empty(s, dtype=dt), d) for a in ranks}) == 1 for _, s, dt, d, _, _ in CASES)
check(same, "ranks 0..3 decide identically for every shape")

# 7. bad threshold / switch off
os.environ["GLM_ROCE_AG_DIM0_NCCL_ABOVE"] = "4TiB"
G._S["above"] = None
try:
    G.register()
    check(False, "bad GLM_ROCE_AG_DIM0_NCCL_ABOVE fails register()")
except ValueError as exc:
    check(True, f"bad GLM_ROCE_AG_DIM0_NCCL_ABOVE fails register() ({exc})")
os.environ["GLM_ROCE_AG_DIM0_NCCL_ABOVE"] = "8MiB"
G._S["above"] = None
check(not G.to_nccl(torch.empty((720, H), dtype=bf), 0) and G.to_nccl(torch.empty((1440, H), dtype=bf), 0),
      "GLM_ROCE_AG_DIM0_NCCL_ABOVE=8MiB: 5.9 MB shard stays RoCE, 11.8 MB goes to NCCL")
os.environ["GLM_ROCE_AG_DIM0_NCCL"] = "0"
n_meta = len(sys.meta_path)
G.register()
check(len(sys.meta_path) == n_meta, "switch off: register() adds no finder")

# sitecustomize: the glm_ab key and the registration block
sc = open(os.path.join(REPO, "overlay", "sitecustomize.py")).read()
check('glm_ab.KNOWN.setdefault("GLM_ROCE_AG_DIM0_NCCL", "bool")' in sc, "sitecustomize: glm_ab key")
check(sc.find("glm_roce_gather_route.register()") > sc.find("glm_pf3_moe.register()") > 0,
      "sitecustomize: gather-route block registered after the pf3 block")

print(f"ALL {'PASS' if not FAIL else 'FAIL'}: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)

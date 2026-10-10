"""In-image import check for the in-boot A/B harness and the diagnostics hooks (no GPU needed).

Run inside the serving image with the overlay on PYTHONPATH (so sitecustomize runs at interpreter start), e.g.

  docker run --rm --network none --memory 6g --entrypoint python3 -v $OVL:/overlay:ro \
    -v $OVL/overlay/glm5next_kda.py:$VLLM_PKG/models/glm5next/nvidia/kda.py:ro -e PYTHONPATH=/overlay/overlay \
    -e GLM_AB_VARIANTS=3 -e GLM_AB_ALLOW_AA=1 -e GLM_AB_V0=GLM_TARGET_VOCAB_ARGMAX=1+GLM_KDA_STASH=1 \
    -e GLM_AB_V1=GLM_TARGET_VOCAB_ARGMAX=1+GLM_KDA_STASH=1 -e GLM_AB_V2=GLM_TARGET_VOCAB_ARGMAX=0+GLM_KDA_NOCOPY=1 \
    -e GLM_KDA_STASH=1 -e GLM_TARGET_VOCAB_ARGMAX=1 -e GLM_DIAG=1 $IMAGE /overlay/tests/test_glm_ab_image.py

Checks that the hooks landed on the image's real classes: CudaGraphManager.capture / run_fullgraph /
profile_memory, Worker.glm_ab_switch / glm_ab_status / glm_diag_snapshot, EngineCore.collective_rpc (with the
original signature, which the engine core's msgspec argument conversion inspects), GPUModelRunner.execute_model
(diag) and .sample (argmax), Triton JITFunction._do_compile / Autotuner.check_disk_cache / _bench, and the KDA
dispatcher chain (nocopy -> stash -> stock) with the drift guards of the exact adapters passing.
"""
import inspect
import os
import sys

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


import glm_ab  # noqa: E402  (already configured by sitecustomize)

check("harness armed from sitecustomize", glm_ab.ACTIVE and glm_ab.N == int(os.environ["GLM_AB_VARIANTS"]))
check("union put GLM_KDA_NOCOPY in the env", os.environ.get("GLM_KDA_NOCOPY") == "1")
print(glm_ab.describe())

import vllm.v1.worker.gpu.cudagraph_utils as cu  # noqa: E402

cls = cu.CudaGraphManager
check("CudaGraphManager hooked", getattr(cls, "_glm_ab", False))
for m in ("capture", "run_fullgraph", "profile_memory"):
    check(f"CudaGraphManager.{m} wrapped", hasattr(getattr(cls, m), "__wrapped__"))
check("ModelCudaGraphManager inherits the base run_fullgraph via super()",
      "super().run_fullgraph" in inspect.getsource(cu.ModelCudaGraphManager.run_fullgraph))
check("ModelCudaGraphManager.capture calls super().capture",
      "super().capture" in inspect.getsource(cu.ModelCudaGraphManager.capture))
check("profile_memory calls CudaGraphManager.capture by class name",
      "CudaGraphManager.capture(" in inspect.getsource(cls.profile_memory.__wrapped__))

import vllm.v1.worker.gpu_worker as gw  # noqa: E402

for m in ("glm_ab_switch", "glm_ab_status", "glm_diag_snapshot"):
    check(f"Worker.{m}", callable(getattr(gw.Worker, m, None)))

import vllm.v1.engine.core as core  # noqa: E402

rpc = core.EngineCore.collective_rpc
check("EngineCore.collective_rpc wrapped", getattr(core.EngineCore, "_glm_ab", False) and hasattr(rpc, "__wrapped__"))
check("EngineCore.collective_rpc keeps its signature",
      list(inspect.signature(rpc).parameters) == ["self", "method", "timeout", "args", "kwargs"])

import vllm.v1.worker.gpu.model_runner as mr  # noqa: E402

check("GPUModelRunner.execute_model wrapped (diag)", hasattr(mr.GPUModelRunner.execute_model, "__wrapped__"))
check("GPUModelRunner.sample hooked (argmax)", hasattr(mr.GPUModelRunner.sample, "__wrapped__"))

import triton.runtime.autotuner as at  # noqa: E402
import triton.runtime.jit as jit  # noqa: E402

check("JITFunction._do_compile wrapped", hasattr(jit.JITFunction._do_compile, "__wrapped__"))
check("Autotuner.check_disk_cache wrapped", hasattr(at.Autotuner.check_disk_cache, "__wrapped__"))
check("Autotuner._bench wrapped", hasattr(at.Autotuner._bench, "__wrapped__"))

import vllm.distributed.device_communicators.pynccl as pn  # noqa: E402

check("PyNcclCommunicator.all_reduce wrapped (diag)", hasattr(pn.PyNcclCommunicator.all_reduce, "__wrapped__"))

if os.environ.get("GLM_INDEXER_WARMUP_FIX") == "1":
    import vllm.v1.attention.backends.mla.indexer as idx  # noqa: E402
    check("indexer warm-up keys wrapped (boot fix)",
          hasattr(idx.BuildPrefillChunkMetadataKernel.get_warmup_keys, "__wrapped__"))

import vllm.models.glm5next.nvidia.kda as kda  # noqa: E402

d = kda.fused_recurrent_kda
check("kda dispatcher is nocopy", getattr(d, "__module__", "") == "glm_kda_nocopy")
prev = getattr(d, "__wrapped__", None)
check("nocopy wraps the stash dispatcher", prev is not None and getattr(prev, "__module__", "") == "glm_kda_stash")
from vllm.third_party.flash_linear_attention.ops import kda as fla_kda  # noqa: E402

check("stash dispatcher is not the stock kernel", prev is not fla_kda.fused_recurrent_kda)

bad = [n for n, c in ok if not c]
print(f"{len(ok) - len(bad)}/{len(ok)} PASS")
sys.exit(1 if bad else 0)

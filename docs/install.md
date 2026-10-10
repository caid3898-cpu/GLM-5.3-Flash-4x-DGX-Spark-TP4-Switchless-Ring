# Install the accepted LVKP-S-L2 recipe

Four ARM64 DGX Sparks, a working switched RoCE fabric, Docker with NVIDIA runtime,
SSH between the head and peers, and local checkpoint storage are prerequisites.
The launcher does not configure the network. Keep management/Tailscale separate
from the high-speed fabric; never expose the model port on the public router.
Set both actual HCA names and GID index for your topology. The tested switched
fleet uses two available rails, not the older switchless NCCL patch.

On the head, choose the directory used by the example configuration:

```sh
git clone https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4.git "$HOME/glm53-flash-4x-spark"
cd "$HOME/glm53-flash-4x-spark"
```

## Image and NCCL

Build only on a stopped/non-serving node. The exact base is pinned by digest in
Dockerfile.roce, ARM64 tonyd2wild v11 DFlash2 with vLLM487ecf187. The vendored
RoCEnante subset pins b12x b58f34eaf978277621efced6678e6713fd7122e4 and retains its
license/provenance; the shim installs through .pth without editing vLLM files.

```sh
docker pull --platform linux/arm64 ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
```

Default build tests verify CPU/import/proxy ABI. They are not the four-rank
transport/quality gate. Distribute this built image with docker save/load, or
build the same pinned source on each node. Local image IDs can differ after
independent builds; record each node's actual identity and source checksums.
The deployment used separate IDs, not one universal image hash. This publication
has not performed another fresh-image/fresh-clone model boot.

The host bind mount supplies official NCCL v2.30.7-1, commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`, compiled for sm121. Do not substitute
the historical switchless-patched library. With a CUDA13 toolchain on an idle
build host, a fresh directory and adequate build memory:

```sh
git clone https://github.com/NVIDIA/nccl.git nccl-build
git -C nccl-build checkout --detach 73cf112295c33aee2b895f329f592f2a9b4b0f97
make -C nccl-build -j4 src.build NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
mkdir "$HOME/nccl-2.30.7"
cp -a nccl-build/build/lib/libnccl.so* "$HOME/nccl-2.30.7/"
```

Copy the same resulting directory to peers. Set NCCL_HOST_DIR accordingly. The
launcher explicitly preloads libnccl.so.2.30.7. Toolchain/build-path details can
change binary hashes, so a rebuild requires functional transport checks; this is
a source-pinned build recipe, not a promise of bit-reproducible library output.

## Configure and launch

Prepare weights using [weights.md](weights.md). Put the complete repository at
the same chosen absolute OVERLAY_REMOTE path on all nodes; run the launcher from
the head. Paths must not contain whitespace. Edit management hostnames, fabric
IPs/HCA, checkpoint paths and NCCL location in the example:

```sh
cp .env.example .env
# Edit .env for your actual topology; profiles/current.env holds the flattened accepted model flags.
bash start.sh serve
```

The draft-length policy the scheduler reads ships as `profiles/levers_policy.json` and is synced with the
overlay (`SPEC_PROBE_CONTROL=/overlay/profiles/levers_policy.json`); only the head's scheduler reads it.
Without it the scheduler silently falls back to the launch table, so keep it in place.

Cold caches are supported: FlashInfer, Triton, TileLang, CUDA, Torch/Inductor,
vLLM, b12x, the L2 helper and the overlay's own CUDA extensions compile/populate
their per-node caches under `$OVERLAY_REMOTE/cache` on first use. The overlay
extensions are JIT-built with the image's nvcc from the sources in `overlay/`:
the dense 8-bit kernels (`glm_dense_fast_csrc/`, `cache/glm_dense_fast`), FlashKDA
(`glm_flashkda_csrc/`, about 25 s per node, then cached; it self-tests on every
rank and fails closed to the stock prefill), the mHC BF16 kernel and the routed-MoE
prefill kernels (`glm_pf3_csrc/`, about 15 s per node on the first long prefill;
a failed build leaves every path on stock).
No undocumented precompiled cache is required for correctness. A cold boot can
take substantially longer than a warm boot. Warm cache contents are specific to
source/compiler/GPU versions and are not distributed as a replacement for source.
Do not copy large caches while serving or measuring. Never build on a serving head.

The launcher starts peers3,2,1 then head0 and refuses existing container names or overlapping bind-mounted source paths
before source synchronization, even for stopped preserved containers. Docker
daemon/permission failures stop preflight rather than being treated as absence. `stop` stops but preserves containers, and signals
only verified auxiliary PIDs using Linux pidfds; it never removes containers or
uses process-name-wide killing. A new deployment needs a fresh CTN and mount path.
Restart an unchanged preserved deployment with explicit docker start, peers first.
Memory compaction is an optional operational aid for fragmentation, not a hidden
recipe helper or correctness prerequisite. current.env turns compaction on
(`COMPACT_MEM=1`; it is skipped without a NOPASSWD rule for
`/usr/local/sbin/spark-compact-mem.sh`); boot warm-up is off, so schedule controlled
warmup yourself before measurement. `DRY=1 bash start.sh serve` prints remote commands and the intended synchronization
without making SSH/Docker/rsync calls. It does not validate remote readiness.

Check all-rank source/import/transport logs, localhost:8093 health, model identity,
and the published quality/measurement gates before accepting a rebuild. The
launcher binds localhost. Tailscale forwarding is separately administered and
must be checked independently; no auth keys or service secrets are included here.

"""Production shapes, per-step call counts and GB10 constants for the GLM-5.3-Flash TP4 Marlin tuning (per rank).

Pure Python (no torch): shared by the GPU sweeps, the summarizer, the overlay tests and the CPU tests.

Sources: the 2026-09-28 per-step call budgets (c1 / c4 production traces,
profiled on the fleet), the served tensor shapes (dense/bench_dense_gpu.py, moe/bench_moe_gpu.py) and the live kernel
launch parameters seen in the trace (grid, dynamic smem, registers).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------------------------------
# GB10 (SM121). The GPU scripts re-read these from the device and refuse to run if they differ.
SMS = 48
SMEM_OPTIN = 101376            # cudaDevAttrMaxSharedMemoryPerBlockOptin (99 KiB); the trace shows 101376 on dense
SMEM_PER_SM = 102400           # cudaDevAttrMaxSharedMemoryPerMultiprocessor (100 KiB)
SMEM_RESERVED_PER_BLOCK = 1024  # driver-reserved shared memory per resident CTA on SM8x+
REGS_PER_SM = 65536
MAX_THREADS_PER_SM = 1536
CEIL_GBS = 245.0               # measured copy ceiling used by the budgets

# ---------------------------------------------------------------------------------------------------------------
# Decode step shapes. M = tokens in the target forward: c1 runs k=3 (M=4) or k=7 (M=8), c4 runs M=16 (k=3) or 32.
# CUDA graph capture sizes (profiles/current.env): 1 2 4 6 8 12 16 18 24 32 ... -> tuning buckets (0,4] (4,8]
# (8,16] (16,32]; M above 32 is never touched by the overlay.
M_TUNED = (4, 8, 16, 32)
M_BUCKETS = ((0, 4), (4, 8), (8, 16), (16, 32))      # (lo, hi]: the entry tuned at hi serves this range
STEP_M = {"c1": 4, "c1_k7": 8, "c4": 16, "c4_k7": 32}

# Dense Marlin W8A16 layers (FP8 weights; MXFP8 = e8m0 scales per 32, block-FP8 = BF16 scales per 128x128 block).
# name: K, N (logical), PN (padded, what Marlin sees), fmt, calls per decode step, stream ('main' or 'aux').
DENSE = {
    "kda_in_proj":    dict(K=4096, N=6416, PN=6464, fmt="mxfp8",  calls=34, stream="main"),
    "kda_o_proj":     dict(K=2048, N=4096, PN=4096, fmt="mxfp8",  calls=34, stream="main"),
    "mla_qkv_a":      dict(K=4096, N=2048, PN=2048, fmt="fp8blk", calls=11, stream="main"),
    "mla_q_b":        dict(K=1536, N=4096, PN=4096, fmt="fp8blk", calls=11, stream="main"),
    "mla_o":          dict(K=4096, N=4096, PN=4096, fmt="fp8blk", calls=11, stream="main"),
    "shared_gate_up": dict(K=4096, N=1024, PN=1024, fmt="fp8blk", calls=42, stream="aux"),
    "shared_down":    dict(K=512,  N=4096, PN=4096, fmt="fp8blk", calls=42, stream="aux"),
}
# group size along K of the weight scales, and the Marlin template's group_blocks (= group / 16)
FMT_GROUP = {"mxfp8": 32, "fp8blk": 128}
FMT_GROUP_BLOCKS = {"mxfp8": 2, "fp8blk": 8}


def dense_bytes(name: str) -> int:
    """Weight + scale bytes streamed by one call (per rank)."""
    s = DENSE[name]
    k, pn = s["K"], s["PN"]
    if s["fmt"] == "mxfp8":
        return k * pn + (k // 32) * pn
    return k * pn + (k // 128) * pn * 2


# Routed MoE (NVFP4 Marlin, fused_marlin_moe): E experts, hidden K, intermediate I per rank, top-8, block_m 8.
MOE = dict(E=288, K=4096, I=512, TOPK=8, CLAMP=10.0, calls=42)
MOE_GEMMS = {
    # size_n, size_k as the Marlin op sees them; top_k argument; bytes per distinct expert (weights + fp8 scales)
    "gate_up": dict(size_n=2 * 512, size_k=4096, top_k=8, bytes_per_expert=4096 * 1024 // 2 + 256 * 1024),
    "down":    dict(size_n=4096, size_k=512, top_k=1, bytes_per_expert=512 * 4096 // 2 + 32 * 4096),
}
MOE_BYTES_PER_EXPERT = sum(g["bytes_per_expert"] for g in MOE_GEMMS.values())   # 3,538,944

# Distinct experts per layer (U) at each M: the review's c1-like 20-28 and c4-like 64-75 plus the M=8/32 points of
# the 09-28 bench. Tuning minimises the mean time over these (uniform weights) and rejects a config that loses more
# than MOE_MAX_REGRESS at any single U.
MOE_U = {4: (18, 20, 22, 25, 28), 8: (32, 40, 48), 16: (60, 64, 70, 75, 80), 32: (110, 130, 160)}
MOE_MAX_REGRESS = 0.01

# Live trace kernel parameters (stock choices), for the CPU model tests.
TRACE_STOCK = {
    "moe": dict(grid=144, threads=128, thread_n_blocks=8, thread_k_blocks=4, smem=32768, regs=94),
    "kda_in_proj": dict(grid=48, threads=128, thread_n_blocks=4, thread_k_blocks=8, smem=101376),
    "kda_o_proj": dict(grid=48, threads=256, thread_n_blocks=8, thread_k_blocks=8, smem=101376),
    "mla": dict(grid=48, threads=256, thread_n_blocks=8, thread_k_blocks=8, smem=101376),
    "shared_gate_up": dict(grid=48, threads=128, thread_n_blocks=4, thread_k_blocks=8, smem=101376),
    "shared_down": dict(grid=48, threads=256, thread_n_blocks=8, thread_k_blocks=8, smem=101376),
}

# Live (profiled) vs isolated stock kernel times, us (glm-prof-20260928 traces via tools/trace_overlap.py; isolated
# from ../glm-kernels-20260928/dense/dense.jsonl stock medians, which include the in_proj slice copy of ~2 us).
LIVE_VS_ISOLATED = {
    ("kda_in_proj", 4): (137.3, 119.9), ("kda_in_proj", 16): (143.7, 121.9),
    ("mla_qkv_a", 4): (40.8, 39.5), ("mla_q_b", 4): (31.4, 29.9), ("mla_o", 4): (77.7, 75.5),
    ("mla_qkv_a", 16): (41.8, 40.7), ("mla_q_b", 16): (33.0, 30.7), ("mla_o", 16): (81.9, 77.1),
}


def m_bucket(m: int):
    """Tuned M whose entry serves this token count, or None (not a decode shape)."""
    for lo, hi in M_BUCKETS:
        if lo < m <= hi:
            return hi
    return None

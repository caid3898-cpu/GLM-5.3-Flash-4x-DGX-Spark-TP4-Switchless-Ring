"""CPU model of vLLM's Marlin launch-config logic (pure Python), for candidate enumeration and tests.

Mirrors, line for line where it matters, vLLM d26f46973:
  dense  csrc/libtorch_stable/quantization/marlin/marlin.cu  get_scales_cache_size / get_kernel_cache_size /
         is_valid_config / determine_exec_config / the marlin_mm small-grid switch to (128, 64, 128)
  MoE    csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu   the same plus sh_block_meta, blocks_per_sm and the
         thread_k / thread_n / blocks_per_sm override path (threads = thread_k * thread_n / 64)
  generated kernels: generate_kernels.py (THREAD_CONFIGS, 256-thread rule, stages = 4)

Nothing here launches anything; the GPU scripts use it to enumerate candidates and to predict the stock choice,
and they cross-check the prediction against the trace / the device (CPU tests pin the trace's choices).
"""
from __future__ import annotations

from dataclasses import dataclass

import shapes as S

MIN_THREAD_N = 64
MIN_THREAD_K = 64
MAX_THREAD_N = 256


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Tile:
    thread_k: int
    thread_n: int
    threads: int

    @property
    def key(self) -> str:
        return f"{self.thread_k}x{self.thread_n}x{self.threads}"


# vLLM's config tables (ops.cu / marlin.cu), in priority order
SMALL_BATCH = (Tile(128, 128, 256), Tile(64, 128, 128), Tile(128, 64, 128))
LARGE_BATCH = (Tile(64, 256, 256), Tile(64, 128, 128), Tile(128, 64, 128))


def generated_tiles(m_blocks: float):
    """Tiles instantiated by generate_kernels.py for thread_m_blocks = m_blocks (0.5 = m_block_size_8)."""
    out = []
    for tk, tn, th in ((128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)):
        if th == 256:
            if m_blocks <= 1 and (tk, tn) != (128, 128):
                continue
            if m_blocks > 1 and (tk, tn) != (64, 256):
                continue
        out.append(Tile(tk, tn, th))
    return out


# ------------------------------------------------------------------------------------------------ cache sizes
def scales_cache_size(tile: Tile, group_size: int, stages: int, has_act_order=False, is_k_full=True) -> int:
    tb_n, tb_k = tile.thread_n, tile.thread_k
    if group_size == -1:
        tb_groups = 1
    elif group_size == 0:
        tb_groups = div_ceil(tb_k, 32)
    else:
        tb_groups = div_ceil(tb_k, group_size)
    if has_act_order and not is_k_full:
        return max(tb_groups * stages * 2, 32) * tb_n * 2
    return tb_groups * tb_n * 2 * stages


def kernel_cache_size(tile: Tile, thread_m_blocks: int, num_bits: int, group_size: int, stages: int,
                      moe: bool, a_8bit: bool = False) -> int:
    """get_kernel_cache_size (dense when moe=False; the MoE version adds the block metadata)."""
    pack_factor = 32 // num_bits
    tb_k, tb_n, tb_m = tile.thread_k, tile.thread_n, thread_m_blocks * 16
    sh_a = stages * (tb_m * tb_k) * (1 if a_8bit else 2)
    sh_b = stages * (tb_k * tb_n // pack_factor) * 4
    sh_red = tb_m * (tb_n + 8) * 2
    sh_bias = tb_n * 2
    tmp = (sh_red if sh_b > sh_red else sh_b) + sh_bias
    tmp = max(max(sh_b, sh_red), tmp)
    sh_s = scales_cache_size(tile, group_size, stages)
    total = tmp + sh_a + sh_s
    if moe:
        total += tb_m * 16
    return total


def is_valid(tile: Tile, prob_n: int, prob_k: int, thread_m_blocks: int, num_bits: int, group_size: int,
             stages: int, max_shared_mem: int, moe: bool) -> bool:
    if -1 in (tile.thread_k, tile.thread_n, tile.threads):
        return False
    if prob_k % tile.thread_k or prob_n % tile.thread_n:
        return False
    if tile.thread_n < MIN_THREAD_N or tile.thread_k < MIN_THREAD_K:
        return False
    if tile.threads < 128:
        return False
    return kernel_cache_size(tile, thread_m_blocks, num_bits, group_size, stages, moe) <= max_shared_mem


# ------------------------------------------------------------------------------------------------ MoE (ops.cu)
@dataclass(frozen=True)
class MoeExec:
    tile: Tile
    blocks_per_sm: int

    @property
    def grid(self) -> int:
        return S.SMS * self.blocks_per_sm

    @property
    def smem(self) -> int:
        """Dynamic shared memory the launch requests (ops.cu: optin, or optin / bps - 1024 when bps > 1)."""
        return S.SMEM_OPTIN if self.blocks_per_sm == 1 else S.SMEM_OPTIN // self.blocks_per_sm - 1024


def moe_auto(prob_m: int, prob_n: int, prob_k: int, top_k: int, block_m: int, num_bits=4, group_size=16,
             stages=4, regs=None) -> MoeExec:
    """determine_exec_config (ops.cu:274). regs: {tile.key: registers per thread}; unknown tiles assume 128
    (the value only matters when it binds, which it does not on GB10: shared memory binds first)."""
    thread_m_blocks = div_ceil(block_m, 16)
    table = LARGE_BATCH if thread_m_blocks > 1 else SMALL_BATCH
    regs = regs or {}
    best, count = MoeExec(Tile(-1, -1, -1), 1), 0
    for tile in table:
        if not is_valid(tile, prob_n, prob_k, thread_m_blocks, num_bits, group_size, stages,
                        S.SMEM_OPTIN - 512, moe=True):
            continue
        if tile not in generated_tiles(0.5 if block_m == 8 else thread_m_blocks):
            continue
        cache = kernel_cache_size(tile, thread_m_blocks, num_bits, group_size, stages, moe=True)
        reg_size = max(regs.get(tile.key, 128), 1) * tile.threads * 4
        allow = min(255 * 1024 // reg_size, S.SMEM_OPTIN // (cache + 1536))
        allow = max(min(allow, 4), 1) if thread_m_blocks == 1 else max(min(allow, 2), 1)
        if prob_n // tile.thread_n * prob_m * top_k * 4 < S.SMS * allow:
            allow = max(prob_n // tile.thread_n * prob_m * top_k * 4 // S.SMS, 1)
        if allow > count:
            count = allow
            best = MoeExec(tile, count)
    return best


def moe_override_valid(thread_k: int, thread_n: int, blocks_per_sm: int, prob_n: int, prob_k: int, block_m: int,
                       num_bits=4, group_size=16, stages=4, regs_per_thread=128):
    """Would ops.moe_wna16_marlin_gemm accept thread_k/thread_n/blocks_per_sm, and are all CTAs co-resident?
    Returns (ok, reason)."""
    tile = Tile(thread_k, thread_n, thread_k * thread_n // 64)
    thread_m_blocks = div_ceil(block_m, 16)
    bps = 1 if blocks_per_sm == -1 else blocks_per_sm
    if prob_n % thread_n or prob_k % thread_k:
        return False, "divisibility"
    smem = S.SMEM_OPTIN if bps == 1 else S.SMEM_OPTIN // bps - 1024
    if not is_valid(tile, prob_n, prob_k, thread_m_blocks, num_bits, group_size, stages, smem, moe=True):
        return False, "shared memory / tile limits"
    if tile not in generated_tiles(0.5 if block_m == 8 else thread_m_blocks):
        return False, "kernel not generated"
    # the grid is persistent and stream-K CTAs spin on each other: all sms * bps CTAs must fit at once
    if bps * (smem + S.SMEM_RESERVED_PER_BLOCK) > S.SMEM_PER_SM:
        return False, "not co-resident (shared memory per SM)"
    if bps * tile.threads * regs_per_thread > S.REGS_PER_SM or bps * tile.threads > S.MAX_THREADS_PER_SM:
        return False, "not co-resident (registers / threads)"
    return True, ""


def moe_candidates(gemm: str, block_m: int = 8, max_bps: int = 4):
    """Exposed-parameter candidates for one MoE GEMM: [(thread_k, thread_n, blocks_per_sm)] that the op accepts."""
    g = S.MOE_GEMMS[gemm]
    out = []
    for tile in generated_tiles(0.5 if block_m == 8 else div_ceil(block_m, 16)):
        for bps in range(1, max_bps + 1):
            ok, _ = moe_override_valid(tile.thread_k, tile.thread_n, bps, g["size_n"], g["size_k"], block_m)
            if ok:
                out.append((tile.thread_k, tile.thread_n, bps))
    return out


# ------------------------------------------------------------------------------------------------ dense (marlin.cu)
def dense_m_variant(m: int):
    """(thread_m_blocks, m_block_size_8) marlin_mm picks for M <= 64 (one split): 16-bit activations."""
    thread_m_blocks = min(div_ceil(m, 16), 4)
    return thread_m_blocks, m <= 8


def dense_auto(m: int, prob_n: int, prob_k: int, fmt: str, stages=4) -> Tile:
    """determine_exec_config + the small-grid switch in marlin_mm (marlin.cu:452-468), bps is always 1."""
    group = S.FMT_GROUP[fmt]
    thread_m_blocks, m8 = dense_m_variant(m)
    table = LARGE_BATCH if thread_m_blocks > 1 else SMALL_BATCH
    chosen = Tile(-1, -1, -1)
    for tile in table:
        if not is_valid(tile, prob_n, prob_k, thread_m_blocks, 8, group, stages, S.SMEM_OPTIN - 512, moe=False):
            continue
        if tile not in generated_tiles(0.5 if m8 else thread_m_blocks):
            continue
        chosen = tile
        break
    if chosen.thread_n != -1:
        if prob_n // chosen.thread_n * div_ceil(m, thread_m_blocks * 16) * 4 <= S.SMS:
            small = Tile(128, 64, 128)
            if is_valid(small, prob_n, prob_k, thread_m_blocks, 8, group, stages, S.SMEM_OPTIN, moe=False):
                chosen = small
    return chosen


# ------------------------------------------------------------------------------------------------ JIT candidates
# Tiles the Marlin template supports without edits: threads = thread_k * thread_n / 64 keeps b_sh_wr_iters == 2
# (every generated config has it; the main loop issues the next stage's fetch at k == b_sh_wr_iters - 2), the
# per-warp n-extent needs thread_n >= 64 and the k-warp reduction (red_off = threads / thread_n for 8-bit weights)
# a power of two. EXTENDED tiles use b_sh_wr_iters == 4 (threads = thread_k * thread_n / 128): the template looks
# generic in b_sh_wr_iters but no shipped config exercises it, so they are opt-in and numerics-gated like all.
JIT_TILES = (
    Tile(64, 128, 128), Tile(128, 64, 128),
    Tile(128, 128, 256), Tile(64, 256, 256), Tile(256, 64, 256),
    Tile(256, 128, 512), Tile(128, 256, 512), Tile(512, 64, 512),
)
JIT_TILES_EXTENDED = (Tile(128, 128, 128), Tile(256, 64, 128), Tile(64, 256, 128), Tile(256, 128, 256),
                      Tile(128, 256, 256), Tile(512, 64, 256))


def b_sh_wr_iters(tile: Tile, num_bits: int) -> int:
    pack = 32 // num_bits
    b_sh_stride = ((tile.thread_n // 16) * 16 * 16 // pack) // 4
    b_thread_vecs = 1 if num_bits == 4 else 2
    return b_sh_stride * (tile.thread_k // 16) // (tile.threads * b_thread_vecs)


def template_ok(tile: Tile, num_bits: int, group_blocks: int) -> tuple[bool, str]:
    """Static constraints of marlin_template.h for 16-bit activations."""
    if tile.thread_n < 64 or tile.thread_k < 64 or tile.threads < 128 or tile.threads % 32:
        return False, "min tile / threads"
    it = b_sh_wr_iters(tile, num_bits)
    if it < 2 or it & (it - 1):
        return False, f"b_sh_wr_iters={it}"
    tb_n_warps = tile.thread_n // 16 // 4
    if tb_n_warps < 1 or (tile.threads // 32) % tb_n_warps:
        return False, "warp layout"
    b_sh_stride = ((tile.thread_n // 16) * 16 * 16 // (32 // num_bits)) // 4
    if b_sh_stride % tile.threads and tile.threads % b_sh_stride:
        return False, "B fetch layout"  # fetch_to_shared: count = div_ceil(b_sh_stride, threads)
    if 2 * (tile.thread_n // 16) > tile.threads:
        return False, "write-out layout"
    b_sh_stride_threads = b_sh_stride // (1 if num_bits == 4 else 2)
    red_off = tile.threads // b_sh_stride_threads // 2
    if red_off >= 1 and red_off & (red_off - 1):
        return False, "k-warp reduction"
    tkb = tile.thread_k // 16
    if group_blocks > 0:
        if group_blocks < tkb and tkb % group_blocks:
            return False, "group straddles k tile"
        if group_blocks >= tkb and group_blocks % tkb:
            return False, "k tile straddles group"
    return True, ""


def stages_ok(tile: Tile, group_blocks: int, stages: int) -> bool:
    """fetch_to_shared loads a scale row when `pipe % div_ceil(group_blocks, thread_k_blocks) == 0`, with pipe
    the stage slot: a group spanning g > 1 k tiles needs stages % g == 0 to stay aligned (4 always is)."""
    if stages < 2:
        return False
    g = div_ceil(group_blocks, tile.thread_k // 16) if group_blocks > 0 else 1
    return stages % g == 0


@dataclass(frozen=True)
class JitCfg:
    """One JIT dense launch: template instance (fmt, m variant, tile, stages) + runtime grid / smem / n_out."""
    fmt: str
    m_variant: str          # "m8" (M <= 8, m_block_size_8), "m16" (thread_m_blocks 1), "m32" (2), "m48", "m64"
    tile: Tile
    stages: int
    grid: int
    smem: int               # dynamic shared memory bytes requested at launch

    @property
    def thread_m_blocks(self) -> int:
        return {"m8": 1, "m16": 1, "m32": 2, "m48": 3, "m64": 4}[self.m_variant]

    @property
    def m_max(self) -> int:
        return {"m8": 8, "m16": 16, "m32": 32, "m48": 48, "m64": 64}[self.m_variant]

    @property
    def key(self) -> str:
        return f"{self.fmt}/{self.m_variant}/{self.tile.key}/s{self.stages}/g{self.grid}/sm{self.smem}"

    def to_json(self) -> dict:
        return {"fmt": self.fmt, "m_variant": self.m_variant, "thread_k": self.tile.thread_k,
                "thread_n": self.tile.thread_n, "threads": self.tile.threads, "stages": self.stages,
                "grid": self.grid, "smem": self.smem}

    @staticmethod
    def from_json(d: dict) -> "JitCfg":
        return JitCfg(d["fmt"], d["m_variant"], Tile(d["thread_k"], d["thread_n"], d["threads"]), d["stages"],
                      d["grid"], d["smem"])


def m_variant_for(m: int) -> str:
    if m <= 8:
        return "m8"
    return {1: "m16", 2: "m32", 3: "m48", 4: "m64"}[min(div_ceil(m, 16), 4)]


def jit_smem_needed(fmt: str, m_variant: str, tile: Tile, stages: int) -> int:
    return kernel_cache_size(tile, {"m8": 1, "m16": 1, "m32": 2, "m48": 3, "m64": 4}[m_variant], 8,
                             S.FMT_GROUP[fmt], stages, moe=False)


def max_coresident(threads: int, smem: int, regs_per_thread: int = 128) -> int:
    by_smem = S.SMEM_PER_SM // (smem + S.SMEM_RESERVED_PER_BLOCK)
    by_regs = S.REGS_PER_SM // max(1, threads * regs_per_thread)
    by_thr = S.MAX_THREADS_PER_SM // threads
    return max(0, min(by_smem, by_regs, by_thr))


def dense_jit_candidates(name: str, m: int, stages=(4,), extra_stages=(2, 3, 6), extra_stage_tiles=None,
                         extended=False, grids_per_sm=(1, 2, 3), smem_modes=("stock", "exact"),
                         regs_per_thread=128):
    """JIT candidates for one dense shape at token count m. The stock-equivalent config (same tile, stages 4,
    grid = SMS, smem = optin) is always first: it is the harness parity check."""
    s = S.DENSE[name]
    fmt, pn, k = s["fmt"], s["PN"], s["K"]
    gb = S.FMT_GROUP_BLOCKS[fmt]
    mv = m_variant_for(m)
    stock_tile = dense_auto(m, pn, k, fmt)
    tiles = list(JIT_TILES) + (list(JIT_TILES_EXTENDED) if extended else [])
    if extra_stage_tiles is None:
        extra_stage_tiles = {Tile(128, 64, 128), Tile(64, 128, 128), Tile(128, 128, 256), Tile(256, 64, 256)}
    out = [JitCfg(fmt, mv, stock_tile, 4, S.SMS, S.SMEM_OPTIN)]
    seen = {out[0]}
    for tile in tiles:
        if pn % tile.thread_n or k % tile.thread_k:
            continue
        if not template_ok(tile, 8, gb)[0]:
            continue
        st_list = list(stages) + ([x for x in extra_stages if x not in stages] if tile in extra_stage_tiles else [])
        for st in st_list:
            if not stages_ok(tile, gb, st):
                continue
            need = jit_smem_needed(fmt, mv, tile, st)
            if need > S.SMEM_OPTIN:
                continue
            for bps in grids_per_sm:
                for mode in smem_modes:
                    if mode == "stock":
                        smem = S.SMEM_OPTIN if bps == 1 else S.SMEM_OPTIN // bps - 1024
                    else:
                        smem = need
                    if smem < need:
                        continue
                    if max_coresident(tile.threads, smem, regs_per_thread) < bps:
                        continue
                    c = JitCfg(fmt, mv, tile, st, S.SMS * bps, smem)
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
    return out


def jit_instances(cands):
    """Distinct template instances (fmt, m_variant, tile, stages) needed by a candidate list."""
    return sorted({(c.fmt, c.m_variant, c.tile, c.stages) for c in cands},
                  key=lambda t: (t[0], t[1], t[2].thread_k, t[2].thread_n, t[2].threads, t[3]))


# ------------------------------------------------------------------------------------------------ MoE JIT
@dataclass(frozen=True)
class MoeJitCfg:
    """One JIT MoE launch (NVFP4 weights): tile, stages, grid, smem. block_m 8 only (the decode tile)."""
    tile: Tile
    stages: int
    grid: int
    smem: int

    @property
    def key(self) -> str:
        return f"nvfp4/m8/{self.tile.key}/s{self.stages}/g{self.grid}/sm{self.smem}"

    def to_json(self) -> dict:
        return {"thread_k": self.tile.thread_k, "thread_n": self.tile.thread_n, "threads": self.tile.threads,
                "stages": self.stages, "grid": self.grid, "smem": self.smem}

    @staticmethod
    def from_json(d: dict) -> "MoeJitCfg":
        return MoeJitCfg(Tile(d["thread_k"], d["thread_n"], d["threads"]), d["stages"], d["grid"], d["smem"])


MOE_JIT_TILES = (Tile(64, 128, 128), Tile(128, 64, 128), Tile(128, 128, 256), Tile(64, 256, 256),
                 Tile(256, 64, 256))


def moe_jit_candidates(gemm: str, stages=(2, 3, 4, 6), grids_per_sm=(1, 2, 3, 4), regs_per_thread=96):
    g = S.MOE_GEMMS[gemm]
    out = []
    for tile in MOE_JIT_TILES:
        if g["size_n"] % tile.thread_n or g["size_k"] % tile.thread_k:
            continue
        if not template_ok(tile, 4, 1)[0]:
            continue
        for st in stages:
            if not stages_ok(tile, 1, st):
                continue
            need = kernel_cache_size(tile, 1, 4, 16, st, moe=True)
            for bps in grids_per_sm:
                smem = need
                if max_coresident(tile.threads, smem, regs_per_thread) < bps:
                    continue
                out.append(MoeJitCfg(tile, st, S.SMS * bps, smem))
    return out

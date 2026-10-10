// Decode-specialised W8A16 GEMM over vLLM's served Marlin FP8 buffers (MXFP8 e8m0 / block-FP8 bf16 scales).
//
//   out[M, N] (bf16) = x[M, K] (bf16) @ W[K, N],  M <= 32, weights read in place from the Marlin int32 tensor
//   [K/16, 4*PN] (8-bit, non-a8 layout of gptq_marlin_repack) and the Marlin-permuted scales.
//
// Design (all choices aimed at a pure DRAM stream on GB10):
//   * Work unit = TPW whole 64-column Marlin tiles x a K range. One warp streams its tile rows into registers.
//     XPOSE = 0: a lane reads its own 32 contiguous bytes of each 1 KB tile row (two 16-byte loads 32 bytes apart;
//     each load instruction touches all 32 sectors of the row half-used, the pair covers them fully).
//     XPOSE = 1: every load instruction reads 512 contiguous bytes (lane L: bytes 16L..16L+15, then 512 + 16L..),
//     and the fragments are redistributed through a 1 KB per-warp shared buffer (XOR-swizzled 16-byte chunks,
//     conflict-free on both sides) right before use. Either way no shared-memory pipeline for weights.
//   * Software pipeline: a D-deep register ring of weight + scale loads, activations one step ahead.
//   * Dequant = Marlin's bit construction + __hmul2 by the scale (bit-identical operands), then Marlin's M <= 8
//     transposed m16n8k16 MMA (weights as A, 8 activation rows as B); MG such groups cover M <= 8*MG.
//   * CTA = KS warps splitting the CTA's K range; the KS partial tiles are summed in shared memory in fixed warp
//     order. Optional deterministic split over K across GS CTAs (grid.y) for grids that would otherwise leave SMs
//     idle: fp32 partials in a workspace, the last-arriving CTA of a column block sums them in split order and resets
//     its ticket (CUDA-graph safe). No atomics on data, so results are run-to-run deterministic.
//   * Epilogue writes the unpadded bf16 [M, N] directly (no padded output + slice copy).
//
// Numerics: operands identical to Marlin's; fp32 accumulation; only the fp32 summation order differs.
// Credits: Marlin (Frantar & Alistarh, IST-DASLab; Neural Magic; vLLM contributors) for the weight layout, the
// dequantisation bit tricks, the transposed small-M MMA and the stream-K/global-reduce ideas this simplifies.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <type_traits>

#include "glmk_common.cuh"

namespace glmk {

struct DenseArgs {
  const __nv_bfloat16* x;
  long long lda;
  const uint32_t* w;
  const uint8_t* s;
  __nv_bfloat16* out;
  long long ldc;
  float* ws;
  int* cnt;
  int M, N, PN, KT, ntiles;
  int vec_out;
};

template <bool MX, int MG, int TPW, int D, int LDG, int XPOSE>
__global__ void __launch_bounds__(256) dense_w8a16_kernel(const DenseArgs p) {
  static_assert(D % 2 == 0, "D must be even (activation double buffer is indexed by step parity)");
  constexpr int MPAD = 8 * MG;
  constexpr int OSTR = 64 * TPW + 4;  // fp32 smem row stride: 2*OSTR = 8 (mod 32) -> conflict-free fragment stores
  using SV = std::conditional_t<MX, uint2, uint4>;
  extern __shared__ float4 o_s4[];
  float* o_s = reinterpret_cast<float*>(o_s4);
  __shared__ int s_last;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, KS = blockDim.x >> 5;
  // XPOSE staging buffer of this warp: 64 x 16-byte chunks after the reduction tile
  uint4* xbuf = reinterpret_cast<uint4*>(o_s4 + (MPAD * OSTR) / 4) + warp * 64;
  const int GS = gridDim.y, gsi = blockIdx.y;
  const int tile0 = blockIdx.x * TPW;
  const int ntv = min(TPW, p.ntiles - tile0);

  const int r0 = (int)((long long)p.KT * gsi / GS);
  const int r1 = (int)((long long)p.KT * (gsi + 1) / GS);
  const int kb = r0 + (int)((long long)(r1 - r0) * warp / KS);
  const int ke = r0 + (int)((long long)(r1 - r0) * (warp + 1) / KS);
  const int n = ke - kb;

  const size_t wrow = (size_t)4 * p.PN;
  const uint32_t* wp = p.w + (size_t)tile0 * 256 + lane * (XPOSE ? 4 : 8);
  const int scol = tile0 * 64 + 8 * (lane >> 2);

  const __nv_bfloat16* xr[MG];
  bool xv[MG];
#pragma unroll
  for (int g = 0; g < MG; ++g) {
    const int m = 8 * g + (lane >> 2);
    xv[g] = m < p.M;
    xr[g] = p.x + (size_t)(xv[g] ? m : 0) * p.lda + 2 * (lane & 3);
  }

  float acc[TPW][4][MG][4];
#pragma unroll
  for (int t = 0; t < TPW; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int g = 0; g < MG; ++g)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[t][j][g][r] = 0.f;

  uint4 wq[D][TPW][2];
  SV sq[D][TPW];
  uint32_t xb[2][MG][2];

  auto issue = [&](uint4 (&wd)[TPW][2], SV (&sd)[TPW], int kt) {
    const uint32_t* wr = wp + (size_t)kt * wrow;
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      if (t < ntv) {
        wd[t][0] = ld_stream_v4<LDG>(wr + t * 256);
        wd[t][1] = ld_stream_v4<LDG>(wr + t * 256 + (XPOSE ? 128 : 4));
        if constexpr (MX) {
          sd[t] = ld_nc_v2(p.s + (size_t)(kt >> 1) * p.PN + scol + t * 64);
        } else {
          sd[t] = ld_nc_v4(p.s + ((size_t)(kt >> 3) * p.PN + scol + t * 64) * 2);
        }
      }
    }
  };
  auto issue_x = [&](uint32_t (&xd)[MG][2], int kt) {
#pragma unroll
    for (int g = 0; g < MG; ++g) {
      if (xv[g]) {
        const __nv_bfloat16* a = xr[g] + (size_t)kt * 16;
        xd[g][0] = ld_nc_b32(a);
        xd[g][1] = ld_nc_b32(a + 8);
      } else {
        xd[g][0] = 0u;
        xd[g][1] = 0u;
      }
    }
  };
  auto compute = [&](const uint4 (&wd)[TPW][2], const SV (&sd)[TPW], const uint32_t (&xd)[MG][2]) {
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      if (t < ntv) {
        uint4 sv;
        if constexpr (MX) {
          sv = make_uint4(sd[t].x, sd[t].y, 0u, 0u);
        } else {
          sv = sd[t];
        }
        if constexpr (XPOSE) {
          // chunk c (16 bytes of the 1 KB row) lives at c ^ ((c >> 3) & 1): the contiguous stores (c = lane,
          // 32 + lane) and the fragment reads (c = 2 lane, 2 lane + 1) are both bank-conflict free.
          const int c0 = lane, c1 = 32 + lane, r0 = 2 * lane, r1 = 2 * lane + 1;
          xbuf[c0 ^ ((c0 >> 3) & 1)] = wd[t][0];
          xbuf[c1 ^ ((c1 >> 3) & 1)] = wd[t][1];
          __syncwarp();
          const uint4 a = xbuf[r0 ^ ((r0 >> 3) & 1)];
          const uint4 b = xbuf[r1 ^ ((r1 >> 3) & 1)];
          __syncwarp();
          tile_fp8<MX, MG>(acc[t], a, b, sv, xd);
        } else {
          tile_fp8<MX, MG>(acc[t], wd[t][0], wd[t][1], sv, xd);
        }
      }
    }
  };

  // ---- pipelined K loop: loads for step i + D - 1 are in flight while step i is multiplied ----
#pragma unroll
  for (int u = 0; u < D - 1; ++u)
    if (u < n) issue(wq[u], sq[u], kb + u);
  if (n > 0) issue_x(xb[0], kb);
  for (int base = 0; base < n; base += D) {
#pragma unroll
    for (int u = 0; u < D; ++u) {
      const int i = base + u;
      if (i + D - 1 < n) issue(wq[(u + D - 1) % D], sq[(u + D - 1) % D], kb + i + D - 1);
      if (i + 1 < n) issue_x(xb[(u + 1) & 1], kb + i + 1);
      if (i < n) compute(wq[u], sq[u], xb[u & 1]);
    }
  }

  // ---- intra-CTA reduction over the KS warps, fixed order 0..KS-1 ----
  for (int w = 0; w < KS; ++w) {
    if (warp == w) {
#pragma unroll
      for (int t = 0; t < TPW; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int g = 0; g < MG; ++g)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int m = 8 * g + 2 * (lane & 3) + (r & 1);
              const int c = t * 64 + 16 * j + (lane >> 2) + 8 * (r >> 1);
              float* dst = &o_s[m * OSTR + c];
              *dst = (w == 0) ? acc[t][j][g][r] : (*dst + acc[t][j][g][r]);
            }
    }
    __syncthreads();
  }

  const int Mv = min(p.M, MPAD);
  constexpr int CH = TPW * 8;  // 8-column chunks per row
  auto store_chunk = [&](int m, int c8, const float* v) {
    const int n0 = tile0 * 64 + c8 * 8;
    __nv_bfloat16* dst = p.out + (size_t)m * p.ldc + n0;
    if (p.vec_out && n0 + 8 <= p.N) {
      uint4 o;
      o.x = as_u32(__floats2bfloat162_rn(v[0], v[1]));
      o.y = as_u32(__floats2bfloat162_rn(v[2], v[3]));
      o.z = as_u32(__floats2bfloat162_rn(v[4], v[5]));
      o.w = as_u32(__floats2bfloat162_rn(v[6], v[7]));
      *reinterpret_cast<uint4*>(dst) = o;
    } else {
      for (int e = 0; e < 8 && n0 + e < p.N; ++e) dst[e] = __float2bfloat16_rn(v[e]);
    }
  };

  if (GS == 1) {
    for (int idx = threadIdx.x; idx < Mv * CH; idx += blockDim.x) {
      const int m = idx / CH, c8 = idx % CH;
      if (c8 >= ntv * 8 || tile0 * 64 + c8 * 8 >= p.N) continue;
      float v[8];
      const float4 a = *reinterpret_cast<const float4*>(&o_s[m * OSTR + c8 * 8]);
      const float4 b = *reinterpret_cast<const float4*>(&o_s[m * OSTR + c8 * 8 + 4]);
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
      store_chunk(m, c8, v);
    }
    return;
  }

  // ---- deterministic split-K across the GS CTAs of this column block ----
  for (int idx = threadIdx.x; idx < Mv * CH * 2; idx += blockDim.x) {
    const int m = idx / (CH * 2), c4 = idx % (CH * 2);
    if (c4 >= ntv * 16) continue;
    float4* dst = reinterpret_cast<float4*>(p.ws + ((size_t)gsi * MPAD + m) * p.PN + tile0 * 64 + c4 * 4);
    *dst = *reinterpret_cast<const float4*>(&o_s[m * OSTR + c4 * 4]);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old = atomicAdd(&p.cnt[blockIdx.x], 1);
    s_last = (old == GS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < Mv * CH; idx += blockDim.x) {
    const int m = idx / CH, c8 = idx % CH;
    if (c8 >= ntv * 8 || tile0 * 64 + c8 * 8 >= p.N) continue;
    float v[8];
#pragma unroll 1
    for (int g = 0; g < GS; ++g) {
      const float4* src = reinterpret_cast<const float4*>(p.ws + ((size_t)g * MPAD + m) * p.PN + tile0 * 64 + c8 * 8);
      const float4 a = ld_cg_f4(src), b = ld_cg_f4(src + 1);
      if (g == 0) {
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
      } else {
        v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w; v[4] += b.x; v[5] += b.y; v[6] += b.z; v[7] += b.w;
      }
    }
    store_chunk(m, c8, v);
  }
  if (threadIdx.x == 0) p.cnt[blockIdx.x] = 0;
}

// ------------------------------------------------------------------------------------------------------------------
// host side
// ------------------------------------------------------------------------------------------------------------------
using KernelFn = void (*)(const DenseArgs);

template <bool MX, int MG, int TPW, int D, int XP>
static KernelFn pick_ldg(int ldg) {
  switch (ldg) {
    case 0: return dense_w8a16_kernel<MX, MG, TPW, D, 0, XP>;
#ifdef GLMK_ALL_LDG
    case 1: return dense_w8a16_kernel<MX, MG, TPW, D, 1, XP>;
    case 2: return dense_w8a16_kernel<MX, MG, TPW, D, 2, XP>;
#endif
    default: return nullptr;
  }
}

template <bool MX, int MG, int TPW, int D>
static KernelFn pick_xp(int xp, int ldg) {
  return xp ? pick_ldg<MX, MG, TPW, D, 1>(ldg) : pick_ldg<MX, MG, TPW, D, 0>(ldg);
}

template <bool MX, int MG, int TPW>
static KernelFn pick_d(int d, int xp, int ldg) {
  switch (d) {
    case 2: return pick_xp<MX, MG, TPW, 2>(xp, ldg);
    case 4: return pick_xp<MX, MG, TPW, 4>(xp, ldg);
    default: return nullptr;
  }
}

template <bool MX, int MG>
static KernelFn pick_tpw(int tpw, int d, int xp, int ldg) {
  switch (tpw) {
    case 1: return pick_d<MX, MG, 1>(d, xp, ldg);
    case 2: return pick_d<MX, MG, 2>(d, xp, ldg);
    default: return nullptr;
  }
}

template <bool MX>
static KernelFn pick_mg(int mg, int tpw, int d, int xp, int ldg) {
  switch (mg) {
    case 1: return pick_tpw<MX, 1>(tpw, d, xp, ldg);
    case 2: return pick_tpw<MX, 2>(tpw, d, xp, ldg);
    case 4: return pick_tpw<MX, 4>(tpw, d, xp, ldg);
    default: return nullptr;
  }
}

static int mg_for(int M) { return M <= 8 ? 1 : (M <= 16 ? 2 : 4); }

static KernelFn dense_kernel_for(bool mx, int M, int tpw, int d, int xp, int ldg) {
  const int mg = mg_for(M);
  return mx ? pick_mg<true>(mg, tpw, d, xp, ldg) : pick_mg<false>(mg, tpw, d, xp, ldg);
}

// reduction tile [8*MG][64*TPW + 4] fp32, then (XPOSE) 1 KB per warp
static size_t dense_smem(int M, int tpw, int ks, int xp) {
  return (size_t)8 * mg_for(M) * (64 * tpw + 4) * sizeof(float) + (xp ? (size_t)ks * 1024 : 0);
}

void dense_w8a16(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, torch::Tensor& out,
                 torch::Tensor& ws, torch::Tensor& cnt, bool mx, int64_t N, int64_t tpw, int64_t ks, int64_t gs,
                 int64_t d, int64_t xp, int64_t ldg) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x: bf16 [M,K]");
  TORCH_CHECK(x.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0, "x alignment");
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kInt && w.is_contiguous(), "w: int32 [K/16, 4*PN]");
  const int M = (int)x.size(0), K = (int)x.size(1), KT = (int)w.size(0), PN = (int)(w.size(1) / 4);
  TORCH_CHECK(KT * 16 == K && w.size(1) % 4 == 0 && PN % 64 == 0, "w shape vs x");
  TORCH_CHECK(M >= 1 && M <= 32, "M must be 1..32");
  TORCH_CHECK(N >= 1 && N <= PN, "N <= padded N");
  TORCH_CHECK(s.is_contiguous() && s.element_size() == (mx ? 1 : 2), "scales dtype");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0,
              "weight / scale base pointers must be 16-byte aligned");
  if (mx) {
    TORCH_CHECK(K % 32 == 0 && s.numel() == (int64_t)(K / 32) * PN, "MX scales [K/32, PN]");
  } else {
    TORCH_CHECK(K % 128 == 0 && s.numel() == (int64_t)(K / 128) * PN, "block scales [K/128, PN]");
  }
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == M && out.size(1) == N &&
                  out.stride(1) == 1,
              "out: bf16 [M, N]");
  TORCH_CHECK(ks >= 1 && ks <= 8 && gs >= 1 && gs <= KT, "ks 1..8, gs 1..K/16");
  const int ntiles = PN / 64;
  const int nblk = (ntiles + (int)tpw - 1) / (int)tpw;
  if (gs > 1) {
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= gs * 8 * mg_for(M) * (int64_t)PN, "workspace size");
    TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= nblk, "counter size");
  }
  KernelFn fn = dense_kernel_for(mx, M, (int)tpw, (int)d, (int)xp, (int)ldg);
  TORCH_CHECK(fn != nullptr, "no kernel instance for (tpw, d, xp, ldg)");
  const c10::cuda::CUDAGuard guard(x.device());
  DenseArgs a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.lda = x.stride(0);
  a.w = reinterpret_cast<const uint32_t*>(w.data_ptr());
  a.s = reinterpret_cast<const uint8_t*>(s.data_ptr());
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.ldc = out.stride(0);
  a.ws = gs > 1 ? ws.data_ptr<float>() : nullptr;
  a.cnt = gs > 1 ? cnt.data_ptr<int>() : nullptr;
  a.M = M;
  a.N = (int)N;
  a.PN = PN;
  a.KT = KT;
  a.ntiles = ntiles;
  a.vec_out = (out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) ? 1 : 0;
  const size_t smem = dense_smem(M, (int)tpw, (int)ks, (int)xp);
  dim3 grid(nblk, (unsigned)gs), block((unsigned)(32 * ks));
  fn<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [max resident CTAs per SM, registers per thread, local (spill) bytes per thread, static smem]
std::vector<int64_t> dense_kernel_info(bool mx, int64_t M, int64_t tpw, int64_t ks, int64_t d, int64_t xp,
                                       int64_t ldg) {
  KernelFn fn = dense_kernel_for(mx, (int)M, (int)tpw, (int)d, (int)xp, (int)ldg);
  TORCH_CHECK(fn != nullptr, "no kernel instance");
  int blocks = 0;
  const size_t smem = dense_smem((int)M, (int)tpw, (int)ks, (int)xp);
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, (int)(32 * ks), smem));
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
  return {blocks, attr.numRegs, (int64_t)attr.localSizeBytes, (int64_t)attr.sharedSizeBytes};
}

}  // namespace glmk

// Shared device helpers for the GLM-5.3-Flash decode kernels (GB10 / SM121).
//
// The weight dequantisation routines below are line-for-line ports of vLLM's Marlin `dequant.h`
// (csrc/libtorch_stable/quantization/marlin/dequant.h: FE4M3fn -> bf16, FE2M1f -> bf16, FE4M3fn / FE8M0fnu scale
// -> bf16) and `marlin_mma.h` (`mma_trans`, the M <= 8 transposed m16n8k16 trick). Using the identical bit
// constructions and the identical __hmul2 scale application gives bit-identical MMA operands to the served Marlin
// kernels, so the only numerical difference left is the fp32 summation order.
//
// Marlin: Elias Frantar and Dan Alistarh (IST-DASLab), "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference
// on Large Language Models"; extended by Neural Magic and the vLLM contributors (FP8/FP4/MX support, MoE, stream-K,
// vllm#24722). Apache-2.0.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace glmk {

// ------------------------------------------------------------------------------------------------------------------
// Global loads. Everything is inline asm marked volatile so that the software pipeline order written in the kernels
// (issue loads for step i + D - 1, then consume step i) is the order ptxas sees; the MMAs are volatile as well.
// ------------------------------------------------------------------------------------------------------------------

// Streamed weights: read exactly once per call. MODE 0: no L1 allocation; MODE 1: + 256 B L2 prefetch hint;
// MODE 2: plain non-coherent load (L1 allocating).
template <int MODE>
__device__ __forceinline__ uint4 ld_stream_v4(const void* p) {
  uint4 r;
  if constexpr (MODE == 0) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
  } else if constexpr (MODE == 1) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
  } else {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
  }
  return r;
}

// Scales and activations: small, reused across lanes of a quad (scales) and across tiles / CTAs (activations),
// so they go through L1.
__device__ __forceinline__ uint4 ld_nc_v4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc_v2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_nc_b32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.b32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

// Workspace written earlier in the same kernel by other CTAs: bypass L1 (it may hold a line from a previous call).
__device__ __forceinline__ float4 ld_cg_f4(const float4* p) { return __ldcg(p); }
__device__ __forceinline__ float ld_cg_f(const float* p) { return __ldcg(p); }

// ------------------------------------------------------------------------------------------------------------------
// bf16 helpers
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t u) {
  return *reinterpret_cast<const __nv_bfloat162*>(&u);
}
__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) { return *reinterpret_cast<const uint32_t*>(&v); }

__device__ __forceinline__ __nv_bfloat162 bcast_lo(__nv_bfloat162 v) { return __bfloat162bfloat162(v.x); }
__device__ __forceinline__ __nv_bfloat162 bcast_hi(__nv_bfloat162 v) { return __bfloat162bfloat162(v.y); }

// ------------------------------------------------------------------------------------------------------------------
// Marlin dequantisation (ports of dequant.h; "reverse indexing is intentional because weights are permuted").
// All shifts are done on uint32_t, which reproduces Marlin's bit patterns (its masks >= 0x80000000 are unsigned
// literals, so its shifts are logical too).
// ------------------------------------------------------------------------------------------------------------------

// dequant<nv_bfloat162, kFE4M3fn, skip_flop=true>: e4m3 bits placed into bf16 (value = e4m3 * 2^-120).
__device__ __forceinline__ void dq_fp8_bits(uint32_t q, __nv_bfloat162* fb) {
  constexpr uint32_t MASK = 0x7F007F00u;
  uint32_t out1 = (q & 0x80008000u) | ((q & MASK) >> 4);
  q <<= 8;
  uint32_t out2 = (q & 0x80008000u) | ((q & MASK) >> 4);
  fb[1] = as_bf162(out1);
  fb[0] = as_bf162(out2);
}

// dequant<nv_bfloat162, kFE4M3fn, skip_flop=false>: the bits above times 2^120 (exponent bias), in bf16.
__device__ __forceinline__ void dq_fp8_biased(uint32_t q, __nv_bfloat162* fb) {
  dq_fp8_bits(q, fb);
  constexpr uint32_t BIAS = (120u + 127u) << 23;  // 2^120 as fp32 bits
  const __nv_bfloat162 bias_reg = __float2bfloat162_rn(__uint_as_float(BIAS));
  fb[1] = __hmul2(fb[1], bias_reg);
  fb[0] = __hmul2(fb[0], bias_reg);
}

// dequant<nv_bfloat162, kFE2M1f, skip_flop=true>: e2m1 bits placed into bf16 (value = fp4 * 2^-126).
__device__ __forceinline__ void dq_fp4_bits(uint32_t q, __nv_bfloat162* fb) {
  constexpr uint32_t MASK = 0x70007000u;
  uint32_t out1 = (q & 0x80008000u) | ((q & MASK) >> 6);
  q <<= 4;
  uint32_t out2 = (q & 0x80008000u) | ((q & MASK) >> 6);
  fb[1] = as_bf162(out1);
  fb[0] = as_bf162(out2);
}

// dequant_fp8_scales<nv_bfloat162, kFE4M3fn> (NVFP4 group scales in Marlin's S0E5M3 form).
__device__ __forceinline__ void dq_scale_e4m3(uint32_t q, __nv_bfloat162* fs) {
  constexpr uint32_t MASK = 0x7F007F00u;
  uint32_t out1 = ((q & 0x80008000u) >> 1) | ((q & MASK) >> 4);
  q <<= 8;
  uint32_t out2 = ((q & 0x80008000u) >> 1) | ((q & MASK) >> 4);
  fs[1] = as_bf162(out1);
  fs[0] = as_bf162(out2);
}

// dequant_fp8_scales<nv_bfloat162, kFE8M0fnu> (MX e8m0 scales).
__device__ __forceinline__ void dq_scale_e8m0(uint32_t q, __nv_bfloat162* fs) {
  uint32_t out1 = (q & 0xFF00FF00u) >> 1;
  q <<= 7;
  uint32_t out2 = q & 0x7F807F80u;
  fs[1] = as_bf162(out1);
  fs[0] = as_bf162(out2);
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor core MMA, Marlin's mma_trans (m_block_size_8 path): the 16 x 16 weight fragment is the A operand
// {b0[0], b1[0], b0[1], b1[1]} (rows = 16 output columns n, cols = 16 k), the activations are the B operand
// (16 k x 8 rows of M). Accumulator element r of lane l is C[n = l/4 + 8*(r>>1)][m = 2*(l%4) + (r&1)].
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_trans(float* c, const __nv_bfloat162* fb0, const __nv_bfloat162* fb1,
                                          uint32_t xb0, uint32_t xb1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(as_u32(fb0[0])), "r"(as_u32(fb1[0])), "r"(as_u32(fb0[1])), "r"(as_u32(fb1[1])), "r"(xb0), "r"(xb1),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ------------------------------------------------------------------------------------------------------------------
// One 16 x 64 weight tile slice (one k16 row of a 64-column Marlin tile) as held by one lane, dequantised and
// multiplied into MG accumulator groups. `acc` layout: acc[j][g][r], j = n16 group, g = 8-row group of M.
// ------------------------------------------------------------------------------------------------------------------

// 8-bit weights (32 bytes per lane = words j*2 + side), MX (e8m0 per 32 k, 8 scale bytes per lane) or block
// (bf16 per 128 k, 8 bf16 per lane).
template <bool MX, int MG>
__device__ __forceinline__ void tile_fp8(float (&acc)[4][MG][4], const uint4& w0, const uint4& w1,
                                         const uint4& sv, const uint32_t (&xb)[MG][2]) {
  const uint32_t q[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  __nv_bfloat162 fs[4];
  if constexpr (MX) {
    dq_scale_e8m0(sv.x, fs);
    dq_scale_e8m0(sv.y, fs + 2);
  } else {
    fs[0] = as_bf162(sv.x);
    fs[1] = as_bf162(sv.y);
    fs[2] = as_bf162(sv.z);
    fs[3] = as_bf162(sv.w);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __nv_bfloat162 fb0[2], fb1[2];
    if constexpr (MX) {
      dq_fp8_biased(q[2 * j + 0], fb0);
      dq_fp8_biased(q[2 * j + 1], fb1);
    } else {
      dq_fp8_bits(q[2 * j + 0], fb0);
      dq_fp8_bits(q[2 * j + 1], fb1);
    }
    // Marlin scale<>(): frag_b0 by element 0 of frag_s[j], frag_b1 by element 1.
    const __nv_bfloat162 s0 = bcast_lo(fs[j]);
    const __nv_bfloat162 s1 = bcast_hi(fs[j]);
    fb0[0] = __hmul2(fb0[0], s0);
    fb0[1] = __hmul2(fb0[1], s0);
    fb1[0] = __hmul2(fb1[0], s1);
    fb1[1] = __hmul2(fb1[1], s1);
#pragma unroll
    for (int g = 0; g < MG; ++g) mma_trans(acc[j][g], fb0, fb1, xb[g][0], xb[g][1]);
  }
}

// 4-bit NVFP4 weights (16 bytes per lane = word j per n16 group), S0E5M3 group-16 scales (8 bytes per lane).
template <int MG>
__device__ __forceinline__ void tile_fp4(float (&acc)[4][MG][4], const uint4& w, const uint2& sv,
                                         const uint32_t (&xb)[MG][2]) {
  const uint32_t q[4] = {w.x, w.y, w.z, w.w};
  __nv_bfloat162 fs[4];
  dq_scale_e4m3(sv.x, fs);
  dq_scale_e4m3(sv.y, fs + 2);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __nv_bfloat162 fb0[2], fb1[2];
    // Marlin (kFE2M1f): b_quant_1 = word, b_quant_0 = word << 8.
    dq_fp4_bits(q[j] << 8, fb0);
    dq_fp4_bits(q[j], fb1);
    const __nv_bfloat162 s0 = bcast_lo(fs[j]);
    const __nv_bfloat162 s1 = bcast_hi(fs[j]);
    fb0[0] = __hmul2(fb0[0], s0);
    fb0[1] = __hmul2(fb0[1], s0);
    fb1[0] = __hmul2(fb1[0], s1);
    fb1[1] = __hmul2(fb1[1], s1);
#pragma unroll
    for (int g = 0; g < MG; ++g) mma_trans(acc[j][g], fb0, fb1, xb[g][0], xb[g][1]);
  }
}

__host__ __device__ constexpr int ilog2c(int v) { return v <= 1 ? 0 : 1 + ilog2c(v >> 1); }

}  // namespace glmk

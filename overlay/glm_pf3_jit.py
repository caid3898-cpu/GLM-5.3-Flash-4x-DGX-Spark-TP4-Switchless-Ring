# SPDX-License-Identifier: Apache-2.0
"""JIT build of vLLM's MoE Marlin template for PREFILL tiles (glm-prefill3-20260928), with an optional SiLU epilogue.

Item 2 of the prefill3 study. The kernels are vLLM's own MoE Marlin template (vendored copy in ./csrc, vLLM commit in
csrc/VLLM_COMMIT, identical to the image's), explicitly instantiated for:
  * thread_m_blocks 1..4 (moe_block_size 16/32/48/64), any tile that satisfies the template's static constraints,
    stages 2..6, with the launch grid and dynamic shared memory as runtime arguments (co-residency checked);
  * act=True: the same kernel with ONE edit in write_result (anchored patch below): instead of storing the bf16
    gate_up tile, threads of the left half of each tile row read the gate int4 (8 bf16) and the up int4 at the same
    offset in the right half, apply vLLM's act_and_mul_kernel math (silu_and_mul_with_clamp: packed_compute
    <act_first=true, HAS_CLAMP=true>, same operations and the same two bf16 rounding points) and store 8 bf16 into
    an [M*top_k, N/2] output. This needs the served w13 columns permuted so that each thread_n tile holds
    gate[j*tn/2 .. (j+1)*tn/2) followed by up[same] (permute_gate_up below; 64-column Marlin tile granularity).
    The GEMM accumulation and its bf16 rounding are untouched: the value fed to the activation is exactly the value
    the stock kernel stores, so gate_up+act is bit-identical to stock gate_up (same config) + stock act.
    limit / alpha / beta travel in `a_scales_ptr` (read only for 8-bit activations by the stock body), as a
    3-float device tensor, so they are runtime values exactly as in the stock op (no constant folding).

Built and loaded by overlay/glm_pf3_moe.py (GLM_PF3_*) on the first eligible prefill call. Credits: Marlin (Elias Frantar,
IST-DASLab) and its vLLM port / MoE / NVFP4 / stream-K (Neural Magic, vLLM contributors, vllm#24722); the build
machinery is ours (diagnostics/glm-marlin-tune-20260928). The SiLU-in-epilogue idea is
credited to Matt Mastracci (moe_prefill fc1 epilogue fusion); nothing of his code is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "glm_pf3_csrc")
MOE_TEMPLATE = "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
VENDORED = ("core/scalar_type.hpp", "libtorch_stable/quantization/marlin/kernel.h",
            "libtorch_stable/quantization/marlin/marlin.cuh", "libtorch_stable/quantization/marlin/marlin_dtypes.cuh",
            "libtorch_stable/quantization/marlin/marlin_mma.h", "libtorch_stable/quantization/marlin/dequant.h",
            "libtorch_stable/moe/marlin_moe_wna16/kernel.h", MOE_TEMPLATE)
NVCC_FLAGS = ["-O3", "-std=c++17", "-DENABLE_FP8", "--expt-relaxed-constexpr",
              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
              "-static-global-template-stub=false"]
CHUNK = 1

# The act edit: first statements inside the store branch of write_result (anchor = the verbatim stock lines).
ACT_ANCHOR = ("      int row = c_gl_wr / c_gl_stride;\n"
              "      if (row < block_num_valid_tokens) {\n"
              "        int64_t sorted_row = sh_block_sorted_ids[row];\n")
ACT_PATCH = ("      int row = c_gl_wr / c_gl_stride;\n"
             "      if (row < block_num_valid_tokens) {\n"
             "#if GLM_PF3_ACT\n"
             "        {  // glm-pf3: fused silu_and_mul_with_clamp on the finished bf16 tile (see pf3_moe_jit.py)\n"
             "          const int glm_j = threadIdx.x % (2 * thread_n_blocks);\n"
             "          if (glm_j < thread_n_blocks) {\n"
             "            const int64_t glm_row = sh_block_sorted_ids[row];\n"
             "            const int64_t glm_idx =\n"
             "                glm_row * (prob_n / 16) + (int64_t)slice_col * thread_n_blocks + glm_j;\n"
             "            C[glm_idx] = glm_pf3_act_int4(sh_red[c_sh_rd], sh_red[c_sh_rd + thread_n_blocks],\n"
             "                                          a_scales_ptr);\n"
             "          }\n"
             "          c_gl_wr += c_gl_wr_delta;\n"
             "          c_sh_rd += c_sh_rd_delta;\n"
             "          continue;\n"
             "        }\n"
             "#endif\n"
             "        int64_t sorted_row = sh_block_sorted_ids[row];\n")

_HELPERS = r"""
#include <cuda_bf16.h>
// glm-pf3: vLLM act_and_mul_kernel<..., packed_silu_kernel, act_first=true, use_vec, HAS_CLAMP=true> per pair,
// operation for operation (activation_kernels.cu packed_compute + packed_silu_kernel + cast_to_packed).
static __device__ __forceinline__ __nv_bfloat162 glm_pf3_act2(__nv_bfloat162 gate, __nv_bfloat162 up, float limit,
                                                              float alpha, float beta) {
  float2 u = __bfloat1622float2(up);
  float2 g = __bfloat1622float2(gate);
  g.x = fminf(g.x, limit);
  g.y = fminf(g.y, limit);
  u.x = fmaxf(fminf(u.x, limit), -limit);
  u.y = fmaxf(fminf(u.y, limit), -limit);
  gate = __float22bfloat162_rn(g);
  float2 f = __bfloat1622float2(gate);
  f.x = f.x / (1.0f + expf(-f.x * alpha));
  f.y = f.y / (1.0f + expf(-f.y * alpha));
  float2 a = __bfloat1622float2(__float22bfloat162_rn(f));
  a.x *= u.x + beta;
  a.y *= u.y + beta;
  return __float22bfloat162_rn(a);
}
static __device__ __forceinline__ int4 glm_pf3_act_int4(int4 g, int4 u, const float* __restrict__ p) {
  const float limit = p[0], alpha = p[1], beta = p[2];
  int4 o;
  const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g);
  const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&u);
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; i++) o2[i] = glm_pf3_act2(g2[i], u2[i], limit, alpha, beta);
  return o;
}
"""

_KERNELS_HEAD = """// generated by pf3_moe_jit.py: explicit instantiations of vLLM's MoE Marlin template
// clang-format off
#define GLM_PF3_ACT {act}
@HELPERS@
#define MARLIN_NAMESPACE_NAME {ns}
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "marlin_moe_template_pf3.h"

namespace {ns} {{
"""

_LAUNCH_COMMON = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>
#include <algorithm>

namespace {
struct DevInfo { int sms = 0, optin = 0; };
DevInfo dev_info(int dev) {
  static DevInfo cache[64];
  DevInfo& d = cache[dev];
  if (d.sms == 0) {
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&d.optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return d;
}
// Raise the dynamic smem cap once; every CTA must be co-resident (stream-K CTAs spin on each other's locks).
template <typename F>
void prepare_launch(F fn, int kid, int threads, int smem, int grid, int dev, int* smem_set, const char* what) {
  DevInfo d = dev_info(dev);
  TORCH_CHECK(smem <= d.optin, what, ": smem ", smem, " > opt-in ", d.optin);
  if (smem_set[kid] < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set[kid] = smem;
  }
  int per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (const void*)fn, threads, smem));
  TORCH_CHECK(per_sm > 0 && grid <= per_sm * d.sms, what, ": grid ", grid, " is not co-resident (", per_sm,
              " CTAs/SM)");
}
int cache_size(int tk, int tn, int tmb, int stages) {   // get_kernel_cache_size, NVFP4 (4 bit, group 16), MoE
  const int pack = 8, tb_m = tmb * 16;
  int sh_a = stages * tb_m * tk * 2, sh_b = stages * (tk * tn / pack) * 4, sh_red = tb_m * (tn + 8) * 2;
  int sh_bias = tn * 2;
  int tmp = (sh_b > sh_red ? sh_red : sh_b) + sh_bias;
  tmp = std::max(std::max(sh_b, sh_red), tmp);
  int sh_s = ((tk + 15) / 16) * tn * 2 * stages;
  return tmp + sh_a + sh_s + tb_m * 16;
}
}  // namespace
"""

_LAUNCH = r"""// generated by pf3_moe_jit.py: MoE Marlin launcher (NVFP4, bf16, moe_block_size = 16 * thread_m_blocks)
// clang-format off
#define MARLIN_NAMESPACE_NAME @NS@
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
@COMMON@
namespace @NS@ {
using KernelFn = void (*)(MARLIN_KERNEL_PARAMS);
struct Entry { int tmb, thread_k, thread_n, threads, stages; KernelFn fn; };
static const Entry kTable[] = {
@TABLE@
};
static constexpr int kN = sizeof(kTable) / sizeof(Entry);
static int kSmemSet[kN > 0 ? kN : 1] = {0};
}  // namespace @NS@

// Contract of ops.moe_wna16_marlin_gemm for bf16 activations, NVFP4 weights (fp8 e4m3 group-16 scales, fp32 per-expert
// global scale), no bias / zero points / act_order, use_atomic_add=False, use_fp32_reduce=True.
// ACT=1: output [size_m * top_k, size_n / 2] = silu_and_mul_with_clamp(gate_up) (act_params = [limit, alpha, beta]),
// weights column-permuted per thread_n tile, mul_topk_weights must be false.
at::Tensor @OP@(const at::Tensor& a, const c10::optional<at::Tensor>& c_or_none, const at::Tensor& b_q_weight,
                const at::Tensor& b_scales, const at::Tensor& global_scale, at::Tensor& workspace,
                const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                const at::Tensor& num_tokens_past_padded, const at::Tensor& topk_weights, int64_t moe_block_size,
                int64_t top_k, bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
                int64_t kernel_id, int64_t grid, int64_t smem, const c10::optional<at::Tensor>& act_params) {
  using namespace @NS@;
  constexpr bool kAct = @ACT@;
  TORCH_CHECK(kernel_id >= 0 && kernel_id < kN, "@OP@: kernel_id out of range");
  const Entry& e = kTable[kernel_id];
  TORCH_CHECK(moe_block_size == 16 * e.tmb, "@OP@: moe_block_size ", moe_block_size, " != 16 * tmb ", e.tmb);
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous(), "@OP@: a bf16 contiguous");
  TORCH_CHECK(a.size(0) == size_m && a.size(1) == size_k, "@OP@: a shape");
  TORCH_CHECK(b_q_weight.is_contiguous() && b_q_weight.size(1) * 16 == size_k && b_q_weight.size(2) / 16 * 8 == size_n,
              "@OP@: b_q_weight shape (nvfp4, E x K/16 x 2N)");
  TORCH_CHECK(b_scales.scalar_type() == c10::ScalarType::Float8_e4m3fn && b_scales.dim() == 3 &&
              b_scales.size(2) == size_n && b_scales.size(1) * 16 == size_k, "@OP@: scales fp8 e4m3 group 16");
  TORCH_CHECK(global_scale.scalar_type() == at::kFloat, "@OP@: global_scale fp32");
  TORCH_CHECK(size_n % e.thread_n == 0 && size_k % e.thread_k == 0, "@OP@: tile divisibility");
  const int need = cache_size(e.thread_k, e.thread_n, e.tmb, e.stages);
  TORCH_CHECK(smem >= need, "@OP@: smem ", smem, " < required ", need);
  TORCH_CHECK(workspace.scalar_type() == at::kInt && workspace.numel() >= grid + 2, "@OP@: workspace too small");
  const float* ap = nullptr;
  if (kAct) {
    TORCH_CHECK(!mul_topk_weights, "@OP@: the act epilogue needs mul_topk_weights = false");
    TORCH_CHECK(e.thread_n % 128 == 0, "@OP@: act needs thread_n % 128 == 0 (64-column halves)");
    TORCH_CHECK(act_params.has_value() && act_params->is_cuda() && act_params->scalar_type() == at::kFloat &&
                act_params->numel() == 3, "@OP@: act_params = float32[3] (limit, alpha, beta) on the GPU");
    ap = act_params->data_ptr<float>();
  }
  const int dev = a.get_device();
  c10::cuda::CUDAGuard guard(a.device());
  prepare_launch(e.fn, (int)kernel_id, e.threads, (int)smem, (int)grid, dev, kSmemSet, "@OP@");
  auto opts = a.options();
  const int64_t n_out = kAct ? size_n / 2 : size_n;
  at::Tensor c;
  if (c_or_none.has_value()) {
    c = *c_or_none;
    TORCH_CHECK(c.is_contiguous() && c.size(0) == size_m * top_k && c.size(1) == n_out, "@OP@: c shape");
  } else {
    c = at::empty({size_m * top_k, n_out}, opts);
  }
  at::Tensor c_tmp = at::empty({(grid + 2) * 16 * e.tmb * e.thread_n}, opts.dtype(at::kFloat));
  at::Tensor none_f = at::empty({0}, opts.dtype(at::kFloat));
  at::Tensor none_h = at::empty({0}, opts);
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  e.fn<<<(int)grid, e.threads, (int)smem, stream>>>(
      (const int4*)a.data_ptr(), (const int4*)b_q_weight.data_ptr(), (int4*)c.data_ptr(), (int4*)c_tmp.data_ptr(),
      (const int4*)none_h.data_ptr(), kAct ? ap : (const float*)none_f.data_ptr(), (const int4*)b_scales.data_ptr(),
      (const float*)global_scale.data_ptr(), (const int4*)none_h.data_ptr(), (const int*)nullptr,
      (const int32_t*)sorted_token_ids.data_ptr(), (const int32_t*)expert_ids.data_ptr(),
      (const int32_t*)num_tokens_past_padded.data_ptr(), (const float*)topk_weights.data_ptr(), (int)top_k,
      mul_topk_weights, (int)b_scales.size(1), (int)size_m, (int)size_n, (int)size_k, (int*)workspace.data_ptr(),
      /*has_bias=*/false, /*use_atomic_add=*/false, /*use_fp32_reduce=*/true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

std::vector<int64_t> @OP@_table() {
  using namespace @NS@;
  std::vector<int64_t> out;
  for (int i = 0; i < kN; i++) {
    const Entry& e = kTable[i];
    cudaFuncAttributes attr;
    int regs = -1, local = -1;
    if (e.fn != nullptr && cudaFuncGetAttributes(&attr, (const void*)e.fn) == cudaSuccess) {
      regs = attr.numRegs;
      local = (int)attr.localSizeBytes;
    }
    out.insert(out.end(), {e.tmb, e.thread_k, e.thread_n, e.threads, e.stages, regs, local});
  }
  return out;
}

TORCH_LIBRARY_FRAGMENT(glm_pf3, m) {
  m.def("@OP@(Tensor a, Tensor? c, Tensor b_q_weight, Tensor b_scales, Tensor global_scale, Tensor(a!) workspace, "
        "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_past_padded, Tensor topk_weights, "
        "int moe_block_size, int top_k, bool mul_topk_weights, int size_m, int size_n, int size_k, int kernel_id, "
        "int grid, int smem, Tensor? act_params) -> Tensor");
  m.def("@OP@_table() -> int[]", &@OP@_table);
}
TORCH_LIBRARY_IMPL(glm_pf3, CUDA, m) { m.impl("@OP@", &@OP@); }
"""

# Stand-alone elementwise twin of the act epilogue, for the exhaustive bitwise test against the stock op.
_ACT_TEST = r"""// generated by pf3_moe_jit.py: act epilogue math as an elementwise kernel (test only) + exact sum8_add
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>
@HELPERS@
__global__ void glm_pf3_act_rows(const int4* __restrict__ in, int4* __restrict__ out, int d8, const float* p) {
  const int64_t r = blockIdx.x;
  for (int i = threadIdx.x; i < d8; i += blockDim.x)
    out[r * d8 + i] = glm_pf3_act_int4(in[r * 2 * d8 + i], in[r * 2 * d8 + d8 + i], p);
}
void act_rows(const at::Tensor& x, at::Tensor& y, const at::Tensor& p) {
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(1) % 16 == 0 && y.size(0) == x.size(0) && y.size(1) * 2 == x.size(1));
  TORCH_CHECK(p.scalar_type() == at::kFloat && p.numel() == 3 && p.is_cuda());
  c10::cuda::CUDAGuard g(x.device());
  glm_pf3_act_rows<<<(int)x.size(0), 128, 0, at::cuda::getCurrentCUDAStream(x.get_device())>>>(
      (const int4*)x.data_ptr(), (int4*)y.data_ptr(), (int)(x.size(1) / 16), p.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Exact fused MoE finalize: y = shared + moe_sum(x3) with the stock rounding tree
//   r = bf16(((0.f + x[t,0]) + x[t,1]) + ... + x[t,7])   (moe_sum_vec_kernel: fp32 in slot order)
//   y = bf16(float(shared) + float(r))                    (aten::add, bf16 tensors, alpha 1: opmath fp32)
// c10::BFloat16(float) in device code on sm_80+ is __float2bfloat16 (round to nearest even, NaN -> 0x7FFF): the same
// intrinsic here, so NaN / inf / -0 map bit for bit too.
__device__ __forceinline__ unsigned short glm_pf3_c10_bf16(float f) { return __bfloat16_as_ushort(__float2bfloat16(f)); }
__device__ __forceinline__ float glm_pf3_bf2f(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }

template <int TOPK>
__global__ void glm_pf3_sum_add(const int4* __restrict__ x, const int4* __restrict__ sh, int4* __restrict__ y,
                                int64_t T, int d8) {
  const int64_t total = T * d8;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = i / d8, v = i % d8;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int4 p = x[(t * TOPK + k) * d8 + v];
      const unsigned short* ps = reinterpret_cast<const unsigned short*>(&p);
#pragma unroll
      for (int j = 0; j < 8; j++) acc[j] += glm_pf3_bf2f(ps[j]);
    }
    int4 s = sh[t * d8 + v], o;
    const unsigned short* ss = reinterpret_cast<const unsigned short*>(&s);
    unsigned short* os = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float r = glm_pf3_bf2f(glm_pf3_c10_bf16(acc[j]));
      os[j] = glm_pf3_c10_bf16(glm_pf3_bf2f(ss[j]) + r);
    }
    y[t * d8 + v] = o;
  }
}
void sum8_add(const at::Tensor& x3, const at::Tensor& shared, at::Tensor& y) {
  TORCH_CHECK(x3.is_contiguous() && shared.is_contiguous() && y.is_contiguous() && x3.dim() == 3 && x3.size(1) == 8 &&
              x3.scalar_type() == at::kBFloat16 && shared.scalar_type() == at::kBFloat16 &&
              y.scalar_type() == at::kBFloat16 && x3.size(2) % 8 == 0 && shared.size(0) == x3.size(0) &&
              shared.size(1) == x3.size(2) && y.sizes() == shared.sizes());
  c10::cuda::CUDAGuard g(x3.device());
  const int64_t T = x3.size(0);
  const int d8 = (int)(x3.size(2) / 8);
  const int block = 256;
  const int64_t total = T * d8;
  const int grid = (int)std::min<int64_t>((total + block - 1) / block, 65535);
  glm_pf3_sum_add<8><<<grid, block, 0, at::cuda::getCurrentCUDAStream(x3.get_device())>>>(
      (const int4*)x3.data_ptr(), (const int4*)shared.data_ptr(), (int4*)y.data_ptr(), T, d8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(glm_pf3, m) {
  m.def("act_rows(Tensor x, Tensor(a!) y, Tensor p) -> ()");
  m.def("sum8_add(Tensor x3, Tensor shared, Tensor(a!) y) -> ()");
}
TORCH_LIBRARY_IMPL(glm_pf3, CUDA, m) {
  m.impl("act_rows", &act_rows);
  m.impl("sum8_add", &sum8_add);
}
"""


class PatchError(RuntimeError):
    pass


# Column remap (act kernels only): tile `slice_col` of width tn reads gate columns [slice_col*tn/2, +tn/2) and the up
# columns at the same offset in the second half of w13, straight from the SERVED (unpermuted) Marlin buffers. Marlin
# lays a k16 row out as N/64 contiguous 64-column tiles (32 int4 of B, 4 int4 of fp8 scales), so moving whole
# 64-column tiles is exactly what the offline permutation (permute_gate_up) did: same operands in the same tile
# positions, hence the same MMA sequence. Valid when every thread owns one B column (threads > b_sh_stride), checked
# statically. Non-act kernels expand the macros to the stock expressions.
REMAP_DEFS = (
    "#if GLM_PF3_ACT\n"
    "  #define GLM_PF3_BCOL(sc) ((b_sh_stride / 2) * (sc) + (((int)(threadIdx.x % b_sh_stride) >= b_sh_stride / 2) "
    "? (b_gl_stride / 2 - b_sh_stride / 2) : 0))\n"
    "  #define GLM_PF3_SCOL(sc) ((s_sh_stride / 2) * (sc) + (((int)(threadIdx.x % s_sh_stride) >= s_sh_stride / 2) "
    "? (s_gl_stride / 2 - s_sh_stride / 2) : 0))\n"
    "#else\n"
    "  #define GLM_PF3_BCOL(sc) (b_sh_stride * (sc))\n"
    "  #define GLM_PF3_SCOL(sc) (s_sh_stride * (sc))\n"
    "#endif\n")
REMAP_EDITS = (
    ("  b_gl_rd += B_expert_off + b_sh_stride * slice_col;\n",
     "  b_gl_rd += B_expert_off + GLM_PF3_BCOL(slice_col);\n"
     "#if GLM_PF3_ACT\n  static_assert(threads > b_sh_stride, \"glm-pf3 remap needs one B column per thread\");\n#endif\n", 1),
    ("        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;\n",
     "        b_gl_rd += GLM_PF3_BCOL(slice_col) + b_gl_rd_delta_o * slice_row;\n", 1),
    ("s_sh_stride * slice_col + threadIdx.x % s_sh_stride;", "GLM_PF3_SCOL(slice_col) + threadIdx.x % s_sh_stride;", 2),
)


def patch_act(text: str) -> str:
    n = text.count(ACT_ANCHOR)
    if n != 1:
        raise PatchError(f"act anchor found {n}x, expected 1x (vendored MoE template changed)")
    if "a_scales_ptr" not in text:
        raise PatchError("a_scales_ptr missing from the template")
    text = text.replace(ACT_ANCHOR, ACT_PATCH)
    for old, new, cnt in REMAP_EDITS:
        if text.count(old) != cnt:
            raise PatchError(f"remap anchor found {text.count(old)}x, expected {cnt}x: {old.strip()[:70]!r}")
        text = text.replace(old, new)
    return REMAP_DEFS + text


def kernel_expr(inst) -> str:
    tmb, tk, tn, th, st = inst["tmb"], inst["tk"], inst["tn"], inst["th"], inst["stages"]
    return (f"Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), {th}, "
            f"{tmb}, {tn // 16}, {tk // 16}, false, {st}, 1, false>")


def ns_of(act: bool) -> str:
    return "glm_pf3_moe_act" if act else "glm_pf3_moe"


def op_of(act: bool) -> str:
    return "moe_gemm_act" if act else "moe_gemm"


def read_vendored():
    out = {}
    for rel in VENDORED:
        with open(os.path.join(CSRC, rel)) as f:
            out[rel] = f.read()
    return out


def render_texts(instances):
    """{file name: text}. instances: list of dicts tmb, tk, tn, th, stages, act (bool)."""
    src = read_vendored()
    out = {"marlin_moe_template_pf3.h": patch_act(src[MOE_TEMPLATE])}
    for act in (False, True):
        lst = [i for i in instances if bool(i["act"]) == act]
        for part in range(0, len(lst), CHUNK):
            body = _KERNELS_HEAD.format(act=int(act), ns=ns_of(act)).replace("@HELPERS@", _HELPERS)
            for inst in lst[part:part + CHUNK]:
                body += f"template __global__ void {kernel_expr(inst)}( MARLIN_KERNEL_PARAMS );\n"
            out[f"moe_{'act' if act else 'plain'}_{part // CHUNK}.cu"] = body + f"\n}}  // namespace {ns_of(act)}\n"
        rows = [f"    {{{i['tmb']}, {i['tk']}, {i['tn']}, {i['th']}, {i['stages']}, &{ns_of(act)}::{kernel_expr(i)}}},"
                for i in lst] or ["    {0, 0, 0, 0, 0, nullptr}"]
        out[f"launch_{'act' if act else 'plain'}.cu"] = (
            _LAUNCH.replace("@COMMON@", _LAUNCH_COMMON).replace("@TABLE@", "\n".join(rows))
            .replace("@NS@", ns_of(act)).replace("@OP@", op_of(act)).replace("@ACT@", "true" if act else "false"))
    out["act_test.cu"] = _ACT_TEST.replace("@HELPERS@", _HELPERS)
    return out


def _after_dirs():
    out = []
    try:
        import nvidia
        for base in getattr(nvidia, "__path__", []):
            for sub in ("cu13/include", "cu12/include"):
                d = os.path.join(base, sub)
                if os.path.isdir(d):
                    out.append(d)
    except ImportError:
        pass
    return out


class Ext:
    def __init__(self, instances):
        import torch
        self.ops = torch.ops.glm_pf3
        self.ids = {}
        for act in (False, True):
            flat = list(getattr(self.ops, f"{op_of(act)}_table")())
            for i in range(len(flat) // 7):
                tmb, tk, tn, th, st, regs, local = flat[i * 7:(i + 1) * 7]
                if tmb > 0:
                    self.ids[(act, tmb, tk, tn, th, st)] = (i, regs, local)

    def kid(self, act, tmb, tk, tn, th, st):
        return self.ids[(bool(act), tmb, tk, tn, th, st)]


def build(instances, build_root, gencode="arch=compute_121a,code=sm_121a", jobs=16, verbose=False):
    import torch
    texts = render_texts(instances)
    h = hashlib.sha256()
    for name in sorted(texts):
        h.update(name.encode() + b"\0" + texts[name].encode())
    for rel in VENDORED:
        with open(os.path.join(CSRC, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + f.read())
    h.update(json.dumps([gencode, NVCC_FLAGS, torch.__version__, torch.version.cuda]).encode())
    tag = h.hexdigest()[:16]
    root = os.path.join(build_root, tag)
    src_dir, bdir = os.path.join(root, "src"), os.path.join(root, "build")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)
    files = []
    for name, text in texts.items():
        p = os.path.join(src_dir, name)
        with open(p, "w") as f:
            f.write(text)
        if name.endswith(".cu"):
            files.append(p)
    lock = os.path.join(bdir, "lock")
    if os.path.exists(lock) and not os.path.exists(os.path.join(bdir, f"glm_pf3_{tag}.so")):
        # a killed earlier build leaves torch's FileBaton lock behind (a new build would wait on it forever): set it
        # aside by renaming (nothing is deleted)
        os.rename(lock, os.path.join(bdir, f"stale-lock-{int(time.time())}"))
    os.environ["MAX_JOBS"] = str(jobs)
    from torch.utils.cpp_extension import load
    after, nv_after = [], []
    for d in _after_dirs():
        after += ["-idirafter", d]
        nv_after += ["-Xcompiler", "-idirafter," + d]
    incs = [CSRC, os.path.join(CSRC, "libtorch_stable", "quantization", "marlin"), src_dir]
    t0 = time.time()
    load(name=f"glm_pf3_{tag}", sources=files, extra_include_paths=incs, extra_cflags=["-O3", "-std=c++17"] + after,
         extra_cuda_cflags=NVCC_FLAGS + ["-gencode=" + gencode] + nv_after, build_directory=bdir,
         is_python_module=False, verbose=verbose)
    return Ext(instances), tag, time.time() - t0


# ------------------------------------------------------------------------------------------------------------------
# instance plan
# ------------------------------------------------------------------------------------------------------------------
PLAIN_TILES = ((64, 256, 256), (128, 128, 256), (64, 128, 128), (128, 64, 128), (256, 64, 256), (128, 256, 512))
ACT_TILES = ((64, 256, 256), (128, 128, 256), (64, 128, 128), (128, 256, 512))
OPTIN = 101376


def default_instances():
    """Plain: block_m 32/48/64 x every PLAIN tile the template accepts x stages 2..6 that fit the 99 KiB opt-in.
    Act (gate_up only, thread_n % 128 == 0): block_m 48/64 x ACT tiles x stages 2..5."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "vendor"))
    import marlin_model as MM

    def fits(tile, tmb, st):
        return MM.kernel_cache_size(MM.Tile(*tile), tmb, 4, 16, st, True) <= OPTIN

    out = []
    for tmb in (2, 3, 4):
        for tile in PLAIN_TILES:
            if not MM.template_ok(MM.Tile(*tile), 4, 1)[0]:
                continue
            for st in (2, 3, 4, 5, 6):
                if fits(tile, tmb, st):
                    out.append(dict(tmb=tmb, tk=tile[0], tn=tile[1], th=tile[2], stages=st, act=False))
    for tmb in (3, 4):
        for tile in ACT_TILES:
            for st in (2, 3, 4, 5):
                if fits(tile, tmb, st):
                    out.append(dict(tmb=tmb, tk=tile[0], tn=tile[1], th=tile[2], stages=st, act=True))
    return out


# ------------------------------------------------------------------------------------------------------------------
# weight permutation for the act epilogue
# ------------------------------------------------------------------------------------------------------------------
def gate_up_perm(N2: int, tn: int):
    """Column order for thread_n tile width tn over a [gate | up] output of N2 columns: tile j holds
    gate[j*tn/2 : (j+1)*tn/2] then up[j*tn/2 : (j+1)*tn/2]. Returns the list of source columns (length N2)."""
    half, h = N2 // 2, tn // 2
    assert N2 % tn == 0 and h % 64 == 0
    cols = []
    for j in range(N2 // tn):
        cols += list(range(j * h, (j + 1) * h)) + list(range(half + j * h, half + (j + 1) * h))
    return cols


def permute_gate_up(w1, s1, tn: int):
    """Served NVFP4 Marlin w13 [E, K/16, 2*N2] int32 and scales [E, K/16, N2] fp8 -> column-permuted copies, moving
    whole 64-column Marlin tiles (a k16 row holds N2/64 contiguous 128-word tiles; scales are permuted within 64)."""
    import torch
    E, kt, words = w1.shape
    N2 = s1.shape[2]
    assert words == 2 * N2 and N2 % 64 == 0
    blocks = [c // 64 for c in gate_up_perm(N2, tn)[::64]]
    idx = torch.tensor(blocks, device=w1.device)
    w = w1.view(E, kt, N2 // 64, 128).index_select(2, idx).reshape(E, kt, words).contiguous()
    s = s1.view(torch.uint8).view(E, s1.shape[1], N2 // 64, 64).index_select(2, idx).reshape(s1.shape) \
        .contiguous().view(s1.dtype)
    return w, s

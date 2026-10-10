// Python bindings for the decode W8A16 kernel only (glmk_dense.cu, unchanged copy from
// diagnostics/glm-cuda-kernels-20260928/csrc). The routed-MoE kernel of that study is not built: it loses to
// Marlin on every measured shape.
#include <torch/extension.h>

#include <vector>

namespace glmk {
void dense_w8a16(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, torch::Tensor& out,
                 torch::Tensor& ws, torch::Tensor& cnt, bool mx, int64_t N, int64_t tpw, int64_t ks, int64_t gs,
                 int64_t d, int64_t xp, int64_t ldg);
std::vector<int64_t> dense_kernel_info(bool mx, int64_t M, int64_t tpw, int64_t ks, int64_t d, int64_t xp,
                                       int64_t ldg);
}  // namespace glmk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_w8a16", &glmk::dense_w8a16, "decode W8A16 GEMM over Marlin FP8 buffers");
  m.def("dense_kernel_info", &glmk::dense_kernel_info, "[ctas/SM, regs, local bytes, static smem]");
}

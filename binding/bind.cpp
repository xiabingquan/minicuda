#include <torch/extension.h>

// 前向声明, 实现在 csrc/vector_add/vector_add.cu
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Element-wise vector addition (CUDA)");
}

#include <torch/extension.h>

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a);
torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("vector_add", &vector_add, "Element-wise vector addition (CUDA)",
            py::arg("a"), py::arg("b"));
      m.def("saxpy", &saxpy, "Compute z = a * x + y (CUDA)",
            py::arg("x"), py::arg("y"), py::arg("a"));
      m.def("matrix_add", &matrix_add, "Element-wise matrix addition (CUDA)",
            py::arg("a"), py::arg("b"));
}

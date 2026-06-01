#include <torch/extension.h>

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor vector_add_raw(torch::Tensor a, torch::Tensor b);
torch::Tensor cpu_large_vector_add_async(torch::Tensor a, torch::Tensor b, int buffer_size);
torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a);
torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b);
torch::Tensor rgb_to_grayscale(torch::Tensor inp);
torch::Tensor transpose_naive(torch::Tensor inp);
torch::Tensor transpose_shared(torch::Tensor inp);
torch::Tensor dot_product(torch::Tensor a, torch::Tensor b);
torch::Tensor gemv(torch::Tensor A, torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("vector_add", &vector_add, "Element-wise vector addition (CUDA)",
            py::arg("a"), py::arg("b"));
      m.def("vector_add_raw", &vector_add_raw, "Element-wise vector addition (raw CUDA, no PyTorch alloc)",
            py::arg("a"), py::arg("b"));
      m.def("cpu_large_vector_add_async", &cpu_large_vector_add_async,
            "CPU-input/output vector addition with chunked GPU pipeline + ping-pong pinned buffers",
            py::arg("a"), py::arg("b"), py::arg("buffer_size") = 1024);
      m.def("saxpy", &saxpy, "Compute z = a * x + y (CUDA)",
            py::arg("x"), py::arg("y"), py::arg("a"));
      m.def("matrix_add", &matrix_add, "Element-wise matrix addition (CUDA)",
            py::arg("a"), py::arg("b"));
      m.def("rgb_to_grayscale", &rgb_to_grayscale, "RGB to grayscale conversion (CUDA)",
            py::arg("inp"));
      m.def("transpose_naive", &transpose_naive, "Naive matrix transpose (CUDA)",
            py::arg("inp"));
      m.def("transpose_shared", &transpose_shared, "Shared memory matrix transpose (CUDA)",
            py::arg("inp"));
      m.def("dot_product", &dot_product, "Vector dot product with shared memory reduction (CUDA)",
            py::arg("a"), py::arg("b"));
      m.def("gemv", &gemv, "Matrix-vector multiply y = A @ x (CUDA)",
            py::arg("A"), py::arg("x"));
}

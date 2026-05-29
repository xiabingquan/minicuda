#include <torch/types.h>

__global__ void vector_add_kernel(float* a, float* b, float* c, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    c[idx] = a[idx] + b[idx];
  }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
  TORCH_CHECK(b.device().is_cuda(), "b must be on CUDA");
  TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");

  auto c = torch::empty_like(a);

  int len = c.numel();
  int block_size = 256;
  int grid_size = (len + block_size - 1) / block_size;

  vector_add_kernel<<<grid_size, block_size>>>(
    a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), len
  );

  return c;
}

#include <torch/types.h>

__global__ void matrix_add_kernel(float *a, float *b, float *c, int numel, int offset)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  int i = row * offset + col;
  if (i < numel)
  {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
  TORCH_CHECK(b.device().is_cuda(), "b must be on CUDA");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

  auto c = a.new_empty(a.sizes());

  dim3 block_size(16, 16);
  dim3 grid_size(
      (a.size(0) + 15) / 16,
      (a.size(1) + 15) / 16);

  matrix_add_kernel<<<grid_size, block_size>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), a.numel(), a.size(1));

  return c;
}

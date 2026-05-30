#include <torch/types.h>
#include <cuda_runtime.h>

__global__ void vector_add_raw_kernel(float *a, float *b, float *c, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor vector_add_raw(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
  TORCH_CHECK(b.device().is_cuda(), "b must be on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch");
  TORCH_CHECK(a.dim() == 1, "must be 1D");

  int n = a.size(0);
  float *c_ptr;
  cudaMalloc(&c_ptr, n * sizeof(float));

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  vector_add_raw_kernel<<<grid_size, block_size>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c_ptr, n);

  return torch::from_blob(
      c_ptr, {n},
      /*deleter=*/[](void *p)
      { cudaFree(p); },
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

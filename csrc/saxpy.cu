#include <torch/types.h>

__global__ void saxpy_kernel(float* x, float* y, float a, float* z, int n) {
  int num_threads = gridDim.x * blockDim.x;
  int st = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i = st; i < n; i+=num_threads) {
    z[i] = a * x[i] + y[i];
  }
}

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a) {
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(y.device().is_cuda(), "y must be on CUDA");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  auto z = x.new_empty(x.sizes());

  int block_size = 256;
  int grid_size = 8;
  int n = x.numel();

  saxpy_kernel<<<grid_size, block_size>>>(
    x.data_ptr<float>(), y.data_ptr<float>(), a, z.data_ptr<float>(), n
  );

  return z;
}

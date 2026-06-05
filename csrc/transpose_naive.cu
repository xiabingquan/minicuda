#include <cuda_runtime.h>
#include <torch/types.h>

__global__ void transpose_naive_kernel(float *inp, float *out, int m, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n) {
    out[j * m + i] = inp[i * n + j];
  }
}

torch::Tensor transpose_naive(torch::Tensor inp) {
  TORCH_CHECK(inp.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(inp.dim() == 2, "input must be 2D");
  TORCH_CHECK(inp.dtype() == torch::kFloat32, "input must be float32");

  int m = inp.size(0);
  int n = inp.size(1);
  torch::Tensor out = inp.new_empty({n, m});

  dim3 block_size(32, 32);
  dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  transpose_naive_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), m,
                                                    n);

  return out;
}

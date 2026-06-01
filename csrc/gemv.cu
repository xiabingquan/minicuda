#include <torch/types.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

__global__ void gemv_kernel(float *A, float *x, float *y, int m, int d)
{
  int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  __shared__ float buf[WARP_SIZE][WARP_SIZE];
  buf[threadIdx.y][threadIdx.x] = 0.0f;

  if (row_idx < m)
  {
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
      buf[threadIdx.y][threadIdx.x] += A[d * row_idx + i] * x[i];
    }
  }
  __syncthreads();

  for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1)
  {
    if (threadIdx.x < stride)
    {
      buf[threadIdx.y][threadIdx.x] += buf[threadIdx.y][threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && row_idx < m)
  {
    y[row_idx] = buf[threadIdx.y][0];
  }
}

torch::Tensor gemv(torch::Tensor A, torch::Tensor x)
{
  TORCH_CHECK(A.is_cuda() && x.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && x.dim() == 1, "A must be 2D, x must be 1D");
  TORCH_CHECK(A.size(1) == x.size(0), "dimension mismatch");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "inputs must be float32");

  int m = A.size(0), d = A.size(1);
  torch::Tensor y = x.new_empty({m});

  dim3 block_size(WARP_SIZE, WARP_SIZE);
  int grid_size = (m + block_size.y - 1) / block_size.y;

  gemv_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), m, d);
  return y;
}

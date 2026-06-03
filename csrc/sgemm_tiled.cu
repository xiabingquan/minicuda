#include <torch/types.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr int WARP_SIZE = 32;

__global__ void sgemm_shared_kernel(float *A, float *B, float *C, int M, int K, int N)
{
  int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int col_idx = blockDim.x * blockIdx.x + threadIdx.x;

  float tmp = 0.0f;
  __shared__ float buf_A[WARP_SIZE][WARP_SIZE + 1];
  __shared__ float buf_B[WARP_SIZE][WARP_SIZE + 1];

  for (int i = 0; i < (K + WARP_SIZE - 1); i += WARP_SIZE)
  {
    int row_dim_idx = i + threadIdx.x;
    int col_dim_idx = i + threadIdx.y;
    if (row_dim_idx < K && row_idx < M)
    {
      buf_A[threadIdx.y][threadIdx.x] = A[K * row_idx + row_dim_idx];
    }
    else
    {
      buf_A[threadIdx.y][threadIdx.x] = 0;
    }
    if (col_dim_idx < K && col_idx < N)
    {
      buf_B[threadIdx.y][threadIdx.x] = B[N * col_dim_idx + col_idx];
    }
    else
    {
      buf_B[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    for (int j = 0; j < WARP_SIZE; j++)
    {
      tmp += buf_A[threadIdx.y][j] * buf_B[j][threadIdx.x];
    }
    __syncthreads();
  }

  if (row_idx < M && col_idx < N)
  {
    C[row_idx * N + col_idx] = tmp;
  }
}

torch::Tensor sgemm_shared(torch::Tensor A, torch::Tensor B)
{
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch: A is MxK, B is KxN");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "inputs must be float32");

  int M = A.size(0), K = A.size(1), N = B.size(1); // （M, K) * (K, N) -> (M, N)
  torch::Tensor C = A.new_empty({M, N});

  dim3 block_size(WARP_SIZE, WARP_SIZE);
  dim3 grid_size(
      (N + block_size.x - 1) / block_size.x,
      (M + block_size.y - 1) / block_size.y);

  sgemm_shared_kernel<<<grid_size, block_size>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  return C;
}

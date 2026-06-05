#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdio>

__global__ void sgemm_naive_kernel(float *A, float *B, float *C, int M, int K, int N) {
  int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int col_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (row_idx < M && col_idx < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; i++) {
      tmp += A[row_idx * K + i] * B[i * N + col_idx];
    }
    C[row_idx * N + col_idx] = tmp;
  }
}

torch::Tensor sgemm_naive(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch: A is MxK, B is KxN");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "inputs must be float32");

  int M = A.size(0), K = A.size(1), N = B.size(1);  // （M, K) * (K, N) -> (M, N)
  torch::Tensor C = A.new_empty({M, N});

  dim3 block_size(16, 16);
  dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

  sgemm_naive_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                C.data_ptr<float>(), M, K, N);
  return C;
}

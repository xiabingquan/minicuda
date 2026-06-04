#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdio>

constexpr int BM = 128;  // block tile M
constexpr int BN = 128;  // block tile N
constexpr int BK = 8;    // block tile K
constexpr int TM = 8;    // thread tile M (每个 thread 算 TM 行)
constexpr int TN = 8;    // thread tile N (每个 thread 算 TN 列)

__global__ void sgemm_vectorized_kernel(float *A, float *B, float *C, int M, int K, int N) {
  __shared__ float buf_A[BM][BK];
  __shared__ float buf_B[BN][BK];

  float tmp[TM][TN];
  float reg_A[TM][TN];
  float reg_B[TM][TN];

  int row_st_idx = blockIdx.y * BM + threadIdx.y * TM;
  int row_ed_idx = row_st_idx + TM;
  int col_st_idx = blockIdx.x * BN + threadIdx.x * TN;
  int col_ed_idx = col_st_idx + TN;

  for (int li = 0; li < TM; li++) {
    int i = blockIdx.y * BM + threadIdx.y * TM + i;  // row_idx
    if (i < M) {
      for (int k = 0; k < (K + BK - 1); k += BK) {
        if (k < K) {
          float4 *d_tmp = reinterpret_cast<float4 *>(&A[i * K + k]);
        } else {
          reg_A[li][k] = 0.0f;
        }
      }
    } else {
      for (int k = 0; k < K; k++) {
        reg_A[li][k] = 0.0f;
      }
    }
  }
}

torch::Tensor sgemm_vectorized(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch: A is MxK, B is KxN");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "inputs must be float32");

  int M = A.size(0), K = A.size(1), N = B.size(1);
  torch::Tensor C = torch::zeros({M, N}, A.options());

  constexpr int BLOCK_X = BN / TN;  // 16
  constexpr int BLOCK_Y = BM / TM;  // 16
  dim3 block_size(BLOCK_X, BLOCK_Y);
  dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_vectorized_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                     C.data_ptr<float>(), M, K, N);
  return C;
}

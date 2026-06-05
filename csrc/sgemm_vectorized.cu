#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdio>

constexpr int BM = 128;  // block tile M
constexpr int BN = 128;  // block tile N
constexpr int BK = 8;    // block tile K
constexpr int TM = 8;    // thread tile M (每个 thread 算 TM 行)
constexpr int TN = 8;    // thread tile N (每个 thread 算 TN 列)

__global__ void sgemm_vectorized_kernel(float* A, float* B, float* C, int M, int K, int N) {
  __shared__ float buf_A[BM][BK];
  __shared__ float buf_B[BK][BN];

  float tmp[TM][TN] = {};
  float reg_A[TM];
  float reg_B[TN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (int k = 0; k < K; k += BK) {
    // load buf_A: [BM][BK], each thread loads one float4
    {
      int a_row = tid / (BK / 4);
      int a_col = (tid % (BK / 4)) * 4;
      int global_row = blockIdx.y * BM + a_row;
      int global_col = k + a_col;
      if (global_row < M && global_col < K) {
        float4 tmp4 = reinterpret_cast<float4*>(&A[global_row * K + global_col])[0];
        buf_A[a_row][a_col] = tmp4.x;
        buf_A[a_row][a_col + 1] = tmp4.y;
        buf_A[a_row][a_col + 2] = tmp4.z;
        buf_A[a_row][a_col + 3] = tmp4.w;
      } else {
        buf_A[a_row][a_col] = 0.0f;
        buf_A[a_row][a_col + 1] = 0.0f;
        buf_A[a_row][a_col + 2] = 0.0f;
        buf_A[a_row][a_col + 3] = 0.0f;
      }
    }

    // load buf_B: [BK][BN], each thread loads one float4
    {
      int b_row = tid / (BN / 4);
      int b_col = (tid % (BN / 4)) * 4;
      int global_row = k + b_row;
      int global_col = blockIdx.x * BN + b_col;
      if (global_row < K && global_col < N) {
        float4 tmp4 = reinterpret_cast<float4*>(&B[global_row * N + global_col])[0];
        buf_B[b_row][b_col] = tmp4.x;
        buf_B[b_row][b_col + 1] = tmp4.y;
        buf_B[b_row][b_col + 2] = tmp4.z;
        buf_B[b_row][b_col + 3] = tmp4.w;
      } else {
        buf_B[b_row][b_col] = 0.0f;
        buf_B[b_row][b_col + 1] = 0.0f;
        buf_B[b_row][b_col + 2] = 0.0f;
        buf_B[b_row][b_col + 3] = 0.0f;
      }
    }

    __syncthreads();

    // compute: outer product per BK step
    for (int bk = 0; bk < BK; bk++) {
      for (int i = 0; i < TM; i++) {
        reg_A[i] = buf_A[threadIdx.y * TM + i][bk];
      }
      for (int j = 0; j < TN; j++) {
        reg_B[j] = buf_B[bk][threadIdx.x * TN + j];
      }
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          tmp[i][j] += reg_A[i] * reg_B[j];
        }
      }
    }

    __syncthreads();
  }

  // write back
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int row_idx = blockIdx.y * BM + threadIdx.y * TM + i;
      int col_idx = blockIdx.x * BN + threadIdx.x * TN + j;
      if (row_idx < M && col_idx < N) {
        C[row_idx * N + col_idx] = tmp[i][j];
      }
    }
  }
}

torch::Tensor sgemm_vectorized(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "dimension mismatch: A is MxK, B is KxN");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(A.size(0) % 4 == 0 && A.size(1) % 4 == 0 && B.size(1) % 4 == 0,
              "M, K, N must be multiples of 4 for float4 vectorized load");

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

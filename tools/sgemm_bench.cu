#include <cstdlib>

#include "../csrc/sgemm_naive.cu"
#include "../csrc/sgemm_tiled.cu"
#include "../csrc/sgemm_vectorized.cu"

// ==================== Benchmark ====================

using KernelFn = void (*)(float *, float *, float *, int, int, int);

struct BenchResult {
  float ms;
  float gflops;
  bool correct;
};

BenchResult bench_kernel(KernelFn kernel, dim3 grid, dim3 block, torch::Tensor A, torch::Tensor B,
                         torch::Tensor ref, int M, int K, int N, int warmup, int repeats) {
  torch::Tensor C = torch::zeros({M, N}, A.options());

  for (int i = 0; i < warmup; i++)
    kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < repeats; i++)
    kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms;
  cudaEventElapsedTime(&total_ms, start, stop);
  float avg_ms = total_ms / repeats;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float gflops = 2.0 * M * N * K / (avg_ms * 1e6);
  bool correct = torch::allclose(C, ref, 1e-3, 1e-3);

  return {avg_ms, gflops, correct};
}

int main(int argc, char **argv) {
  int M = 1024, K = 1024, N = 1024;
  if (argc >= 4) {
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = atoi(argv[3]);
  }

  const int warmup = 10, repeats = 50;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto A = torch::randn({M, K}, opts);
  auto B = torch::randn({K, N}, opts);
  auto ref = torch::mm(A, B);

  printf("SGEMM Benchmark: M=%d, K=%d, N=%d  (warmup=%d, repeats=%d)\n\n", M, K, N, warmup,
         repeats);
  printf("%-12s  %10s  %12s  %s\n", "Kernel", "Time", "GFLOPS", "Check");
  printf("----------------------------------------------\n");

  // cuBLAS (torch::mm)
  {
    for (int i = 0; i < warmup; i++) torch::mm(A, B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) torch::mm(A, B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeats;
    float gflops = 2.0 * M * N * K / (avg_ms * 1e6);

    printf("%-12s  %8.3f ms  %8.1f      ref\n", "cuBLAS", avg_ms, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Naive
  {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    auto r = bench_kernel(sgemm_naive_kernel, grid, block, A, B, ref, M, K, N, warmup, repeats);
    printf("%-12s  %8.3f ms  %8.1f      %s\n", "Naive", r.ms, r.gflops,
           r.correct ? "PASS" : "FAIL");
  }

  // Tiled (shared memory)
  {
    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid((N + WARP_SIZE - 1) / WARP_SIZE, (M + WARP_SIZE - 1) / WARP_SIZE);
    auto r = bench_kernel(sgemm_shared_kernel, grid, block, A, B, ref, M, K, N, warmup, repeats);
    printf("%-12s  %8.3f ms  %8.1f      %s\n", "Tiled", r.ms, r.gflops,
           r.correct ? "PASS" : "FAIL");
  }

  // Vectorized (float4 + register tiling)
  {
    constexpr int BLOCK_X = BN / TN;
    constexpr int BLOCK_Y = BM / TM;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto r =
        bench_kernel(sgemm_vectorized_kernel, grid, block, A, B, ref, M, K, N, warmup, repeats);
    printf("%-12s  %8.3f ms  %8.1f      %s\n", "Vectorized", r.ms, r.gflops,
           r.correct ? "PASS" : "FAIL");
  }

  return 0;
}

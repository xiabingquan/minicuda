#include <torch/types.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void transpose_shared_kernel(float *inp, float *out, int m, int n)
{
  __shared__ float tile[32][33];

  int li = threadIdx.y; // row within tile
  int lj = threadIdx.x; // col within tile

  // Load from input (coalesced read)
  int i = blockIdx.y * blockDim.y + li;
  int j = blockIdx.x * blockDim.x + lj;
  if (i < m && j < n)
    tile[li][lj] = inp[i * n + j];

  __syncthreads();

  // Write to output (coalesced write)
  int o_i = blockIdx.x * blockDim.y + li;
  int o_j = blockIdx.y * blockDim.x + lj;
  if (o_i < n && o_j < m)
    out[o_i * m + o_j] = tile[lj][li];
}

torch::Tensor transpose_shared(torch::Tensor inp)
{
  TORCH_CHECK(inp.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(inp.dim() == 2, "input must be 2D");
  TORCH_CHECK(inp.dtype() == torch::kFloat32, "input must be float32");

  int m = inp.size(0);
  int n = inp.size(1);
  torch::Tensor out = inp.new_empty({n, m});

  dim3 block_size(32, 32);
  dim3 grid_size(
      (n + block_size.x - 1) / block_size.x,
      (m + block_size.y - 1) / block_size.y);
  transpose_shared_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), m, n);

  return out;
}

// int main()
// {
//   auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

//   // 方阵
//   auto a = torch::randn({128, 128}, opts);
//   auto out = transpose_shared(a);
//   TORCH_CHECK(torch::equal(out, a.t()), "square matrix test failed");
//   printf("[PASS] 128x128\n");

//   // 非方阵
//   auto b = torch::randn({37, 123}, opts);
//   out = transpose_shared(b);
//   TORCH_CHECK(torch::equal(out, b.t()), "non-square matrix test failed");
//   printf("[PASS] 37x123\n");

//   // 大矩阵
//   auto c = torch::randn({1024, 2048}, opts);
//   out = transpose_shared(c);
//   TORCH_CHECK(torch::equal(out, c.t()), "large matrix test failed");
//   printf("[PASS] 1024x2048\n");

//   printf("All tests passed.\n");
//   return 0;
// }

#include <torch/types.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <utility>

__global__ void vector_add_kernel(float *a, float *b, float *c, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor cpu_large_vector_add_async(torch::Tensor a, torch::Tensor b, int buffer_size = 1024)
{
  nvtxRangePushA("async_total");
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "a and b must be on CPU");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "must be float32");
  TORCH_CHECK(a.dim() == 1 && b.dim() == 1, "must be 1D");
  TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch");

  std::vector<float *> buffers(2);
  cudaMallocHost(&buffers[0], buffer_size * sizeof(float) * 2);
  cudaMallocHost(&buffers[1], buffer_size * sizeof(float) * 2);

  cudaStream_t current;
  cudaStreamCreate(&current);

  int n = a.size(0), cur_offset = 0;
  auto a_ptr = a.data_ptr<float>(), b_ptr = b.data_ptr<float>();
  float *a_d_ptr, *b_d_ptr, *c_d_ptr;
  cudaMalloc(&a_d_ptr, buffer_size * sizeof(float));
  cudaMalloc(&b_d_ptr, buffer_size * sizeof(float));
  cudaMalloc(&c_d_ptr, n * sizeof(float));

  int cur_real_size = 0, next_real_size = 0;
  int cur_buffer_idx = 0, next_buffer_idx = 1;
  int num_loop = (n + buffer_size - 1) / buffer_size + 1;

  int block_size = 256, grid_size = -1;
  for (int i = 0; i < num_loop; i++)
  {
    if (i != 0)
    {
      nvtxRangePushA("gpu_submit");
      cudaStreamSynchronize(current);
      cudaMemcpyAsync(a_d_ptr, buffers[cur_buffer_idx], cur_real_size * sizeof(float), cudaMemcpyHostToDevice, current);
      cudaMemcpyAsync(b_d_ptr, buffers[cur_buffer_idx] + buffer_size, cur_real_size * sizeof(float), cudaMemcpyHostToDevice, current);
      grid_size = (cur_real_size + block_size - 1) / block_size;
      vector_add_kernel<<<grid_size, block_size>>>(a_d_ptr, b_d_ptr, c_d_ptr + cur_offset, cur_real_size);
      cur_offset += cur_real_size;
      nvtxRangePop();
    }
    if (i != num_loop - 1)
    {
      nvtxRangePushA("cpu_memcpy");
      next_real_size = min(buffer_size, n - cur_offset);
      memcpy(buffers[next_buffer_idx], a_ptr + cur_offset, next_real_size * sizeof(float));
      memcpy(buffers[next_buffer_idx] + buffer_size, b_ptr + cur_offset, next_real_size * sizeof(float));
      nvtxRangePop();
    }
    std::swap(cur_buffer_idx, next_buffer_idx);
    cur_real_size = next_real_size;
  }

  nvtxRangePushA("final_sync_d2h");
  cudaStreamSynchronize(current);

  auto c = torch::empty({n}, a.options());
  cudaMemcpy(c.data_ptr<float>(), c_d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  nvtxRangePop();

  cudaFreeHost(buffers[0]);
  cudaFreeHost(buffers[1]);
  cudaFree(a_d_ptr);
  cudaFree(b_d_ptr);
  cudaFree(c_d_ptr);
  cudaStreamDestroy(current);

  nvtxRangePop();
  return c;
}

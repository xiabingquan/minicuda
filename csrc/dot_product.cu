#include <torch/types.h>
#include <cuda_runtime.h>

template <int N>
__global__ void dot_product_kernel(float *a, float *b, float *c, int n)
{
  int li = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ float buf[N];

  if (i < n)
  {
    buf[li] = a[i] * b[i];
  }
  else
  {
    buf[li] = 0.;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (li < stride)
    {
      buf[li] += buf[li + stride];
    }
    __syncthreads();
  }
  if (li == 0)
  {
    atomicAdd(c, buf[0]);
  }
}

torch::Tensor dot_product(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.dim() == 1 && b.dim() == 1, "inputs must be 1D");
  TORCH_CHECK(a.size(0) == b.size(0), "size mismatch");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be float32");

  int n = a.size(0);
  torch::Tensor c = a.new_zeros({1});

  constexpr int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  dot_product_kernel<block_size><<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  return c;
}

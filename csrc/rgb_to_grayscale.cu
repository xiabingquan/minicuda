#include <torch/torch.h>

__global__ void rgb_to_grayscale_kernel(float *inp, float *out, int row, int col)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int offset = col * i + j;
  if (i < row && j < col)
  {
    out[offset] = 0.299 * inp[offset] + 0.587 * inp[row * col + offset] + 0.114 * inp[row * col * 2 + offset];
  }
}

torch::Tensor rgb_to_grayscale(torch::Tensor inp)
{
  TORCH_CHECK(inp.device().is_cuda(), "input must be on CUDA");
  TORCH_CHECK(inp.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(inp.dim() == 3, "input must be 3D (C, H, W)");
  TORCH_CHECK(inp.size(0) == 3, "input must have 3 channels");
  TORCH_CHECK(inp.is_contiguous(), "input must be contiguous");

  int row = inp.size(1), col = inp.size(2);
  auto out = inp.new_zeros({row, col});

  dim3 block_size(16, 16);
  dim3 grid_size(
      (row + block_size.x - 1) / block_size.x,
      (col + block_size.y - 1) / block_size.y);

  rgb_to_grayscale_kernel<<<grid_size, block_size>>>(
      inp.data_ptr<float>(), out.data_ptr<float>(), row, col);

  return out;
}

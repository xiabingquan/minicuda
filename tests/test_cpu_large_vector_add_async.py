import torch
from minicuda import cpu_large_vector_add_async


def test_basic():
    """Basic test: result should match a + b."""
    n = 1 << 16
    a = torch.randn(n, device="cpu")
    b = torch.randn(n, device="cpu")
    c = cpu_large_vector_add_async(a, b)
    assert c.device.type == "cpu", "output must be on CPU"
    assert torch.allclose(c, a + b, atol=1e-6), "basic test failed"


def test_smaller_than_buffer():
    """Tensor smaller than buffer_size."""
    a = torch.randn(100, device="cpu")
    b = torch.randn(100, device="cpu")
    c = cpu_large_vector_add_async(a, b, 1024)
    assert torch.allclose(c, a + b, atol=1e-6), "smaller-than-buffer test failed"


def test_equal_to_buffer():
    """Tensor exactly equals buffer_size."""
    n = 1024
    a = torch.randn(n, device="cpu")
    b = torch.randn(n, device="cpu")
    c = cpu_large_vector_add_async(a, b, 1024)
    assert torch.allclose(c, a + b, atol=1e-6), "equal-to-buffer test failed"


def test_non_aligned():
    """Tensor not a multiple of buffer_size."""
    n = 5000
    a = torch.randn(n, device="cpu")
    b = torch.randn(n, device="cpu")
    c = cpu_large_vector_add_async(a, b, 1024)
    assert torch.allclose(c, a + b, atol=1e-6), "non-aligned test failed"


def test_large():
    """Large tensor with default buffer_size."""
    n = 1 << 20
    a = torch.randn(n, device="cpu")
    b = torch.randn(n, device="cpu")
    c = cpu_large_vector_add_async(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "large tensor test failed"

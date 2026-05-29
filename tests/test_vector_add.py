import torch
from minicuda import vector_add


def test_vector_add_basic():
    """Basic test: result should match torch.add."""
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "basic test failed"


def test_vector_add_large():
    """Large size test: grid should cover all elements."""
    n = 1_000_000
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "large test failed"


def test_vector_add_non_aligned():
    """Non-aligned length test: length is not a multiple of block_size."""
    n = 999
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "non-aligned test failed"


def test_vector_add_multidim():
    """Multi-dim tensor test: kernel treats it as 1D via numel()."""
    a = torch.randn(32, 64, device="cuda")
    b = torch.randn(32, 64, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "multidim test failed"


if __name__ == "__main__":
    test_vector_add_basic()
    test_vector_add_large()
    test_vector_add_non_aligned()
    test_vector_add_multidim()
    print("All tests passed!")

import torch
from minicuda import vector_add_raw


def test_vector_add_raw_basic():
    """Basic test: result should match torch.add."""
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = vector_add_raw(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "basic test failed"


def test_vector_add_raw_large():
    """Large size test."""
    n = 1_000_000
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add_raw(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "large test failed"


def test_vector_add_raw_non_aligned():
    """Non-aligned length: not a multiple of block_size."""
    n = 999
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add_raw(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "non-aligned test failed"

import torch
from minicuda import matrix_add


def test_matrix_add_basic():
    """Basic test: result should match a + b."""
    a = torch.randn(16, 16, device="cuda")
    b = torch.randn(16, 16, device="cuda")
    c = matrix_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "basic test failed"


def test_matrix_add_non_aligned():
    """Non-aligned shape: not a multiple of block_size (16)."""
    a = torch.randn(17, 33, device="cuda")
    b = torch.randn(17, 33, device="cuda")
    c = matrix_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "non-aligned test failed"


def test_matrix_add_large():
    """Large matrix test."""
    a = torch.randn(512, 1024, device="cuda")
    b = torch.randn(512, 1024, device="cuda")
    c = matrix_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "large test failed"

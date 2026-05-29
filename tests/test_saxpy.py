import torch
from minicuda import saxpy


def test_saxpy_basic():
    """Basic test: result should match a * x + y."""
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")
    a = 2.5
    z = saxpy(x, y, a)
    assert torch.allclose(z, a * x + y, atol=1e-6), "basic test failed"


def test_saxpy_large():
    """Large size test: grid-stride loop should cover all elements."""
    n = 1_000_000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    a = -1.0
    z = saxpy(x, y, a)
    assert torch.allclose(z, a * x + y, atol=1e-6), "large test failed"


def test_saxpy_zero_scalar():
    """Zero scalar: should degenerate to z = y."""
    x = torch.randn(512, device="cuda")
    y = torch.randn(512, device="cuda")
    z = saxpy(x, y, 0.0)
    assert torch.allclose(z, y, atol=1e-6), "zero scalar test failed"

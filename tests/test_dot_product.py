import torch
from minicuda import dot_product


def test_basic():
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    out = dot_product(a, b)
    ref = torch.dot(a, b)
    assert torch.allclose(out.view([]), ref, atol=1e-3), \
        f"basic test failed: got {out.item()}, expected {ref.item()}"


def test_non_aligned():
    a = torch.randn(1023, device="cuda")
    b = torch.randn(1023, device="cuda")
    out = dot_product(a, b)
    ref = torch.dot(a, b)
    assert torch.allclose(out.view([]), ref, atol=1e-3), \
        f"non-aligned test failed: got {out.item()}, expected {ref.item()}"


def test_large():
    a = torch.randn(100000, device="cuda")
    b = torch.randn(100000, device="cuda")
    out = dot_product(a, b)
    ref = torch.dot(a, b)
    assert torch.allclose(out.view([]), ref, rtol=1e-2, atol=1e-1), \
        f"large test failed: got {out.item()}, expected {ref.item()}"

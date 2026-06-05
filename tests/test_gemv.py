import torch
from minicuda import gemv


def test_square():
    A = torch.randn(64, 64, device="cuda")
    x = torch.randn(64, device="cuda")
    out = gemv(A, x)
    ref = torch.mv(A, x)
    assert torch.allclose(out, ref, atol=1e-4), "square test failed"


def test_non_square():
    A = torch.randn(37, 123, device="cuda")
    x = torch.randn(123, device="cuda")
    out = gemv(A, x)
    ref = torch.mv(A, x)
    assert torch.allclose(out, ref, atol=1e-4), "non-square test failed"


def test_large():
    A = torch.randn(1024, 2048, device="cuda")
    x = torch.randn(2048, device="cuda")
    out = gemv(A, x)
    ref = torch.mv(A, x)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), "large test failed"

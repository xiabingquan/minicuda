import torch
from minicuda import sgemm_vectorized


def test_square():
    A = torch.randn(256, 256, device="cuda")
    B = torch.randn(256, 256, device="cuda")
    out = sgemm_vectorized(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "square test failed"


def test_large():
    A = torch.randn(1024, 1024, device="cuda")
    B = torch.randn(1024, 1024, device="cuda")
    out = sgemm_vectorized(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), "large test failed"


def test_non_square_aligned():
    A = torch.randn(128, 64, device="cuda")
    B = torch.randn(64, 256, device="cuda")
    out = sgemm_vectorized(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "non-square aligned test failed"


def test_non_aligned():
    A = torch.randn(36, 64, device="cuda")
    B = torch.randn(64, 124, device="cuda")
    out = sgemm_vectorized(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "non-aligned test failed"

import torch
from minicuda import sgemm_naive, sgemm_shared


def test_naive_square():
    A = torch.randn(128, 128, device="cuda")
    B = torch.randn(128, 128, device="cuda")
    out = sgemm_naive(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "naive square test failed"


def test_naive_non_square():
    A = torch.randn(37, 64, device="cuda")
    B = torch.randn(64, 123, device="cuda")
    out = sgemm_naive(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "naive non-square test failed"


def test_tiled_square():
    A = torch.randn(128, 128, device="cuda")
    B = torch.randn(128, 128, device="cuda")
    out = sgemm_shared(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "tiled square test failed"


def test_tiled_non_square():
    A = torch.randn(37, 64, device="cuda")
    B = torch.randn(64, 123, device="cuda")
    out = sgemm_shared(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, atol=1e-3), "tiled non-square test failed"


def test_tiled_large():
    A = torch.randn(512, 1024, device="cuda")
    B = torch.randn(1024, 256, device="cuda")
    out = sgemm_shared(A, B)
    ref = torch.mm(A, B)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), "tiled large test failed"

import torch
from minicuda import transpose_naive


def test_square():
    inp = torch.randn(128, 128, device="cuda")
    out = transpose_naive(inp)
    assert torch.equal(out, inp.T), "square matrix test failed"


def test_non_square():
    inp = torch.randn(37, 123, device="cuda")
    out = transpose_naive(inp)
    assert torch.equal(out, inp.T), "non-square matrix test failed"


def test_large():
    inp = torch.randn(1024, 2048, device="cuda")
    out = transpose_naive(inp)
    assert torch.equal(out, inp.T), "large matrix test failed"

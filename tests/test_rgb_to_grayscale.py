import torch
from minicuda import rgb_to_grayscale


def test_rgb_to_grayscale_basic():
    """Basic test: result should match weighted sum of channels."""
    inp = torch.rand(3, 16, 16, device="cuda")
    out = rgb_to_grayscale(inp)
    expected = 0.299 * inp[0] + 0.587 * inp[1] + 0.114 * inp[2]
    assert torch.allclose(out, expected, atol=1e-6), "basic test failed"


def test_rgb_to_grayscale_non_aligned():
    """Non-aligned shape: H, W not multiples of block_size (16)."""
    inp = torch.rand(3, 17, 33, device="cuda")
    out = rgb_to_grayscale(inp)
    expected = 0.299 * inp[0] + 0.587 * inp[1] + 0.114 * inp[2]
    assert torch.allclose(out, expected, atol=1e-6), "non-aligned test failed"


def test_rgb_to_grayscale_large():
    """Large image test."""
    inp = torch.rand(3, 1080, 1920, device="cuda")
    out = rgb_to_grayscale(inp)
    expected = 0.299 * inp[0] + 0.587 * inp[1] + 0.114 * inp[2]
    assert torch.allclose(out, expected, atol=1e-6), "large test failed"


def test_rgb_to_grayscale_pure_channels():
    """Pure color test: verify each coefficient individually."""
    h, w = 8, 8
    # Pure red -> gray = 0.299
    inp = torch.zeros(3, h, w, device="cuda")
    inp[0] = 1.0
    out = rgb_to_grayscale(inp)
    assert torch.allclose(out, torch.full((h, w), 0.299, device="cuda"), atol=1e-6), "pure red failed"

    # Pure green -> gray = 0.587
    inp = torch.zeros(3, h, w, device="cuda")
    inp[1] = 1.0
    out = rgb_to_grayscale(inp)
    assert torch.allclose(out, torch.full((h, w), 0.587, device="cuda"), atol=1e-6), "pure green failed"

    # Pure blue -> gray = 0.114
    inp = torch.zeros(3, h, w, device="cuda")
    inp[2] = 1.0
    out = rgb_to_grayscale(inp)
    assert torch.allclose(out, torch.full((h, w), 0.114, device="cuda"), atol=1e-6), "pure blue failed"

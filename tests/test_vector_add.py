import torch
from minicuda import vector_add


def test_vector_add_basic():
    """基本功能测试: 结果与 torch.add 对齐。"""
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "basic test failed"


def test_vector_add_large():
    """大尺寸测试: 验证 grid 能正确覆盖所有元素。"""
    n = 1_000_000
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "large test failed"


def test_vector_add_non_aligned():
    """非对齐长度测试: 长度不是 block_size 的整数倍。"""
    n = 999
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = vector_add(a, b)
    assert torch.allclose(c, a + b, atol=1e-6), "non-aligned test failed"


def test_vector_add_multidim():
    """多维张量测试: 虽然 kernel 按一维处理, 但 numel() 应正确覆盖。"""
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

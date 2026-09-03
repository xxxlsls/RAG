import torch
import torchvision

print("=" * 50)
print("PyTorch 安装校验")
print("=" * 50)

# 1. 版本信息
print(f"\n✓ PyTorch 版本: {torch.__version__}")
print(f"✓ Torchvision 版本: {torchvision.__version__}")

# 2. CUDA 支持
print(f"\n{'=' * 50}")
print("CUDA 支持检查")
print("=" * 50)
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # 3. 简单计算测试
    print(f"\n{'=' * 50}")
    print("GPU 计算测试")
    print("=" * 50)

    try:
        # 创建测试张量并移到 GPU
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = torch.mm(x, y)
        print("✓ GPU 矩阵乘法测试成功")
        print(f"  结果设备: {z.device}")
    except Exception as e:
        print(f"✗ GPU 计算测试失败: {e}")
else:
    print("\n⚠ CUDA 不可用，将使用 CPU")
    print("提示：如果你安装了 CUDA 版本的 PyTorch，请检查：")
    print("  1. NVIDIA 显卡驱动是否正确安装")
    print("  2. CUDA Toolkit 版本是否匹配")
    print("  3. 环境变量是否正确配置")

print(f"\n{'=' * 50}")


"""
==================================================
PyTorch 安装校验
==================================================

✓ PyTorch 版本: 2.7.1+cu118
✓ Torchvision 版本: 0.22.1+cu118

==================================================
CUDA 支持检查
==================================================
✓ CUDA 可用: True
✓ CUDA 版本: 11.8
✓ cuDNN 版本: 90100
✓ GPU 数量: 1
  - GPU 0: GeForce RTX 2060

==================================================
GPU 计算测试
==================================================
✓ GPU 矩阵乘法测试成功
  结果设备: cuda:0

==================================================
"""
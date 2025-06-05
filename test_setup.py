#!/usr/bin/env python3
"""
环境和代码测试脚本
用于验证项目设置是否正确
"""

import sys
import os
import importlib

# 先导入ale_py以注册ALE命名空间
try:
    import ale_py
except ImportError:
    pass

# 添加gymnasium到gym的兼容性映射
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    pass

def test_imports():
    """
    测试必要的库是否可以正常导入
    """
    print("测试库导入...")
    
    required_packages = [
        'torch',
        'numpy', 
        'gym',
        'cv2',
        'matplotlib',
        'tqdm'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n错误: 以下包导入失败: {failed_imports}")
        print("请确保已正确安装 conda 环境")
        return False
    
    print("\n所有必要的包都已正确安装！")
    return True

def test_gym_environment():
    """
    测试 Gym 环境是否可以正常创建
    """
    print("\n测试 Gym 环境...")
    
    try:
        import gymnasium as gym
        
        # 测试创建环境
        env = gym.make('ALE/Assault-v5', render_mode="rgb_array")
        print(f"✓ 环境创建成功: {env.spec.id}")
        
        # 测试环境基本功能
        obs, info = env.reset()
        print(f"✓ 环境重置成功，观测形状: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ 环境步进成功，动作空间大小: {env.action_space.n}")
        
        env.close()
        print("✓ Gym 环境测试通过")
        return True
        
    except Exception as e:
        print(f"✗ Gym 环境测试失败: {e}")
        return False

def test_custom_modules():
    """
    测试自定义模块是否可以正常导入
    """
    print("\n测试自定义模块...")
    
    # 添加 src 目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    modules = [
        'model',
        'agent', 
        'utils'
    ]
    
    failed_modules = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}.py")
        except ImportError as e:
            print(f"✗ {module}.py: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n错误: 以下模块导入失败: {failed_modules}")
        return False
    
    print("\n所有自定义模块都可以正常导入！")
    return True

def test_model_creation():
    """
    测试模型是否可以正常创建
    """
    print("\n测试模型创建...")
    
    try:
        import torch
        from model import DQN, DuelingDQN
        
        # 测试 DQN 模型
        input_shape = (4, 84, 84)  # 4帧堆叠，84x84像素
        n_actions = 14  # Assault 游戏的动作数
        
        dqn = DQN(input_shape, n_actions)
        print(f"✓ DQN 模型创建成功，参数数量: {sum(p.numel() for p in dqn.parameters()):,}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, *input_shape)
        output = dqn(dummy_input)
        print(f"✓ DQN 前向传播成功，输出形状: {output.shape}")
        
        # 测试 Dueling DQN 模型
        dueling_dqn = DuelingDQN(input_shape, n_actions)
        print(f"✓ Dueling DQN 模型创建成功，参数数量: {sum(p.numel() for p in dueling_dqn.parameters()):,}")
        
        # 测试前向传播
        output = dueling_dqn(dummy_input)
        print(f"✓ Dueling DQN 前向传播成功，输出形状: {output.shape}")
        
        print("\n模型创建测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False

def test_environment_wrapper():
    """
    测试环境包装器
    """
    print("\n测试环境包装器...")
    
    try:
        from utils import make_env
        
        # 创建包装后的环境
        env = make_env('ALE/Assault-v5')
        print(f"✓ 包装环境创建成功")
        
        # 测试环境功能
        obs, info = env.reset()
        print(f"✓ 环境重置成功，观测形状: {obs.shape}")
        
        # 检查观测是否为堆叠的帧
        if len(obs.shape) == 3 and obs.shape[0] == 4:
            print("✓ 帧堆叠正常工作")
        else:
            print(f"✗ 帧堆叠异常，期望形状 (4, H, W)，实际形状 {obs.shape}")
            return False
        
        # 测试几步
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
        
        print("✓ 环境包装器测试通过")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ 环境包装器测试失败: {e}")
        return False

def test_device_availability():
    """
    测试计算设备可用性
    """
    print("\n测试计算设备...")
    
    try:
        import torch
        
        # 检查 CUDA 可用性
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA 版本: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            print("✓ 使用 CPU 设备")
        
        # 测试张量创建和运算
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        print(f"✓ 设备运算测试通过，结果形状: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 设备测试失败: {e}")
        return False

def main():
    """
    运行所有测试
    """
    print("="*50)
    print("强化学习项目环境测试")
    print("="*50)
    
    tests = [
        test_imports,
        test_gym_environment,
        test_custom_modules,
        test_model_creation,
        test_environment_wrapper,
        test_device_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print("\n" + "="*50)
    print(f"测试结果: {passed}/{total} 通过")
    print("="*50)
    
    if passed == total:
        print("🎉 所有测试通过！项目环境配置正确。")
        print("\n可以开始训练了:")
        print("  python src/train.py")
        print("  或者运行: ./run_training.sh")
        return True
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
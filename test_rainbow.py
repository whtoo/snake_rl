#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rainbow DQN 实现测试脚本
验证所有组件是否能正常工作
"""

import torch
import numpy as np
from src.model import RainbowDQN
from src.agent import RainbowAgent
from src.buffers.n_step_buffers import NStepBuffer
from src.utils import make_env

def test_noisy_linear():
    """测试噪声线性层"""
    print("测试噪声线性层...")
    from src.model import NoisyLinear
    
    layer = NoisyLinear(64, 32, factorised=True)
    x = torch.randn(10, 64)
    
    # 测试前向传播
    output1 = layer(x)
    output2 = layer(x)
    
    # 噪声网络在没有重置噪声的情况下，连续调用应该产生相同的结果
    assert torch.equal(output1, output2), "噪声网络在没有重置噪声的情况下，连续调用输出应该相同"
    
    # 测试重置噪声
    layer.sample_noise() # Correct method name
    output3 = layer(x)
    assert not torch.equal(output1, output3), "重置噪声后输出应该不同"
    
    print("✅ 噪声线性层测试通过")

def test_rainbow_dqn():
    """测试 Rainbow DQN 网络"""
    print("测试 Rainbow DQN 网络...")
    
    input_shape = (4, 84, 84)
    n_actions = 6
    n_atoms = 51
    
    # 测试标准 Q 网络模式
    model_standard = RainbowDQN(
        input_shape=input_shape,
        n_actions=n_actions,
        use_noisy=False,
        use_distributional=False
    )
    
    x = torch.randn(2, *input_shape)
    output = model_standard(x)
    assert output.shape == (2, n_actions), f"标准模式输出形状错误: {output.shape}"
    
    # 测试分布式 Q 网络模式
    model_distributional = RainbowDQN(
        input_shape=input_shape,
        n_actions=n_actions,
        use_noisy=False,
        use_distributional=True,
        n_atoms=n_atoms
    )
    
    output = model_distributional(x)
    assert output.shape == (2, n_actions, n_atoms), f"分布式模式输出形状错误: {output.shape}"
    
    # 测试噪声网络
    model_noisy = RainbowDQN(
        input_shape=input_shape,
        n_actions=n_actions,
        use_noisy=True,
        use_distributional=False
    )
    
    output1 = model_noisy(x)
    output2 = model_noisy(x)
    # 噪声网络在没有重置噪声的情况下，连续调用应该产生相同的结果
    assert torch.equal(output1, output2), "RainbowDQN (noisy) 在没有重置噪声的情况下，连续调用输出应该相同"
    
    model_noisy.sample_noise() # Correct method name
    output3 = model_noisy(x)
    assert not torch.equal(output1, output3), "重置噪声后输出应该不同"
    
    print("✅ Rainbow DQN 网络测试通过")

def test_n_step_buffer():
    """测试 N 步缓冲区"""
    print("测试 N 步缓冲区...")
    
    buffer = NStepBuffer(n_step=3, gamma=0.99)
    
    # 添加经验
    states = [np.random.randn(4, 84, 84) for _ in range(5)]
    actions = [0, 1, 2, 0, 1]
    rewards = [1.0, 2.0, -1.0, 0.5, 1.5]
    dones = [False, False, False, False, True]
    
    experiences = []
    for i in range(5):
        next_state = states[i+1] if i < 4 else states[i]
        exp = buffer.add(states[i], actions[i], rewards[i], next_state, dones[i])
        if exp is not None:
            experiences.append(exp)
    
    # 游戏结束，获取剩余经验
    remaining = buffer.get_last_n_step()
    experiences.extend(remaining)
    
    # 验证 n 步回报计算
    assert len(experiences) > 0, "应该有 N 步经验"
    
    # 验证第一个 3 步经验的回报
    expected_reward = rewards[0] + 0.99 * rewards[1] + 0.99**2 * rewards[2]
    assert abs(experiences[0][2] - expected_reward) < 1e-6, f"N步回报计算错误: {experiences[0][2]} vs {expected_reward}"
    
    print("✅ N 步缓冲区测试通过")

def test_rainbow_agent():
    """测试 Rainbow 智能体"""
    print("测试 Rainbow 智能体...")
    
    # 创建简单环境
    try:
        env = make_env("ALE/Assault-v5")
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
    except:
        print("⚠️  无法创建 Atari 环境，使用模拟参数")
        input_shape = (4, 84, 84)
        n_actions = 6
        env = None
    
    # 创建模型
    model = RainbowDQN(
        input_shape=input_shape,
        n_actions=n_actions,
        use_noisy=True,
        use_distributional=True,
        n_atoms=51
    )
    
    target_model = RainbowDQN(
        input_shape=input_shape,
        n_actions=n_actions,
        use_noisy=True,
        use_distributional=True,
        n_atoms=51
    )
    
    device = torch.device("cpu")
    
    # 创建 Rainbow 智能体
    agent = RainbowAgent(
        model=model,
        target_model=target_model,
        env=env,
        device=device,
        n_step=3,
        use_noisy=True,
        use_distributional=True,
        batch_size=4,  # 小批量便于测试
        buffer_size=1000
    )
    
    # 测试动作选择
    state = np.random.randn(*input_shape)
    action = agent.select_action(state)
    assert 0 <= action < n_actions, f"动作超出范围: {action}"
    
    # 测试经验存储
    next_state = np.random.randn(*input_shape)
    agent.store_experience(state, action, 1.0, next_state, False)
    
    # 添加更多经验以便测试更新
    for _ in range(20):
        state = np.random.randn(*input_shape)
        action = np.random.randint(n_actions)
        reward = np.random.randn()
        next_state = np.random.randn(*input_shape)
        done = np.random.random() < 0.1
        agent.store_experience(state, action, reward, next_state, done)
    
    # 测试模型更新
    if len(agent.memory) >= agent.batch_size:
        loss = agent.update_model()
        assert loss >= 0, f"损失应该非负: {loss}"
        print(f"  模型更新损失: {loss:.6f}")
    
    print("✅ Rainbow 智能体测试通过")

def main():
    """运行所有测试"""
    print("=" * 50)
    print("Rainbow DQN 实现测试")
    print("=" * 50)
    
    try:
        test_noisy_linear()
        test_rainbow_dqn()
        test_n_step_buffer()
        test_rainbow_agent()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！Rainbow DQN 实现正确！")
        print("=" * 50)
        
        print("\n使用示例:")
        print("python run_training.py rainbow 100 --use_noisy --use_distributional")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
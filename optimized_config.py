#!/usr/bin/env python3
"""
优化配置文件 - Rainbow DQN 损失优化参数

本文件包含经过优化的超参数设置，旨在降低训练损失并提高模型性能。
使用方法：python src/train.py --config optimized_config.py
"""

# 基础训练参数
OPTIMIZED_CONFIG = {
    # 核心超参数优化
    'lr': 3e-5,                    # 降低学习率：从 1e-4 到 3e-5
    'batch_size': 64,              # 增加批量大小：从 32 到 64
    'target_update': 2000,         # 增加目标网络更新间隔：从 1000 到 2000
    'gamma': 0.99,                 # 保持折扣因子不变
    
    # 缓冲区和经验回放
    'buffer_size': 100000,         # 保持缓冲区大小
    'prioritized_replay': True,    # 启用优先经验回放
    
    # 探索策略（非噪声网络）
    'epsilon_start': 1.0,
    'epsilon_final': 0.01,
    'epsilon_decay': 150000,       # 增加衰减步数，更平缓的探索
    
    # Rainbow DQN 特有参数
    'use_noisy': True,             # 启用噪声网络
    'use_distributional': True,    # 启用分布式Q学习
    'n_atoms': 51,                 # 分布原子数
    'v_min': -10,                  # 值函数范围
    'v_max': 10,
    
    # 自适应 N-step 参数优化
    'base_n_step': 3,
    'max_n_step': 8,               # 降低最大N步：从 10 到 8
    'adapt_n_step_freq': 1500,     # 增加调整频率：从 1000 到 1500
    'td_error_threshold_low': 0.08, # 降低阈值：从 0.1 到 0.08
    'td_error_threshold_high': 0.4, # 降低阈值：从 0.5 到 0.4
    
    # 状态增强参数
    'use_state_augmentation': True,
    'aug_noise_scale': 3.0,        # 降低噪声强度：从 5.0 到 3.0
    
    # 训练控制
    'episodes': 2000,              # 增加训练回合数
    'save_interval': 50,           # 更频繁保存：从 100 到 50
    'eval_interval': 10,           # 保持评估间隔
    
    # 网络结构优化参数
    'noisy_sigma_init': 0.2,       # 降低噪声初始化：从 0.4 到 0.2
    'grad_clip_norm': 1.0,         # 更严格的梯度裁剪：从 10 到 1.0
    
    # 学习率调度参数
    'use_lr_scheduler': True,      # 启用学习率调度
    'lr_decay_factor': 0.95,       # 学习率衰减因子
    'lr_decay_steps': 10000,       # 每10000步衰减一次
    'min_lr': 1e-6,                # 最小学习率
    
    # 损失函数优化
    'use_huber_loss': True,        # 启用 Huber Loss
    'huber_delta': 1.0,            # Huber Loss 的 delta 参数
    
    # 预热训练参数
    'warmup_steps': 1000,          # 预热步数
    'warmup_lr_factor': 0.1,       # 预热期学习率因子
    
    # 监控和日志
    'log_loss_freq': 100,          # 每100步记录详细损失
    'log_grad_norm': True,         # 记录梯度范数
    'log_q_values': True,          # 记录Q值统计
}

# 验证配置
def validate_config(config):
    """
    验证配置参数的合理性
    """
    assert config['lr'] > 0, "学习率必须为正数"
    assert config['batch_size'] > 0, "批量大小必须为正数"
    assert 0 < config['gamma'] <= 1, "折扣因子必须在(0,1]范围内"
    assert config['v_min'] < config['v_max'], "值函数范围设置错误"
    assert config['base_n_step'] <= config['max_n_step'], "N步参数设置错误"
    print("✓ 配置验证通过")

# 配置比较函数
def compare_with_default():
    """
    与默认配置进行比较
    """
    default_config = {
        'lr': 1e-4,
        'batch_size': 32,
        'target_update': 1000,
        'max_n_step': 10,
        'grad_clip_norm': 10,
        'noisy_sigma_init': 0.4,
    }
    
    print("配置对比：")
    for key in default_config:
        if key in OPTIMIZED_CONFIG:
            print(f"{key}: {default_config[key]} → {OPTIMIZED_CONFIG[key]}")

if __name__ == "__main__":
    validate_config(OPTIMIZED_CONFIG)
    compare_with_default()
    print("\n优化配置已准备就绪！")
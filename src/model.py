import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DQN(nn.Module):
    """
    深度Q网络 (DQN) 模型
    输入: 游戏画面 (预处理后的图像)
    输出: 每个可能动作的Q值
    """

    def __init__(self, input_shape, n_actions):
        """
        初始化DQN模型

        参数:
            input_shape: 输入图像的形状 (通道, 高度, 宽度)
            n_actions: 可用动作的数量
        """
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 计算卷积层输出的特征图大小
        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """
        计算卷积层输出的特征图大小
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像张量，形状为 (batch_size, channels, height, width)

        返回:
            各个动作的Q值
        """
        # 确保输入是浮点型并归一化到[0,1]
        x = x.float() / 255.0

        # 通过卷积层
        conv_out = self.conv(x)

        # 展平卷积输出
        conv_out = conv_out.view(conv_out.size(0), -1)

        # 通过全连接层
        return self.fc(conv_out)


class DuelingDQN(nn.Module):
    """
    Dueling DQN 架构
    将Q值分解为状态价值函数V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """

    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class NoisyLinear(nn.Module):
    """
    噪声线性层 - 替代传统的 epsilon-greedy 探索

    参数:
        in_features: 输入特征数
        out_features: 输出特征数
        sigma_init: 噪声初始化标准差
        factorised: 是否使用因子化噪声
    """

    def __init__(self, in_features, out_features, sigma_init=0.4, factorised=True):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.factorised = factorised

        # 权重参数：均值和标准差
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_weight = nn.Parameter(
            torch.zeros(out_features, in_features))

        # 偏置参数：均值和标准差
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))

        # 噪声缓存（不是参数，不参与梯度计算）
        self.register_buffer(
            'epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)

        self.sigma_weight.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(
            self.sigma_init / math.sqrt(self.out_features))

    def sample_noise(self):
        """采样噪声"""
        device = self.mu_weight.device
        if self.factorised:
            # 因子化噪声：减少参数数量
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            # 使用外积生成因子化噪声
            noise_weight = torch.outer(epsilon_out, epsilon_in)
            self.epsilon_weight.data = noise_weight.to(device)
            self.epsilon_bias.data = epsilon_out.to(device)
        else:
            # 独立噪声
            self.epsilon_weight.data = torch.randn(
                self.out_features, self.in_features, device=device)
            self.epsilon_bias.data = torch.randn(
                self.out_features, device=device)

    def _scale_noise(self, size):
        """缩放噪声"""
        device = self.mu_weight.device
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        """前向传播"""
        if self.training:
            # 训练时使用噪声 - 确保类型一致
            epsilon_w = getattr(self, 'epsilon_weight')  # type: torch.Tensor
            epsilon_b = getattr(self, 'epsilon_bias')    # type: torch.Tensor
            weight = self.mu_weight + self.sigma_weight * epsilon_w
            bias = self.mu_bias + self.sigma_bias * epsilon_b
        else:
            # 评估时使用均值
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN 网络架构
    集成 Dueling + Noisy Networks + Distributional DQN
    """

    def __init__(self, input_shape, n_actions, n_atoms=51, v_min=-10, v_max=10,
                 use_noisy=True, use_distributional=False):
        super(RainbowDQN, self).__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional

        # 卷积特征提取层（复用现有设计）
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # 选择线性层类型
        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        # Dueling 架构
        if use_distributional:
            # 分布式输出
            self.value_stream = nn.Sequential(
                LinearLayer(conv_out_size, 512),
                nn.ReLU(),
                LinearLayer(512, n_atoms)
            )
            self.advantage_stream = nn.Sequential(
                LinearLayer(conv_out_size, 512),
                nn.ReLU(),
                LinearLayer(512, n_actions * n_atoms)
            )
        else:
            # 标量输出
            self.value_stream = nn.Sequential(
                LinearLayer(conv_out_size, 512),
                nn.ReLU(),
                LinearLayer(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                LinearLayer(conv_out_size, 512),
                nn.ReLU(),
                LinearLayer(512, n_actions)
            )

    def _get_conv_out(self, shape):
        """计算卷积输出尺寸"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """前向传播"""
        x = x.float() / 255.0
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        if self.use_distributional:
            # 分布式 Dueling
            batch_size = x.size(0)
            value = value.view(batch_size, 1, self.n_atoms)
            advantage = advantage.view(
                batch_size, self.n_actions, self.n_atoms)

            # Dueling 架构合并
            q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
            # 应用 softmax 得到概率分布
            q_dist = F.softmax(q_dist, dim=2)

            return q_dist
        else:
            # 标量 Dueling
            return value + advantage - advantage.mean(dim=1, keepdim=True)

    def sample_noise(self):
        """为所有噪声层采样新的噪声"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.sample_noise()

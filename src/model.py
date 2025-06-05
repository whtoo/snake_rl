import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
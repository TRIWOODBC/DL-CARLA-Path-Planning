"""
模型定义 - PilotNet CNN 架构
"""
import torch
import torch.nn as nn


class PilotNet(nn.Module):
    """
    NVIDIA PilotNet 端到端自动驾驶模型
    输入: 3x66x200 (RGB 图像)
    输出: 1 (转向角)
    """
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 输入: 3x66x200
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # -> 24x31x98
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # -> 36x14x47
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # -> 48x5x22
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # -> 64x3x20
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64x1x18
            nn.ReLU()
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 输出一个值：转向角
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

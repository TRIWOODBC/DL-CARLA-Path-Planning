"""
模型定义 - PilotNet 神经网络
基于 NVIDIA 端到端自动驾驶论文
"""
import torch
import torch.nn as nn


class PilotNet(nn.Module):
    """
    PilotNet: 端到端自动驾驶卷积神经网络
    
    输入: 3x66x200 的 RGB 图像
    输出: 1 个值（转向角）
    
    网络结构:
    - 5 层卷积层（带 BatchNorm 和 Dropout）
    - 4 层全连接层
    """
    
    def __init__(self, dropout_rate=0.2):
        super(PilotNet, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 输入: 3x66x200
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # -> 24x31x98
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # -> 36x14x47
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # -> 48x5x22
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # -> 64x3x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64x1x18
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(50, 10),
            nn.ReLU(),
            
            nn.Linear(10, 1)  # 输出转向角
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class PilotNetLegacy(nn.Module):
    """
    原始 PilotNet（无 BatchNorm 和 Dropout）
    用于加载旧模型权重
    """
    
    def __init__(self):
        super(PilotNetLegacy, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def load_model(model_path, device, use_legacy=False):
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        device: 运行设备 (cuda/cpu)
        use_legacy: 是否使用旧版模型结构
    
    Returns:
        加载好的模型
    """
    if use_legacy:
        model = PilotNetLegacy()
    else:
        model = PilotNet()
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

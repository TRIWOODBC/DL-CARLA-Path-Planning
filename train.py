import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- 1. 超参数与设置 ----------
# 请根据你的实际情况修改这些路径和参数
DATA_DIR = "data_more"  # 包含所有图像文件夹和CSV文件的根目录
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
VALIDATION_CSV = os.path.join(DATA_DIR, "validation_labels.csv")

# 训练参数
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4  # 学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"将使用设备: {DEVICE}")

# ---------- 2. 自定义数据集类 (CarlaDataset) ----------
class CarlaDataset(Dataset):
    def __init__(self, data_dir, csv_file, is_training=True):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(csv_file)
        self.is_training = is_training

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['filename'])
        
        # 使用OpenCV读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            # 返回一个黑色的图像和0标签作为容错处理
            return torch.zeros(3, 66, 200), torch.tensor([0.0], dtype=torch.float32)

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        steer = float(row['steer'])

        # --- 数据增强与预处理 ---
        # 1. 数据增强：只在训练时进行水平翻转
        if self.is_training and random.random() > 0.5:
            image = cv2.flip(image, 1)  # 1表示水平翻转
            steer = -steer

        # 2. 图像裁剪 (Crop)：去掉天空和车头盖
        # 这个裁剪区域[60:-25, :, :]是针对Udacity数据集的经典设置，对CARLA同样有效
        image = image[60:-25, :, :]

        # 3. 图像缩放 (Resize)：缩放到模型期望的输入尺寸
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

        # 4. 归一化与格式转换
        # (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        # 转换为浮点数并归一化到 [0, 1]
        image = torch.from_numpy(image).float() / 255.0
        
        # 标签也转换为Tensor
        label = torch.tensor([steer], dtype=torch.float32)

        return image, label

# ---------- 3. 模型架构 (PilotNet) ----------
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 输入: 3x66x200
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # -> 24x31x98
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # -> 36x14x47
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # -> 48x5x22
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), # -> 64x3x20
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> 64x1x18
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
            nn.Linear(10, 1) # 输出一个值：转向角
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x

# ---------- 4. 训练与验证主逻辑 ----------
def main():
    # 实例化数据集和数据加载器
    train_dataset = CarlaDataset(data_dir=DATA_DIR, csv_file=TRAIN_CSV, is_training=True)
    val_dataset = CarlaDataset(data_dir=DATA_DIR, csv_file=VALIDATION_CSV, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 实例化模型、损失函数和优化器
    model = PilotNet().to(DEVICE)
    criterion = nn.MSELoss() # 均方误差损失，适合回归任务
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练历史记录
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    # 开始训练循环
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): # 在验证阶段不计算梯度
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for images, labels in progress_bar_val:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                progress_bar_val.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")

        # --- 保存最佳模型 ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best model saved with validation loss: {best_val_loss:.6f}")

    print("训练完成!")
    
    # ---------- 5. 绘制并保存损失曲线图 ----------
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()


if __name__ == '__main__':
    main()
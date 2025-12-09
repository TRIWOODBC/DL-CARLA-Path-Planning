"""
训练脚本 - PilotNet 端到端自动驾驶模型训练
功能：
  - 使用统一配置
  - 学习率调度器
  - 早停机制
  - 数据增强
  - 训练曲线可视化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS, DEVICE,
    MODEL_PATH, CROP_TOP, CROP_BOTTOM, INPUT_WIDTH, INPUT_HEIGHT
)
from model import PilotNet

# 数据目录（使用合并后的数据）
DATA_DIR = "data_merged"
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
VALIDATION_CSV = os.path.join(DATA_DIR, "validation_labels.csv")

# 早停配置
EARLY_STOP_PATIENCE = 7  # 连续多少个epoch没有改善就停止


class CarlaDataset(Dataset):
    """CARLA 自动驾驶数据集"""
    
    def __init__(self, csv_file, is_training=True):
        self.labels_df = pd.read_csv(csv_file)
        self.is_training = is_training
        print(f"加载数据集: {len(self.labels_df)} 条数据")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        # filename 已经包含完整路径（如 data_v3/images_000/000001.png）
        img_path = row['filename']
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            return torch.zeros(3, INPUT_HEIGHT, INPUT_WIDTH), torch.tensor([0.0], dtype=torch.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steer = float(row['steer'])

        # 数据增强（仅训练时）
        if self.is_training:
            image, steer = self._augment(image, steer)

        # 预处理
        image = self._preprocess(image)
        label = torch.tensor([steer], dtype=torch.float32)

        return image, label
    
    def _augment(self, image, steer):
        """数据增强"""
        # 1. 随机水平翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            steer = -steer
        
        # 2. 随机亮度调整
        if random.random() > 0.5:
            factor = 0.5 + random.random()  # 0.5 ~ 1.5
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # 3. 随机添加阴影
        if random.random() > 0.7:
            image = self._add_shadow(image)
        
        return image, steer
    
    def _add_shadow(self, image):
        """添加随机阴影"""
        h, w = image.shape[:2]
        x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
        
        shadow = np.zeros_like(image)
        shadow[:, x1:x2] = 1
        
        image = image.copy()
        image[:, x1:x2] = (image[:, x1:x2] * 0.5).astype(np.uint8)
        return image
    
    def _preprocess(self, image):
        """图像预处理"""
        # 1. 裁剪（去掉天空和车头）
        image = image[CROP_TOP:-CROP_BOTTOM, :, :]
        
        # 2. 缩放到模型输入尺寸
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 3. 转换格式 (H,W,C) -> (C,H,W)
        image = np.transpose(image, (2, 0, 1))
        
        # 4. 归一化到 [0, 1]
        image = torch.from_numpy(image).float() / 255.0
        
        return image


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    progress = tqdm(loader, desc="训练中")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress = tqdm(loader, desc="验证中")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader.dataset)


def plot_history(history, save_path='loss_curve.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], label='Learning Rate', color='green')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"训练曲线已保存: {save_path}")


def main():
    print("=" * 60)
    print("PilotNet 训练脚本")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"权重衰减: {WEIGHT_DECAY}")
    print(f"最大轮数: {EPOCHS}")
    print("=" * 60)
    
    # 数据加载
    print("\n加载数据...")
    train_dataset = CarlaDataset(TRAIN_CSV, is_training=True)
    val_dataset = CarlaDataset(VALIDATION_CSV, is_training=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 模型、损失函数、优化器
    print("\n初始化模型...")
    model = PilotNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器：验证损失不下降时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 早停
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*40}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"\nTrain Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, MODEL_PATH)
            print(f"✅ 保存最佳模型 (val_loss: {val_loss:.6f})")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠️ 早停触发！连续 {EARLY_STOP_PATIENCE} 轮无改善")
            break
    
    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存: {MODEL_PATH}")
    
    # 绘制训练曲线
    plot_history(history)


if __name__ == '__main__':
    main()

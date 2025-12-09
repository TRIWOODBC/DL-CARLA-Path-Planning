"""
测试脚本 - 在测试集上评估模型性能
注意：测试集只用于最终评估，不用于调参！
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from config import (
    BATCH_SIZE, DEVICE, MODEL_PATH,
    CROP_TOP, CROP_BOTTOM, INPUT_WIDTH, INPUT_HEIGHT
)
from model import PilotNet

# 测试数据
DATA_DIR = "data_merged"
TEST_CSV = os.path.join(DATA_DIR, "test_labels.csv")


class CarlaTestDataset(Dataset):
    """测试数据集（无数据增强）"""
    
    def __init__(self, csv_file):
        self.labels_df = pd.read_csv(csv_file)
        print(f"加载测试集: {len(self.labels_df)} 条数据")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = row['filename']
        
        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros(3, INPUT_HEIGHT, INPUT_WIDTH), torch.tensor([0.0])
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steer = float(row['steer'])
        
        # 预处理（与训练时一致，但无数据增强）
        image = image[CROP_TOP:-CROP_BOTTOM, :, :]
        image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0
        
        return image, torch.tensor([steer], dtype=torch.float32)


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="测试中"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def plot_results(preds, labels, save_path='test_results.png'):
    """可视化测试结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 预测 vs 真实值散点图
    ax1 = axes[0]
    ax1.scatter(labels, preds, alpha=0.5, s=10)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='理想预测')
    ax1.set_xlabel('真实转向值')
    ax1.set_ylabel('预测转向值')
    ax1.set_title('预测 vs 真实')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    
    # 2. 误差分布
    ax2 = axes[1]
    errors = preds - labels
    ax2.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('预测误差')
    ax2.set_ylabel('频数')
    ax2.set_title(f'误差分布 (MAE={np.mean(np.abs(errors)):.4f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. 按真实值分段的误差
    ax3 = axes[2]
    bins = np.linspace(-1, 1, 11)
    bin_indices = np.digitize(labels, bins)
    bin_errors = []
    bin_centers = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(np.abs(errors[mask])))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    ax3.bar(bin_centers, bin_errors, width=0.15, color='green', alpha=0.7)
    ax3.set_xlabel('转向值区间')
    ax3.set_ylabel('平均绝对误差')
    ax3.set_title('不同转向值的误差')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"结果图已保存: {save_path}")


def main():
    print("=" * 60)
    print("模型测试脚本")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"模型: {MODEL_PATH}")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 train.py 训练模型！")
        return
    
    # 检查测试数据
    if not os.path.exists(TEST_CSV):
        print(f"❌ 测试数据不存在: {TEST_CSV}")
        print("请先运行 process_data.py 处理数据！")
        return
    
    # 加载数据
    print("\n加载测试数据...")
    test_dataset = CarlaTestDataset(TEST_CSV)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 加载模型
    print("\n加载模型...")
    model = PilotNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型来自 Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?'):.6f}")
    else:
        model.load_state_dict(checkpoint)
    
    criterion = nn.MSELoss()
    
    # 评估
    print("\n开始测试...")
    test_loss, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    
    # 计算指标
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    
    # 分段统计
    left_mask = labels < -0.05
    right_mask = labels > 0.05
    straight_mask = ~(left_mask | right_mask)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"测试样本数: {len(labels)}")
    print(f"MSE Loss: {test_loss:.6f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print()
    print("分段误差:")
    if np.sum(left_mask) > 0:
        print(f"  左转 (steer < -0.05): MAE = {np.mean(np.abs(preds[left_mask] - labels[left_mask])):.4f}, 样本数 = {np.sum(left_mask)}")
    if np.sum(straight_mask) > 0:
        print(f"  直行 (-0.05 ~ 0.05): MAE = {np.mean(np.abs(preds[straight_mask] - labels[straight_mask])):.4f}, 样本数 = {np.sum(straight_mask)}")
    if np.sum(right_mask) > 0:
        print(f"  右转 (steer > 0.05): MAE = {np.mean(np.abs(preds[right_mask] - labels[right_mask])):.4f}, 样本数 = {np.sum(right_mask)}")
    print("=" * 60)
    
    # 可视化
    print("\n生成结果图...")
    plot_results(preds, labels)


if __name__ == '__main__':
    main()

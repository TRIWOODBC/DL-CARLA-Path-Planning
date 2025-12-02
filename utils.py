"""
工具函数 - 图像处理、数据增强等公共函数
"""
import cv2
import numpy as np
import torch
import random

from config import (
    CROP_TOP, CROP_BOTTOM, INPUT_WIDTH, INPUT_HEIGHT
)


def preprocess_image(image, for_training=False):
    """
    图像预处理（用于训练和推理）
    
    Args:
        image: BGR 格式的原始图像 (H, W, 3)
        for_training: 是否为训练模式（启用数据增强）
    
    Returns:
        处理后的图像 tensor (1, 3, 66, 200)
    """
    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 裁剪（去掉天空和车头盖）
    image = image[CROP_TOP:-CROP_BOTTOM, :, :]
    
    # 缩放到模型输入尺寸
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # 归一化到 [0, 1]
    image = torch.from_numpy(image).float() / 255.0
    
    # 添加 batch 维度
    image = image.unsqueeze(0)
    
    return image


def augment_image(image, steer):
    """
    数据增强（仅用于训练）
    
    Args:
        image: RGB 格式图像 (H, W, 3)
        steer: 转向角标签
    
    Returns:
        增强后的图像和转向角
    """
    # 1. 随机水平翻转
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        steer = -steer
    
    # 2. 随机亮度调整
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    # 3. 随机对比度调整
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)
        mean = np.mean(image)
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    # 4. 随机添加阴影
    if random.random() > 0.5:
        image = add_random_shadow(image)
    
    return image, steer


def add_random_shadow(image):
    """
    添加随机阴影
    
    Args:
        image: RGB 图像
    
    Returns:
        添加阴影后的图像
    """
    h, w = image.shape[:2]
    
    # 随机生成阴影区域
    x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
    
    # 创建阴影遮罩
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    shadow_mask[:, x1:x2] = random.uniform(0.3, 0.7)
    
    # 应用阴影
    image = image.astype(np.float32)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - shadow_mask)
    
    return np.clip(image, 0, 255).astype(np.uint8)


def smooth_steering(current_steer, prev_steer, alpha=0.3, max_delta=0.2):
    """
    转向平滑处理
    
    Args:
        current_steer: 当前预测转向
        prev_steer: 上一帧转向
        alpha: 平滑系数 (0=无平滑, 1=完全保持上帧)
        max_delta: 单帧最大变化
    
    Returns:
        平滑后的转向值
    """
    # 指数平滑
    smoothed = alpha * prev_steer + (1 - alpha) * current_steer
    
    # 限制变化速率
    delta = smoothed - prev_steer
    if abs(delta) > max_delta:
        smoothed = prev_steer + max_delta * np.sign(delta)
    
    return float(np.clip(smoothed, -1.0, 1.0))


def to_kmh(velocity):
    """
    将 CARLA 速度向量转换为 km/h
    
    Args:
        velocity: carla.Vector3D 速度向量
    
    Returns:
        速度（km/h）
    """
    return 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5


def get_speed(velocity):
    """
    计算速度（m/s）
    
    Args:
        velocity: carla.Vector3D 速度向量
    
    Returns:
        速度（m/s）
    """
    import math
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

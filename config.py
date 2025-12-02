"""
配置文件 - 统一管理所有参数
"""
import os
import torch

# ==================== 路径配置 ====================
# CARLA 路径
CARLA_EGG_PATH = r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg'

# 数据目录
DATA_DIR = "data_more"
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
VALIDATION_CSV = os.path.join(DATA_DIR, "validation_labels.csv")

# 模型保存路径
MODEL_PATH = "best_model.pth"

# ==================== CARLA 配置 ====================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 采集配置
TOWNS = ['Town01', 'Town03', 'Town05', 'Town07']
NPC_COUNTS = [30, 80, 140]  # 稀疏/中/拥挤
COLLECTION_FPS = 10
MIN_MOVE_M = 0.3  # 两帧累计位移<此阈值则跳过保存
DURATION_PER_SCENARIO_SEC = 180  # 每个场景跑几分钟

# 相机配置（采集和推理保持一致！）
CAMERA_X = 0.5  # 相机前后位置
CAMERA_Z = 1.6  # 相机高度
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90

# ==================== 图像预处理配置 ====================
# 裁剪参数（去掉天空和车头盖）
CROP_TOP = 60
CROP_BOTTOM = 25

# 模型输入尺寸
INPUT_WIDTH = 200
INPUT_HEIGHT = 66

# ==================== 训练配置 ====================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5  # L2 正则化
NUM_WORKERS = 4

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据处理配置 ====================
STEER_THRESHOLD = 0.02  # 小于此值视为直行
STRAIGHT_SAMPLE_RATIO = 0.1  # 直行数据保留比例
VALIDATION_SPLIT = 0.2  # 验证集比例

# ==================== 驾驶控制配置 ====================
# 油门控制
BASE_THROTTLE = 0.5
MIN_THROTTLE = 0.25

# 转向控制
STEER_GAIN = 1.5  # 转向增益（不要太大！）
STEER_SMOOTH_ALPHA = 0.3  # 平滑系数 (0=无平滑, 1=完全保持上一帧)
MAX_STEER_DELTA = 0.2  # 单帧最大转向变化
STEER_DEADZONE = 0.01  # 死区

# 油门与转向关联
THROTTLE_STEER_SCALE = 0.5  # 转向时降低油门的系数

# ==================== 天气预设 ====================
# 在 collect_data.py 中使用
WEATHER_PRESETS = [
    "ClearNoon",
    "CloudySunset",
    "WetNoon",
    "MidRainSunset",
    "SoftRainSunset",
]

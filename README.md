# DL-CARLA 自动驾驶路径规划

基于 CARLA 0.9.10.1 的端到端自动驾驶毕业设计项目，使用 PilotNet CNN 从单目图像直接预测转向角，实现行为克隆式驾驶控制。

## 1. 快速开始

1) 准备环境（推荐 Conda）：

```bash
conda env create -f environment.yml
conda activate carla_env
```

2) 安装 CARLA 0.9.10.1，并在 [config.py](config.py) 中将 `CARLA_EGG_PATH` 指向本机 `carla-0.9.10-py3.7-win-amd64.egg` 路径。

3) 启动 CARLA 服务器（默认 2000 端口）：

```bash
E:\CARLA_0.9.10.1\WindowsNoEditor\CarlaUE4.exe
```

4) 运行流程：
- 采集数据：`python collect_data.py`
- 训练模型：`python train.py`
- 测试评估：`python test.py`
- 自动驾驶：`python drive.py`

## 1.1 CARLA 下载与配置

- 下载地址（CARLA 0.9.10.1）：https://github.com/carla-simulator/carla/releases/tag/0.9.10.1
- Windows 选择 `CARLA_0.9.10.1.zip`，解压后路径类似 `E:\CARLA_0.9.10.1\WindowsNoEditor`
- Python API egg 在 `WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg`
- 将 `CARLA_EGG_PATH` 写入 [config.py](config.py)，并确保本机 Python 版本为 3.7

## 2. 目录与数据

```
.
├── config.py           # 全局配置（路径、相机、训练、控制参数）
├── collect_data.py     # 多地图/多天气数据采集脚本
├── train.py            # 数据集加载、训练与最优模型保存
├── test.py             # 验证/测试评估与可视化
├── drive.py            # 推理与 CARLA 内驾驶控制
├── model.py            # PilotNet 模型定义
├── utils.py            # 图像预处理、速度换算等工具
├── best_model.pth      # 训练好的模型权重（如已存在）
├── data_merged_all/    # 默认数据根目录
│   ├── train_labels.csv
│   ├── validation_labels.csv
│   ├── test_labels.csv (如有)
│   └── images_xxx/     # 采集的图像子目录
└── environment.yml     # Conda 环境文件
```

数据 CSV 至少需要列：`filename, steer, throttle, brake, speed_kmh`。默认路径与文件名在 [config.py](config.py) 中定义。

## 3. 配置要点

- 路径与数据集：`DATA_DIR`、`TRAIN_CSV`、`VALIDATION_CSV`、`MODEL_PATH`
- 相机与输入：`CAMERA_WIDTH`、`CAMERA_HEIGHT`、`CROP_TOP`、`CROP_BOTTOM`、`INPUT_WIDTH`、`INPUT_HEIGHT`
- 训练参数：`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`NUM_WORKERS`、`DEVICE`
- 采集参数：`TOWNS`、`NPC_COUNTS`、`COLLECTION_FPS`、`MIN_MOVE_M`、`DURATION_PER_SCENARIO_SEC`、`WEATHER_PRESETS`
- 驾驶控制：`BASE_THROTTLE`、`STEER_GAIN`、`THROTTLE_STEER_SCALE` 等

所有参数集中于 [config.py](config.py)。根据 GPU 情况，`DEVICE` 会自动选择 CUDA/CPU。

## 4. 训练与评估

- 训练：`python train.py`
	- 自动保存验证集最优模型到 `best_model.pth`
	- 输出 `loss_curve.png` 供可视化

- 测试：`python test.py`
	- 读取 `MODEL_PATH`（默认 `best_model.pth`）
	- 输出指标并可生成可视化结果

## 5. 自动驾驶 (drive.py)

- 加载 `best_model.pth` 并在 CARLA 中驾驶
- 支持模型预测 + 几何控制融合、速度 PI 控制、转向平滑与限幅
- 启动前确保 CARLA 服务器已运行，且 `MODEL_PATH` 指向可用权重

## 6. 注意事项

- CARLA Python API 绑定：保持 Python 3.7，并确保 `CARLA_EGG_PATH` 正确
- 显存/内存不足：可调低 `BATCH_SIZE`、`NUM_WORKERS`，或改用 CPU
- 数据裁剪与尺寸需与训练一致，否则推理结果会偏移

## 7. 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

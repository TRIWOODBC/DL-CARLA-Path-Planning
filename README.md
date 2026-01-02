# DL-CARLA 自动驾驶路径规划

基于 CARLA 0.9.10.1 的端到端自动驾驶毕业设计项目，使用 PilotNet CNN 从单目图像直接预测转向角，实现行为克隆式驾驶控制。

## English Version (Full)

**DL-CARLA End-to-End Driving (CARLA 0.9.10.1, PilotNet)** — Train a PilotNet CNN to predict steering directly from monocular RGB images and drive in CARLA. This section mirrors the Chinese content with full English instructions.

### 1. Quick Start

1) Environment (Conda):

```bash
conda env create -f environment.yml
conda activate carla_env
```

2) CARLA: Download 0.9.10.1 from https://github.com/carla-simulator/carla/releases/tag/0.9.10.1 (Windows: `CARLA_0.9.10.1.zip`), extract to a path like `E:\CARLA_0.9.10.1\WindowsNoEditor`.

3) Python API egg: located at `WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg`. Put this path into `CARLA_EGG_PATH` in [config.py](config.py) (Python 3.7 required).

4) Start CARLA server (default 2000): run `CarlaUE4.exe` from `WindowsNoEditor`.

5) Pipeline:
- Collect data: `python collect_data.py`
- Train: `python train.py`
- Test/evaluate: `python test.py`
- Drive in CARLA: `python drive.py`

### 1.1 End-to-End Flow (zero to driving)

1) Create env: `conda env create -f environment.yml && conda activate carla_env`
2) Configure CARLA: set `CARLA_EGG_PATH` in [config.py](config.py) to the 0.9.10.1 egg path
3) Launch CARLA server: run `CarlaUE4.exe`
4) Data collection: `python collect_data.py` -> generates `images_xxx/` and `labels_xxx.csv` under [data_merged_all](data_merged_all)
5) Prepare datasets:
	 - Merge valid samples into `train_labels.csv` and `validation_labels.csv` (default under [data_merged_all](data_merged_all))
	 - CSV needs columns at least: `filename, steer, throttle, brake, speed_kmh`
6) Train: `python train.py` -> best weights saved to `best_model.pth`, `loss_curve.png` plotted
7) Test/validate: `python test.py` (reads `MODEL_PATH`, default `best_model.pth`)
8) Drive: `python drive.py` (CARLA server running, `MODEL_PATH` pointing to weights)
9) Tune and repeat: adjust crops/resolution/control gains in [config.py](config.py), then repeat steps 4-8.

### 2. Structure and Data

```
.
├── config.py           # Global config (paths, camera, training, control)
├── collect_data.py     # Multi-town/weather data collection
├── train.py            # Dataset loading, training, best-model saving
├── test.py             # Validation/testing and visualization
├── drive.py            # Inference and CARLA driving control
├── model.py            # PilotNet definition
├── utils.py            # Preprocess, speed conversion, helpers
├── best_model.pth      # Trained weights (if present)
├── data_merged_all/    # Default data root
│   ├── train_labels.csv
│   ├── validation_labels.csv
│   ├── test_labels.csv (optional)
│   └── images_xxx/
└── environment.yml     # Conda environment
```

CSV columns needed: `filename, steer, throttle, brake, speed_kmh`. Paths and filenames are defined in [config.py](config.py).

### 3. Key Configuration

- Paths/data: `DATA_DIR`, `TRAIN_CSV`, `VALIDATION_CSV`, `MODEL_PATH`
- Camera/input: `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CROP_TOP`, `CROP_BOTTOM`, `INPUT_WIDTH`, `INPUT_HEIGHT`
- Training: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `NUM_WORKERS`, `DEVICE`
- Collection: `TOWNS`, `NPC_COUNTS`, `COLLECTION_FPS`, `MIN_MOVE_M`, `DURATION_PER_SCENARIO_SEC`, `WEATHER_PRESETS`
- Driving control: `BASE_THROTTLE`, `STEER_GAIN`, `THROTTLE_STEER_SCALE`, etc.

All configurable in [config.py](config.py). `DEVICE` auto-selects CUDA/CPU if available.

### 4. Training and Evaluation

- Train: `python train.py`
	- Saves best validation model to `best_model.pth`
	- Plots `loss_curve.png`

- Test: `python test.py`
	- Loads `MODEL_PATH` (default `best_model.pth`)
	- Outputs metrics and optional visuals

### 5. Driving (drive.py)

- Loads `best_model.pth` and drives in CARLA
- Blends model prediction with geometric control, PI speed control, smoothing and rate limits
- Ensure CARLA server is running and `MODEL_PATH` points to valid weights

### 6. Notes

- CARLA Python API: stick to Python 3.7 and set `CARLA_EGG_PATH` correctly
- OOM or memory issues: lower `BATCH_SIZE`, `NUM_WORKERS`, or force CPU
- Keep preprocessing consistent: crops and input size must match training

### 7. License

MIT License — see [LICENSE](LICENSE)

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

## 1.2 端到端流程（从零到自动驾驶）

1) 创建环境：`conda env create -f environment.yml && conda activate carla_env`
2) 配置 CARLA：下载 0.9.10.1，解压后把 egg 路径填入 [config.py](config.py) 的 `CARLA_EGG_PATH`
3) 启动 CARLA 服务器：运行 `CarlaUE4.exe`（默认 2000 端口）
4) 采集数据：`python collect_data.py` 会在 [data_merged_all](data_merged_all) 下生成 `images_xxx/` 与 `labels_xxx.csv`
5) 准备训练集：
	- 将有效样本合并成 `train_labels.csv`、`validation_labels.csv`（默认已在 [data_merged_all](data_merged_all)）
	- CSV 需包含至少 `filename, steer, throttle, brake, speed_kmh`
6) 训练：`python train.py`，保存最优权重到 `best_model.pth`，输出 `loss_curve.png`
7) 测试/验证：`python test.py`（会读取 `MODEL_PATH`，默认 `best_model.pth`）
8) 自动驾驶：`python drive.py`（确保 CARLA 服务器已运行，`MODEL_PATH` 指向可用权重）
9) 若需微调：调整 [config.py](config.py) 中的相机裁剪、输入分辨率、控制增益等，再重复 4-8 步。

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

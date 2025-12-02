# -*- coding: utf-8 -*-
import sys, os, time, math, random, queue
import numpy as np
import cv2
import torch
import torch.nn as nn

# ========= 配置区 =========
CARLA_EGG_PATH = r'D:\\CARLA_0.9.10.1\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.10-py3.7-win-amd64.egg'
WORLD_NAME = 'Town03'
USE_LOAD_WORLD = False
USE_SYNC = False
FIXED_DELTA = 0.05

CAM_RES_X, CAM_RES_Y = 1280, 720
MODEL_PATH = 'best_model.pth'

# 模型标签单位：'norm'（[-1,1]）、'deg'（度）、'rad'（弧度）
LABEL_UNITS = 'norm'
MAX_STEER_DEG = 35.0

# 控制参数
BASE_THROTTLE = 0.40
MIN_THROTTLE = 0.22
MAX_THROTTLE = 0.60
THROTTLE_STEER_SCALE = 0.6

TARGET_SPD = 4.8    # m/s
KP_SPD = 0.22       # 速度P控制

# ====== 模型主导融合 & 稳定化 ======
STEER_GAIN = 1.0         # 重要：对 norm 标签用 ~1.0，防止偏置被放大
MODEL_EMA_ALPHA = 0.35   # 模型转向 EMA 平滑（0~1）

# 动态融合参数（模型主导）
BLEND_BETA_MIN = 0.60    # 直路/不确定时模型最小占比
BLEND_BETA_MAX = 0.85    # 弯道/信息丰富时模型最大占比
CURVE_SENS = 1.6         # 根据 |steer_geo| 提升模型占比的敏感度（越大越快靠近 MAX）

# 速率限制，避免瞬时翻向
MAX_DSTEER = 0.05        # 每步最大方向变化

# Pure Pursuit/Stanley 参数
WHEEL_BASE = 2.8
STEER_MAX_RAD = math.radians(35.0)
LOOKAHEAD_MIN = 4.0
LOOKAHEAD_MAX = 10.0
LOOKAHEAD_GAIN = 0.85     # 稍增，几何更平滑
K_STANLEY = 0.6           # 横误差增益，略保守，减少撞限

# 其他
ANTI_STUCK = True
SHOW_WINDOW = False
WIGGLE_STEPS = 0
# =========================

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 加入 CARLA egg
if CARLA_EGG_PATH not in sys.path:
    sys.path.append(CARLA_EGG_PATH)
import carla  # noqa


# ---------- 模型 ----------
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
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


# ---------- 预处理 ----------
def preprocess_image(image_bgr):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = img[60:-25, :, :]
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float() / 255.0
    return img.unsqueeze(0)  # (1,3,66,200)


def map_pred_to_steer(raw):
    # 将原始 raw（可能未限幅）映射到控制域
    # 先做 tanh 限幅，避免异常值
    raw = math.tanh(raw)
    if LABEL_UNITS == 'norm':
        steer_norm = raw
    elif LABEL_UNITS == 'deg':
        steer_norm = raw / MAX_STEER_DEG
    elif LABEL_UNITS == 'rad':
        steer_norm = raw / math.radians(MAX_STEER_DEG)
    else:
        steer_norm = raw
    return float(np.clip(steer_norm * STEER_GAIN, -1.0, 1.0))


def speed_of(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


# ---------- 坐标/几何工具 ----------
def world_to_vehicle(vec_world, veh_transform):
    yaw = math.radians(veh_transform.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = vec_world.x - veh_transform.location.x
    dy = vec_world.y - veh_transform.location.y
    x_body = c * dx - s * dy
    y_body = s * dx + c * dy
    return x_body, y_body


def stanley_pure_pursuit_steer(world, vehicle, lookahead_m):
    car_tf = vehicle.get_transform()
    car_loc = car_tf.location

    wp_now = world.get_map().get_waypoint(car_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp_now is None:
        return 0.0

    wps = wp_now.next(lookahead_m)
    wp_target = wps[0] if wps else wp_now

    x_t, y_t = world_to_vehicle(wp_target.transform.location, car_tf)
    alpha = math.atan2(y_t, max(1e-3, x_t))

    x_c, y_c = world_to_vehicle(wp_now.transform.location, car_tf)
    e_y = y_c

    delta_pp = math.atan2(2.0 * WHEEL_BASE * math.sin(alpha) / max(1.0, lookahead_m), 1.0)

    delta_st = alpha + math.atan2(K_STANLEY * e_y, max(0.1, speed_of(vehicle)))

    delta = 0.6 * delta_pp + 0.4 * delta_st

    steer_norm = float(np.clip(delta / STEER_MAX_RAD, -1.0, 1.0))
    return steer_norm


# ========== 自标定：运行期零点偏置 ==========
CALIB_FRAMES = 90     # 标定帧数（起步直行阶段），可调 60~150
raw_accum = 0.0
raw_n = 0
raw_bias = 0.0        # 动态估计到的零点偏置


def main():
    ego_vehicle, camera, synchronous_enabled = None, None, False
    try:
        print('1) connecting...', flush=True)
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.load_world(WORLD_NAME) if USE_LOAD_WORLD else client.get_world()
        print('2) world ready:', world.get_map().name, '(async)' if not USE_SYNC else '(sync)', flush=True)

        if USE_SYNC:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DELTA
            world.apply_settings(settings)
            synchronous_enabled = True

        # 模型
        print('3) loading model...', flush=True)
        print('   MODEL_PATH =', os.path.abspath(MODEL_PATH), flush=True)
        if not os.path.exists(MODEL_PATH):
            print('   [ERROR] best_model.pth 不存在！', flush=True)
            return
        device = torch.device('cpu')
        model = PilotNet().to(device)
        state = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        print('   model ready on CPU.', flush=True)

        # 车辆
        bp_lib = world.get_blueprint_library()
        cand = bp_lib.filter('vehicle.tesla.model3')
        veh_bp = random.choice(cand) if len(cand) else bp_lib.filter('vehicle.*')[0]
        spawn = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(veh_bp, spawn)

        # 相机
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAM_RES_X))
        cam_bp.set_attribute('image_size_y', str(CAM_RES_Y))
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', '0.0' if USE_SYNC else str(FIXED_DELTA))
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=ego_vehicle
        )

        # 观众视角（可选）
        try:
            spectator = world.get_spectator()
            tf = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(x=-15, z=3),
                carla.Rotation(pitch=-10, yaw=tf.rotation.yaw)
            ))
        except Exception:
            pass

        # 图像队列
        img_q = queue.Queue()
        camera.listen(img_q.put)
        print('4) actors spawned. starting loop...', flush=True)

        # 起步助推（直线起步）
        for _ in range(15):
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))
            if USE_SYNC: world.tick()
            else: time.sleep(FIXED_DELTA)

        prev_time = time.time()
        step = 0
        same_sign_count = 0
        prev_sign = 0
        prev_steer = 0.0

        # 模型转向 EMA 状态
        steer_model_ema = 0.0

        global raw_accum, raw_n, raw_bias

        while True:
            if USE_SYNC:
                world.tick()

            try:
                carla_img = img_q.get(timeout=1.0)
            except queue.Empty:
                print('no frame', flush=True)
                continue

            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            frame_bgr = arr.reshape((carla_img.height, carla_img.width, 4))[:, :, :3]

            if SHOW_WINDOW:
                cv2.imshow('Ego Camera', frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ========= 模型推理 =========
            with torch.no_grad():
                inp = preprocess_image(frame_bgr).to(device)
                raw_val = float(model(inp).item())

            # 自标定：累计前 CALIB_FRAMES 帧的均值作为零点偏置
            if raw_n < CALIB_FRAMES:
                raw_accum += raw_val
                raw_n += 1
                raw_bias = raw_accum / max(1, raw_n)

            raw = raw_val - raw_bias
            steer_model = map_pred_to_steer(raw)

            # 轻度 EMA 平滑
            steer_model_ema = (1.0 - MODEL_EMA_ALPHA) * steer_model_ema + MODEL_EMA_ALPHA * steer_model
            steer_model = float(np.clip(steer_model_ema, -1.0, 1.0))

            # ========= 几何转向 =========
            v = speed_of(ego_vehicle)
            Ld = np.clip(v * LOOKAHEAD_GAIN + LOOKAHEAD_MIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
            steer_geo = stanley_pure_pursuit_steer(world, ego_vehicle, float(Ld))

            # ========= 动态融合（模型主导） =========
            # 用 |steer_geo| 作为曲率 proxy：越弯，越提升模型占比
            curve = abs(steer_geo)
            beta = BLEND_BETA_MIN + (BLEND_BETA_MAX - BLEND_BETA_MIN) * (1.0 - math.exp(-CURVE_SENS * curve))
            beta = float(np.clip(beta, BLEND_BETA_MIN, BLEND_BETA_MAX))

            steer = (1.0 - beta) * steer_geo + beta * steer_model

            # 速率限制（避免瞬时翻向）
            steer = float(np.clip(steer, prev_steer - MAX_DSTEER, prev_steer + MAX_DSTEER))
            steer = float(np.clip(steer, -1.0, 1.0))
            prev_steer = steer

            # ========= 速度控制 =========
            throttle_base = BASE_THROTTLE + KP_SPD * (TARGET_SPD - v)
            throttle_base = float(np.clip(throttle_base, MIN_THROTTLE, MAX_THROTTLE))
            throttle = throttle_base * (1.0 - THROTTLE_STEER_SCALE * abs(steer))
            throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

            # 超速轻刹
            brake = 0.0
            if v > TARGET_SPD + 0.8:
                brake = min(0.25, 0.06 * (v - TARGET_SPD))

            # 反卡死
            if ANTI_STUCK and v < 0.2 and step > 50 and brake == 0.0:
                throttle = max(throttle, 0.35)

            # 应用控制
            ctrl = carla.VehicleControl()
            ctrl.steer = steer
            ctrl.throttle = 0.0 if brake > 0 else throttle
            ctrl.brake = brake
            ctrl.hand_brake = False
            ctrl.reverse = False
            ego_vehicle.apply_control(ctrl)

            # 打印
            now = time.time()
            hz = 1.0 / max(1e-3, (now - prev_time))
            prev_time = now
            print(
                f'raw:{raw_val:+.6f} raw_bias:{raw_bias:+.4f} raw_corr:{raw:+.4f} '
                f'steer_m:{steer_model:+.3f} steer_geo:{steer_geo:+.3f} beta:{beta:.2f} '
                f'steer:{steer:+.3f} thr:{throttle:.2f} brk:{brake:.2f} v:{v:.2f} '
                f'Ld:{Ld:.1f} hz:{hz:.1f}', flush=True
            )

            step += 1
            if not USE_SYNC:
                time.sleep(FIXED_DELTA)

    except KeyboardInterrupt:
        print('\n[CTRL+C] stopping...', flush=True)
    finally:
        print('cleaning up...', flush=True)
        try:
            if camera:
                camera.stop()
                camera.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            if synchronous_enabled:
                s = world.get_settings()
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                world.apply_settings(s)
        except Exception as e:
            print('cleanup error:', e, flush=True)
        cv2.destroyAllWindows()
        print('done.', flush=True)


if __name__ == '__main__':
    main()

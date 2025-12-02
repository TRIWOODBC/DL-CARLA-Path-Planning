# -*- coding: utf-8 -*-
import sys, os, time, math, random, queue
import numpy as np
import cv2
import torch
import torch.nn as nn

# ========= 配置区 =========
CARLA_EGG_PATH = r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg'
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

TARGET_SPD = 4.8    # m/s，先保守，稳定后可 5.5~6.0
KP_SPD = 0.22       # 速度P控制

# 转向融合
STEER_GAIN = 10.0   # 模型增益（若 LABEL_UNITS='rad' 可用 8~10）
BLEND_BETA = 0.55   # β：模型占比，Stanley 70% + 模型 30%

# Pure Pursuit/Stanley 参数
WHEEL_BASE = 2.8               # 近似特斯拉Model3轴距（米）
STEER_MAX_RAD = math.radians(35.0)  # 最大方向角用于归一化
LOOKAHEAD_MIN = 4.0
LOOKAHEAD_MAX = 10.0
LOOKAHEAD_GAIN = 0.8           # Ld = min(max, max(min, v*GAIN+MIN))

# 其他
ANTI_STUCK = True
SHOW_WINDOW = False            # 若GUI卡顿就 False
WIGGLE_STEPS = 0               # 现在不摆头，避免起步就蹭边
# =========================

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 先强制CPU，避免CUDA初始化卡住

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
    # 把世界坐标的向量投到车体坐标系（前为+X，左为+Y）
    yaw = math.radians(veh_transform.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = vec_world.x - veh_transform.location.x
    dy = vec_world.y - veh_transform.location.y
    # 旋转到车体坐标
    x_body = c * dx - s * dy
    y_body = s * dx + c * dy
    return x_body, y_body


def stanley_pure_pursuit_steer(world, vehicle, lookahead_m):
    """
    计算面向 lookahead_m 处车道中心点的转向（归一化到 [-1,1]）
    组合了 Pure Pursuit（目标点角度） + 轻度Stanley（横向误差）
    """
    car_tf = vehicle.get_transform()
    car_loc = car_tf.location
    car_yaw = math.radians(car_tf.rotation.yaw)

    # 最近车道中心（投影到道路）
    wp_now = world.get_map().get_waypoint(car_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp_now is None:
        return 0.0

    # 前方 lookahead_m 的参考点
    wps = wp_now.next(lookahead_m)
    wp_target = wps[0] if wps else wp_now

    # 目标点在车体坐标系的位置
    x_t, y_t = world_to_vehicle(wp_target.transform.location, car_tf)
    # 目标朝向与车辆朝向的误差（Pure Pursuit）
    alpha = math.atan2(y_t, max(1e-3, x_t))  # 车体系下到目标点的方位角
    # 横向误差（Stanley）
    x_c, y_c = world_to_vehicle(wp_now.transform.location, car_tf)
    e_y = y_c  # 当前中心线点到车体的横向误差（左为正）

    # Pure Pursuit 转角（轮角）
    # delta_pp = atan2(2*L*sin(alpha)/Ld, 1)
    delta_pp = math.atan2(2.0 * WHEEL_BASE * math.sin(alpha) / max(1.0, lookahead_m), 1.0)

    # Stanley 修正：heading误差用 alpha 代替，横向误差项 e_y
    k_st = 0.8
    delta_st = alpha + math.atan2(k_st * e_y, max(0.1, speed_of(vehicle)))

    # 融合（权重可再调）
    delta = 0.6 * delta_pp + 0.4 * delta_st

    # 归一化到 [-1,1] 控制
    steer_norm = float(np.clip(delta / STEER_MAX_RAD, -1.0, 1.0))
    return steer_norm


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

            # 模型推理
            with torch.no_grad():
                inp = preprocess_image(frame_bgr).to(device)
                raw = float(model(inp).item())
            steer_model = map_pred_to_steer(raw)

            # 几何转向（主控）
            v = speed_of(ego_vehicle)
            Ld = np.clip(v * LOOKAHEAD_GAIN + LOOKAHEAD_MIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
            steer_geo = stanley_pure_pursuit_steer(world, ego_vehicle, float(Ld))

            # 融合
            steer = (1.0 - BLEND_BETA) * steer_geo + BLEND_BETA * steer_model
            steer = float(np.clip(steer, -1.0, 1.0))

            # 防自旋：若持续同向大打角且车速上不来，强行压小
            sign = 1 if steer > 0 else (-1 if steer < 0 else 0)
            if sign != 0 and sign == prev_sign:
                same_sign_count += 1
            else:
                same_sign_count = 0
            prev_sign = sign

            if same_sign_count > 20 and abs(steer) > 0.20 and v < 1.2:
                steer *= 0.5  # 降半
            if same_sign_count > 40 and abs(steer) > 0.25 and v < 0.8:
                steer *= 0.35 # 更狠一点

            # 速度控制 + 弯中降油
            throttle_base = BASE_THROTTLE + KP_SPD * (TARGET_SPD - v)
            throttle_base = float(np.clip(throttle_base, MIN_THROTTLE, MAX_THROTTLE))
            throttle = throttle_base * (1.0 - THROTTLE_STEER_SCALE * abs(steer))
            throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

            # 超速轻刹
            brake = 0.0
            if v > TARGET_SPD + 0.8:
                brake = min(0.25, 0.06 * (v - TARGET_SPD))

            # 反卡死（可用可不用：几乎不动时小幅脉冲）
            # 但在有Stanley主控后通常不再需要激进倒车，这里只留轻微处理
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
            print(f'raw:{raw:.6f} steer_m:{steer_model:+.3f} steer_geo:{steer_geo:+.3f} '
                  f'steer:{steer:+.3f} thr:{throttle:.2f} brk:{brake:.2f} v:{v:.2f} '
                  f'Ld:{Ld:.1f} same:{same_sign_count} hz:{hz:.1f}', flush=True)

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

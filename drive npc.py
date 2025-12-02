# -*- coding: utf-8 -*-
import os, sys, time, math, random, queue
import numpy as np
import cv2
import torch
import torch.nn as nn

# =============== 基本配置 ===============
CARLA_EGG_PATH = r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg'
WORLD_NAME = 'Town03'
USE_LOAD_WORLD = False

# 性能/稳定性
USE_SYNC = True            # 同步模式更稳
FIXED_DELTA = 0.08         # 12.5Hz，想更顺可设 0.10
CAM_RES_X, CAM_RES_Y = 640, 360   # 降分辨率减负
SHOW_WINDOW = False        # False 更省资源
SEED = 42

# NPC / Traffic Manager
ENABLE_NPC = True
NPC_NUM = 8
NPC_MIN_DIST_TO_EGO = 20.0
TM_PORT = 8000
TM_GLOBAL_GAP = 2.5
TM_SPEED_DIFF = -10.0      # 负值=更快，正值=更慢

# 模型与控制参数
MODEL_PATH = 'best_model.pth'
LABEL_UNITS = 'norm'       # 'norm'([-1,1]) / 'deg' / 'rad'
MAX_STEER_DEG = 35.0
STEER_GAIN = 10.0          # 若 LABEL_UNITS='rad'，8~10 更合适
BLEND_BETA = 0.30          # 融合权重：steer = (1-β)*几何 + β*模型

# 几何控制（Stanley + PurePursuit）
WHEEL_BASE = 2.8
STEER_MAX_RAD = math.radians(35.0)
LOOKAHEAD_MIN = 4.0
LOOKAHEAD_MAX = 10.0
LOOKAHEAD_GAIN = 0.8

# 纵向控制
BASE_THROTTLE = 0.40
MIN_THROTTLE = 0.22
MAX_THROTTLE = 0.60
THROTTLE_STEER_SCALE = 0.6
TARGET_SPD = 4.8
KP_SPD = 0.22

# 保护
ANTI_STUCK = True

# =============== 设备与CARLA导入 ===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[device]', device)

if CARLA_EGG_PATH not in sys.path:
    sys.path.append(CARLA_EGG_PATH)
import carla  # noqa


# =============== 模型定义 ===============
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


# =============== 工具函数 ===============
def preprocess_image(image_bgr):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = img[60:-25, :, :]
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float() / 255.0
    return img.unsqueeze(0)

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
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def world_to_vehicle(vec_world, veh_tf):
    yaw = math.radians(veh_tf.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = vec_world.x - veh_tf.location.x
    dy = vec_world.y - veh_tf.location.y
    x_body = c*dx - s*dy
    y_body = s*dx + c*dy
    return x_body, y_body

def stanley_pure_pursuit_steer(world, vehicle, lookahead_m):
    car_tf = vehicle.get_transform()
    car_loc = car_tf.location
    wp_now = world.get_map().get_waypoint(
        car_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp_now is None:
        return 0.0
    wps = wp_now.next(lookahead_m)
    wp_target = wps[0] if wps else wp_now
    x_t, y_t = world_to_vehicle(wp_target.transform.location, car_tf)
    alpha = math.atan2(y_t, max(1e-3, x_t))
    x_c, y_c = world_to_vehicle(wp_now.transform.location, car_tf)
    e_y = y_c
    delta_pp = math.atan2(2.0 * WHEEL_BASE * math.sin(alpha) / max(1.0, lookahead_m), 1.0)
    k_st = 0.8
    v = speed_of(vehicle)
    delta_st = alpha + math.atan2(k_st * e_y, max(0.1, v))
    delta = 0.6 * delta_pp + 0.4 * delta_st
    steer_norm = float(np.clip(delta / STEER_MAX_RAD, -1.0, 1.0))
    return steer_norm

# —— 追尾镜头 & 顶部标注 ——
def follow_ego_spectator(world, vehicle, dist=12.0, height=3.0, pitch=-12.0):
    try:
        sp = world.get_spectator()
        tf = vehicle.get_transform()
        back_vec = carla.Location(
            -dist * math.cos(math.radians(tf.rotation.yaw)),
            -dist * math.sin(math.radians(tf.rotation.yaw)),
            height
        )
        sp.set_transform(carla.Transform(
            tf.location + back_vec,
            carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw)
        ))
    except Exception:
        pass

def mark_ego(world, vehicle, text='EGO', life_time=0.5):
    try:
        loc = vehicle.get_location() + carla.Location(z=2.4)
        world.debug.draw_string(
            loc, text, draw_shadow=False,
            color=carla.Color(r=255, g=255, b=0),
            life_time=life_time, persistent_lines=False
        )
    except Exception:
        pass


# =============== NPC 相关 ===============
def spawn_npc(world, client, ego_vehicle, npc_num=8, tm_port=8000, min_dist=20.0):
    tm = client.get_trafficmanager(tm_port)
    tm.set_global_distance_to_leading_vehicle(TM_GLOBAL_GAP)
    tm.set_synchronous_mode(USE_SYNC)
    tm.global_percentage_speed_difference(TM_SPEED_DIFF)
    try:
        tm.set_hybrid_physics_mode(True)
        tm.set_random_device_seed(SEED)
    except Exception:
        pass

    bp_lib = world.get_blueprint_library()
    v_bps = bp_lib.filter('vehicle.*')
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    npc_list = []
    ego_loc = ego_vehicle.get_transform().location

    def far_enough(sp):
        loc = sp.location
        dx, dy = (loc.x - ego_loc.x), (loc.y - ego_loc.y)
        return (dx*dx + dy*dy) >= (min_dist * min_dist)

    for sp in spawn_points:
        if len(npc_list) >= npc_num:
            break
        if not far_enough(sp):
            continue
        bp = random.choice(v_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        if bp.has_attribute('driver_id'):
            did = random.choice(bp.get_attribute('driver_id').recommended_values)
            bp.set_attribute('driver_id', did)
        bp.set_attribute('role_name', 'autopilot')

        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            npc_list.append(npc)

    print(f'[NPC] spawned {len(npc_list)} vehicles.', flush=True)
    return tm, npc_list


# =============== 主流程 ===============
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ego_vehicle = camera = None
    npc_list = []
    tm = None
    world = None

    try:
        print('1) connecting...', flush=True)
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.load_world(WORLD_NAME) if USE_LOAD_WORLD else client.get_world()
        print('2) world ready:', world.get_map().name, '(sync)' if USE_SYNC else '(async)', flush=True)

        # 天气简化（可选）
        try:
            world.set_weather(carla.WeatherParameters.ClearNoon)
        except Exception:
            pass

        # 同步设置
        if USE_SYNC:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DELTA
            world.apply_settings(settings)

        # 加载模型
        print('3) loading model...', flush=True)
        if not os.path.exists(MODEL_PATH):
            print('   [ERROR] best_model.pth 不存在 ->', os.path.abspath(MODEL_PATH), flush=True)
            return
        model = PilotNet().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print('   model ready on', device, flush=True)

        # Ego 车辆
        bp_lib = world.get_blueprint_library()
        cand = bp_lib.filter('vehicle.tesla.model3')
        veh_bp = random.choice(cand) if len(cand) else bp_lib.filter('vehicle.*')[0]
        spawn = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.try_spawn_actor(veh_bp, spawn)
        if ego_vehicle is None:
            for sp in world.get_map().get_spawn_points():
                ego_vehicle = world.try_spawn_actor(veh_bp, sp)
                if ego_vehicle: break
        if ego_vehicle is None:
            print('[ERROR] ego_vehicle spawn failed.', flush=True)
            return

        # 相机
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAM_RES_X))
        cam_bp.set_attribute('image_size_y', str(CAM_RES_Y))
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', '0.0' if USE_SYNC else str(FIXED_DELTA))
        try:
            cam_bp.set_attribute('enable_postprocess_effects', 'false')
        except Exception:
            pass

        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=ego_vehicle
        )

        # 初始追尾视角
        follow_ego_spectator(world, ego_vehicle)

        # 图像队列
        img_q = queue.Queue()
        camera.listen(img_q.put)

        # 生成 NPC
        if ENABLE_NPC:
            tm, npc_list = spawn_npc(world, client, ego_vehicle,
                                     npc_num=NPC_NUM, tm_port=TM_PORT, min_dist=NPC_MIN_DIST_TO_EGO)

        print('4) actors spawned. starting loop...', flush=True)

        # 起步助推
        for _ in range(12):
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))
            if USE_SYNC: world.tick()
            else: time.sleep(FIXED_DELTA)

        # 主循环
        prev_time = time.time()
        prev_sign, same_sign_count = 0, 0
        step = 0

        while True:
            if USE_SYNC:
                world.tick()

            try:
                carla_img = img_q.get(timeout=1.0)
            except queue.Empty:
                print('no frame')
                continue

            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            frame_bgr = arr.reshape((carla_img.height, carla_img.width, 4))[:, :, :3]

            if SHOW_WINDOW:
                cv2.imshow('Ego Camera', frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 模型推理
            with torch.no_grad():
                inp = preprocess_image(frame_bgr).to(device, non_blocking=True)
                raw = float(model(inp).item())
            steer_model = map_pred_to_steer(raw)

            # 几何主控
            v = speed_of(ego_vehicle)
            Ld = float(np.clip(v * LOOKAHEAD_GAIN + LOOKAHEAD_MIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX))
            steer_geo = stanley_pure_pursuit_steer(world, ego_vehicle, Ld)

            # 融合与保护
            steer = (1.0 - BLEND_BETA) * steer_geo + BLEND_BETA * steer_model
            steer = float(np.clip(steer, -1.0, 1.0))
            sign = 1 if steer > 0 else (-1 if steer < 0 else 0)
            if sign != 0 and sign == prev_sign:
                same_sign_count += 1
            else:
                same_sign_count = 0
            prev_sign = sign
            if same_sign_count > 20 and abs(steer) > 0.20 and v < 1.2:
                steer *= 0.5
            if same_sign_count > 40 and abs(steer) > 0.25 and v < 0.8:
                steer *= 0.35

            # 纵向控制
            throttle_base = BASE_THROTTLE + KP_SPD * (TARGET_SPD - v)
            throttle_base = float(np.clip(throttle_base, MIN_THROTTLE, MAX_THROTTLE))
            throttle = throttle_base * (1.0 - THROTTLE_STEER_SCALE * abs(steer))
            throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

            brake = 0.0
            if v > TARGET_SPD + 0.8:
                brake = min(0.25, 0.06 * (v - TARGET_SPD))
            if ANTI_STUCK and v < 0.2 and step > 50 and brake == 0.0:
                throttle = max(throttle, 0.35)

            # 控制应用
            ctrl = carla.VehicleControl()
            ctrl.steer = steer
            ctrl.throttle = 0.0 if brake > 0 else throttle
            ctrl.brake = brake
            ctrl.hand_brake = False
            ctrl.reverse = False
            ego_vehicle.apply_control(ctrl)

            # 追尾镜头 + 标注
            follow_ego_spectator(world, ego_vehicle)
            mark_ego(world, ego_vehicle)

            # 日志
            now = time.time()
            hz = 1.0 / max(1e-3, (now - prev_time))
            prev_time = now
            print(f'raw:{raw:+.6f} s_m:{steer_model:+.3f} s_geo:{steer_geo:+.3f} '
                  f's:{steer:+.3f} thr:{throttle:.2f} brk:{brake:.2f} v:{v:.2f} Ld:{Ld:.1f} hz:{hz:.1f}',
                  flush=True)

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
            for npc in npc_list:
                try:
                    npc.set_autopilot(False)
                except Exception:
                    pass
                npc.destroy()
            if ENABLE_NPC and tm:
                try:
                    tm.set_synchronous_mode(False)
                except Exception:
                    pass
            if world and USE_SYNC:
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

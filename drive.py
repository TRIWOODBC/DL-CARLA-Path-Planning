"""
自动驾驶脚本 V2 - Stanley/Pure Pursuit + 模型融合
参考更稳定的几何控制方法
"""
import sys
import os
import time
import math
import random
import queue
import numpy as np
import cv2
import torch

from config import (
    CARLA_EGG_PATH, CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    MODEL_PATH, DEVICE,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
    CROP_TOP, CROP_BOTTOM, INPUT_WIDTH, INPUT_HEIGHT,
)
from model import PilotNet

sys.path.append(CARLA_EGG_PATH)
import carla


# ========= 控制参数 =========
# 速度控制
TARGET_SPEED = 5.0      # 目标速度 m/s（约18km/h）
BASE_THROTTLE = 0.40
MIN_THROTTLE = 0.22
MAX_THROTTLE = 0.60
KP_SPEED = 0.20         # 速度P控制增益
THROTTLE_STEER_SCALE = 0.5  # 转弯时降油门

# 转向融合
STEER_GAIN = 1.5        # 模型输出增益
BLEND_BETA = 0.4        # 模型占比（0.4 = 模型40% + 几何60%）

# Pure Pursuit / Stanley 参数
WHEEL_BASE = 2.8        # 轴距（米）
STEER_MAX_RAD = math.radians(35.0)  # 最大转向角
LOOKAHEAD_MIN = 4.0     # 最小前瞻距离
LOOKAHEAD_MAX = 12.0    # 最大前瞻距离
LOOKAHEAD_GAIN = 0.8    # 前瞻距离 = 速度 * GAIN + MIN

# 其他
SHOW_WINDOW = True
# ============================


def speed_of(vehicle):
    """获取车辆速度 (m/s)"""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def world_to_vehicle(point_world, vehicle_transform):
    """世界坐标转车体坐标"""
    yaw = math.radians(vehicle_transform.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = point_world.x - vehicle_transform.location.x
    dy = point_world.y - vehicle_transform.location.y
    x_body = c * dx - s * dy
    y_body = s * dx + c * dy
    return x_body, y_body


class AutonomousDriverV2:
    """自动驾驶控制器 V2 - 几何+模型融合"""
    
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.model = None
        self.image_queue = queue.Queue()
        self.prev_steer = 0.0
        self.same_sign_count = 0
        self.prev_sign = 0
        
    def connect(self):
        """连接 CARLA"""
        print("连接 CARLA...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.get_world()
        print(f"已连接，当前地图: {self.world.get_map().name}")
        
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {MODEL_PATH}")
        self.model = PilotNet().to(DEVICE)
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"模型加载成功，设备: {DEVICE}")
        
    def spawn_vehicle(self):
        """生成车辆"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"车辆已生成: {self.vehicle.type_id}")
        
    def setup_camera(self):
        """设置摄像头"""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self.image_queue.put)
        print("摄像头已设置")
        
    def setup_spectator(self):
        """设置观众视角"""
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(x=-10, z=5),
            carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
        ))
        
    def preprocess_image(self, image_bgr):
        """预处理图像"""
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = img[CROP_TOP:-CROP_BOTTOM, :, :]
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0
        return img.unsqueeze(0)
    
    def predict_steering(self, image_bgr):
        """模型预测转向"""
        img_tensor = self.preprocess_image(image_bgr).to(DEVICE)
        
        with torch.no_grad():
            raw = self.model(img_tensor).item()
        
        # 应用增益并限制范围
        steer = float(np.clip(raw * STEER_GAIN, -1.0, 1.0))
        return raw, steer
    
    def stanley_pure_pursuit_steer(self, lookahead_m):
        """
        Stanley + Pure Pursuit 几何控制
        返回归一化转向 [-1, 1]
        """
        car_tf = self.vehicle.get_transform()
        car_loc = car_tf.location
        
        # 获取当前车道
        wp_now = self.world.get_map().get_waypoint(
            car_loc, project_to_road=True, 
            lane_type=carla.LaneType.Driving
        )
        if wp_now is None:
            return 0.0
        
        # 获取前方目标点
        wps = wp_now.next(lookahead_m)
        wp_target = wps[0] if wps else wp_now
        
        # 目标点在车体坐标系的位置
        x_t, y_t = world_to_vehicle(wp_target.transform.location, car_tf)
        
        # Pure Pursuit: 到目标点的方位角
        alpha = math.atan2(y_t, max(1e-3, x_t))
        
        # 横向误差（Stanley用）
        x_c, y_c = world_to_vehicle(wp_now.transform.location, car_tf)
        e_y = y_c  # 横向误差
        
        # Pure Pursuit 转角
        delta_pp = math.atan2(
            2.0 * WHEEL_BASE * math.sin(alpha) / max(1.0, lookahead_m), 
            1.0
        )
        
        # Stanley 修正
        k_stanley = 0.8
        v = speed_of(self.vehicle)
        delta_st = alpha + math.atan2(k_stanley * e_y, max(0.1, v))
        
        # 融合 Pure Pursuit (60%) + Stanley (40%)
        delta = 0.6 * delta_pp + 0.4 * delta_st
        
        # 归一化到 [-1, 1]
        steer_norm = float(np.clip(delta / STEER_MAX_RAD, -1.0, 1.0))
        return steer_norm
    
    def compute_control(self, steer, speed):
        """计算油门和刹车"""
        # 速度P控制
        throttle = BASE_THROTTLE + KP_SPEED * (TARGET_SPEED - speed)
        throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))
        
        # 转弯时降油门
        throttle *= (1.0 - THROTTLE_STEER_SCALE * abs(steer))
        throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))
        
        # 超速刹车
        brake = 0.0
        if speed > TARGET_SPEED + 1.0:
            brake = min(0.3, 0.08 * (speed - TARGET_SPEED))
        
        return throttle, brake
    
    def anti_spin_protection(self, steer, speed):
        """防自旋保护：持续同向大转角时减弱"""
        sign = 1 if steer > 0 else (-1 if steer < 0 else 0)
        
        if sign != 0 and sign == self.prev_sign:
            self.same_sign_count += 1
        else:
            self.same_sign_count = 0
        self.prev_sign = sign
        
        # 持续同向转动且车速很低时，减弱转向
        if self.same_sign_count > 25 and abs(steer) > 0.2 and speed < 1.5:
            steer *= 0.5
        if self.same_sign_count > 50 and abs(steer) > 0.25 and speed < 1.0:
            steer *= 0.3
            
        return steer
    
    def update_spectator(self):
        """更新观众视角"""
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        spectator.set_transform(carla.Transform(
            transform.location - forward * 10 + carla.Location(z=5),
            carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
        ))
    
    def run(self):
        """主循环"""
        print("\n" + "=" * 60)
        print("开始自动驾驶 V2！（Stanley + Pure Pursuit + 模型融合）")
        print("按 Q 或 Ctrl+C 退出")
        print("=" * 60 + "\n")
        
        # 起步助推
        print("起步中...")
        for _ in range(20):
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.5, steer=0.0, brake=0.0
            ))
            time.sleep(0.05)
        
        step = 0
        try:
            while True:
                # 获取图像
                try:
                    carla_image = self.image_queue.get(timeout=2.0)
                except queue.Empty:
                    print("摄像头超时")
                    continue
                
                # 转换图像
                img = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
                img = img.reshape((carla_image.height, carla_image.width, 4))[:, :, :3]
                
                # 显示图像
                if SHOW_WINDOW:
                    cv2.imshow('Camera', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 获取速度
                speed = speed_of(self.vehicle)
                
                # 计算前瞻距离
                lookahead = np.clip(
                    speed * LOOKAHEAD_GAIN + LOOKAHEAD_MIN,
                    LOOKAHEAD_MIN, LOOKAHEAD_MAX
                )
                
                # 模型预测
                raw_pred, steer_model = self.predict_steering(img)
                
                # 几何控制
                steer_geo = self.stanley_pure_pursuit_steer(lookahead)
                
                # 融合：几何为主 + 模型辅助
                steer = (1.0 - BLEND_BETA) * steer_geo + BLEND_BETA * steer_model
                steer = float(np.clip(steer, -1.0, 1.0))
                
                # 防自旋保护
                steer = self.anti_spin_protection(steer, speed)
                
                # 计算油门刹车
                throttle, brake = self.compute_control(steer, speed)
                
                # 低速防卡死
                if speed < 0.3 and step > 50 and brake == 0:
                    throttle = max(throttle, 0.4)
                
                # 应用控制
                control = carla.VehicleControl()
                control.steer = steer
                control.throttle = 0.0 if brake > 0 else throttle
                control.brake = brake
                control.hand_brake = False
                control.reverse = False
                self.vehicle.apply_control(control)
                
                # 更新视角
                self.update_spectator()
                
                # 打印状态
                print(f"\rraw:{raw_pred:+.4f} model:{steer_model:+.3f} geo:{steer_geo:+.3f} "
                      f"final:{steer:+.3f} thr:{throttle:.2f} brk:{brake:.2f} "
                      f"v:{speed:.1f}m/s Ld:{lookahead:.1f}", end="")
                
                step += 1
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n用户中断")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理中...")
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        cv2.destroyAllWindows()
        print("清理完成")


def main():
    driver = AutonomousDriverV2()
    
    try:
        driver.connect()
        driver.load_model()
        driver.spawn_vehicle()
        driver.setup_camera()
        driver.setup_spectator()
        driver.run()
    finally:
        driver.cleanup()


if __name__ == '__main__':
    main()

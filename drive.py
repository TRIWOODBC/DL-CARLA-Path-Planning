import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
sys.path.append(r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
import carla
import random
import time
import queue
import math


# ---------- 1. 模型定义 (必须和训练时完全一样！) ----------
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


# ---------- 2. 图像预处理函数 ----------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:-25, :, :]
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float() / 255.0
    image = image.unsqueeze(0)
    return image


def main():
    ego_vehicle = None
    camera = None
    try:
        # CARLA 初始化
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town03')

        # 加载模型
        print('正在加载模型...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PilotNet().to(device)
        state = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f'模型加载成功，将使用设备: {device}')

        # 创建车辆和摄像头
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

        # 只在一开始设置一次第三人称视角
        spectator = world.get_spectator()
        transform = ego_vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(
                transform.location + carla.Location(x=-15, z=3),  # 后方15米，高3米
                carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
            )
        )

        # 图像队列
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 控制参数（可调）
        BASE_THROTTLE = 0.50
        MIN_THROTTLE = 0.25
        STEER_SMOOTH_ALPHA = 0.0
        MAX_STEER_DELTA = 1.0
        STEER_GAIN = 10.0
        STEER_DEADZONE = 0.0
        THROTTLE_STEER_SCALE = 0.0

        prev_steer = 0.0

        print('车辆已准备就绪，按 CTRL+C 退出。')

        step = 0

        while True:
            carla_image = image_queue.get()

            img = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            img = img.reshape((carla_image.height, carla_image.width, 4))[:, :, :3]

            cv2.imshow('Ego Vehicle Camera', img)
            if cv2.waitKey(1) == ord('q'):
                break

            image_tensor = preprocess_image(img).to(device)
            with torch.no_grad():
                raw = float(model(image_tensor).item())  # 你原来的 predicted_steer

            # 如果你的训练标签其实是“角度（度）”，解开下一行映射到[-1,1]，否则保持注释：
            # raw = raw / 35.0

            # 前 60 帧做摆头自检：先右后左，确认控制链路确实生效
            if step < 60:
                steer = 0.40 if step < 30 else -0.40
                wiggle = True
            else:
                # ★ 无平滑、无死区、直接增益后裁剪
                steer = float(np.clip(raw * STEER_GAIN, -1.0, 1.0))
                wiggle = False

            # 根据转向幅度适度降油门，防止高速直冲
            throttle = BASE_THROTTLE * max(0.0, 1.0 - THROTTLE_STEER_SCALE * abs(steer))
            throttle = max(MIN_THROTTLE, min(BASE_THROTTLE, throttle))

            velocity = ego_vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = 0.0
            control.hand_brake = False   # ★ 确保没拉手刹
            control.reverse = False
            ego_vehicle.apply_control(control)

            print(f'raw:{raw:.6f} applied_steer:{steer:.6f} throttle:{throttle:.2f} speed:{speed:.2f} wiggle:{wiggle}', flush=True)

            step += 1
            time.sleep(0.05)  # 保留你的异步结构

    except KeyboardInterrupt:
        print('\n用户中断，准备退出')
    finally:
        print('\n正在清理场景...')
        if camera:
            camera.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        cv2.destroyAllWindows()
        print('清理完成。')


if __name__ == '__main__':
    main()
"""
数据采集脚本 - CARLA 自动驾驶数据采集
功能：
  - 多地图、多天气、多NPC密度采集
  - 自动过滤静止帧
  - 实时进度显示
  - 转向数据统计
"""
import sys
import os
import time
import csv
import random
import numpy as np
import cv2

from config import (
    CARLA_EGG_PATH, DATA_DIR, CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    TOWNS, NPC_COUNTS, COLLECTION_FPS, MIN_MOVE_M, DURATION_PER_SCENARIO_SEC,
    CAMERA_X, CAMERA_Z, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV, WEATHER_PRESETS
)
from utils import to_kmh

sys.path.append(CARLA_EGG_PATH)
import carla


def get_weather(name):
    """获取天气参数"""
    W = carla.WeatherParameters
    return {
        "ClearNoon": W.ClearNoon,
        "CloudySunset": W.CloudySunset,
        "WetNoon": W.WetNoon,
        "MidRainSunset": W.MidRainSunset,
        "SoftRainSunset": W.SoftRainSunset,
        "HardRainNoon": W.HardRainNoon,
        "ClearSunset": W.ClearSunset,
    }.get(name, W.ClearNoon)


def setup_traffic_manager(tm):
    """配置交通管理器，让NPC行为更激进，产生更多转弯数据"""
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.5)  # 跟车距离
    tm.global_percentage_speed_difference(-20)  # 比限速快20%
    # 增加变道和超车概率
    tm.set_random_device_seed(random.randint(0, 1000))


def create_ego_vehicle(world, tm, blueprints, spawn_points):
    """创建主车辆，优先选择普通轿车"""
    # 优先选择 Tesla Model 3 或其他轿车
    preferred = ['vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.bmw.grandtourer']
    bp = None
    for name in preferred:
        bp = world.get_blueprint_library().find(name)
        if bp:
            break
    if not bp:
        bp = random.choice(blueprints)
    
    # 尝试多个生成点
    random.shuffle(spawn_points)
    for sp in spawn_points[:10]:
        ego = world.try_spawn_actor(bp, sp)
        if ego:
            # 配置更激进的驾驶行为
            tm.ignore_lights_percentage(ego, 0)  # 遵守红绿灯
            tm.distance_to_leading_vehicle(ego, 2.0)  # 跟车距离
            tm.vehicle_percentage_speed_difference(ego, -10)  # 比限速快10%
            return ego
    return None


def create_camera(world, ego):
    """创建RGB相机"""
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    cam_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    cam_bp.set_attribute('fov', str(CAMERA_FOV))
    cam_bp.set_attribute('sensor_tick', str(1.0 / COLLECTION_FPS))
    
    # 相机位置：车头前方，稍微抬高
    transform = carla.Transform(carla.Location(x=CAMERA_X, z=CAMERA_Z))
    return world.spawn_actor(cam_bp, transform, attach_to=ego)


def spawn_npcs(world, tm, blueprints, spawn_points, count):
    """生成NPC车辆"""
    npcs = []
    random.shuffle(spawn_points)
    for sp in spawn_points[:count]:
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        v = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            # 随机设置不同的驾驶风格
            tm.vehicle_percentage_speed_difference(v, random.uniform(-30, 30))
            npcs.append(v)
    return npcs


def collect_scenario(world, ego, camera, spectator, scenario_id, town, weather, npc_n):
    """采集单个场景的数据"""
    out_dir = os.path.join(DATA_DIR, f"images_{scenario_id:03d}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, f"labels_{scenario_id:03d}.csv")
    
    # 写入CSV表头
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            "filename", "steer", "throttle", "brake", 
            "speed_kmh", "town", "weather", "npc", "frame"
        ])
    
    # 相机缓冲
    img_buf = {}
    camera.listen(lambda img: img_buf.update({img.frame: img}))
    
    # 采集状态
    last_loc = ego.get_transform().location
    moved = 0.0
    count = 0
    t_start = time.time()
    t_end = t_start + DURATION_PER_SCENARIO_SEC
    
    # 统计信息
    steer_sum = 0.0
    left_count = 0
    right_count = 0
    
    try:
        while time.time() < t_end:
            world.tick()
            frame = world.get_snapshot().frame
            image = img_buf.pop(frame, None)
            
            # 计算移动距离
            cur = ego.get_transform().location
            dist = ((cur.x - last_loc.x)**2 + (cur.y - last_loc.y)**2 + (cur.z - last_loc.z)**2)**0.5
            moved += dist
            last_loc = cur
            
            # 过滤静止帧
            if moved < MIN_MOVE_M or not image:
                continue
            moved = 0.0
            
            # 保存图像
            arr = np.frombuffer(image.raw_data, np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
            fname = f"{count:06d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), arr)
            
            # 记录控制数据
            ctrl = ego.get_control()
            speed = to_kmh(ego.get_velocity())
            
            with open(out_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    fname, ctrl.steer, ctrl.throttle, ctrl.brake,
                    speed, town, weather, npc_n, frame
                ])
            
            # 统计转向
            steer_sum += abs(ctrl.steer)
            if ctrl.steer < -0.05:
                left_count += 1
            elif ctrl.steer > 0.05:
                right_count += 1
            
            # 更新观众视角
            tr = ego.get_transform()
            spectator.set_transform(carla.Transform(
                tr.location + carla.Location(x=-6, z=3), tr.rotation
            ))
            
            count += 1
            
            # 显示进度
            elapsed = time.time() - t_start
            if count % 50 == 0:
                print(f"\r  进度: {elapsed:.0f}s/{DURATION_PER_SCENARIO_SEC}s | 帧数: {count} | 左转: {left_count} | 右转: {right_count}", end="")
    
    finally:
        camera.stop()
    
    # 返回统计信息
    avg_steer = steer_sum / max(count, 1)
    return count, left_count, right_count, avg_steer


def main():
    """主函数"""
    print("=" * 60)
    print("CARLA 数据采集脚本")
    print("=" * 60)
    
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    scenario_id = 0
    total_frames = 0
    total_left = 0
    total_right = 0

    for town in TOWNS:
        print(f"\n{'='*60}")
        print(f"加载地图: {town}")
        print(f"{'='*60}")
        
        # 加载地图，增加重试机制
        try:
            client.set_timeout(120.0)  # 加载地图需要更长超时
            client.load_world(town)
            time.sleep(5)  # 等待地图完全加载
            client.set_timeout(CARLA_TIMEOUT)  # 恢复正常超时
        except Exception as e:
            print(f"  ❌ 加载地图 {town} 失败: {e}")
            print(f"  跳过此地图，继续下一个...")
            continue
        
        world = client.get_world()
        
        # 重新获取 TrafficManager（切换地图后需要重新获取）
        try:
            tm = client.get_trafficmanager(8000)
        except Exception as e:
            print(f"  ⚠️ TrafficManager 错误，尝试使用其他端口...")
            tm = client.get_trafficmanager(8100)
        
        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / COLLECTION_FPS
        world.apply_settings(settings)
        setup_traffic_manager(tm)

        try:
            for weather in WEATHER_PRESETS:
                world.set_weather(get_weather(weather))
                
                for npc_n in NPC_COUNTS:
                    print(f"\n场景 {scenario_id}: {town} | {weather} | NPC={npc_n}")
                    
                    # 清除所有车辆
                    world.tick()
                    for a in world.get_actors().filter('*vehicle*'):
                        a.destroy()
                    world.tick()

                    # 获取蓝图和生成点
                    bps = list(world.get_blueprint_library().filter('*vehicle*'))
                    spawns = list(world.get_map().get_spawn_points())

                    # 生成NPC
                    npcs = spawn_npcs(world, tm, bps, spawns, npc_n)
                    print(f"  生成 NPC: {len(npcs)}/{npc_n}")

                    # 创建主车辆
                    ego = create_ego_vehicle(world, tm, bps, spawns)
                    if not ego:
                        print("  ❌ 无法生成主车辆，跳过")
                        continue
                    ego.set_autopilot(True, tm.get_port())

                    # 创建相机
                    camera = create_camera(world, ego)
                    spectator = world.get_spectator()

                    try:
                        # 采集数据
                        frames, left, right, avg_steer = collect_scenario(
                            world, ego, camera, spectator,
                            scenario_id, town, weather, npc_n
                        )
                        
                        print(f"\n  ✅ 完成: {frames} 帧 | 左转: {left} | 右转: {right} | 平均转向: {avg_steer:.4f}")
                        
                        total_frames += frames
                        total_left += left
                        total_right += right
                        scenario_id += 1
                        
                    finally:
                        camera.destroy()
                        ego.destroy()
                        for n in npcs:
                            try:
                                n.destroy()
                            except:
                                pass

        finally:
            # 恢复异步模式
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)

    # 打印总结
    print(f"\n{'='*60}")
    print("采集完成!")
    print(f"{'='*60}")
    print(f"总场景数: {scenario_id}")
    print(f"总帧数: {total_frames}")
    print(f"左转帧数: {total_left} ({100*total_left/max(total_frames,1):.1f}%)")
    print(f"右转帧数: {total_right} ({100*total_right/max(total_frames,1):.1f}%)")
    print(f"直行帧数: {total_frames - total_left - total_right} ({100*(total_frames-total_left-total_right)/max(total_frames,1):.1f}%)")
    print(f"数据保存至: {DATA_DIR}")


if __name__ == '__main__':
    main()

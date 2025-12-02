import random, time, os, csv, numpy as np, cv2, sys
sys.path.append(r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
import carla

# ---------- 参数 ----------
TOWNS = ['Town01','Town03','Town05','Town07']   # 想快点就改成 ['Town05']
NPC_COUNTS = [30, 80, 140]                      # 稀疏/中/拥挤
FPS = 10                                        # 10 FPS
MIN_MOVE_M = 0.3                                # 两帧累计位移<此阈值则跳过保存
SAVE_DIR = "data_more"                          # 输出目录
DURATION_PER_SCENARIO_SEC = 180                 # 每个场景跑几分钟

# ---------- 采集 ----------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

def to_kmh(v):
    return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

os.makedirs(SAVE_DIR, exist_ok=True)
scenario_id = 0

# 天气预设
W = carla.WeatherParameters
WEATHERS = [
    ("ClearNoon",   W.ClearNoon),
    ("CloudySunset",W.CloudySunset),
    ("WetNoon",     W.WetNoon),
    ("MidRainSunset",W.MidRainSunset),
    ("SoftRainSunset",   W.SoftRainSunset),
]

for town in TOWNS:
    client.load_world(town)
    world = client.get_world()

    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.0)  # 贴近点，更多交互

    try:
        for wname, wparam in WEATHERS:
            world.set_weather(wparam)

            for npc_n in NPC_COUNTS:
                # 清场
                world.tick()
                for a in world.get_actors().filter('*vehicle*'):
                    a.destroy()
                world.tick()

                # 车辆蓝图 & 生成点
                blueprints = world.get_blueprint_library().filter('*vehicle*')
                spawns = world.get_map().get_spawn_points()

                # NPC
                import random as _R
                _R.shuffle(spawns)
                spawned = 0
                for sp in spawns:
                    if spawned >= npc_n: break
                    bp = _R.choice(blueprints)
                    v = world.try_spawn_actor(bp, sp)
                    if v:
                        v.set_autopilot(True, tm.get_port())
                        spawned += 1

                # Ego
                ego_bp = _R.choice(blueprints)
                ego = world.try_spawn_actor(ego_bp, _R.choice(spawns))
                if ego is None:
                    continue

                # 相机（稍前、抬高，避免仪表台）
                cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                cam_bp.set_attribute('sensor_tick', str(1.0/FPS))
                cam_bp.set_attribute('image_size_x', '1280')
                cam_bp.set_attribute('image_size_y', '720')
                cam_bp.set_attribute('fov', '90')
                cam = world.spawn_actor(
                    cam_bp,
                    carla.Transform(carla.Location(x=0.5, z=1.6)),
                    attach_to=ego
                )

                # Ego 也交给 TM（简单稳妥）
                ego.set_autopilot(True, tm.get_port())

                # 观众跟随
                spectator = world.get_spectator()

                # 输出目录与CSV
                out_imgs = os.path.join(SAVE_DIR, f"images_{scenario_id:03d}")
                os.makedirs(out_imgs, exist_ok=True)
                out_csv = os.path.join(SAVE_DIR, f"labels_{scenario_id:03d}.csv")
                with open(out_csv, 'w', newline='') as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["filename","steer","throttle","brake","speed_kmh","town","weather","npc","frame"])

                # 异步相机缓冲（按帧号取）
                img_buf = {}
                def on_cam(image):
                    img_buf[image.frame] = image
                cam.listen(on_cam)

                # 采集主循环（同步 tick）
                last_loc = ego.get_transform().location
                moved_accum = 0.0
                frame_count = 0
                t_end = time.time() + DURATION_PER_SCENARIO_SEC

                try:
                    while time.time() < t_end:
                        world.tick()
                        snap = world.get_snapshot()
                        frame = snap.frame
                        image = img_buf.pop(frame, None)

                        # 去冗余：不动就不存
                        cur = ego.get_transform().location
                        dx = ((cur.x-last_loc.x)**2 + (cur.y-last_loc.y)**2 + (cur.z-last_loc.z)**2)**0.5
                        moved_accum += dx
                        last_loc = cur
                        if moved_accum < MIN_MOVE_M:
                            continue
                        moved_accum = 0.0

                        if image is None:
                            continue

                        # 存图（原图；裁剪训练时再做）
                        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
                        fname = f"{frame_count:06d}.png"
                        cv2.imwrite(os.path.join(out_imgs, fname), arr)

                        # 读取控制（TM自动驾驶也会写到VehicleControl；同步模式下时序一致）
                        ctrl = ego.get_control()
                        spd = to_kmh(ego.get_velocity())

                        with open(out_csv, 'a', newline='') as f:
                            wcsv = csv.writer(f)
                            wcsv.writerow([fname, float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake),
                                           spd, town, wname, npc_n, int(frame)])

                        # 视角跟随
                        tr = ego.get_transform()
                        spectator.set_transform(carla.Transform(tr.location + carla.Location(x=-6, z=3), tr.rotation))

                        frame_count += 1

                finally:
                    cam.stop(); cam.destroy()
                    ego.destroy()
                    for v in world.get_actors().filter('*vehicle*'):
                        try: v.destroy()
                        except: pass

                scenario_id += 1
                print(f"[OK] {town} {wname} NPC={npc_n} -> {frame_count} frames, saved to {out_imgs}")

    finally:
        # 退出前恢复异步
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

print("[DONE] all scenarios")

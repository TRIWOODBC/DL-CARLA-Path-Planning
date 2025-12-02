"""
数据采集脚本
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
    W = carla.WeatherParameters
    return {"ClearNoon": W.ClearNoon, "CloudySunset": W.CloudySunset,
            "WetNoon": W.WetNoon, "MidRainSunset": W.MidRainSunset,
            "SoftRainSunset": W.SoftRainSunset}.get(name, W.ClearNoon)


def main():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    os.makedirs(DATA_DIR, exist_ok=True)
    scenario_id = 0

    for town in TOWNS:
        print(f"Loading {town}")
        client.load_world(town)
        world = client.get_world()
        tm = client.get_trafficmanager(8000)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / COLLECTION_FPS
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        try:
            for weather in WEATHER_PRESETS:
                world.set_weather(get_weather(weather))
                for npc_n in NPC_COUNTS:
                    print(f"Scene: {town} | {weather} | NPC={npc_n}")
                    
                    world.tick()
                    for a in world.get_actors().filter('*vehicle*'):
                        a.destroy()
                    world.tick()

                    bps = world.get_blueprint_library().filter('*vehicle*')
                    spawns = list(world.get_map().get_spawn_points())
                    random.shuffle(spawns)

                    npcs = []
                    for sp in spawns[:npc_n]:
                        v = world.try_spawn_actor(random.choice(bps), sp)
                        if v:
                            v.set_autopilot(True, tm.get_port())
                            npcs.append(v)

                    ego = world.try_spawn_actor(random.choice(bps), random.choice(spawns))
                    if not ego:
                        continue

                    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                    cam_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
                    cam_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
                    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=CAMERA_X, z=CAMERA_Z)), attach_to=ego)
                    ego.set_autopilot(True, tm.get_port())

                    out_dir = os.path.join(DATA_DIR, f"images_{scenario_id:03d}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_csv = os.path.join(DATA_DIR, f"labels_{scenario_id:03d}.csv")
                    with open(out_csv, 'w', newline='') as f:
                        csv.writer(f).writerow(["filename","steer","throttle","brake","speed_kmh","town","weather","npc","frame"])

                    img_buf = {}
                    cam.listen(lambda img: img_buf.update({img.frame: img}))

                    last_loc = ego.get_transform().location
                    moved = 0.0
                    count = 0
                    t_end = time.time() + DURATION_PER_SCENARIO_SEC

                    try:
                        while time.time() < t_end:
                            world.tick()
                            frame = world.get_snapshot().frame
                            image = img_buf.pop(frame, None)
                            cur = ego.get_transform().location
                            moved += ((cur.x-last_loc.x)**2+(cur.y-last_loc.y)**2+(cur.z-last_loc.z)**2)**0.5
                            last_loc = cur
                            if moved < MIN_MOVE_M or not image:
                                continue
                            moved = 0.0
                            arr = np.frombuffer(image.raw_data, np.uint8).reshape((image.height, image.width, 4))[:,:,:3]
                            fname = f"{count:06d}.png"
                            cv2.imwrite(os.path.join(out_dir, fname), arr)
                            ctrl = ego.get_control()
                            with open(out_csv, 'a', newline='') as f:
                                csv.writer(f).writerow([fname, ctrl.steer, ctrl.throttle, ctrl.brake, to_kmh(ego.get_velocity()), town, weather, npc_n, frame])
                            count += 1
                    finally:
                        cam.stop()
                        cam.destroy()
                        ego.destroy()
                        for n in npcs:
                            try: n.destroy()
                            except: pass

                    print(f"  OK: {count} frames")
                    scenario_id += 1
        finally:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)

    print(f"Done: {scenario_id} scenarios")


if __name__ == '__main__':
    main()

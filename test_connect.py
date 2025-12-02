import sys
sys.path.append(r'D:\CARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg')
import carla
print('1) creating client...', flush=True)
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
print('2) get world...', flush=True)
world = client.get_world()
print('3) connected. map =', world.get_map().name, flush=True)

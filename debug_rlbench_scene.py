import os
import numpy as np
from pyrep.const import RenderMode

print("===== ENV =====")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("QT_QPA_PLATFORM:", os.environ.get("QT_QPA_PLATFORM"))
print("QT_XCB_GL_INTEGRATION:", os.environ.get("QT_XCB_GL_INTEGRATION"))
print("QT_OPENGL:", os.environ.get("QT_OPENGL"))
print("COPPELIASIM_ROOT:", os.environ.get("COPPELIASIM_ROOT"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("LIBGL_DRIVERS_PATH:", os.environ.get("LIBGL_DRIVERS_PATH"))

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import PutToiletRollOnStand

def get_render_mode():
    mode = os.environ.get("DEBUG_RENDER_MODE", "OPENGL").strip().upper()
    if mode == "OPENGL3":
        return RenderMode.OPENGL3
    if mode == "OPENGL":
        return RenderMode.OPENGL
    if mode == "OPENGL_AUXILIARY":
        return RenderMode.OPENGL_AUXILIARY
    raise ValueError(f"Unknown DEBUG_RENDER_MODE={mode}")

DATASET = "/data/xzha593/projects/X-ICM/data/unseen_tasks/test"

enabled = os.environ.get("DEBUG_CAMERAS", "front").strip()

if enabled.lower() in ["none", "off", ""]:
    enabled_cameras = set()
elif enabled.lower() == "all":
    enabled_cameras = {"front", "left_shoulder", "right_shoulder", "wrist"}
else:
    enabled_cameras = {x.strip() for x in enabled.split(",") if x.strip()}

print("===== DEBUG CAMERAS =====")
print("enabled_cameras:", enabled_cameras)

def make_cam(name):
    use_rgb = name in enabled_cameras
    return CameraConfig(
        rgb=use_rgb,
        depth=False,
        mask=False,
        point_cloud=False,
        image_size=(128, 128),
        render_mode=RenderMode.OPENGL,
    )

print("===== BUILD OBS CONFIG =====")

obs_config = ObservationConfig(
    left_shoulder_camera=make_cam("left_shoulder"),
    right_shoulder_camera=make_cam("right_shoulder"),
    overhead_camera=CameraConfig(
        rgb=False,
        depth=False,
        mask=False,
        point_cloud=False,
        image_size=(128, 128),
        render_mode=get_render_mode(),
    ),
    wrist_camera=make_cam("wrist"),
    front_camera=make_cam("front"),
)

print("DEBUG_RENDER_MODE:", os.environ.get("DEBUG_RENDER_MODE", "OPENGL"))
print("Resolved render_mode:", get_render_mode())

obs_config.joint_velocities = True
obs_config.joint_positions = True
obs_config.joint_forces = False
obs_config.gripper_open = True
obs_config.gripper_pose = True
obs_config.gripper_matrix = False
obs_config.gripper_touch_forces = False
obs_config.task_low_dim_state = True

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete(),
)

print("===== LAUNCH RLBench ENV =====")
env = Environment(
    action_mode=action_mode,
    dataset_root=DATASET,
    obs_config=obs_config,
    headless=True,
)

env.launch()
print("Environment launched")

print("===== LOAD TASK =====")
task = env.get_task(PutToiletRollOnStand)
print("Task loaded:", task)

task.set_variation(-1)

print("===== LOAD DEMO =====")
demos = task.get_demos(
    amount=1,
    live_demos=False,
    from_episode_number=0,
)
print("Demo loaded:", len(demos))

print("===== RESET TO DEMO =====")
descriptions, obs = task.reset_to_demo(demos[0])
print("Reset OK")
print("Descriptions:", descriptions)

print("===== OBS CHECK =====")
for name in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]:
    arr = getattr(obs, name, None)
    if arr is None:
        print(name, "None")
    else:
        print(
            name,
            type(arr),
            arr.shape,
            arr.dtype,
            "min=", np.min(arr),
            "max=", np.max(arr),
        )

print("===== LOW-DIM CHECK =====")
print("joint_positions:", None if obs.joint_positions is None else np.asarray(obs.joint_positions).shape)
print("gripper_pose:", None if obs.gripper_pose is None else np.asarray(obs.gripper_pose).shape)
print("task_low_dim_state:", None if obs.task_low_dim_state is None else np.asarray(obs.task_low_dim_state).shape)

env.shutdown()
print("RLBench camera debug OK")
"""
References: https://github.com/LightwheelAI/leisaac
Unified data generation script using state machines.

Selects the appropriate state machine based on --task and runs the recording loop.
Episode count is driven by --object_poses: each ``status == "full"`` entry in the
file yields one replayed episode. Object placements are written via
``RigidObject.write_root_pose_to_sim`` after each ``env.reset()``.

Usage:
    python scripts/datagen/generate.py \
        --task HCIS-CupStacking-SingleArm-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --use_lerobot_recorder \
        --lerobot_dataset_repo_id MidoriChou/ai_final_3 \
        --object_poses data/ai_final_3/object_poses.json
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import math
import os
import random
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/dataset.hdf5",
    help="HDF5 output path used only when --use_lerobot_recorder is not set.",
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--target_demo_count",
    type=int,
    default=None,
    help=(
        "Total number of successful demos to keep generating until. With --resume, "
        "existing demos in the selected recorder dataset are counted and new demos are appended."
    ),
)
parser.add_argument(
    "--object_poses",
    type=str,
    required=True,
    help="Path to the per-episode object_poses.json (UMI schema). Episode count = number of status=='full' entries.",
)
parser.add_argument(
    "--samples_per_pose",
    type=int,
    default=1,
    help="Number of randomized trajectories to generate from each object_poses entry, starting at entry 0.",
)
parser.add_argument(
    "--pose_jitter_xy",
    type=float,
    default=0.0,
    help="Uniform random x/y jitter in meters applied independently to each object pose.",
)
parser.add_argument(
    "--pose_jitter_yaw_deg",
    type=float,
    default=0.0,
    help="Uniform random yaw jitter in degrees applied independently to each object pose.",
)
parser.add_argument(
    "--min_cup_distance",
    type=float,
    default=0.08,
    help="Minimum initial xy distance between blue_cup and pink_cup after jitter.",
)
parser.add_argument(
    "--max_cup_distance",
    type=float,
    default=0.20,
    help="Maximum initial xy distance between blue_cup and pink_cup after jitter.",
)
parser.add_argument(
    "--cup_workspace_x_range",
    type=float,
    nargs=2,
    default=(0.35, 0.55),
    metavar=("MIN", "MAX"),
    help="Allowed world x range for cup initial poses after jitter.",
)
parser.add_argument(
    "--cup_workspace_y_range",
    type=float,
    nargs=2,
    default=(-0.36, -0.16),
    metavar=("MIN", "MAX"),
    help="Allowed world y range for cup initial poses after jitter.",
)
parser.add_argument(
    "--disable_cup_workspace_clamp",
    action="store_true",
    help="Disable cup workspace regularization and use raw object_poses+jitter positions.",
)
parser.add_argument(
    "--cup_pair_flip_prob",
    type=float,
    default=0.5,
    help=(
        "Probability of flipping the blue/pink relative layout around their midpoint after jitter. "
        "Use 0 to preserve object_poses ordering."
    ),
)
parser.add_argument(
    "--sample_cup_layout",
    action="store_true",
    help="Sample cup center, distance, and angle uniformly inside the safe workspace instead of clamping raw poses.",
)
parser.add_argument(
    "--cup_layout_profile",
    choices=("source", "eval", "eval_wide"),
    default="source",
    help=(
        "Cup start-pose distribution. 'source' uses object_poses plus the workspace options; "
        "'eval' samples the fixed cup-stacking evaluation distribution; 'eval_wide' adds a small "
        "reach-safe margin around the evaluation distribution."
    ),
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")
parser.add_argument(
    "--lerobot_dataset_local_dir",
    type=str,
    default=None,
    help=(
        "Local LeRobot dataset directory used to count existing episodes when resuming. "
        "Defaults to ./data/<repo_name> for repo ids like user/repo."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import simulator.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.datagen.state_machine import PickOrangeStateMachine
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

from simulator.datagen.state_machine.cup_stacking import CupStackingStateMachine
from simulator.datagen.state_machine.cutlery_arrangement import CutleryArrangementStateMachine
from simulator.datagen.state_machine.toy_blocks_collection import ToyBlocksCollectionStateMachine
from simulator.utils.cup_stacking_layout import (
    CUP_STACKING_EVAL_SPAWNS,
    CUP_STACKING_EVAL_WIDE_SPAWNS,
    sample_cup_stacking_eval_layout,
    sample_cup_stacking_eval_wide_layout,
)
from simulator.utils.object_poses_loader import load_episode_poses

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
    "HCIS-ToyBlocksCollection-SingleArm-v0": (ToyBlocksCollectionStateMachine, "keyboard"),
    "HCIS-CutleryArrangement-SingleArm-v0": (CutleryArrangementStateMachine, "keyboard"),
}


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _get_dataset_export_mode(args_cli):
    if args_cli.use_lerobot_recorder:
        if args_cli.resume:
            return EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
        return DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    if args_cli.resume:
        return EnhanceDatasetExportMode.EXPORT_ALL_RESUME
    return DatasetExportMode.EXPORT_ALL


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    """Configure termination and recorder settings on env_cfg."""
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            # The default IsaacLab RecorderManager is constructed during gym.make.
            # Keep recorder terms active, but suppress its HDF5 file handler until
            # we replace it with LeRobotRecorderManager after env creation.
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = _get_dataset_export_mode(args_cli)
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = _get_dataset_export_mode(args_cli)
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
            env_cfg.recorders.dataset_export_dir_path = output_dir
            env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    """Replace the default recorder manager with streaming or lerobot recorder."""
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import (
            LeRobotRecorderManager,
        )

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env_cfg.recorders.dataset_export_mode = _get_dataset_export_mode(args_cli)
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def _resolve_lerobot_dataset_dir(args_cli):
    """Return the local LeRobot dataset directory used by lerobot-train, if present."""
    if args_cli.lerobot_dataset_local_dir:
        return args_cli.lerobot_dataset_local_dir
    if not args_cli.lerobot_dataset_repo_id:
        return None
    repo_name = args_cli.lerobot_dataset_repo_id.rsplit("/", 1)[-1]
    hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    hf_lerobot_home = os.path.expanduser(
        os.environ.get("HF_LEROBOT_HOME", os.path.join(hf_home, "lerobot"))
    )
    candidates = [
        os.path.join(hf_lerobot_home, args_cli.lerobot_dataset_repo_id),
        os.path.join("data", repo_name),
        args_cli.lerobot_dataset_repo_id,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _get_lerobot_episode_count(args_cli):
    dataset_dir = _resolve_lerobot_dataset_dir(args_cli)
    if not dataset_dir:
        return 0
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return 0
    with open(info_path) as f:
        info = json.load(f)
    return int(info.get("total_episodes", 0))


def _validate_lerobot_resume_dataset(args_cli):
    """Fail early when an interrupted LeRobot dataset is not safely resumable."""
    if not (args_cli.record and args_cli.use_lerobot_recorder and args_cli.resume):
        return

    dataset_dir = _resolve_lerobot_dataset_dir(args_cli)
    if not dataset_dir or not os.path.isdir(dataset_dir):
        raise ValueError(
            "--resume requested, but the local LeRobot dataset directory does not exist: "
            f"{dataset_dir}"
        )

    info_path = os.path.join(dataset_dir, "meta", "info.json")
    episodes_dir = os.path.join(dataset_dir, "meta", "episodes")
    has_episode_metadata = os.path.isdir(episodes_dir) and any(
        filename.endswith(".parquet")
        for _, _, filenames in os.walk(episodes_dir)
        for filename in filenames
    )
    if not os.path.isfile(info_path) or not has_episode_metadata:
        raise ValueError(
            f"LeRobot dataset at {dataset_dir} is incomplete and cannot be resumed safely. "
            "Preserve or remove that directory, then start a clean dataset without --resume."
        )


def _get_resume_recorded_demo_count(env, args_cli):
    if args_cli.use_lerobot_recorder:
        count = _get_lerobot_episode_count(args_cli)
        dataset_dir = _resolve_lerobot_dataset_dir(args_cli)
        print(f"Resume recording from LeRobot dataset {dataset_dir} with {count} demonstrations.")
        return count
    count = env.recorder_manager._dataset_file_handler.get_num_episodes()
    print(f"Resume recording from existing HDF5 dataset file with {count} demonstrations.")
    return count


def _get_exported_successful_episode_count(env):
    return getattr(env.recorder_manager, "exported_successful_episode_count", 0)


def _ensure_physics_sim_view(env):
    """Ensure PhysX tensor views exist before robot/object handles are touched."""
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_physics_sim_view() is not None:
        return

    print("[INFO] PhysX simulation view is not ready; resetting simulator before manager setup.")
    env.sim.reset()
    env.scene.update(dt=env.physics_dt)
    if not hasattr(env, "recorder_manager"):
        env.load_managers()


def _quat_with_yaw_delta(quat, yaw_delta):
    """Apply a world-z yaw delta to a wxyz quaternion."""
    half_yaw = 0.5 * yaw_delta
    yaw_quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    w1, x1, y1, z1 = yaw_quat
    w2, x2, y2, z2 = quat
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _jitter_episode_poses(
    poses,
    rng,
    args_cli,
    xy_jitter: float,
    yaw_jitter_rad: float,
    min_cup_distance: float,
):
    """Return a jittered copy of episode poses without mutating the source."""
    jittered = {}
    for name, (pos, quat) in poses.items():
        dx = rng.uniform(-xy_jitter, xy_jitter) if xy_jitter > 0.0 else 0.0
        dy = rng.uniform(-xy_jitter, xy_jitter) if xy_jitter > 0.0 else 0.0
        dyaw = rng.uniform(-yaw_jitter_rad, yaw_jitter_rad) if yaw_jitter_rad > 0.0 else 0.0
        jittered[name] = (
            (pos[0] + dx, pos[1] + dy, pos[2]),
            _quat_with_yaw_delta(quat, dyaw) if dyaw != 0.0 else quat,
        )

    if "blue_cup" not in jittered or "pink_cup" not in jittered:
        return jittered

    if not args_cli.disable_cup_workspace_clamp:
        jittered = _regularize_cup_stacking_poses(jittered, rng, args_cli)

    if min_cup_distance <= 0.0:
        return jittered

    blue_pos = jittered["blue_cup"][0]
    pink_pos = jittered["pink_cup"][0]
    cup_distance = math.hypot(blue_pos[0] - pink_pos[0], blue_pos[1] - pink_pos[1])
    if cup_distance >= min_cup_distance:
        return jittered

    # If independent jitter makes the cups too close, push them apart along their
    # current separation direction while preserving the sampled midpoint.
    if cup_distance < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux = (blue_pos[0] - pink_pos[0]) / cup_distance
        uy = (blue_pos[1] - pink_pos[1]) / cup_distance
    mid_x = 0.5 * (blue_pos[0] + pink_pos[0])
    mid_y = 0.5 * (blue_pos[1] + pink_pos[1])
    half_distance = 0.5 * min_cup_distance

    blue_quat = jittered["blue_cup"][1]
    pink_quat = jittered["pink_cup"][1]
    jittered["blue_cup"] = (
        (mid_x + ux * half_distance, mid_y + uy * half_distance, blue_pos[2]),
        blue_quat,
    )
    jittered["pink_cup"] = (
        (mid_x - ux * half_distance, mid_y - uy * half_distance, pink_pos[2]),
        pink_quat,
    )
    return jittered


def _prepare_episode_poses(poses, rng, args_cli):
    """Select and randomize the start-pose distribution for one episode."""
    if args_cli.cup_layout_profile == "eval":
        return sample_cup_stacking_eval_layout(rng)
    if args_cli.cup_layout_profile == "eval_wide":
        return sample_cup_stacking_eval_wide_layout(rng)
    return _jitter_episode_poses(
        poses,
        rng,
        args_cli,
        args_cli.pose_jitter_xy,
        math.radians(args_cli.pose_jitter_yaw_deg),
        args_cli.min_cup_distance,
    )


def _regularize_cup_stacking_poses(poses, rng, args_cli):
    """Keep cup starts in the reliable workspace of the scripted IK policy."""
    blue_pos, blue_quat = poses["blue_cup"]
    pink_pos, pink_quat = poses["pink_cup"]
    x_min, x_max = args_cli.cup_workspace_x_range
    y_min, y_max = args_cli.cup_workspace_y_range
    min_distance = max(args_cli.min_cup_distance, 0.0)
    max_distance = max(args_cli.max_cup_distance, min_distance)

    if args_cli.sample_cup_layout:
        for _ in range(100):
            angle = rng.uniform(-math.pi, math.pi)
            ux, uy = math.cos(angle), math.sin(angle)
            distance = rng.uniform(min_distance, max_distance)
            half_dx = 0.5 * distance * ux
            half_dy = 0.5 * distance * uy
            if x_min + abs(half_dx) <= x_max - abs(half_dx) and y_min + abs(half_dy) <= y_max - abs(half_dy):
                break
        else:
            raise ValueError("Unable to sample cup layout inside workspace; reduce --max_cup_distance or widen ranges.")
    else:
        dx = blue_pos[0] - pink_pos[0]
        dy = blue_pos[1] - pink_pos[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            angle = rng.uniform(-math.pi, math.pi)
            ux, uy = math.cos(angle), math.sin(angle)
        else:
            ux, uy = dx / distance, dy / distance
        should_flip = rng.random() < args_cli.cup_pair_flip_prob
        if should_flip:
            print("[pose] flipped blue/pink relative layout")
            ux, uy = -ux, -uy
        distance = min(max(distance, min_distance), max_distance)
        half_dx = 0.5 * distance * ux
        half_dy = 0.5 * distance * uy

    center_x_min = x_min + abs(half_dx)
    center_x_max = x_max - abs(half_dx)
    center_y_min = y_min + abs(half_dy)
    center_y_max = y_max - abs(half_dy)
    if args_cli.sample_cup_layout:
        center_x = rng.uniform(center_x_min, center_x_max)
        center_y = rng.uniform(center_y_min, center_y_max)
        print(
            "[pose] sampled cup layout "
            f"center=({center_x:.3f}, {center_y:.3f}) distance={distance:.3f}"
        )
    else:
        center_x = 0.5 * (blue_pos[0] + pink_pos[0])
        center_y = 0.5 * (blue_pos[1] + pink_pos[1])
        center_x = min(max(center_x, center_x_min), center_x_max)
        center_y = min(max(center_y, center_y_min), center_y_max)

    regularized = dict(poses)
    regularized["blue_cup"] = ((center_x + half_dx, center_y + half_dy, blue_pos[2]), blue_quat)
    regularized["pink_cup"] = ((center_x - half_dx, center_y - half_dy, pink_pos[2]), pink_quat)
    return regularized


def _apply_episode_poses(env, poses, args_cli, episode_seed: int):
    """Write per-object root poses for the current episode into the sim."""
    rng = random.Random(episode_seed)
    poses = _prepare_episode_poses(poses, rng, args_cli)

    device = env.device
    for name, (pos, quat) in poses.items():
        obj = env.scene[name]
        pose_tensor = torch.tensor(
            [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
            device=device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        obj.write_root_pose_to_sim(pose_tensor)
        w, x, y, z = quat
        yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        print(
            f"  [pose] {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
            f"yaw={yaw_deg:+6.1f}°"
        )


def _should_stop_generating(
    args_cli,
    current_recorded_demo_count: int,
    attempted_episode_count: int,
    total_episodes: int,
):
    if args_cli.record and args_cli.target_demo_count is not None:
        return current_recorded_demo_count >= args_cli.target_demo_count
    return attempted_episode_count >= total_episodes


def _progress_counts(
    args_cli,
    current_recorded_demo_count: int,
    attempted_episode_count: int,
    total_episodes: int,
):
    if args_cli.record and args_cli.target_demo_count is not None:
        return current_recorded_demo_count, args_cli.target_demo_count
    return min(attempted_episode_count, total_episodes), total_episodes


def _planned_pose_attempt_count(args_cli, pose_count: int) -> int:
    return pose_count * args_cli.samples_per_pose


def _pose_idx_for_attempt(args_cli, attempted_episode_count: int, pose_count: int) -> int:
    return (attempted_episode_count // args_cli.samples_per_pose) % pose_count


def _pose_sample_idx_for_attempt(args_cli, attempted_episode_count: int) -> int:
    return attempted_episode_count % args_cli.samples_per_pose


# z below which a task object is considered to have fallen off the table.
# Objects sit at object_z ≈ 0.05; anything under the table surface trips this.
_FALL_THRESHOLD_Z: float = 0.0


def _any_object_fell(env, object_names, z_threshold: float) -> bool:
    """Return True if any named scene object has root_pos_w.z below z_threshold."""
    for name in object_names:
        try:
            obj = env.scene[name]
        except KeyError:
            continue
        if torch.any(obj.data.root_pos_w[:, 2] < z_threshold).item():
            return True
    return False


def _print_episode_diagnostics(env, label: str):
    """Print task-specific final state for success/failure inspection."""
    try:
        blue_cup = env.scene["blue_cup"]
        pink_cup = env.scene["pink_cup"]
    except KeyError:
        return

    blue_pos = (blue_cup.data.root_pos_w - env.scene.env_origins)[0]
    pink_pos = (pink_cup.data.root_pos_w - env.scene.env_origins)[0]
    dx = blue_pos[0] - pink_pos[0]
    dy = blue_pos[1] - pink_pos[1]
    dz = blue_pos[2] - pink_pos[2]
    xy_dist = torch.linalg.norm(torch.stack((dx, dy))).item()
    print(
        f"[{label} Debug] blue-pink "
        f"dx={dx.item():+.3f} dy={dy.item():+.3f} dz={dz.item():+.3f} "
        f"xy_dist={xy_dist:.3f}"
    )
    print(
        f"[{label} Debug] final poses "
        f"blue=({blue_pos[0].item():.3f}, {blue_pos[1].item():.3f}, {blue_pos[2].item():.3f}) "
        f"pink=({pink_pos[0].item():.3f}, {pink_pos[1].item():.3f}, {pink_pos[2].item():.3f})"
    )
    if label == "Fail":
        print(
            "[Fail Debug] success requires "
            "|dx|<0.050, |dy|<0.050, dz>0.100, and no task object below z=0.000"
        )
        if blue_pos[2].item() < _FALL_THRESHOLD_Z or pink_pos[2].item() < _FALL_THRESHOLD_Z:
            print("[Fail Debug] at least one cup is below the table threshold.")


def _on_episode_done(
    env,
    sm,
    args_cli,
    episodes,
    next_pose_idx,
    attempted_episode_count,
    episode_seed_base,
    resume_recorded_demo_count,
    current_recorded_demo_count,
    start_record_state,
):
    """Handle end-of-episode logic.

    Returns (next_pose_idx, attempted_episode_count, current_recorded_demo_count, start_record_state, should_break).
    """
    total_episodes = _planned_pose_attempt_count(args_cli, len(episodes))

    try:
        success = sm.check_success(env)
    except Exception as e:
        print("Success check failed:", e)
        success = False

    print("Episode success!" if success else "Episode failed!")
    _print_episode_diagnostics(env, "Success" if success else "Fail")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record and success:
        auto_terminate(env, True)
        current_recorded_demo_count += 1
    else:
        auto_terminate(env, False)

    # Recorder managers export the just-finished episode during reset. Flush
    # before checking the target count so the final requested demo is not lost
    # when generation exits immediately after reaching the target.
    env.reset()

    if (
        args_cli.record
        and _get_exported_successful_episode_count(env) + resume_recorded_demo_count
        > current_recorded_demo_count
    ):
        current_recorded_demo_count = (
            _get_exported_successful_episode_count(env) + resume_recorded_demo_count
        )
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    if _should_stop_generating(args_cli, current_recorded_demo_count, attempted_episode_count, total_episodes):
        if args_cli.record and args_cli.target_demo_count is not None:
            print(f"Reached target demo count {args_cli.target_demo_count}. Exiting the app.")
        else:
            print(f"Replayed all {total_episodes} pose attempts. Exiting the app.")
        return next_pose_idx, attempted_episode_count, current_recorded_demo_count, start_record_state, True, success

    sm.reset()
    auto_terminate(env, False)
    next_pose_idx = _pose_idx_for_attempt(args_cli, attempted_episode_count, len(episodes))
    next_sample_idx = _pose_sample_idx_for_attempt(args_cli, attempted_episode_count)
    print(
        f"[pose] using object_poses[{next_pose_idx}] "
        f"sample {next_sample_idx + 1}/{args_cli.samples_per_pose}"
    )
    _apply_episode_poses(env, episodes[next_pose_idx], args_cli, episode_seed_base + attempted_episode_count)
    attempted_episode_count += 1

    return next_pose_idx, attempted_episode_count, current_recorded_demo_count, start_record_state, False, success


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]

    if args_cli.use_lerobot_recorder:
        if not args_cli.lerobot_dataset_repo_id:
            raise ValueError("--use_lerobot_recorder requires --lerobot_dataset_repo_id.")
        output_dir = ""
        output_file_name = "unused_lerobot_dataset"
    else:
        output_dir = os.path.dirname(args_cli.dataset_file)
        output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    if getattr(env_cfg, "object_pose_cfg", None) is None:
        raise ValueError(
            f"Task '{task_name}' env_cfg has no 'object_pose_cfg' attribute; "
            "cannot resolve anchor frame for --object_poses."
        )
    episodes = load_episode_poses(args_cli.object_poses, env_cfg.object_pose_cfg)
    if not episodes:
        raise ValueError(
            f"No 'status==full' episodes in {args_cli.object_poses}; nothing to replay."
        )
    print(f"Loaded {len(episodes)} replay episodes from {args_cli.object_poses}")
    if args_cli.target_demo_count is not None and args_cli.target_demo_count <= 0:
        raise ValueError("--target_demo_count must be a positive integer when provided.")
    if args_cli.target_demo_count is not None and not args_cli.record:
        raise ValueError("--target_demo_count counts recorded successful demos, so it requires --record.")
    _validate_lerobot_resume_dataset(args_cli)
    if args_cli.samples_per_pose <= 0:
        raise ValueError("--samples_per_pose must be a positive integer.")
    if args_cli.pose_jitter_xy < 0.0:
        raise ValueError("--pose_jitter_xy must be >= 0.")
    if args_cli.pose_jitter_yaw_deg < 0.0:
        raise ValueError("--pose_jitter_yaw_deg must be >= 0.")
    if args_cli.min_cup_distance < 0.0:
        raise ValueError("--min_cup_distance must be >= 0.")
    if args_cli.max_cup_distance < args_cli.min_cup_distance:
        raise ValueError("--max_cup_distance must be >= --min_cup_distance.")
    if args_cli.cup_workspace_x_range[0] >= args_cli.cup_workspace_x_range[1]:
        raise ValueError("--cup_workspace_x_range MIN must be less than MAX.")
    if args_cli.cup_workspace_y_range[0] >= args_cli.cup_workspace_y_range[1]:
        raise ValueError("--cup_workspace_y_range MIN must be less than MAX.")
    if not 0.0 <= args_cli.cup_pair_flip_prob <= 1.0:
        raise ValueError("--cup_pair_flip_prob must be in [0, 1].")
    if args_cli.cup_layout_profile in {"eval", "eval_wide"}:
        if task_name != "HCIS-CupStacking-SingleArm-v0":
            raise ValueError(
                "--cup_layout_profile=eval or eval_wide is only supported for HCIS-CupStacking-SingleArm-v0."
            )
        spawns = (
            CUP_STACKING_EVAL_SPAWNS
            if args_cli.cup_layout_profile == "eval"
            else CUP_STACKING_EVAL_WIDE_SPAWNS
        )
        print(f"[pose] using cup-stacking {args_cli.cup_layout_profile} start distribution:")
        for name, spawn in spawns.items():
            print(f"  [pose] {name}: x={spawn.x_range}, y={spawn.y_range}, z={spawn.center[2]:.3f}")
    if args_cli.sample_cup_layout:
        workspace_width = args_cli.cup_workspace_x_range[1] - args_cli.cup_workspace_x_range[0]
        workspace_depth = args_cli.cup_workspace_y_range[1] - args_cli.cup_workspace_y_range[0]
        if max(workspace_width, workspace_depth) < args_cli.min_cup_distance:
            raise ValueError("--sample_cup_layout workspace is too small for --min_cup_distance.")

    is_direct_env = "Direct" in task_name
    _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    _ensure_physics_sim_view(env)

    # disable gravity for every robot link prim
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record and not args_cli.use_lerobot_recorder:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # one-time state machine setup (e.g. FK calibration)
    sm = SMClass()
    sm.setup(env)
    env.reset()
    sm.reset()

    if args_cli.record and args_cli.use_lerobot_recorder:
        _replace_recorder_manager(env, env_cfg, args_cli)

    fall_check_object_names = tuple(getattr(sm, "task_object_names", ()))

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = _get_resume_recorded_demo_count(env, args_cli)
    current_recorded_demo_count = resume_recorded_demo_count

    if (
        args_cli.record
        and args_cli.target_demo_count is not None
        and current_recorded_demo_count >= args_cli.target_demo_count
    ):
        print(
            f"Existing dataset already has {current_recorded_demo_count} demonstrations, "
            f"which meets target {args_cli.target_demo_count}. Nothing to do."
        )
        env.close()
        simulation_app.close()
        return

    episode_seed_base = int(env_cfg.seed)
    attempted_episode_count = 0
    next_pose_idx = _pose_idx_for_attempt(args_cli, attempted_episode_count, len(episodes))
    next_sample_idx = _pose_sample_idx_for_attempt(args_cli, attempted_episode_count)
    print(
        f"[pose] using object_poses[{next_pose_idx}] "
        f"sample {next_sample_idx + 1}/{args_cli.samples_per_pose}"
    )
    _apply_episode_poses(env, episodes[next_pose_idx], args_cli, episode_seed_base + attempted_episode_count)
    attempted_episode_count += 1

    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    cnt = 1
    success_ID = []
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, device)

                if sm.is_episode_done:
                    (
                        next_pose_idx,
                        attempted_episode_count,
                        current_recorded_demo_count,
                        start_record_state,
                        should_break,
                        success,
                    ) = _on_episode_done(
                        env,
                        sm,
                        args_cli,
                        episodes,
                        next_pose_idx,
                        attempted_episode_count,
                        episode_seed_base,
                        resume_recorded_demo_count,
                        current_recorded_demo_count,
                        start_record_state,
                    )
                    if success:
                        progress_count, progress_total = _progress_counts(
                            args_cli,
                            current_recorded_demo_count,
                            attempted_episode_count,
                            _planned_pose_attempt_count(args_cli, len(episodes)),
                        )
                        print(f"\033[92m[Data Usage]{progress_count}/{progress_total} success.\033[0m")
                        success_ID.append(cnt)
                        cnt += 1
                    else:
                        progress_count, progress_total = _progress_counts(
                            args_cli,
                            current_recorded_demo_count,
                            attempted_episode_count,
                            _planned_pose_attempt_count(args_cli, len(episodes)),
                        )
                        print(f"\033[91m[Data Usage]{progress_count}/{progress_total} fail.\033[0m")
                    if should_break:
                        break
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                    if fall_check_object_names and _any_object_fell(
                        env, fall_check_object_names, _FALL_THRESHOLD_Z
                    ):
                        print(
                            "[INFO] Task object fell off the table; aborting this "
                            "episode and skipping to next."
                        )
                        sm._episode_done = True

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()
    
    print(success_ID)


if __name__ == "__main__":
    main()

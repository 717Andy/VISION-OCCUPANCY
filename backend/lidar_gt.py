"""Ego-frame nuScenes LIDAR_TOP returns used as occupancy ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nuscenes_loader import CAMERA_IDS, DATA_ROOT, load_synchronized_frame
from projection import EGO_BOUNDS, camera_to_ego, camera_visible_points

LIDAR_DIR = DATA_ROOT / "lidar"
# Store one point per cell so the clip stays small and still finer than the UI voxels.
STORE_VOXEL_M = 0.1
MAX_STORED_POINTS = 30_000


def lidar_path(frame_index: int) -> Path:
    return LIDAR_DIR / f"{int(frame_index):04d}.npy"


def camera_space_lidar(frame_index: int, points: np.ndarray | None = None) -> np.ndarray:
    """Lidar returns restricted to the surfaces the six cameras can see.

    When the frame has no calibration, the sweep is returned unchanged so a
    missing clip does not erase an explicitly supplied cloud.
    """
    cloud = load_lidar_ego_points(frame_index) if points is None else np.asarray(points, dtype=np.float32)
    cloud = cloud.reshape(-1, 3)
    if cloud.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        frame = load_synchronized_frame(int(frame_index))
    except Exception:
        return cloud.astype(np.float32, copy=False)
    cameras = [
        {
            "intrinsic": frame.cameras[camera_id].intrinsic,
            "rotation": frame.cameras[camera_id].rotation,
            "translation": frame.cameras[camera_id].translation,
        }
        for camera_id in CAMERA_IDS
    ]
    return camera_visible_points(cloud, cameras)


def load_lidar_ego_points(frame_index: int) -> np.ndarray:
    """Return Nx3 ego-frame lidar points, or an empty array when the clip has none."""
    path = lidar_path(frame_index)
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float32)
    points = np.load(path)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return points


def read_nuscenes_lidar_bin(path: Path) -> np.ndarray:
    """Load a nuScenes .pcd.bin sweep (x, y, z, intensity, ring) in the sensor frame."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if raw.size % 5 == 0:
        cols = 5
    elif raw.size % 4 == 0:
        cols = 4
    else:
        raise ValueError(f"Unexpected lidar width in {path} ({raw.size} floats)")
    return raw.reshape(-1, cols)[:, :3].astype(np.float64, copy=False)


def lidar_sensor_to_ego(
    points_sensor: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """calibrated_sensor maps lidar sensor coordinates into the ego frame."""
    return camera_to_ego(points_sensor, rotation_wxyz, translation)


def decimate_ego_points(
    points: np.ndarray,
    voxel_m: float = STORE_VOXEL_M,
    max_points: int = MAX_STORED_POINTS,
    bounds: np.ndarray = EGO_BOUNDS,
) -> np.ndarray:
    """Keep one point per coarse cell inside the occupancy bounds."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mins = bounds[:, 0]
    maxs = bounds[:, 1]
    mask = np.all((pts >= mins) & (pts <= maxs), axis=1)
    pts = pts[mask]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    size = float(max(voxel_m, 0.05))
    idx = np.floor((pts - mins) / size).astype(np.int32)
    _, unique_idx = np.unique(idx, axis=0, return_index=True)
    kept = pts[np.sort(unique_idx)]
    if kept.shape[0] > max_points:
        select = np.linspace(0, kept.shape[0] - 1, max_points).astype(np.int64)
        kept = kept[select]
    return kept.astype(np.float32)


def save_lidar_ego_points(frame_index: int, points_ego: np.ndarray) -> Path:
    path = lidar_path(frame_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = decimate_ego_points(points_ego)
    np.save(path, stored)
    return path

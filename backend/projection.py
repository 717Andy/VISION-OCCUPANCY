"""Project MiDaS relative depth into ego-frame voxels.

Pinhole unprojection follows P_3D = d · K^{-1} · [u, v, 1]^T, then the
camera-to-ego extrinsics stored with each nuScenes sample. Relative
disparity is mapped to metric depth with a scale-and-shift inverse-depth
fit onto the physical interval [0.5 m, 50 m].
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEAR_M = 0.5
FAR_M = 50.0
# nuScenes camera JPEGs are captured at 1600×900; prepared clip is 640×360.
ORIGINAL_IMAGE_SIZE = (1600, 900)

# Ego-frame crop used for occupancy (x forward, y left, z up).
EGO_BOUNDS = np.array(
    [
        [-15.0, 40.0],
        [-15.0, 15.0],
        [-1.0, 5.0],
    ],
    dtype=np.float64,
)

MAX_POINTS = 80_000
MAX_VOXELS = 4_000


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Scale a 3×3 pinhole K from one image resolution to another."""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    k[0, :] *= sx
    k[1, :] *= sy
    return k


def disparity_to_metric_depth(
    disparity: np.ndarray,
    near: float = NEAR_M,
    far: float = FAR_M,
) -> np.ndarray:
    """Scale-and-shift relative disparity onto metric depth in [near, far].

    MiDaS_small emits inverse-depth-like values (larger = closer). After
    min-max normalizing to relative disparity r ∈ [0, 1]:

        d = 1 / (r · (1/near − 1/far) + 1/far)

    so r=1 maps to `near` and r=0 maps to `far`.
    """
    disp = np.asarray(disparity, dtype=np.float32)
    dmin = float(np.min(disp))
    dmax = float(np.max(disp))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax - dmin < 1e-6:
        mid = np.float32((near + far) / 2.0)
        return np.full(disp.shape, mid, dtype=np.float32)

    relative = (disp - np.float32(dmin)) / np.float32(dmax - dmin)
    inv_near = 1.0 / near
    inv_far = 1.0 / far
    metric = 1.0 / (relative * (inv_near - inv_far) + inv_far)
    return np.clip(metric, near, far).astype(np.float32)


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 3,
) -> np.ndarray:
    """Vectorized P_3D = d · K^{-1} · [u, v, 1]^T in the camera optical frame."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be a 2D array, got shape {depth.shape}")
    height, width = depth.shape
    step = max(int(stride), 1)

    us = np.arange(0, width, step, dtype=np.float64) + 0.5
    vs = np.arange(0, height, step, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(us, vs)
    dd = depth[::step, ::step].astype(np.float64)
    valid = np.isfinite(dd) & (dd > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    pix = np.stack(
        [uu[valid], vv[valid], np.ones(np.count_nonzero(valid), dtype=np.float64)],
        axis=0,
    )
    k_inv = np.linalg.inv(np.asarray(K, dtype=np.float64).reshape(3, 3))
    rays = k_inv @ pix
    points = (dd[valid] * rays).T
    if points.shape[0] > MAX_POINTS:
        idx = np.linspace(0, points.shape[0] - 1, MAX_POINTS).astype(np.int64)
        points = points[idx]
    return points


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3×3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def camera_to_ego(points_cam: np.ndarray, rotation_wxyz: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Transform camera-optical points into the nuScenes ego frame.

    Ego is x-forward, y-left, z-up. Camera optical is x-right, y-down, z-forward.
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    rot = quat_to_rotmat(rotation_wxyz)
    trans = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return (rot @ pts.T).T + trans


def clip_to_bounds(points: np.ndarray, bounds: np.ndarray = EGO_BOUNDS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    mins = bounds[:, 0]
    maxs = bounds[:, 1]
    mask = np.all((pts >= mins) & (pts <= maxs), axis=1)
    return pts[mask]


def _cap_voxels(centers: np.ndarray, occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if centers.shape[0] <= MAX_VOXELS:
        return centers, occupancy
    # Spatial stride keeps surround coverage instead of only the densest cells.
    select = np.linspace(0, centers.shape[0] - 1, MAX_VOXELS).astype(np.int64)
    return centers[select], occupancy[select]


def _voxelize_numpy(
    points: np.ndarray,
    voxel_size: float,
    occupancy_threshold: float,
    bounds: np.ndarray,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    origin = bounds[:, 0] if origin is None else np.asarray(origin, dtype=np.float64)
    idx = np.floor((pts - origin) / voxel_size).astype(np.int32)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    ref = max(float(np.percentile(counts, 75)), 3.0)
    occupancy = np.clip(counts.astype(np.float32) / ref, 0.0, 1.0)
    keep = occupancy >= np.float32(occupancy_threshold)
    kept = uniq[keep]
    occ = occupancy[keep]
    centers = (origin + (kept.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)
    return _cap_voxels(centers, occ)


def voxelize_occupancy(
    points: np.ndarray,
    voxel_size: float,
    occupancy_threshold: float = 0.38,
    bounds: np.ndarray = EGO_BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a metric point cloud into a discrete occupancy grid.

    Occupied cells are histogrammed in NumPy, then instantiated as an Open3D
    `VoxelGrid` so the live path matches the depth-to-voxel spatial projection.
    """
    size = float(max(voxel_size, 0.05))
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    origin = pts.min(axis=0)
    centers, occ = _voxelize_numpy(pts, size, occupancy_threshold, bounds, origin=origin)
    if centers.shape[0] == 0:
        return centers, occ

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
        o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)
    except Exception as exc:  # pragma: no cover - Open3D optional at import time
        logger.warning("Open3D voxelization unavailable (%s); using NumPy grid", exc)
    return centers, occ


def pack_occupancy(centers: np.ndarray, occupancy: np.ndarray) -> bytes:
    """Binary protocol: uint32 count + float32 x,y,z,prob per voxel."""
    packed = np.column_stack(
        (
            np.asarray(centers, dtype=np.float32).reshape(-1, 3),
            np.asarray(occupancy, dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32, copy=False)
    header = np.array([packed.shape[0]], dtype=np.uint32)
    return header.tobytes() + packed.tobytes()


def project_cameras_to_voxels(
    camera_payloads: list[dict[str, Any]],
    voxel_size: float,
    occupancy_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-camera metric depth into one ego-frame occupancy grid.

    Each payload must contain:
      disparity: HxW relative depth from MiDaS
      intrinsic: 3×3 K at ORIGINAL_IMAGE_SIZE
      translation, rotation: nuScenes calibrated_sensor extrinsics
    """
    clouds: list[np.ndarray] = []
    for payload in camera_payloads:
        disparity = np.asarray(payload["disparity"], dtype=np.float32)
        height, width = disparity.shape[:2]
        k_depth = scale_intrinsics(
            payload["intrinsic"],
            ORIGINAL_IMAGE_SIZE,
            (width, height),
        )
        metric = disparity_to_metric_depth(disparity)
        cam_pts = unproject_depth(metric, k_depth)
        ego_pts = camera_to_ego(cam_pts, payload["rotation"], payload["translation"])
        clouds.append(ego_pts)

    if not clouds:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    merged = np.concatenate(clouds, axis=0)
    return voxelize_occupancy(merged, voxel_size, occupancy_threshold)

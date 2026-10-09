"""Ground-plane inverse perspective mapping.

Each camera ray is intersected with the ego plane z = 0. A ground cell is
occupied only where the image has a strong vertical edge, so a flat road is
not painted solid. There is no learned depth and lidar is not an input.
"""

from __future__ import annotations

import numpy as np

from nuscenes_loader import CAMERA_IDS, ORIGINAL_IMAGE_SIZE, load_rgb, load_synchronized_frame
from projection import cell_centers, quat_to_rotmat, scale_intrinsics

IPM_NAME = "Ground-plane IPM"
IPM_ID = "ipm-ground"
_STRIDE = 4
_GRADIENT_QUANTILE = 0.92
_MIN_GRADIENT = 12.0


def ipm_points(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    *,
    stride: int = _STRIDE,
    gradient_quantile: float = _GRADIENT_QUANTILE,
    source_size: tuple[int, int] = ORIGINAL_IMAGE_SIZE,
) -> np.ndarray:
    """Ego points where vertical image edges meet the ground plane."""
    rgb = np.asarray(image)
    if rgb.ndim == 3:
        gray = rgb.mean(axis=2)
    else:
        gray = rgb
    gray = np.asarray(gray, dtype=np.float64)
    if gray.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    height, width = gray.shape
    gradient = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    if float(gradient.max()) < _MIN_GRADIENT:
        return np.zeros((0, 3), dtype=np.float32)
    cutoff = max(float(np.quantile(gradient, gradient_quantile)), _MIN_GRADIENT)
    rows, cols = np.nonzero(gradient >= cutoff)
    step = max(int(stride), 1)
    rows = rows[::step]
    cols = cols[::step]
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    k = scale_intrinsics(intrinsic, source_size, (width, height))
    rays = np.linalg.inv(k) @ np.stack((cols, rows, np.ones(cols.shape[0])), axis=0)
    rotation = quat_to_rotmat(rotation_wxyz)
    direction = (rotation @ rays).T
    origin = np.asarray(translation, dtype=np.float64).reshape(3)
    denom = direction[:, 2]
    scale = np.divide(-origin[2], denom, out=np.full(denom.shape, np.nan), where=np.abs(denom) > 1e-4)
    valid = np.isfinite(scale) & (scale > 0.5)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    points = origin.reshape(1, 3) + scale[valid, None] * direction[valid]
    return points.astype(np.float32)


def predict_points(frame_index: int) -> np.ndarray:
    """Raw ground-plane hits for one synchronized frame."""
    synced = load_synchronized_frame(int(frame_index))
    clouds = []
    for camera_id in CAMERA_IDS:
        sample = synced.cameras[camera_id]
        clouds.append(
            ipm_points(
                load_rgb(sample.image_path),
                sample.intrinsic,
                sample.rotation,
                sample.translation,
            )
        )
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)


_CENTER_CACHE: dict[tuple[int, float], np.ndarray] = {}


def predict_centers(frame_index: int, voxel_m: float = 1.0) -> np.ndarray:
    """Occupied ground cells. `voxel_m` is the grid used for this call."""
    key = (int(frame_index), round(float(voxel_m), 3))
    cached = _CENTER_CACHE.get(key)
    if cached is not None:
        return cached
    centers = cell_centers(predict_points(key[0]), key[1])
    _CENTER_CACHE[key] = centers
    return centers

"""Project MiDaS relative depth into ego-frame voxels.

Pinhole unprojection follows P_3D = d · K^{-1} · [u, v, 1]^T, then the
camera-to-ego extrinsics stored with each nuScenes sample. Relative
disparity is an inverse-depth signal. Each camera is scaled so rays that
see the road meet the ego ground plane (z = 0) at the calibrated camera
height. A fixed [0.5 m, 50 m] stretch remains as the fallback when that
fit is unavailable.
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
# Dense "sensor" occupancy for Split GT (same fused cloud, lower keep-threshold).
GT_OCCUPANCY_SCALE = 0.55
GT_OCCUPANCY_FLOOR = 0.05


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Scale a 3×3 pinhole K from one image resolution to another."""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    k[0, :] *= sx
    k[1, :] *= sy
    return k


def _fit_inverse_depth(disparity: np.ndarray, inv_depth: np.ndarray) -> tuple[float, float] | None:
    """Least-squares 1/d = a · disparity + b, refit on relative-depth inliers."""

    def solve(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        design = np.column_stack((xs, np.ones(xs.shape[0], dtype=np.float64)))
        slope, intercept = np.linalg.lstsq(design, ys, rcond=None)[0]
        return float(slope), float(intercept)

    slope, intercept = solve(disparity, inv_depth)
    if not np.isfinite(slope) or slope <= 1e-8:
        return None
    predicted = slope * disparity + intercept
    depth_hat = 1.0 / np.clip(predicted, 1e-6, None)
    depth_true = 1.0 / inv_depth
    inliers = np.abs(depth_hat - depth_true) / depth_true < 0.25
    needed = max(32, int(0.2 * disparity.shape[0]))
    if int(np.count_nonzero(inliers)) < needed:
        return None
    slope, intercept = solve(disparity[inliers], inv_depth[inliers])
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 1e-8:
        return None
    return slope, intercept


def _apply_inverse_depth(disparity: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    inverse = slope * np.asarray(disparity, dtype=np.float64) + intercept
    depth = np.full(inverse.shape, FAR_M, dtype=np.float64)
    valid = np.isfinite(inverse) & (inverse > 1.0 / (FAR_M * 1.5))
    depth[valid] = 1.0 / inverse[valid]
    return np.clip(depth, NEAR_M, FAR_M).astype(np.float32)


def metric_depth_from_ground(
    disparity: np.ndarray,
    K: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    z_ground: float = 0.0,
) -> np.ndarray:
    """Scale relative disparity so road rays intersect the ego ground plane.

    For a pixel whose ray points downward, the metric depth that lands on
    z = `z_ground` is fixed by the camera height. MiDaS only supplies the
    inverse-depth shape: 1/d = a · disparity + b, fit on the lower image
    and applied to the whole frame. Objects closer than the road stay
    closer, so they lift above the plane. Falls back to the fixed
    [near, far] stretch when the camera has no usable road samples.
    """
    disp = np.asarray(disparity, dtype=np.float64)
    if disp.ndim != 2:
        raise ValueError(f"Disparity must be a 2D array, got shape {disp.shape}")
    fallback = disparity_to_metric_depth(disp)
    cam_z = float(np.asarray(translation, dtype=np.float64).reshape(3)[2])
    if cam_z < 0.3:
        return fallback

    height, width = disp.shape
    stride = 8
    us = np.arange(0, width, stride, dtype=np.float64) + 0.5
    vs = np.arange(0, height, stride, dtype=np.float64) + 0.5
    if us.size == 0 or vs.size == 0:
        return fallback
    uu, vv = np.meshgrid(us, vs)
    samples = disp[::stride, ::stride][: uu.shape[0], : uu.shape[1]]
    pixels = np.stack(
        (uu.ravel(), vv.ravel(), np.ones(uu.size, dtype=np.float64)),
        axis=0,
    )
    k_inv = np.linalg.inv(np.asarray(K, dtype=np.float64).reshape(3, 3))
    ray_z = (quat_to_rotmat(rotation_wxyz) @ (k_inv @ pixels))[2]
    safe_z = np.where(np.abs(ray_z) < 1e-4, np.nan, ray_z)
    ground_depth = (float(z_ground) - cam_z) / safe_z
    flat = samples.ravel()
    road = (
        (ray_z < -0.05)
        & (ground_depth > 4.0)
        & (ground_depth < 30.0)
        & (vv.ravel() > height * 0.55)
        & np.isfinite(flat)
        & (flat > 0.0)
    )
    if int(np.count_nonzero(road)) < 32:
        return fallback

    fit = _fit_inverse_depth(flat[road], 1.0 / ground_depth[road])
    if fit is None:
        return fallback
    return _apply_inverse_depth(disp, fit[0], fit[1])


def disparity_to_metric_depth(
    disparity: np.ndarray,
    near: float = NEAR_M,
    far: float = FAR_M,
) -> np.ndarray:
    """Scale-and-shift relative disparity onto metric depth in [near, far].

    MiDaS_small emits inverse-depth-like values (larger = closer). After
    min-max normalizing to relative disparity r ∈ [0, 1]:

        d = 1 / (r · (1/near − 1/far) + 1/far)

    so r=1 maps to `near` and r=0 maps to `far`. This ignores camera height;
    `metric_depth_from_ground` is the live path.
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


# Angular bin for the camera z-buffer. Eight native pixels is about one
# occupancy cell across the nuScenes depth range.
VISIBLE_PIXEL_BIN = 8
VISIBLE_SURFACE_M = 0.4
VISIBLE_MIN_DEPTH_M = 0.5


def camera_visible_points(
    points_ego: np.ndarray,
    cameras: list[dict[str, Any]],
    image_size: tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    pixel_bin: int = VISIBLE_PIXEL_BIN,
    surface_m: float = VISIBLE_SURFACE_M,
) -> np.ndarray:
    """Keep lidar returns that are the first surface in the camera images.

    Each camera keeps points that land inside its image and sit within
    `surface_m` of the nearest return in that angular bin. Returns outside
    every frame, behind the cameras, or hidden behind a closer surface are
    dropped. The result is the lidar measurement of the space the feeds see.
    """
    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0 or not cameras:
        return pts.astype(np.float32, copy=False)

    width, height = int(image_size[0]), int(image_size[1])
    bin_size = max(int(pixel_bin), 1)
    grid_w = (width + bin_size - 1) // bin_size
    grid_h = (height + bin_size - 1) // bin_size
    visible = np.zeros(pts.shape[0], dtype=bool)
    thickness = float(max(surface_m, 0.0))

    for camera in cameras:
        intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64).reshape(3, 3)
        rotation = quat_to_rotmat(np.asarray(camera["rotation"], dtype=np.float64))
        translation = np.asarray(camera["translation"], dtype=np.float64).reshape(1, 3)
        cam = (pts - translation) @ rotation
        depth = cam[:, 2]
        in_front = depth > VISIBLE_MIN_DEPTH_M
        if not np.any(in_front):
            continue
        pix = cam @ intrinsic.T
        u = np.divide(pix[:, 0], depth, out=np.full(depth.shape, -1.0), where=in_front)
        v = np.divide(pix[:, 1], depth, out=np.full(depth.shape, -1.0), where=in_front)
        inside = in_front & (u >= 0.0) & (v >= 0.0) & (u < width) & (v < height)
        if not np.any(inside):
            continue
        ui = np.minimum((u[inside] / bin_size).astype(np.int64), grid_w - 1)
        vi = np.minimum((v[inside] / bin_size).astype(np.int64), grid_h - 1)
        flat = vi * grid_w + ui
        zbuf = np.full(grid_h * grid_w, np.inf, dtype=np.float64)
        np.minimum.at(zbuf, flat, depth[inside])
        on_surface = depth[inside] <= zbuf[flat] + thickness
        visible[np.flatnonzero(inside)[on_surface]] = True

    return pts[visible].astype(np.float32)


def _points_visible_to_camera(
    points_ego: np.ndarray,
    camera: dict[str, Any],
    image_size: tuple[int, int] = ORIGINAL_IMAGE_SIZE,
    pixel_bin: int = VISIBLE_PIXEL_BIN,
    surface_m: float = VISIBLE_SURFACE_M,
) -> np.ndarray:
    """Returns from `points_ego` that form this camera's nearest surface."""
    kept = camera_visible_points(points_ego, [camera], image_size, pixel_bin, surface_m)
    return kept


def _cell_keys_from_points(
    points: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> set[tuple[int, int, int]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return set()
    idx = np.floor((pts - origin) / float(voxel_size)).astype(np.int64)
    return {tuple(int(v) for v in row) for row in idx.tolist()}


def _centers_from_keys(
    keys: set[tuple[int, int, int]],
    voxel_size: float,
    origin: np.ndarray,
) -> np.ndarray:
    if not keys:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.array(sorted(keys), dtype=np.float64)
    return (origin + (idx + 0.5) * float(voxel_size)).astype(np.float32)


def _free_keys_along_rays(
    hits: np.ndarray,
    camera_xyz: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    bounds: np.ndarray,
    occupied: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Cells a camera ray crosses before it reaches a measured hit."""
    pts = np.asarray(hits, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return set()
    cam = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    size = float(voxel_size)
    step = size * 0.5
    mins = np.asarray(bounds[:, 0], dtype=np.float64)
    maxs = np.asarray(bounds[:, 1], dtype=np.float64)
    free: set[tuple[int, int, int]] = set()
    for start in range(0, pts.shape[0], 256):
        chunk = pts[start : start + 256]
        delta = chunk - cam
        dist = np.linalg.norm(delta, axis=1)
        usable = dist > step
        if not np.any(usable):
            continue
        max_steps = int(np.max(dist[usable]) / step)
        if max_steps <= 1:
            continue
        distances = (np.arange(1, max_steps, dtype=np.float64) * step)[None, :]
        valid = distances < (dist[:, None] - step * 0.25)
        samples = cam + (distances / dist[:, None])[..., None] * delta[:, None, :]
        flat = samples[valid]
        if flat.shape[0] == 0:
            continue
        inside = np.all((flat >= mins) & (flat <= maxs), axis=1)
        flat = flat[inside]
        if flat.shape[0] == 0:
            continue
        idx = np.floor((flat - origin) / size).astype(np.int64)
        uniq = np.unique(idx, axis=0)
        for row in uniq.tolist():
            key = (int(row[0]), int(row[1]), int(row[2]))
            if key not in occupied:
                free.add(key)
    return free


def camera_known_space(
    points_ego: np.ndarray,
    cameras: list[dict[str, Any]] | None,
    voxel_size: float,
    origin: np.ndarray | None = None,
    bounds: np.ndarray = EGO_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupied hits, free cells on the way to those hits, and occupancy weights.

    Unknown space is everything else. Without camera calibration the occupied
    cloud is the lidar sweep and free space is empty, so callers can fall back
    to treating unlabeled cells as empty.
    """
    origin_vec = EGO_BOUNDS[:, 0] if origin is None else np.asarray(origin, dtype=np.float64)
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    if cameras:
        visible = camera_visible_points(points, cameras)
    else:
        visible = points.astype(np.float32, copy=False)
    occupied, weights = voxelize_lidar(visible, voxel_size, bounds, cap=False)
    if not cameras or occupied.shape[0] == 0:
        return occupied, np.zeros((0, 3), dtype=np.float32), weights
    occupied_keys = _cell_keys_from_points(occupied, voxel_size, origin_vec)
    free: set[tuple[int, int, int]] = set()
    for camera in cameras:
        hits = _points_visible_to_camera(points, camera)
        free |= _free_keys_along_rays(
            hits,
            np.asarray(camera["translation"], dtype=np.float64),
            voxel_size,
            origin_vec,
            bounds,
            occupied_keys,
        )
    free -= occupied_keys
    return occupied, _centers_from_keys(free, voxel_size, origin_vec), weights


def _keys_near(
    source: set[tuple[int, int, int]],
    target: set[tuple[int, int, int]],
    radius: int,
) -> set[tuple[int, int, int]]:
    if radius <= 0:
        return source & target
    matched: set[tuple[int, int, int]] = set()
    for x, y, z in source:
        found = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if (x + dx, y + dy, z + dz) in target:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            matched.add((x, y, z))
    return matched


def known_space_counts(
    pred_centers: np.ndarray,
    occupied_centers: np.ndarray,
    free_centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
    tolerance: int = 0,
    unlabeled_is_free: bool = False,
) -> dict[str, Any]:
    """IoU on cells labeled occupied or free. Unknown predictions are ignored.

    A predicted cell counts only when that exact cell is known. Tolerance is a
    Chebyshev radius used to match a known prediction to an occupied cell, so a
    one-bin miss can still be a true positive. mIoU is the mean of that IoU
    over height bands, which are not semantic classes.
    """
    size = float(max(voxel_size, 0.05))
    origin_vec = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    radius = max(int(tolerance), 0)
    pred = _cell_keys_from_points(pred_centers, size, origin_vec)
    occupied = _cell_keys_from_points(occupied_centers, size, origin_vec)
    free = _cell_keys_from_points(free_centers, size, origin_vec) - occupied
    if unlabeled_is_free:
        pred_known = pred
        false_positive_pool = pred - occupied
    else:
        known = occupied | free
        pred_known = pred & known
        false_positive_pool = pred_known
    true_positive_keys = _keys_near(pred_known, occupied, radius)
    false_positive_keys = false_positive_pool - true_positive_keys
    false_negative_keys = occupied - _keys_near(occupied, pred_known, radius)
    binary = _iou_from_counts(len(true_positive_keys), len(false_positive_keys), len(false_negative_keys))

    bands: list[dict[str, Any]] = []
    band_ious: list[float] = []
    for label, class_name in HEIGHT_BANDS:
        def in_band(key: tuple[int, int, int], name: str = class_name) -> bool:
            return _class_of_key(key, origin_vec, size) == name

        tp = sum(1 for key in true_positive_keys if in_band(key))
        fp = sum(1 for key in false_positive_keys if in_band(key))
        fn = sum(1 for key in false_negative_keys if in_band(key))
        if tp + fp + fn == 0:
            continue
        iou = _iou_from_counts(tp, fp, fn)
        band_ious.append(iou)
        bands.append({"band": label, "tp": tp, "fp": fp, "fn": fn, "iou": iou})

    distance_zones, miou_drop = _distance_decay(
        true_positive_keys, false_positive_keys, false_negative_keys, origin_vec, size
    )
    return {
        "tp": len(true_positive_keys),
        "fp": len(false_positive_keys),
        "fn": len(false_negative_keys),
        "iou": binary,
        "miou": float(sum(band_ious) / len(band_ious)) if band_ious else binary,
        "bands": bands,
        "distance_zones": distance_zones,
        "miou_drop": miou_drop,
        "tolerance_cells": radius,
        "grid_m": size,
        "unknown": "ignored",
    }


def _horizontal_range(key: tuple[int, int, int], origin: np.ndarray, voxel_size: float) -> float:
    x = float(origin[0] + (key[0] + 0.5) * voxel_size)
    y = float(origin[1] + (key[1] + 0.5) * voxel_size)
    return float(np.hypot(x, y))


def _zone_miou(
    true_positive: set[tuple[int, int, int]],
    false_positive: set[tuple[int, int, int]],
    false_negative: set[tuple[int, int, int]],
    origin: np.ndarray,
    voxel_size: float,
) -> float | None:
    if not true_positive and not false_positive and not false_negative:
        return None
    ious: list[float] = []
    for _label, class_name in HEIGHT_BANDS:
        def in_band(key: tuple[int, int, int], name: str = class_name) -> bool:
            return _class_of_key(key, origin, voxel_size) == name

        tp = sum(1 for key in true_positive if in_band(key))
        fp = sum(1 for key in false_positive if in_band(key))
        fn = sum(1 for key in false_negative if in_band(key))
        if tp + fp + fn:
            ious.append(_iou_from_counts(tp, fp, fn))
    if not ious:
        return _iou_from_counts(len(true_positive), len(false_positive), len(false_negative))
    return float(sum(ious) / len(ious))


def _distance_decay(
    true_positive: set[tuple[int, int, int]],
    false_positive: set[tuple[int, int, int]],
    false_negative: set[tuple[int, int, int]],
    origin: np.ndarray,
    voxel_size: float,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Height-band mIoU inside 0–10 m, 10–25 m, and 25 m+, plus the drop between them."""
    zones: list[dict[str, Any]] = []
    by_zone: dict[str, float | None] = {}
    for name, label, low, high in DISTANCE_ZONES:
        def in_zone(key: tuple[int, int, int], lo: float = low, hi: float = high) -> bool:
            distance = _horizontal_range(key, origin, voxel_size)
            return lo <= distance < hi

        tp = {key for key in true_positive if in_zone(key)}
        fp = {key for key in false_positive if in_zone(key)}
        fn = {key for key in false_negative if in_zone(key)}
        miou = _zone_miou(tp, fp, fn, origin, voxel_size)
        zones.append(
            {
                "zone": name,
                "label": label,
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
                "miou": miou,
                "error": None if miou is None else float(1.0 - miou),
            }
        )
        by_zone[name] = miou

    def drop(start: str, end: str) -> float | None:
        left = by_zone[start]
        right = by_zone[end]
        if left is None or right is None:
            return None
        return float(left - right)

    return zones, {
        "near_to_mid": drop("near", "mid"),
        "mid_to_far": drop("mid", "far"),
        "near_to_far": drop("near", "far"),
    }


def known_space_error_voxels(
    pred_centers: np.ndarray,
    occupied_centers: np.ndarray,
    free_centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
    *,
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact known-cell mistakes: predicted free cells, and missed occupied cells."""
    counts_origin = EGO_BOUNDS[:, 0] if origin is None else np.asarray(origin, dtype=np.float64)
    size = float(max(voxel_size, 0.05))
    pred = _cell_keys_from_points(pred_centers, size, counts_origin)
    occupied = _cell_keys_from_points(occupied_centers, size, counts_origin)
    free = _cell_keys_from_points(free_centers, size, counts_origin) - occupied
    mistakes = (pred & free) | (occupied - pred)
    centers = _centers_from_keys(mistakes, size, counts_origin)
    occupancy = np.ones((centers.shape[0],), dtype=np.float32)
    if cap:
        return _cap_voxels(centers, occupancy)
    return centers, occupancy


def display_label_clouds(
    occupied: np.ndarray,
    free: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
    bounds: np.ndarray = EGO_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capped free and unknown clouds for the viewer. Scoring does not use these."""
    free_occ = np.ones((free.shape[0],), dtype=np.float32)
    free_centers, free_occ = _cap_voxels(free, free_occ)
    unknown = _unknown_display_centers(occupied, free, voxel_size, origin, bounds)
    unknown_occ = np.ones((unknown.shape[0],), dtype=np.float32)
    return free_centers, free_occ, unknown, unknown_occ


def _unknown_display_centers(
    occupied: np.ndarray,
    free: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None,
    bounds: np.ndarray,
) -> np.ndarray:
    size = float(max(voxel_size, 0.05))
    origin_vec = np.asarray(bounds[:, 0] if origin is None else origin, dtype=np.float64)
    counts = [max(int(np.floor((hi - lo) / size)), 0) for lo, hi in bounds]
    nx, ny, nz = counts
    del nx
    total = ny * nz * counts[0]
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    taken = _cell_keys_from_points(occupied, size, origin_vec) | _cell_keys_from_points(
        free, size, origin_vec
    )
    stride = max(total // (MAX_VOXELS * 4), 1)
    chosen = np.arange(0, total, stride, dtype=np.int64)
    plane = ny * nz
    ix = chosen // plane
    rem = chosen % plane
    iy = rem // nz
    iz = rem % nz
    keep_x: list[int] = []
    keep_y: list[int] = []
    keep_z: list[int] = []
    for x, y, z in zip(ix.tolist(), iy.tolist(), iz.tolist()):
        if (int(x), int(y), int(z)) not in taken:
            keep_x.append(int(x))
            keep_y.append(int(y))
            keep_z.append(int(z))
    if not keep_x:
        return np.zeros((0, 3), dtype=np.float32)
    xs = np.array(keep_x)
    ys = np.array(keep_y)
    zs = np.array(keep_z)
    if xs.size > MAX_VOXELS:
        select = np.linspace(0, xs.size - 1, MAX_VOXELS).astype(np.int64)
        xs, ys, zs = xs[select], ys[select], zs[select]
    centers = np.column_stack(
        (
            origin_vec[0] + (xs.astype(np.float64) + 0.5) * size,
            origin_vec[1] + (ys.astype(np.float64) + 0.5) * size,
            origin_vec[2] + (zs.astype(np.float64) + 0.5) * size,
        )
    )
    return centers.astype(np.float32)


def cell_centers(points: np.ndarray, voxel_size: float, bounds: np.ndarray = EGO_BOUNDS) -> np.ndarray:
    """Unique occupied cell centers. A cell is occupied when it contains any point."""
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    origin = np.asarray(bounds[:, 0], dtype=np.float64)
    size = float(max(voxel_size, 0.05))
    idx = np.floor((pts - origin) / size).astype(np.int64)
    span = np.floor((bounds[:, 1] - origin) / size).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < span), axis=1)
    idx = idx[inside]
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    uniq = np.unique(idx, axis=0)
    return (origin + (uniq.astype(np.float64) + 0.5) * size).astype(np.float32)


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
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    origin = bounds[:, 0] if origin is None else np.asarray(origin, dtype=np.float64)
    idx = np.floor((pts - origin) / voxel_size).astype(np.int32)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    ref = max(float(np.percentile(counts, 75)), 3.0)
    occupancy = np.clip(counts.astype(np.float32) / ref, 0.0, 1.0)
    centers_all = (origin + (uniq.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)
    # The road is much denser than cars, so a percentile threshold deletes
    # the elevated surface. Image stride often leaves one hit in an object
    # cell, same as a lidar return, so keep every raised cell.
    elevated = centers_all[:, 2] >= np.float32(0.5)
    keep = (occupancy >= np.float32(occupancy_threshold)) | elevated
    kept = uniq[keep]
    occ = occupancy[keep]
    centers = (origin + (kept.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)
    if cap:
        return _cap_voxels(centers, occ)
    return centers, occ


def voxelize_occupancy(
    points: np.ndarray,
    voxel_size: float,
    occupancy_threshold: float = 0.38,
    bounds: np.ndarray = EGO_BOUNDS,
    *,
    origin: np.ndarray | None = None,
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a metric point cloud into a discrete occupancy grid.

    Occupied cells are histogrammed in NumPy, then instantiated as an Open3D
    `VoxelGrid` so the live path matches the depth-to-voxel spatial projection.
    """
    size = float(max(voxel_size, 0.05))
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    grid_origin = pts.min(axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    centers, occ = _voxelize_numpy(
        pts, size, occupancy_threshold, bounds, origin=grid_origin, cap=cap
    )
    if cap:
        _touch_open3d(centers, size)
    return centers, occ


def voxelize_lidar(
    points: np.ndarray,
    voxel_size: float,
    bounds: np.ndarray = EGO_BOUNDS,
    *,
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize ego-frame lidar returns. A cell is occupied if it contains a return.

    This grid is independent of the camera depth cloud and ignores the vision
    occupancy slider so Ground Truth stays the sensor measurement.
    """
    size = float(max(voxel_size, 0.05))
    pts = clip_to_bounds(points, bounds)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    origin = bounds[:, 0]
    idx = np.floor((pts - origin) / size).astype(np.int32)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    ref = max(float(np.percentile(counts, 90)), 1.0)
    occupancy = np.clip(counts.astype(np.float32) / ref, 0.0, 1.0)
    centers = (origin + (uniq.astype(np.float64) + 0.5) * size).astype(np.float32)
    if cap:
        centers, occupancy = _cap_voxels(centers, occupancy)
        _touch_open3d(centers, size)
    return centers, occupancy


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


def _cell_index_map(
    centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> dict[tuple[int, int, int], np.ndarray]:
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return {}
    idx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    uniq, first = np.unique(idx, axis=0, return_index=True)
    return {tuple(int(v) for v in ijk): pts[i].astype(np.float32) for ijk, i in zip(uniq, first)}


def discrepancy_voxels(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
    *,
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cells occupied in exactly one of the two grids.

    Indices use the same origin and voxel size as `grid_miou`, so a cell that
    both matrices occupy is not a discrepancy even if the float centers differ
    by a rounding error.
    """
    size = float(max(voxel_size, 0.05))
    origin_vec = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    pred = _cell_index_map(pred_centers, size, origin_vec)
    gt = _cell_index_map(gt_centers, size, origin_vec)
    only = (pred.keys() - gt.keys()) | (gt.keys() - pred.keys())
    if not only:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    centers = np.stack([pred[key] if key in pred else gt[key] for key in sorted(only)]).astype(np.float32)
    occupancy = np.ones((centers.shape[0],), dtype=np.float32)
    if cap:
        return _cap_voxels(centers, occupancy)
    return centers, occupancy


def pack_occupancy_pair(
    pred_centers: np.ndarray,
    pred_occupancy: np.ndarray,
    gt_centers: np.ndarray,
    gt_occupancy: np.ndarray,
    err_centers: np.ndarray | None = None,
    err_occupancy: np.ndarray | None = None,
    free_centers: np.ndarray | None = None,
    free_occupancy: np.ndarray | None = None,
    unknown_centers: np.ndarray | None = None,
    unknown_occupancy: np.ndarray | None = None,
) -> bytes:
    """uint32 pred, gt, discrepancy, free, and unknown counts, then those grids."""
    blocks = [
        (pred_centers, pred_occupancy),
        (gt_centers, gt_occupancy),
        (err_centers, err_occupancy),
        (free_centers, free_occupancy),
        (unknown_centers, unknown_occupancy),
    ]
    packed = []
    counts = []
    for centers, occupancy in blocks:
        if centers is None or occupancy is None:
            centers = np.zeros((0, 3), dtype=np.float32)
            occupancy = np.zeros((0,), dtype=np.float32)
        blob = pack_occupancy(centers, occupancy)
        counts.append(int(np.frombuffer(blob[:4], dtype=np.uint32)[0]))
        packed.append(blob[4:])
    header = np.array(counts, dtype=np.uint32)
    return header.tobytes() + b"".join(packed)


def grid_miou(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
    tolerance: int = 0,
) -> float:
    """Occupancy IoU on quantized voxel indices.

    `tolerance` is a Chebyshev radius in cells. A depth surface and a
    single-sweep lidar shell often miss by one 0.2 m bin; radius 1 counts
    that neighbor as overlap. Radius 0 is exact cell equality.
    """
    size = float(max(voxel_size, 0.05))
    origin_vec = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    radius = max(int(tolerance), 0)

    def keys(centers: np.ndarray) -> set[tuple[int, int, int]]:
        pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return set()
        idx = np.floor((pts - origin_vec) / size).astype(np.int64)
        return {tuple(int(v) for v in row) for row in idx.tolist()}

    pred_keys = keys(pred_centers)
    gt_keys = keys(gt_centers)
    if not pred_keys and not gt_keys:
        return 1.0
    if radius == 0:
        union = pred_keys | gt_keys
        if not union:
            return 1.0
        return float(len(pred_keys & gt_keys) / len(union))

    def within(source: set[tuple[int, int, int]], target: set[tuple[int, int, int]]) -> int:
        matched = 0
        for x, y, z in source:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if (x + dx, y + dy, z + dz) in target:
                            matched += 1
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        return matched

    true_positive = within(pred_keys, gt_keys)
    false_negative = len(gt_keys) - within(gt_keys, pred_keys)
    false_positive = len(pred_keys) - true_positive
    denom = true_positive + false_positive + false_negative
    if denom == 0:
        return 1.0
    return float(true_positive / denom)


# Height bands match the viewer: z < 0.45 driveable, z < 2.3 vehicle, else pedestrian.
SEMANTIC_CLASSES = ("driveable", "vehicle", "pedestrian")
# Display names for the same z cuts. These are height bands, not object classes.
HEIGHT_BANDS = (
    ("below 0.45 m", "driveable"),
    ("0.45 to 2.3 m", "vehicle"),
    ("above 2.3 m", "pedestrian"),
)
# Horizontal range from the ego origin. 10 m belongs to mid, 25 m belongs to far.
DISTANCE_ZONES = (
    ("near", "0–10 m", 0.0, 10.0),
    ("mid", "10–25 m", 10.0, 25.0),
    ("far", "25 m+", 25.0, float("inf")),
)


def semantic_class_for_z(z: float) -> str:
    if z < 0.45:
        return "driveable"
    if z < 2.3:
        return "vehicle"
    return "pedestrian"


def _cell_keys(
    centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> set[tuple[int, int, int]]:
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return set()
    idx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    return {tuple(int(v) for v in row) for row in idx.tolist()}


def occupancy_counts(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact-cell occupancy counts against a ground-truth grid.

    IoU = TP / (TP + FP + FN), with TP the cells both grids occupy, FP the
    predicted cells that are empty in ground truth, and FN the ground-truth
    cells the prediction missed. mIoU is the unweighted mean of that IoU over
    the driveable, vehicle, and pedestrian height bands that contain any cell.
    """
    size = float(max(voxel_size, 0.05))
    origin_vec = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    pred_keys = _cell_keys(pred_centers, size, origin_vec)
    gt_keys = _cell_keys(gt_centers, size, origin_vec)
    true_positive = len(pred_keys & gt_keys)
    false_positive = len(pred_keys - gt_keys)
    false_negative = len(gt_keys - pred_keys)
    binary = _iou_from_counts(true_positive, false_positive, false_negative)

    per_class: list[dict[str, Any]] = []
    class_ious: list[float] = []
    for name in SEMANTIC_CLASSES:
        pred_band = {key for key in pred_keys if _class_of_key(key, origin_vec, size) == name}
        gt_band = {key for key in gt_keys if _class_of_key(key, origin_vec, size) == name}
        tp = len(pred_band & gt_band)
        fp = len(pred_band - gt_band)
        fn = len(gt_band - pred_band)
        if tp + fp + fn == 0:
            continue
        iou = _iou_from_counts(tp, fp, fn)
        class_ious.append(iou)
        per_class.append({"class": name, "tp": tp, "fp": fp, "fn": fn, "iou": iou})

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "iou": binary,
        "miou": float(sum(class_ious) / len(class_ious)) if class_ious else binary,
        "classes": per_class,
    }


def _class_of_key(key: tuple[int, int, int], origin: np.ndarray, voxel_size: float) -> str:
    z = float(origin[2] + (key[2] + 0.5) * voxel_size)
    return semantic_class_for_z(z)


def _iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def gt_occupancy_threshold(occupancy_threshold: float) -> float:
    """Keep-threshold for the denser ground-truth pane."""
    return max(float(occupancy_threshold) * GT_OCCUPANCY_SCALE, GT_OCCUPANCY_FLOOR)


def fuse_camera_points(camera_payloads: list[dict[str, Any]]) -> np.ndarray:
    """Unproject each camera's disparity and concatenate in the ego frame."""
    clouds: list[np.ndarray] = []
    for payload in camera_payloads:
        disparity = np.asarray(payload["disparity"], dtype=np.float32)
        height, width = disparity.shape[:2]
        k_depth = scale_intrinsics(
            payload["intrinsic"],
            ORIGINAL_IMAGE_SIZE,
            (width, height),
        )
        metric = metric_depth_from_ground(
            disparity,
            k_depth,
            payload["rotation"],
            payload["translation"],
        )
        cam_pts = unproject_depth(metric, k_depth)
        ego_pts = camera_to_ego(cam_pts, payload["rotation"], payload["translation"])
        clouds.append(ego_pts)

    if not clouds:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(clouds, axis=0)


def _touch_open3d(centers: np.ndarray, voxel_size: float) -> None:
    if centers.shape[0] == 0:
        return
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
        o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    except Exception as exc:  # pragma: no cover - Open3D optional at import time
        logger.warning("Open3D voxelization unavailable (%s); using NumPy grid", exc)


def _voxel_keys(centers: np.ndarray, voxel_size: float, origin: np.ndarray) -> np.ndarray:
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.floor((pts - origin) / voxel_size).astype(np.int64)


def _gt_display_voxels(
    pred_centers: np.ndarray,
    pred_occ: np.ndarray,
    gt_centers: np.ndarray,
    gt_occ: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Show prediction cells plus extra lower-threshold GT cells (denser pane)."""
    if gt_centers.shape[0] == 0:
        return pred_centers, pred_occ
    pred_keyset = {tuple(row) for row in _voxel_keys(pred_centers, voxel_size, origin).tolist()}
    extra_mask = np.array(
        [tuple(row) not in pred_keyset for row in _voxel_keys(gt_centers, voxel_size, origin).tolist()],
        dtype=bool,
    )
    extra_c = gt_centers[extra_mask]
    extra_o = gt_occ[extra_mask]
    if extra_c.shape[0] > MAX_VOXELS:
        extra_c, extra_o = _cap_voxels(extra_c, extra_o)
    if extra_c.shape[0] == 0:
        return pred_centers, pred_occ
    return (
        np.concatenate([pred_centers, extra_c], axis=0),
        np.concatenate([pred_occ, extra_o], axis=0),
    )


def voxelize_aligned(
    points: np.ndarray,
    voxel_size: float,
    occupancy_threshold: float,
    origin: np.ndarray,
    bounds: np.ndarray = EGO_BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram occupancy on a shared origin so pred/GT indices line up."""
    size = float(max(voxel_size, 0.05))
    return _voxelize_numpy(points, size, occupancy_threshold, bounds, origin=origin, cap=False)


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
    merged = fuse_camera_points(camera_payloads)
    return voxelize_occupancy(merged, voxel_size, occupancy_threshold)


def project_cameras_to_voxel_pair(
    camera_payloads: list[dict[str, Any]],
    lidar_points: np.ndarray,
    voxel_size: float,
    occupancy_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Vision occupancy from cameras, ground truth from lidar, plus grid mIoU.

    mIoU is measured on the full grids, sharing the ego-bounds origin, before
    the display cap. The cap is only a transport limit and would otherwise
    drop overlapping cells.
    """
    origin = EGO_BOUNDS[:, 0]
    merged = fuse_camera_points(camera_payloads)
    pred_centers, pred_occ = voxelize_occupancy(
        merged,
        voxel_size,
        occupancy_threshold,
        origin=origin,
        cap=False,
    )
    occupied, free, weights = camera_known_space(
        lidar_points, camera_payloads, voxel_size, origin=origin
    )
    # One cell of slack on the live grid. Unknown cells are not errors.
    score = known_space_counts(
        pred_centers, occupied, free, voxel_size, origin=origin, tolerance=1
    )
    miou = float(score["iou"])
    err_centers, err_occ = known_space_error_voxels(
        pred_centers, occupied, free, voxel_size, origin=origin
    )
    pred_centers, pred_occ = _cap_voxels(pred_centers, pred_occ)
    gt_centers, gt_occ = _cap_voxels(occupied, weights)
    free_c, free_o, unk_c, unk_o = display_label_clouds(occupied, free, voxel_size, origin=origin)
    size = float(max(voxel_size, 0.05))
    _touch_open3d(pred_centers, size)
    _touch_open3d(gt_centers, size)
    return (
        pred_centers,
        pred_occ,
        gt_centers,
        gt_occ,
        err_centers,
        err_occ,
        miou,
        score,
        free_c,
        free_o,
        unk_c,
        unk_o,
    )

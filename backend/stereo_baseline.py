"""Multi-view stereo occupancy from overlapping surround cameras.

SIFT features are matched between camera pairs that share a field of view,
kept when they agree with a fundamental matrix, and triangulated into the
ego frame. Metric depth from MiDaS is not used, and lidar is not an input.
"""

from __future__ import annotations

import numpy as np

from nuscenes_loader import ORIGINAL_IMAGE_SIZE, load_rgb, load_synchronized_frame
from projection import cell_centers, quat_to_rotmat, scale_intrinsics

STEREO_NAME = "Multi-view stereo"
STEREO_ID = "stereo-sift"
PAIRS = (
    ("CAM_FRONT_LEFT", "CAM_FRONT"),
    ("CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK"),
    ("CAM_BACK", "CAM_BACK_RIGHT"),
)
_MAX_FEATURES = 2000
_RATIO = 0.75
_MAX_REPROJECT_PX = 6.0


def projection_matrix(
    intrinsic: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    image_size: tuple[int, int],
    source_size: tuple[int, int] = ORIGINAL_IMAGE_SIZE,
) -> np.ndarray:
    """3×4 ego-to-pixel matrix. `image_size` is (width, height) of the photo."""
    k = scale_intrinsics(intrinsic, source_size, image_size)
    rotation = quat_to_rotmat(rotation_wxyz)
    cam_from_ego = rotation.T
    t_cam = -cam_from_ego @ np.asarray(translation, dtype=np.float64).reshape(3)
    return k @ np.column_stack((cam_from_ego, t_cam))


def triangulate_matches(matrix_a: np.ndarray, matrix_b: np.ndarray, pixels_a: np.ndarray, pixels_b: np.ndarray) -> np.ndarray:
    """Ego points for corresponding pixels. Rows that fail the cheirality check are dropped."""
    left = np.asarray(pixels_a, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(pixels_b, dtype=np.float64).reshape(-1, 2)
    if left.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for multi-view stereo") from exc
    hom = cv2.triangulatePoints(matrix_a, matrix_b, left.T, right.T)
    points = (hom[:3] / hom[3:4]).T
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    left = left[finite]
    right = right[finite]
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keep = _in_front(matrix_a, points) & _in_front(matrix_b, points)
    keep &= _reprojection_ok(matrix_a, points, left) & _reprojection_ok(matrix_b, points, right)
    return points[keep].astype(np.float32)


def _in_front(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.column_stack((points, np.ones(points.shape[0])))
    cam = matrix @ hom.T
    return cam[2] > 0.5


def _reprojection_ok(matrix: np.ndarray, points: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    hom = np.column_stack((points, np.ones(points.shape[0])))
    projected = matrix @ hom.T
    uv = (projected[:2] / projected[2:3]).T
    return np.linalg.norm(uv - pixels, axis=1) < _MAX_REPROJECT_PX


def _match_pair(image_a: np.ndarray, image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY) if image_b.ndim == 3 else image_b
    sift = cv2.SIFT_create(nfeatures=_MAX_FEATURES)
    key_a, desc_a = sift.detectAndCompute(gray_a, None)
    key_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(key_a) < 8 or len(key_b) < 8:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    left = []
    right = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < _RATIO * second.distance:
            left.append(key_a[best.queryIdx].pt)
            right.append(key_b[best.trainIdx].pt)
    if len(left) < 8:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    pixels_a = np.asarray(left, dtype=np.float64)
    pixels_b = np.asarray(right, dtype=np.float64)
    _fundamental, mask = cv2.findFundamentalMat(pixels_a, pixels_b, cv2.FM_RANSAC, 2.0, 0.99)
    if mask is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 8:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return pixels_a[inliers], pixels_b[inliers]


def predict_points(frame_index: int) -> np.ndarray:
    """Triangulated ego points from the four overlapping camera pairs."""
    synced = load_synchronized_frame(int(frame_index))
    clouds = []
    for left_id, right_id in PAIRS:
        left = synced.cameras[left_id]
        right = synced.cameras[right_id]
        image_a = load_rgb(left.image_path)
        image_b = load_rgb(right.image_path)
        pixels_a, pixels_b = _match_pair(image_a, image_b)
        if pixels_a.shape[0] == 0:
            continue
        size_a = (image_a.shape[1], image_a.shape[0])
        size_b = (image_b.shape[1], image_b.shape[0])
        matrix_a = projection_matrix(left.intrinsic, left.rotation, left.translation, size_a)
        matrix_b = projection_matrix(right.intrinsic, right.rotation, right.translation, size_b)
        points = triangulate_matches(matrix_a, matrix_b, pixels_a, pixels_b)
        if points.shape[0]:
            clouds.append(points)
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)


_CENTER_CACHE: dict[tuple[int, float], np.ndarray] = {}


def predict_centers(frame_index: int, voxel_m: float = 1.0) -> np.ndarray:
    """Occupied cells that contain a triangulated stereo point."""
    key = (int(frame_index), round(float(voxel_m), 3))
    cached = _CENTER_CACHE.get(key)
    if cached is not None:
        return cached
    centers = cell_centers(predict_points(key[0]), key[1])
    _CENTER_CACHE[key] = centers
    return centers

"""Ray-surface occupancy model.

Each camera pixel predicts one metric depth and whether that ray hits a
surface. Kept pixels are unprojected into the ego frame and binned on the
1.0 m grid. There is no voxel feature volume and no 3D convolution.
Lidar is the training label only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nuscenes_loader import CAMERA_IDS, ORIGINAL_IMAGE_SIZE, load_manifest, load_synchronized_frame
from projection import EGO_BOUNDS, camera_known_space, camera_to_ego, quat_to_rotmat, unproject_depth, voxelize_lidar
from voxnet_baseline import BENCH_VOXEL_M, held_out

logger = logging.getLogger(__name__)

RAY_SURFACE_NAME = "Ray surface"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "ray_surface.pt"
IMAGE_SIZE = (160, 90)  # width, height
NATIVE_SIZE = ORIGINAL_IMAGE_SIZE
_CHECKPOINT_ID: str | None = None
_RUNTIME: dict[str, object] | None = None


class RaySurfaceNet(nn.Module):
    """Full-resolution depth and keep maps from one RGB image."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(size=(IMAGE_SIZE[1], IMAGE_SIZE[0]), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)
        self.keep_head = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.constant_(self.depth_head.bias, 1.0)
        nn.init.constant_(self.keep_head.bias, -1.5)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """image is (1, 3, H, W). Returns depth (1, 1, H, W) and keep logits."""
        feat = self.up(self.trunk(image))
        # Scale the softplus so a small conv activation can reach the far ego bound.
        depth = 8.0 * F.softplus(self.depth_head(feat)) + 0.5
        keep = self.keep_head(feat)
        return depth, keep


def checkpoint_id() -> str:
    global _CHECKPOINT_ID
    if _CHECKPOINT_ID is not None:
        return _CHECKPOINT_ID
    if not CHECKPOINT_PATH.is_file():
        return "missing"
    import hashlib

    _CHECKPOINT_ID = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()[:12]
    return _CHECKPOINT_ID


def _scaled_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    width, height = IMAGE_SIZE
    native_w, native_h = NATIVE_SIZE
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    k[0] *= width / float(native_w)
    k[1] *= height / float(native_h)
    return k


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def surface_cells(
    depth: np.ndarray,
    keep: np.ndarray,
    intrinsic: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Occupied 1.0 m cell centers for one depth map. Unkept pixels are dropped."""
    depth_map = np.asarray(depth, dtype=np.float32)
    kept = np.asarray(keep, dtype=bool)
    if depth_map.shape != kept.shape:
        raise ValueError(f"Depth {depth_map.shape} does not match keep mask {kept.shape}")
    masked = np.where(kept, depth_map, np.float32(0.0))
    cam_points = unproject_depth(masked, intrinsic, stride=1)
    if cam_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ego = camera_to_ego(cam_points, rotation_wxyz, translation)
    centers, _occ = voxelize_lidar(ego.astype(np.float32), BENCH_VOXEL_M, cap=False)
    return centers


def _occupied_free(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    from lidar_gt import load_lidar_ego_points

    points = load_lidar_ego_points(frame_index)
    try:
        synced = load_synchronized_frame(frame_index)
        cameras = [
            {
                "intrinsic": synced.cameras[camera_id].intrinsic,
                "rotation": synced.cameras[camera_id].rotation,
                "translation": synced.cameras[camera_id].translation,
            }
            for camera_id in CAMERA_IDS
        ]
    except Exception:
        cameras = None
    occupied, free, _weights = camera_known_space(
        points, cameras, BENCH_VOXEL_M, origin=EGO_BOUNDS[:, 0]
    )
    return occupied, free


def _surface_target(
    occupied: np.ndarray,
    intrinsic: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest occupied-cell depth per pixel, and the positive keep mask."""
    width, height = IMAGE_SIZE
    depth = np.zeros((height, width), dtype=np.float32)
    positive = np.zeros((height, width), dtype=bool)
    if occupied.size == 0:
        return depth, positive
    k = _scaled_intrinsic(intrinsic)
    rotation = quat_to_rotmat(np.asarray(rotation_wxyz, dtype=np.float64))
    trans = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    cam = (occupied.astype(np.float64) - trans) @ rotation
    z = cam[:, 2]
    in_front = z > 0.5
    if not np.any(in_front):
        return depth, positive
    pix = cam @ k.T
    u = np.divide(pix[:, 0], z, out=np.full(z.shape, -1.0), where=in_front)
    v = np.divide(pix[:, 1], z, out=np.full(z.shape, -1.0), where=in_front)
    inside = in_front & (u >= 0.0) & (v >= 0.0) & (u < width) & (v < height)
    if not np.any(inside):
        return depth, positive
    ui = np.minimum(u[inside].astype(np.int64), width - 1)
    vi = np.minimum(v[inside].astype(np.int64), height - 1)
    flat = vi * width + ui
    zbuf = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, flat, z[inside])
    hit = np.isfinite(zbuf)
    depth.reshape(-1)[hit] = zbuf[hit].astype(np.float32)
    positive.reshape(-1)[hit] = True
    return depth, positive


def _camera_loss(
    depth_pred: torch.Tensor,
    keep_logit: torch.Tensor,
    depth_target: np.ndarray,
    positive: np.ndarray,
    keep_weight: float,
) -> torch.Tensor:
    pos = torch.from_numpy(positive)
    target = torch.from_numpy(depth_target)
    pred = depth_pred[0, 0]
    logit = keep_logit[0, 0]
    if bool(pos.any()):
        err = (pred[pos] - target[pos]).abs()
        # Far pixels are rare. Weight by depth so a horizon miss moves the head.
        weights = target[pos].clamp(min=1.0)
        depth_loss = (err * weights).sum() / weights.sum()
    else:
        depth_loss = pred.sum() * 0.0
    positive_count = pos.sum().clamp(min=1.0)
    negative_count = ((~pos).sum()).clamp(min=1.0)
    pos_weight = (negative_count / positive_count).clamp(max=30.0)
    keep_loss = F.binary_cross_entropy_with_logits(
        logit, pos.to(dtype=logit.dtype), pos_weight=pos_weight.reshape(1), reduction="mean"
    )
    return depth_loss + float(keep_weight) * keep_loss


def _predict_maps(
    model: RaySurfaceNet, frame_index: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    synced = load_synchronized_frame(frame_index)
    maps = []
    for camera_id in CAMERA_IDS:
        sample = synced.cameras[camera_id]
        image = _image_tensor(sample.image_path).unsqueeze(0)
        with torch.inference_mode():
            depth, keep_logit = model(image)
            keep_prob = torch.sigmoid(keep_logit)
        maps.append(
            (
                depth[0, 0].cpu().numpy(),
                keep_prob[0, 0].cpu().numpy(),
                _scaled_intrinsic(sample.intrinsic),
                np.asarray(sample.rotation, dtype=np.float64),
                np.asarray(sample.translation, dtype=np.float64),
            )
        )
    return maps


def _centers_from_maps(
    maps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> np.ndarray:
    clouds = []
    for depth, keep_prob, intrinsic, rotation, translation in maps:
        centers = surface_cells(depth, keep_prob >= float(threshold), intrinsic, rotation, translation)
        if centers.shape[0]:
            clouds.append(centers)
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0)


def _merge_centers(parts: np.ndarray) -> np.ndarray:
    if parts.shape[0] == 0:
        return parts
    merged, _occ = voxelize_lidar(parts, BENCH_VOXEL_M, cap=False)
    return merged


def load_baseline() -> dict[str, object]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing {CHECKPOINT_PATH}. Train it with `python -m ray_surface_baseline`.")
    try:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = RaySurfaceNet()
    model.trunk.load_state_dict(blob["trunk"])
    model.up.load_state_dict(blob["up"])
    model.depth_head.load_state_dict(blob["depth_head"])
    model.keep_head.load_state_dict(blob["keep_head"])
    model.eval()
    _RUNTIME = {"model": model, "threshold": round(float(blob["threshold"]), 4)}
    return _RUNTIME


def predict_centers(frame_index: int) -> np.ndarray:
    """Occupied cell centers. Lidar is not read here."""
    runtime = load_baseline()
    model: RaySurfaceNet = runtime["model"]  # type: ignore[assignment]
    model.eval()
    maps = _predict_maps(model, int(frame_index))
    return _merge_centers(_centers_from_maps(maps, float(runtime["threshold"])))


def _select_threshold(
    model: RaySurfaceNet,
    frame_ids: list[int],
) -> float:
    from projection import known_space_counts

    cached = []
    model.eval()
    for frame_index in frame_ids:
        occupied, free = _occupied_free(frame_index)
        cached.append((_predict_maps(model, frame_index), occupied, free))
    best_t = 0.5
    best_iou = -1.0
    for threshold in np.linspace(0.05, 0.9, 18):
        scores = []
        for maps, occupied, free in cached:
            pred = _merge_centers(_centers_from_maps(maps, float(threshold)))
            score = known_space_counts(
                pred,
                occupied,
                free,
                BENCH_VOXEL_M,
                origin=EGO_BOUNDS[:, 0],
                tolerance=0,
            )
            scores.append(float(score["iou"]))
        mean_iou = float(np.mean(scores)) if scores else 0.0
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_t = float(threshold)
    print(f"ray-surface threshold {best_t:.2f} train IoU {best_iou:.3f}", flush=True)
    return best_t


def fit_baseline(epochs: int = 12, lr: float = 1e-3, keep_weight: float = 1.0) -> dict[str, object]:
    """Fit on the train split only. Held-out frames are not in the loss or the threshold."""
    torch.manual_seed(0)
    frame_count = int(load_manifest()["frame_count"])
    train_ids = [i for i in range(frame_count) if not held_out(i)]
    model = RaySurfaceNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        losses = []
        for frame_index in train_ids:
            synced = load_synchronized_frame(frame_index)
            occupied, _free = _occupied_free(frame_index)
            optimizer.zero_grad(set_to_none=True)
            loss = None
            for camera_id in CAMERA_IDS:
                sample = synced.cameras[camera_id]
                image = _image_tensor(sample.image_path).unsqueeze(0)
                depth_pred, keep_logit = model(image)
                depth_target, positive = _surface_target(
                    occupied, sample.intrinsic, sample.rotation, sample.translation
                )
                term = _camera_loss(depth_pred, keep_logit, depth_target, positive, keep_weight)
                loss = term if loss is None else loss + term
            assert loss is not None
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(
            f"ray-surface epoch {epoch} loss {float(np.mean(losses)):.4f} keep_weight {keep_weight:.1f}",
            flush=True,
        )

    model.eval()
    threshold = _select_threshold(model, train_ids)
    blob = {
        "trunk": model.trunk.state_dict(),
        "up": model.up.state_dict(),
        "depth_head": model.depth_head.state_dict(),
        "keep_head": model.keep_head.state_dict(),
        "threshold": threshold,
        "keep_weight": float(keep_weight),
        "voxel_m": BENCH_VOXEL_M,
        "train_frames": train_ids,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, CHECKPOINT_PATH)
    global _RUNTIME, _CHECKPOINT_ID
    _RUNTIME = None
    _CHECKPOINT_ID = None
    print(f"ray-surface threshold {threshold:.2f}", flush=True)
    return blob


if __name__ == "__main__":
    fit_baseline()

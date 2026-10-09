"""Lift-Splat occupancy baseline.

A small image encoder predicts a categorical depth distribution. Each pixel
splats its features into the ego voxel along that distribution, and a 3D
head reads the volume. This is not the VoxNet grid sampler and it does not
read MiDaS depth. Lidar is the training label only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nuscenes_loader import CAMERA_IDS, load_manifest, load_synchronized_frame
from projection import EGO_BOUNDS, quat_to_rotmat
from voxnet_baseline import BENCH_VOXEL_M, _centers_from_mask, grid_centers, held_out

logger = logging.getLogger(__name__)

LSS_NAME = "Lift-Splat"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "lss_baseline.pt"
IMAGE_SIZE = (128, 72)
FEATURE_CHANNELS = 8
DEPTH_BINS = 6
BIN_CENTERS_M = (2.0, 6.0, 12.0, 20.0, 30.0, 42.0)
_CHECKPOINT_ID: str | None = None
_RUNTIME: dict[str, object] | None = None


class LiftSplatOccupancy(nn.Module):
    """Depth distribution over image features, splatted into a voxel volume."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, FEATURE_CHANNELS, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth = nn.Conv2d(FEATURE_CHANNELS, DEPTH_BINS, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv3d(FEATURE_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward_logits(self, volume: torch.Tensor) -> torch.Tensor:
        """volume is (1, C, Z, Y, X)."""
        return self.head(volume)


def checkpoint_id() -> str:
    global _CHECKPOINT_ID
    if _CHECKPOINT_ID is not None:
        return _CHECKPOINT_ID
    if not CHECKPOINT_PATH.is_file():
        return "missing"
    import hashlib

    _CHECKPOINT_ID = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()[:12]
    return _CHECKPOINT_ID


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _splat_volume(model: LiftSplatOccupancy, frame_index: int, shape: tuple[int, int, int]) -> torch.Tensor:
    """(1, C, Z, Y, X) feature volume. Gradients flow through the depth weights."""
    nx, ny, nz = shape
    cells = nx * ny * nz
    acc = torch.zeros((FEATURE_CHANNELS, cells), dtype=torch.float32)
    weight = torch.zeros((cells,), dtype=torch.float32)
    origin = torch.as_tensor(EGO_BOUNDS[:, 0], dtype=torch.float32)
    synced = load_synchronized_frame(frame_index)
    for camera_id in CAMERA_IDS:
        sample = synced.cameras[camera_id]
        image = _image_tensor(sample.image_path).unsqueeze(0)
        feat = model.encoder(image)
        prob = torch.softmax(model.depth(feat), dim=1)[0]
        feat_map = feat[0]
        _, feat_h, feat_w = feat_map.shape
        k = torch.as_tensor(sample.intrinsic, dtype=torch.float32).clone()
        k[0] *= feat_w / 1600.0
        k[1] *= feat_h / 900.0
        rotation = torch.as_tensor(quat_to_rotmat(sample.rotation), dtype=torch.float32)
        translation = torch.as_tensor(sample.translation, dtype=torch.float32)
        vs = torch.arange(feat_h, dtype=torch.float32) + 0.5
        us = torch.arange(feat_w, dtype=torch.float32) + 0.5
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        pixels = torch.stack((grid_u.reshape(-1), grid_v.reshape(-1), torch.ones(feat_h * feat_w)), dim=0)
        rays = torch.linalg.inv(k) @ pixels
        flat_feat = feat_map.reshape(FEATURE_CHANNELS, -1)
        for bin_index, depth_m in enumerate(BIN_CENTERS_M):
            cam = rays * float(depth_m)
            ego = (rotation @ cam).T + translation
            idx = torch.floor((ego - origin) / BENCH_VOXEL_M).to(torch.long)
            valid = (
                (idx[:, 0] >= 0)
                & (idx[:, 0] < nx)
                & (idx[:, 1] >= 0)
                & (idx[:, 1] < ny)
                & (idx[:, 2] >= 0)
                & (idx[:, 2] < nz)
            )
            if not bool(valid.any()):
                continue
            chosen = idx[valid]
            flat = (chosen[:, 0] * ny + chosen[:, 1]) * nz + chosen[:, 2]
            gate = prob[bin_index].reshape(-1)[valid]
            acc.index_add_(1, flat, flat_feat[:, valid] * gate.unsqueeze(0))
            weight.index_add_(0, flat, gate)
    volume = acc / weight.clamp(min=1e-3).unsqueeze(0)
    volume = volume.view(FEATURE_CHANNELS, nx, ny, nz)
    return volume.permute(0, 3, 2, 1).unsqueeze(0).contiguous()


def load_baseline() -> dict[str, object]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing {CHECKPOINT_PATH}. Train it with `python -m lss_baseline`.")
    try:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = LiftSplatOccupancy()
    model.encoder.load_state_dict(blob["encoder"])
    model.depth.load_state_dict(blob["depth"])
    model.head.load_state_dict(blob["head"])
    model.eval()
    centers, shape = grid_centers(float(blob.get("voxel_m", BENCH_VOXEL_M)))
    _RUNTIME = {
        "model": model,
        "threshold": round(float(blob["threshold"]), 4),
        "centers": centers,
        "shape": shape,
    }
    return _RUNTIME


def predict_centers(frame_index: int, voxel_m: float = 1.0) -> np.ndarray:
    """Occupied cell centers. The network is defined on the 1 m grid."""
    del voxel_m
    runtime = load_baseline()
    model: LiftSplatOccupancy = runtime["model"]  # type: ignore[assignment]
    centers_np: np.ndarray = runtime["centers"]  # type: ignore[assignment]
    shape: tuple[int, int, int] = runtime["shape"]  # type: ignore[assignment]
    with torch.inference_mode():
        volume = _splat_volume(model, int(frame_index), shape)
        prob = torch.sigmoid(model.forward_logits(volume))[0, 0].cpu().numpy()
    return _centers_from_mask(prob >= float(runtime["threshold"]), centers_np, shape)


def _lidar_target(frame_index: int, shape: tuple[int, int, int]) -> torch.Tensor:
    from lidar_gt import camera_space_lidar

    nx, ny, nz = shape
    target = np.zeros((nx, ny, nz), dtype=np.float32)
    points = camera_space_lidar(frame_index)
    if points.shape[0]:
        origin = EGO_BOUNDS[:, 0].astype(np.float64)
        idx = np.floor((points.astype(np.float64) - origin) / BENCH_VOXEL_M).astype(np.int64)
        valid = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < nz)
        )
        idx = idx[valid]
        if idx.shape[0]:
            target[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return torch.from_numpy(np.transpose(target, (2, 1, 0))).view(1, 1, nz, ny, nx).contiguous()


def fit_baseline(epochs: int = 3, lr: float = 1e-3) -> dict[str, object]:
    """Fit on the train split only. Held-out frames are not in the loss or the threshold."""
    torch.manual_seed(0)
    frame_count = int(load_manifest()["frame_count"])
    train_ids = [i for i in range(frame_count) if not held_out(i)]
    model = LiftSplatOccupancy()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _centers_np, shape = grid_centers()
    for epoch in range(epochs):
        losses = []
        for frame_index in train_ids:
            volume = _splat_volume(model, frame_index, shape)
            logits = model.forward_logits(volume)
            target = _lidar_target(frame_index, shape)
            positive = target.sum().clamp(min=1.0)
            negative = target.numel() - target.sum()
            pos_weight = (negative / positive).clamp(max=30.0).reshape(1)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"lss epoch {epoch} loss {float(np.mean(losses)):.4f}", flush=True)

    model.eval()
    threshold = _select_threshold(model, train_ids, shape)
    blob = {
        "encoder": model.encoder.state_dict(),
        "depth": model.depth.state_dict(),
        "head": model.head.state_dict(),
        "threshold": threshold,
        "voxel_m": BENCH_VOXEL_M,
        "train_frames": train_ids,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, CHECKPOINT_PATH)
    global _RUNTIME, _CHECKPOINT_ID
    _RUNTIME = None
    _CHECKPOINT_ID = None
    print(f"lss threshold {threshold:.2f}", flush=True)
    return blob


def _select_threshold(model: LiftSplatOccupancy, frame_ids: list[int], shape: tuple[int, int, int]) -> float:
    from lidar_gt import camera_space_lidar
    from projection import occupancy_counts, voxelize_lidar

    centers_np, _shape = grid_centers()
    probs = []
    gts = []
    with torch.inference_mode():
        for frame_index in frame_ids:
            volume = _splat_volume(model, frame_index, shape)
            probs.append(torch.sigmoid(model.forward_logits(volume))[0, 0].cpu().numpy())
            gt, _ = voxelize_lidar(camera_space_lidar(frame_index), BENCH_VOXEL_M, cap=False)
            gts.append(gt)
    best_t = 0.5
    best_iou = -1.0
    for threshold in np.linspace(0.2, 0.8, 7):
        scores = []
        for prob, gt in zip(probs, gts):
            pred = _centers_from_mask(prob >= threshold, centers_np, shape)
            scores.append(occupancy_counts(pred, gt, BENCH_VOXEL_M, origin=EGO_BOUNDS[:, 0])["iou"])
        mean_iou = float(np.mean(scores)) if scores else 0.0
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_t = float(threshold)
    print(f"lss threshold {best_t:.2f} train IoU {best_iou:.3f}", flush=True)
    return best_t


if __name__ == "__main__":
    fit_baseline()

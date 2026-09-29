"""VoxNet-style 3D CNN occupancy baseline.

Camera images are encoded with a small 2D conv net. Those feature maps are
sampled into an ego-frame voxel volume by projecting each cell through the
calibrated cameras. A stack of 3D convolutions, the VoxNet pattern of
Conv3d-ReLU blocks, then predicts an occupancy logit per cell. Metric depth
from MiDaS is not an input. Lidar is the training target and the score, not
a feature.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from lidar_gt import camera_space_lidar
from nuscenes_loader import CAMERA_IDS, load_manifest, load_synchronized_frame
from projection import EGO_BOUNDS, quat_to_rotmat

logger = logging.getLogger(__name__)

BENCH_VOXEL_M = 1.0
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "voxnet_baseline.pt"
IMAGE_SIZE = (128, 72)  # width, height
FEATURE_CHANNELS = 8


def held_out(frame_index: int) -> bool:
    """Every fifth frame, including 0, is kept out of the fit."""
    return int(frame_index) % 5 == 0


def grid_centers(voxel_size: float = BENCH_VOXEL_M) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Cell centers inside the ego bounds. Shape is (X, Y, Z)."""
    size = float(voxel_size)
    axes = []
    for lo, hi in EGO_BOUNDS:
        axes.append(np.arange(lo + size * 0.5, hi, size, dtype=np.float32))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    centers = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    shape = (int(xx.shape[0]), int(xx.shape[1]), int(xx.shape[2]))
    return centers, shape


class VoxNetOccupancy(nn.Module):
    """Shared 2D encoder plus a dense 3D conv occupancy head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, FEATURE_CHANNELS, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv3d(FEATURE_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward_logits(self, volume: torch.Tensor) -> torch.Tensor:
        """volume is (N, C, Z, Y, X). Returns logits of the same spatial shape."""
        return self.head(volume)


_RUNTIME: dict[str, object] | None = None
_PRED_CACHE: OrderedDict[int, np.ndarray] = OrderedDict()
_PRED_CACHE_LIMIT = 8


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _feature_volume(
    model: VoxNetOccupancy,
    frame_index: int,
    centers: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Scatter projected image features into an (1, C, Z, Y, X) volume."""
    synced = load_synchronized_frame(frame_index)
    nx, ny, nz = shape
    acc = centers.new_zeros((FEATURE_CHANNELS, centers.shape[0]))
    weight = centers.new_zeros((1, centers.shape[0]))
    for camera_id in CAMERA_IDS:
        sample = synced.cameras[camera_id]
        image = _image_tensor(sample.image_path).unsqueeze(0)
        feat = model.encoder(image)
        _, _, feat_h, feat_w = feat.shape
        k = torch.as_tensor(sample.intrinsic, dtype=torch.float32)
        k = k.clone()
        k[0] *= feat_w / 1600.0
        k[1] *= feat_h / 900.0
        rotation = torch.as_tensor(quat_to_rotmat(sample.rotation), dtype=torch.float32)
        translation = torch.as_tensor(sample.translation, dtype=torch.float32)
        sampled, valid = _sample_features(feat, centers, k, rotation, translation, feat_h, feat_w)
        acc = acc + sampled * valid
        weight = weight + valid
    volume = acc / weight.clamp(min=1.0)
    volume = volume.view(FEATURE_CHANNELS, nx, ny, nz)
    # Conv3d depth axis is Z, the short ego-up dimension.
    volume = volume.permute(0, 3, 2, 1).unsqueeze(0).contiguous()
    return volume


def _sample_features(
    feat: torch.Tensor,
    centers: torch.Tensor,
    intrinsic: torch.Tensor,
    rotation_ego_from_cam: torch.Tensor,
    translation: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear-sample one camera feature map at each voxel center."""
    offset = centers - translation.view(1, 3)
    cam = offset @ rotation_ego_from_cam
    depth = cam[:, 2]
    pix = cam @ intrinsic.T
    u = pix[:, 0] / depth.clamp(min=1e-4)
    v = pix[:, 1] / depth.clamp(min=1e-4)
    valid = (depth > 0.5) & (u >= 0) & (v >= 0) & (u <= feat_w - 1) & (v <= feat_h - 1)
    gx = (u / max(feat_w - 1, 1)) * 2.0 - 1.0
    gy = (v / max(feat_h - 1, 1)) * 2.0 - 1.0
    gx = torch.where(valid, gx, torch.zeros_like(gx))
    gy = torch.where(valid, gy, torch.zeros_like(gy))
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.view(FEATURE_CHANNELS, -1)
    return sampled, valid.to(sampled.dtype).view(1, -1)


def _lidar_target(frame_index: int, shape: tuple[int, int, int]) -> torch.Tensor:
    """Binary occupancy volume (1, 1, Z, Y, X). Lidar is the label, not a feature."""
    points = camera_space_lidar(frame_index)
    nx, ny, nz = shape
    target = np.zeros((nx, ny, nz), dtype=np.float32)
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
    volume = torch.from_numpy(np.transpose(target, (2, 1, 0))).view(1, 1, nz, ny, nx)
    return volume.contiguous()


def _centers_from_mask(mask_zyx: np.ndarray, centers: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    # mask is (Z, Y, X) to match the conv output. centers are flattened (X, Y, Z).
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).reshape(-1)
    if mask_xyz.shape[0] != centers.shape[0]:
        raise RuntimeError(f"Mask {mask_xyz.shape} does not match grid {shape}")
    del nx, ny, nz
    return centers[mask_xyz.astype(bool)].astype(np.float32)


def load_baseline() -> dict[str, object]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT_PATH}. Train it with `python -m voxnet_baseline`."
        )
    try:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = VoxNetOccupancy()
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["head"])
    model.eval()
    centers, shape = grid_centers(float(blob.get("voxel_m", BENCH_VOXEL_M)))
    _RUNTIME = {
        "model": model,
        "threshold": round(float(blob["threshold"]), 4),
        "centers": centers,
        "shape": shape,
        "heldout": blob.get("heldout"),
    }
    return _RUNTIME


def predict_centers(frame_index: int) -> np.ndarray:
    """Occupied cell centers from the 3D CNN. Lidar is not read here."""
    index = int(frame_index)
    cached = _PRED_CACHE.get(index)
    if cached is not None:
        _PRED_CACHE.move_to_end(index)
        return cached
    runtime = load_baseline()
    model: VoxNetOccupancy = runtime["model"]  # type: ignore[assignment]
    centers_np: np.ndarray = runtime["centers"]  # type: ignore[assignment]
    shape: tuple[int, int, int] = runtime["shape"]  # type: ignore[assignment]
    centers = torch.from_numpy(centers_np)
    with torch.inference_mode():
        volume = _feature_volume(model, index, centers, shape)
        logits = model.forward_logits(volume)
        prob = torch.sigmoid(logits)[0, 0]
        mask = (prob >= float(runtime["threshold"])).cpu().numpy()
    predicted = _centers_from_mask(mask, centers_np, shape)
    _PRED_CACHE[index] = predicted
    while len(_PRED_CACHE) > _PRED_CACHE_LIMIT:
        _PRED_CACHE.popitem(last=False)
    return predicted


def fit_baseline(epochs: int = 6, lr: float = 1e-3) -> dict[str, object]:
    """Fit the CNN on frames that are not in the held-out split and save it."""
    torch.manual_seed(0)
    frame_count = int(load_manifest()["frame_count"])
    train_ids = [i for i in range(frame_count) if not held_out(i)]
    eval_ids = [i for i in range(frame_count) if held_out(i)]
    model = VoxNetOccupancy()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    centers_np, shape = grid_centers()
    centers = torch.from_numpy(centers_np)
    for epoch in range(epochs):
        losses = []
        for frame_index in train_ids:
            volume = _feature_volume(model, frame_index, centers, shape)
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
        logger.info("voxnet epoch %s loss %.4f", epoch, float(np.mean(losses)))
        print(f"voxnet epoch {epoch} loss {float(np.mean(losses)):.4f}", flush=True)

    threshold = _select_threshold(model, train_ids, centers, centers_np, shape)
    heldout = _heldout_summary(model, eval_ids, threshold, centers, centers_np, shape)
    blob = {
        "encoder": model.encoder.state_dict(),
        "head": model.head.state_dict(),
        "threshold": threshold,
        "voxel_m": BENCH_VOXEL_M,
        "train_frames": train_ids,
        "eval_frames": eval_ids,
        "heldout": heldout,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, CHECKPOINT_PATH)
    global _RUNTIME
    _RUNTIME = None
    _PRED_CACHE.clear()
    print(
        "voxnet held-out IoU {iou:.3f} mIoU {miou:.3f} (tp {tp} fp {fp} fn {fn})".format(
            **heldout["voxnet"]
        ),
        flush=True,
    )
    return blob


def _occupied_centers(
    model: VoxNetOccupancy,
    frame_index: int,
    threshold: float,
    centers: torch.Tensor,
    centers_np: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    with torch.inference_mode():
        volume = _feature_volume(model, frame_index, centers, shape)
        prob = torch.sigmoid(model.forward_logits(volume))[0, 0].cpu().numpy()
    return _centers_from_mask(prob >= threshold, centers_np, shape)


def _heldout_summary(
    model: VoxNetOccupancy,
    frame_ids: list[int],
    threshold: float,
    centers: torch.Tensor,
    centers_np: np.ndarray,
    shape: tuple[int, int, int],
) -> dict[str, object]:
    """Pool held-out TP/FP/FN. These frames never entered the fit or the threshold pick."""
    from benchmark import micro_summary, monocular_centers

    model.eval()

    def predict(frame_index: int) -> np.ndarray:
        return _occupied_centers(model, frame_index, threshold, centers, centers_np, shape)

    mono_threshold = 0.38
    summary: dict[str, object] = {
        "frames": len(frame_ids),
        "monocular_threshold": mono_threshold,
        "voxnet": micro_summary(predict, frame_ids),
    }
    try:
        summary["monocular"] = micro_summary(
            lambda frame_index: monocular_centers(frame_index, mono_threshold),
            frame_ids,
        )
    except FileNotFoundError as exc:
        logger.warning("Skipping monocular held-out summary: %s", exc)
    return summary


def _select_threshold(
    model: VoxNetOccupancy,
    frame_ids: list[int],
    centers: torch.Tensor,
    centers_np: np.ndarray,
    shape: tuple[int, int, int],
) -> float:
    from projection import occupancy_counts, voxelize_lidar

    model.eval()
    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    with torch.inference_mode():
        for frame_index in frame_ids:
            volume = _feature_volume(model, frame_index, centers, shape)
            prob = torch.sigmoid(model.forward_logits(volume))[0, 0].cpu().numpy()
            probs.append(prob)
            gt, _ = voxelize_lidar(camera_space_lidar(frame_index), BENCH_VOXEL_M, cap=False)
            gts.append(gt)
    best_t = 0.5
    best_iou = -1.0
    for threshold in np.linspace(0.2, 0.8, 13):
        scores = []
        for prob, gt in zip(probs, gts):
            pred = _centers_from_mask(prob >= threshold, centers_np, shape)
            scores.append(occupancy_counts(pred, gt, BENCH_VOXEL_M, origin=EGO_BOUNDS[:, 0])["iou"])
        mean_iou = float(np.mean(scores)) if scores else 0.0
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_t = float(threshold)
    print(f"voxnet threshold {best_t:.2f} train IoU {best_iou:.3f}", flush=True)
    return best_t


if __name__ == "__main__":
    fit_baseline()

"""Score the monocular depth pipeline and the VoxNet baseline on one lidar grid.

Both predictions are voxelized at 1.0 m and compared with exact cell equality.
The live HUD keeps its own 0.2 m score with a one-cell tolerance; this table
does not replace that number.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from lidar_gt import load_lidar_ego_points
from midas_engine import DEPTH_CACHE_ROOT
from nuscenes_loader import CAMERA_IDS, load_synchronized_frame
from projection import (
    EGO_BOUNDS,
    SEMANTIC_CLASSES,
    fuse_camera_points,
    occupancy_counts,
    voxelize_lidar,
    voxelize_occupancy,
)
from voxnet_baseline import BENCH_VOXEL_M, held_out, load_baseline, predict_centers

MONOCULAR_NAME = "Monocular depth"
VOXNET_NAME = "VoxNet 3D CNN"
SCORING = (
    "exact 1.0 m cells; IoU = TP / (TP + FP + FN); "
    "mIoU = unweighted mean over driveable, vehicle, and pedestrian height bands"
)


def monocular_centers(frame_index: int, threshold: float) -> np.ndarray:
    """Pipeline A: cached MiDaS disparity, unproject with K, voxelize at 1.0 m."""
    synced = load_synchronized_frame(frame_index)
    payloads: list[dict[str, Any]] = []
    for camera_id in CAMERA_IDS:
        sample = synced.cameras[camera_id]
        cache_path = DEPTH_CACHE_ROOT / f"{int(frame_index):04d}" / f"{camera_id}.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing depth cache {cache_path}")
        payloads.append(
            {
                "disparity": np.load(cache_path),
                "intrinsic": sample.intrinsic,
                "translation": sample.translation,
                "rotation": sample.rotation,
            }
        )
    points = fuse_camera_points(payloads)
    centers, _occupancy = voxelize_occupancy(
        points,
        BENCH_VOXEL_M,
        occupancy_threshold=float(threshold),
        origin=EGO_BOUNDS[:, 0],
        cap=False,
    )
    return centers


def gt_centers_for_frame(frame_index: int) -> np.ndarray:
    centers, _occupancy = voxelize_lidar(
        load_lidar_ego_points(frame_index),
        BENCH_VOXEL_M,
        cap=False,
    )
    return centers


def score_prediction(name: str, pred_centers: np.ndarray, gt_centers: np.ndarray) -> dict[str, Any]:
    counts = occupancy_counts(pred_centers, gt_centers, BENCH_VOXEL_M, origin=EGO_BOUNDS[:, 0])
    return {
        "pipeline": name,
        "tp": int(counts["tp"]),
        "fp": int(counts["fp"]),
        "fn": int(counts["fn"]),
        "iou": float(counts["iou"]),
        "miou": float(counts["miou"]),
        "classes": [
            {
                "class": str(row["class"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "iou": float(row["iou"]),
            }
            for row in counts["classes"]
        ],
    }


def micro_summary(predict: Callable[[int], np.ndarray], frame_ids: list[int]) -> dict[str, float | int]:
    """Pool TP/FP/FN across frames, then compute IoU and per-band mIoU."""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    class_tp = {name: 0 for name in SEMANTIC_CLASSES}
    class_fp = {name: 0 for name in SEMANTIC_CLASSES}
    class_fn = {name: 0 for name in SEMANTIC_CLASSES}
    for frame_index in frame_ids:
        row = score_prediction("", predict(int(frame_index)), gt_centers_for_frame(int(frame_index)))
        true_positive += row["tp"]
        false_positive += row["fp"]
        false_negative += row["fn"]
        for band in row["classes"]:
            class_tp[band["class"]] += band["tp"]
            class_fp[band["class"]] += band["fp"]
            class_fn[band["class"]] += band["fn"]
    class_ious: list[float] = []
    for name in SEMANTIC_CLASSES:
        denom = class_tp[name] + class_fp[name] + class_fn[name]
        if denom:
            class_ious.append(class_tp[name] / denom)
    denom = true_positive + false_positive + false_negative
    iou = 1.0 if denom == 0 else true_positive / denom
    miou = float(sum(class_ious) / len(class_ious)) if class_ious else iou
    return {
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "iou": float(iou),
        "miou": float(miou),
    }


def build_frame_benchmark(frame_index: int, threshold: float) -> dict[str, Any]:
    """One-frame comparison of both pipelines against synchronized lidar."""
    index = int(frame_index)
    gt_centers = gt_centers_for_frame(index)
    mono = monocular_centers(index, float(threshold))
    voxnet = predict_centers(index)
    runtime = load_baseline()
    heldout = runtime.get("heldout")
    return {
        "voxel_m": float(BENCH_VOXEL_M),
        "frame_index": index,
        "split": "held-out" if held_out(index) else "train",
        "monocular_threshold": float(threshold),
        "voxnet_threshold": float(runtime["threshold"]),
        "scoring": SCORING,
        "rows": [
            score_prediction(MONOCULAR_NAME, mono, gt_centers),
            score_prediction(VOXNET_NAME, voxnet, gt_centers),
        ],
        "heldout": heldout if isinstance(heldout, dict) else None,
    }

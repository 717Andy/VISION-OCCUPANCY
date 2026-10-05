"""Score the monocular depth pipeline and the VoxNet baseline on one lidar grid.

Both predictions are voxelized at 1.0 m and compared with exact cell equality.
The live HUD keeps its own 0.2 m score with a one-cell tolerance; this table
does not replace that number.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from midas_engine import DEPTH_CACHE_ROOT
from nuscenes_loader import CAMERA_IDS, load_synchronized_frame
from projection import (
    EGO_BOUNDS,
    HEIGHT_BANDS,
    camera_known_space,
    fuse_camera_points,
    known_space_counts,
    voxelize_occupancy,
)
from voxnet_baseline import BENCH_VOXEL_M, held_out, load_baseline, predict_centers

MONOCULAR_NAME = "Monocular depth"
VOXNET_NAME = "VoxNet 3D CNN"
SCORING = (
    "exact 1.0 m cells; unknown cells ignored; "
    "IoU = TP / (TP + FP + FN) on occupied and free cells; "
    "height-band mIoU is the mean over below 0.45 m, 0.45 to 2.3 m, and above 2.3 m"
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


def known_grids_for_frame(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Occupied camera-visible hits and the free cells on the rays in front of them."""
    from lidar_gt import load_lidar_ego_points
    from nuscenes_loader import CAMERA_IDS as camera_ids

    points = load_lidar_ego_points(frame_index)
    try:
        synced = load_synchronized_frame(frame_index)
        cameras = [
            {
                "intrinsic": synced.cameras[camera_id].intrinsic,
                "rotation": synced.cameras[camera_id].rotation,
                "translation": synced.cameras[camera_id].translation,
            }
            for camera_id in camera_ids
        ]
    except Exception:
        cameras = None
    occupied, free, _weights = camera_known_space(
        points, cameras, BENCH_VOXEL_M, origin=EGO_BOUNDS[:, 0]
    )
    return occupied, free


def gt_centers_for_frame(frame_index: int) -> np.ndarray:
    occupied, _free = known_grids_for_frame(frame_index)
    return occupied


def score_prediction(
    name: str,
    pred_centers: np.ndarray,
    occupied_centers: np.ndarray,
    free_centers: np.ndarray | None = None,
) -> dict[str, Any]:
    free = np.zeros((0, 3), dtype=np.float32) if free_centers is None else free_centers
    counts = known_space_counts(
        pred_centers,
        occupied_centers,
        free,
        BENCH_VOXEL_M,
        origin=EGO_BOUNDS[:, 0],
        tolerance=0,
        unlabeled_is_free=free_centers is None,
    )
    return {
        "pipeline": name,
        "tp": int(counts["tp"]),
        "fp": int(counts["fp"]),
        "fn": int(counts["fn"]),
        "iou": float(counts["iou"]),
        "miou": float(counts["miou"]),
        "bands": [
            {
                "band": str(row["band"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "iou": float(row["iou"]),
            }
            for row in counts["bands"]
        ],
        "distance_zones": counts.get("distance_zones") or [],
        "miou_drop": counts.get("miou_drop") or {},
    }


def micro_summary(predict: Callable[[int], np.ndarray], frame_ids: list[int]) -> dict[str, float | int]:
    """Pool TP/FP/FN across frames, then compute IoU and the height-band mean."""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    band_names = [label for label, _class_name in HEIGHT_BANDS]
    band_tp = {name: 0 for name in band_names}
    band_fp = {name: 0 for name in band_names}
    band_fn = {name: 0 for name in band_names}
    for frame_index in frame_ids:
        occupied, free = known_grids_for_frame(int(frame_index))
        row = score_prediction("", predict(int(frame_index)), occupied, free)
        true_positive += row["tp"]
        false_positive += row["fp"]
        false_negative += row["fn"]
        for band in row["bands"]:
            band_tp[band["band"]] += band["tp"]
            band_fp[band["band"]] += band["fp"]
            band_fn[band["band"]] += band["fn"]
    band_ious: list[float] = []
    for name in band_names:
        denom = band_tp[name] + band_fp[name] + band_fn[name]
        if denom:
            band_ious.append(band_tp[name] / denom)
    denom = true_positive + false_positive + false_negative
    iou = 1.0 if denom == 0 else true_positive / denom
    miou = float(sum(band_ious) / len(band_ious)) if band_ious else iou
    return {
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "iou": float(iou),
        "miou": float(miou),
    }


def build_frame_benchmark(frame_index: int, threshold: float) -> dict[str, Any]:
    """One-frame comparison of both pipelines against camera-visible lidar."""
    from voxnet_baseline import checkpoint_id

    index = int(frame_index)
    occupied, free = known_grids_for_frame(index)
    rows: list[dict[str, Any]] = []
    try:
        mono = monocular_centers(index, float(threshold))
        rows.append(score_prediction(MONOCULAR_NAME, mono, occupied, free))
    except FileNotFoundError as exc:
        rows.append({"pipeline": MONOCULAR_NAME, "error": str(exc)})
    voxnet = predict_centers(index)
    rows.append(score_prediction(VOXNET_NAME, voxnet, occupied, free))
    runtime = load_baseline()
    heldout = runtime.get("heldout")
    return {
        "voxel_m": float(BENCH_VOXEL_M),
        "tolerance_cells": 0,
        "unknown": "ignored",
        "frame_index": index,
        "split": "held-out" if held_out(index) else "train",
        "monocular_threshold": float(threshold),
        "voxnet_threshold": float(runtime["threshold"]),
        "checkpoint_id": checkpoint_id(),
        "scoring": SCORING,
        "rows": rows,
        "heldout": heldout if isinstance(heldout, dict) else None,
    }


def evaluation_document(
    frame_index: int | None,
    heldout: bool,
    threshold: float = 0.38,
) -> dict[str, Any]:
    """Comparison table for one frame or the held-out list. Counts are uncapped."""
    from nuscenes_loader import load_manifest
    from voxnet_baseline import checkpoint_id

    manifest = load_manifest()
    document: dict[str, Any] = {
        "scene_name": manifest.get("scene_name"),
        "checkpoint_id": checkpoint_id(),
        "scoring": SCORING,
        "voxel_m": float(BENCH_VOXEL_M),
        "tolerance_cells": 0,
        "unknown": "ignored",
        "display_cap": "not applied",
    }
    if heldout:
        count = int(manifest.get("frame_count", 0))
        document["frames"] = []
        for index in range(count):
            if not held_out(index):
                continue
            try:
                document["frames"].append(build_frame_benchmark(index, threshold))
            except Exception as exc:
                document["frames"].append({"frame_index": index, "error": str(exc)})
        return document
    index = 0 if frame_index is None else int(frame_index)
    document["frame"] = build_frame_benchmark(index, threshold)
    return document


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Regenerate the occupancy comparison table.")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.38)
    args = parser.parse_args()
    print(json.dumps(evaluation_document(args.frame, args.heldout, args.threshold), indent=2))


if __name__ == "__main__":
    main()

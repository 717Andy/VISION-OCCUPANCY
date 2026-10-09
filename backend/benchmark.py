"""Score registered occupancy predictors on one lidar grid.

Both predictions are voxelized at 1.0 m and compared with exact cell equality.
The live HUD keeps its own 0.2 m score with a one-cell tolerance; this table
does not replace that number.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
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
IPM_NAME = "Ground-plane IPM"
STEREO_NAME = "Multi-view stereo"
LSS_NAME = "Lift-Splat"
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
        "precision": _ratio(int(counts["tp"]), int(counts["fp"])),
        "recall": _ratio(int(counts["tp"]), int(counts["fn"])),
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


def _ratio(numerator: int, other: int) -> float | None:
    denom = numerator + other
    if denom <= 0:
        return None
    return float(numerator / denom)


def live_centers(name: str, frame_index: int, voxel_m: float) -> np.ndarray:
    """Cell centers for the model drawn in the prediction pane."""
    if name == IPM_NAME:
        from ipm_baseline import predict_centers as ipm_centers

        return ipm_centers(frame_index, voxel_m)
    if name == STEREO_NAME:
        from stereo_baseline import predict_centers as stereo_centers

        return stereo_centers(frame_index, voxel_m)
    for registered, predict, _identity in registered_predictors():
        if registered == name:
            return predict(int(frame_index), 0.38)
    raise KeyError(name)


def registered_predictors() -> list[tuple[str, Any, Any]]:
    """Name, predict(frame, threshold) -> centers, checkpoint id callable."""
    from ipm_baseline import IPM_ID
    from ipm_baseline import predict_centers as ipm_centers
    from camera_space_baseline import CAMERA_SPACE_NAME
    from camera_space_baseline import checkpoint_id as camera_space_checkpoint_id
    from camera_space_baseline import predict_centers as camera_space_centers
    from lss_baseline import checkpoint_id as lss_checkpoint_id
    from lss_baseline import predict_centers as lss_centers
    from stereo_baseline import STEREO_ID
    from stereo_baseline import predict_centers as stereo_centers
    from voxnet_baseline import checkpoint_id as voxnet_checkpoint_id

    return [
        (MONOCULAR_NAME, monocular_centers, lambda: "midas-cache"),
        (VOXNET_NAME, lambda frame, _threshold: predict_centers(frame), voxnet_checkpoint_id),
        (IPM_NAME, lambda frame, _threshold: ipm_centers(frame), lambda: IPM_ID),
        (STEREO_NAME, lambda frame, _threshold: stereo_centers(frame), lambda: STEREO_ID),
        (LSS_NAME, lambda frame, _threshold: lss_centers(frame), lss_checkpoint_id),
        (CAMERA_SPACE_NAME, lambda frame, _threshold: camera_space_centers(frame), camera_space_checkpoint_id),
    ]


_HELDOUT: dict[str, Any] | None = None


def comparison_heldout() -> dict[str, Any]:
    """Held-out scores for every predictor that can run. Cached for the process."""
    global _HELDOUT
    if _HELDOUT is not None:
        return _HELDOUT
    from nuscenes_loader import load_manifest

    count = int(load_manifest().get("frame_count", 0))
    frame_ids = [index for index in range(count) if held_out(index)]
    scores: list[dict[str, Any]] = []
    runtime = load_baseline()
    stored = runtime.get("heldout")
    if isinstance(stored, dict):
        for key, pipeline in (("monocular", MONOCULAR_NAME), ("voxnet", VOXNET_NAME)):
            row = stored.get(key)
            if isinstance(row, dict) and "iou" in row:
                scores.append({"pipeline": pipeline, "iou": row.get("iou"), "miou": row.get("miou")})
    from ipm_baseline import predict_centers as ipm_centers
    from stereo_baseline import predict_centers as stereo_centers

    for pipeline, predict in ((IPM_NAME, ipm_centers), (STEREO_NAME, stereo_centers)):
        try:
            summary = micro_summary(predict, frame_ids)
        except Exception as exc:
            logger_warning = exc
            scores.append({"pipeline": pipeline, "error": str(logger_warning)})
            continue
        scores.append({"pipeline": pipeline, "iou": summary["iou"], "miou": summary["miou"]})
    try:
        from lss_baseline import predict_centers as lss_centers

        summary = micro_summary(lss_centers, frame_ids)
        scores.append({"pipeline": LSS_NAME, "iou": summary["iou"], "miou": summary["miou"]})
    except Exception as exc:
        scores.append({"pipeline": LSS_NAME, "error": str(exc)})
    try:
        from camera_space_baseline import CAMERA_SPACE_NAME as camera_name
        from camera_space_baseline import predict_centers as camera_space_centers

        summary = micro_summary(camera_space_centers, frame_ids)
        scores.append({"pipeline": camera_name, "iou": summary["iou"], "miou": summary["miou"]})
    except Exception as exc:
        scores.append({"pipeline": "Camera-space", "error": str(exc)})
    _HELDOUT = {"frames": len(frame_ids), "scores": scores}
    return _HELDOUT


def build_frame_benchmark(frame_index: int, threshold: float) -> dict[str, Any]:
    """One-frame comparison of both pipelines against camera-visible lidar."""
    from voxnet_baseline import checkpoint_id

    index = int(frame_index)
    occupied, free = known_grids_for_frame(index)
    rows: list[dict[str, Any]] = []
    for name, predict, identity in registered_predictors():
        try:
            started = time.perf_counter()
            centers = predict(index, float(threshold))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            row = score_prediction(name, centers, occupied, free)
            row["checkpoint_id"] = identity()
            row["latency_ms"] = float(elapsed_ms)
            rows.append(row)
        except Exception as exc:
            rows.append({"pipeline": name, "error": str(exc)})
    runtime = load_baseline()
    stored = runtime.get("heldout")
    heldout: dict[str, Any] = dict(stored) if isinstance(stored, dict) else {}
    summary = comparison_heldout()
    heldout["frames"] = summary["frames"]
    heldout["scores"] = summary["scores"]
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


RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"


def save_run(document: dict[str, Any]) -> dict[str, Any]:
    """Write one evaluation document and return it with an id."""
    from datetime import datetime, timezone

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint = str(document.get("checkpoint_id") or "none")
    run_id = f"{stamp}_{checkpoint}"
    stored = {"id": run_id, **document}
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(stored))
    return stored


def list_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    summaries = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        frame = data.get("frame") or {}
        summaries.append(
            {
                "id": data.get("id", path.stem),
                "scene_name": data.get("scene_name"),
                "checkpoint_id": data.get("checkpoint_id"),
                "heldout": "frames" in data and "frame" not in data,
                "frame_index": frame.get("frame_index"),
                "rows": [
                    {
                        "pipeline": row.get("pipeline"),
                        "iou": row.get("iou"),
                        "tp": row.get("tp"),
                        "fp": row.get("fp"),
                        "fn": row.get("fn"),
                    }
                    for row in frame.get("rows", [])
                    if isinstance(row, dict)
                ],
            }
        )
    return summaries


def load_run(run_id: str) -> dict[str, Any]:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(run_id)
    return json.loads(path.read_text())


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Regenerate the occupancy comparison table.")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.38)
    args = parser.parse_args()
    document = evaluation_document(args.frame, args.heldout, args.threshold)
    saved = save_run(document)
    document["run_id"] = saved["id"]
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()

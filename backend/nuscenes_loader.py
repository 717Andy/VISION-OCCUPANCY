"""Load a prepared nuScenes camera clip from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent / "data" / "nuscenes"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

# Native nuScenes camera resolution; prepared JPEGs are resized copies.
ORIGINAL_IMAGE_SIZE = (1600, 900)
PREPARED_IMAGE_SIZE = (640, 360)

CAMERA_IDS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


@dataclass(frozen=True)
class CameraSample:
    """One camera at one keyframe: RGB path plus calibrated K / extrinsics."""

    camera_id: str
    image_path: Path
    intrinsic: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    timestamp: int


@dataclass(frozen=True)
class SynchronizedFrame:
    """Six surround cameras captured at the same nuScenes sample timestep."""

    index: int
    timestamp: int
    time_s: float
    cameras: dict[str, CameraSample]


class ClipNotPrepared(FileNotFoundError):
    """Raised when the local nuScenes clip has not been extracted yet."""


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ClipNotPrepared(f"Missing {MANIFEST_PATH}. Run backend/prepare_nuscenes.py")
    return json.loads(MANIFEST_PATH.read_text())


def reload_manifest() -> dict[str, Any]:
    load_manifest.cache_clear()
    return load_manifest()


def resolve_frame_path(frame_index: int, camera_id: str) -> Path:
    if camera_id not in CAMERA_IDS:
        raise KeyError(f"Unknown camera '{camera_id}'")

    manifest = load_manifest()
    frames = manifest["frames"]
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"Frame {frame_index} out of range")

    relative = frames[frame_index]["cameras"].get(camera_id)
    if not relative:
        raise FileNotFoundError(f"No image for {camera_id} at frame {frame_index}")

    path = (DATA_ROOT / relative).resolve()
    if DATA_ROOT.resolve() not in path.parents and path != DATA_ROOT.resolve():
        raise ValueError("Refusing to serve path outside the nuScenes data root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_synchronized_frame(frame_index: int) -> SynchronizedFrame:
    """Load all six camera images plus per-camera K / extrinsics for a timestep."""
    manifest = load_manifest()
    frames = manifest["frames"]
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"Frame {frame_index} out of range")

    row = frames[frame_index]
    calibration = row.get("calibration") or {}
    cameras: dict[str, CameraSample] = {}
    for camera_id in CAMERA_IDS:
        calib = calibration.get(camera_id)
        if not calib:
            raise FileNotFoundError(f"Missing calibration for {camera_id} at frame {frame_index}")
        intrinsic = np.asarray(calib["intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"Intrinsic K for {camera_id} must be 3×3, got {intrinsic.shape}")
        cameras[camera_id] = CameraSample(
            camera_id=camera_id,
            image_path=resolve_frame_path(frame_index, camera_id),
            intrinsic=intrinsic,
            translation=np.asarray(calib["translation"], dtype=np.float64).reshape(3),
            rotation=np.asarray(calib["rotation"], dtype=np.float64).reshape(4),
            timestamp=int(calib.get("timestamp", row["timestamp"])),
        )

    if len(cameras) != len(CAMERA_IDS):
        raise FileNotFoundError(f"Expected 6 cameras at frame {frame_index}, got {len(cameras)}")

    return SynchronizedFrame(
        index=int(row.get("index", frame_index)),
        timestamp=int(row["timestamp"]),
        time_s=float(row.get("time_s", 0.0)),
        cameras=cameras,
    )

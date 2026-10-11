"""Load a prepared nuScenes camera clip from disk."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent / "data" / "nuscenes"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
TRAINING_SCENE = "scene-0103"
_SCENE: ContextVar[str] = ContextVar("nuscenes_scene", default=TRAINING_SCENE)

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


def get_scene() -> str:
    return _SCENE.get()


def set_scene(name: str) -> object:
    return _SCENE.set(name)


def reset_scene(token: object) -> None:
    _SCENE.reset(token)  # type: ignore[arg-type]


@contextmanager
def scene_context(name: str | None) -> Iterator[None]:
    if not name:
        yield
        return
    token = set_scene(name)
    try:
        yield
    finally:
        reset_scene(token)


def scene_root(name: str | None = None) -> Path:
    """scene-0103 stays at the original data root. Later scenes get a folder."""
    scene = name or get_scene()
    if scene == TRAINING_SCENE:
        return DATA_ROOT
    return DATA_ROOT / "scenes" / scene


def list_scenes() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    roots = [DATA_ROOT]
    extra = DATA_ROOT / "scenes"
    if extra.is_dir():
        roots.extend(sorted(path for path in extra.iterdir() if path.is_dir()))
    for root in roots:
        path = root / "manifest.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        name = str(data.get("scene_name") or root.name)
        found.append(
            {
                "name": name,
                "description": data.get("description") or "",
                "location": data.get("location") or "",
                "frame_count": int(data.get("frame_count") or 0),
                "training_scene": bool(data.get("training_scene", name == TRAINING_SCENE)),
            }
        )
    return found


@lru_cache(maxsize=8)
def _load_manifest(scene: str) -> dict[str, Any]:
    path = scene_root(scene) / "manifest.json"
    if not path.exists():
        raise ClipNotPrepared(f"Missing {path}. Run backend/prepare_nuscenes.py")
    return json.loads(path.read_text())


def load_manifest() -> dict[str, Any]:
    return _load_manifest(get_scene())


def reload_manifest() -> dict[str, Any]:
    _load_manifest.cache_clear()
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

    path = (scene_root() / relative).resolve()
    root = scene_root().resolve()
    if root not in path.parents and path != root:
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

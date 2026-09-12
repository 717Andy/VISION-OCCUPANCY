"""Load a prepared nuScenes camera clip from disk."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).resolve().parent / "data" / "nuscenes"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

CAMERA_IDS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


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

#!/usr/bin/env python3
"""Download nuScenes mini and extract one synchronized 6-camera clip."""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "nuscenes"
CLIP_DIR = DATA_ROOT / "frames"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
ARCHIVE_PATH = Path("/tmp/nuscenes/v1.0-mini.tgz")
EXTRACT_ROOT = Path("/tmp/nuscenes/extract")
DOWNLOAD_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"

CAMERA_IDS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)
TARGET_SIZE = (640, 360)


def download_archive() -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 4_000_000_000:
        print(f"Using existing archive {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size} bytes)")
        return
    print(f"Downloading {DOWNLOAD_URL} -> {ARCHIVE_PATH}")
    urllib.request.urlretrieve(DOWNLOAD_URL, ARCHIVE_PATH)


def load_table(meta_dir: Path, name: str) -> list[dict]:
    return json.loads((meta_dir / f"{name}.json").read_text())


def index_by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def choose_scene(scenes: list[dict]) -> dict:
    """Prefer a ~20s keyframe clip (nuScenes keyframes are 2 Hz)."""
    return min(scenes, key=lambda scene: abs(int(scene.get("nbr_samples", 0)) - 40))


def collect_samples(samples: list[dict], scene_token: str) -> list[dict]:
    linked = [sample for sample in samples if sample["scene_token"] == scene_token]
    linked.sort(key=lambda sample: sample["timestamp"])
    return linked


def archive_index(tar: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], dict[str, tarfile.TarInfo]]:
    json_members: list[tarfile.TarInfo] = []
    samples_index: dict[str, tarfile.TarInfo] = {}
    for member in tar.getmembers():
        name = member.name.lstrip("./")
        if name.endswith(".json") and "v1.0-mini/" in name:
            json_members.append(member)
        elif name.endswith(".jpg") and "samples/" in name:
            rel = name[name.find("samples/") :]
            samples_index[rel] = member
    return json_members, samples_index


def extract_clip(tar: tarfile.TarFile) -> dict:
    json_members, samples_index = archive_index(tar)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    tar.extractall(EXTRACT_ROOT, members=json_members, filter="data")

    matches = list(EXTRACT_ROOT.rglob("scene.json"))
    if not matches:
        raise FileNotFoundError("Could not find v1.0-mini metadata in archive")
    meta_dir = matches[0].parent

    scenes = load_table(meta_dir, "scene")
    samples = load_table(meta_dir, "sample")
    sample_data_rows = load_table(meta_dir, "sample_data")
    calibrated = index_by_token(load_table(meta_dir, "calibrated_sensor"))
    sensors = index_by_token(load_table(meta_dir, "sensor"))
    logs = index_by_token(load_table(meta_dir, "log"))

    scene = choose_scene(scenes)
    scene_samples = collect_samples(samples, scene["token"])
    location = logs.get(scene.get("log_token", ""), {}).get("location", "")
    scene_sample_tokens = {sample["token"] for sample in scene_samples}

    cameras_by_sample: dict[str, dict[str, dict]] = {token: {} for token in scene_sample_tokens}
    for row in sample_data_rows:
        if row["sample_token"] not in scene_sample_tokens:
            continue
        if not row.get("is_key_frame"):
            continue
        if row.get("fileformat") != "jpg":
            continue
        calib = calibrated.get(row["calibrated_sensor_token"])
        if not calib:
            continue
        sensor = sensors.get(calib["sensor_token"])
        if not sensor or sensor.get("channel") not in CAMERA_IDS:
            continue
        cameras_by_sample[row["sample_token"]][sensor["channel"]] = row

    needed: list[tarfile.TarInfo] = []
    planned: list[tuple[int, str, dict, tarfile.TarInfo]] = []
    for index, sample in enumerate(scene_samples):
        for camera_id in CAMERA_IDS:
            row = cameras_by_sample.get(sample["token"], {}).get(camera_id)
            if not row:
                continue
            filename = row["filename"].replace("\\", "/")
            rel = filename[filename.find("samples/") :] if "samples/" in filename else filename
            member = samples_index.get(rel)
            if member is None:
                print(f"Missing archive member for {filename}")
                continue
            needed.append(member)
            planned.append((index, camera_id, row, member))

    unique_needed = list({member.name: member for member in needed}.values())
    unique_needed.sort(key=lambda member: member.offset)
    print(f"Extracting {len(unique_needed)} camera JPEGs for {scene.get('name')}...")
    tar.extractall(EXTRACT_ROOT, members=unique_needed, filter="data")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = [{"index": i, "timestamp": s["timestamp"], "cameras": {}, "calibration": {}} for i, s in enumerate(scene_samples)]
    first_ts = scene_samples[0]["timestamp"] if scene_samples else 0

    for index, camera_id, row, member in planned:
        src = EXTRACT_ROOT / member.name.lstrip("./")
        if not src.exists():
            # tar may strip a leading folder
            alt = EXTRACT_ROOT / Path(member.name).name
            candidates = list(EXTRACT_ROOT.rglob(Path(member.name).name))
            src = candidates[0] if candidates else src
        if not src.exists():
            print(f"Extracted file missing: {member.name}")
            continue

        frame_dir = CLIP_DIR / f"{index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        dest = frame_dir / f"{camera_id}.jpg"
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb = rgb.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            rgb.save(dest, format="JPEG", quality=82, optimize=True)

        calib = calibrated[row["calibrated_sensor_token"]]
        frames[index]["time_s"] = (frames[index]["timestamp"] - first_ts) / 1e6
        frames[index]["cameras"][camera_id] = str(dest.relative_to(DATA_ROOT))
        frames[index]["calibration"][camera_id] = {
            "intrinsic": calib.get("camera_intrinsic"),
            "translation": calib.get("translation"),
            "rotation": calib.get("rotation"),
            "timestamp": row["timestamp"],
        }

    duration_s = frames[-1].get("time_s", 0.0) if frames else 0.0
    sample_hz = 2.0 if duration_s <= 0 else (len(frames) - 1) / duration_s

    manifest = {
        "scene_name": scene.get("name", "unknown"),
        "description": scene.get("description", ""),
        "location": location,
        "scene_token": scene["token"],
        "duration_s": round(duration_s, 3),
        "sample_hz": round(sample_hz, 3),
        "frame_count": len(frames),
        "cameras": [
            {"id": cam, "label": cam.replace("CAM_", "").replace("_", " ").title()}
            for cam in CAMERA_IDS
        ],
        "frames": frames,
    }
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(frames)} frames from {manifest['scene_name']} ({location}) -> {MANIFEST_PATH}")
    return manifest


def main() -> None:
    download_archive()
    print("Opening archive (this can take a minute)...")
    with tarfile.open(ARCHIVE_PATH, "r:*") as tar:
        extract_clip(tar)


if __name__ == "__main__":
    main()

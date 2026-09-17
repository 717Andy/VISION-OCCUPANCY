import asyncio
import json
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from nuscenes_loader import (
    CAMERA_IDS,
    ORIGINAL_IMAGE_SIZE,
    PREPARED_IMAGE_SIZE,
    ClipNotPrepared,
    load_manifest,
    resolve_frame_path,
)
from pipeline import PerceptionPipeline

pipeline = PerceptionPipeline()


def _warmup_worker() -> None:
    pipeline.warmup()
    try:
        manifest = load_manifest()
        pipeline.precompute_depth_cache(int(manifest.get("frame_count", 0)))
    except ClipNotPrepared:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warmup_worker, daemon=True, name="midas-warmup").start()
    yield


app = FastAPI(title="VisionOccupancy R&D Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gpu_label() -> str:
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=1,
                text=True,
            )
            return f"{int(out.strip().splitlines()[0])}%"
    except Exception:
        pass
    if pipeline.engine.using_cuda:
        return "CUDA"
    return "CPU"


@app.get("/api/scene")
def get_scene():
    try:
        manifest = load_manifest()
        manifest["original_image_size"] = list(ORIGINAL_IMAGE_SIZE)
        manifest["prepared_image_size"] = list(PREPARED_IMAGE_SIZE)
        return manifest
    except ClipNotPrepared as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/frames/{frame_index}/cameras/{camera_id}")
def get_camera_frame(frame_index: int, camera_id: str):
    if camera_id not in CAMERA_IDS:
        raise HTTPException(status_code=404, detail="Unknown camera")
    try:
        path = resolve_frame_path(frame_index, camera_id)
    except ClipNotPrepared as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/telemetry")
def telemetry():
    scene_name = None
    frame_count = 0
    try:
        manifest = load_manifest()
        scene_name = manifest.get("scene_name")
        frame_count = manifest.get("frame_count", 0)
    except ClipNotPrepared:
        pass
    return {
        "gpu": gpu_label(),
        "device": pipeline.engine.device,
        "scene_name": scene_name,
        "frame_count": frame_count,
        "voxel_source": pipeline.voxel_source,
        "midas_ready": pipeline.engine.available,
                        "last_depth_ms": pipeline.last_depth_ms,
        "last_project_ms": pipeline.last_project_ms,
        "last_miou": pipeline.last_miou,
        "last_frame_index": pipeline.last_frame_index,
    }


@app.get("/api/health")
def health():
    clip_ready = Path(__file__).resolve().parent.joinpath("data/nuscenes/manifest.json").exists()
    return {
        "ok": True,
        "clip_ready": clip_ready,
        "midas_ready": pipeline.engine.available,
        "voxel_source": pipeline.voxel_source,
        "device": pipeline.engine.device,
    }


@app.websocket("/ws/occupancy")
async def occupancy_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to VisionOccupancy Pipeline WebSocket.")

    threshold = 0.38
    voxel_size = 0.2
    frame_index = 0
    is_paused = False
    dirty = True

    async def receive_controls():
        nonlocal threshold, voxel_size, frame_index, is_paused, dirty
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                if "threshold" in data or "occupancy_threshold" in data:
                    threshold = float(data.get("occupancy_threshold", data.get("threshold")))
                    dirty = True
                if "voxel_size" in data:
                    voxel_size = float(data["voxel_size"])
                    dirty = True
                if "frame_index" in data:
                    frame_index = max(int(data["frame_index"]), 0)
                    dirty = True
                if "paused" in data:
                    is_paused = bool(data["paused"])
        except Exception:
            pass

    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            if dirty and not is_paused:
                idx = frame_index
                vs = voxel_size
                th = threshold
                dirty = False
                start_time = time.perf_counter()
                payload = await asyncio.to_thread(
                    pipeline.occupancy_for_frame,
                    idx,
                    vs,
                    th,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "occupancy_meta",
                            "frame_index": idx,
                            "voxel_source": pipeline.voxel_source,
                            "depth_ms": pipeline.last_depth_ms,
                            "project_ms": pipeline.last_project_ms,
                            "elapsed_ms": elapsed_ms,
                            "device": pipeline.engine.device,
                            "miou": pipeline.last_miou,
                        }
                    )
                )
                await websocket.send_bytes(payload)
            else:
                await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        control_task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

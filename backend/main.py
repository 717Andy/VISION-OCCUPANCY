import asyncio
import json
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from nuscenes_loader import CAMERA_IDS, ClipNotPrepared, load_manifest, resolve_frame_path

app = FastAPI(title="VisionOccupancy R&D Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRID_X, GRID_Y, GRID_Z = 32, 32, 16


class PerceptionPipeline:
    """Simulates 2D-to-3D depth projection & voxelization with a trigonometric occupancy wave."""

    def __init__(self) -> None:
        self.frame_count = 0

    def generate_occupancy_tensor(self, threshold: float = 0.4) -> bytes:
        self.frame_count += 1
        time_factor = self.frame_count * 0.1

        binary_data = bytearray()
        active_count = 0

        for z in range(GRID_Z):
            for y in range(GRID_Y):
                for x in range(GRID_X):
                    nx, ny, nz = (x - 16) / 16.0, (y - 16) / 16.0, z / 16.0
                    prob = (math.sin(nx * 3.0 + time_factor) * math.cos(ny * 3.0 + time_factor) + 1.0) / 2.0
                    if prob >= threshold:
                        binary_data.extend(
                            struct.pack("<ffff", float(x - 16), float(y - 16), float(z), prob)
                        )
                        active_count += 1

        header = struct.pack("<I", active_count)
        return header + binary_data


pipeline = PerceptionPipeline()


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
    return "CPU"


@app.get("/api/scene")
def get_scene():
    try:
        return load_manifest()
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
        "scene_name": scene_name,
        "frame_count": frame_count,
        "voxel_source": "trigonometric-wave",
    }


@app.get("/api/health")
def health():
    clip_ready = Path(__file__).resolve().parent.joinpath("data/nuscenes/manifest.json").exists()
    return {"ok": True, "clip_ready": clip_ready}


@app.websocket("/ws/occupancy")
async def occupancy_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to VisionOccupancy Pipeline WebSocket.")

    threshold = 0.38
    is_paused = False

    async def receive_controls():
        nonlocal threshold, is_paused
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                if "threshold" in data:
                    threshold = float(data["threshold"])
                if "paused" in data:
                    is_paused = bool(data["paused"])
        except Exception:
            pass

    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            if not is_paused:
                start_time = time.perf_counter()
                payload = pipeline.generate_occupancy_tensor(threshold=threshold)
                await websocket.send_bytes(payload)
                elapsed = time.perf_counter() - start_time
                await asyncio.sleep(max(0.0, 0.033 - elapsed))
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        control_task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

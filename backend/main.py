import asyncio
import json
import math
import struct
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VisionOccupancy R&D Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Parameters
GRID_X, GRID_Y, GRID_Z = 32, 32, 16
NUM_VOXELS = GRID_X * GRID_Y * GRID_Z

class PerceptionPipeline:
    """Simulates 2D-to-3D depth projection & voxelization pipeline."""
    def __init__(self):
        self.frame_count = 0

    def generate_occupancy_tensor(self, threshold: float = 0.4) -> bytes:
        """
        Generates 3D voxel grid tensor: [X, Y, Z, Probability, Distance]
        Packs active voxels into binary Float32Array format for low latency.
        """
        self.frame_count += 1
        time_factor = self.frame_count * 0.1
        
        binary_data = bytearray()
        active_count = 0

        # Simulate camera frustum depth wave projection
        for z in range(GRID_Z):
            for y in range(GRID_Y):
                for x in range(GRID_X):
                    # Normalized spatial coordinates
                    nx, ny, nz = (x - 16) / 16.0, (y - 16) / 16.0, z / 16.0
                    dist = math.sqrt(nx**2 + ny**2 + nz**2)
                    
                    # Simulated monocular depth probability wave
                    prob = (math.sin(nx * 3.0 + time_factor) * 
                            math.cos(ny * 3.0 + time_factor) + 1.0) / 2.0
                    
                    # Apply user threshold
                    if prob >= threshold:
                        # Pack 5 float32 fields per voxel: [x, y, z, prob, dist]
                        binary_data.extend(struct.pack('<ffff', float(x - 16), float(y - 16), float(z), prob))
                        active_count += 1

        # Header: Pack total active voxel count as unsigned int (4 bytes)
        header = struct.pack('<I', active_count)
        return header + binary_data

pipeline = PerceptionPipeline()

@app.websocket("/ws/occupancy")
async def occupancy_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to VisionOccupancy Pipeline WebSocket.")
    
    threshold = 0.4
    is_paused = False

    async def receive_controls():
        nonlocal threshold, is_paused
        try:
            while True:
                # Listen for client control updates (JSON text frames)
                message = await websocket.receive_text()
                data = json.loads(message)
                if "threshold" in data:
                    threshold = float(data["threshold"])
                if "paused" in data:
                    is_paused = bool(data["paused"])
        except Exception:
            pass

    # Run listener as background task
    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            if not is_paused:
                start_time = time.perf_counter()
                
                # Generate binary payload
                payload = pipeline.generate_occupancy_tensor(threshold=threshold)
                
                # Send binary data frame to client
                await websocket.send_bytes(payload)
                
                # Target ~30 FPS server output
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
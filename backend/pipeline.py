"""Occupancy pipeline: trigonometric fallback plus MiDaS + Open3D path."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from midas_engine import MidasDepthEngine
from nuscenes_loader import CAMERA_IDS, ClipNotPrepared, load_rgb, load_synchronized_frame
from projection import pack_occupancy, project_cameras_to_voxels

logger = logging.getLogger(__name__)

GRID_X, GRID_Y, GRID_Z = 32, 32, 16
OCCUPANCY_CACHE_LIMIT = 48


class PerceptionPipeline:
    """2D-to-3D occupancy: MiDaS depth + Open3D voxels, with a wave fallback."""

    def __init__(self, depth_engine: MidasDepthEngine | None = None) -> None:
        self.frame_count = 0
        self.last_depth_ms = 0.0
        self.last_project_ms = 0.0
        self.last_frame_index = 0
        self.voxel_source = "trigonometric-wave"
        self.engine = depth_engine if depth_engine is not None else MidasDepthEngine()
        self._occupancy_cache: OrderedDict[tuple[int, float, float], bytes] = OrderedDict()
        self._cache_lock = threading.Lock()

        x = np.arange(GRID_X, dtype=np.float32)
        y = np.arange(GRID_Y, dtype=np.float32)
        z = np.arange(GRID_Z, dtype=np.float32)
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing="ij")
        self.nx = (self.xx - 16.0) / 16.0
        self.ny = (self.yy - 16.0) / 16.0

    def generate_occupancy_tensor(self, threshold: float = 0.4) -> bytes:
        self.frame_count += 1
        time_factor = np.float32(self.frame_count * 0.1)
        prob = (np.sin(self.nx * 3.0 + time_factor) * np.cos(self.ny * 3.0 + time_factor) + 1.0) / 2.0
        mask = prob >= np.float32(threshold)

        packed = np.column_stack(
            (
                self.xx[mask] - 16.0,
                self.yy[mask] - 16.0,
                self.zz[mask],
                prob[mask],
            )
        ).astype(np.float32, copy=False)

        header = np.array([packed.shape[0]], dtype=np.uint32)
        return header.tobytes() + packed.tobytes()

    def occupancy_for_frame(
        self,
        frame_index: int,
        voxel_size: float = 0.2,
        threshold: float = 0.38,
    ) -> bytes:
        """Build occupancy for a synchronized 6-camera nuScenes timestep."""
        key = (int(frame_index), round(float(voxel_size), 3), round(float(threshold), 3))
        with self._cache_lock:
            cached = self._occupancy_cache.get(key)
            if cached is not None:
                self._occupancy_cache.move_to_end(key)
                self.last_frame_index = int(frame_index)
                self.last_depth_ms = 0.0
                self.last_project_ms = 0.0
                return cached

        if not self.engine.ensure_loaded():
            self.voxel_source = "trigonometric-wave"
            return self.generate_occupancy_tensor(threshold=threshold)

        try:
            payload = self._midas_occupancy(frame_index, voxel_size, threshold)
            self.voxel_source = "midas-open3d"
        except ClipNotPrepared:
            self.voxel_source = "trigonometric-wave"
            payload = self.generate_occupancy_tensor(threshold=threshold)
        except Exception as exc:
            logger.warning("MiDaS occupancy failed for frame %s: %s", frame_index, exc)
            self.voxel_source = "trigonometric-wave"
            payload = self.generate_occupancy_tensor(threshold=threshold)

        with self._cache_lock:
            self._occupancy_cache[key] = payload
            self._occupancy_cache.move_to_end(key)
            while len(self._occupancy_cache) > OCCUPANCY_CACHE_LIMIT:
                self._occupancy_cache.popitem(last=False)
        self.last_frame_index = int(frame_index)
        return payload

    def _midas_occupancy(self, frame_index: int, voxel_size: float, threshold: float) -> bytes:
        synced = load_synchronized_frame(frame_index)
        payloads: list[dict[str, Any]] = []
        depth_ms = 0.0
        for camera_id in CAMERA_IDS:
            sample = synced.cameras[camera_id]
            rgb = load_rgb(sample.image_path)
            disparity = self.engine.infer_cached(frame_index, camera_id, rgb)
            depth_ms += float(self.engine.last_infer_ms)
            payloads.append(
                {
                    "disparity": disparity,
                    "intrinsic": sample.intrinsic,
                    "translation": sample.translation,
                    "rotation": sample.rotation,
                }
            )

        start = time.perf_counter()
        centers, occupancy = project_cameras_to_voxels(payloads, voxel_size, threshold)
        self.last_project_ms = (time.perf_counter() - start) * 1000.0
        self.last_depth_ms = depth_ms
        return pack_occupancy(centers, occupancy)

    def warmup(self) -> None:
        """Load MiDaS, touch Open3D, and cache depth for frame 0."""
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
            o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)
        except Exception as exc:
            logger.warning("Open3D warmup failed: %s", exc)
        if not self.engine.ensure_loaded():
            return
        try:
            self.occupancy_for_frame(0)
        except Exception as exc:
            logger.warning("Pipeline warmup failed: %s", exc)

    def precompute_depth_cache(self, frame_count: int) -> None:
        if not self.engine.ensure_loaded():
            return
        for index in range(max(int(frame_count), 0)):
            try:
                synced = load_synchronized_frame(index)
            except Exception:
                return
            for camera_id in CAMERA_IDS:
                sample = synced.cameras[camera_id]
                try:
                    self.engine.infer_cached(index, camera_id, load_rgb(sample.image_path))
                except Exception as exc:
                    logger.warning("Depth cache failed %s/%s: %s", index, camera_id, exc)
                    return

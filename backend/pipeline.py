"""Occupancy pipeline: trigonometric fallback plus MiDaS + Open3D path."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from lidar_gt import load_lidar_ego_points
from midas_engine import MidasDepthEngine
from nuscenes_loader import CAMERA_IDS, ClipNotPrepared, load_rgb, load_synchronized_frame
from projection import (
    EGO_BOUNDS,
    discrepancy_voxels,
    grid_miou,
    pack_occupancy,
    pack_occupancy_pair,
    project_cameras_to_voxel_pair,
    voxelize_lidar,
)

logger = logging.getLogger(__name__)

GRID_X, GRID_Y, GRID_Z = 32, 32, 16
OCCUPANCY_CACHE_LIMIT = 48


class PerceptionPipeline:
    """2D-to-3D occupancy: MiDaS depth + Open3D voxels, with a wave fallback."""

    def __init__(self, depth_engine: MidasDepthEngine | None = None) -> None:
        self.frame_count = 0
        self.last_depth_ms = 0.0
        self.last_project_ms = 0.0
        self.last_miou = 0.0
        self.last_frame_index = 0
        self.voxel_source = "trigonometric-wave"
        self.gt_source = "unavailable"
        self.engine = depth_engine if depth_engine is not None else MidasDepthEngine()
        self._occupancy_cache: OrderedDict[tuple[int, float, float], tuple[bytes, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

        x = np.arange(GRID_X, dtype=np.float32)
        y = np.arange(GRID_Y, dtype=np.float32)
        z = np.arange(GRID_Z, dtype=np.float32)
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing="ij")
        self.nx = (self.xx - 16.0) / 16.0
        self.ny = (self.yy - 16.0) / 16.0

    def _wave_voxels(self, threshold: float, time_factor: np.float32) -> tuple[np.ndarray, np.ndarray]:
        prob = (np.sin(self.nx * 3.0 + time_factor) * np.cos(self.ny * 3.0 + time_factor) + 1.0) / 2.0
        mask = prob >= np.float32(threshold)
        centers = np.column_stack(
            (
                self.xx[mask] - 16.0,
                self.yy[mask] - 16.0,
                self.zz[mask],
            )
        ).astype(np.float32, copy=False)
        occupancy = prob[mask].astype(np.float32, copy=False)
        return centers, occupancy

    def generate_occupancy_tensor(self, threshold: float = 0.4) -> bytes:
        self.frame_count += 1
        time_factor = np.float32(self.frame_count * 0.1)
        centers, occupancy = self._wave_voxels(threshold, time_factor)
        return pack_occupancy(centers, occupancy)

    def _pack_with_lidar(
        self,
        frame_index: int,
        pred_centers: np.ndarray,
        pred_occ: np.ndarray,
        voxel_size: float,
        lidar_points: np.ndarray | None = None,
    ) -> bytes:
        points = load_lidar_ego_points(frame_index) if lidar_points is None else lidar_points
        gt_centers, gt_occ = voxelize_lidar(points, voxel_size)
        self.last_miou = grid_miou(pred_centers, gt_centers, voxel_size, tolerance=1)
        self.gt_source = "lidar-top" if gt_centers.shape[0] else "unavailable"
        err_centers, err_occ = discrepancy_voxels(
            pred_centers, gt_centers, voxel_size, origin=EGO_BOUNDS[:, 0]
        )
        return pack_occupancy_pair(pred_centers, pred_occ, gt_centers, gt_occ, err_centers, err_occ)

    def generate_occupancy_pair(
        self,
        threshold: float = 0.4,
        frame_index: int = 0,
        voxel_size: float = 1.0,
        lidar_points: np.ndarray | None = None,
    ) -> bytes:
        """Wave prediction plus lidar ground truth when the depth model is unavailable."""
        self.frame_count += 1
        time_factor = np.float32(self.frame_count * 0.1)
        pred_centers, pred_occ = self._wave_voxels(threshold, time_factor)
        return self._pack_with_lidar(
            frame_index, pred_centers, pred_occ, voxel_size, lidar_points
        )

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
                payload, miou = cached
                self.last_frame_index = int(frame_index)
                self.last_depth_ms = 0.0
                self.last_project_ms = 0.0
                self.last_miou = miou
                return payload

        if not self.engine.ensure_loaded():
            self.voxel_source = "trigonometric-wave"
            payload = self.generate_occupancy_pair(
                threshold=threshold, frame_index=frame_index, voxel_size=voxel_size
            )
        else:
            try:
                payload = self._midas_occupancy(frame_index, voxel_size, threshold)
                self.voxel_source = "midas-open3d"
            except ClipNotPrepared:
                self.voxel_source = "trigonometric-wave"
                payload = self.generate_occupancy_pair(
                    threshold=threshold, frame_index=frame_index, voxel_size=voxel_size
                )
            except Exception as exc:
                logger.warning("MiDaS occupancy failed for frame %s: %s", frame_index, exc)
                self.voxel_source = "trigonometric-wave"
                payload = self.generate_occupancy_pair(
                    threshold=threshold, frame_index=frame_index, voxel_size=voxel_size
                )

        with self._cache_lock:
            self._occupancy_cache[key] = (payload, float(self.last_miou))
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
        pred_c, pred_o, gt_c, gt_o, err_c, err_o, miou = project_cameras_to_voxel_pair(
            payloads,
            load_lidar_ego_points(frame_index),
            voxel_size,
            threshold,
        )
        self.last_project_ms = (time.perf_counter() - start) * 1000.0
        self.last_depth_ms = depth_ms
        self.last_miou = miou
        self.gt_source = "lidar-top" if gt_c.shape[0] else "unavailable"
        return pack_occupancy_pair(pred_c, pred_o, gt_c, gt_o, err_c, err_o)

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

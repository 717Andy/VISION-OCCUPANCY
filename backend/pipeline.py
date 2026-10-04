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
    camera_known_space,
    discrepancy_voxels,
    display_label_clouds,
    known_space_counts,
    known_space_error_voxels,
    pack_occupancy,
    pack_occupancy_pair,
    project_cameras_to_voxel_pair,
    _cap_voxels,
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
        self.last_metric: dict[str, Any] | None = None
        self.last_frame_index = 0
        self.last_benchmark: dict[str, Any] | None = None
        self.voxel_source = "trigonometric-wave"
        self.gt_source = "unavailable"
        self.engine = depth_engine if depth_engine is not None else MidasDepthEngine()
        self._occupancy_cache: OrderedDict[tuple[int, float, float], tuple[bytes, float, dict[str, Any]]] = OrderedDict()
        self._benchmark_cache: OrderedDict[tuple[int, float], dict[str, Any]] = OrderedDict()
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

    def _cameras_for_frame(self, frame_index: int) -> list[dict[str, Any]] | None:
        try:
            frame = load_synchronized_frame(int(frame_index))
        except Exception:
            return None
        return [
            {
                "intrinsic": frame.cameras[camera_id].intrinsic,
                "rotation": frame.cameras[camera_id].rotation,
                "translation": frame.cameras[camera_id].translation,
            }
            for camera_id in CAMERA_IDS
        ]

    def _store_score(self, score: dict[str, Any], voxel_size: float, unlabeled_is_free: bool) -> None:
        self.last_miou = float(score["iou"])
        self.last_metric = {
            "iou": float(score["iou"]),
            "height_band_miou": float(score["miou"]),
            "tp": int(score["tp"]),
            "fp": int(score["fp"]),
            "fn": int(score["fn"]),
            "grid_m": float(voxel_size),
            "tolerance_cells": int(score["tolerance_cells"]),
            "unknown": "ignored" if not unlabeled_is_free else "treated_as_free",
            "bands": score["bands"],
        }

    def _record_known_score(
        self,
        pred_centers: np.ndarray,
        occupied: np.ndarray,
        free: np.ndarray,
        voxel_size: float,
        *,
        unlabeled_is_free: bool,
    ) -> None:
        score = known_space_counts(
            pred_centers,
            occupied,
            free,
            voxel_size,
            origin=EGO_BOUNDS[:, 0],
            tolerance=1,
            unlabeled_is_free=unlabeled_is_free,
        )
        self._store_score(score, voxel_size, unlabeled_is_free)

    def _pack_with_lidar(
        self,
        frame_index: int,
        pred_centers: np.ndarray,
        pred_occ: np.ndarray,
        voxel_size: float,
        lidar_points: np.ndarray | None = None,
    ) -> bytes:
        points = load_lidar_ego_points(frame_index) if lidar_points is None else lidar_points
        cameras = self._cameras_for_frame(frame_index)
        occupied, free, weights = camera_known_space(
            points, cameras, voxel_size, origin=EGO_BOUNDS[:, 0]
        )
        self._record_known_score(
            pred_centers,
            occupied,
            free,
            voxel_size,
            unlabeled_is_free=cameras is None,
        )
        self.gt_source = "lidar-visible" if occupied.shape[0] else "unavailable"
        if cameras is None:
            err_centers, err_occ = discrepancy_voxels(
                pred_centers, occupied, voxel_size, origin=EGO_BOUNDS[:, 0]
            )
        else:
            err_centers, err_occ = known_space_error_voxels(
                pred_centers, occupied, free, voxel_size, origin=EGO_BOUNDS[:, 0]
            )
        gt_centers, gt_occ = _cap_voxels(occupied, weights)
        free_c, free_o, unk_c, unk_o = display_label_clouds(
            occupied, free, voxel_size, origin=EGO_BOUNDS[:, 0]
        )
        return pack_occupancy_pair(
            pred_centers,
            pred_occ,
            gt_centers,
            gt_occ,
            err_centers,
            err_occ,
            free_c,
            free_o,
            unk_c,
            unk_o,
        )

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
                payload, miou, metric = cached
                self.last_frame_index = int(frame_index)
                self.last_depth_ms = 0.0
                self.last_project_ms = 0.0
                self.last_miou = miou
                self.last_metric = metric
            else:
                payload = None
        if payload is not None:
            self._refresh_benchmark(int(frame_index), float(threshold))
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

        self._refresh_benchmark(int(frame_index), float(threshold))
        with self._cache_lock:
            self._occupancy_cache[key] = (payload, float(self.last_miou), dict(self.last_metric or {}))
            self._occupancy_cache.move_to_end(key)
            while len(self._occupancy_cache) > OCCUPANCY_CACHE_LIMIT:
                self._occupancy_cache.popitem(last=False)
        self.last_frame_index = int(frame_index)
        return payload

    def _refresh_benchmark(self, frame_index: int, threshold: float) -> None:
        """Attach the 1.0 m monocular-vs-VoxNet table without changing HUD mIoU."""
        key = (int(frame_index), round(float(threshold), 3))
        with self._cache_lock:
            cached = self._benchmark_cache.get(key)
            if cached is not None:
                self._benchmark_cache.move_to_end(key)
                self.last_benchmark = cached
                return
        try:
            from benchmark import build_frame_benchmark

            table = build_frame_benchmark(int(frame_index), float(threshold))
        except Exception as exc:
            logger.warning("Occupancy benchmark unavailable for frame %s: %s", frame_index, exc)
            self.last_benchmark = None
            return
        with self._cache_lock:
            self._benchmark_cache[key] = table
            self._benchmark_cache.move_to_end(key)
            while len(self._benchmark_cache) > OCCUPANCY_CACHE_LIMIT:
                self._benchmark_cache.popitem(last=False)
            self.last_benchmark = table

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
        pred_c, pred_o, gt_c, gt_o, err_c, err_o, _miou, score, free_c, free_o, unk_c, unk_o = (
            project_cameras_to_voxel_pair(
                payloads,
                load_lidar_ego_points(frame_index),
                voxel_size,
                threshold,
            )
        )
        self.last_project_ms = (time.perf_counter() - start) * 1000.0
        self.last_depth_ms = depth_ms
        self._store_score(score, voxel_size, unlabeled_is_free=False)
        self.gt_source = "lidar-visible" if gt_c.shape[0] else "unavailable"
        return pack_occupancy_pair(
            pred_c, pred_o, gt_c, gt_o, err_c, err_o, free_c, free_o, unk_c, unk_o
        )

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

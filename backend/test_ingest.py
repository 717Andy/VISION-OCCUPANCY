"""nuScenes 6-camera ingest plus playback-driven occupancy."""

from __future__ import annotations

import struct
import unittest

import numpy as np

from nuscenes_loader import (
    CAMERA_IDS,
    ORIGINAL_IMAGE_SIZE,
    load_synchronized_frame,
)
from pipeline import PerceptionPipeline


class FakeMidasEngine:
    """Deterministic HxW disparity without downloading weights."""

    def __init__(self) -> None:
        self.last_infer_ms = 1.25
        self.available = True
        self.device = "cpu"
        self.using_cuda = False

    def ensure_loaded(self) -> bool:
        return True

    def infer_cached(self, frame_index: int, camera_id: str, image_rgb: np.ndarray) -> np.ndarray:
        del frame_index, camera_id
        self.assert_rgb(image_rgb)
        height, width = 32, 48
        yy, xx = np.mgrid[0:height, 0:width]
        disparity = 1.5 + 6.0 * np.exp(-((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2) / 90.0)
        return disparity.astype(np.float32)

    @staticmethod
    def assert_rgb(image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise AssertionError(f"expected RGB, got {image_rgb.shape}")


class MatrixIngestionTests(unittest.TestCase):
    def test_synchronized_six_camera_timestep_exposes_k(self):
        frame = load_synchronized_frame(0)
        self.assertEqual(len(frame.cameras), 6)
        self.assertEqual(set(frame.cameras), set(CAMERA_IDS))
        timestamps = [sample.timestamp for sample in frame.cameras.values()]
        self.assertTrue(max(timestamps) - min(timestamps) < 200_000)
        for sample in frame.cameras.values():
            self.assertEqual(sample.intrinsic.shape, (3, 3))
            self.assertGreater(sample.intrinsic[0, 0], 100.0)
            self.assertEqual(sample.translation.shape, (3,))
            self.assertEqual(sample.rotation.shape, (4,))
            rgb = np.asarray(__import__("PIL").Image.open(sample.image_path))
            self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(ORIGINAL_IMAGE_SIZE, (1600, 900))

    def test_second_timestep_is_a_new_sync_set(self):
        a = load_synchronized_frame(0)
        b = load_synchronized_frame(1)
        self.assertNotEqual(a.timestamp, b.timestamp)
        self.assertEqual(len(b.cameras), 6)


class PlaybackOccupancyTests(unittest.TestCase):
    def test_frame_index_changes_occupancy_payload(self):
        pipeline = PerceptionPipeline(depth_engine=FakeMidasEngine())
        first = pipeline.occupancy_for_frame(0, voxel_size=0.4, threshold=0.1)
        second = pipeline.occupancy_for_frame(1, voxel_size=0.4, threshold=0.1)
        self.assertEqual(pipeline.voxel_source, "midas-open3d")
        self.assertGreater(struct.unpack_from("<I", first, 0)[0], 0)
        self.assertGreater(struct.unpack_from("<I", second, 0)[0], 0)
        self.assertEqual(pipeline.last_frame_index, 1)

    def test_voxels_are_metric_ego_coordinates(self):
        pipeline = PerceptionPipeline(depth_engine=FakeMidasEngine())
        payload = pipeline.occupancy_for_frame(0, voxel_size=0.5, threshold=0.0)
        count = struct.unpack_from("<I", payload, 0)[0]
        self.assertGreater(count, 0)
        x, y, z, prob = struct.unpack_from("<ffff", payload, 4)
        self.assertTrue(math_isfinite(x, y, z, prob))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertTrue(-20.0 <= x <= 45.0)
        self.assertTrue(-20.0 <= y <= 20.0)
        self.assertTrue(-2.0 <= z <= 6.0)


def math_isfinite(*values: float) -> bool:
    return all(np.isfinite(value) for value in values)


class SceneApiTests(unittest.TestCase):
    def test_scene_endpoint_exposes_intrinsics_and_image_sizes(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.get("/api/scene")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["original_image_size"], [1600, 900])
        self.assertEqual(data["prepared_image_size"], [640, 360])
        frame = data["frames"][0]
        self.assertEqual(len(frame["cameras"]), 6)
        self.assertEqual(len(frame["calibration"]), 6)
        k = frame["calibration"]["CAM_FRONT"]["intrinsic"]
        self.assertEqual(len(k), 3)
        self.assertEqual(len(k[0]), 3)


if __name__ == "__main__":
    unittest.main()

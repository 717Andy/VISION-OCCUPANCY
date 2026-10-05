import math
import struct
import time
import unittest

import numpy as np

from nuscenes_loader import CAMERA_IDS
from pipeline import PerceptionPipeline


class OccupancyTensorTests(unittest.TestCase):
    def test_header_matches_active_count(self):
        pipeline = PerceptionPipeline()
        payload = pipeline.generate_occupancy_tensor(threshold=0.5)
        count = struct.unpack_from("<I", payload, 0)[0]
        body = payload[4:]
        self.assertEqual(len(body) % 16, 0)
        self.assertEqual(count, len(body) // 16)
        self.assertGreater(count, 0)

    def test_higher_threshold_fewer_voxels(self):
        pipeline = PerceptionPipeline()
        low = pipeline.generate_occupancy_tensor(threshold=0.2)
        pipeline.frame_count = 0
        high = pipeline.generate_occupancy_tensor(threshold=0.8)
        low_count = struct.unpack_from("<I", low, 0)[0]
        high_count = struct.unpack_from("<I", high, 0)[0]
        self.assertGreater(low_count, high_count)

    def test_probability_in_unit_interval(self):
        pipeline = PerceptionPipeline()
        payload = pipeline.generate_occupancy_tensor(threshold=0.1)
        count = struct.unpack_from("<I", payload, 0)[0]
        for i in range(min(count, 32)):
            _x, _y, _z, prob = struct.unpack_from("<ffff", payload, 4 + i * 16)
            self.assertGreaterEqual(prob, 0.1)
            self.assertLessEqual(prob, 1.0)
            self.assertTrue(math.isfinite(prob))


    def test_ground_truth_comes_from_lidar_not_the_prediction(self):
        pipeline = PerceptionPipeline()
        lidar = np.array(
            [[12.0, 0.0, 0.4], [12.15, 0.05, 0.45], [18.0, -3.0, 1.2]],
            dtype=np.float32,
        )
        payload = pipeline.generate_occupancy_pair(
            threshold=0.5, frame_index=3, voxel_size=0.5, lidar_points=lidar
        )
        pred_n, gt_n, err_n, free_n, unk_n = struct.unpack_from("<IIIII", payload, 0)
        self.assertGreater(pred_n, 0)
        self.assertGreater(gt_n, 0)
        self.assertGreater(err_n, 0)
        self.assertEqual(len(payload), 20 + (pred_n + gt_n + err_n + free_n + unk_n) * 16)
        self.assertEqual(pipeline.gt_source, "lidar-visible")
        self.assertLess(pipeline.last_miou, 1.0)
        gt = np.frombuffer(
            payload[20 + pred_n * 16 : 20 + (pred_n + gt_n) * 16], dtype=np.float32
        ).reshape(gt_n, 4)
        self.assertTrue(np.any(np.abs(gt[:, 0] - 12.0) < 1.0))

    def test_missing_lidar_does_not_copy_prediction(self):
        pipeline = PerceptionPipeline()
        payload = pipeline.generate_occupancy_pair(
            threshold=0.5,
            frame_index=99_999,
            voxel_size=1.0,
            lidar_points=np.zeros((0, 3), dtype=np.float32),
        )
        pred_n, gt_n, err_n, _free_n, _unk_n = struct.unpack_from("<IIIII", payload, 0)
        self.assertGreater(pred_n, 0)
        self.assertEqual(gt_n, 0)
        self.assertGreater(err_n, 0)
        self.assertLessEqual(err_n, pred_n)
        self.assertEqual(pipeline.gt_source, "unavailable")

    def test_vectorized_generation_is_realtime(self):
        pipeline = PerceptionPipeline()
        pipeline.generate_occupancy_tensor(threshold=0.38)  # warmup
        start = time.perf_counter()
        for _ in range(10):
            pipeline.generate_occupancy_tensor(threshold=0.38)
        elapsed = (time.perf_counter() - start) / 10
        self.assertLess(elapsed, 0.02, f"occupancy generation took {elapsed:.4f}s")


class ManifestPathTests(unittest.TestCase):
    def test_known_camera_ids(self):
        self.assertEqual(len(CAMERA_IDS), 6)
        self.assertIn("CAM_FRONT", CAMERA_IDS)


if __name__ == "__main__":
    unittest.main()

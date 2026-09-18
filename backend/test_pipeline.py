import math
import struct
import time
import unittest

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


    def test_occupancy_pair_has_denser_ground_truth(self):
        pipeline = PerceptionPipeline()
        payload = pipeline.generate_occupancy_pair(threshold=0.5)
        pred_n, gt_n = struct.unpack_from("<II", payload, 0)
        self.assertGreater(pred_n, 0)
        self.assertGreaterEqual(gt_n, pred_n)
        self.assertEqual(len(payload), 8 + (pred_n + gt_n) * 28)
        self.assertGreater(pipeline.last_miou, 0.0)
        self.assertLessEqual(pipeline.last_miou, 1.0)

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

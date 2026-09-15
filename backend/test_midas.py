"""MiDaS engine tests. CUDA <30 ms is enforced only when a GPU is present."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _engine():
    from midas_engine import MidasDepthEngine

    return MidasDepthEngine(autoload=True)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class MidasEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _engine()
        if not cls.engine.available:
            raise unittest.SkipTest(cls.engine.error or "MiDaS weights unavailable")

    def test_output_is_2d_numpy_float32(self):
        rgb = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        depth = self.engine.infer(rgb)
        self.assertIsInstance(depth, np.ndarray)
        self.assertEqual(depth.ndim, 2)
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreater(depth.shape[0], 1)
        self.assertGreater(depth.shape[1], 1)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required for 30ms SLA")
    def test_gpu_depth_pass_under_30ms(self):
        rgb = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        self.engine.infer(rgb)  # warmup
        samples = []
        for _ in range(5):
            self.engine.infer(rgb)
            samples.append(self.engine.last_infer_ms)
        median = float(np.median(samples))
        self.assertLess(median, 30.0, f"GPU depth pass took {median:.2f} ms")


if __name__ == "__main__":
    unittest.main()

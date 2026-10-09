"""A kept ray occupies the far cell and leaves the free cells in front empty."""

from __future__ import annotations

import unittest

import numpy as np

from projection import EGO_BOUNDS, quat_to_rotmat
from ray_surface_baseline import surface_cells


def _forward_quat() -> np.ndarray:
    """Camera optical axes: x right, y down, z along ego +x."""
    rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    trace = float(np.trace(rotation))
    scale = np.sqrt(trace + 1.0) * 2.0
    quat = np.array(
        [
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ],
        dtype=np.float64,
    )
    recovered = quat_to_rotmat(quat)
    if not np.allclose(recovered, rotation, atol=1e-6):
        raise AssertionError("forward quaternion does not match the camera axes")
    return quat


class RaySurfaceGeometryTests(unittest.TestCase):
    def test_one_ray_occupies_only_the_far_cell(self):
        intrinsic = np.array([[10.0, 0.0, 5.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]])
        depth = np.zeros((11, 11), dtype=np.float32)
        keep = np.zeros((11, 11), dtype=bool)
        depth[5, 5] = 10.0
        keep[5, 5] = True
        centers = surface_cells(
            depth,
            keep,
            intrinsic,
            _forward_quat(),
            np.array([0.0, 0.0, 1.5]),
        )
        far = np.array([10.5, 0.5, 1.5], dtype=np.float32)
        self.assertEqual(centers.shape, (1, 3))
        self.assertTrue(np.allclose(centers[0], far, atol=1e-4))
        for meters in (1.5, 3.5, 5.5, 7.5, 9.5):
            nearer = np.array([meters, 0.5, 1.5], dtype=np.float32)
            self.assertFalse(np.any(np.all(np.abs(centers - nearer) < 1e-3, axis=1)))

    def test_predicted_cells_stay_inside_ego_bounds(self):
        intrinsic = np.array([[10.0, 0.0, 5.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]])
        depth = np.full((11, 11), 12.0, dtype=np.float32)
        keep = np.zeros((11, 11), dtype=bool)
        keep[5, 5] = True
        centers = surface_cells(
            depth,
            keep,
            intrinsic,
            _forward_quat(),
            np.array([0.0, 0.0, 1.5]),
        )
        self.assertGreater(centers.shape[0], 0)
        self.assertTrue(np.all(centers >= EGO_BOUNDS[:, 0] - 1e-3))
        self.assertTrue(np.all(centers < EGO_BOUNDS[:, 1] + 1e-3))


if __name__ == "__main__":
    unittest.main()

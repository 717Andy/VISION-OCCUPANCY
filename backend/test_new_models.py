"""Geometry checks for the ground-plane and stereo predictors."""

from __future__ import annotations

import unittest

import numpy as np

from ipm_baseline import ipm_points
from projection import quat_to_rotmat
from stereo_baseline import projection_matrix, triangulate_matches


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
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    recovered = quat_to_rotmat(quat)
    if not np.allclose(recovered, rotation, atol=1e-6):
        raise AssertionError("forward quaternion does not match the camera axes")
    return quat


class GroundPlaneTests(unittest.TestCase):
    def test_vertical_edge_lands_on_the_ground_ahead(self):
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        image = np.zeros((100, 100), dtype=np.float32)
        image[65, 50] = 255.0
        points = ipm_points(
            image,
            intrinsic,
            _forward_quat(),
            np.array([0.0, 0.0, 1.5]),
            stride=1,
            gradient_quantile=0.5,
            source_size=(100, 100),
        )
        self.assertGreater(points.shape[0], 0)
        self.assertTrue(np.all(np.abs(points[:, 2]) < 1e-4))
        self.assertTrue(np.any(np.abs(points[:, 0] - 10.0) < 0.5))


class StereoTriangulationTests(unittest.TestCase):
    def test_two_views_recover_a_known_point(self):
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        quat = _forward_quat()
        left = projection_matrix(intrinsic, quat, np.array([0.0, 0.0, 1.5]), (100, 100), (100, 100))
        right = projection_matrix(intrinsic, quat, np.array([0.0, 1.0, 1.5]), (100, 100), (100, 100))
        pixels_a = np.array([[50.0, 65.0]])
        pixels_b = np.array([[60.0, 65.0]])
        points = triangulate_matches(left, right, pixels_a, pixels_b)
        self.assertEqual(points.shape, (1, 3))
        self.assertTrue(np.allclose(points[0], [10.0, 0.0, 0.0], atol=0.05))


if __name__ == "__main__":
    unittest.main()

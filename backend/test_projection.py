"""Unit tests for pinhole unprojection, scale-and-shift depth, and voxelization."""

from __future__ import annotations

import unittest

import numpy as np

from projection import (
    FAR_M,
    NEAR_M,
    camera_to_ego,
    disparity_to_metric_depth,
    grid_miou,
    gt_occupancy_threshold,
    pack_occupancy,
    pack_occupancy_pair,
    quat_to_rotmat,
    scale_intrinsics,
    unproject_depth,
    voxelize_occupancy,
)


class IntrinsicScalingTests(unittest.TestCase):
    def test_scales_fx_fy_cx_cy_to_depth_resolution(self):
        k = np.array(
            [
                [1252.8131021185304, 0.0, 826.588114781398],
                [0.0, 1252.8131021185304, 469.9846626224581],
                [0.0, 0.0, 1.0],
            ]
        )
        scaled = scale_intrinsics(k, (1600, 900), (384, 256))
        self.assertAlmostEqual(scaled[0, 0], 1252.8131021185304 * 384 / 1600)
        self.assertAlmostEqual(scaled[1, 1], 1252.8131021185304 * 256 / 900)
        self.assertAlmostEqual(scaled[0, 2], 826.588114781398 * 384 / 1600)
        self.assertAlmostEqual(scaled[1, 2], 469.9846626224581 * 256 / 900)
        self.assertEqual(scaled[2, 2], 1.0)


class MetricDepthTests(unittest.TestCase):
    def test_scale_and_shift_maps_to_physical_bounds(self):
        disparity = np.array([[0.0, 1.0, 4.0]], dtype=np.float32)
        depth = disparity_to_metric_depth(disparity, near=NEAR_M, far=FAR_M)
        self.assertEqual(depth.shape, disparity.shape)
        self.assertAlmostEqual(float(depth[0, 0]), FAR_M, places=5)
        self.assertAlmostEqual(float(depth[0, 2]), NEAR_M, places=5)
        self.assertTrue(np.all(depth >= NEAR_M))
        self.assertTrue(np.all(depth <= FAR_M))

    def test_constant_disparity_stays_finite_midrange(self):
        depth = disparity_to_metric_depth(np.ones((8, 8), dtype=np.float32) * 3.2)
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue(np.all(depth >= NEAR_M))
        self.assertTrue(np.all(depth <= FAR_M))


class UnprojectionTests(unittest.TestCase):
    def test_pinhole_formula_at_principal_point(self):
        k = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        depth = np.array([[10.0]], dtype=np.float32)
        points = unproject_depth(depth, k, stride=1)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 10.0], atol=1e-6)

    def test_identity_extrinsics_preserve_camera_points(self):
        points = np.array([[1.0, 2.0, 3.0]])
        ego = camera_to_ego(points, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(ego, points)
        self.assertTrue(np.allclose(quat_to_rotmat([1, 0, 0, 0]), np.eye(3)))

    def test_nuscenes_front_optical_axis_is_ego_forward(self):
        # CAM_FRONT from scene-0103: camera z (optical) → ego x (forward)
        rotation = np.array(
            [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754]
        )
        optical = quat_to_rotmat(rotation) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(optical, [1.0, 0.0, 0.0], atol=0.03)


class VoxelizationTests(unittest.TestCase):
    def test_point_cloud_becomes_discrete_occupancy_grid(self):
        clustered = np.array(
            [
                [0.05, 1.05, 0.05],
                [0.08, 1.02, 0.04],
                [0.06, 1.07, 0.08],
            ],
            dtype=np.float64,
        )
        far = np.array([[1.55, 3.25, 0.55]], dtype=np.float64)
        centers, occ = voxelize_occupancy(
            np.vstack([clustered, far]),
            voxel_size=0.2,
            occupancy_threshold=0.0,
        )
        self.assertGreaterEqual(centers.shape[0], 2)
        self.assertEqual(centers.shape[1], 3)
        self.assertEqual(occ.shape[0], centers.shape[0])
        unique = {tuple(np.round(row, 5)) for row in centers}
        self.assertEqual(len(unique), centers.shape[0])
        self.assertTrue(np.all((occ > 0) & (occ <= 1)))

        one_cell, one_occ = voxelize_occupancy(clustered, voxel_size=0.2, occupancy_threshold=0.0)
        self.assertEqual(one_cell.shape[0], 1)
        self.assertEqual(one_occ.shape[0], 1)

    def test_binary_protocol_roundtrip(self):
        centers = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        occ = np.array([0.9], dtype=np.float32)
        payload = pack_occupancy(centers, occ)
        count = np.frombuffer(payload[:4], dtype=np.uint32)[0]
        body = np.frombuffer(payload[4:], dtype=np.float32).reshape(count, 4)
        self.assertEqual(int(count), 1)
        np.testing.assert_allclose(body[0], [1.0, 2.0, 0.5, 0.9])

    def test_pair_protocol_roundtrip(self):
        pred_c = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        pred_o = np.array([0.9], dtype=np.float32)
        gt_c = np.array([[1.0, 2.0, 0.5], [3.0, 0.0, 1.0]], dtype=np.float32)
        gt_o = np.array([0.9, 0.4], dtype=np.float32)
        payload = pack_occupancy_pair(pred_c, pred_o, gt_c, gt_o)
        pred_n, gt_n = np.frombuffer(payload[:8], dtype=np.uint32)
        self.assertEqual(int(pred_n), 1)
        self.assertEqual(int(gt_n), 2)
        body = np.frombuffer(payload[8:], dtype=np.float32)
        self.assertEqual(body.size, (1 + 2) * 4)
        np.testing.assert_allclose(body[:4], [1.0, 2.0, 0.5, 0.9])
        np.testing.assert_allclose(body[4:8], [1.0, 2.0, 0.5, 0.9])
        np.testing.assert_allclose(body[8:], [3.0, 0.0, 1.0, 0.4])

    def test_lower_threshold_keeps_more_ground_truth_voxels(self):
        rng = np.random.default_rng(0)
        clustered = rng.normal(loc=[2.0, 1.0, 0.5], scale=0.04, size=(40, 3))
        sparse = rng.normal(loc=[8.0, -4.0, 1.5], scale=0.12, size=(6, 3))
        points = np.vstack([clustered, sparse])
        pred_c, _ = voxelize_occupancy(points, voxel_size=0.2, occupancy_threshold=0.38)
        gt_c, _ = voxelize_occupancy(
            points,
            voxel_size=0.2,
            occupancy_threshold=gt_occupancy_threshold(0.38),
        )
        self.assertGreaterEqual(gt_c.shape[0], pred_c.shape[0])

    def test_grid_miou_identical_and_disjoint(self):
        cells = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(grid_miou(cells, cells, voxel_size=1.0), 1.0)
        other = np.array([[10.1, 10.1, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(grid_miou(cells, other, voxel_size=1.0), 0.0)
        empty = np.zeros((0, 3), dtype=np.float32)
        self.assertAlmostEqual(grid_miou(empty, empty, voxel_size=1.0), 1.0)


if __name__ == "__main__":
    unittest.main()

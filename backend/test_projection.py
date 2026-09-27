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
    metric_depth_from_ground,
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


class GroundAlignmentTests(unittest.TestCase):
    """Road rays use camera height, not the fixed [0.5 m, 50 m] stretch."""

    FRONT_ROT = np.array(
        [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754]
    )
    FRONT_T = np.array([1.72200568478, 0.00475453292289, 1.49491291905])

    def _road_disparity(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 96, 128
        intrinsic = np.array(
            [[140.0, 0.0, width / 2.0], [0.0, 140.0, height / 2.0], [0.0, 0.0, 1.0]]
        )
        us = np.arange(width, dtype=np.float64) + 0.5
        vs = np.arange(height, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack((uu.ravel(), vv.ravel(), np.ones(uu.size)), axis=0)
        ray_z = (quat_to_rotmat(self.FRONT_ROT) @ (np.linalg.inv(intrinsic) @ pixels))[2]
        ground_depth = (0.0 - self.FRONT_T[2]) / np.where(np.abs(ray_z) < 1e-4, np.nan, ray_z)
        disparity = np.full(uu.size, 1.0, dtype=np.float64)
        road = (ray_z < -0.05) & (ground_depth > 4.0) & (ground_depth < 30.0)
        disparity[road] = 20.0 / ground_depth[road] + 1.0
        disparity = disparity.reshape(height, width)
        # A closer patch must stay above the plane after the same scale.
        disparity[70:85, 50:70] *= 2.2
        return disparity.astype(np.float32), intrinsic

    def test_road_cloud_sits_on_the_ego_ground_plane(self):
        disparity, intrinsic = self._road_disparity()
        depth = metric_depth_from_ground(disparity, intrinsic, self.FRONT_ROT, self.FRONT_T)
        ego = camera_to_ego(
            unproject_depth(depth, intrinsic, stride=2),
            self.FRONT_ROT,
            self.FRONT_T,
        )
        span = ego[(ego[:, 0] > 5.0) & (ego[:, 0] < 25.0) & (np.abs(ego[:, 1]) < 6.0)]
        self.assertGreater(span.shape[0], 30)
        road = span[span[:, 2] < 0.4]
        self.assertGreater(road.shape[0], 20)
        self.assertLess(abs(float(np.median(road[:, 2]))), 0.2)
        # The boosted patch is closer than the road along the same ray, so it lifts off z = 0.
        self.assertLess(float(depth[77, 60]), 4.0)
        ray = np.linalg.inv(intrinsic) @ np.array([60.5, 77.5, 1.0])
        patch = camera_to_ego((float(depth[77, 60]) * ray).reshape(1, 3), self.FRONT_ROT, self.FRONT_T)
        self.assertGreater(float(patch[0, 2]), 0.3)

        stretched = disparity_to_metric_depth(disparity)
        old = camera_to_ego(
            unproject_depth(stretched, intrinsic, stride=2),
            self.FRONT_ROT,
            self.FRONT_T,
        )
        old_span = old[(old[:, 0] > 5.0) & (old[:, 0] < 25.0) & (np.abs(old[:, 1]) < 6.0)]
        self.assertGreater(old_span.shape[0], 10)
        self.assertGreater(abs(float(np.median(old_span[:, 2]))), 0.5)

    def test_low_camera_falls_back_to_bounded_stretch(self):
        disparity = np.linspace(0.0, 4.0, 64, dtype=np.float32).reshape(8, 8)
        intrinsic = np.eye(3)
        depth = metric_depth_from_ground(
            disparity,
            intrinsic,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.1]),
        )
        np.testing.assert_allclose(depth, disparity_to_metric_depth(disparity))


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

    def test_lidar_grid_is_not_a_camera_cloud(self):
        from projection import voxelize_lidar

        camera_like = np.array([[2.0, 1.0, 0.4], [2.05, 1.02, 0.42], [2.02, 0.98, 0.4]], dtype=np.float64)
        lidar = np.array([[20.0, -8.0, 1.5], [20.1, -8.0, 1.55], [8.0, 4.0, 0.2]], dtype=np.float64)
        pred_c, _ = voxelize_occupancy(camera_like, voxel_size=0.5, occupancy_threshold=0.0)
        gt_c, gt_o = voxelize_lidar(lidar, voxel_size=0.5)
        self.assertGreater(gt_c.shape[0], 0)
        self.assertEqual(gt_o.shape[0], gt_c.shape[0])
        pred_keys = {tuple(np.round(row, 2)) for row in pred_c}
        gt_keys = {tuple(np.round(row, 2)) for row in gt_c}
        self.assertTrue(gt_keys.isdisjoint(pred_keys))

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

    def test_one_cell_tolerance_counts_a_neighbor(self):
        left = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        neighbor = np.array([[0.3, 0.1, 0.1]], dtype=np.float32)
        far = np.array([[2.1, 0.1, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(grid_miou(left, neighbor, voxel_size=0.2), 0.0)
        self.assertAlmostEqual(grid_miou(left, neighbor, voxel_size=0.2, tolerance=1), 1.0)
        self.assertAlmostEqual(grid_miou(left, far, voxel_size=0.2, tolerance=1), 0.0)

    def test_elevated_cells_survive_a_road_dominated_threshold(self):
        road = np.tile(np.array([[2.0, 0.0, 0.1]], dtype=np.float64), (40, 1))
        road += np.array([[0.02, 0.0, 0.0]])
        car = np.array([[6.1, 1.0, 1.6], [6.1, 1.0, 1.6]], dtype=np.float64)
        centers, _ = voxelize_occupancy(
            np.vstack([road, car]),
            voxel_size=0.2,
            occupancy_threshold=0.9,
        )
        self.assertTrue(np.any(centers[:, 2] > 1.0))


if __name__ == "__main__":
    unittest.main()

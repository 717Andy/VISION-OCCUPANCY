"""Unit tests for pinhole unprojection, scale-and-shift depth, and voxelization."""

from __future__ import annotations

import unittest

import numpy as np

from projection import (
    EGO_BOUNDS,
    FAR_M,
    NEAR_M,
    camera_to_ego,
    disparity_to_metric_depth,
    grid_miou,
    gt_occupancy_threshold,
    occupancy_counts,
    metric_depth_from_ground,
    camera_visible_points,
    complete_camera_surface,
    discrepancy_voxels,
    pack_occupancy,
    pack_occupancy_pair,
    quat_to_rotmat,
    scale_intrinsics,
    unproject_depth,
    voxelize_occupancy,
)


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


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
        err_c = np.array([[3.0, 0.0, 1.0]], dtype=np.float32)
        err_o = np.array([1.0], dtype=np.float32)
        payload = pack_occupancy_pair(pred_c, pred_o, gt_c, gt_o, err_c, err_o)
        pred_n, gt_n, err_n = np.frombuffer(payload[:12], dtype=np.uint32)
        self.assertEqual(int(pred_n), 1)
        self.assertEqual(int(gt_n), 2)
        self.assertEqual(int(err_n), 1)
        body = np.frombuffer(payload[12:], dtype=np.float32)
        self.assertEqual(body.size, (1 + 2 + 1) * 4)
        np.testing.assert_allclose(body[:4], [1.0, 2.0, 0.5, 0.9])
        np.testing.assert_allclose(body[4:8], [1.0, 2.0, 0.5, 0.9])
        np.testing.assert_allclose(body[8:12], [3.0, 0.0, 1.0, 0.4])
        np.testing.assert_allclose(body[12:], [3.0, 0.0, 1.0, 1.0])

    def test_discrepancy_is_the_per_cell_symmetric_difference(self):
        origin = EGO_BOUNDS[:, 0]
        size = 0.2

        def center(ijk: tuple[int, int, int]) -> np.ndarray:
            return origin + (np.array(ijk, dtype=np.float64) + 0.5) * size

        shared = center((40, 20, 8))
        pred_only = center((80, 20, 8))
        gt_only = center((10, 40, 12))
        pred = np.vstack([shared, pred_only]).astype(np.float32)
        gt = np.vstack([shared, gt_only]).astype(np.float32)
        centers, occ = discrepancy_voxels(pred, gt, size, origin=origin, cap=False)
        got = {tuple(np.round(row, 4)) for row in centers.astype(np.float64)}
        expected = {tuple(np.round(pred_only, 4)), tuple(np.round(gt_only, 4))}
        self.assertEqual(got, expected)
        self.assertTrue(np.all(occ == 1.0))
        empty, _ = discrepancy_voxels(pred, pred, size, origin=origin, cap=False)
        self.assertEqual(empty.shape[0], 0)

    def test_camera_visible_lidar_keeps_the_front_surface_only(self):
        # Camera looks along ego +x: optical z -> ego x, optical x -> ego -y, optical y -> ego -z.
        rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        quat = _quat_wxyz(rotation)
        np.testing.assert_allclose(quat_to_rotmat(quat), rotation, atol=1e-6)
        camera = {
            "intrinsic": np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            "rotation": quat,
            "translation": np.zeros(3),
        }
        front = np.array([10.0, 0.0, 0.0])
        on_surface = np.array([10.2, 0.0, 0.0])
        hidden = np.array([20.0, 0.0, 0.0])
        beside = np.array([10.0, 2.0, 0.0])
        behind = np.array([-5.0, 0.0, 0.0])
        outside = np.array([10.0, 0.0, 8.0])
        points = np.vstack([front, on_surface, hidden, beside, behind, outside])
        kept = camera_visible_points(
            points,
            [camera],
            image_size=(100, 100),
            pixel_bin=4,
            surface_m=0.4,
        )

        def contains(point: np.ndarray) -> bool:
            if kept.shape[0] == 0:
                return False
            return bool(np.any(np.all(np.abs(kept - point) < 1e-3, axis=1)))

        self.assertTrue(contains(front))
        self.assertTrue(contains(on_surface))
        self.assertTrue(contains(beside))
        self.assertFalse(contains(hidden))
        self.assertFalse(contains(behind))
        self.assertFalse(contains(outside))

    def test_real_sweep_drops_lidar_the_cameras_cannot_see(self):
        from lidar_gt import load_lidar_ego_points
        from nuscenes_loader import CAMERA_IDS, load_synchronized_frame

        full = load_lidar_ego_points(0)
        frame = load_synchronized_frame(0)
        cameras = [
            {
                "intrinsic": frame.cameras[camera_id].intrinsic,
                "rotation": frame.cameras[camera_id].rotation,
                "translation": frame.cameras[camera_id].translation,
            }
            for camera_id in CAMERA_IDS
        ]
        visible = camera_visible_points(full, cameras)
        self.assertGreater(full.shape[0], 1000)
        self.assertLess(visible.shape[0], full.shape[0])
        self.assertGreater(visible.shape[0], int(full.shape[0] * 0.5))
        # The near-ego ring is mostly below the images. The feeds keep the farther road.
        self.assertGreater(float(np.median(np.linalg.norm(visible[:, :2], axis=1))), 5.0)

    def test_agreed_camera_gap_is_filled_on_the_near_surface(self):
        rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera = {
            "intrinsic": np.array([[80.0, 0.0, 48.0], [0.0, 80.0, 48.0], [0.0, 0.0, 1.0]]),
            "rotation": _quat_wxyz(rotation),
            "translation": np.zeros(3),
        }
        # Two hits on the same surface with one depth bin between them, plus a
        # farther return on the first ray that the near hit must hide.
        left = np.array([10.0, 1.5, 0.0])
        right = np.array([10.0, -0.5, 0.0])
        hidden = np.array([20.0, 3.0, 0.0])
        surface = complete_camera_surface(
            np.vstack([left, right, hidden]),
            [camera],
            image_size=(96, 96),
            pixel_bin=8,
            fill_radius=1,
            agree_m=0.5,
            min_support=1,
        )
        self.assertGreater(surface.shape[0], 2)
        between = (
            (np.abs(surface[:, 0] - 10.0) < 0.4)
            & (np.abs(surface[:, 1] - 0.5) < 0.4)
            & (np.abs(surface[:, 2] + 0.5) < 0.4)
        )
        self.assertTrue(np.any(between))
        self.assertFalse(np.any(surface[:, 0] > 15.0))

    def test_frame_surface_adds_points_without_leaving_the_measurements(self):
        from lidar_gt import camera_space_lidar, load_lidar_ego_points
        from nuscenes_loader import CAMERA_IDS, load_synchronized_frame
        from projection import voxelize_lidar

        full = load_lidar_ego_points(0)
        frame = load_synchronized_frame(0)
        cameras = [
            {
                "intrinsic": frame.cameras[camera_id].intrinsic,
                "rotation": frame.cameras[camera_id].rotation,
                "translation": frame.cameras[camera_id].translation,
            }
            for camera_id in CAMERA_IDS
        ]
        visible = camera_visible_points(full, cameras)
        space = camera_space_lidar(0)
        visible_cells, _ = voxelize_lidar(visible, 0.2, cap=False)
        space_cells, _ = voxelize_lidar(space, 0.2, cap=False)
        self.assertGreater(space_cells.shape[0], visible_cells.shape[0])
        sample = space[np.random.default_rng(0).choice(space.shape[0], size=200, replace=False)]
        distances = [float(np.linalg.norm(visible - point, axis=1).min()) for point in sample]
        self.assertLess(float(np.median(distances)), 0.5)
        self.assertLess(float(np.percentile(distances, 95)), 1.0)

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

    def test_occupancy_counts_report_tp_fp_fn_and_band_miou(self):
        pred = np.array(
            [
                [0.5, 0.5, -0.5],
                [1.5, 0.5, 1.5],
            ],
            dtype=np.float32,
        )
        gt = np.array(
            [
                [0.5, 0.5, -0.5],
                [2.5, 0.5, 3.5],
            ],
            dtype=np.float32,
        )
        counts = occupancy_counts(pred, gt, voxel_size=1.0, origin=np.zeros(3))
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 1)
        self.assertEqual(counts["fn"], 1)
        self.assertAlmostEqual(counts["iou"], 1.0 / 3.0)
        by_class = {row["class"]: row for row in counts["classes"]}
        self.assertAlmostEqual(by_class["driveable"]["iou"], 1.0)
        self.assertEqual(by_class["vehicle"]["fp"], 1)
        self.assertEqual(by_class["pedestrian"]["fn"], 1)
        self.assertAlmostEqual(counts["miou"], 1.0 / 3.0)
        empty = np.zeros((0, 3), dtype=np.float32)
        both_empty = occupancy_counts(empty, empty, voxel_size=1.0)
        self.assertEqual(both_empty["tp"], 0)
        self.assertAlmostEqual(both_empty["iou"], 1.0)
        self.assertAlmostEqual(both_empty["miou"], 1.0)

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

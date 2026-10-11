"""Ego-pose alignment for a short lidar window."""

from __future__ import annotations

import unittest

import numpy as np

from lidar_gt import (
    SWEEP_WINDOW_US,
    decimate_ego_points,
    rows_in_window,
    sweep_points_in_keyframe_ego,
)


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class SweepWindowTests(unittest.TestCase):
    def test_window_keeps_one_sweep_on_each_side(self):
        rows = [
            {"timestamp": 0},
            {"timestamp": 50_000},
            {"timestamp": 200_000},
        ]
        kept = rows_in_window(rows, 0, SWEEP_WINDOW_US)
        self.assertEqual([row["timestamp"] for row in kept], [0, 50_000])

    def test_keyframe_pose_returns_the_ego_point(self):
        points = np.array([[8.0, -1.0, 0.4]])
        out = sweep_points_in_keyframe_ego(
            points,
            IDENTITY,
            np.zeros(3),
            IDENTITY,
            np.zeros(3),
            IDENTITY,
            np.zeros(3),
        )
        self.assertTrue(np.allclose(out, points))

    def test_ego_motion_puts_a_static_point_back(self):
        # The car has moved +2 m in global x. A world point at x=10 is at ego x=8.
        points = np.array([[8.0, 0.0, 1.0]])
        out = sweep_points_in_keyframe_ego(
            points,
            IDENTITY,
            np.zeros(3),
            IDENTITY,
            np.array([2.0, 0.0, 0.0]),
            IDENTITY,
            np.zeros(3),
        )
        self.assertTrue(np.allclose(out, [[10.0, 0.0, 1.0]]))

    def test_decimate_keeps_a_new_cell_and_drops_a_duplicate(self):
        points = np.array(
            [
                [10.0, 0.0, 1.0],
                [10.02, 0.0, 1.0],
                [12.0, 0.0, 1.0],
            ]
        )
        kept = decimate_ego_points(points)
        self.assertEqual(kept.shape[0], 2)
        self.assertTrue(np.any(np.abs(kept[:, 0] - 12.0) < 0.05))


if __name__ == "__main__":
    unittest.main()

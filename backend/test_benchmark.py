"""Exact-cell mIoU comparison between monocular depth and the VoxNet baseline."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from benchmark import build_frame_benchmark, micro_summary, score_prediction
from projection import EGO_BOUNDS
from voxnet_baseline import CHECKPOINT_PATH, VoxNetOccupancy, predict_centers


class BenchmarkScoreTests(unittest.TestCase):
    def test_score_prediction_uses_exact_cell_counts(self):
        pred = np.array([[0.5, 0.5, -0.5], [1.5, 0.5, 1.5]], dtype=np.float32)
        gt = np.array([[0.5, 0.5, -0.5], [2.5, 0.5, 3.5]], dtype=np.float32)
        row = score_prediction("Monocular depth", pred, gt)
        self.assertEqual(row["pipeline"], "Monocular depth")
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 1, 1))
        self.assertAlmostEqual(row["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(row["miou"], 1.0 / 3.0)
        self.assertAlmostEqual(row["precision"], 0.5)
        self.assertAlmostEqual(row["recall"], 0.5)

    def test_micro_summary_pools_frames_before_dividing(self):
        frames = {
            0: (
                np.array([[0.5, 0.5, -0.5]], dtype=np.float32),
                np.array([[0.5, 0.5, -0.5]], dtype=np.float32),
            ),
            1: (
                np.zeros((0, 3), dtype=np.float32),
                np.array([[2.5, 0.5, 3.5]], dtype=np.float32),
            ),
        }

        def predict(frame_index: int) -> np.ndarray:
            return frames[frame_index][0]

        original = micro_summary.__globals__["known_grids_for_frame"]

        def fake_grids(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
            return frames[frame_index][1], np.zeros((0, 3), dtype=np.float32)

        micro_summary.__globals__["known_grids_for_frame"] = fake_grids
        try:
            summary = micro_summary(predict, [0, 1])
        finally:
            micro_summary.__globals__["known_grids_for_frame"] = original
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (1, 0, 1))
        self.assertAlmostEqual(summary["iou"], 0.5)


class VoxNetHeadTests(unittest.TestCase):
    def test_head_preserves_the_volume_grid(self):
        model = VoxNetOccupancy()
        volume = torch.zeros(1, 8, 6, 4, 5)
        logits = model.forward_logits(volume)
        self.assertEqual(tuple(logits.shape), (1, 1, 6, 4, 5))

    def test_checkpoint_prediction_stays_inside_ego_bounds(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("voxnet checkpoint has not been trained")
        centers = predict_centers(0)
        self.assertEqual(centers.ndim, 2)
        self.assertEqual(centers.shape[1], 3)
        self.assertTrue(np.isfinite(centers).all())
        if centers.shape[0] == 0:
            return
        self.assertTrue(np.all(centers >= EGO_BOUNDS[:, 0] - 1e-3))
        self.assertTrue(np.all(centers < EGO_BOUNDS[:, 1] + 1e-3))

    def test_frame_benchmark_reports_both_pipelines(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("voxnet checkpoint has not been trained")
        table = build_frame_benchmark(0, 0.38)
        self.assertEqual(table["voxel_m"], 1.0)
        self.assertEqual(table["tolerance_cells"], 0)
        self.assertEqual(table["unknown"], "ignored")
        self.assertTrue(table["checkpoint_id"])
        self.assertEqual(table["split"], "held-out")
        self.assertEqual(
            [row["pipeline"] for row in table["rows"]],
            [
                "Monocular depth",
                "VoxNet 3D CNN",
                "Ground-plane IPM",
                "Multi-view stereo",
                "Lift-Splat",
                "Camera-space",
            ],
        )
        scored = [row for row in table["rows"] if "error" not in row]
        self.assertTrue(any(row["pipeline"] == "VoxNet 3D CNN" for row in scored))
        for row in scored:
            denom = row["tp"] + row["fp"] + row["fn"]
            self.assertGreater(denom, 0)
            self.assertAlmostEqual(row["iou"], row["tp"] / denom)
            self.assertGreaterEqual(row["miou"], 0.0)
            self.assertLessEqual(row["miou"], 1.0)
            self.assertTrue(all("band" in item for item in row["bands"]))
        self.assertEqual(table["heldout"]["frames"], 8)
        self.assertIn("tp", table["heldout"]["voxnet"])
        if "monocular" in table["heldout"]:
            self.assertIn("tp", table["heldout"]["monocular"])


class RunStoreTests(unittest.TestCase):
    def test_registry_lists_the_built_in_models(self):
        from benchmark import registered_predictors

        names = [name for name, _predict, _identity in registered_predictors()]
        self.assertEqual(
            names,
            [
                "Monocular depth",
                "VoxNet 3D CNN",
                "Ground-plane IPM",
                "Multi-view stereo",
                "Lift-Splat",
                "Camera-space",
            ],
        )

    def test_saved_run_can_be_listed_and_loaded(self):
        from benchmark import RUNS_DIR, list_runs, load_run, save_run

        saved = save_run(
            {
                "checkpoint_id": "testcid",
                "scene_name": "scene-0103",
                "frame": {
                    "frame_index": 0,
                    "rows": [
                        {"pipeline": "VoxNet 3D CNN", "iou": 0.5, "tp": 1, "fp": 1, "fn": 0}
                    ],
                },
            }
        )
        path = RUNS_DIR / f"{saved['id']}.json"
        try:
            self.assertTrue(path.is_file())
            self.assertIn(saved["id"], [row["id"] for row in list_runs()])
            loaded = load_run(saved["id"])
            self.assertEqual(loaded["checkpoint_id"], "testcid")
            self.assertEqual(loaded["frame"]["rows"][0]["tp"], 1)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

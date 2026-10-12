"""Extra clips stay off the training-scene paths and off the held-out label."""

from __future__ import annotations

import json
import shutil
import unittest

from nuscenes_loader import DATA_ROOT, TRAINING_SCENE, get_scene, list_scenes, scene_context, scene_root


class SceneRootTests(unittest.TestCase):
    def test_monocular_depth_is_saved_for_a_later_scene(self):
        from benchmark import monocular_centers
        from midas_engine import DEPTH_CACHE_ROOT

        self.assertTrue((DEPTH_CACHE_ROOT / "0000" / "CAM_FRONT.npy").is_file())
        night = DEPTH_CACHE_ROOT / "scene-1094" / "0000" / "CAM_FRONT.npy"
        self.assertTrue(night.is_file())
        with scene_context("scene-1094"):
            centers = monocular_centers(0, 0.38)
        self.assertGreater(len(centers), 0)
        self.assertEqual(get_scene(), TRAINING_SCENE)

    def test_training_scene_stays_at_the_original_root(self):
        self.assertEqual(get_scene(), TRAINING_SCENE)
        self.assertEqual(scene_root("scene-0103"), DATA_ROOT)
        self.assertEqual(scene_root("scene-0061"), DATA_ROOT / "scenes" / "scene-0061")

    def test_cross_scene_protocol_keeps_held_out_separate_from_unseen(self):
        from benchmark import cross_scene_protocol

        doc = cross_scene_protocol()
        self.assertEqual(doc["voxel_m"], 1.0)
        by_name = {row["name"]: row for row in doc["scenes"]}
        self.assertEqual(
            list(by_name),
            ["scene-0103", "scene-0061", "scene-0796", "scene-1094"],
        )
        self.assertEqual(by_name["scene-0103"]["label"], "Held-out")
        self.assertEqual(by_name["scene-0103"]["detail"], "held-out")
        self.assertEqual(by_name["scene-0103"]["frames"], 8)
        self.assertEqual(by_name["scene-1094"]["detail"], "night")
        self.assertEqual(by_name["scene-0061"]["label"], "Unseen")
        self.assertGreater(by_name["scene-0061"]["frames"], 8)
        for row in doc["scenes"]:
            camera = next(score for score in row["scores"] if score["pipeline"] == "Camera-space")
            self.assertGreater(camera["iou"], 0.0)
            self.assertNotIn("error", camera)

    def test_prediction_cache_does_not_cross_scenes(self):
        import numpy as np

        from ipm_baseline import _CENTER_CACHE as ipm_cache
        from ipm_baseline import predict_centers as ipm_centers
        from stereo_baseline import _CENTER_CACHE as stereo_cache
        from stereo_baseline import predict_centers as stereo_centers
        from voxnet_baseline import _PRED_CACHE
        from voxnet_baseline import predict_centers as voxnet_centers

        boston = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        night = np.array([[9.0, 0.0, 0.0]], dtype=np.float32)
        _PRED_CACHE.clear()
        ipm_cache.clear()
        stereo_cache.clear()
        _PRED_CACHE[("scene-0103", 0)] = boston
        _PRED_CACHE[("scene-1094", 0)] = night
        ipm_cache[("scene-0103", 0, 1.0)] = boston
        ipm_cache[("scene-1094", 0, 1.0)] = night
        stereo_cache[("scene-0103", 0, 1.0)] = boston
        stereo_cache[("scene-1094", 0, 1.0)] = night
        try:
            with scene_context("scene-0103"):
                self.assertEqual(float(voxnet_centers(0)[0, 0]), 1.0)
                self.assertEqual(float(ipm_centers(0, 1.0)[0, 0]), 1.0)
                self.assertEqual(float(stereo_centers(0, 1.0)[0, 0]), 1.0)
            with scene_context("scene-1094"):
                self.assertEqual(float(voxnet_centers(0)[0, 0]), 9.0)
                self.assertEqual(float(ipm_centers(0, 1.0)[0, 0]), 9.0)
                self.assertEqual(float(stereo_centers(0, 1.0)[0, 0]), 9.0)
        finally:
            _PRED_CACHE.clear()
            ipm_cache.clear()
            stereo_cache.clear()
            self.assertEqual(get_scene(), TRAINING_SCENE)

    def test_listed_scenes_mark_only_the_training_clip(self):
        names = {row["name"]: row for row in list_scenes()}
        self.assertEqual(
            set(names),
            {"scene-0103", "scene-0061", "scene-0796", "scene-1094"},
        )
        training = [name for name, row in names.items() if row["training_scene"]]
        self.assertEqual(training, ["scene-0103"])

    def test_unseen_scene_is_not_called_held_out(self):
        from benchmark import _HELDOUT_BY_SCENE, _frame_split, comparison_heldout
        from nuscenes_loader import _load_manifest

        root = DATA_ROOT / "scenes" / "scene-test"
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "scene_name": "scene-test",
                    "training_scene": False,
                    "frame_count": 0,
                    "frames": [],
                }
            )
        )
        _load_manifest.cache_clear()
        _HELDOUT_BY_SCENE.pop("scene-test", None)
        try:
            with scene_context("scene-test"):
                self.assertEqual(get_scene(), "scene-test")
                self.assertEqual(_frame_split(0), "unseen")
                summary = comparison_heldout()
                self.assertEqual(summary["label"], "Unseen")
                self.assertEqual(summary["frames"], 0)
                self.assertEqual(summary["scene"], "scene-test")
                pipelines = [row["pipeline"] for row in summary["scores"]]
                self.assertIn("VoxNet 3D CNN", pipelines)
                self.assertIn("Ray surface", pipelines)
                self.assertNotIn("monocular", summary)
            with scene_context("scene-0103"):
                self.assertEqual(_frame_split(0), "held-out")
                self.assertEqual(_frame_split(1), "train")
        finally:
            _HELDOUT_BY_SCENE.pop("scene-test", None)
            shutil.rmtree(root, ignore_errors=True)
            _load_manifest.cache_clear()
            self.assertEqual(get_scene(), TRAINING_SCENE)


if __name__ == "__main__":
    unittest.main()

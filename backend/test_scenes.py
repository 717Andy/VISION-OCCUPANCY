"""A second clip stays off the training-scene paths and off the held-out label."""

from __future__ import annotations

import json
import shutil
import unittest

from nuscenes_loader import DATA_ROOT, TRAINING_SCENE, get_scene, list_scenes, scene_context, scene_root


class SceneRootTests(unittest.TestCase):
    def test_training_scene_stays_at_the_original_root(self):
        self.assertEqual(get_scene(), TRAINING_SCENE)
        self.assertEqual(scene_root("scene-0103"), DATA_ROOT)
        self.assertEqual(scene_root("scene-0061"), DATA_ROOT / "scenes" / "scene-0061")

    def test_listed_scenes_mark_only_the_training_clip(self):
        names = {row["name"]: row for row in list_scenes()}
        self.assertIn("scene-0103", names)
        self.assertTrue(names["scene-0103"]["training_scene"])

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

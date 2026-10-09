"""Known-space labels for the camera-space occupancy model."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from camera_space_baseline import CameraSpaceOccupancy, _cell_indices, _splat_volume, known_grids, known_space_volumes
from projection import EGO_BOUNDS
from voxnet_baseline import _centers_from_mask, grid_centers


class CameraSpaceTargetTests(unittest.TestCase):
    def test_mask_covers_only_occupied_and_free_cells(self):
        _centers, shape = grid_centers()
        target, mask = known_space_volumes(0, shape)
        occupied, free = known_grids(0)
        occupied_idx = _cell_indices(occupied, shape)
        free_idx = _cell_indices(free, shape)

        self.assertGreater(occupied_idx.shape[0], 0)
        self.assertGreater(free_idx.shape[0], 0)
        self.assertTrue(np.all(mask[occupied_idx[:, 0], occupied_idx[:, 1], occupied_idx[:, 2]] == 1.0))
        self.assertTrue(np.all(target[occupied_idx[:, 0], occupied_idx[:, 1], occupied_idx[:, 2]] == 1.0))
        self.assertTrue(np.all(mask[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]] == 1.0))
        self.assertTrue(np.all(target[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]] == 0.0))
        self.assertTrue(np.all(target[mask == 0.0] == 0.0))
        known = np.zeros(shape, dtype=bool)
        known[occupied_idx[:, 0], occupied_idx[:, 1], occupied_idx[:, 2]] = True
        known[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]] = True
        self.assertTrue(np.array_equal(mask.astype(bool), known))

    def test_untrained_forward_stays_inside_ego_bounds(self):
        centers, shape = grid_centers()
        model = CameraSpaceOccupancy()
        model.eval()
        with torch.inference_mode():
            volume = _splat_volume(model, 0, shape)
            prob = torch.sigmoid(model.forward_logits(volume))[0, 0].cpu().numpy()
        predicted = _centers_from_mask(prob >= 0.05, centers, shape)
        self.assertGreater(predicted.shape[0], 0)
        self.assertTrue(np.all(predicted >= EGO_BOUNDS[:, 0] - 1e-3))
        self.assertTrue(np.all(predicted < EGO_BOUNDS[:, 1] + 1e-3))


if __name__ == "__main__":
    unittest.main()

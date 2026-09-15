"""Runtime dependency checks for the backend venv."""

from __future__ import annotations

import unittest

from deps import REQUIRED, missing_modules, require_packages


class DependencyCheckTests(unittest.TestCase):
    def test_required_list_includes_pillow(self):
        names = {pip_name for _, pip_name in REQUIRED}
        self.assertIn("pillow", names)
        self.assertIn("numpy", names)

    def test_installed_required_modules_are_importable(self):
        self.assertEqual(missing_modules(REQUIRED), [])
        optional_missing = require_packages()
        self.assertIsInstance(optional_missing, list)


if __name__ == "__main__":
    unittest.main()

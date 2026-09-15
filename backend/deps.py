"""Fail fast with an install recipe when the backend venv is missing packages."""

from __future__ import annotations

import os
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"

# Import name → pip name
REQUIRED = (
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
)
OPTIONAL = (
    ("torch", "torch"),
    ("timm", "timm"),
    ("cv2", "opencv-python-headless"),
    ("open3d", "open3d"),
)


def missing_modules(pairs: tuple[tuple[str, str], ...]) -> list[str]:
    missing: list[str] = []
    for module_name, pip_name in pairs:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def install_help(missing: list[str]) -> str:
    conda_note = ""
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"):
        conda_note = (
            "Your prompt looks like conda is still active (for example `(venv) (base)`).\n"
            "Deactivate conda so only the project venv is used:\n"
            "  conda deactivate\n\n"
        )
    packages = ", ".join(missing)
    return (
        f"Missing Python packages: {packages}\n\n"
        f"{conda_note}"
        "From the backend folder, install into the same interpreter that runs uvicorn:\n"
        "  python -m pip install -r requirements.txt\n"
        "  python -m uvicorn main:app --reload --port 8000\n\n"
        f"Requirements file: {REQUIREMENTS}\n"
        "Pillow provides the `PIL` module. Open3D is optional; without it occupancy "
        "still runs on a NumPy grid."
    )


def require_packages() -> list[str]:
    """Exit if web/runtime deps are missing. Return optional packages that are absent."""
    required_missing = missing_modules(REQUIRED)
    if required_missing:
        raise SystemExit(install_help(required_missing))
    return missing_modules(OPTIONAL)

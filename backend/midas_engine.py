"""PyTorch MiDaS depth engine.

Raw RGB frames are resized to 384×256, run through MiDaS v2.1 small, and
returned as dense 2D NumPy relative-depth (disparity) matrices. CUDA
inference is the production path (acceptance: <30 ms / frame). CPU is a
fallback used when no GPU is present; results are cached as `.npy` so
playback does not re-run the network every seek.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No module named 'PIL'. Install backend deps with "
        "`python -m pip install -r requirements.txt` (Pillow provides PIL)."
    ) from exc

logger = logging.getLogger(__name__)

# Width × height fed to MiDaS_small (R&D downsample).
MIDAS_INPUT_SIZE = (384, 256)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEPTH_CACHE_ROOT = Path(__file__).resolve().parent / "data" / "nuscenes" / "depth"


class MidasDepthEngine:
    """Lazy-loaded MiDaS_small wrapper with optional on-disk depth cache."""

    def __init__(self, device: str | None = None, autoload: bool = False) -> None:
        self._requested_device = device
        self.device = "cpu"
        self.last_infer_ms = 0.0
        self.available = False
        self.error: str | None = None
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        if autoload:
            self.ensure_loaded()

    @property
    def using_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return self.available
        with self._lock:
            if self._loaded:
                return self.available
            try:
                self._load_model()
                self.available = True
                self.error = None
            except Exception as exc:
                self.available = False
                self.error = str(exc)
                logger.warning("MiDaS engine unavailable: %s", exc)
            self._loaded = True
        return self.available

    def _load_model(self) -> None:
        import torch

        if self._requested_device:
            self.device = self._requested_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._trust_hub_repos()
        logger.info("Loading MiDaS_small on %s", self.device)
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, pretrained=True)
        model.to(self.device)
        model.eval()
        self._model = model

    @staticmethod
    def _trust_hub_repos() -> None:
        """Allow MiDaS plus its EfficientNet-lite backbone to load non-interactively."""
        import torch

        hub_dir = Path(torch.hub.get_dir())
        hub_dir.mkdir(parents=True, exist_ok=True)
        trusted_path = hub_dir / "trusted_list"
        existing = trusted_path.read_text() if trusted_path.exists() else ""
        needed = (
            "intel-isl_MiDaS",
            "rwightman_gen-efficientnet-pytorch",
        )
        lines = [line.strip() for line in existing.splitlines() if line.strip()]
        changed = False
        for repo in needed:
            if repo not in lines:
                lines.append(repo)
                changed = True
        if changed or not trusted_path.exists():
            trusted_path.write_text("\n".join(lines) + "\n")

    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run one RGB frame and return an HxW float32 relative-depth map."""
        if not self.ensure_loaded() or self._model is None:
            raise RuntimeError(self.error or "MiDaS model is not available")

        import torch

        tensor = self._preprocess(image_rgb).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = self._model(tensor)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            if prediction.ndim == 4:
                prediction = prediction.squeeze(1)
            if prediction.ndim == 3:
                prediction = prediction.squeeze(0)
            # Canonical 256×384 (H×W) regardless of backbone output stride.
            target_h, target_w = MIDAS_INPUT_SIZE[1], MIDAS_INPUT_SIZE[0]
            if prediction.shape[-2:] != (target_h, target_w):
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            if self.using_cuda:
                torch.cuda.synchronize()
            depth = prediction.detach().float().cpu().numpy().astype(np.float32)
        self.last_infer_ms = (time.perf_counter() - start) * 1000.0
        if depth.ndim != 2:
            raise RuntimeError(f"MiDaS produced non-2D output {depth.shape}")
        return depth

    def infer_cached(self, frame_index: int, camera_id: str, image_rgb: np.ndarray) -> np.ndarray:
        cache_path = DEPTH_CACHE_ROOT / f"{frame_index:04d}" / f"{camera_id}.npy"
        if cache_path.is_file():
            depth = np.load(cache_path)
            if depth.ndim == 2 and depth.dtype == np.float32:
                self.last_infer_ms = 0.0
                return depth
        depth = self.infer(image_rgb)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, depth)
        return depth

    def _preprocess(self, image_rgb: np.ndarray) -> "object":
        import torch

        rgb = np.asarray(image_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        image = image.resize(MIDAS_INPUT_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor

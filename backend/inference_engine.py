"""
Urban Safety AI — Standalone Inference Engine
Supports: ResNet-18 TorchScript (.pt) and raw checkpoint (.pth)
GPU auto-detection, half-precision, lazy loading, thread-safe.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F

logger = logging.getLogger("urban_safety.inference")

CLASS_NAMES = ["accident", "fire", "normal"]


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


class InferenceResult:
    __slots__ = ("label", "confidence", "probabilities", "latency_ms", "device")

    def __init__(self, label, confidence, probabilities, latency_ms, device):
        self.label         = label
        self.confidence    = confidence
        self.probabilities = probabilities
        self.latency_ms    = latency_ms
        self.device        = device

    def to_dict(self) -> dict:
        return {
            "label":          self.label,
            "confidence":     round(self.confidence, 4),
            "probabilities":  {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms":     round(self.latency_ms, 2),
            "device":         self.device,
        }

    def __repr__(self):
        return f"InferenceResult(label={self.label!r}, conf={self.confidence:.3f})"


class InferenceEngine:
    """
    Thread-safe PyTorch classifier with lazy loading.
    Supports TorchScript .pt and standard ResNet checkpoints .pth
    Auto-selects CUDA / MPS / CPU.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path  = model_path
        self._model       = None
        self._lock        = threading.Lock()
        self._transform   = _build_transform()
        self._total       = 0
        self._total_ms    = 0.0

        # Device selection
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._dtype  = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._dtype  = torch.float32
        else:
            self._device = torch.device("cpu")
            self._dtype  = torch.float32

        logger.info("InferenceEngine | device=%s dtype=%s", self._device, self._dtype)

        # Attempt immediate load if path provided
        if model_path and Path(model_path).exists():
            try:
                self._load_model(model_path)
            except Exception as e:
                logger.warning("Could not pre-load model: %s", e)

    @property
    def model_ready(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        if self._device.type == "cuda":
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return self._device.type.upper()

    def load_model(self, path: str):
        """Load or reload a model. Thread-safe."""
        with self._lock:
            self._model_path = path
            self._model = None
        self._load_model(path)

    def predict_file(self, path: str) -> InferenceResult:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self._predict_pil(img)

    def predict_frame(self, frame) -> InferenceResult:
        """Accept BGR numpy frame (OpenCV)."""
        from PIL import Image
        rgb = frame[..., ::-1].copy()
        img = Image.fromarray(rgb)
        return self._predict_pil(img)

    def predict_pil(self, img) -> InferenceResult:
        from PIL import Image
        return self._predict_pil(img.convert("RGB"))

    def stats(self) -> dict:
        return {
            "total":      self._total,
            "avg_ms":     round(self._total_ms / max(1, self._total), 2),
            "device":     str(self._device),
            "ready":      self.model_ready,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _predict_pil(self, img) -> InferenceResult:
        model = self._require_model()
        t0 = time.perf_counter()

        tensor = self._transform(img).unsqueeze(0).to(self._device)
        if self._dtype == torch.float16:
            tensor = tensor.half()

        with torch.no_grad():
            out   = model(tensor)
            probs = F.softmax(out.float(), dim=1)
            conf, idx = torch.max(probs, dim=1)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._total    += 1
        self._total_ms += latency_ms

        label      = CLASS_NAMES[int(idx.item())]
        confidence = float(conf.item())
        prob_dict  = {c: float(probs[0, i].item()) for i, c in enumerate(CLASS_NAMES)}

        return InferenceResult(label, confidence, prob_dict, latency_ms, str(self._device))

    def _require_model(self):
        if self._model is not None:
            return self._model
        if not self._model_path:
            raise RuntimeError("No model path set. Train a model first.")
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._load_model(str(path))
        return self._model

    def _load_model(self, path: str):
        path = Path(path)
        logger.info("Loading model: %s → %s", path.name, self._device)

        if path.suffix == ".pt":
            # TorchScript model
            model = torch.jit.load(str(path), map_location=self._device)
            model.eval()
        else:
            # Raw checkpoint (.pth)
            model = self._load_checkpoint(str(path))

        if self._dtype == torch.float16:
            model = model.half()

        with self._lock:
            self._model = model

        logger.info("Model ready | device=%s", self._device)

    def _load_checkpoint(self, path: str):
        from torchvision import models
        checkpoint = torch.load(path, map_location=self._device)
        num_classes = len(CLASS_NAMES)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        model.to(self._device)
        model.eval()
        return model


class DummyEngine:
    """Fallback engine that returns random results when no model is loaded."""

    model_ready = False
    device_name = "CPU (no model)"

    def predict_file(self, path: str) -> InferenceResult:
        return self._fake()

    def predict_frame(self, frame) -> InferenceResult:
        return self._fake()

    def load_model(self, path: str):
        pass

    def _fake(self) -> InferenceResult:
        import random
        probs = [random.random() for _ in range(3)]
        s = sum(probs)
        probs = [p / s for p in probs]
        idx = probs.index(max(probs))
        prob_dict = {c: probs[i] for i, c in enumerate(CLASS_NAMES)}
        return InferenceResult(
            label=CLASS_NAMES[idx],
            confidence=probs[idx],
            probabilities=prob_dict,
            latency_ms=1.0,
            device="CPU (dummy)",
        )

    def stats(self) -> dict:
        return {"ready": False}

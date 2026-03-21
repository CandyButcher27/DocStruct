"""Offline DocLayNet detector using a local Hugging Face model directory."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

from models.detector import Detection, Detector
from schemas.block import BoundingBox
from utils.logging import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = Path("hf_models") / "deformable-detr-doclaynet"
DOC_LAYOUT_LABEL_MAP: Dict[str, str] = {
    "caption": "caption",
    "footnote": "text",
    "formula": "text",
    "list-item": "text",
    "page-footer": "text",
    "page-header": "header",
    "picture": "figure",
    "section-header": "header",
    "table": "table",
    "text": "text",
    "title": "header",
}


def map_doclaynet_label(label: str) -> str | None:
    normalized = label.strip().lower()
    return DOC_LAYOUT_LABEL_MAP.get(normalized)


class LocalDocLayNetDetector(Detector):
    """Full-layout detector backed by a local DocLayNet checkpoint."""

    def __init__(self, model_path: str | Path = MODEL_DIR, confidence_threshold: float = 0.3):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._processor = None
        self._model = None
        self._ready = False
        self._last_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        required = ["config.json", "preprocessor_config.json"]
        missing = [name for name in required if not (self.model_path / name).exists()]
        if missing:
            self._last_error = f"Missing required model files in {self.model_path}: {', '.join(missing)}"
            logger.error(self._last_error)
            return

        try:
            from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

            logger.info(f"Loading DocLayNet model from {self.model_path}")
            self._processor = AutoImageProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = DeformableDetrForObjectDetection.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model.eval()
            self._ready = True
            self._last_error = None
            logger.info("LocalDocLayNetDetector loaded successfully")
        except Exception as exc:
            self._last_error = str(exc)
            self._ready = False
            logger.error(f"Failed to load DocLayNet model from {self.model_path}: {exc}")

    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        if not self._ready or self._processor is None or self._model is None:
            logger.warning("DocLayNet model not loaded - returning empty detections")
            return []
        if not page_image:
            return []

        try:
            import torch
            from PIL import Image

            image = Image.open(io.BytesIO(page_image)).convert("RGB")
            img_w, img_h = image.size
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([[img_h, img_w]])
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )[0]

            detections: List[Detection] = []
            id2label = self._model.config.id2label
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                raw_label = id2label.get(label_id.item(), "")
                mapped = map_doclaynet_label(raw_label)
                if mapped is None:
                    continue

                x0_px, y0_px, x1_px, y1_px = box.tolist()
                scale_x = page_width / img_w
                scale_y = page_height / img_h
                x0 = x0_px * scale_x
                x1 = x1_px * scale_x
                y0 = page_height - (y1_px * scale_y)
                y1 = page_height - (y0_px * scale_y)
                if x1 <= x0 or y1 <= y0:
                    continue

                detections.append(
                    Detection(
                        bbox=BoundingBox(x0=max(0.0, x0), y0=max(0.0, y0), x1=x1, y1=y1),
                        detection_type=mapped,
                        confidence=round(float(score), 4),
                    )
                )
            return detections
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"LocalDocLayNetDetector.detect() failed: {exc}")
            return []

    def get_model_name(self) -> str:
        return f"doclaynet-local ({self.model_path})"

    def is_ready(self) -> bool:
        return self._ready

    def get_last_error(self) -> str | None:
        return self._last_error

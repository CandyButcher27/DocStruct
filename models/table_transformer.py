"""Offline Table Transformer detector."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from models.detector import Detection, Detector
from schemas.block import BoundingBox
from utils.logging import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = Path("hf_models") / "table-transformer"
CONFIDENCE_THRESHOLD = 0.5


class TableTransformerDetector(Detector):
    _LABEL_MAP = {
        "table": "table",
        "table rotated": "table",
    }

    def __init__(self, model_path: str | Path = MODEL_DIR, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None
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
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection

            logger.info(f"Loading table model from {self.model_path}")
            self._processor = DetrImageProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = TableTransformerForObjectDetection.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model.eval()
            self._ready = True
            self._last_error = None
            logger.info("TableTransformerDetector loaded successfully")
        except Exception as exc:
            self._ready = False
            self._last_error = str(exc)
            logger.error(f"Failed to load {self.model_path}: {exc}")

    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        if not self._ready or self._model is None or self._processor is None:
            logger.warning("TableTransformer model not loaded - returning empty detections")
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
            label_names = self._model.config.id2label
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_str = label_names.get(label_id.item(), "")
                det_type = self._LABEL_MAP.get(label_str)
                if det_type is None:
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
                        detection_type=det_type,
                        confidence=round(float(score), 4),
                    )
                )
            return detections
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"TableTransformerDetector.detect() failed: {exc}")
            return []

    def get_model_name(self) -> str:
        return f"table-transformer-local ({self.model_path})"

    def is_ready(self) -> bool:
        return self._ready

    def get_last_error(self) -> str | None:
        return self._last_error

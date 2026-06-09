"""Optional vision model detector (YOLOv8 / DocLayNet).

Heavy dependencies (``ultralytics``, ``pymupdf``) are imported lazily so the
deterministic core never requires them. When no weights are configured the
pipeline runs geometry-only. DocLayNet's 11 classes are mapped onto the five
DocStruct labels via :data:`docstruct.config.DOCLAYNET_LABEL_MAP`.

Pages are rasterized at :data:`docstruct.config.MODEL_DPI`; detections come back
in image pixels and are divided by the render zoom to return to top-left PDF
points, matching the geometry detector's coordinate space.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from docstruct import config
from docstruct.schema import BoundingBox, Proposal

logger = logging.getLogger(__name__)


def available(weights: Optional[str] = None) -> bool:
    """True if ultralytics, pymupdf, and a weights path are all present."""
    weights = weights or config.MODEL_WEIGHTS
    if not weights:
        return False
    try:
        import ultralytics  # noqa: F401
        import fitz  # noqa: F401
    except ImportError:
        return False
    return True


class ModelDetector:
    """Lazy-loaded YOLOv8 layout detector producing model-source proposals."""

    def __init__(
        self,
        weights: Optional[str] = None,
        dpi: int = config.MODEL_DPI,
        conf_threshold: float = config.MODEL_CONF_THRESHOLD,
    ) -> None:
        self.weights = weights or config.MODEL_WEIGHTS
        self.dpi = dpi
        self.conf_threshold = conf_threshold
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if not self.weights:
                raise RuntimeError(
                    "No model weights configured. Set config.MODEL_WEIGHTS or pass "
                    "weights= to ModelDetector, then install the 'model' extra."
                )
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    def _map_label(self, class_name: str) -> Optional[str]:
        return config.DOCLAYNET_LABEL_MAP.get(class_name.strip().lower())

    def detect(self, pdf_path: str) -> List[Proposal]:
        """Run inference over every page of a PDF."""
        import fitz

        model = self._ensure_model()
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        proposals: List[Proposal] = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                proposals.extend(
                    self._detect_pixmap(
                        pix, model, zoom, page_num, page.rect.width, page.rect.height
                    )
                )
        return proposals

    def _detect_pixmap(
        self, pix, model, zoom, page_num, page_w, page_h
    ) -> List[Proposal]:
        import numpy as np

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = img[:, :, :3]

        results = model(img, conf=self.conf_threshold, verbose=False)
        names = model.names

        proposals: List[Proposal] = []
        counter = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self._map_label(names.get(cls_id, str(cls_id)))
                if label is None:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
                bbox = BoundingBox(
                    x0=x0 / zoom,
                    y0=y0 / zoom,
                    x1=x1 / zoom,
                    y1=y1 / zoom,
                    page_width=page_w,
                    page_height=page_h,
                )
                proposals.append(
                    Proposal(
                        label=label,
                        confidence=round(conf, 4),
                        bbox=bbox,
                        source="model",
                        page_num=page_num,
                        proposal_id=f"model_{page_num}_{counter}",
                    )
                )
                counter += 1
        return proposals

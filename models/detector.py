"""Detector interface and detector factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from schemas.block import BoundingBox

DetectionType = Literal["table", "figure", "text", "header", "caption"]


@dataclass
class ClasswiseThresholdConfig:
    """Per-detection-type confidence thresholds."""

    table: float = 0.7
    figure: float = 0.7
    text: float = 0.5
    header: float = 0.6
    caption: float = 0.6

    def get_threshold(self, detection_type: DetectionType) -> float:
        return getattr(self, detection_type, 0.5)


class Detection(BaseModel):
    """A single detection from a vision model."""

    bbox: BoundingBox
    detection_type: DetectionType
    confidence: float = Field(..., ge=0, le=1)


class Detector(ABC):
    @abstractmethod
    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def is_ready(self) -> bool:
        return True

    def get_last_error(self) -> str | None:
        return None


class StubDetector(Detector):
    def __init__(self, seed: int = 42):
        self.seed = seed
        import random

        random.seed(seed)

    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        return []

    def get_model_name(self) -> str:
        return "stub-detector-v1"


class CombinedLayoutDetector(Detector):
    """Combine local DocLayNet layout detections with TableTransformer tables."""

    def __init__(self, threshold_config: ClasswiseThresholdConfig | None = None, doclaynet_confidence_threshold: float = 0.3):
        from models.doclaynet_detector import LocalDocLayNetDetector
        from models.table_transformer import TableTransformerDetector

        self.threshold_config = threshold_config or ClasswiseThresholdConfig()
        self.doclaynet = LocalDocLayNetDetector(confidence_threshold=doclaynet_confidence_threshold)
        self.table_transformer = TableTransformerDetector(threshold_config=self.threshold_config)

    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        detections: List[Detection] = []

        layout_detections = self.doclaynet.detect(page_image, page_width, page_height)
        table_detections = self.table_transformer.detect(page_image, page_width, page_height)

        # Apply class-wise thresholds to filter detections
        def passes_threshold(det: Detection) -> bool:
            threshold = self.threshold_config.get_threshold(det.detection_type)
            return det.confidence >= threshold

        use_layout_tables = not table_detections
        for det in layout_detections:
            if det.detection_type == "table" and not use_layout_tables:
                continue
            if passes_threshold(det):
                detections.append(det)

        for det in table_detections:
            if passes_threshold(det):
                detections.append(det)

        return detections

    def get_model_name(self) -> str:
        return f"combined-layout ({self.doclaynet.get_model_name()} + {self.table_transformer.get_model_name()})"

    def is_ready(self) -> bool:
        return self.doclaynet.is_ready()

    def get_last_error(self) -> str | None:
        errors = []
        if self.doclaynet.get_last_error():
            errors.append(f"doclaynet={self.doclaynet.get_last_error()}")
        if self.table_transformer.get_last_error():
            errors.append(f"table_transformer={self.table_transformer.get_last_error()}")
        return "; ".join(errors) if errors else None


def create_detector(
    detector_type: str = "stub",
    threshold_config: ClasswiseThresholdConfig | None = None,
    doclaynet_confidence_threshold: float = 0.3,
    model_confidence_threshold: float = 0.7,
) -> Detector:
    if threshold_config is None:
        # Use model_confidence_threshold as the default for all classes if no specific config
        config = ClasswiseThresholdConfig(
            table=model_confidence_threshold,
            figure=model_confidence_threshold,
            text=model_confidence_threshold,
            header=model_confidence_threshold,
            caption=model_confidence_threshold,
        )
    else:
        config = threshold_config
    if detector_type == "stub":
        return StubDetector()
    if detector_type == "table_transformer":
        from models.table_transformer import TableTransformerDetector

        return TableTransformerDetector(threshold_config=config)
    if detector_type == "doclaynet":
        from models.doclaynet_detector import LocalDocLayNetDetector

        return LocalDocLayNetDetector(confidence_threshold=doclaynet_confidence_threshold)
    if detector_type == "combined":
        return CombinedLayoutDetector(
            threshold_config=config,
            doclaynet_confidence_threshold=doclaynet_confidence_threshold,
        )
    raise ValueError(
        f"Unknown detector type: '{detector_type}'. Valid options: stub, table_transformer, doclaynet, combined"
    )

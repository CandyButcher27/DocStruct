"""
Detector interface for ML-based block detection.

This module provides an abstract interface for pretrained vision models.
The stub implementation can be replaced with real models (DETR, LayoutLM, etc.)
without modifying pipeline code.
"""

from typing import List, Dict, Literal
from abc import ABC, abstractmethod
from schemas.block import BoundingBox
from pydantic import BaseModel


DetectionType = Literal["table", "figure", "text", "header"]


class Detection(BaseModel):
    """A single detection from a vision model."""
    bbox: BoundingBox
    detection_type: DetectionType
    confidence: float  # Model's raw confidence score
    
    
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detection from a vision model."""
    bbox: BoundingBox
    detection_type: DetectionType
    confidence: float = Field(..., ge=0, le=1)


class Detector(ABC):
    """
    Abstract base class for layout detection models.
    
    Subclasses must implement the detect() method to return bounding boxes
    and classifications for layout elements.
    """
    
    @abstractmethod
    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        """
        Detect layout elements in a page image.
        
        Args:
            page_image: Raw image bytes (PNG format)
            page_width: Page width in points
            page_height: Page height in points
            
        Returns:
            List of detections with bboxes and classifications
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/version of the detection model."""
        pass


class StubDetector(Detector):
    """
    Stub detector for testing and development.
    
    Uses simple geometric heuristics to simulate ML detection.
    This should be replaced with a real model in production.
    
    NOTE: This is NOT a production implementation. It uses basic rules
    to approximate what a real detector would return.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize stub detector.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        self.seed = seed
        import random
        random.seed(seed)
    
    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        """
        Stub detection using placeholder logic.
        
        In production, this would:
        1. Decode page_image
        2. Run through a pretrained model (e.g., DETR, LayoutLM)
        3. Post-process detections
        4. Return structured results
        
        Args:
            page_image: Raw image bytes
            page_width: Page width in points
            page_height: Page height in points
            
        Returns:
            Empty list (stub implementation)
        """
        # In production, would process image through model
        # For now, return empty to rely on rule-based classification
        return []
    
    def get_model_name(self) -> str:
        """Return stub model identifier."""
        return "stub-detector-v1"


class TableNetDetector(Detector):
    """
    Placeholder for a real table detection model.
    
    To integrate a real model:
    1. Install model dependencies (e.g., transformers, torch)
    2. Load model weights in __init__
    3. Implement detect() to run inference
    4. Ensure deterministic behavior (fixed seed, no dropout)
    
    Example integration with HuggingFace:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        
        class TableNetDetector(Detector):
            def __init__(self):
                self.processor = DetrImageProcessor.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.model = DetrForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.model.eval()
            
            def detect(self, page_image, page_width, page_height):
                # Decode image, run model, post-process
                ...
    """
    
    def __init__(self):
        """Initialize with stub configuration."""
        self.model_loaded = False
    
    def detect(self, page_image: bytes, page_width: float, page_height: float) -> List[Detection]:
        """Stub - would run real model here."""
        return []
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return "tablenet-stub"


def create_detector(detector_type: str = "stub") -> Detector:
    """
    Factory function to create detector instances.
    
    Args:
        detector_type: Type of detector ("stub" or "tablenet")
        
    Returns:
        Detector instance
        
    Raises:
        ValueError: If detector_type is unknown
    """
    if detector_type == "stub":
        return StubDetector()
    elif detector_type == "tablenet":
        return TableNetDetector()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
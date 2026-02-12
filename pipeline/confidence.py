"""
Confidence scoring stage.

Ensures all confidence scores are properly computed and validated.
"""

from typing import Dict, Any
from utils.logging import setup_logger


logger = setup_logger(__name__)


def validate_confidence_breakdown(confidence: Dict[str, float]) -> bool:
    """
    Validate that confidence breakdown is correct.
    
    Args:
        confidence: Confidence dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If confidence is invalid
    """
    required_keys = {"rule_score", "model_score", "geometric_score", "final_confidence"}
    
    if not all(key in confidence for key in required_keys):
        raise ValueError(f"Missing required confidence keys: {required_keys - set(confidence.keys())}")
    
    # Check all scores are in [0, 1]
    for key in required_keys:
        score = confidence[key]
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Confidence score {key}={score} not in [0, 1]")
    
    # Check final confidence is correctly computed
    expected_final = (
        0.3 * confidence["rule_score"] +
        0.5 * confidence["model_score"] +
        0.2 * confidence["geometric_score"]
    )
    
    if abs(confidence["final_confidence"] - expected_final) > 0.01:
        raise ValueError(
            f"final_confidence={confidence['final_confidence']:.3f} does not match "
            f"computed value={expected_final:.3f}"
        )
    
    return True


def ensure_confidence_scores(classified_blocks: list) -> list:
    """
    Ensure all blocks have valid confidence scores.
    
    Args:
        classified_blocks: List of classified blocks
        
    Returns:
        Blocks with validated confidence scores
        
    Raises:
        ValueError: If any confidence is invalid
    """
    for i, block_data in enumerate(classified_blocks):
        confidence = block_data.get("confidence")
        
        if not confidence:
            raise ValueError(f"Block {i} missing confidence")
        
        try:
            validate_confidence_breakdown(confidence)
        except ValueError as e:
            raise ValueError(f"Block {i} has invalid confidence: {e}")
    
    logger.debug(f"Validated confidence for {len(classified_blocks)} blocks")
    
    return classified_blocks
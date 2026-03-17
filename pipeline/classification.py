"""
Block classification stage.

Hybrid classification combining geometric rules with ML detections.
Classifies blocks as: text, header, table, figure, caption.
"""

from typing import List, Dict, Any
from pipeline.layout import LayoutBlock
from models.detector import Detector, Detection
from schemas.block import BoundingBox
from utils.geometry import bbox_overlap
from utils.logging import setup_logger


logger = setup_logger(__name__)


def compute_rule_score(
    block: LayoutBlock,
    block_type: str,
    avg_font_size: float
) -> float:
    """
    Compute rule-based classification score.
    
    Uses font size, position, and text characteristics.
    
    Args:
        block: Layout block to classify
        block_type: Type to score for
        avg_font_size: Average font size on page
        
    Returns:
        Score between 0 and 1
    """
    score = 0.0
    
    if block_type == "header":
        # Headers: larger font, short text, near top of page
        if block.avg_font_size > avg_font_size * 1.3:
            score += 0.4
        if len(block.text.split()) < 15:
            score += 0.3
        if block.bbox and block.bbox.y0 < 100:  # Near top
            score += 0.3
            
    elif block_type == "text":
        # Regular text: normal font, longer content
        if 0.8 < block.avg_font_size / avg_font_size < 1.2:
            score += 0.5
        if len(block.text.split()) > 10:
            score += 0.5
            
    elif block_type == "caption":
        # Captions: smaller font, starts with "Figure" or "Table"
        if block.avg_font_size < avg_font_size * 0.9:
            score += 0.4
        text_lower = block.text.lower()
        if text_lower.startswith(("fig", "figure", "table", "image")):
            score += 0.6
            
    elif block_type == "table":
        # Tables: detected by model or has tabular indicators
        # Rule-based is weak for tables - rely more on model
        if any(char in block.text for char in ['\t', '|']):
            score += 0.3
        # Check for aligned numbers (simple heuristic)
        words = block.text.split()
        if len(words) > 5 and sum(1 for w in words if w.replace('.', '').isdigit()) > len(words) * 0.3:
            score += 0.3
            
    elif block_type == "figure":
        # Figures are primarily detected by model (image blocks)
        # Text blocks rarely classified as figures
        score = 0.0
    
    return min(1.0, score)


def compute_model_score(
    block_bbox: BoundingBox,
    detections: List[Detection],
    target_type: str
) -> float:
    """
    Compute model-based classification score.
    
    Finds best matching detection and returns its confidence.
    
    Args:
        block_bbox: Bounding box of block to classify
        detections: All detections from model
        target_type: Type to score for
        
    Returns:
        Score between 0 and 1
    """
    best_score = 0.0
    
    for detection in detections:
        # Map detection types to block types
        det_type_map = {
            "table": "table",
            "figure": "figure",
            "text": "text",
            "header": "header"
        }
        
        if detection.detection_type in det_type_map:
            mapped_type = det_type_map[detection.detection_type]
            
            if mapped_type == target_type:
                # Calculate overlap
                overlap = bbox_overlap(block_bbox, detection.bbox)
                
                if overlap > 0.5:  # Significant overlap
                    # Combine detection confidence with overlap
                    score = detection.confidence * (0.5 + 0.5 * overlap)
                    best_score = max(best_score, score)
    
    return best_score


def compute_geometric_score(block: LayoutBlock, page_width: float, page_height: float) -> float:
    """
    Compute geometric consistency score.
    
    Checks if block has reasonable size, aspect ratio, and position.
    
    Args:
        block: Layout block
        page_width: Page width
        page_height: Page height
        
    Returns:
        Score between 0 and 1
    """
    if not block.bbox:
        return 0.0
    
    score = 1.0
    
    # Check size is reasonable (not too small or too large)
    area_ratio = block.bbox.area() / (page_width * page_height)
    if area_ratio < 0.001:  # Too small
        score *= 0.5
    elif area_ratio > 0.8:  # Suspiciously large
        score *= 0.7
    
    # Check aspect ratio is reasonable
    aspect_ratio = block.bbox.width() / block.bbox.height() if block.bbox.height() > 0 else 1.0
    if aspect_ratio > 10 or aspect_ratio < 0.1:  # Very extreme aspect ratio
        score *= 0.8
    
    # Check position is within page bounds
    if block.bbox.x0 < 0 or block.bbox.y0 < 0:
        score *= 0.3
    if block.bbox.x1 > page_width or block.bbox.y1 > page_height:
        score *= 0.3
    
    return score


def classify_block(
    block: LayoutBlock,
    detections: List[Detection],
    page_width: float,
    page_height: float,
    avg_font_size: float
) -> Dict[str, Any]:
    """
    Classify a single block using hybrid approach.
    
    Combines rule-based, model-based, and geometric scores.
    
    Args:
        block: Layout block to classify
        detections: Model detections for this page
        page_width: Page width
        page_height: Page height
        avg_font_size: Average font size on page
        
    Returns:
        Classification result with type and confidence breakdown
    """
    block_types = ["text", "header", "table", "caption"]
    
    best_type = "text"
    best_scores = {
        "rule_score": 0.0,
        "model_score": 0.0,
        "geometric_score": 0.0,
        "final_confidence": 0.0
    }
    
    # Compute geometric score (same for all types)
    geometric_score = compute_geometric_score(block, page_width, page_height)
    
    # Try each type
    for block_type in block_types:
        rule_score = compute_rule_score(block, block_type, avg_font_size)
        model_score = compute_model_score(block.bbox, detections, block_type) if block.bbox else 0.0
        
        # Weighted combination: 30% rule, 50% model, 20% geometric
        final_score = 0.3 * rule_score + 0.5 * model_score + 0.2 * geometric_score
        
        if final_score > best_scores["final_confidence"]:
            best_type = block_type
            best_scores = {
                "rule_score": rule_score,
                "model_score": model_score,
                "geometric_score": geometric_score,
                "final_confidence": final_score
            }
    
    return {
        "block_type": best_type,
        "confidence": best_scores
    }


def classify_image_block(
    image_bbox: BoundingBox,
    detections: List[Detection],
    page_width: float,
    page_height: float
) -> Dict[str, Any]:
    """
    Classify an image block (always figure, but compute confidence).
    
    Args:
        image_bbox: Bounding box of image
        detections: Model detections
        page_width: Page width
        page_height: Page height
        
    Returns:
        Classification result for figure
    """
    # Images are always figures
    block_type = "figure"
    
    # Rule score: high for images
    rule_score = 1.0
    
    # Model score: check for figure detections
    model_score = compute_model_score(image_bbox, detections, "figure")
    
    # Geometric score
    geometric_score = 1.0
    area_ratio = image_bbox.area() / (page_width * page_height)
    if area_ratio < 0.001 or area_ratio > 0.8:
        geometric_score = 0.7
    
    final_confidence = 0.3 * rule_score + 0.5 * model_score + 0.2 * geometric_score
    
    return {
        "block_type": block_type,
        "confidence": {
            "rule_score": rule_score,
            "model_score": model_score,
            "geometric_score": geometric_score,
            "final_confidence": final_confidence
        }
    }


def calculate_page_avg_font_size(blocks: List[LayoutBlock]) -> float:
    """
    Calculate average font size across all blocks on a page.
    
    Args:
        blocks: List of layout blocks
        
    Returns:
        Average font size (default 12.0 if no blocks)
    """
    if not blocks:
        return 12.0
    
    total_size = sum(block.avg_font_size for block in blocks)
    return total_size / len(blocks)


def classify_blocks(
    layout_blocks: List[LayoutBlock],
    image_blocks: List[Dict[str, Any]],
    detections: List[Detection],
    page_width: float,
    page_height: float
) -> List[Dict[str, Any]]:
    """
    Classify all blocks on a page.
    
    Args:
        layout_blocks: Text layout blocks
        image_blocks: Image blocks from decomposition
        detections: Model detections
        page_width: Page width
        page_height: Page height
        
    Returns:
        List of classified blocks with confidence scores
    """
    classified_blocks = []
    
    # Calculate average font size for page
    avg_font_size = calculate_page_avg_font_size(layout_blocks)
    
    # Classify text blocks
    for block in layout_blocks:
        classification = classify_block(
            block,
            detections,
            page_width,
            page_height,
            avg_font_size
        )
        
        classified_blocks.append({
            "layout_block": block,
            "block_type": classification["block_type"],
            "confidence": classification["confidence"]
        })
    
    # Classify image blocks
    for image in image_blocks:
        classification = classify_image_block(
            image["bbox"],
            detections,
            page_width,
            page_height
        )
        
        classified_blocks.append({
            "image_block": image,
            "block_type": classification["block_type"],
            "confidence": classification["confidence"]
        })
    
    logger.debug(f"Classified {len(classified_blocks)} blocks")
    
    return classified_blocks
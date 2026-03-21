"""
Table and figure refinement stage.

Extracts structured data from tables using line detection.
Preserves figure metadata.
"""

from typing import List, Dict, Any, Optional
from pipeline.decomposition import PageData
from schemas.block import BoundingBox
from utils.geometry import bbox_overlap
from utils.logging import setup_logger


logger = setup_logger(__name__)


def extract_table_grid(
    table_bbox: BoundingBox,
    lines: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Extract grid structure from ruled table using line detection.
    
    Args:
        table_bbox: Bounding box of table
        lines: Line objects from page
        
    Returns:
        Table structure data or None if no grid found
    """
    if not lines:
        return None
    
    # Find lines within table bbox
    table_lines = []
    for line in lines:
        x0 = min(float(line['x0']), float(line['x1']))
        x1 = max(float(line['x0']), float(line['x1']))
        y0 = min(float(line['y0']), float(line['y1']))
        y1 = max(float(line['y0']), float(line['y1']))
        if x1 <= x0:
            x1 = x0 + 0.1
        if y1 <= y0:
            y1 = y0 + 0.1
        line_bbox = BoundingBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1
        )
        
        if bbox_overlap(table_bbox, line_bbox) > 0.5:
            table_lines.append(line)
    
    if not table_lines:
        return None
    
    # Separate horizontal and vertical lines
    h_lines = [l for l in table_lines if l['orientation'] == 'horizontal']
    v_lines = [l for l in table_lines if l['orientation'] == 'vertical']
    
    if not h_lines or not v_lines:
        return None
    
    # Sort lines
    h_lines.sort(key=lambda l: l['y0'])
    v_lines.sort(key=lambda l: l['x0'])
    
    # Estimate rows and columns
    num_rows = len(h_lines) - 1 if len(h_lines) > 1 else 1
    num_cols = len(v_lines) - 1 if len(v_lines) > 1 else 1
    
    # Build grid structure
    grid = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "row_positions": [l['y0'] for l in h_lines],
        "col_positions": [l['x0'] for l in v_lines],
        "has_ruling": True
    }
    
    logger.debug(f"Extracted table grid: {num_rows}x{num_cols}")
    
    return grid


def refine_table_block(
    block_data: Dict[str, Any],
    page_data: PageData
) -> Dict[str, Any]:
    """
    Refine a table block by extracting grid structure.
    
    Args:
        block_data: Classified block data
        page_data: Raw page data (for line access)
        
    Returns:
        Table structure data
    """
    bbox = None
    if "layout_block" in block_data:
        bbox = block_data["layout_block"].bbox
    
    if not bbox:
        return {"has_ruling": False, "num_rows": 0, "num_cols": 0}
    
    # Try to extract grid
    grid = extract_table_grid(bbox, page_data.lines)
    
    if grid:
        if "table_source" in block_data:
            grid["source"] = block_data["table_source"]
        if "table_evidence" in block_data:
            grid["evidence"] = block_data["table_evidence"]
        return grid
    
    # Fallback: unruled table
    result = {
        "has_ruling": False,
        "num_rows": 0,  # Unknown without parsing
        "num_cols": 0,
        "note": "Unruled table - structure detection not supported in v1"
    }
    if "table_source" in block_data:
        result["source"] = block_data["table_source"]
    if "table_evidence" in block_data:
        result["evidence"] = block_data["table_evidence"]
    return result


def refine_figure_block(
    block_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Refine a figure block by preserving metadata.
    
    Args:
        block_data: Classified block data
        
    Returns:
        Figure metadata
    """
    if "image_block" in block_data:
        image = block_data["image_block"]
        return {
            "width": image.get("width", 0),
            "height": image.get("height", 0),
            "aspect_ratio": image.get("width", 1) / image.get("height", 1) if image.get("height", 0) > 0 else 1.0
        }
    
    return {"width": 0, "height": 0, "aspect_ratio": 1.0}


def refine_blocks(
    classified_blocks: List[Dict[str, Any]],
    page_data: PageData
) -> List[Dict[str, Any]]:
    """
    Refine table and figure blocks with extracted structure.
    
    Args:
        classified_blocks: List of classified blocks
        page_data: Raw page data
        
    Returns:
        Blocks with refined table/figure data
    """
    for block_data in classified_blocks:
        block_type = block_data.get("block_type")
        
        if block_type == "table":
            table_data = refine_table_block(block_data, page_data)
            block_data["table_data"] = table_data
        
        elif block_type == "figure":
            figure_data = refine_figure_block(block_data)
            block_data["image_metadata"] = figure_data
    
    logger.debug(f"Refined {len(classified_blocks)} blocks")
    
    return classified_blocks

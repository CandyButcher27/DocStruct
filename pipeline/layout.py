"""
Layout block formation stage.

Merges text spans into paragraph-level blocks while preserving spatial integrity.
Detects multi-column layouts geometrically.
"""

from typing import List, Tuple, Dict
from pipeline.decomposition import PageData, TextSpan
from schemas.block import BoundingBox
from utils.geometry import merge_bboxes, vertical_distance, is_column_break
from utils.logging import setup_logger


logger = setup_logger(__name__)


class LayoutBlock:
    """
    A preliminary block formed from merged text spans.
    
    Not yet a full Block - lacks classification and confidence.
    """
    
    def __init__(self, page_num: int):
        self.page_num = page_num
        self.spans: List[TextSpan] = []
        self.bbox: BoundingBox | None = None
        self.text: str = ""
        self.avg_font_size: float = 0.0
        self.dominant_font: str = ""
    
    def add_span(self, span: TextSpan) -> None:
        """Add a text span to this block."""
        self.spans.append(span)
        self._update_properties()
    
    def _update_properties(self) -> None:
        """Recompute block properties from spans."""
        if not self.spans:
            return
        
        # Merge bounding boxes
        self.bbox = merge_bboxes([span.bbox for span in self.spans])
        
        # Concatenate text
        self.text = " ".join(span.text for span in self.spans)
        
        # Calculate average font size
        self.avg_font_size = sum(span.font_size for span in self.spans) / len(self.spans)
        
        # Find dominant font (most common)
        font_counts = {}
        for span in self.spans:
            font_counts[span.font_name] = font_counts.get(span.font_name, 0) + 1
        self.dominant_font = max(font_counts, key=font_counts.get) # type: ignore
    
    def __repr__(self) -> str:
        return f"LayoutBlock(text='{self.text[:30]}...', spans={len(self.spans)})"


def should_merge_spans(span1: TextSpan, span2: TextSpan, page_width: float) -> bool:
    """
    Determine if two spans should be merged into the same block.
    
    Criteria:
    - Similar font size (within 20%)
    - Close vertical proximity (< 1.5x font size)
    - Not separated by column break
    - Reading order (left-to-right, top-to-bottom)
    
    Args:
        span1: First text span
        span2: Second text span
        page_width: Page width for column detection
        
    Returns:
        True if spans should be merged
    """
    # Check font size similarity
    font_size_ratio = span2.font_size / span1.font_size if span1.font_size > 0 else 1.0
    if font_size_ratio < 0.8 or font_size_ratio > 1.2:
        return False
    
    # Check vertical distance
    v_dist = vertical_distance(span1.bbox, span2.bbox)
    max_distance = 1.5 * max(span1.font_size, span2.font_size)
    if v_dist > max_distance:
        return False
    
    # Check for column break
    if is_column_break(span1.bbox, span2.bbox, page_width):
        return False
    
    return True


def form_layout_blocks(page_data: PageData) -> List[LayoutBlock]:
    """
    Form layout blocks from text spans using geometric clustering.
    
    Algorithm:
    1. Sort spans by reading order (top-to-bottom, left-to-right)
    2. Greedily merge spans into blocks based on proximity and formatting
    3. Handle multi-column layouts
    
    Args:
        page_data: Raw page data from decomposition
        
    Returns:
        List of layout blocks
    """
    if not page_data.spans:
        logger.debug(f"Page {page_data.page_num}: No spans to form blocks")
        return []
    
    # Sort spans in reading order
    sorted_spans = sorted(
        page_data.spans,
        key=lambda s: (s.bbox.y0, s.bbox.x0)  # Top-to-bottom, left-to-right
    )
    
    blocks = []
    current_block = LayoutBlock(page_data.page_num)
    
    for span in sorted_spans:
        if not current_block.spans:
            # Start first block
            current_block.add_span(span)
        elif should_merge_spans(current_block.spans[-1], span, page_data.width):
            # Merge into current block
            current_block.add_span(span)
        else:
            # Start new block
            if current_block.spans:
                blocks.append(current_block)
            current_block = LayoutBlock(page_data.page_num)
            current_block.add_span(span)
    
    # Add final block
    if current_block.spans:
        blocks.append(current_block)
    
    logger.debug(f"Page {page_data.page_num}: Formed {len(blocks)} layout blocks from {len(sorted_spans)} spans")
    
    return blocks


def detect_columns(blocks: List[LayoutBlock], page_width: float) -> List[List[LayoutBlock]]:
    """
    Detect and group blocks into columns.
    
    Simple column detection based on horizontal clustering.
    
    Args:
        blocks: List of layout blocks
        page_width: Page width
        
    Returns:
        List of columns (each column is a list of blocks)
    """
    if not blocks:
        return []
    
    # Find horizontal clusters
    x_positions = [(b.bbox.x0 + b.bbox.x1) / 2 for b in blocks if b.bbox is not None]
    
    # Filter blocks to only those with valid bounding boxes
    blocks = [b for b in blocks if b.bbox is not None]
    
    # Simple clustering: if there's a large gap, it's a column boundary
    # Use 30% of page width as threshold
    threshold = page_width * 0.3
    
    columns = []
    current_column = []
    last_x = None
    
    # Sort by x position
    if not x_positions:
        return []
    sorted_blocks = sorted(zip(x_positions, blocks), key=lambda x: x[0])
    
    for x, block in sorted_blocks:
        if last_x is None or abs(x - last_x) < threshold:
            current_column.append(block)
        else:
            if current_column:
                columns.append(current_column)
            current_column = [block]
        last_x = x
    
    if current_column:
        columns.append(current_column)
    
    # If only one column detected, return all blocks as single column
    if len(columns) == 1:
        return [blocks]
    
    logger.debug(f"Detected {len(columns)} columns")
    return columns


def process_page_layout(page_data: PageData) -> Tuple[List[LayoutBlock], List[Dict]]:
    """
    Process a page to extract layout blocks and image blocks.
    
    Args:
        page_data: Raw page data
        
    Returns:
        Tuple of (text layout blocks, image blocks)
    """
    logger.debug(f"Processing layout for page {page_data.page_num}")
    
    # Form text blocks
    layout_blocks = form_layout_blocks(page_data)
    
    # Extract image blocks (already have bboxes from decomposition)
    image_blocks = page_data.images
    
    return layout_blocks, image_blocks
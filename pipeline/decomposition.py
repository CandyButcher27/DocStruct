"""
PDF decomposition stage.

Extracts raw text spans, images, and metadata from PDF using pdfplumber.
This stage performs no interpretation - only extraction.
"""

from typing import List, Dict, Any, Optional
import pdfplumber
from pdfplumber.page import Page as PDFPage
from schemas.block import BoundingBox
from utils.logging import setup_logger


logger = setup_logger(__name__)


class TextSpan:
    """
    A single text span with formatting and position.
    
    Represents the atomic unit from PDF extraction.
    """
    
    def __init__(
        self,
        text: str,
        bbox: BoundingBox,
        font_name: str,
        font_size: float,
        page_num: int
    ):
        self.text = text
        self.bbox = bbox
        self.font_name = font_name
        self.font_size = font_size
        self.page_num = page_num
    
    def __repr__(self) -> str:
        return f"TextSpan(text='{self.text[:20]}...', font={self.font_name}, size={self.font_size})"


class PageData:
    """Raw extracted data from a single PDF page."""
    
    def __init__(self, page_num: int, width: float, height: float):
        self.page_num = page_num
        self.width = width
        self.height = height
        self.spans: List[TextSpan] = []
        self.images: List[Dict[str, Any]] = []
        self.lines: List[Dict[str, Any]] = []  # For table detection
    
    def add_span(self, span: TextSpan) -> None:
        """Add a text span to this page."""
        self.spans.append(span)
    
    def add_image(self, image_data: Dict[str, Any]) -> None:
        """Add an image to this page."""
        self.images.append(image_data)
    
    def add_line(self, line_data: Dict[str, Any]) -> None:
        """Add a line (for table detection) to this page."""
        self.lines.append(line_data)


def extract_text_spans(pdf_page: PDFPage, page_num: int) -> List[TextSpan]:
    """
    Extract individual text spans from a PDF page.
    Converts pdfplumber top-origin coordinates to bottom-left origin.
    """
    spans = []

    words = pdf_page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False
    )

    if not words:
        return spans

    page_height = float(pdf_page.height)

    for word in words:
        # Convert coordinate system
        x0 = float(word["x0"])
        x1 = float(word["x1"])

        # pdfplumber gives top-origin distances
        top = float(word["top"])
        bottom = float(word["bottom"])

        # Convert to bottom-left origin
        y0 = page_height - bottom
        y1 = page_height - top

        bbox = BoundingBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1
        )

        span = TextSpan(
            text=word["text"],
            bbox=bbox,
            font_name=word.get("fontname", "Unknown"),
            font_size=float(word.get("height", 10)),
            page_num=page_num
        )

        spans.append(span)

    return spans


def extract_images(pdf_page: PDFPage, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF page.
    
    Args:
        pdf_page: pdfplumber Page object
        page_num: Zero-indexed page number
        
    Returns:
        List of image metadata dictionaries
    """
    images = []
    
    # Extract images using pdfplumber
    page_images = pdf_page.images
    page_height = float(pdf_page.height)
    
    for img in page_images:
        image_data = {
            'bbox': BoundingBox(
                x0=float(img['x0']),
                y0=page_height-float(img['bottom']),
                x1=float(img['x1']),
                y1=page_height-float(img['top'])
            ),
            'width': float(img['width']),
            'height': float(img['height']),
            'page_num': page_num
        }
        images.append(image_data)
    
    return images


def extract_lines(pdf_page: PDFPage) -> List[Dict[str, Any]]:
    """
    Extract line objects for table detection.
    
    Args:
        pdf_page: pdfplumber Page object
        
    Returns:
        List of line dictionaries
    """
    lines = []
    
    # Extract horizontal and vertical lines
    page_lines = pdf_page.lines
    
    for line in page_lines:
        line_data = {
            'x0': float(line['x0']),
            'y0': float(line['top']),
            'x1': float(line['x1']),
            'y1': float(line['bottom']),
            'orientation': 'horizontal' if abs(line['top'] - line['bottom']) < 1 else 'vertical'
        }
        lines.append(line_data)
    
    return lines


def decompose_pdf(pdf_path: str) -> List[PageData]:
    """
    Decompose a PDF into raw extracted data.
    
    This is stage 1 of the pipeline: pure extraction with no interpretation.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of PageData objects (one per page)
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be parsed
    """
    logger.info(f"Decomposing PDF: {pdf_path}")
    
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, pdf_page in enumerate(pdf.pages):
            logger.debug(f"Extracting page {page_num}")
            
            # Create page data container
            page_data = PageData(
                page_num=page_num,
                width=float(pdf_page.width),
                height=float(pdf_page.height)
            )
            
            # Extract text spans
            spans = extract_text_spans(pdf_page, page_num)
            for span in spans:
                page_data.add_span(span)
            
            # Extract images
            images = extract_images(pdf_page, page_num)
            for image in images:
                page_data.add_image(image)
            
            # Extract lines for table detection
            lines = extract_lines(pdf_page)
            for line in lines:
                page_data.add_line(line)
            
            pages_data.append(page_data)
            
            logger.debug(
                f"Page {page_num}: {len(spans)} spans, "
                f"{len(images)} images, {len(lines)} lines"
            )
    
    logger.info(f"Decomposed {len(pages_data)} pages")
    return pages_data
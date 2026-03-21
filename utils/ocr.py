"""
OCR fallback for scanned PDF pages.

When a PDF page contains images but very little/no extractable text
(a scanned document), this module uses Tesseract to recover text spans.

Requires:
  - pytesseract Python package  (pip install pytesseract)
  - Tesseract binary installed on the system:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt install tesseract-ocr
"""

import os
from typing import List, Optional

from utils.logging import setup_logger

logger = setup_logger(__name__)

# Minimum text spans a page must have to be considered "not scanned"
SCANNED_PAGE_SPAN_THRESHOLD = 5


def is_scanned_page(page_data) -> bool:
    """
    Heuristic: return True if the page looks like a scanned image.

    A page is considered scanned when it has at least one embedded image
    but fewer than SCANNED_PAGE_SPAN_THRESHOLD extractable text spans.

    Args:
        page_data: PageData object from decomposition stage.

    Returns:
        True if page appears to be a scanned image.
    """
    has_images = len(page_data.images) > 0
    sparse_text = len(page_data.spans) < SCANNED_PAGE_SPAN_THRESHOLD
    return has_images and sparse_text


def ocr_page(
    image_bytes: bytes,
    page_num: int,
    page_width: float,
    page_height: float,
) -> List:
    """
    Run Tesseract OCR on a rendered page image and return TextSpan objects.

    Requires pytesseract and a system-installed Tesseract binary.
    If Tesseract is not available, returns an empty list gracefully.

    Args:
        image_bytes:  PNG bytes from utils.rendering.render_page_as_png()
        page_num:     Zero-indexed page number (for TextSpan metadata)
        page_width:   PDF page width in points (for coordinate scaling)
        page_height:  PDF page height in points

    Returns:
        List of TextSpan objects, or [] if OCR is unavailable.
    """
    if not image_bytes:
        return []

    # Optional: override Tesseract binary path from env
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        except ImportError:
            pass

    try:
        import pytesseract
        from PIL import Image
        import io

        from pipeline.decomposition import TextSpan
        from schemas.block import BoundingBox

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        # Get word-level bounding boxes from Tesseract
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 3",  # fully automatic page segmentation
        )

        spans: List[TextSpan] = []
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 0:
                continue

            # Tesseract gives pixel top-origin (left, top, width, height)
            left  = data["left"][i]
            top   = data["top"][i]
            width = data["width"][i]
            height_px = data["height"][i]

            # Scale from pixel space to PDF point space
            x0 = left  / img_w * page_width
            x1 = (left + width) / img_w * page_width
            # Convert top-origin to bottom-left origin
            y0 = page_height - ((top + height_px) / img_h * page_height)
            y1 = page_height - (top / img_h * page_height)

            # Guard degenerate boxes
            if x1 <= x0 or y1 <= y0:
                continue

            try:
                bbox = BoundingBox(
                    x0=max(0.0, x0),
                    y0=max(0.0, y0),
                    x1=min(page_width, x1),
                    y1=min(page_height, y1),
                )
                # Use Tesseract confidence as a proxy for font size
                # (real font size unknown from raster; use default 10pt)
                font_size = float(data.get("height", [10])[i]) / img_h * page_height
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=bbox,
                        font_name="OCR",
                        font_size=max(6.0, font_size),
                        page_num=page_num,
                    )
                )
            except Exception:
                continue  # skip invalid boxes

        logger.info(
            f"OCR page {page_num}: extracted {len(spans)} word spans"
        )
        return spans

    except ImportError:
        logger.warning(
            "pytesseract not installed — OCR fallback unavailable. "
            "Install with: pip install pytesseract"
        )
        return []
    except Exception as e:
        logger.error(f"OCR failed on page {page_num}: {e}")
        return []

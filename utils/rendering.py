"""
PDF page rendering utility.

Renders a single PDF page to a PNG-encoded bytes object suitable for
passing to HuggingFace vision models as model input.
"""

import io
from typing import Optional
from utils.logging import setup_logger

logger = setup_logger(__name__)

from pdf2image import convert_from_path
import io

def render_page_as_png(pdf_path: str, page_num: int, dpi: int = 300):
    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num + 1,
            last_page=page_num + 1
        )

        if not images:
            return None

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Failed to render page {page_num} of {pdf_path}: {e}")
        return None
"""
PDF page rendering utility.

Renders a single PDF page to a PNG-encoded bytes object suitable for
passing to HuggingFace vision models as model input.
"""

import io
from utils.logging import setup_logger

logger = setup_logger(__name__)

from pdf2image import convert_from_path
import pdfplumber


def _render_with_pdf2image(pdf_path: str, page_num: int, dpi: int = 300) -> bytes | None:
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num + 1,
        last_page=page_num + 1,
    )
    if not images:
        return None
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def _render_with_pdfplumber(pdf_path: str, page_num: int, dpi: int = 300) -> bytes | None:
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_num < 0 or page_num >= len(pdf.pages):
            return None
        image = pdf.pages[page_num].to_image(resolution=dpi).original.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def render_page_as_png(pdf_path: str, page_num: int, dpi: int = 300) -> bytes | None:
    """Render one PDF page to PNG bytes.

    Tries `pdf2image` first (fast/poppler-backed), then falls back to
    `pdfplumber` rendering if poppler is unavailable or misconfigured.
    """
    pdf2image_error: str | None = None
    try:
        rendered = _render_with_pdf2image(pdf_path, page_num, dpi=dpi)
        if rendered:
            return rendered
    except Exception as e:
        pdf2image_error = str(e)
        logger.warning(f"pdf2image render failed for page {page_num} of {pdf_path}: {e}")

    try:
        rendered = _render_with_pdfplumber(pdf_path, page_num, dpi=dpi)
        if rendered:
            logger.info(f"Used pdfplumber fallback rendering for page {page_num} of {pdf_path}")
            return rendered
    except Exception as e:
        if pdf2image_error:
            logger.error(
                f"Failed to render page {page_num} of {pdf_path}. "
                f"pdf2image error: {pdf2image_error}; pdfplumber error: {e}"
            )
        else:
            logger.error(f"Failed to render page {page_num} of {pdf_path}: {e}")

    if pdf2image_error:
        logger.error(
            f"Failed to render page {page_num} of {pdf_path}; pdf2image failed and pdfplumber fallback returned no image."
        )
    return None

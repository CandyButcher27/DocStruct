"""Typed exception hierarchy so callers can catch DocStruct failures without
importing pdfminer/pdfplumber internals.

Every entry point that opens a PDF routes through :func:`open_pdf`, which
translates the ragged pdfminer exception surface (``PDFSyntaxError``,
``PDFPasswordIncorrect``, bare ``Exception`` from corrupt xref tables) into
these three classes.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import pdfplumber


class DocStructError(Exception):
    """Base class for every error DocStruct raises."""


class InvalidPDFError(DocStructError):
    """The file is not a readable PDF (corrupt, truncated, or not a PDF)."""


class EncryptedPDFError(DocStructError):
    """The PDF is password-protected and no (or a wrong) password was given."""


class EmptyDocumentError(DocStructError):
    """The PDF parsed but yielded no extractable content (e.g. scanned/image-only)."""


@contextlib.contextmanager
def open_pdf(pdf_path: str, *, password: Optional[str] = None) -> Iterator["pdfplumber.PDF"]:
    """Open a PDF, translating pdfminer/pdfplumber failures into DocStruct errors."""
    try:
        pdf = pdfplumber.open(pdf_path, password=password or "")
    except Exception as exc:  # pdfminer raises a wide, undocumented surface
        # pdfplumber wraps the real cause (e.g. PDFPasswordIncorrect, often with an
        # empty message) as a generic PdfminerException, so inspect the whole chain's
        # types and messages, not just the outermost exception.
        signal = ""
        cur: Optional[BaseException] = exc
        while cur is not None:
            signal += f" {cur} {type(cur).__name__}"
            cur = cur.__cause__ or cur.__context__
        signal = signal.lower()
        if "password" in signal or "encrypt" in signal:
            raise EncryptedPDFError(f"{pdf_path}: PDF is encrypted or password is wrong") from exc
        raise InvalidPDFError(f"{pdf_path}: could not open PDF ({exc})") from exc
    try:
        yield pdf
    finally:
        pdf.close()

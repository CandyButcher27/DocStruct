"""Fast determinism tripwire (golden) and malformed-input robustness (fuzz).

The no-LLM/same-in-same-out contract makes exact-equality golden tests uniquely
cheap here: any unintended change to chunk output flips the hash in seconds,
without the ~10-minute retrieval benchmark.
"""

import hashlib
import os

import pytest

import docstruct
from docstruct.errors import DocStructError

_PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw-pdfs")
_GOLDEN_PDF = os.path.join(_PDF_DIR, "doc11.pdf")

# Recompute with the snippet in the commit message if a *deliberate* output change
# lands; an accidental flip is the whole point of this test.
_GOLDEN_N_CHUNKS = 4
_GOLDEN_SHA = "8b4c376ab5e9e2605ab0d26c3307a57bbc63162effb011823d450202d2f25923"
# The corpus is gitignored and rebuilt by scripts/fetch_dataset_v2.py, which
# assigns docN.pdf sequentially — so "doc11.pdf" is not a stable identity, it is
# whatever the eleventh fetch happened to land. A rebuild silently repoints this
# test at a different paper, and the mismatch then reads as a chunking regression.
# Pin the input by content: a different PDF skips, only a real output change fails.
_GOLDEN_PDF_SHA = "e37e2df1c9b0dcb2e9b9d64bbd9d13c6a4bbb2f6c4b1a8e6ab4e5cd7e2a86f1b"


def _file_sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


pytestmark = pytest.mark.skipif(
    not os.path.exists(_GOLDEN_PDF) or _file_sha(_GOLDEN_PDF) != _GOLDEN_PDF_SHA,
    reason="golden corpus PDF absent or replaced by a corpus rebuild",
)


def _chunk_hash(doc) -> str:
    blob = "\x1e".join(f"{c.chunk_type}|{c.page_num}|{c.content}" for c in doc.chunks)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_golden_chunks_are_stable():
    doc = docstruct.parse(_GOLDEN_PDF)
    assert len(doc.chunks) == _GOLDEN_N_CHUNKS
    assert _chunk_hash(doc) == _GOLDEN_SHA


def test_parse_is_deterministic():
    assert _chunk_hash(docstruct.parse(_GOLDEN_PDF)) == _chunk_hash(
        docstruct.parse(_GOLDEN_PDF)
    )


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_zero_byte_pdf_raises_docstruct_error(tmp_path):
    with pytest.raises(DocStructError):
        docstruct.parse(_write(tmp_path, "empty.pdf", b""))


def test_corrupt_pdf_raises_docstruct_error(tmp_path):
    with pytest.raises(DocStructError):
        docstruct.parse(_write(tmp_path, "corrupt.pdf", b"%PDF-1.4\ngarbage\n%%EOF"))


def test_truncated_pdf_never_hangs_or_crashes_uncaught(tmp_path):
    raw = open(_GOLDEN_PDF, "rb").read()
    truncated = _write(tmp_path, "truncated.pdf", raw[: len(raw) // 2])
    try:
        doc = docstruct.parse(truncated)
        assert isinstance(doc.chunks, list)  # empty Document is acceptable
    except DocStructError:
        pass  # a typed error is also acceptable; an uncaught crash is not

import tomllib
from pathlib import Path

import pytest

import docstruct
from docstruct.errors import (
    EncryptedPDFError,
    InvalidPDFError,
    open_pdf,
)


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert docstruct.__version__ == declared


def test_open_zero_byte_file_raises_invalid(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(InvalidPDFError):
        with open_pdf(str(empty)):
            pass


def test_open_corrupt_file_raises_invalid(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nnot a real pdf body\n%%EOF")
    with pytest.raises(InvalidPDFError):
        with open_pdf(str(corrupt)):
            pass


def test_open_encrypted_file_raises_encrypted(tmp_path):
    fitz = pytest.importorskip("fitz")
    enc = tmp_path / "enc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(str(enc), encryption=fitz.PDF_ENCRYPT_AES_256,
             owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(EncryptedPDFError):
        with open_pdf(str(enc)):
            pass


def test_encrypted_file_opens_with_password(tmp_path):
    fitz = pytest.importorskip("fitz")
    enc = tmp_path / "enc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(str(enc), encryption=fitz.PDF_ENCRYPT_AES_256,
             owner_pw="owner", user_pw="user")
    doc.close()
    with open_pdf(str(enc), password="user") as pdf:
        assert len(pdf.pages) == 1

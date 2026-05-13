"""Text extraction from plain text, PDF, and image documents."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".html", ".htm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_text(path: Path) -> str:
    """Extract text from a supported text, PDF, or image file."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in IMAGE_SUFFIXES:
        return extract_image_text(path)
    raise ValueError(
        f"Unsupported file type '{path.suffix}'. Use text, PDF, or image files."
    )


def extract_pdf_text(path: Path) -> str:
    """Extract embedded text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to extract PDF text.") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise RuntimeError(
            "No embedded PDF text was found. Convert the page to an image and use OCR."
        )
    return text


def extract_image_text(path: Path) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Install pytesseract to run OCR on images.") from exc

    try:
        with Image.open(path) as image:
            return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Install Tesseract and ensure it is on PATH."
        ) from exc

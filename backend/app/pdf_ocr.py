"""Extract PDF page text for indexing: native text via PyMuPDF, OCR via Tesseract."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# Default "always": rasterize each page and run Tesseract before Qdrant indexing.
# Set SOURCEMIND_OCR_MODE=hybrid to OCR only when native text is sparse (faster for text PDFs).
OCR_MODE = os.getenv("SOURCEMIND_OCR_MODE", "always").strip().lower()
OCR_MIN_NATIVE_CHARS = int(os.getenv("SOURCEMIND_OCR_MIN_NATIVE_CHARS", "80"))
OCR_ZOOM = float(os.getenv("SOURCEMIND_OCR_ZOOM", "2.0"))


def _native_page_text(page: fitz.Page) -> str:
    return (page.get_text("text") or "").strip()


def _ocr_page_text(page: fitz.Page) -> str:
    z = max(1.0, min(OCR_ZOOM, 4.0))
    mat = fitz.Matrix(z, z)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return (pytesseract.image_to_string(img, config="--oem 3 --psm 3") or "").strip()


def extract_pdf_pages_for_indexing(path: Path) -> list[tuple[int, str]]:
    """
    Return (1-based page number, raw page text) for every page.

    - hybrid: run Tesseract when native text is shorter than OCR_MIN_NATIVE_CHARS.
    - always: rasterize each page and prefer OCR text (still falls back to native if OCR is empty).
    """
    mode = OCR_MODE if OCR_MODE in ("hybrid", "always") else "hybrid"
    doc = fitz.open(path)
    out: list[tuple[int, str]] = []

    try:
        for i in range(len(doc)):
            page = doc[i]
            page_no = i + 1
            native = _native_page_text(page)
            use_ocr = mode == "always" or len(native) < OCR_MIN_NATIVE_CHARS
            text = native

            if use_ocr:
                try:
                    ocr = _ocr_page_text(page)
                    if mode == "always":
                        text = ocr or native
                    else:
                        text = ocr or native
                except pytesseract.TesseractNotFoundError:
                    logger.warning(
                        "Tesseract executable not found; install tesseract-ocr. "
                        "Using native PDF text only for page %s.",
                        page_no,
                    )
                    text = native
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OCR failed for page %s: %s", page_no, exc)
                    text = native

            out.append((page_no, text))
    finally:
        doc.close()

    return out

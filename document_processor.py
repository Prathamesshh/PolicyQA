"""
STEP 2: Document Processor — Row 1 of your diagram
Handles PDF, Image, plain-text; applies OCR when needed.
Returns clean text chunks ready for the RAG index.
"""

import io
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from loguru import logger

# ── lazy imports so missing libs don't break everything ──────────────────────
def _try_import(pkg):
    try:
        import importlib
        return importlib.import_module(pkg)
    except ImportError:
        logger.warning(f"Optional package '{pkg}' not installed — some features disabled.")
        return None


@dataclass
class DocumentChunk:
    text: str
    source: str          # filename
    page: int            # page number (0 for plain text)
    chunk_id: int


class DocumentProcessor:
    """
    Unified processor for the three input types in your diagram:
      1. PDF  → pdfplumber (text layer) OR tesseract OCR (scanned)
      2. Image → tesseract OCR
      3. Plain text / .txt / .docx
    Returns a list of DocumentChunks for downstream indexing.
    """

    def __init__(
        self,
        chunk_size: int = 500,       # tokens per chunk (approx words)
        chunk_overlap: int = 50,
        ocr_lang: str = "eng",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_lang = ocr_lang

    # ─────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────
    def process(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        logger.info(f"Processing {path.name}  (type={suffix})")

        if suffix == ".pdf":
            pages = self._process_pdf(path)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            pages = self._process_image(path)
        elif suffix in {".txt", ".md"}:
            pages = self._process_text(path)
        elif suffix == ".docx":
            pages = self._process_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks = self._chunk_pages(pages, source=path.name)
        logger.success(f"Extracted {len(chunks)} chunks from {path.name}")
        return chunks

    # ─────────────────────────────────────────
    # PDF PROCESSING
    # ─────────────────────────────────────────
    def _process_pdf(self, path: Path) -> List[Dict]:
        pdfplumber = _try_import("pdfplumber")
        if pdfplumber is None:
            raise ImportError("pip install pdfplumber")

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 30:          # scanned page — use OCR
                    logger.info(f"  Page {i+1}: sparse text → OCR fallback")
                    text = self._ocr_pdf_page(page)
                pages.append({"page": i + 1, "text": text})
        return pages

    def _ocr_pdf_page(self, page) -> str:
        """Convert a PDF page to image then OCR it."""
        pytesseract = _try_import("pytesseract")
        if pytesseract is None:
            return ""
        img = page.to_image(resolution=300).original
        return pytesseract.image_to_string(img, lang=self.ocr_lang)

    # ─────────────────────────────────────────
    # IMAGE PROCESSING
    # ─────────────────────────────────────────
    def _process_image(self, path: Path) -> List[Dict]:
        pytesseract = _try_import("pytesseract")
        PIL = _try_import("PIL.Image")
        if pytesseract is None or PIL is None:
            raise ImportError("pip install pytesseract Pillow")

        img = PIL.open(str(path))
        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
        return [{"page": 1, "text": text}]

    # ─────────────────────────────────────────
    # PLAIN TEXT
    # ─────────────────────────────────────────
    def _process_text(self, path: Path) -> List[Dict]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"page": 1, "text": text}]

    # ─────────────────────────────────────────
    # DOCX
    # ─────────────────────────────────────────
    def _process_docx(self, path: Path) -> List[Dict]:
        docx = _try_import("docx")
        if docx is None:
            raise ImportError("pip install python-docx")
        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [{"page": 1, "text": text}]

    # ─────────────────────────────────────────
    # CHUNKING  (sliding window)
    # ─────────────────────────────────────────
    def _chunk_pages(self, pages: List[Dict], source: str) -> List[DocumentChunk]:
        chunks = []
        chunk_id = 0
        for pg in pages:
            words = pg["text"].split()
            words = [w for w in words if w.strip()]   # remove blanks
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, max(1, len(words)), step):
                window = words[start: start + self.chunk_size]
                text = " ".join(window).strip()
                if len(text) < 20:
                    continue
                chunks.append(DocumentChunk(
                    text=self._clean(text),
                    source=source,
                    page=pg["page"],
                    chunk_id=chunk_id,
                ))
                chunk_id += 1
        return chunks

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # strip non-ASCII
        return text.strip()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <file_path>")
        sys.exit(1)
    proc = DocumentProcessor()
    chunks = proc.process(sys.argv[1])
    for c in chunks[:3]:
        print(f"\n[Page {c.page} | Chunk {c.chunk_id}]\n{c.text[:200]}…")

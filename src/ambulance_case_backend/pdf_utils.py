from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def extract_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to read the treatment-instruction PDF. Install package dependencies first."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()

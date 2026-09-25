import os
from pathlib import Path
from typing import List, Dict, Any
from app.utils.logging import logger

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class DocumentLoader:
    """Document text extractor for various file formats."""

    @staticmethod
    def load_file(file_path: str) -> List[Dict[str, Any]]:
        """Load and extract text from a file with page numbers and metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        filename = path.name

        if ext == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif ext in [".txt", ".md", ".markdown", ".csv", ".json"]:
            return DocumentLoader._load_text(path)
        else:
            # Fallback text load
            return DocumentLoader._load_text(path)

    @staticmethod
    def _load_pdf(path: Path) -> List[Dict[str, Any]]:
        pages_data = []
        if not PdfReader:
            # Fallback
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [{"text": text, "page": 1, "filename": path.name}]

        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append({
                        "text": text.strip(),
                        "page": i + 1,
                        "filename": path.name
                    })
        except Exception as e:
            logger.error(f"Error reading PDF {path.name}: {e}")
            text = path.read_text(encoding="utf-8", errors="ignore")
            pages_data.append({"text": text, "page": 1, "filename": path.name})

        return pages_data

    @staticmethod
    def _load_text(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{
            "text": text.strip(),
            "page": 1,
            "filename": path.name
        }]


document_loader = DocumentLoader()

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from app.rag.loaders import document_loader
from app.rag.chunking import text_chunker
from app.rag.hybrid import hybrid_search
from app.utils.logging import logger

DOCUMENTS_DIR = Path("./documents")


class IngestionService:
    """Document Ingestion and Indexing Service."""

    def __init__(self):
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    async def ingest_file(self, file_path: str, filename: str = None) -> Dict[str, Any]:
        """Ingest a file from disk or upload into the knowledge base."""
        path = Path(file_path)
        actual_name = filename or path.name

        dest_path = DOCUMENTS_DIR / actual_name
        if path != dest_path:
            shutil.copy2(path, dest_path)

        # 1. Load pages
        pages_data = document_loader.load_file(str(dest_path))
        
        # 2. Chunk text
        chunks = text_chunker.split_pages(pages_data)

        # 3. Add to Hybrid Vector Store
        await hybrid_search.add_documents(chunks)

        logger.info(f"Ingested document '{actual_name}' with {len(chunks)} chunks.")
        return {
            "filename": actual_name,
            "chunks_count": len(chunks),
            "pages_count": len(pages_data),
            "size": dest_path.stat().st_size,
            "status": "indexed"
        }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete a document file and its vector index."""
        dest_path = DOCUMENTS_DIR / filename
        file_deleted = False
        if dest_path.exists():
            dest_path.unlink()
            file_deleted = True

        chunks_removed = hybrid_search.delete_document(filename)
        return {
            "filename": filename,
            "file_deleted": file_deleted,
            "chunks_removed": chunks_removed,
            "status": "deleted"
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents with size and chunk count metadata."""
        if not DOCUMENTS_DIR.exists():
            return []
        
        docs_list = []
        for f in DOCUMENTS_DIR.glob("*"):
            if f.is_file() and not f.name.startswith("."):
                size = f.stat().st_size
                chunks = hybrid_search.get_document_chunk_count(f.name)
                docs_list.append({
                    "filename": f.name,
                    "size": size,
                    "chunks": chunks,
                    "extension": f.suffix.lower().replace(".", "")
                })
        return sorted(docs_list, key=lambda x: x["filename"])


ingestion_service = IngestionService()

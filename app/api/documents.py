"""
Documents API - File Upload, List, Delete
Files are stored in ./documents/, embedded into the Hybrid Vector Store (in-memory + JSON),
AND their metadata is persisted to the PostgreSQL uploaded_documents table.
"""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.rag.ingestion import ingestion_service
from app.database.postgres import get_db_session
from app.utils.logging import logger

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_DOCUMENTS = 30
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


async def _upsert_doc_in_db(filename: str, size: int, chunks: int, ext: str):
    """Persist or update document metadata in PostgreSQL."""
    try:
        async with get_db_session() as db:
            await db.execute(
                text("""
                    INSERT INTO uploaded_documents (filename, size_bytes, chunks_count, extension)
                    VALUES (:filename, :size, :chunks, :ext)
                    ON CONFLICT (filename) DO UPDATE
                        SET size_bytes = EXCLUDED.size_bytes,
                            chunks_count = EXCLUDED.chunks_count,
                            extension = EXCLUDED.extension,
                            uploaded_at = NOW()
                """),
                {"filename": filename, "size": size, "chunks": chunks, "ext": ext},
            )
    except Exception as e:
        logger.warning(f"Could not persist document metadata to DB: {e}")


async def _delete_doc_from_db(filename: str):
    """Remove document metadata from PostgreSQL."""
    try:
        async with get_db_session() as db:
            await db.execute(
                text("DELETE FROM uploaded_documents WHERE filename = :filename"),
                {"filename": filename},
            )
    except Exception as e:
        logger.warning(f"Could not delete document metadata from DB: {e}")


async def _list_docs_from_db():
    """List documents from PostgreSQL (fallback to filesystem if DB unavailable)."""
    try:
        async with get_db_session() as db:
            result = await db.execute(
                text("SELECT filename, size_bytes, chunks_count, extension, uploaded_at FROM uploaded_documents ORDER BY uploaded_at DESC")
            )
            rows = result.fetchall()
            return [
                {
                    "filename": r.filename,
                    "size": r.size_bytes,
                    "chunks": r.chunks_count,
                    "extension": r.extension,
                    "uploaded_at": str(r.uploaded_at),
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"Could not fetch documents from DB, falling back to filesystem: {e}")
        return None


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT, MD, CSV) into the Hybrid RAG engine."""
    # 1. Check maximum files limit
    existing_docs = ingestion_service.list_documents()
    is_new_file = not any(d["filename"] == file.filename for d in existing_docs)
    if len(existing_docs) >= MAX_DOCUMENTS and is_new_file:
        raise HTTPException(
            status_code=400,
            detail=f"Storage limit reached: Maximum of {MAX_DOCUMENTS} documents allowed. Please delete an existing file."
        )

    try:
        temp_dir = Path("./data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / file.filename

        # 2. Check file size during write
        total_size = 0
        with open(temp_file, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    if temp_file.exists():
                        temp_file.unlink()
                    raise HTTPException(
                        status_code=400,
                        detail="File size exceeds the maximum limit of 500 MB."
                    )
                buffer.write(chunk)

        result = await ingestion_service.ingest_file(str(temp_file), filename=file.filename)

        # 3. Persist metadata to PostgreSQL
        ext = Path(file.filename).suffix.lower().replace(".", "")
        await _upsert_doc_in_db(
            filename=result["filename"],
            size=result["size"],
            chunks=result["chunks_count"],
            ext=ext,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_documents():
    """List all indexed documents in the knowledge base."""
    # Try DB first, fall back to filesystem
    db_docs = await _list_docs_from_db()
    if db_docs is not None:
        docs = db_docs
    else:
        docs = ingestion_service.list_documents()

    return {
        "documents": docs,
        "total_count": len(docs),
        "max_limit": MAX_DOCUMENTS,
        "max_size_mb": 500
    }


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document and purge its vector embeddings and keyword index."""
    try:
        res = ingestion_service.delete_document(filename)
        await _delete_doc_from_db(filename)
        return res
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file/{filename}")
async def get_document_file(filename: str):
    """Serve a document file so the user can inspect and verify it directly."""
    doc_path = Path("./documents") / filename
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found.")
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    return FileResponse(path=str(doc_path), filename=filename, media_type=media_type)

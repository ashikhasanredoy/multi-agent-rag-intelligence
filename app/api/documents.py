import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.ingestion import ingestion_service
from app.utils.logging import logger

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_DOCUMENTS = 30
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


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
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_documents():
    """List all indexed documents in the knowledge base."""
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
        return res
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from app.memory.models import (
    MemoryItem,
    MemoryAddRequest,
    MemoryRecallRequest,
    MemoryRecallResult,
    MemoryCategory
)
from app.memory.store import memory_store
from app.utils.logging import logger

router = APIRouter(prefix="/memory", tags=["Memory"])


@router.get("", response_model=dict)
async def list_memories(category: Optional[MemoryCategory] = None):
    """List all stored long-term memories with optional category filtering."""
    try:
        memories = memory_store.list_memories(category=category)
        return {
            "memories": [m.model_dump() for m in memories],
            "total_count": len(memories),
            "categories": [c.value for c in MemoryCategory]
        }
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=MemoryItem)
async def add_memory(payload: MemoryAddRequest):
    """Add a new custom memory item directly to Long-Term Memory."""
    try:
        item = await memory_store.add_memory(
            content=payload.content,
            category=payload.category or MemoryCategory.GENERAL,
            importance=payload.importance or 3,
            metadata=payload.metadata or {"source": "manual_api"}
        )
        return item
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recall", response_model=List[MemoryRecallResult])
async def recall_memories(payload: MemoryRecallRequest):
    """Recall relevant long-term memories matching a query string."""
    try:
        results = await memory_store.recall_memories(
            query=payload.query,
            top_k=payload.top_k or 5,
            category=payload.category,
            min_score=payload.min_score or 0.30
        )
        return results
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    deleted = await memory_store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    return {"status": "success", "message": f"Memory '{memory_id}' deleted."}


@router.delete("")
async def clear_all_memories():
    """Clear all long-term memories."""
    count = await memory_store.clear_memories()
    return {"status": "success", "message": f"Cleared {count} memories."}

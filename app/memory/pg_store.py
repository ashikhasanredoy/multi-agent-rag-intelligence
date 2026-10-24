"""
PostgreSQL-backed Long-Term Memory Store.
Replaces the JSON file-based store with a proper async Postgres DB.
Features:
  - Semantic vector search (cosine similarity via Python)
  - Keyword search
  - Hybrid scoring
  - Category filtering
  - Async CRUD using SQLAlchemy
"""
import json
import math
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from sqlalchemy import text

from app.memory.models import MemoryItem, MemoryCategory, MemoryRecallResult
from app.database.postgres import get_db_session
from app.services.embeddings import embedding_service
from app.utils.logging import logger


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


class PostgresLongTermMemoryStore:
    """
    PostgreSQL-backed Long-Term Memory Store using async SQLAlchemy.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Internal DB helpers
    # ------------------------------------------------------------------ #
    def _row_to_item(self, row) -> MemoryItem:
        embedding = None
        if row.embedding:
            try:
                embedding = json.loads(row.embedding)
            except Exception:
                embedding = None
        meta = row.metadata if isinstance(row.metadata, dict) else {}
        return MemoryItem(
            id=row.id,
            content=row.content,
            category=MemoryCategory(row.category),
            importance=row.importance,
            embedding=embedding,
            created_at=str(row.created_at),
            last_accessed_at=str(row.last_accessed_at),
            access_count=row.access_count,
            metadata=meta,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def add_memory(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        importance: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        embed: bool = True,
    ) -> MemoryItem:
        """Add a new memory item or refresh an existing duplicate."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        # Check for near-duplicate
        existing = await self._find_duplicate(clean_content, category)
        if existing:
            async with self._lock:
                await self._touch_memory(existing.id, metadata)
            return existing

        # Generate embedding
        embedding = None
        if embed:
            try:
                embedding = await embedding_service.get_embedding(clean_content)
            except Exception as e:
                logger.warning(f"Could not embed memory: {e}")

        item = MemoryItem(
            content=clean_content,
            category=category,
            importance=max(1, min(5, importance)),
            embedding=embedding,
            metadata=metadata or {},
        )

        embedding_json = json.dumps(embedding) if embedding else None
        meta_json = json.dumps(item.metadata) if item.metadata else "{}"

        async with self._lock:
            async with get_db_session() as db:
                await db.execute(
                    text("""
                        INSERT INTO long_term_memories
                            (id, content, category, importance, embedding,
                             created_at, last_accessed_at, access_count, metadata)
                        VALUES
                            (:id, :content, :category, :importance, :embedding,
                             :created_at, :last_accessed_at, :access_count, :metadata::jsonb)
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {
                        "id": item.id,
                        "content": item.content,
                        "category": item.category.value,
                        "importance": item.importance,
                        "embedding": embedding_json,
                        "created_at": item.created_at,
                        "last_accessed_at": item.last_accessed_at,
                        "access_count": item.access_count,
                        "metadata": meta_json,
                    },
                )

        logger.info(f"Added memory [{item.id}] ({category.value}): '{clean_content[:50]}'")
        return item

    async def _find_duplicate(
        self, content: str, category: MemoryCategory
    ) -> Optional[MemoryItem]:
        try:
            async with get_db_session() as db:
                result = await db.execute(
                    text("""
                        SELECT * FROM long_term_memories
                        WHERE LOWER(content) = LOWER(:content)
                          AND category = :category
                        LIMIT 1
                    """),
                    {"content": content, "category": category.value},
                )
                row = result.fetchone()
                if row:
                    return self._row_to_item(row)
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
        return None

    async def _touch_memory(self, memory_id: str, metadata: Optional[Dict] = None):
        now = datetime.utcnow().isoformat()
        try:
            async with get_db_session() as db:
                await db.execute(
                    text("""
                        UPDATE long_term_memories
                        SET last_accessed_at = :now,
                            access_count = access_count + 1,
                            metadata = COALESCE(metadata, '{}'::jsonb) || :meta::jsonb
                        WHERE id = :id
                    """),
                    {
                        "now": now,
                        "meta": json.dumps(metadata or {}),
                        "id": memory_id,
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to touch memory: {e}")

    async def recall_memories(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[MemoryCategory] = None,
        min_score: float = 0.30,
    ) -> List[MemoryRecallResult]:
        """Recall memories using hybrid semantic + keyword scoring."""
        if not query.strip():
            return []

        # Fetch all memories from DB (or filtered by category)
        items = await self.list_memories(category=category)
        if not items:
            return []

        query_tokens = set(query.lower().split())
        query_embedding = None
        try:
            query_embedding = await embedding_service.get_embedding(query)
        except Exception as e:
            logger.warning(f"Embedding failed during recall: {e}")

        scored: List[Tuple[MemoryItem, float]] = []

        for item in items:
            sem_score = 0.0
            if query_embedding and item.embedding:
                sem_score = max(0.0, cosine_similarity(query_embedding, item.embedding))

            item_tokens = set(item.content.lower().split())
            overlap = len(query_tokens.intersection(item_tokens))
            kw_score = min(1.0, overlap / max(1, len(query_tokens)))

            importance_boost = (item.importance - 1) * 0.05

            if query_embedding and item.embedding:
                final_score = 0.7 * sem_score + 0.3 * kw_score + importance_boost
            else:
                final_score = kw_score + importance_boost

            if final_score >= min_score:
                scored.append((item, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = [
            MemoryRecallResult(memory=mem, relevance_score=round(score, 4))
            for mem, score in scored[:top_k]
        ]

        # Update access stats
        if results:
            ids = [r.memory.id for r in results]
            asyncio.create_task(self._touch_many(ids))

        return results

    async def _touch_many(self, ids: List[str]):
        now = datetime.utcnow().isoformat()
        try:
            async with get_db_session() as db:
                for mid in ids:
                    await db.execute(
                        text("""
                            UPDATE long_term_memories
                            SET last_accessed_at = :now, access_count = access_count + 1
                            WHERE id = :id
                        """),
                        {"now": now, "id": mid},
                    )
        except Exception as e:
            logger.warning(f"Failed to touch memories: {e}")

    async def list_memories(
        self, category: Optional[MemoryCategory] = None
    ) -> List[MemoryItem]:
        """List all memories sorted by created_at descending."""
        try:
            async with get_db_session() as db:
                if category:
                    result = await db.execute(
                        text("""
                            SELECT * FROM long_term_memories
                            WHERE category = :cat
                            ORDER BY created_at DESC
                        """),
                        {"cat": category.value},
                    )
                else:
                    result = await db.execute(
                        text("SELECT * FROM long_term_memories ORDER BY created_at DESC")
                    )
                rows = result.fetchall()
                return [self._row_to_item(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            async with get_db_session() as db:
                result = await db.execute(
                    text("DELETE FROM long_term_memories WHERE id = :id"),
                    {"id": memory_id},
                )
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Deleted memory: {memory_id}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def clear_memories(self) -> int:
        """Clear all stored memories."""
        try:
            async with get_db_session() as db:
                result = await db.execute(
                    text("DELETE FROM long_term_memories")
                )
                count = result.rowcount
                logger.info(f"Cleared all {count} memories from PostgreSQL.")
                return count
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0

    # Backward-compat sync wrapper used by memory API router
    def list_memories_sync_unsafe(self) -> List[MemoryItem]:
        """WARNING: Only call from sync context if event loop already running."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.list_memories())
        except Exception:
            return []


# Singleton
memory_store = PostgresLongTermMemoryStore()

"""
PostgreSQL Database Layer
Handles async connection pool, table creation, and session management.
Used for Long-Term Memory persistence.
"""
import json
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from app.config.settings import get_settings
from app.utils.logging import logger


# Global engine and session factory (lazy-initialized)
_engine = None
_async_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.POSTGRES_URL
        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def _get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async database session."""
    session_factory = _get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """
    Create all required tables if they don't exist.
    Called at application startup.
    """
    create_memories_table = """
    CREATE TABLE IF NOT EXISTS long_term_memories (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT 'general',
        importance  INTEGER NOT NULL DEFAULT 3,
        embedding   TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        access_count INTEGER NOT NULL DEFAULT 0,
        metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    """

    create_documents_table = """
    CREATE TABLE IF NOT EXISTS uploaded_documents (
        id              SERIAL PRIMARY KEY,
        filename        TEXT NOT NULL UNIQUE,
        size_bytes      BIGINT NOT NULL DEFAULT 0,
        chunks_count    INTEGER NOT NULL DEFAULT 0,
        extension       TEXT NOT NULL DEFAULT '',
        uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    """

    try:
        engine = _get_engine()
        async with engine.begin() as conn:
            await conn.execute(text(create_memories_table))
            await conn.execute(text(create_documents_table))
        logger.info("PostgreSQL tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL tables: {e}")
        raise


async def close_db():
    """Close the database connection pool."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("PostgreSQL connection pool closed.")

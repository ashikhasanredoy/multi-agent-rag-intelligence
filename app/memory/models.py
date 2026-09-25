import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    USER_PROFILE = "user_profile"       # Name, role, company, contact, location
    PREFERENCE = "preference"           # Coding style, concise answers, dark mode, tone
    FACT = "fact"                       # Domain facts, project knowledge, facts user stated
    PROJECT = "project"                 # Active projects, codebase details, tech stack
    INSTRUCTION = "instruction"         # Persistent custom rules or behavioral constraints
    GENERAL = "general"                 # Miscellaneous long-term observations


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    content: str = Field(..., description="The remembered fact, preference, or piece of knowledge.")
    category: MemoryCategory = Field(default=MemoryCategory.GENERAL, description="Category of memory.")
    importance: int = Field(default=3, ge=1, le=5, description="Salience score from 1 (low) to 5 (critical).")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for semantic search.")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=2, description="Fact or preference to remember.")
    category: Optional[MemoryCategory] = Field(default=MemoryCategory.GENERAL)
    importance: Optional[int] = Field(default=3, ge=1, le=5)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MemoryRecallRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query to recall relevant memories for.")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    category: Optional[MemoryCategory] = None
    min_score: Optional[float] = Field(default=0.35, ge=0.0, le=1.0)


class MemoryRecallResult(BaseModel):
    memory: MemoryItem
    relevance_score: float

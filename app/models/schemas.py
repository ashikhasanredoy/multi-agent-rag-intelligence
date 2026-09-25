from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    agent: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier. If omitted, a new conversation context is generated."
    )
    message: str = Field(..., min_length=1, description="The user's query or prompt message.")
    stream: bool = Field(default=False, description="Whether to stream the assistant's response.")


class SourceCitation(BaseModel):
    document: str = Field(..., description="Filename or title of the source document.")
    page: Optional[int] = Field(default=None, description="Page number of the reference if applicable.")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk reference.")
    score: Optional[float] = Field(default=None, description="Relevance / confidence score of the chunk.")
    snippet: Optional[str] = Field(default=None, description="Brief snippet excerpt.")


class SupervisorDecision(BaseModel):
    intent: str = Field(..., description="Determined intent (e.g., general, rag, web, code, multi_agent).")
    agents: List[str] = Field(default_factory=list, description="List of agents chosen to process the query.")
    requires_rag: bool = False
    requires_web: bool = False
    requires_code: bool = False
    confidence: float = 1.0


class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="Active conversation identifier.")
    answer: str = Field(..., description="Assistant's final response.")
    agent: str = Field(default="general_agent", description="The primary agent or workflow that answered.")
    sources: List[SourceCitation] = Field(default_factory=list, description="Citations used in the response.")
    confidence: Optional[float] = Field(default=1.0, description="Overall confidence score.")
    latency_seconds: Optional[float] = Field(default=None, description="Execution latency in seconds.")


class HealthCheck(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Any] = Field(default_factory=dict)

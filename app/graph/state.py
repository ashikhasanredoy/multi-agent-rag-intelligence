from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from app.models.schemas import SourceCitation


class AgentState(TypedDict, total=False):
    conversation_id: str
    user_query: str
    messages: List[Dict[str, Any]]
    intent: str
    selected_agents: List[str]
    rewritten_query: Optional[str]
    retrieved_documents: List[Any]
    web_results: List[Any]
    code_context: Optional[str]
    agent_results: Dict[str, Any]
    draft_answer: str
    critique: Dict[str, Any]
    retry_count: int
    sources: List[SourceCitation]
    final_answer: str
    confidence: float
    active_agent: str
    latency_seconds: float
    relevant_memories: List[str]
    working_dialogue: Optional[str]
    error: Optional[str]

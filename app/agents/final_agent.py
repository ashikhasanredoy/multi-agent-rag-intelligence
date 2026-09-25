from typing import Dict, Any, List
from app.models.schemas import ChatResponse, SourceCitation
from app.utils.logging import logger


class FinalAgent:
    """Final Response Synthesizer & Formatter."""

    async def format_response(
        self,
        conversation_id: str,
        query: str,
        draft_answer: str,
        agent_name: str,
        sources: List[SourceCitation],
        confidence: float,
        latency: float
    ) -> ChatResponse:
        logger.info(f"FinalAgent assembling response from {agent_name}")
        
        return ChatResponse(
            conversation_id=conversation_id,
            answer=draft_answer,
            agent=agent_name,
            sources=sources,
            confidence=confidence,
            latency_seconds=latency
        )


final_agent = FinalAgent()

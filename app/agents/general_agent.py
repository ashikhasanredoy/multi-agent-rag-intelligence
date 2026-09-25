from typing import Dict, Any, Optional, List
from app.services.llm import llm_service
from app.utils.logging import logger


class GeneralAgent:
    """General Conversation & Reasoning Agent with Long-Term Memory recall."""

    async def execute(
        self,
        query: str,
        memory_context: Optional[str] = None,
        dialogue_context: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"GeneralAgent executing query: {query}")
        
        system_prompt = (
            "You are a helpful, intelligent AI assistant in a Multi-Agent Intelligence Platform.\n"
            "You maintain continuous awareness of user preferences, identity, and past discussions "
            "using long-term memory."
        )

        messages = []
        
        # Inject memory and dialogue context if available
        context_parts = []
        if memory_context:
            context_parts.append(f"### Relevant Long-Term Memory:\n{memory_context}")
        if dialogue_context:
            context_parts.append(f"### Recent Conversation Context:\n{dialogue_context}")

        if context_parts:
            combined_context = "\n\n".join(context_parts)
            messages.append({
                "role": "user",
                "content": f"[SYSTEM CONTEXT]\n{combined_context}\n\n[USER QUERY]\n{query}"
            })
        else:
            messages.append({"role": "user", "content": query})
        
        answer = await llm_service.chat(
            messages=messages,
            system=system_prompt,
            temperature=0.3
        )

        return {
            "draft_answer": answer,
            "sources": [],
            "agent": "general_agent",
            "confidence": 1.0
        }


general_agent = GeneralAgent()

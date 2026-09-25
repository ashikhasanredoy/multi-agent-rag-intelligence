from typing import Dict, Any, Optional
from app.tools.code_tool import code_tool
from app.services.llm import llm_service
from app.utils.logging import logger


class CodeAgent:
    """Code Analysis, Debugging, and Math Evaluation Agent with Memory Awareness."""

    async def execute(
        self,
        query: str,
        memory_context: Optional[str] = None,
        dialogue_context: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"CodeAgent analyzing query: {query}")
        
        system_prompt = (
            "You are an expert Software Engineer & Code Analysis Agent.\n"
            "Analyze the code or technical problem thoroughly, provide clean explanations, "
            "adhere strictly to user preferences and coding standards, "
            "and supply production-grade solutions with syntax highlighting."
        )

        messages = []
        context_parts = []
        if memory_context:
            context_parts.append(f"### Relevant User Preferences & Instructions:\n{memory_context}")
        if dialogue_context:
            context_parts.append(f"### Recent Conversation:\n{dialogue_context}")

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
            temperature=0.1
        )

        return {
            "draft_answer": answer,
            "sources": [],
            "agent": "code_agent",
            "confidence": 0.95
        }


code_agent = CodeAgent()

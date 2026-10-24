from typing import Dict, Any, Optional, List
from app.tools.code_tool import code_tool
from app.services.llm import llm_service
from app.models.schemas import SourceCitation
from app.utils.logging import logger


class CodeAgent:
    """
    Code Generation, Debugging, and Technical Implementation Agent.
    Operates independently of RAG or web search success.
    Capable of incorporating project documentation when retrieved.
    """

    async def execute(
        self,
        query: str,
        project_context: Optional[str] = None,
        sources: Optional[List[SourceCitation]] = None,
        memory_context: Optional[str] = None,
        dialogue_context: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"CodeAgent generating solution for query: {query}")
        
        system_prompt = (
            "You are an expert Senior Software Engineer & Code Generation Agent.\n"
            "Your responsibility is to generate clean, production-ready, well-structured code with necessary dependencies, "
            "edge-case handling, and clear explanations.\n\n"
            "RULES:\n"
            "1. ALWAYS provide a complete, working implementation with syntax highlighting.\n"
            "2. Identify the target programming language (e.g. C++, Python, JavaScript, Java, Rust, Go).\n"
            "3. If relevant project files or documentation context are provided, adhere strictly to existing project conventions.\n"
            "4. NEVER output '[NO_RELEVANT_DOCS]' or claim inability to write code because documents are missing. Generate code using standard language specifications and algorithms.\n"
            "5. Consider key edge cases (e.g. negative weights in Bellman-Ford, file path errors, resource cleanup) and include comments explaining crucial logic."
        )

        messages = []
        context_parts = []
        if project_context:
            context_parts.append(f"### Relevant Project Documents & Code Context:\n{project_context}")
        if memory_context:
            context_parts.append(f"### Relevant User Preferences & Instructions:\n{memory_context}")
        if dialogue_context:
            context_parts.append(f"### Recent Conversation:\n{dialogue_context}")

        if context_parts:
            combined_context = "\n\n".join(context_parts)
            messages.append({
                "role": "user",
                "content": f"[SYSTEM CONTEXT]\n{combined_context}\n\n[USER CODING REQUEST]\n{query}"
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
            "sources": sources or [],
            "agent": "code_agent",
            "confidence": 0.98
        }


code_agent = CodeAgent()

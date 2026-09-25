from typing import Dict, Any, List
from app.tools.search_tool import search_tool
from app.services.llm import llm_service
from app.models.schemas import SourceCitation
from app.utils.logging import logger


class WebAgent:
    """
    Web Search & Real-Time Agent.
    Fetches real-time web, news, and market information, then synthesizes accurate grounded responses.
    """

    async def execute(self, query: str) -> Dict[str, Any]:
        logger.info(f"WebAgent executing query: {query}")
        
        # 1. Search for live data & stock quotes
        web_results = await search_tool.search_web(query, max_results=5)
        
        # Format sources
        sources: List[SourceCitation] = []
        context_snippets = []
        for r in web_results:
            sources.append(SourceCitation(
                document=r.get("title", "Web Evidence"),
                snippet=r.get("snippet", ""),
                score=0.95
            ))
            context_snippets.append(f"Title: {r.get('title')}\nSource: {r.get('source')}\nSnippet: {r.get('snippet')}\n")

        evidence_text = "\n---\n".join(context_snippets) if context_snippets else "No external search results found."

        # 2. Instruct LLM to synthesize based strictly on the retrieved real-time evidence
        system_prompt = (
            "You are a specialized Web & Market Intelligence Agent in a Multi-Agent system.\n"
            "Your task is to answer the user's question using the REAL-TIME LIVE SEARCH EVIDENCE provided below.\n"
            "Rules:\n"
            "1. Base your answer directly on the verified live data provided.\n"
            "2. State the exact prices, facts, or numbers retrieved from the live data.\n"
            "3. DO NOT claim that your knowledge cutoff prevents you from answering, because live data is supplied.\n"
            "4. Format the output cleanly with bullet points, currencies, and timestamps."
        )

        user_content = f"User Question:\n{query}\n\nReal-Time Retrieved Evidence:\n{evidence_text}\n\nPlease generate a direct, accurate answer."

        messages = [{"role": "user", "content": user_content}]
        draft_answer = await llm_service.chat(
            messages=messages,
            system=system_prompt,
            temperature=0.1
        )

        return {
            "draft_answer": draft_answer,
            "sources": sources,
            "raw_evidence": web_results,
            "agent": "web_agent",
            "confidence": 0.95 if web_results else 0.5
        }


web_agent = WebAgent()

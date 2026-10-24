from typing import Dict, Any, List
from app.tools.retrieval_tool import retrieval_tool
from app.services.llm import llm_service
from app.models.schemas import SourceCitation
from app.utils.logging import logger


class RAGAgent:
    """Document Ingestion & Knowledge Base Retrieval Agent."""

    async def execute(self, query: str) -> Dict[str, Any]:
        logger.info(f"RAGAgent executing query: {query}")
        
        retrieved_docs = await retrieval_tool.retrieve_documents(query, top_k=4)
        
        sources: List[SourceCitation] = []
        context_blocks = []

        for doc in retrieved_docs:
            fname = doc.get("filename", "Knowledge Base Document")
            sources.append(SourceCitation(
                document=fname,
                page=doc.get("page", 1),
                chunk_id=doc.get("chunk_id"),
                snippet=doc.get("content", "")[:200] + "...",
                url=f"/documents/file/{fname}" if fname else None,
                score=doc.get("score", 0.95)
            ))
            context_blocks.append(
                f"[Document: {doc.get('filename')}, Page: {doc.get('page')}]\n{doc.get('content')}"
            )

        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No matching documents found in knowledge base."

        system_prompt = (
            "You are a specialized Document Intelligence & RAG Agent.\n"
            "Your task is to answer the user's question accurately using ONLY the provided verified document evidence.\n\n"
            "STRICT RULES:\n"
            "1. State the exact policy deadlines, section rules, and details specified in the evidence.\n"
            "2. Cite the document name and section/page if available.\n"
            "3. RESEARCH PROPOSALS vs FINDINGS: If the document is a Research Proposal, you MUST clearly distinguish between Research Questions, Hypotheses (e.g. H1-H4), Proposed Methodology, and Experimental Results. NEVER describe proposed hypotheses or intended experiments as proven findings or completed results if no experiments were conducted.\n"
            "4. If the retrieved documents DO NOT contain information to answer the user's question, respond with: '[NO_RELEVANT_DOCS] The provided documents do not contain information to answer this question.'\n"
            "5. Keep the answer structured and clear with bullet points and headings.\n"
            "6. NEVER include code snippets or programming examples unless the source document itself contains code."
        )

        user_content = f"User Question:\n{query}\n\nRetrieved Policy Documents Context:\n{context_text}\n\nPlease generate a grounded, accurate response."

        messages = [{"role": "user", "content": user_content}]
        answer = await llm_service.chat(
            messages=messages,
            system=system_prompt,
            temperature=0.1
        )

        return {
            "draft_answer": answer,
            "sources": sources,
            "agent": "rag_agent",
            "confidence": 0.95 if retrieved_docs else 0.4
        }


rag_agent = RAGAgent()

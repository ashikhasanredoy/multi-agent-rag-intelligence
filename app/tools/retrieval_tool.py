from typing import List, Dict, Any
from app.rag.hybrid import hybrid_search
from app.utils.logging import logger


class RetrievalTool:
    """Document & Knowledge Base Hybrid Retrieval Tool."""

    @staticmethod
    async def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks from indexed documents using Hybrid Search."""
        try:
            results = await hybrid_search.search(query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []


retrieval_tool = RetrievalTool()

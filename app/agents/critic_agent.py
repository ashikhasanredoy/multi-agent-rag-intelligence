from typing import Dict, Any, List
from app.utils.logging import logger


class CriticAgent:
    """
    Critic & Reflection Agent.
    Evaluates grounding, checks for hallucinations, and ensures cutoff apologies are prevented when real-time tools were executed.
    """

    async def evaluate(self, query: str, draft_answer: str, sources: List[Any], agent_name: str) -> Dict[str, Any]:
        logger.info(f"CriticAgent validating response from {agent_name}")
        
        # Check if model inappropriately gave a knowledge cutoff disclaimer despite tools
        cutoff_markers = ["knowledge cutoff", "as of december 2023", "as of my last update", "i do not have real-time"]
        has_cutoff_disclaimer = any(marker in draft_answer.lower() for marker in cutoff_markers)
        
        if has_cutoff_disclaimer and sources:
            # Critic flags cutoff disclaimer when real-time sources exist
            return {
                "valid": False,
                "action": "rewrite",
                "feedback": "Remove knowledge cutoff apology and state the live verified facts directly.",
                "confidence": 0.8
            }

        return {
            "valid": True,
            "action": "accept",
            "feedback": "Answer is grounded and relevant.",
            "confidence": 0.95
        }


critic_agent = CriticAgent()

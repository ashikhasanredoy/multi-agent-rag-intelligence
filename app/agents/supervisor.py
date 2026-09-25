import json
from typing import Dict, Any, List
from app.models.schemas import SupervisorDecision
from app.services.llm import llm_service
from app.utils.logging import logger


class SupervisorAgent:
    """
    Supervisor / Router Agent.
    Evaluates user intent and dynamically assigns work to specialized agents.
    """

    WEB_KEYWORDS = [
        "current stock", "stock price", "market cap", "weather today",
        "latest release", "who won", "breaking news", "live price", "today's price",
        "ticker"
    ]

    RAG_KEYWORDS = [
        "policy", "under this policy", "lost laptop", "security breach", "deadline",
        "byod", "classification", "tier 1", "tier 2", "tier 3", "tier 4", "restricted",
        "document", "uploaded", "paper", "pdf", "file", "my notes",
        "chapter", "dataset in the paper", "according to the doc", "dlp", "mdm",
        "whistleblower", "surveillance", "monitoring", "cctv", "turnstile"
    ]

    CODE_KEYWORDS = [
        "def ", "function", "class ", "syntax error", "bug", "python",
        "javascript", "calculate", "math", "evaluate", "traceback", "import ",
        "sql", "regex", "algorithm"
    ]

    async def route(self, query: str, context: List[Dict[str, str]] = None) -> SupervisorDecision:
        lower_q = query.lower()

        # 1. RAG heuristic for document, policy, and compliance inquiries
        if any(w in lower_q for w in self.RAG_KEYWORDS):
            return SupervisorDecision(
                intent="rag",
                agents=["rag_agent"],
                requires_rag=True,
                confidence=0.95
            )

        # 2. Web search heuristic for live market, news, stock prices
        if any(w in lower_q for w in self.WEB_KEYWORDS):
            return SupervisorDecision(
                intent="web",
                agents=["web_agent"],
                requires_web=True,
                confidence=0.95
            )

        # 3. Code heuristic
        if any(w in lower_q for w in self.CODE_KEYWORDS):
            return SupervisorDecision(
                intent="code",
                agents=["code_agent"],
                requires_code=True,
                confidence=0.90
            )

        # 4. Conversational / general fallback
        return SupervisorDecision(
            intent="general",
            agents=["general_agent"],
            confidence=1.0
        )


supervisor_agent = SupervisorAgent()

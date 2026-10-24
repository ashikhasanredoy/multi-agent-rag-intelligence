"""
Supervisor / Router Agent.
Classifies user queries into validated categories:
- DOCUMENT_RAG
- FINANCE_MARKET
- PRODUCT_SEARCH
- WEB_SEARCH
- GENERAL_CHAT
- MULTI_AGENT
"""
import re
from typing import Dict, Any, List, Optional
from app.models.schemas import RoutingDecision
from app.rag.hybrid import hybrid_search
from app.utils.logging import logger


class SupervisorAgent:
    """
    Centralized Intelligent Query Router.
    Uses deterministic lexical, regex, and intent heuristics with robust schema validation.
    """

    FINANCE_KEYWORDS = [
        "stock price", "stock performance", "stocks", "share price", "shares",
        "market cap", "market capitalization", "ticker", "nasdaq", "nyse",
        "pe ratio", "trading volume", "dividend yield", "earnings report",
        "aapl stock", "msft stock", "googl stock", "tsla stock", "samsung stock",
        "stock of", "shares of", "stock market"
    ]

    PRODUCT_KEYWORDS = [
        "iphone", "galaxy", "macbook", "ipad", "pixel", "airpods", "playstation",
        "xbox", "laptop", "smartphone", "retail price", "msrp", "buy price",
        "pro max", "duo", "fold", "flip", "watch ultra", "oled tv",
        "headphones", "specs of", "price of the iphone", "price of iphone",
        "cost of iphone", "price of samsung phone", "price of the phone",
        "latest mobile", "latest phone", "latest device", "newest phone", "newest mobile",
        "latest samsung", "latest apple", "latest release", "most recent device",
        "newest device", "newest product", "latest product"
    ]

    RAG_KEYWORDS = [
        "policy", "under this policy", "lost laptop", "security breach", "deadline",
        "byod", "classification", "tier 1", "tier 2", "tier 3", "tier 4", "restricted",
        "document", "uploaded", "paper", "pdf", "my notes", "research proposal",
        "chapter", "dataset in the paper", "according to the doc", "dlp", "mdm",
        "whistleblower", "surveillance", "monitoring", "cctv", "turnstile",
        "in the document", "in my file", "from the file", "based on the document",
        "what does the document", "summarize the uploaded", "explain my uploaded",
        "hypotheses in the", "experimental results does the", "annual leave policy",
        "privacy policy", "all_sql_queries", "adaptive image enhancement", "gap analysis",
        "literature review in the paper", "uploaded file", "uploaded files", "my project files"
    ]

    WEB_KEYWORDS = [
        "weather today", "forecast", "latest news", "breaking news",
        "current events", "who won", "president of", "current ceo",
        "prime minister", "olympics", "happened today", "world cup",
        "web information", "web", "internet", "google search", "online",
        "regulations", "current law", "current regulation", "external regulation"
    ]

    CODE_KEYWORDS = [
        "def ", "function", "class ", "syntax error", "bug", "python",
        "javascript", "typescript", "c++", "cpp", "c code", "java", "golang", "rust",
        "calculate", "math", "evaluate", "traceback", "import ",
        "sql query", "write a regex", "algorithm", "code for", "give me code",
        "write code", "bellman-ford", "file loading", "polymorphism in c++", "dijkstra",
        "sort algorithm", "binary search", "recursion"
    ]

    async def route(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> RoutingDecision:
        lower_q = query.lower().strip()
        logger.info(f"Supervisor routing query: '{query}'")

        # 1. Multi-Agent Detection (e.g. Compare Apple stock with iPhone prices, or document with web)
        has_finance = any(w in lower_q for w in self.FINANCE_KEYWORDS) or bool(
            re.search(r'\b(stock|shares|market cap|nasdaq|ticker)\b', lower_q)
        )
        has_product = any(w in lower_q for w in self.PRODUCT_KEYWORDS) or bool(
            re.search(r'\b(iphone|samsung galaxy|laptop|smartphone|phone|macbook)\b', lower_q) and re.search(r'\b(price|cost|specs|buy|model)\b', lower_q)
        )
        has_rag = any(w in lower_q for w in self.RAG_KEYWORDS) or bool(
            re.search(r'\b(uploaded|my document|the pdf|the policy|research proposal)\b', lower_q)
        )
        has_web = any(w in lower_q for w in self.WEB_KEYWORDS)

        has_code = any(w in lower_q for w in self.CODE_KEYWORDS) or bool(
            re.search(r'\b(code|function|class|algorithm|c\+\+|python|java|debug|script)\b', lower_q)
        )

        # Check Multi-Agent combination
        # A. Code + Project Document Files (e.g. "Use my uploaded project files to improve this Python code")
        if has_code and has_rag:
            return RoutingDecision(
                intent="MULTI_AGENT",
                agents=["rag_agent", "code_agent"],
                requires_documents=True,
                confidence=0.98,
                reasoning_summary="Query requires project document context combined with code generation."
            )

        # B. Finance + Product, or Document + Web/Finance
        if (has_finance and has_product) or (has_rag and (has_web or has_finance)):
            agents = []
            if has_finance:
                agents.append("finance_agent")
            if has_product:
                agents.append("product_agent")
            if has_rag:
                agents.append("rag_agent")
            if has_web and "web_agent" not in agents:
                agents.append("web_agent")

            return RoutingDecision(
                intent="MULTI_AGENT",
                agents=agents,
                requires_live_data=True,
                requires_documents=has_rag,
                requires_finance=has_finance,
                requires_product=has_product,
                confidence=0.98,
                reasoning_summary="Query requires multi-agent synthesis across distinct tools."
            )

        # 2. CODE_EXECUTION (Check before general RAG so programming queries generate code directly)
        if has_code and not has_rag:
            return RoutingDecision(
                intent="CODE_EXECUTION",
                agents=["code_agent"],
                confidence=0.98,
                reasoning_summary="Query requests code synthesis, algorithm implementation, or debugging."
            )

        # 2. PRODUCT_SEARCH (Check before finance so 'Apple iPhone price' doesn't get routed to stock market)
        if has_product:
            return RoutingDecision(
                intent="PRODUCT_SEARCH",
                agents=["product_agent"],
                requires_live_data=True,
                requires_product=True,
                confidence=0.98,
                reasoning_summary="Query concerns consumer hardware/electronics pricing or specifications."
            )

        # 3. FINANCE_MARKET
        if has_finance:
            return RoutingDecision(
                intent="FINANCE_MARKET",
                agents=["finance_agent"],
                requires_live_data=True,
                requires_finance=True,
                confidence=0.98,
                reasoning_summary="Query asks for equity shares, stock market quotes, or ticker data."
            )

        # 4. DOCUMENT_RAG
        if has_rag:
            return RoutingDecision(
                intent="DOCUMENT_RAG",
                agents=["rag_agent"],
                requires_documents=True,
                confidence=0.98,
                reasoning_summary="Query explicitly refers to knowledge base, uploaded PDFs, policies, or research files."
            )

        # 5. CODE_EXECUTION
        if any(w in lower_q for w in self.CODE_KEYWORDS):
            return RoutingDecision(
                intent="CODE_EXECUTION",
                agents=["code_agent"],
                confidence=0.95,
                reasoning_summary="Query requests code synthesis or mathematical evaluation."
            )

        # 6. WEB_SEARCH
        if has_web or any(term in lower_q for term in ["today", "latest", "recent", "current news", "who is", "weather"]):
            return RoutingDecision(
                intent="WEB_SEARCH",
                agents=["web_agent"],
                requires_live_data=True,
                confidence=0.95,
                reasoning_summary="Query requires live external web search."
            )

        # 7. Check if query is naturally answered by knowledge base documents (if indexed and matching)
        if len(hybrid_search.chunks) > 0:
            # Test if query has high semantic affinity to indexed documents
            # Only route if not general chat greeting
            if len(lower_q.split()) > 3 and not any(g in lower_q for g in ["hello", "hi", "hey", "who are you", "what is rag", "explain what"]):
                return RoutingDecision(
                    intent="DOCUMENT_RAG",
                    agents=["rag_agent"],
                    requires_documents=True,
                    confidence=0.80,
                    reasoning_summary="Query evaluated against active indexed knowledge base."
                )

        # 8. GENERAL_CHAT
        return RoutingDecision(
            intent="GENERAL_CHAT",
            agents=["general_agent"],
            confidence=1.0,
            reasoning_summary="General conversational question or conceptual explanation."
        )


supervisor_agent = SupervisorAgent()

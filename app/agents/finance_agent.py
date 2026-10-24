from typing import Dict, Any, List
from datetime import datetime
from app.tools.finance_tool import finance_tool
from app.services.llm import llm_service
from app.models.schemas import SourceCitation
from app.utils.logging import logger


class FinanceAgent:
    """
    Finance & Stock Market Intelligence Agent.
    Strictly handles public company equities, tickers, market capitalization, and financial quotes.
    Never handles consumer electronic devices or retail product prices.
    """

    async def execute(self, query: str) -> Dict[str, Any]:
        logger.info(f"FinanceAgent executing financial query: {query}")
        
        quotes = await finance_tool.get_quotes_for_query(query)
        
        sources: List[SourceCitation] = []
        snippets = []

        for q in quotes:
            sources.append(SourceCitation(
                document=f"{q['symbol']} - Yahoo Finance",
                source_type="finance",
                snippet=q.get("snippet", ""),
                url=q.get("url"),
                score=1.0,
                retrieved_at=q.get("retrieved_at", datetime.utcnow().isoformat())
            ))
            snippets.append(q.get("snippet", ""))

        if not quotes:
            # Fallback direct lookup if ticker was mentioned
            resolved = finance_tool.resolve_ticker(query)
            if resolved:
                quote = finance_tool.get_stock_quote(resolved)
                if quote:
                    curr = quote['currency']
                    quote_str = f"Share price for {quote['symbol']} on {quote['exchange']} is {quote['price']} {curr} (Previous close: {quote.get('previous_close')} {curr})."
                    snippets.append(quote_str)
                    sources.append(SourceCitation(
                        document=f"{quote['symbol']} - Yahoo Finance",
                        source_type="finance",
                        snippet=quote_str,
                        url=f"https://finance.yahoo.com/quote/{quote['symbol']}",
                        score=1.0,
                        retrieved_at=datetime.utcnow().isoformat()
                    ))

        evidence_text = "\n\n".join(snippets) if snippets else "No active financial ticker quotes found for the requested company."

        system_prompt = (
            "You are an expert Financial Markets & Equities Intelligence Agent.\n"
            "Your role is to report verified financial market data, stock quotes, market caps, and exchange information.\n"
            "STRICT RULES:\n"
            "1. ONLY answer with the verified financial data provided below.\n"
            "2. Always specify the exact ticker symbol, trading exchange, currency, and quote price.\n"
            "3. If multiple companies are mentioned (e.g. Apple and Samsung), provide a clean side-by-side or bulleted comparison.\n"
            "4. NEVER confuse stock prices with product retail prices.\n"
            "5. If financial data for a requested company or ticker cannot be resolved, state that clearly."
        )

        user_content = f"User Financial Inquiry:\n{query}\n\nVerified Financial Market Evidence:\n{evidence_text}\n\nPlease generate a clear financial response."

        messages = [{"role": "user", "content": user_content}]
        draft_answer = await llm_service.chat(
            messages=messages,
            system=system_prompt,
            temperature=0.0
        )

        return {
            "draft_answer": draft_answer,
            "sources": sources,
            "agent": "finance_agent",
            "confidence": 0.98 if sources else 0.5
        }


finance_agent = FinanceAgent()

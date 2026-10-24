import re
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from app.tools.search_tool import search_tool
from app.services.llm import llm_service
from app.models.schemas import SourceCitation
from app.utils.logging import logger


class ProductAgent:
    """
    Consumer Electronics, Latest Devices, & Product Search Agent.
    Specializes in:
    - Identifying the newest confirmed phone/device releases.
    - Inspecting explicit product names, announcement dates, and availability dates.
    - Distinguishing the newest model overall vs within a particular series (e.g. S-series vs Foldables).
    - Verifying official manufacturer sources, newsrooms, and official product pages.
    - Never confusing a search failure with a fact being unavailable.
    - Grounding answers without substituting stock prices or inventing unannounced models.
    """

    @staticmethod
    def _is_latest_device_query(query: str) -> bool:
        lower = query.lower()
        patterns = [
            "latest", "newest", "current release", "most recent", "new release",
            "recently launched", "latest mobile", "latest phone", "latest device",
            "new phone", "new mobile"
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _extract_potential_products(query: str) -> List[str]:
        # Split on conjunctions to research each product separately if comparing
        parts = re.split(r'\band\b|\bvs\b|\bcompare\b|,', query, flags=re.IGNORECASE)
        products = []
        for p in parts:
            cleaned = p.strip()
            cleaned = re.sub(
                r'^(what is|what are|the|current|price of|how much is|how much does|cost of|specs of|tell me about)\s+',
                '', cleaned, flags=re.IGNORECASE
            ).strip()
            if len(cleaned) > 2:
                products.append(cleaned)
        return products if products else [query]

    async def execute(self, query: str) -> Dict[str, Any]:
        logger.info(f"ProductAgent executing query: {query}")
        sources: List[SourceCitation] = []
        evidence_sections = []
        is_latest_query = self._is_latest_device_query(query)

        if is_latest_query:
            # Broad / Latest device query: run search queries in parallel
            search_queries = [
                f"{query} announcement release date official",
                f"{query} flagship smartphones newsroom"
            ]
            search_tasks = [search_tool.search_web(sq, max_results=3) for sq in search_queries]
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            all_results = []
            for res in search_results_list:
                if isinstance(res, list):
                    all_results.extend(res)

            # Deduplicate by url/title
            seen_urls = set()
            snippets = []
            for r in all_results:
                url = r.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                sources.append(SourceCitation(
                    document=r.get("title", "Official Product Research"),
                    source_type="product",
                    snippet=r.get("snippet", ""),
                    url=url,
                    score=0.95,
                    retrieved_at=datetime.utcnow().isoformat()
                ))
                snippets.append(f"Title: {r.get('title')}\nSource: {r.get('source')}\nURL: {url}\nDetails: {r.get('snippet')}\n")

            evidence_sections.append(f"### Web Search & Release Evidence:\n" + "\n---\n".join(snippets))
        else:
            # Targeted product inquiries: run all product searches in parallel
            products = self._extract_potential_products(query)[:3]
            search_tasks = [
                search_tool.search_web(f"{prod} official release price specs announcement", max_results=4)
                for prod in products
            ]
            results_per_product = await asyncio.gather(*search_tasks, return_exceptions=True)

            for prod, results in zip(products, results_per_product):
                snippets = []
                if isinstance(results, list):
                    for r in results:
                        sources.append(SourceCitation(
                            document=r.get("title", "Product Research"),
                            source_type="product",
                            snippet=r.get("snippet", ""),
                            url=r.get("url"),
                            score=0.90,
                            retrieved_at=datetime.utcnow().isoformat()
                        ))
                        snippets.append(f"- [{r.get('title')}] {r.get('snippet')} (URL: {r.get('url')})")

                section_text = "\n".join(snippets) if snippets else "No official product release or pricing listings found."
                evidence_sections.append(f"### Target Product: {prod}\n{section_text}")

        combined_evidence = "\n\n".join(evidence_sections)

        system_prompt = (
            "You are a Senior Consumer Electronics & Product Intelligence Agent.\n"
            "Your role is to synthesize verified web and release evidence to answer questions about the latest products, newest devices, pricing, and releases.\n\n"
            "MANDATORY GUIDELINES:\n"
            "1. SYNTHESIZE THE EVIDENCE: Extract explicit product names, announcement dates, and release dates from the evidence. Do NOT merely repeat that snippets lack information if confirmed facts are present.\n"
            "2. DISTINGUISH DATES: Clearly distinguish the announcement date (when it was unveiled) from the availability/shipping release date.\n"
            "3. DISTINGUISH PRODUCT SERIES: When identifying the latest device, clarify whether it is the newest flagship overall (e.g. Galaxy S series) or newest within another category (e.g. Galaxy Z Fold/Flip series or Galaxy A series).\n"
            "4. BROAD QUESTIONS (e.g. 'What is Samsung's latest mobile device?'): Identify the newest confirmed flagship phone release and briefly list other recent key lineup releases (e.g. foldables) if helpful.\n"
            "5. VERIFY EXISTENCE vs SPECULATION: If a product does not exist or has never been announced (e.g. 'iPhone 18 Pro Max' or 'iPhone Duo'), state clearly that it is unannounced or fictional, and note the current actual generation.\n"
            "6. NEVER CONFUSE STOCKS WITH PRODUCTS: Never substitute company stock prices (AAPL) for product prices.\n"
            "7. HONEST LIMITATIONS: Do not confuse a search failure with a fact being unavailable. If sources disagree or details are pending, explain the discrepancy honestly.\n"
            "8. DIRECT CITATIONS: Reference the verified source links and maintain clear markdown formatting."
        )

        user_content = f"User Query:\n{query}\n\nVerified Evidence & Product News Sources:\n{combined_evidence}\n\nPlease generate a thorough, grounded synthesis."

        messages = [{"role": "user", "content": user_content}]
        draft_answer = await llm_service.chat(
            messages=messages,
            system=system_prompt,
            temperature=0.0
        )

        return {
            "draft_answer": draft_answer,
            "sources": sources,
            "agent": "product_agent",
            "confidence": 0.95 if sources else 0.5
        }


product_agent = ProductAgent()

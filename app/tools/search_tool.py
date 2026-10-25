"""
Real-Time Web Search and Market Intelligence Tooling.

This module provides data retrieval capabilities for multi-agent workflows,
including:
- Real-time equity market data via Yahoo Finance (`yfinance`).
- Web and breaking news intelligence via DuckDuckGo Search (`duckduckgo_search`).
- Factual grounding and encyclopedic summaries via Wikipedia REST and MediaWiki APIs.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional

import httpx

from app.utils.logging import logger

try:
    import yfinance as yf
except ImportError:
    yf = None


class SearchTool:
    """
    Search and real-time data retrieval service.

    Coordinates multi-provider search requests across market data endpoints,
    search engines, and encyclopedic knowledge graphs with built-in deduplication
    and error resilience.
    """

    @staticmethod
    def get_stock_quote(symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real-time financial market quotes for a given equity ticker symbol.

        Utilizes Yahoo Finance FastInfo to retrieve current trading price, currency,
        prior close, and total market capitalization.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA').

        Returns:
            A dictionary containing parsed market data fields, or None if the symbol
            is invalid, unreachable, or the yfinance dependency is unavailable.
        """
        if not yf:
            return None
        try:
            ticker = yf.Ticker(symbol.strip().upper())
            fast = ticker.fast_info
            price = fast.get("lastPrice") or fast.get("regularMarketPrice") or fast.get("last_price")
            currency = fast.get("currency", "USD")
            prev_close = fast.get("previousClose") or fast.get("previous_close")
            market_cap = fast.get("marketCap")

            if price:
                return {
                    "symbol": symbol.upper(),
                    "price": round(float(price), 2),
                    "currency": currency,
                    "previous_close": round(float(prev_close), 2) if prev_close else None,
                    "market_cap": market_cap,
                    "source": "Yahoo Finance Real-Time Market Data",
                }
        except Exception as e:
            logger.warning(f"Error fetching stock quote for {symbol}: {e}")
        return None

    @staticmethod
    async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute an asynchronous multi-source web search to retrieve grounded evidence.

        Implements a two-phase retrieval pipeline:
        1. Primary Search Provider: DuckDuckGo News and Text search executed off-thread
           to gather breaking news and organic search snippets without blocking the event loop.
        2. Knowledge Grounding Fallback: Wikipedia MediaWiki & REST APIs to acquire authoritative
           factual extracts and canonical reference URLs.

        Args:
            query: Natural language or keyword query string.
            max_results: Target number of results per provider stream.

        Returns:
            A deduplicated list of search result dictionaries containing title,
            snippet, url, and source attribution.
        """
        results: List[Dict[str, Any]] = []
        seen_urls = set()

        def add_result(title: str, snippet: str, url: str, source: str) -> None:
            """
            Sanitize and deduplicate incoming search result records.

            Enforces snippet length thresholds to discard low-information noise and ensures
            unique URL tracking across diverse provider responses.
            """
            if url and url in seen_urls:
                return
            if snippet and len(snippet.strip()) > 20:
                if url:
                    seen_urls.add(url)
                results.append({
                    "title": title or "Web Result",
                    "snippet": snippet.strip(),
                    "url": url or "",
                    "source": source,
                })

        # Phase 1: DuckDuckGo text and news retrieval
        # Offloads synchronous network I/O from the DuckDuckGo client into a dedicated
        # thread pool to ensure non-blocking event loop execution.
        try:
            from duckduckgo_search import DDGS

            def run_ddg() -> List[Dict[str, Any]]:
                """
                Query DuckDuckGo for chronological news and organic web rankings.
                Prioritizes recent news items for timely events, followed by web results.
                """
                out: List[Dict[str, Any]] = []
                try:
                    with DDGS() as ddgs:
                        try:
                            news_items = list(ddgs.news(query, max_results=max_results))
                            for n in news_items:
                                out.append({
                                    "title": n.get("title", ""),
                                    "snippet": f"[{n.get('date', '')[:10]}] {n.get('body', '')}",
                                    "url": n.get("url", ""),
                                    "source": n.get("source", "DuckDuckGo News"),
                                })
                        except Exception:
                            pass

                        try:
                            text_items = list(ddgs.text(query, max_results=max_results))
                            for t in text_items:
                                out.append({
                                    "title": t.get("title", ""),
                                    "snippet": t.get("body", ""),
                                    "url": t.get("href", ""),
                                    "source": "DuckDuckGo Web",
                                })
                        except Exception:
                            pass
                except Exception as ex:
                    logger.warning(f"DuckDuckGo search error: {ex}")
                return out

            ddg_results = await asyncio.to_thread(run_ddg)
            for item in ddg_results:
                add_result(item["title"], item["snippet"], item["url"], item["source"])
        except Exception as e:
            logger.warning(f"DDG search tool failed: {e}")

        # Phase 2: Wikipedia MediaWiki search and lead summary extraction
        # Queries the MediaWiki action API to identify relevant topics, then leverages
        # the REST v1 page summary endpoint for canonical, citation-grade lead context.
        try:
            wiki_headers = {"User-Agent": "MultiAgentResearchBot/2.0 (research@multiagent.internal)"}
            async with httpx.AsyncClient(headers=wiki_headers, timeout=5.0) as client:
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3,
                }
                resp = await client.get(search_url, params=params)
                if resp.status_code == 200:
                    wiki_data = resp.json()
                    search_items = wiki_data.get("query", {}).get("search", [])
                    for item in search_items:
                        title = item.get("title")
                        clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                        try:
                            sum_resp = await client.get(
                                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                            )
                            if sum_resp.status_code == 200:
                                extract = sum_resp.json().get("extract", "")
                                if extract:
                                    add_result(title, extract, page_url, "Wikipedia Official Release Summary")
                                    continue
                        except Exception:
                            pass

                        add_result(title, clean_snippet, page_url, "Wikipedia")
        except Exception as e:
            logger.debug(f"Wikipedia search skipped: {e}")

        return results[:max_results * 2]


search_tool = SearchTool()

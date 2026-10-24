import re
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from app.utils.logging import logger

try:
    import yfinance as yf
except ImportError:
    yf = None


class SearchTool:
    """
    Search & Real-Time Data Tool.
    Retrieves live web data, financial quotes, and factual summaries.
    """

    @staticmethod
    def get_stock_quote(symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch live financial stock quotes via Yahoo Finance."""
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
                    "source": "Yahoo Finance Real-Time Market Data"
                }
        except Exception as e:
            logger.warning(f"Error fetching stock quote for {symbol}: {e}")
        return None

    @staticmethod
    async def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for real-time information, articles, news, product specs, or facts.
        Combines DuckDuckGo News & Text with Wikipedia Search & Extract fallbacks.
        """
        results = []
        seen_urls = set()

        def add_result(title: str, snippet: str, url: str, source: str):
            if url and url in seen_urls:
                return
            if snippet and len(snippet.strip()) > 20:
                if url:
                    seen_urls.add(url)
                results.append({
                    "title": title or "Web Result",
                    "snippet": snippet.strip(),
                    "url": url or "",
                    "source": source
                })

        # 1. DuckDuckGo Text & News Search
        try:
            from duckduckgo_search import DDGS
            def run_ddg():
                out = []
                try:
                    with DDGS() as ddgs:
                        # Try news first for latest announcements/products
                        try:
                            news_items = list(ddgs.news(query, max_results=max_results))
                            for n in news_items:
                                out.append({
                                    "title": n.get("title", ""),
                                    "snippet": f"[{n.get('date', '')[:10]}] {n.get('body', '')}",
                                    "url": n.get("url", ""),
                                    "source": n.get("source", "DuckDuckGo News")
                                })
                        except Exception:
                            pass

                        # If text search has results, include them
                        try:
                            text_items = list(ddgs.text(query, max_results=max_results))
                            for t in text_items:
                                out.append({
                                    "title": t.get("title", ""),
                                    "snippet": t.get("body", ""),
                                    "url": t.get("href", ""),
                                    "source": "DuckDuckGo Web"
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

        # 2. Wikipedia Structured Search & Extract (for accurate announcement and release dates)
        try:
            wiki_headers = {'User-Agent': 'MultiAgentResearchBot/2.0 (research@multiagent.internal)'}
            async with httpx.AsyncClient(headers=wiki_headers, timeout=5.0) as client:
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3
                }
                resp = await client.get(search_url, params=params)
                if resp.status_code == 200:
                    wiki_data = resp.json()
                    search_items = wiki_data.get("query", {}).get("search", [])
                    for item in search_items:
                        title = item.get("title")
                        clean_snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        
                        # Fetch full lead summary extract for the top page
                        try:
                            sum_resp = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
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

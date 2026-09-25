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
        Search the web for real-time information, financial prices, or current events.
        Includes dedicated real-time financial ticker resolution.
        """
        results = []
        lower_query = query.lower()

        # 1. Check for stock queries (e.g. Apple, Samsung, Microsoft, Google, Tesla, Nvidia)
        stock_mappings = {
            "apple": "AAPL",
            "aapl": "AAPL",
            "samsung": "005930.KS",
            "samsung electronics": "005930.KS",
            "microsoft": "MSFT",
            "msft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "amzn": "AMZN",
            "tesla": "TSLA",
            "tsla": "TSLA",
            "nvidia": "NVDA",
            "nvda": "NVDA",
            "meta": "META",
            "facebook": "META"
        }

        # If query asks for stock / share price
        if any(term in lower_query for term in ["stock", "price", "shares", "market cap", "quote", "ticker"]):
            for name, ticker in stock_mappings.items():
                if name in lower_query:
                    # Run ticker lookup in threadpool
                    quote = await asyncio.to_thread(SearchTool.get_stock_quote, ticker)
                    if quote:
                        display_name = name.title()
                        curr = quote['currency']
                        price_str = f"{quote['price']} {curr}"
                        prev_str = f", Previous Close: {quote['previous_close']} {curr}" if quote.get('previous_close') else ""
                        results.append({
                            "title": f"Live Market Data: {display_name} ({quote['symbol']})",
                            "snippet": f"Real-time market price for {display_name} ({quote['symbol']}) is {price_str}{prev_str}. (Source: {quote['source']})",
                            "url": f"https://finance.yahoo.com/quote/{quote['symbol']}",
                            "source": "Yahoo Finance (Live Market)"
                        })

        # 2. General Web Search via DuckDuckGo / Wikipedia fallback
        try:
            from duckduckgo_search import DDGS
            def run_ddg():
                try:
                    with DDGS() as ddgs:
                        return list(ddgs.text(query, max_results=max_results))
                except Exception as ex:
                    logger.warning(f"DuckDuckGo search error: {ex}")
                    return []

            ddg_results = await asyncio.to_thread(run_ddg)
            for item in ddg_results:
                results.append({
                    "title": item.get("title", "Web Result"),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                    "source": item.get("href", "Web Search")
                })
        except Exception as e:
            logger.warning(f"DDG search tool failed: {e}")

        # 3. Wikipedia Summary API fallback for general factual inquiries
        if len(results) < 2:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    # Extract search keywords
                    words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', '', query).split() if len(w) > 3]
                    keyword = words[0] if words else query
                    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{keyword}"
                    resp = await client.get(wiki_url)
                    if resp.status_code == 200:
                        wiki_data = resp.json()
                        if "extract" in wiki_data:
                            results.append({
                                "title": wiki_data.get("title", keyword),
                                "snippet": wiki_data.get("extract", ""),
                                "url": wiki_data.get("content_urls", {}).get("desktop", {}).get("page", wiki_url),
                                "source": "Wikipedia"
                            })
            except Exception as e:
                logger.debug(f"Wikipedia fallback skipped: {e}")

        return results


search_tool = SearchTool()

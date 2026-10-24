import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.utils.logging import logger

try:
    import yfinance as yf
except ImportError:
    yf = None


class FinanceTool:
    """
    Dedicated Financial Market Data Tool.
    Exclusively resolves public company stocks, tickers, ETFs, and market indices.
    Never returns consumer product prices.
    """

    COMPANY_TO_TICKER = {
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

    @classmethod
    def resolve_ticker(cls, company_or_symbol: str) -> Optional[str]:
        cleaned = company_or_symbol.strip().lower()
        if cleaned in cls.COMPANY_TO_TICKER:
            return cls.COMPANY_TO_TICKER[cleaned]
        for name, ticker in cls.COMPANY_TO_TICKER.items():
            if name in cleaned:
                return ticker
        if len(company_or_symbol.strip()) <= 5 and company_or_symbol.strip().isalpha():
            return company_or_symbol.strip().upper()
        return None

    @classmethod
    def get_stock_quote(cls, symbol: str) -> Optional[Dict[str, Any]]:
        if not yf:
            logger.warning("yfinance is not installed.")
            return None
        try:
            ticker_obj = yf.Ticker(symbol.strip().upper())
            fast = ticker_obj.fast_info
            price = fast.get("lastPrice") or fast.get("regularMarketPrice") or fast.get("last_price")
            currency = fast.get("currency", "USD")
            prev_close = fast.get("previousClose") or fast.get("previous_close")
            market_cap = fast.get("marketCap")
            exchange = fast.get("exchange", "Unknown Exchange")
            
            if price is not None:
                return {
                    "symbol": symbol.upper(),
                    "price": round(float(price), 2),
                    "currency": currency,
                    "previous_close": round(float(prev_close), 2) if prev_close else None,
                    "market_cap": market_cap,
                    "exchange": exchange,
                    "quote_timestamp": datetime.utcnow().isoformat(),
                    "source": "Yahoo Finance (Live Market)"
                }
        except Exception as e:
            logger.warning(f"Error fetching stock quote for {symbol}: {e}")
        return None

    @classmethod
    async def get_quotes_for_query(cls, query: str) -> List[Dict[str, Any]]:
        lower = query.lower()
        results = []
        found_tickers = set()

        for name, ticker in cls.COMPANY_TO_TICKER.items():
            if name in lower and ticker not in found_tickers:
                found_tickers.add(ticker)
                quote = await asyncio.to_thread(cls.get_stock_quote, ticker)
                if quote:
                    curr = quote['currency']
                    prev = f", Previous Close: {quote['previous_close']} {curr}" if quote.get('previous_close') else ""
                    mcap = f", Market Cap: {quote['market_cap']:,}" if quote.get('market_cap') else ""
                    results.append({
                        "title": f"Financial Market Quote: {name.title()} ({quote['symbol']})",
                        "symbol": quote["symbol"],
                        "price": quote["price"],
                        "currency": quote["currency"],
                        "exchange": quote["exchange"],
                        "snippet": f"Real-time share price for {name.title()} ({quote['symbol']} on {quote['exchange']}) is {quote['price']} {curr}{prev}{mcap}. (Source: {quote['source']})",
                        "url": f"https://finance.yahoo.com/quote/{quote['symbol']}",
                        "source": "Yahoo Finance",
                        "source_type": "finance",
                        "retrieved_at": quote["quote_timestamp"]
                    })
        return results


finance_tool = FinanceTool()

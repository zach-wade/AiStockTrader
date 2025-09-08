"""
Polygon Data Provider

Migrates and adapts the old Polygon client functionality to the new architecture.
Provides market data, news, and fundamentals from Polygon.io API.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from httpx import AsyncClient

from src.application.interfaces.market_data import (
    Bar,
    IMarketDataProvider,
    MarketDataError,
    PriceUpdate,
    Quote,
    RateLimitError,
)
from src.domain.value_objects import Price

logger = logging.getLogger(__name__)


class PolygonDataProvider(IMarketDataProvider):
    """
    Polygon.io data provider implementation.

    Provides access to market data, news, fundamentals, and corporate actions
    through the Polygon.io API with rate limiting and error handling.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Polygon data provider.

        Args:
            api_key: Polygon API key. If None, reads from POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key is required")

        self.base_url = "https://api.polygon.io"
        self._client: AsyncClient | None = None
        self._rate_limiter = RateLimiter(5, 1)  # 5 requests per second

    async def connect(self) -> None:
        """Establish connection to Polygon API."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0), headers={"Authorization": f"Bearer {self.api_key}"}
            )
        logger.info("Connected to Polygon.io API")

    async def disconnect(self) -> None:
        """Close connection to Polygon API."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Disconnected from Polygon.io API")

    async def get_current_price(self, symbol: str) -> Price:
        """
        Get the latest price for a symbol.

        Args:
            symbol: The symbol to get price for

        Returns:
            Latest price

        Raises:
            MarketDataError: If price fetch fails
        """
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            if data.get("results"):
                price_value = data["results"]["p"]
                return Price(Decimal(str(price_value)))
            else:
                raise MarketDataError(f"No price data for {symbol}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Polygon API rate limit exceeded")
            raise MarketDataError(f"Failed to get price: {e}")
        except Exception as e:
            raise MarketDataError(f"Failed to get price: {e}")

    async def get_current_quote(self, symbol: str) -> Quote:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: The symbol to get quote for

        Returns:
            Latest quote with bid/ask
        """
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            url = f"{self.base_url}/v2/last/nbbo/{symbol}"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            if data.get("results"):
                result = data["results"]
                from src.domain.value_objects import Symbol

                return Quote(
                    symbol=Symbol(symbol),
                    bid_price=Price(Decimal(str(result["P"]))),
                    ask_price=Price(Decimal(str(result["S"]))),
                    bid_size=result.get("s", 0),
                    ask_size=result.get("z", 0),
                    timestamp=datetime.fromtimestamp(result["t"] / 1000),
                )
            else:
                raise MarketDataError(f"No quote data for {symbol}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Polygon API rate limit exceeded")
            raise MarketDataError(f"Failed to get quote: {e}")
        except Exception as e:
            raise MarketDataError(f"Failed to get quote: {e}")

    async def get_historical_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[Bar]:
        """
        Get historical price bars for a symbol.

        Args:
            symbol: The symbol to get bars for
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            List of price bars
        """
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        # Convert timeframe to Polygon format
        timeframe_map = {
            "1Min": "1/minute",
            "5Min": "5/minute",
            "15Min": "15/minute",
            "1Hour": "1/hour",
            "1Day": "1/day",
        }

        polygon_timeframe = timeframe_map.get(timeframe, "1/day")
        multiplier, timespan = polygon_timeframe.split("/")

        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000}

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            bars = []

            if data.get("results"):
                for bar_data in data["results"]:
                    bar = Bar(
                        symbol=symbol,
                        open=Price(Decimal(str(bar_data["o"]))),
                        high=Price(Decimal(str(bar_data["h"]))),
                        low=Price(Decimal(str(bar_data["l"]))),
                        close=Price(Decimal(str(bar_data["c"]))),
                        volume=bar_data["v"],
                        timestamp=datetime.fromtimestamp(bar_data["t"] / 1000),
                    )
                    bars.append(bar)

            return bars

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Polygon API rate limit exceeded")
            raise MarketDataError(f"Failed to get historical bars: {e}")
        except Exception as e:
            raise MarketDataError(f"Failed to get historical bars: {e}")

    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[PriceUpdate]:
        """Stream real-time price updates."""
        raise NotImplementedError("Real-time streaming not yet implemented")
        # This would require WebSocket implementation
        yield  # type: ignore

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """Stream real-time quote updates."""
        raise NotImplementedError("Real-time streaming not yet implemented")
        # This would require WebSocket implementation
        yield  # type: ignore

    async def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        await self._ensure_connected()

        try:
            url = f"{self.base_url}/v1/marketstatus/now"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            return data.get("market", "closed") == "open"
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            return data.get("results", {})
        except Exception as e:
            raise MarketDataError(f"Failed to get symbol info: {e}")

    async def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._client:
            await self.connect()

    # Additional Polygon-specific methods

    async def get_news(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get news articles from Polygon.

        Args:
            symbol: Optional symbol to filter news
            limit: Maximum number of articles to return

        Returns:
            List of news articles
        """
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            url = f"{self.base_url}/v2/reference/news"
            params = {"limit": limit}
            if symbol:
                params["ticker"] = symbol

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data.get("results", [])

        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return []

    async def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: The symbol to get fundamentals for

        Returns:
            Dictionary of fundamental data
        """
        await self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            return data.get("results", {})

        except Exception as e:
            logger.error(f"Failed to get fundamentals: {e}")
            return {}


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old requests outside the window
            self.requests = [
                req_time for req_time in self.requests if now - req_time < self.window_seconds
            ]

            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.window_seconds - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Remove the expired request
                    self.requests.pop(0)

            # Record this request
            self.requests.append(now)

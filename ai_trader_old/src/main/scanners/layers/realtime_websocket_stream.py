# File: src/main/scanners/layers/realtime_websocket_stream.py
"""
WebSocket-based Real-time Data Streaming for Hunter-Killer Scanner

Provides sub-second market data updates for rapid opportunity detection.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from typing import Any

# Third-party imports
import numpy as np
import websockets

logger = logging.getLogger(__name__)


@dataclass
class RealtimeQuote:
    """Real-time quote data."""

    symbol: str
    timestamp: datetime
    price: float
    size: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: float
    conditions: list[str] = field(default_factory=list)

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.price > 0:
            return (self.spread / self.price) * 10000
        return 0.0


@dataclass
class RealtimeTrade:
    """Real-time trade data."""

    symbol: str
    timestamp: datetime
    price: float
    size: int
    conditions: list[str] = field(default_factory=list)
    exchange: str = ""


class WebSocketDataStream:
    """
    High-performance WebSocket data streaming for real-time market data.

    Supports multiple data providers:
    - Alpaca WebSocket API
    - Polygon.io WebSocket API (if available)
    - IEX Cloud streaming (backup)
    """

    def __init__(
        self,
        provider: str = "alpaca",
        api_key: str = None,
        api_secret: str = None,
        feed: str = "iex",
    ):
        """
        Initialize WebSocket stream.

        Args:
            provider: Data provider ("alpaca", "polygon", "iex")
            api_key: API key for authentication
            api_secret: API secret for authentication
            feed: Data feed type for Alpaca ("iex" or "sip")
        """
        self.provider = provider
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed

        # WebSocket connections
        self.ws_conn = None
        self.authenticated = False

        # Subscriptions
        self.quote_symbols: set[str] = set()
        self.trade_symbols: set[str] = set()

        # Callbacks
        self.quote_callbacks: list[Callable[[RealtimeQuote], None]] = []
        self.trade_callbacks: list[Callable[[RealtimeTrade], None]] = []

        # Data buffers for aggregation
        self.quote_buffer: dict[str, list[RealtimeQuote]] = defaultdict(list)
        self.trade_buffer: dict[str, list[RealtimeTrade]] = defaultdict(list)

        # Performance metrics
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = None

        # Connection URLs
        self.ws_urls = {
            "alpaca": f"wss://stream.data.alpaca.markets/v2/{feed}",
            "polygon": "wss://socket.polygon.io/stocks",
            "iex": "wss://cloud-sse.iexapis.com/stable/stocksUS",
        }

        logger.info(f"WebSocket stream initialized for {provider} ({feed} feed)")

    async def connect(self):
        """Connect to WebSocket stream."""
        url = self.ws_urls.get(self.provider)
        if not url:
            raise ValueError(f"Unknown provider: {self.provider}")

        try:
            logger.info(f"Connecting to {self.provider} WebSocket...")

            if self.provider == "alpaca":
                await self._connect_alpaca(url)
            elif self.provider == "polygon":
                await self._connect_polygon(url)
            elif self.provider == "iex":
                await self._connect_iex(url)

            logger.info(f"Connected to {self.provider} WebSocket")

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def _connect_alpaca(self, url: str):
        """Connect to Alpaca WebSocket."""
        self.ws_conn = await websockets.connect(url)

        # Authenticate
        auth_data = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
        await self.ws_conn.send(json.dumps(auth_data))

        # Wait for auth response
        response = await self.ws_conn.recv()
        msg = json.loads(response)

        if msg[0]["T"] == "success" and msg[0]["msg"] == "authenticated":
            self.authenticated = True
            logger.info("Alpaca WebSocket authenticated")
        else:
            raise Exception(f"Authentication failed: {msg}")

    async def _connect_polygon(self, url: str):
        """Connect to Polygon WebSocket."""
        # Polygon uses token-based auth in URL
        auth_url = f"{url}?apikey={self.api_key}"
        self.ws_conn = await websockets.connect(auth_url)
        self.authenticated = True
        logger.info("Polygon WebSocket connected")

    async def _connect_iex(self, url: str):
        """Connect to IEX Cloud WebSocket."""
        # IEX uses token in URL
        auth_url = f"{url}?token={self.api_key}"
        self.ws_conn = await websockets.connect(auth_url)
        self.authenticated = True
        logger.info("IEX WebSocket connected")

    async def subscribe_quotes(self, symbols: list[str]):
        """Subscribe to real-time quotes."""
        if not self.authenticated:
            raise Exception("Not authenticated")

        self.quote_symbols.update(symbols)

        if self.provider == "alpaca":
            sub_data = {"action": "subscribe", "quotes": symbols}
            await self.ws_conn.send(json.dumps(sub_data))

        elif self.provider == "polygon":
            for symbol in symbols:
                await self.ws_conn.send(
                    json.dumps({"action": "subscribe", "params": f"Q.{symbol}"})
                )

        logger.info(f"Subscribed to quotes for {len(symbols)} symbols")

    async def subscribe_trades(self, symbols: list[str]):
        """Subscribe to real-time trades."""
        if not self.authenticated:
            raise Exception("Not authenticated")

        self.trade_symbols.update(symbols)

        if self.provider == "alpaca":
            sub_data = {"action": "subscribe", "trades": symbols}
            await self.ws_conn.send(json.dumps(sub_data))

        elif self.provider == "polygon":
            for symbol in symbols:
                await self.ws_conn.send(
                    json.dumps({"action": "subscribe", "params": f"T.{symbol}"})
                )

        logger.info(f"Subscribed to trades for {len(symbols)} symbols")

    async def unsubscribe_quotes(self, symbols: list[str]):
        """Unsubscribe from quotes."""
        self.quote_symbols.difference_update(symbols)

        if self.provider == "alpaca":
            unsub_data = {"action": "unsubscribe", "quotes": symbols}
            await self.ws_conn.send(json.dumps(unsub_data))

    async def unsubscribe_trades(self, symbols: list[str]):
        """Unsubscribe from trades."""
        self.trade_symbols.difference_update(symbols)

        if self.provider == "alpaca":
            unsub_data = {"action": "unsubscribe", "trades": symbols}
            await self.ws_conn.send(json.dumps(unsub_data))

    def add_quote_callback(self, callback: Callable[[RealtimeQuote], None]):
        """Add callback for quote updates."""
        self.quote_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable[[RealtimeTrade], None]):
        """Add callback for trade updates."""
        self.trade_callbacks.append(callback)

    async def stream(self):
        """Main streaming loop."""
        if not self.ws_conn:
            await self.connect()

        try:
            while True:
                message = await self.ws_conn.recv()
                await self._process_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._reconnect()
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            self.error_count += 1
            await self._reconnect()

    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            self.message_count += 1
            self.last_message_time = datetime.now(UTC)

            if self.provider == "alpaca":
                await self._process_alpaca_message(message)
            elif self.provider == "polygon":
                await self._process_polygon_message(message)
            elif self.provider == "iex":
                await self._process_iex_message(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1

    async def _process_alpaca_message(self, message: str):
        """Process Alpaca WebSocket message."""
        data = json.loads(message)

        for msg in data:
            msg_type = msg.get("T")

            if msg_type == "q":  # Quote
                quote = RealtimeQuote(
                    symbol=msg["S"],
                    timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                    price=(msg["bp"] + msg["ap"]) / 2,  # Mid price
                    size=0,
                    bid=msg["bp"],
                    ask=msg["ap"],
                    bid_size=msg["bs"],
                    ask_size=msg["as"],
                    volume=0,  # Not in quote message
                    conditions=msg.get("c", []),
                )

                # Buffer and notify
                self.quote_buffer[quote.symbol].append(quote)
                for callback in self.quote_callbacks:
                    try:
                        callback(quote)
                    except Exception as e:
                        logger.error(f"Error in quote callback: {e}")

            elif msg_type == "t":  # Trade
                trade = RealtimeTrade(
                    symbol=msg["S"],
                    timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                    price=msg["p"],
                    size=msg["s"],
                    conditions=msg.get("c", []),
                    exchange=msg.get("x", ""),
                )

                # Buffer and notify
                self.trade_buffer[trade.symbol].append(trade)
                for callback in self.trade_callbacks:
                    try:
                        callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")

    async def _process_polygon_message(self, message: str):
        """Process Polygon WebSocket message."""
        data = json.loads(message)

        for msg in data:
            ev_type = msg.get("ev")

            if ev_type == "Q":  # Quote
                quote = RealtimeQuote(
                    symbol=msg["sym"],
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000, tz=UTC),
                    price=(msg["bp"] + msg["ap"]) / 2,
                    size=0,
                    bid=msg["bp"],
                    ask=msg["ap"],
                    bid_size=msg["bs"],
                    ask_size=msg["as"],
                    volume=0,
                    conditions=[],
                )

                self.quote_buffer[quote.symbol].append(quote)
                for callback in self.quote_callbacks:
                    callback(quote)

            elif ev_type == "T":  # Trade
                trade = RealtimeTrade(
                    symbol=msg["sym"],
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000, tz=UTC),
                    price=msg["p"],
                    size=msg["s"],
                    conditions=msg.get("c", []),
                    exchange=msg.get("x", ""),
                )

                self.trade_buffer[trade.symbol].append(trade)
                for callback in self.trade_callbacks:
                    callback(trade)

    async def _process_iex_message(self, message: str):
        """Process IEX Cloud WebSocket message."""
        # IEX has different message format
        # Implementation depends on specific IEX streaming setup
        pass

    async def _reconnect(self):
        """Reconnect to WebSocket stream."""
        logger.info("Attempting to reconnect...")

        # Close existing connection
        if self.ws_conn:
            await self.ws_conn.close()

        # Wait before reconnecting
        await asyncio.sleep(5)

        # Reconnect
        await self.connect()

        # Re-subscribe
        if self.quote_symbols:
            await self.subscribe_quotes(list(self.quote_symbols))
        if self.trade_symbols:
            await self.subscribe_trades(list(self.trade_symbols))

    def get_latest_quote(self, symbol: str) -> RealtimeQuote | None:
        """Get latest quote for a symbol."""
        quotes = self.quote_buffer.get(symbol, [])
        return quotes[-1] if quotes else None

    def get_latest_trade(self, symbol: str) -> RealtimeTrade | None:
        """Get latest trade for a symbol."""
        trades = self.trade_buffer.get(symbol, [])
        return trades[-1] if trades else None

    def get_volume_profile(self, symbol: str, window_seconds: int = 60) -> dict[str, Any]:
        """Get volume profile for recent trades."""
        trades = self.trade_buffer.get(symbol, [])
        if not trades:
            return {}

        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=window_seconds)

        recent_trades = [t for t in trades if t.timestamp >= cutoff]

        if not recent_trades:
            return {}

        prices = [t.price for t in recent_trades]
        sizes = [t.size for t in recent_trades]

        return {
            "trade_count": len(recent_trades),
            "total_volume": sum(sizes),
            "avg_price": np.mean(prices),
            "price_std": np.std(prices),
            "max_price": max(prices),
            "min_price": min(prices),
            "vwap": np.average(prices, weights=sizes) if sizes else 0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        return {
            "provider": self.provider,
            "connected": self.ws_conn is not None and not self.ws_conn.closed,
            "authenticated": self.authenticated,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_message_time": (
                self.last_message_time.isoformat() if self.last_message_time else None
            ),
            "quote_symbols": len(self.quote_symbols),
            "trade_symbols": len(self.trade_symbols),
            "quote_buffer_size": sum(len(quotes) for quotes in self.quote_buffer.values()),
            "trade_buffer_size": sum(len(trades) for trades in self.trade_buffer.values()),
        }

    def clear_buffers(self, max_age_seconds: int = 300):
        """Clear old data from buffers."""
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)

        # Clear old quotes
        for symbol in list(self.quote_buffer.keys()):
            self.quote_buffer[symbol] = [
                q for q in self.quote_buffer[symbol] if q.timestamp >= cutoff
            ]
            if not self.quote_buffer[symbol]:
                del self.quote_buffer[symbol]

        # Clear old trades
        for symbol in list(self.trade_buffer.keys()):
            self.trade_buffer[symbol] = [
                t for t in self.trade_buffer[symbol] if t.timestamp >= cutoff
            ]
            if not self.trade_buffer[symbol]:
                del self.trade_buffer[symbol]

    async def close(self):
        """Close WebSocket connection."""
        if self.ws_conn:
            await self.ws_conn.close()
            logger.info("WebSocket connection closed")

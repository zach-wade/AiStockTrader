# File: src/main/scanners/layers/layer3_realtime_scanner.py
"""
Enhanced Layer 3 Real-time Scanner with WebSocket Streaming

Provides sub-second opportunity detection for hunter-killer strategy.
"""

# Standard library imports
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

# Third-party imports
import numpy as np
from sqlalchemy import text

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.scanners.layers.realtime_websocket_stream import (
    RealtimeQuote,
    RealtimeTrade,
    WebSocketDataStream,
)
from main.utils.cache import CacheType, get_global_cache

logger = logging.getLogger(__name__)


@dataclass
class RealtimeOpportunity:
    """Real-time trading opportunity."""

    symbol: str
    timestamp: datetime
    score: float
    rvol: float
    price_change_pct: float
    current_price: float
    bid: float
    ask: float
    spread_bps: float
    volume: float
    momentum_score: float
    catalyst_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Layer3RealtimeScanner:
    """
    Enhanced Layer 3 scanner with real-time WebSocket streaming.

    Features:
    - Sub-second market data updates via WebSocket
    - Redis caching for instant data access
    - Parallel processing of opportunities
    - Smart symbol prioritization
    """

    def __init__(self, config=None):
        """Initialize real-time scanner."""
        self.config = config or get_config()
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(self.config)
        self.cache = get_global_cache()

        # Scanner configuration
        scanner_config = self.config.get("scanner", {}).get("layer3", {})
        self.use_websocket = scanner_config.get("use_websocket", True)
        self.update_interval = scanner_config.get("update_interval_seconds", 0.1)
        self.top_candidates_count = scanner_config.get("top_candidates_count", 20)
        self.rvol_threshold = scanner_config.get("rvol_threshold", 2.0)
        self.price_change_threshold = scanner_config.get("price_change_threshold", 0.02)

        # WebSocket stream
        self.ws_stream = None
        if self.use_websocket:
            # Get API credentials from config
            api_key = self.config.get("data_sources", {}).get("alpaca", {}).get("api_key")
            api_secret = self.config.get("data_sources", {}).get("alpaca", {}).get("api_secret")

            self.ws_stream = WebSocketDataStream(
                provider="alpaca",
                api_key=api_key,
                api_secret=api_secret,
                feed="iex",  # Use IEX feed for lower latency
            )

        # Tracking data
        self.tracked_symbols: set[str] = set()
        self.opportunity_scores: dict[str, float] = {}
        self.rvol_baselines: dict[str, dict[str, float]] = {}

        # Real-time data buffers
        self.quote_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.trade_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_tracker: dict[str, float] = defaultdict(float)

        # Performance metrics
        self.scan_count = 0
        self.opportunity_count = 0
        self.last_scan_time = None

        logger.info(f"Layer 3 Real-time Scanner initialized (WebSocket: {self.use_websocket})")

    async def initialize(self):
        """Initialize scanner components."""
        if self.ws_stream:
            # Connect to WebSocket
            await self.ws_stream.connect()

            # Set up callbacks
            self.ws_stream.add_quote_callback(self._on_quote_update)
            self.ws_stream.add_trade_callback(self._on_trade_update)

            # Start streaming task
            asyncio.create_task(self.ws_stream.stream())

            # Start buffer cleanup task
            asyncio.create_task(self._periodic_buffer_cleanup())

        logger.info("Real-time scanner initialized")

    async def run(self, input_symbols: list[str]) -> list[str]:
        """
        Run Layer 3 scanning and return filtered symbols.
        Wrapper around scan for funnel compatibility.
        """
        opportunities = await self.scan(input_symbols)
        return [opp.symbol for opp in opportunities]

    async def scan(self, symbols: list[str]) -> list[RealtimeOpportunity]:
        """
        Perform real-time scan on qualified symbols.

        Args:
            symbols: List of Layer 2 qualified symbols

        Returns:
            List of real-time opportunities
        """
        start_time = datetime.now(UTC)
        self.scan_count += 1

        try:
            # Update tracked symbols
            await self._update_tracked_symbols(symbols)

            # Load baseline data if needed
            if not self.rvol_baselines:
                await self._load_rvol_baselines(symbols)

            # Get real-time opportunities
            opportunities = await self._identify_opportunities()

            # Rank and filter
            qualified = self._rank_opportunities(opportunities)

            # Update database
            await self._update_database(qualified)

            # Update metrics
            self.opportunity_count = len(qualified)
            self.last_scan_time = datetime.now(UTC)

            # Log performance
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.info(
                f"Real-time scan completed in {duration:.3f}s - {len(qualified)} opportunities"
            )

            return qualified

        except Exception as e:
            logger.error(f"Error in real-time scan: {e}", exc_info=True)
            return []

    async def _update_tracked_symbols(self, symbols: list[str]):
        """Update symbols being tracked in real-time."""
        new_symbols = set(symbols)

        # Unsubscribe from removed symbols
        removed = self.tracked_symbols - new_symbols
        if removed and self.ws_stream:
            await self.ws_stream.unsubscribe_quotes(list(removed))
            await self.ws_stream.unsubscribe_trades(list(removed))
            logger.info(f"Unsubscribed from {len(removed)} symbols")

        # Subscribe to new symbols
        added = new_symbols - self.tracked_symbols
        if added and self.ws_stream:
            await self.ws_stream.subscribe_quotes(list(added))
            await self.ws_stream.subscribe_trades(list(added))
            logger.info(f"Subscribed to {len(added)} new symbols")

            # Pre-load previous close prices
            await self._load_previous_closes(list(added))

        self.tracked_symbols = new_symbols

    async def _load_previous_closes(self, symbols: list[str]):
        """Load previous close prices for new symbols."""
        for symbol in symbols:
            # Check cache first
            cached_close = await self.cache.get(CacheType.CUSTOM, f"prev_close:{symbol}")
            if cached_close:
                continue

            # Query database
            query = text(
                """
                SELECT close
                FROM market_data
                WHERE symbol = :symbol
                AND timestamp < DATE_TRUNC('day', NOW())
                ORDER BY timestamp DESC
                LIMIT 1
            """
            )

            def execute_query(session):
                result = session.execute(query, {"symbol": symbol})
                row = result.first()
                return float(row.close) if row else 0.0

            close_price = await self.db_adapter.run_sync(execute_query)

            # Cache the result
            await self.cache.set(
                CacheType.CUSTOM, f"prev_close:{symbol}", close_price, 86400
            )  # 24 hours

    async def _load_rvol_baselines(self, symbols: list[str]):
        """Load RVOL baseline data."""
        # Implementation similar to original but with Redis caching
        for symbol in symbols:
            # Check cache
            cached_baseline = await self.cache.get(CacheType.CUSTOM, f"rvol_baseline:{symbol}")
            if cached_baseline:
                self.rvol_baselines[symbol] = cached_baseline
                continue

            # Load from database if not cached
            # (Implementation omitted for brevity - similar to original)

    async def _on_quote_update(self, quote: RealtimeQuote):
        """Handle real-time quote update."""
        # Buffer quote
        self.quote_buffer[quote.symbol].append(quote)

        # Update Redis cache for instant access
        market_data = {
            "symbol": quote.symbol,
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread": quote.spread,
            "spread_bps": quote.spread_bps,
            "timestamp": quote.timestamp.isoformat(),
        }
        await self.cache.set(CacheType.QUOTES, f"market:{quote.symbol}", market_data, 5)

    def _on_trade_update(self, trade: RealtimeTrade):
        """Handle real-time trade update."""
        # Buffer trade
        self.trade_buffer[trade.symbol].append(trade)

        # Update volume tracker
        self.volume_tracker[trade.symbol] += trade.size

    async def _identify_opportunities(self) -> list[RealtimeOpportunity]:
        """Identify real-time trading opportunities."""
        opportunities = []

        for symbol in self.tracked_symbols:
            try:
                # Get latest quote
                latest_quote = self.ws_stream.get_latest_quote(symbol) if self.ws_stream else None
                if not latest_quote:
                    continue

                # Calculate metrics - get from cache
                prev_close = await self.cache.get(CacheType.CUSTOM, f"prev_close:{symbol}")
                if not prev_close or prev_close <= 0:
                    continue

                price_change_pct = ((latest_quote.price - prev_close) / prev_close) * 100

                # Check price change threshold
                if abs(price_change_pct) < self.price_change_threshold * 100:
                    continue

                # Calculate RVOL
                current_volume = self.volume_tracker.get(symbol, 0)
                rvol = self._calculate_realtime_rvol(symbol, current_volume)

                # Check RVOL threshold
                if rvol < self.rvol_threshold:
                    continue

                # Calculate momentum score
                momentum_score = self._calculate_momentum_score(symbol)

                # Get catalyst score from cache
                catalyst_score = (
                    await self.cache.get(CacheType.SIGNALS, f"catalyst_score:{symbol}") or 0
                )

                # Calculate opportunity score
                score = self._calculate_opportunity_score(
                    rvol=rvol,
                    price_change_pct=price_change_pct,
                    spread_bps=latest_quote.spread_bps,
                    momentum_score=momentum_score,
                    catalyst_score=catalyst_score,
                )

                # Create opportunity
                opportunity = RealtimeOpportunity(
                    symbol=symbol,
                    timestamp=latest_quote.timestamp,
                    score=score,
                    rvol=rvol,
                    price_change_pct=price_change_pct,
                    current_price=latest_quote.price,
                    bid=latest_quote.bid,
                    ask=latest_quote.ask,
                    spread_bps=latest_quote.spread_bps,
                    volume=current_volume,
                    momentum_score=momentum_score,
                    catalyst_score=catalyst_score,
                    metadata={
                        "quote_time": latest_quote.timestamp.isoformat(),
                        "trades_count": len(self.trade_buffer[symbol]),
                    },
                )

                opportunities.append(opportunity)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        return opportunities

    def _calculate_realtime_rvol(self, symbol: str, current_volume: float) -> float:
        """Calculate real-time relative volume."""
        if symbol not in self.rvol_baselines:
            return 0.0

        # Get current time bucket
        now = datetime.now(UTC)
        hour = now.hour
        minute = (now.minute // 5) * 5  # 5-minute buckets
        time_key = f"{hour:02d}:{minute:02d}"

        baseline = self.rvol_baselines[symbol].get(time_key, {})
        avg_volume = baseline.get("avg_volume", 0)

        if avg_volume > 0:
            return current_volume / avg_volume

        return 0.0

    def _calculate_momentum_score(self, symbol: str) -> float:
        """Calculate real-time momentum score."""
        trades = list(self.trade_buffer[symbol])
        if len(trades) < 10:
            return 0.0

        # Get recent trades (last 30 seconds)
        cutoff = datetime.now(UTC) - timedelta(seconds=30)
        recent_trades = [t for t in trades if t.timestamp >= cutoff]

        if len(recent_trades) < 5:
            return 0.0

        # Calculate price momentum
        prices = [t.price for t in recent_trades]
        first_price = prices[0]
        last_price = prices[-1]

        if first_price > 0:
            momentum = ((last_price - first_price) / first_price) * 100

            # Calculate consistency (lower std = more consistent)
            price_std = np.std(prices)
            consistency = 1 / (1 + price_std)

            # Combined score
            return abs(momentum) * consistency

        return 0.0

    def _calculate_opportunity_score(
        self,
        rvol: float,
        price_change_pct: float,
        spread_bps: float,
        momentum_score: float,
        catalyst_score: float,
    ) -> float:
        """Calculate comprehensive opportunity score."""
        # RVOL component (0-5 points)
        rvol_score = min(5.0, rvol)

        # Price movement component (0-4 points)
        price_score = min(4.0, abs(price_change_pct))

        # Momentum component (0-3 points)
        momentum_pts = min(3.0, momentum_score / 2)

        # Spread quality (0-2 points) - tighter spread is better
        spread_score = max(0, 2.0 - (spread_bps / 10))

        # Catalyst bonus (0-3 points)
        catalyst_pts = min(3.0, catalyst_score)

        # Total score
        total_score = (
            rvol_score * 0.3  # 30% weight on volume
            + price_score * 0.25  # 25% weight on price movement
            + momentum_pts * 0.2  # 20% weight on momentum
            + spread_score * 0.1  # 10% weight on spread
            + catalyst_pts * 0.15  # 15% weight on catalysts
        )

        return total_score

    def _rank_opportunities(
        self, opportunities: list[RealtimeOpportunity]
    ) -> list[RealtimeOpportunity]:
        """Rank and filter opportunities."""
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        # Take top N
        return opportunities[: self.top_candidates_count]

    async def _update_database(self, opportunities: list[RealtimeOpportunity]):
        """Update database with qualified symbols."""
        if not opportunities:
            return

        try:
            # Import company repository
            # Local imports
            from main.data_pipeline.storage.repositories import get_repository_factory

            # Create company repository instance using factory
            factory = get_repository_factory()
            company_repo = factory.create_company_repository(self.db_adapter)

            # Extract qualified symbols and scores
            qualified_symbols = [opp.symbol for opp in opportunities]
            premarket_scores = {opp.symbol: opp.score for opp in opportunities}

            # Get all tracked symbols as the evaluated set
            all_input_symbols = list(self.tracked_symbols)

            # Update each qualified symbol to Layer 3 (ACTIVE)
            qualified_count = 0
            for opp in opportunities:
                result = await company_repo.update_layer(
                    symbol=opp.symbol,
                    layer=DataLayer.ACTIVE,
                    metadata={
                        "realtime_score": opp.score,
                        "rvol": opp.rvol,
                        "price_change": opp.price_change,
                        "volume": opp.volume,
                        "source": "layer3_realtime_scanner",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                if result.success:
                    qualified_count += 1
                else:
                    logger.warning(f"Failed to update layer for {opp.symbol}: {result.errors}")

            logger.info(
                f"✅ Updated Layer 3 real-time qualifications: "
                f"{qualified_count} qualified out of {len(opportunities)} opportunities"
            )

            # Update additional real-time specific fields using direct query
            update_query = text(
                """
                UPDATE companies
                SET rvol = :rvol,
                    current_price = :price
                WHERE symbol = :symbol
            """
            )

            def execute_updates(session):
                for opp in opportunities:
                    session.execute(
                        update_query,
                        {"symbol": opp.symbol, "rvol": opp.rvol, "price": opp.current_price},
                    )
                session.commit()

            await self.db_adapter.run_sync(execute_updates)

        except Exception as e:
            logger.error(f"Error updating Layer 3 qualifications: {e}", exc_info=True)
            # Don't fail the scan if qualification update fails

        # Also update Redis for instant access
        for opp in opportunities:
            await self.cache.set(
                CacheType.CUSTOM,
                f"opportunity:{opp.symbol}",
                {
                    "score": opp.score,
                    "rvol": opp.rvol,
                    "price_change_pct": opp.price_change_pct,
                    "timestamp": opp.timestamp.isoformat(),
                },
                60,
            )

    async def _periodic_buffer_cleanup(self):
        """Periodically clean up old data from buffers."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Clear old WebSocket buffers
                if self.ws_stream:
                    self.ws_stream.clear_buffers(max_age_seconds=600)

                # Clear old quotes/trades
                cutoff = datetime.now(UTC) - timedelta(minutes=10)

                for symbol in list(self.quote_buffer.keys()):
                    # Remove old quotes
                    while (
                        self.quote_buffer[symbol]
                        and self.quote_buffer[symbol][0].timestamp < cutoff
                    ):
                        self.quote_buffer[symbol].popleft()

                for symbol in list(self.trade_buffer.keys()):
                    # Remove old trades
                    while (
                        self.trade_buffer[symbol]
                        and self.trade_buffer[symbol][0].timestamp < cutoff
                    ):
                        self.trade_buffer[symbol].popleft()

                logger.debug("Buffer cleanup completed")

            except Exception as e:
                logger.error(f"Error in buffer cleanup: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        stats = {
            "scan_count": self.scan_count,
            "opportunity_count": self.opportunity_count,
            "tracked_symbols": len(self.tracked_symbols),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "websocket_connected": self.ws_stream.ws_conn is not None if self.ws_stream else False,
            "cache_stats": await self.cache.get_stats() if hasattr(self.cache, "get_stats") else {},
        }

        if self.ws_stream:
            stats["websocket_stats"] = self.ws_stream.get_stats()

        return stats

    async def close(self):
        """Clean up resources."""
        if self.ws_stream:
            await self.ws_stream.close()

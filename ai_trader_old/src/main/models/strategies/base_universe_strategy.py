"""
Base class for universe-based strategies.

This module provides a foundation for strategies that operate on filtered
universes of stocks, with built-in universe management and filtering capabilities.
"""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Set

# Third-party imports
import pandas as pd

# Local imports
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.models.common import MarketData
from main.models.strategies.base_strategy import BaseStrategy, Signal

# from main.utils.core import create_event_tracker  # TODO: Implement event tracker
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class UniverseFilter:
    """Configuration for universe filtering."""

    min_price: float = 5.0
    max_price: float = 1000.0
    min_volume: int = 100000
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    exchanges: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    exclude_symbols: Set[str] = field(default_factory=set)


@dataclass
class UniverseStats:
    """Statistics about the universe."""

    total_symbols: int
    filtered_symbols: int
    filter_reasons: Dict[str, int]
    sector_distribution: Dict[str, int]
    market_cap_distribution: Dict[str, int]


class BaseUniverseStrategy(BaseStrategy):
    """
    Base class for strategies that operate on filtered stock universes.

    Provides universe filtering, ranking, and management capabilities
    that derived strategies can leverage.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_engine: UnifiedFeatureEngine,
        universe_filter: Optional[UniverseFilter] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize universe strategy.

        Args:
            config: Strategy configuration
            feature_engine: Feature engine for data
            universe_filter: Universe filtering configuration
            metrics_collector: Optional metrics collector
        """
        super().__init__(config, feature_engine)

        self.universe_filter = universe_filter or UniverseFilter()
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker(f"strategy_{self.name}")

        # Universe management
        self._current_universe: Set[str] = set()
        self._universe_ranks: Dict[str, float] = {}
        self._last_universe_update: Optional[datetime] = None
        self._universe_update_frequency = config.get("universe_update_hours", 24)

        # Universe statistics
        self._universe_stats: Optional[UniverseStats] = None

    @abstractmethod
    async def rank_universe(self, universe: List[str]) -> Dict[str, float]:
        """
        Rank symbols in the universe by desirability.

        Args:
            universe: List of symbols in the universe

        Returns:
            Dictionary mapping symbols to ranking scores (higher is better)
        """
        pass

    @abstractmethod
    async def generate_signals_for_universe(
        self, ranked_universe: Dict[str, float]
    ) -> List[Signal]:
        """
        Generate signals for the ranked universe.

        Args:
            ranked_universe: Symbols with their ranking scores

        Returns:
            List of trading signals
        """
        pass

    async def execute(self, timestamp: datetime) -> List[Signal]:
        """
        Execute strategy on the filtered universe.

        Args:
            timestamp: Current timestamp

        Returns:
            List of trading signals
        """
        try:
            # Update universe if needed
            if self._should_update_universe(timestamp):
                await self._update_universe(timestamp)

            # Get current universe as list
            universe_list = list(self._current_universe)

            if not universe_list:
                logger.warning(f"{self.name}: Empty universe after filtering")
                return []

            # Rank universe
            self._universe_ranks = await self.rank_universe(universe_list)

            # Generate signals based on rankings
            signals = await self.generate_signals_for_universe(self._universe_ranks)

            # Track execution
            self._track_execution(timestamp, len(signals))

            return signals

        except Exception as e:
            logger.error(f"{self.name}: Error in execute: {e}")
            return []

    async def _update_universe(self, timestamp: datetime) -> None:
        """Update the trading universe."""
        logger.info(f"{self.name}: Updating universe")

        # Get all available symbols
        all_symbols = await self._get_all_symbols()

        # Apply filters
        filtered_universe, stats = await self._apply_universe_filters(all_symbols, timestamp)

        # Update state
        self._current_universe = filtered_universe
        self._universe_stats = stats
        self._last_universe_update = timestamp

        # Log statistics
        logger.info(
            f"{self.name}: Universe updated - "
            f"{stats.filtered_symbols}/{stats.total_symbols} symbols passed filters"
        )

        # Track metrics
        if self.metrics:
            self.metrics.gauge(f"strategy.{self.name}.universe_size", stats.filtered_symbols)

    async def _apply_universe_filters(
        self, symbols: List[str], timestamp: datetime
    ) -> tuple[Set[str], UniverseStats]:
        """Apply filters to create the trading universe."""
        filtered = set()
        filter_reasons = {}
        sector_dist = {}
        mcap_dist = {}

        for symbol in symbols:
            # Skip excluded symbols
            if symbol in self.universe_filter.exclude_symbols:
                filter_reasons["excluded"] = filter_reasons.get("excluded", 0) + 1
                continue

            try:
                # Get market data
                market_data = await self.feature_engine.get_market_data(symbol, timestamp)

                if not market_data:
                    filter_reasons["no_data"] = filter_reasons.get("no_data", 0) + 1
                    continue

                # Price filter
                if market_data.price < self.universe_filter.min_price:
                    filter_reasons["price_too_low"] = filter_reasons.get("price_too_low", 0) + 1
                    continue

                if market_data.price > self.universe_filter.max_price:
                    filter_reasons["price_too_high"] = filter_reasons.get("price_too_high", 0) + 1
                    continue

                # Volume filter
                if market_data.volume < self.universe_filter.min_volume:
                    filter_reasons["volume_too_low"] = filter_reasons.get("volume_too_low", 0) + 1
                    continue

                # Get reference data
                ref_data = await self._get_reference_data(symbol)

                if ref_data:
                    # Market cap filter
                    market_cap = ref_data.get("market_cap")
                    if market_cap:
                        if (
                            self.universe_filter.min_market_cap
                            and market_cap < self.universe_filter.min_market_cap
                        ):
                            filter_reasons["mcap_too_low"] = (
                                filter_reasons.get("mcap_too_low", 0) + 1
                            )
                            continue

                        if (
                            self.universe_filter.max_market_cap
                            and market_cap > self.universe_filter.max_market_cap
                        ):
                            filter_reasons["mcap_too_high"] = (
                                filter_reasons.get("mcap_too_high", 0) + 1
                            )
                            continue

                        # Track market cap distribution
                        if market_cap < 1e9:
                            mcap_dist["small"] = mcap_dist.get("small", 0) + 1
                        elif market_cap < 10e9:
                            mcap_dist["mid"] = mcap_dist.get("mid", 0) + 1
                        else:
                            mcap_dist["large"] = mcap_dist.get("large", 0) + 1

                    # Exchange filter
                    exchange = ref_data.get("exchange")
                    if (
                        self.universe_filter.exchanges
                        and exchange not in self.universe_filter.exchanges
                    ):
                        filter_reasons["wrong_exchange"] = (
                            filter_reasons.get("wrong_exchange", 0) + 1
                        )
                        continue

                    # Sector filter
                    sector = ref_data.get("sector")
                    if sector:
                        if (
                            self.universe_filter.sectors
                            and sector not in self.universe_filter.sectors
                        ):
                            filter_reasons["wrong_sector"] = (
                                filter_reasons.get("wrong_sector", 0) + 1
                            )
                            continue

                        if (
                            self.universe_filter.exclude_sectors
                            and sector in self.universe_filter.exclude_sectors
                        ):
                            filter_reasons["excluded_sector"] = (
                                filter_reasons.get("excluded_sector", 0) + 1
                            )
                            continue

                        # Track sector distribution
                        sector_dist[sector] = sector_dist.get(sector, 0) + 1

                # Symbol passed all filters
                filtered.add(symbol)

            except Exception as e:
                logger.debug(f"Error filtering {symbol}: {e}")
                filter_reasons["error"] = filter_reasons.get("error", 0) + 1

        # Create statistics
        stats = UniverseStats(
            total_symbols=len(symbols),
            filtered_symbols=len(filtered),
            filter_reasons=filter_reasons,
            sector_distribution=sector_dist,
            market_cap_distribution=mcap_dist,
        )

        return filtered, stats

    async def _get_all_symbols(self) -> List[str]:
        """Get all available symbols from data provider."""
        # This would typically query a reference data service
        # For now, use symbols from feature engine
        return await self.feature_engine.get_available_symbols()

    async def _get_reference_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get reference data for a symbol."""
        # This would typically query a reference data service
        # For now, return None (filters based on ref data will be skipped)
        return None

    def _should_update_universe(self, timestamp: datetime) -> bool:
        """Check if universe should be updated."""
        if self._last_universe_update is None:
            return True

        hours_since_update = (timestamp - self._last_universe_update).total_seconds() / 3600
        return hours_since_update >= self._universe_update_frequency

    def get_universe_size(self) -> int:
        """Get current universe size."""
        return len(self._current_universe)

    def get_universe_stats(self) -> Optional[UniverseStats]:
        """Get universe statistics."""
        return self._universe_stats

    def get_top_ranked_symbols(self, n: int = 10) -> List[tuple[str, float]]:
        """Get top N ranked symbols."""
        sorted_ranks = sorted(self._universe_ranks.items(), key=lambda x: x[1], reverse=True)
        return sorted_ranks[:n]

    def _track_execution(self, timestamp: datetime, signal_count: int) -> None:
        """Track strategy execution."""
        self.event_tracker.track(
            "execution",
            {
                "timestamp": timestamp.isoformat(),
                "universe_size": len(self._current_universe),
                "signal_count": signal_count,
                "top_symbol": (
                    self.get_top_ranked_symbols(1)[0][0] if self._universe_ranks else None
                ),
            },
        )

        if self.metrics:
            self.metrics.increment(
                f"strategy.{self.name}.executions", tags={"signal_count": str(signal_count)}
            )

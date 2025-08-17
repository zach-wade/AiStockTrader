"""
Symbol selection for backtesting and strategy universe construction.

Provides intelligent symbol selection based on various criteria including:
- Market capitalization and liquidity requirements
- Volatility and price characteristics
- Sector and industry classification
- Historical data availability
- Trading volume patterns
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Local imports
from main.utils.core import ErrorHandlingMixin, ensure_utc, get_last_us_trading_day, get_logger
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric, timer

logger = get_logger(__name__)


@dataclass
class SymbolCriteria:
    """Criteria for symbol selection."""

    # Market cap filters
    min_market_cap: float | None = None
    max_market_cap: float | None = None

    # Liquidity filters
    min_avg_volume: float | None = None
    min_avg_dollar_volume: float | None = None

    # Price filters
    min_price: float | None = 5.0
    max_price: float | None = None

    # Volatility filters
    min_volatility: float | None = None
    max_volatility: float | None = None

    # Data availability
    min_history_days: int = 252  # 1 year default
    max_missing_data_pct: float = 0.05  # 5% max missing data

    # Sector/industry filters
    sectors: list[str] | None = None
    exclude_sectors: list[str] | None = None
    industries: list[str] | None = None
    exclude_industries: list[str] | None = None

    # Exchange filters
    exchanges: list[str] | None = None
    exclude_otc: bool = True

    # Additional filters
    exclude_etfs: bool = False
    exclude_adrs: bool = False
    require_options: bool = False


@dataclass
class SymbolStats:
    """Statistics for a symbol."""

    symbol: str
    market_cap: float | None = None
    avg_volume: float | None = None
    avg_dollar_volume: float | None = None
    current_price: float | None = None
    volatility: float | None = None
    history_days: int = 0
    missing_data_pct: float = 0.0
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    has_options: bool = False
    is_etf: bool = False
    is_adr: bool = False


class SymbolSelector(ErrorHandlingMixin):
    """
    Selects symbols for backtesting based on specified criteria.

    Features:
    - Multi-criteria filtering
    - Historical data validation
    - Liquidity and tradability checks
    - Sector/industry classification
    - Performance optimization for large universes
    """

    def __init__(self, db_pool: DatabasePool, config: dict[str, Any] | None = None):
        """
        Initialize symbol selector.

        Args:
            db_pool: Database connection pool
            config: Optional configuration
        """
        super().__init__()
        self.db_pool = db_pool
        self.config = config or {}

        # Cache settings
        self._cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        self._symbol_cache: dict[str, SymbolStats] = {}
        self._cache_timestamp: datetime | None = None

        # Performance settings
        self._batch_size = self.config.get("batch_size", 100)

    @timer
    async def select_symbols(
        self, criteria: SymbolCriteria, as_of_date: datetime | None = None, limit: int | None = None
    ) -> list[str]:
        """
        Select symbols based on criteria.

        Args:
            criteria: Selection criteria
            as_of_date: Date for selection (default: latest trading day)
            limit: Maximum number of symbols to return

        Returns:
            List of selected symbol codes
        """
        as_of_date = as_of_date or get_last_us_trading_day()
        as_of_date = ensure_utc(as_of_date)

        with self._handle_error("selecting symbols"):
            # Get all candidate symbols
            candidates = await self._get_candidate_symbols(as_of_date)

            # Apply filters
            filtered = await self._apply_filters(candidates, criteria, as_of_date)

            # Sort by liquidity/market cap
            sorted_symbols = self._sort_symbols(filtered, criteria)

            # Apply limit if specified
            if limit:
                sorted_symbols = sorted_symbols[:limit]

            # Record metrics
            record_metric(
                "symbol_selector.symbols_selected",
                len(sorted_symbols),
                tags={"criteria": criteria.__class__.__name__},
            )

            logger.info(
                f"Selected {len(sorted_symbols)} symbols from " f"{len(candidates)} candidates"
            )

            return sorted_symbols

    async def get_symbol_stats(
        self, symbols: list[str], as_of_date: datetime | None = None
    ) -> dict[str, SymbolStats]:
        """
        Get statistics for symbols.

        Args:
            symbols: List of symbols
            as_of_date: Date for stats

        Returns:
            Dictionary mapping symbols to their stats
        """
        as_of_date = as_of_date or get_last_us_trading_day()
        stats = {}

        # Process in batches
        for i in range(0, len(symbols), self._batch_size):
            batch = symbols[i : i + self._batch_size]
            batch_stats = await self._get_batch_stats(batch, as_of_date)
            stats.update(batch_stats)

        return stats

    async def validate_data_availability(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        min_coverage: float = 0.95,
    ) -> dict[str, bool]:
        """
        Validate data availability for symbols.

        Args:
            symbols: Symbols to validate
            start_date: Start of period
            end_date: End of period
            min_coverage: Minimum data coverage required

        Returns:
            Dictionary mapping symbols to availability status
        """
        results = {}

        async with self.db_pool.acquire() as conn:
            for symbol in symbols:
                coverage = await self._check_data_coverage(conn, symbol, start_date, end_date)
                results[symbol] = coverage >= min_coverage

        return results

    async def _get_candidate_symbols(self, as_of_date: datetime) -> list[str]:
        """Get all candidate symbols from database."""
        async with self.db_pool.acquire() as conn:
            # Get active symbols as of date
            query = """
                SELECT DISTINCT symbol
                FROM symbol_master
                WHERE status = 'active'
                AND listing_date <= $1
                AND (delisting_date IS NULL OR delisting_date > $1)
                ORDER BY symbol
            """

            rows = await conn.fetch(query, as_of_date)
            return [row["symbol"] for row in rows]

    async def _apply_filters(
        self, symbols: list[str], criteria: SymbolCriteria, as_of_date: datetime
    ) -> list[str]:
        """Apply all filters to symbol list."""
        # Get stats for all symbols
        all_stats = await self.get_symbol_stats(symbols, as_of_date)

        filtered = []
        for symbol, stats in all_stats.items():
            if self._passes_criteria(stats, criteria):
                filtered.append(symbol)

        return filtered

    def _passes_criteria(self, stats: SymbolStats, criteria: SymbolCriteria) -> bool:
        """Check if symbol passes all criteria."""
        # Market cap filters
        if criteria.min_market_cap and stats.market_cap:
            if stats.market_cap < criteria.min_market_cap:
                return False

        if criteria.max_market_cap and stats.market_cap:
            if stats.market_cap > criteria.max_market_cap:
                return False

        # Liquidity filters
        if criteria.min_avg_volume and stats.avg_volume:
            if stats.avg_volume < criteria.min_avg_volume:
                return False

        if criteria.min_avg_dollar_volume and stats.avg_dollar_volume:
            if stats.avg_dollar_volume < criteria.min_avg_dollar_volume:
                return False

        # Price filters
        if criteria.min_price and stats.current_price:
            if stats.current_price < criteria.min_price:
                return False

        if criteria.max_price and stats.current_price:
            if stats.current_price > criteria.max_price:
                return False

        # Volatility filters
        if criteria.min_volatility and stats.volatility:
            if stats.volatility < criteria.min_volatility:
                return False

        if criteria.max_volatility and stats.volatility:
            if stats.volatility > criteria.max_volatility:
                return False

        # Data availability
        if stats.history_days < criteria.min_history_days:
            return False

        if stats.missing_data_pct > criteria.max_missing_data_pct:
            return False

        # Sector/industry filters
        if criteria.sectors and stats.sector:
            if stats.sector not in criteria.sectors:
                return False

        if criteria.exclude_sectors and stats.sector:
            if stats.sector in criteria.exclude_sectors:
                return False

        # Exchange filters
        if criteria.exchanges and stats.exchange:
            if stats.exchange not in criteria.exchanges:
                return False

        if criteria.exclude_otc and stats.exchange:
            if "OTC" in stats.exchange:
                return False

        # Type filters
        if criteria.exclude_etfs and stats.is_etf:
            return False

        if criteria.exclude_adrs and stats.is_adr:
            return False

        if criteria.require_options and not stats.has_options:
            return False

        return True

    def _sort_symbols(self, symbols: list[str], criteria: SymbolCriteria) -> list[str]:
        """Sort symbols by relevance."""
        # For now, sort by liquidity (can be enhanced)
        # This is a placeholder - actual implementation would use stats
        return sorted(symbols)

    async def _get_batch_stats(
        self, symbols: list[str], as_of_date: datetime
    ) -> dict[str, SymbolStats]:
        """Get stats for a batch of symbols."""
        stats = {}

        async with self.db_pool.acquire() as conn:
            # This is a simplified version - actual implementation
            # would join multiple tables and calculate real stats
            for symbol in symbols:
                stats[symbol] = SymbolStats(
                    symbol=symbol,
                    # Placeholder values - implement actual queries
                    market_cap=1e9,
                    avg_volume=1e6,
                    avg_dollar_volume=1e7,
                    current_price=100.0,
                    volatility=0.25,
                    history_days=500,
                    missing_data_pct=0.02,
                    sector="Technology",
                    exchange="NASDAQ",
                )

        return stats

    async def _check_data_coverage(
        self, conn, symbol: str, start_date: datetime, end_date: datetime
    ) -> float:
        """Check data coverage percentage for symbol."""
        # Placeholder - implement actual coverage check
        return 0.98

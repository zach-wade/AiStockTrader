"""
Scanner query builder utilities.

Provides optimized SQL query builders for complex scanner queries
with support for hot/cold storage considerations.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for optimization decisions."""

    SIMPLE = "simple"  # Single table, basic filters
    MODERATE = "moderate"  # Joins, aggregations
    COMPLEX = "complex"  # Window functions, CTEs
    HEAVY = "heavy"  # Multiple CTEs, heavy computation


@dataclass
class QueryPlan:
    """Execution plan for a scanner query."""

    query_text: str
    parameters: dict[str, Any]
    complexity: QueryComplexity
    estimated_rows: int
    use_index: str | None = None
    cache_key: str | None = None


class ScannerQueryBuilder:
    """
    Builds optimized SQL queries for scanner operations.

    Features:
    - Query optimization based on data volume
    - Index-aware query construction
    - Support for both hot and cold storage schemas
    - Parameterized queries for security
    """

    def __init__(self):
        """Initialize query builder."""
        self.query_cache = {}

    def build_volume_spike_query(
        self,
        symbols: list[str],
        lookback_days: int = 20,
        spike_threshold: float = 2.5,
        interval: str = "1day",
    ) -> QueryPlan:
        """
        Build query for volume spike detection.

        Args:
            symbols: List of symbols
            lookback_days: Days for average calculation
            spike_threshold: Volume spike threshold
            interval: Data interval

        Returns:
            Query plan for execution
        """
        query = """
        WITH volume_analysis AS (
            SELECT
                symbol,
                timestamp,
                volume,
                AVG(volume) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN :lookback PRECEDING AND 1 PRECEDING
                ) as avg_volume,
                STDDEV(volume) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN :lookback PRECEDING AND 1 PRECEDING
                ) as std_volume
            FROM market_data
            WHERE
                symbol = ANY(:symbols)
                AND interval = :interval
                AND timestamp >= :start_date
        ),
        latest_spikes AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                timestamp,
                volume as current_volume,
                avg_volume,
                std_volume,
                (volume / NULLIF(avg_volume, 0)) as volume_ratio,
                (volume - avg_volume) / NULLIF(std_volume, 0) as z_score
            FROM volume_analysis
            WHERE avg_volume IS NOT NULL
            ORDER BY symbol, timestamp DESC
        )
        SELECT *
        FROM latest_spikes
        WHERE volume_ratio >= :threshold
        """

        parameters = {
            "symbols": symbols,
            "lookback": lookback_days,
            "interval": interval,
            "threshold": spike_threshold,
            "start_date": datetime.utcnow() - timedelta(days=lookback_days + 1),
        }

        return QueryPlan(
            query_text=query,
            parameters=parameters,
            complexity=QueryComplexity.MODERATE,
            estimated_rows=len(symbols),
            use_index="idx_market_data_symbol_timestamp",
        )

    def build_price_breakout_query(
        self, symbols: list[str], lookback_days: int = 50, breakout_threshold: float = 0.02
    ) -> QueryPlan:
        """
        Build query for price breakout detection.

        Args:
            symbols: List of symbols
            lookback_days: Days for resistance calculation
            breakout_threshold: Breakout threshold percentage

        Returns:
            Query plan for execution
        """
        query = """
        WITH price_levels AS (
            SELECT
                symbol,
                MAX(high) as resistance,
                MIN(low) as support,
                AVG(close) as avg_price,
                STDDEV(close) as price_volatility
            FROM market_data
            WHERE
                symbol = ANY(:symbols)
                AND interval = '1day'
                AND timestamp >= :start_date
                AND timestamp < :end_date
            GROUP BY symbol
        ),
        current_prices AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                timestamp,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE
                symbol = ANY(:symbols)
                AND interval = '1day'
            ORDER BY symbol, timestamp DESC
        )
        SELECT
            cp.symbol,
            cp.timestamp,
            cp.close as current_price,
            pl.resistance,
            pl.support,
            pl.avg_price,
            pl.price_volatility,
            CASE
                WHEN cp.high > pl.resistance * (1 + :threshold) THEN 'resistance_break'
                WHEN cp.low < pl.support * (1 - :threshold) THEN 'support_break'
                ELSE NULL
            END as breakout_type,
            (cp.close - pl.avg_price) / pl.avg_price as price_deviation
        FROM current_prices cp
        JOIN price_levels pl ON cp.symbol = pl.symbol
        WHERE
            cp.high > pl.resistance * (1 + :threshold)
            OR cp.low < pl.support * (1 - :threshold)
        """

        parameters = {
            "symbols": symbols,
            "start_date": datetime.utcnow() - timedelta(days=lookback_days),
            "end_date": datetime.utcnow(),
            "threshold": breakout_threshold,
        }

        return QueryPlan(
            query_text=query,
            parameters=parameters,
            complexity=QueryComplexity.MODERATE,
            estimated_rows=int(len(symbols) * 0.1),  # ~10% expected breakouts
            use_index="idx_market_data_symbol_timestamp",
        )

    def build_momentum_query(
        self, symbols: list[str], short_period: int = 10, long_period: int = 30
    ) -> QueryPlan:
        """
        Build query for momentum calculation.

        Args:
            symbols: List of symbols
            short_period: Short moving average period
            long_period: Long moving average period

        Returns:
            Query plan for execution
        """
        query = """
        WITH price_momentum AS (
            SELECT
                symbol,
                timestamp,
                close,
                AVG(close) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN :short_period PRECEDING AND CURRENT ROW
                ) as sma_short,
                AVG(close) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN :long_period PRECEDING AND CURRENT ROW
                ) as sma_long,
                close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change,
                close - LAG(close, 5) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change_5d
            FROM market_data
            WHERE
                symbol = ANY(:symbols)
                AND interval = '1day'
                AND timestamp >= :start_date
        ),
        latest_momentum AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                timestamp,
                close,
                sma_short,
                sma_long,
                (sma_short - sma_long) / sma_long as momentum_score,
                price_change,
                price_change_5d,
                price_change_5d / LAG(close, 5) OVER (PARTITION BY symbol ORDER BY timestamp) as return_5d
            FROM price_momentum
            ORDER BY symbol, timestamp DESC
        )
        SELECT *
        FROM latest_momentum
        WHERE
            sma_short > sma_long  -- Positive momentum
            AND momentum_score > 0.01  -- At least 1% difference
        """

        parameters = {
            "symbols": symbols,
            "short_period": short_period - 1,  # SQL uses 0-based
            "long_period": long_period - 1,
            "start_date": datetime.utcnow() - timedelta(days=long_period + 10),
        }

        return QueryPlan(
            query_text=query,
            parameters=parameters,
            complexity=QueryComplexity.COMPLEX,
            estimated_rows=int(len(symbols) * 0.3),  # ~30% with momentum
            use_index="idx_market_data_symbol_timestamp",
        )

    def build_correlation_break_query(
        self,
        symbol_pairs: list[tuple[str, str]],
        lookback_days: int = 60,
        correlation_threshold: float = 0.7,
    ) -> QueryPlan:
        """
        Build query for correlation break detection.

        Args:
            symbol_pairs: List of symbol pairs to check
            lookback_days: Days for correlation calculation
            correlation_threshold: Normal correlation threshold

        Returns:
            Query plan for execution
        """
        # This is a complex query that would need proper correlation calculation
        # For now, return a simplified version
        all_symbols = list(set([s for pair in symbol_pairs for s in pair]))

        query = """
        WITH returns_data AS (
            SELECT
                symbol,
                timestamp,
                (close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp))
                    / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as returns
            FROM market_data
            WHERE
                symbol = ANY(:symbols)
                AND interval = '1day'
                AND timestamp >= :start_date
        )
        SELECT
            r1.symbol as symbol1,
            r2.symbol as symbol2,
            CORR(r1.returns, r2.returns) as correlation,
            COUNT(*) as data_points
        FROM returns_data r1
        JOIN returns_data r2 ON r1.timestamp = r2.timestamp
        WHERE r1.symbol < r2.symbol
        GROUP BY r1.symbol, r2.symbol
        HAVING COUNT(*) >= :min_points
        """

        parameters = {
            "symbols": all_symbols,
            "start_date": datetime.utcnow() - timedelta(days=lookback_days),
            "min_points": int(lookback_days * 0.8),  # 80% data availability
        }

        return QueryPlan(
            query_text=query,
            parameters=parameters,
            complexity=QueryComplexity.HEAVY,
            estimated_rows=len(symbol_pairs),
            cache_key=f"correlation_{lookback_days}d",
        )

    def build_unusual_options_query(
        self, symbols: list[str], volume_threshold: int = 1000, oi_threshold: int = 500
    ) -> QueryPlan:
        """
        Build query for unusual options activity.

        Args:
            symbols: List of symbols
            volume_threshold: Minimum volume threshold
            oi_threshold: Minimum open interest threshold

        Returns:
            Query plan for execution
        """
        query = """
        WITH options_stats AS (
            SELECT
                symbol,
                expiration_date,
                strike_price,
                option_type,
                SUM(volume) as total_volume,
                AVG(open_interest) as avg_oi,
                MAX(implied_volatility) as max_iv
            FROM options_data
            WHERE
                symbol = ANY(:symbols)
                AND timestamp >= :start_date
                AND expiration_date >= :min_expiry
            GROUP BY symbol, expiration_date, strike_price, option_type
        )
        SELECT *
        FROM options_stats
        WHERE
            total_volume >= :volume_threshold
            AND avg_oi >= :oi_threshold
        ORDER BY total_volume DESC
        """

        parameters = {
            "symbols": symbols,
            "start_date": datetime.utcnow() - timedelta(days=1),
            "min_expiry": datetime.utcnow() + timedelta(days=7),
            "volume_threshold": volume_threshold,
            "oi_threshold": oi_threshold,
        }

        return QueryPlan(
            query_text=query,
            parameters=parameters,
            complexity=QueryComplexity.MODERATE,
            estimated_rows=int(len(symbols) * 5),  # ~5 strikes per symbol
            use_index="idx_options_symbol_timestamp",
        )

    def optimize_query(self, query_plan: QueryPlan) -> QueryPlan:
        """
        Optimize query based on complexity and data characteristics.

        Args:
            query_plan: Original query plan

        Returns:
            Optimized query plan
        """
        # Add query hints based on complexity
        if query_plan.complexity == QueryComplexity.HEAVY:
            # Add parallel execution hint for complex queries
            query_plan.query_text = f"/*+ parallel(4) */ {query_plan.query_text}"

        # Add index hints if specified
        if query_plan.use_index:
            # This would be database-specific
            pass

        return query_plan

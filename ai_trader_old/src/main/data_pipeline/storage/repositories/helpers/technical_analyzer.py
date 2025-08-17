"""
Technical Analyzer Helper

Handles technical indicator calculations and analysis for repositories.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
import json

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


class TechnicalAnalyzer:
    """
    Calculates technical indicators and metrics.

    Separates complex technical analysis logic from repository operations.
    """

    def __init__(self, db_adapter: IAsyncDatabase):
        """Initialize the technical analyzer."""
        self.db_adapter = db_adapter

        self.default_indicators = [
            "rsi",
            "macd",
            "bb_upper",
            "bb_lower",
            "sma_20",
            "sma_50",
            "sma_200",
            "volume_ratio",
            "atr",
            "adx",
        ]

    async def add_indicators(self, df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
        """Add technical indicators to a DataFrame."""
        # In production, would calculate or fetch real indicators
        # This is a placeholder implementation
        for indicator in indicators:
            if indicator in self.default_indicators:
                # Add placeholder values
                df[indicator] = np.random.randn(len(df)) * 10 + 50

        return df

    async def get_indicators(
        self, symbol: str, start_date: datetime, end_date: datetime, indicators: list[str]
    ) -> pd.DataFrame:
        """Get or calculate technical indicators."""
        try:
            # Check if indicators are stored
            query = """
                SELECT timestamp, indicators
                FROM technical_indicators
                WHERE symbol = $1
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp
            """

            results = await self.db_adapter.fetch_all(query, symbol.upper(), start_date, end_date)

            if results:
                # Parse stored indicators
                data = []
                for row in results:
                    ind_dict = json.loads(row["indicators"]) if row["indicators"] else {}
                    filtered = {k: v for k, v in ind_dict.items() if k in indicators}
                    filtered["timestamp"] = row["timestamp"]
                    data.append(filtered)

                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                return df

            # Calculate if not stored
            return await self._calculate_indicators(symbol, start_date, end_date, indicators)

        except Exception as e:
            logger.error(f"Error getting indicators: {e}")
            return pd.DataFrame()

    async def calculate_relative_strength(
        self, symbols: list[str], benchmark: str, period_days: int = 30
    ) -> dict[str, float]:
        """Calculate relative strength vs benchmark."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=period_days)

            # Get benchmark performance
            benchmark_perf = await self._get_performance(benchmark, start_date, end_date)

            if benchmark_perf is None:
                return {}

            # Calculate RS for each symbol
            rs_scores = {}
            for symbol in symbols:
                symbol_perf = await self._get_performance(symbol, start_date, end_date)

                if symbol_perf is not None:
                    rs = (symbol_perf / benchmark_perf) * 100 if benchmark_perf != 0 else 0
                    rs_scores[symbol] = round(rs, 2)

            return rs_scores

        except Exception as e:
            logger.error(f"Error calculating relative strength: {e}")
            return {}

    async def calculate_momentum_scores(
        self, symbols: list[str], lookback_periods: list[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """Calculate momentum scores for symbols."""
        try:
            momentum_data = []

            for symbol in symbols:
                scores = {"symbol": symbol}

                for period in lookback_periods:
                    perf = await self._get_performance(
                        symbol, datetime.now(UTC) - timedelta(days=period), datetime.now(UTC)
                    )

                    if perf is not None:
                        scores[f"momentum_{period}d"] = round(perf, 2)

                # Composite score
                valid_scores = [v for k, v in scores.items() if k != "symbol" and v is not None]
                if valid_scores:
                    scores["composite_momentum"] = round(sum(valid_scores) / len(valid_scores), 2)

                momentum_data.append(scores)

            return pd.DataFrame(momentum_data)

        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return pd.DataFrame()

    async def calculate_volatility_metrics(
        self, symbol: str, period_days: int = 30
    ) -> dict[str, float]:
        """Calculate volatility metrics."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=period_days)

            # Get daily returns
            query = """
                SELECT
                    close,
                    LAG(close) OVER (ORDER BY timestamp) as prev_close
                FROM market_data_1h
                WHERE symbol = $1
                AND interval = '1day'
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp
            """

            results = await self.db_adapter.fetch_all(query, symbol.upper(), start_date, end_date)

            if len(results) < 2:
                return {}

            # Calculate returns
            returns = []
            for row in results[1:]:
                if row["prev_close"] and row["prev_close"] != 0:
                    ret = (row["close"] - row["prev_close"]) / row["prev_close"]
                    returns.append(ret)

            if not returns:
                return {}

            returns_array = np.array(returns)

            return {
                "volatility": float(np.std(returns_array) * np.sqrt(252)),
                "daily_volatility": float(np.std(returns_array)),
                "max_drawdown": float(np.min(returns_array)),
                "max_gain": float(np.max(returns_array)),
            }

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {}

    async def find_support_resistance(
        self, symbol: str, lookback_days: int = 100
    ) -> dict[str, list[float]]:
        """Find support and resistance levels."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=lookback_days)

            query = """
                SELECT high, low, close
                FROM market_data_1h
                WHERE symbol = $1
                AND interval = '1day'
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp
            """

            results = await self.db_adapter.fetch_all(query, symbol.upper(), start_date, end_date)

            if len(results) < 20:
                return {"support": [], "resistance": []}

            df = pd.DataFrame([dict(r) for r in results])

            # Find local peaks and troughs
            resistance = self._find_peaks(df["high"].values)
            support = self._find_troughs(df["low"].values)

            # Cluster nearby levels
            resistance = self._cluster_levels(resistance)
            support = self._cluster_levels(support)

            return {
                "support": sorted(support)[:5],
                "resistance": sorted(resistance, reverse=True)[:5],
            }

        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {"support": [], "resistance": []}

    # Private helper methods
    async def _get_performance(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> float | None:
        """Calculate performance over a period."""
        query = """
            SELECT
                first_value(close) OVER (ORDER BY timestamp) as start_price,
                last_value(close) OVER (ORDER BY timestamp) as end_price
            FROM market_data_1h
            WHERE symbol = $1
            AND interval = '1day'
            AND timestamp >= $2
            AND timestamp <= $3
            LIMIT 1
        """

        result = await self.db_adapter.fetch_one(query, symbol.upper(), start_date, end_date)

        if result and result["start_price"] and result["end_price"]:
            return ((result["end_price"] - result["start_price"]) / result["start_price"]) * 100

        return None

    async def _calculate_indicators(
        self, symbol: str, start_date: datetime, end_date: datetime, indicators: list[str]
    ) -> pd.DataFrame:
        """Calculate indicators from market data."""
        # Simplified implementation
        # In production would calculate real indicators
        return pd.DataFrame()

    def _find_peaks(self, prices: np.ndarray, window: int = 5) -> list[float]:
        """Find local peaks in price data."""
        peaks = []
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] > prices[i + j] for j in range(1, window + 1)
            ):
                peaks.append(float(prices[i]))
        return peaks

    def _find_troughs(self, prices: np.ndarray, window: int = 5) -> list[float]:
        """Find local troughs in price data."""
        troughs = []
        for i in range(window, len(prices) - window):
            if all(prices[i] < prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] < prices[i + j] for j in range(1, window + 1)
            ):
                troughs.append(float(prices[i]))
        return troughs

    def _cluster_levels(self, levels: list[float], threshold: float = 0.02) -> list[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

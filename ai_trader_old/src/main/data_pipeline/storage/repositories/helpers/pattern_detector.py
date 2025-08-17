"""
Pattern Detector Helper

Handles technical pattern detection for scanner repository.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


class PatternDetector:
    """
    Detects technical chart patterns.

    Identifies common patterns like head and shoulders, flags,
    triangles, and other technical formations.
    """

    def __init__(self, db_adapter: IAsyncDatabase):
        """Initialize the pattern detector."""
        self.db_adapter = db_adapter

        self.supported_patterns = [
            "head_and_shoulders",
            "inverse_head_and_shoulders",
            "double_top",
            "double_bottom",
            "triangle_ascending",
            "triangle_descending",
            "wedge_rising",
            "wedge_falling",
            "flag_bullish",
            "flag_bearish",
            "pennant",
            "cup_and_handle",
            "breakout_resistance",
            "breakdown_support",
        ]

    async def detect_patterns(
        self, symbol: str, patterns: list[str], lookback_days: int = 30
    ) -> list[dict[str, Any]]:
        """
        Detect specified patterns in price data.

        Args:
            symbol: Stock symbol
            patterns: List of pattern names to detect
            lookback_days: Days of history to analyze

        Returns:
            List of detected patterns with details
        """
        try:
            # Get price data
            price_data = await self._get_price_data(symbol, lookback_days)

            if price_data.empty:
                return []

            detected = []

            for pattern in patterns:
                if pattern not in self.supported_patterns:
                    logger.warning(f"Pattern '{pattern}' not supported")
                    continue

                # Detect specific pattern
                result = await self._detect_pattern(pattern, price_data)

                if result:
                    detected.append(
                        {
                            "pattern": pattern,
                            "symbol": symbol,
                            "detected_at": datetime.now(UTC),
                            "confidence": result.get("confidence", 0),
                            "details": result.get("details", {}),
                            "price_at_detection": price_data.iloc[-1]["close"],
                        }
                    )

            return detected

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    async def _get_price_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Get historical price data for pattern detection."""
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=lookback_days)

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data_1h
            WHERE symbol = $1
            AND interval = '1day'
            AND timestamp >= $2
            AND timestamp <= $3
            ORDER BY timestamp ASC
        """

        results = await self.db_adapter.fetch_all(query, symbol.upper(), start_date, end_date)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in results])
        df.set_index("timestamp", inplace=True)
        return df

    async def _detect_pattern(
        self, pattern: str, price_data: pd.DataFrame
    ) -> dict[str, Any] | None:
        """Detect a specific pattern in price data."""

        # Pattern detection methods
        pattern_methods = {
            "head_and_shoulders": self._detect_head_and_shoulders,
            "double_top": self._detect_double_top,
            "double_bottom": self._detect_double_bottom,
            "triangle_ascending": self._detect_triangle_ascending,
            "triangle_descending": self._detect_triangle_descending,
            "breakout_resistance": self._detect_breakout_resistance,
            "breakdown_support": self._detect_breakdown_support,
        }

        method = pattern_methods.get(pattern)
        if method:
            return method(price_data)

        # Placeholder for unsupported patterns
        return None

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect head and shoulders pattern."""
        if len(df) < 20:
            return None

        # Find local peaks
        highs = df["high"].values
        peaks = self._find_peaks(highs, window=5)

        if len(peaks) < 3:
            return None

        # Check for head and shoulders formation
        # Look for 3 peaks where middle is highest
        for i in range(len(peaks) - 2):
            left_shoulder = highs[peaks[i]]
            head = highs[peaks[i + 1]]
            right_shoulder = highs[peaks[i + 2]]

            # Head should be higher than shoulders
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal (within 5%)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

                if shoulder_diff < 0.05:
                    return {
                        "confidence": 0.7 + (0.3 * (1 - shoulder_diff)),
                        "details": {
                            "left_shoulder_idx": int(peaks[i]),
                            "head_idx": int(peaks[i + 1]),
                            "right_shoulder_idx": int(peaks[i + 2]),
                            "neckline": float(min(left_shoulder, right_shoulder)),
                        },
                    }

        return None

    def _detect_double_top(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect double top pattern."""
        if len(df) < 15:
            return None

        highs = df["high"].values
        peaks = self._find_peaks(highs, window=5)

        if len(peaks) < 2:
            return None

        # Look for two peaks at similar levels
        for i in range(len(peaks) - 1):
            peak1 = highs[peaks[i]]
            peak2 = highs[peaks[i + 1]]

            # Peaks should be within 3% of each other
            diff = abs(peak1 - peak2) / peak1

            if diff < 0.03:
                # Find valley between peaks
                valley_start = peaks[i]
                valley_end = peaks[i + 1]
                valley = min(highs[valley_start:valley_end])

                return {
                    "confidence": 0.6 + (0.4 * (1 - diff)),
                    "details": {
                        "first_peak_idx": int(peaks[i]),
                        "second_peak_idx": int(peaks[i + 1]),
                        "resistance_level": float((peak1 + peak2) / 2),
                        "support_level": float(valley),
                    },
                }

        return None

    def _detect_double_bottom(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect double bottom pattern."""
        if len(df) < 15:
            return None

        lows = df["low"].values
        troughs = self._find_troughs(lows, window=5)

        if len(troughs) < 2:
            return None

        # Look for two troughs at similar levels
        for i in range(len(troughs) - 1):
            trough1 = lows[troughs[i]]
            trough2 = lows[troughs[i + 1]]

            # Troughs should be within 3% of each other
            diff = abs(trough1 - trough2) / trough1

            if diff < 0.03:
                # Find peak between troughs
                peak_start = troughs[i]
                peak_end = troughs[i + 1]
                peak = max(lows[peak_start:peak_end])

                return {
                    "confidence": 0.6 + (0.4 * (1 - diff)),
                    "details": {
                        "first_trough_idx": int(troughs[i]),
                        "second_trough_idx": int(troughs[i + 1]),
                        "support_level": float((trough1 + trough2) / 2),
                        "resistance_level": float(peak),
                    },
                }

        return None

    def _detect_triangle_ascending(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect ascending triangle pattern."""
        if len(df) < 10:
            return None

        highs = df["high"].values
        lows = df["low"].values

        # Check for flat resistance and rising support
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)

        # Ascending triangle: flat top, rising bottom
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            return {
                "confidence": 0.7,
                "details": {
                    "resistance_level": float(np.mean(highs[-5:])),
                    "support_slope": float(low_slope),
                    "apex_distance": int(self._calculate_apex_distance(highs, lows)),
                },
            }

        return None

    def _detect_triangle_descending(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect descending triangle pattern."""
        if len(df) < 10:
            return None

        highs = df["high"].values
        lows = df["low"].values

        # Check for falling resistance and flat support
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)

        # Descending triangle: falling top, flat bottom
        if high_slope < -0.01 and abs(low_slope) < 0.01:
            return {
                "confidence": 0.7,
                "details": {
                    "support_level": float(np.mean(lows[-5:])),
                    "resistance_slope": float(high_slope),
                    "apex_distance": int(self._calculate_apex_distance(highs, lows)),
                },
            }

        return None

    def _detect_breakout_resistance(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect resistance breakout."""
        if len(df) < 20:
            return None

        # Find resistance level from recent highs
        recent_highs = df["high"].values[-20:-1]
        resistance = np.percentile(recent_highs, 95)

        current_price = df["close"].values[-1]

        # Check for breakout (price above resistance)
        if current_price > resistance * 1.02:  # 2% above resistance
            # Check volume confirmation
            avg_volume = df["volume"].values[-20:-1].mean()
            current_volume = df["volume"].values[-1]

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            return {
                "confidence": min(0.9, 0.5 + (volume_ratio - 1) * 0.2),
                "details": {
                    "resistance_level": float(resistance),
                    "breakout_price": float(current_price),
                    "volume_ratio": float(volume_ratio),
                    "breakout_percentage": float((current_price - resistance) / resistance * 100),
                },
            }

        return None

    def _detect_breakdown_support(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect support breakdown."""
        if len(df) < 20:
            return None

        # Find support level from recent lows
        recent_lows = df["low"].values[-20:-1]
        support = np.percentile(recent_lows, 5)

        current_price = df["close"].values[-1]

        # Check for breakdown (price below support)
        if current_price < support * 0.98:  # 2% below support
            # Check volume confirmation
            avg_volume = df["volume"].values[-20:-1].mean()
            current_volume = df["volume"].values[-1]

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            return {
                "confidence": min(0.9, 0.5 + (volume_ratio - 1) * 0.2),
                "details": {
                    "support_level": float(support),
                    "breakdown_price": float(current_price),
                    "volume_ratio": float(volume_ratio),
                    "breakdown_percentage": float((support - current_price) / support * 100),
                },
            }

        return None

    # Helper methods
    def _find_peaks(self, prices: np.ndarray, window: int = 5) -> list[int]:
        """Find local peaks in price data."""
        peaks = []
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] > prices[i + j] for j in range(1, window + 1)
            ):
                peaks.append(i)
        return peaks

    def _find_troughs(self, prices: np.ndarray, window: int = 5) -> list[int]:
        """Find local troughs in price data."""
        troughs = []
        for i in range(window, len(prices) - window):
            if all(prices[i] < prices[i - j] for j in range(1, window + 1)) and all(
                prices[i] < prices[i + j] for j in range(1, window + 1)
            ):
                troughs.append(i)
        return troughs

    def _calculate_slope(self, values: np.ndarray) -> float:
        """Calculate slope of values using linear regression."""
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _calculate_apex_distance(self, highs: np.ndarray, lows: np.ndarray) -> int:
        """Calculate distance to triangle apex."""
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)

        if abs(high_slope - low_slope) < 0.001:
            return 999  # Lines are parallel

        # Calculate intersection point
        distance = (highs[-1] - lows[-1]) / abs(high_slope - low_slope)
        return min(int(distance), 999)

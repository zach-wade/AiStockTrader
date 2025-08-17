# File: ai_trader/scanners/technical_scanner.py

# Standard library imports
from collections import defaultdict
from datetime import UTC, datetime
import logging
from typing import Any

# Third-party imports
from omegaconf import DictConfig
import pandas as pd

# Local imports
from main.events.types import AlertType, ScanAlert
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerRepository
from main.utils.core import timer
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector

from ..catalyst_scanner_base import CatalystScannerBase

logger = logging.getLogger(__name__)


class TechnicalScanner(CatalystScannerBase):
    """
    Scans for technical, price-action-based catalysts like breakouts and gaps.

    Now uses the repository pattern with hot/cold storage awareness for
    optimal performance when analyzing technical patterns.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: IEventBus | None = None,
        metrics_collector: ScannerMetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
    ):
        """
        Initializes the TechnicalScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "TechnicalScanner", config, repository, event_bus, metrics_collector, cache_manager
        )

        # Scanner-specific parameters
        self.params = self.config.get("scanners.technical", {})
        self.breakout_pct = self.params.get("breakout_pct", 2.0)
        self.gap_pct = self.params.get("gap_pct", 2.0)
        self.lookback_days = self.params.get("lookback_days", 30)
        self.volume_confirmation = self.params.get("volume_confirmation", True)
        self.min_price = self.params.get("min_price", 5.0)
        self.use_cache = self.params.get("use_cache", True)

        # Technical indicators to calculate
        self.indicators = ["sma_20", "sma_50", "rsi", "bollinger_bands"]

        # Track initialization
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.name}")
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._initialized = False

    async def scan(self, symbols: list[str], **kwargs) -> list[ScanAlert]:
        """
        Scan symbols for technical patterns like breakouts and gaps.

        Uses repository pattern for data access with automatic hot/cold
        storage routing based on data age.

        Args:
            symbols: List of stock symbols to scan
            **kwargs: Additional scanner-specific parameters

        Returns:
            List of ScanAlert objects for technical signals
        """
        if not self._initialized:
            await self.initialize()

        with timer() as t:
            logger.info(f"📈 Technical Scanner: Analyzing {len(symbols)} symbols...")

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = (
                        f"technical_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_days}"
                    )
                    cached_alerts = await self.cache.get_cached_result(
                        self.name, "batch", cache_key
                    )
                    if cached_alerts is not None:
                        logger.info("Using cached results for technical scan")
                        return cached_alerts

                # Get technical data using repository (hot/cold aware)
                # This will use hot storage for recent data, cold for historical
                technical_data = await self.repository.get_technical_data(
                    symbols=symbols,
                    indicators=self.indicators,
                    lookback_days=self.lookback_days + 5,  # Buffer for calculations
                )

                # Get volume statistics if volume confirmation enabled
                volume_stats = None
                if self.volume_confirmation:
                    volume_stats = await self.repository.get_volume_statistics(
                        symbols=symbols, lookback_days=20
                    )

                alerts = []
                for symbol, df in technical_data.items():
                    if df is None or df.empty or len(df) < self.lookback_days:
                        continue

                    # Skip low-priced stocks if configured
                    if self.min_price > 0 and df["close"].iloc[-1] < self.min_price:
                        continue

                    # Convert numeric columns from Decimal to float if needed
                    numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(float)

                    # Calculate technical indicators if not in data
                    df = self._calculate_indicators(df)

                    # Check for different technical patterns
                    breakout_alerts = await self._check_breakout(symbol, df, volume_stats)
                    gap_alerts = await self._check_gap(symbol, df)
                    momentum_alerts = await self._check_momentum(symbol, df)

                    alerts.extend(breakout_alerts)
                    alerts.extend(gap_alerts)
                    alerts.extend(momentum_alerts)

                # Deduplicate alerts for same symbol
                alerts = self.deduplicate_alerts(alerts)

                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=300,  # 5 minute TTL for technical data
                    )

                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)

                logger.info(
                    f"✅ Technical Scanner: Found {len(alerts)} technical signals "
                    f"in {t.elapsed * 1000:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(self.name, t.elapsed * 1000, len(symbols))

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Technical Scanner: {e}", exc_info=True)

                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(self.name, type(e).__name__, str(e))

                return []

    async def run(self, universe: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Legacy method for backward compatibility.

        Args:
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to a list of their technical catalyst signals.
        """
        # Use the new scan method
        alerts = await self.scan(universe)

        # Convert to legacy format
        catalyst_signals = defaultdict(list)
        for alert in alerts:
            signal = {
                "score": alert.metadata.get(
                    "raw_score", alert.score * 5.0
                ),  # Convert from 0-1 to 0-5 scale
                "reason": alert.metadata.get("reason", ""),
                "signal_type": alert.metadata.get("signal_type", "technical"),
            }
            catalyst_signals[alert.symbol].append(signal)

        return dict(catalyst_signals)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators if not already in data."""
        # Simple Moving Averages
        if "sma_20" not in df.columns:
            df["sma_20"] = df["close"].rolling(window=20).mean()
        if "sma_50" not in df.columns:
            df["sma_50"] = df["close"].rolling(window=50).mean()

        # RSI
        if "rsi" not in df.columns:
            df["rsi"] = self._calculate_rsi(df["close"])

        # Bollinger Bands
        if "bb_upper" not in df.columns:
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Volume metrics
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def _check_breakout(
        self, symbol: str, df: pd.DataFrame, volume_stats: dict | None
    ) -> list[ScanAlert]:
        """Check for various breakout patterns."""
        alerts = []

        if len(df) < 21:
            return alerts

        # Price breakout
        high_20d = df["high"].iloc[-21:-1].max()
        current_price = df["close"].iloc[-1]
        current_volume = df["volume"].iloc[-1]

        # Volume confirmation
        volume_confirmed = True
        if volume_stats and symbol in volume_stats:
            avg_volume = volume_stats[symbol].get("avg_volume", 0)
            volume_confirmed = current_volume > avg_volume * 1.5

        # Resistance breakout
        if high_20d > 0 and current_price > high_20d * (1 + self.breakout_pct / 100):
            if not self.volume_confirmation or volume_confirmed:
                breakout_pct = (current_price / high_20d - 1) * 100
                score = min(breakout_pct / 10, 1.0)  # Normalize to 0-1

                # Boost score for volume confirmation
                if volume_confirmed:
                    score = min(score * 1.2, 1.0)

                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.TECHNICAL_BREAKOUT,
                    score=score,
                    metadata={
                        "pattern": "resistance_breakout",
                        "breakout_pct": round(breakout_pct, 2),
                        "resistance_level": round(high_20d, 2),
                        "current_price": round(current_price, 2),
                        "volume_confirmed": volume_confirmed,
                        "reason": f"Resistance Breakout: {breakout_pct:.1f}% above 20-day high",
                    },
                )
                alerts.append(alert)

                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name, AlertType.TECHNICAL_BREAKOUT, symbol, score
                    )

        # Moving average breakout
        if "sma_50" in df.columns and len(df) > 50:
            sma_50 = df["sma_50"].iloc[-1]
            prev_close = df["close"].iloc[-2]

            if prev_close <= sma_50 and current_price > sma_50 * 1.01:
                score = 0.7  # Fixed score for MA breakout

                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.TECHNICAL_BREAKOUT,
                    score=score,
                    metadata={
                        "pattern": "ma_breakout",
                        "ma_period": 50,
                        "ma_value": round(sma_50, 2),
                        "current_price": round(current_price, 2),
                        "reason": f"50-day MA Breakout at ${sma_50:.2f}",
                    },
                )
                alerts.append(alert)

        return alerts

    async def _check_gap(self, symbol: str, df: pd.DataFrame) -> list[ScanAlert]:
        """Check for gap patterns."""
        alerts = []

        if len(df) < 2:
            return alerts

        prev_close = df["close"].iloc[-2]
        today_open = df["open"].iloc[-1]
        current_close = df["close"].iloc[-1]

        # Gap up
        if prev_close > 0 and today_open > prev_close * (1 + self.gap_pct / 100):
            gap_pct = (today_open / prev_close - 1) * 100

            # Check if gap held (didn't fill)
            gap_held = df["low"].iloc[-1] > prev_close

            score = min(gap_pct / 10, 1.0)  # Normalize to 0-1
            if gap_held:
                score = min(score * 1.1, 1.0)  # Boost for held gaps

            alert = self.create_alert(
                symbol=symbol,
                alert_type=AlertType.MOMENTUM,
                score=score,
                metadata={
                    "pattern": "gap_up",
                    "gap_pct": round(gap_pct, 2),
                    "prev_close": round(prev_close, 2),
                    "today_open": round(today_open, 2),
                    "gap_held": gap_held,
                    "reason": f"Gap Up: {gap_pct:.1f}% {'(held)' if gap_held else ''}",
                },
            )
            alerts.append(alert)

        return alerts

    async def _check_momentum(self, symbol: str, df: pd.DataFrame) -> list[ScanAlert]:
        """Check for momentum patterns."""
        alerts = []

        if len(df) < 20:
            return alerts

        # RSI momentum
        if "rsi" in df.columns:
            current_rsi = df["rsi"].iloc[-1]
            prev_rsi = df["rsi"].iloc[-2]

            # Bullish RSI divergence or momentum
            if 30 < current_rsi < 70 and current_rsi > prev_rsi + 5:
                score = 0.6

                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.MOMENTUM,
                    score=score,
                    metadata={
                        "pattern": "rsi_momentum",
                        "current_rsi": round(current_rsi, 2),
                        "rsi_change": round(current_rsi - prev_rsi, 2),
                        "reason": f"RSI Momentum: {current_rsi:.1f} (+{current_rsi - prev_rsi:.1f})",
                    },
                )
                alerts.append(alert)

        # Price momentum (consecutive up days)
        last_5_changes = df["close"].pct_change().iloc[-5:]
        up_days = (last_5_changes > 0).sum()

        if up_days >= 4:
            total_gain = ((df["close"].iloc[-1] / df["close"].iloc[-6]) - 1) * 100
            # Ensure score is between 0 and 1
            raw_score = total_gain / 10
            score = max(0.0, min(raw_score, 0.8))  # Clamp between 0 and 0.8

            alert = self.create_alert(
                symbol=symbol,
                alert_type=AlertType.MOMENTUM,
                score=score,
                metadata={
                    "pattern": "consecutive_gains",
                    "up_days": up_days,
                    "total_gain_pct": round(total_gain, 2),
                    "reason": f"Strong Momentum: {up_days}/5 up days, +{total_gain:.1f}%",
                },
            )
            alerts.append(alert)

        return alerts

# File: ai_trader/scanners/options_scanner.py
"""
Options Scanner - detects catalyst signals through unusual options activity.
Uses repository pattern for efficient access to options data.

V3 Enhancement: Options-based catalyst signal detection
"""

# Standard library imports
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.events.types import AlertType, ScanAlert
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerRepository
from main.scanners.catalyst_scanner_base import CatalystScannerBase
from main.utils.core import timer
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector

logger = logging.getLogger(__name__)


class OptionsScanner(CatalystScannerBase):
    """
    Scanner for detecting options-based catalyst signals.
    Identifies unusual options activity that often precedes significant price movements.

    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access options data and historical patterns.
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
        Initializes the OptionsScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "OptionsScanner", config, repository, event_bus, metrics_collector, cache_manager
        )

        # Scanner thresholds
        self.params = self.config.get("scanners.options", {})
        self.min_signal_strength = self.params.get("min_signal_strength", 0.3)  # 0-1 scale
        self.iv_threshold = self.params.get("iv_threshold", 80.0)  # 80th percentile IV
        self.volume_threshold = self.params.get("volume_threshold", 2.0)  # 2x normal volume
        self.flow_threshold = self.params.get("flow_threshold", 0.3)  # Flow pressure threshold
        self.lookback_days = self.params.get("lookback_days", 30)
        self.use_cache = self.params.get("use_cache", True)

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
        Scan for options-based catalyst signals.

        Uses repository pattern for efficient options data access with hot storage
        for recent activity and cold storage for historical patterns.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List of ScanAlert objects
        """
        if not self._initialized:
            await self.initialize()

        with timer() as t:
            logger.info(f"📈 Options Scanner: Analyzing {len(symbols)} symbols...")

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = (
                        f"options_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_days}"
                    )
                    cached_alerts = await self.cache.get_cached_result(
                        self.name, "batch", cache_key
                    )
                    if cached_alerts is not None:
                        logger.info("Using cached results for options scan")
                        return cached_alerts

                # Build query filter for options data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(UTC) - timedelta(days=self.lookback_days),
                    end_date=datetime.now(UTC),
                )

                # Get options data from repository
                # This will use hot storage for recent data, cold for historical
                options_data = await self.repository.get_options_data(
                    symbols=symbols, query_filter=query_filter
                )

                # Get IV ranks from repository
                iv_ranks = await self.repository.get_iv_ranks(symbols)

                alerts = []
                for symbol, symbol_options_data in options_data.items():
                    if not symbol_options_data:
                        continue

                    # Process options data for this symbol
                    symbol_alerts = await self._process_symbol_options_data(
                        symbol, symbol_options_data, iv_ranks.get(symbol, 0)
                    )
                    alerts.extend(symbol_alerts)

                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)

                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=300,  # 5 minute TTL for options data
                    )

                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)

                logger.info(
                    f"✅ Options Scanner: Found {len(alerts)} signals " f"in {t.elapsed_ms:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(self.name, t.elapsed_ms, len(symbols))

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Options Scanner: {e}", exc_info=True)

                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(self.name, type(e).__name__, str(e))

                return []

    async def run(self, universe: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Legacy method for backward compatibility.

        Args:
            universe: List of symbols to scan

        Returns:
            Dict mapping symbol to catalyst signal data
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
                "signal_type": "options",
                "metadata": {
                    "iv_rank": alert.metadata.get("iv_rank"),
                    "dominant_signal": alert.metadata.get("dominant_signal"),
                    "signal_quality": alert.metadata.get("signal_quality"),
                    "flow_pressure": alert.metadata.get("flow_pressure"),
                },
            }
            catalyst_signals[alert.symbol].append(signal)

        return dict(catalyst_signals)

    async def _process_symbol_options_data(
        self, symbol: str, options_data: list[dict[str, Any]], iv_rank: float
    ) -> list[ScanAlert]:
        """
        Process options data for a symbol and generate alerts.

        Args:
            symbol: Stock symbol
            options_data: List of options activity data
            iv_rank: Current IV rank

        Returns:
            List of alerts for this symbol
        """
        alerts = []

        # Analyze options activity patterns
        analysis = self._analyze_options_activity(options_data)

        # Calculate signal score
        score = 0.0
        reasons = []
        dominant_signal = None

        # 1. IV Rank Signal
        if iv_rank > self.iv_threshold:
            score_contribution = (iv_rank - self.iv_threshold) / 20.0
            score += min(score_contribution * 0.3, 0.3)  # Max 30% from IV
            reasons.append(f"High IV rank: {iv_rank:.1f}th percentile")
            if iv_rank > 95:
                dominant_signal = "extreme_iv"

        # 2. Unusual Volume
        if analysis["volume_ratio"] > self.volume_threshold:
            score += min(analysis["volume_ratio"] / 10, 0.3)  # Max 30% from volume
            reasons.append(f'Unusual volume: {analysis["volume_ratio"]:.1f}x normal')

        # 3. Options Flow
        if abs(analysis["flow_pressure"]) > self.flow_threshold:
            score += min(abs(analysis["flow_pressure"]), 0.3)  # Max 30% from flow
            direction = "bullish" if analysis["flow_pressure"] > 0 else "bearish"
            reasons.append(f'Strong {direction} flow: {analysis["flow_pressure"]:.2f}')
            if abs(analysis["flow_pressure"]) > 0.6:
                dominant_signal = f"{direction}_flow"

        # 4. Sweep Detection
        if analysis["call_sweeps"] > 0:
            score += 0.2
            reasons.append(f'{analysis["call_sweeps"]} call sweeps')
            dominant_signal = "call_sweeps"

        if analysis["put_sweeps"] > 0:
            score += 0.2
            reasons.append(f'{analysis["put_sweeps"]} put sweeps')
            dominant_signal = "put_sweeps"

        # 5. Gamma Squeeze Potential
        if analysis["gamma_squeeze_potential"]:
            score += 0.3
            reasons.append("Gamma squeeze setup detected")
            dominant_signal = "gamma_squeeze"

        # Only create alert if score meets threshold
        if score >= self.min_signal_strength and reasons:
            # Determine alert type
            if dominant_signal in ["extreme_iv", "high_iv"]:
                alert_type = AlertType.HIGH_IV_RANK
            elif dominant_signal in ["call_sweeps", "put_sweeps", "bullish_flow", "bearish_flow"]:
                alert_type = AlertType.LARGE_OPTIONS_FLOW
            else:
                alert_type = AlertType.UNUSUAL_OPTIONS_ACTIVITY

            # Determine signal quality
            if dominant_signal in ["gamma_squeeze", "extreme_iv", "call_sweeps", "put_sweeps"]:
                signal_quality = "high"
            elif dominant_signal in ["bullish_flow", "bearish_flow"]:
                signal_quality = "medium"
            else:
                signal_quality = "low"

            alert = self.create_alert(
                symbol=symbol,
                alert_type=alert_type,
                score=score,
                metadata={
                    "iv_rank": iv_rank,
                    "volume_ratio": analysis["volume_ratio"],
                    "flow_pressure": analysis["flow_pressure"],
                    "call_sweeps": analysis["call_sweeps"],
                    "put_sweeps": analysis["put_sweeps"],
                    "gamma_squeeze_potential": analysis["gamma_squeeze_potential"],
                    "dominant_signal": dominant_signal,
                    "signal_quality": signal_quality,
                    "reason": f"Options signal: {'; '.join(reasons)}",
                    "raw_score": score * 5.0,  # Convert to 0-5 scale for legacy
                },
            )
            alerts.append(alert)

            # Record metric
            if self.metrics:
                self.metrics.record_alert_generated(self.name, alert_type, symbol, score)

        return alerts

    def _analyze_options_activity(self, options_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze options activity data to extract patterns and signals."""
        analysis = {
            "volume_ratio": 0.0,
            "flow_pressure": 0.0,
            "call_sweeps": 0,
            "put_sweeps": 0,
            "gamma_squeeze_potential": False,
            "total_volume": 0,
            "call_volume": 0,
            "put_volume": 0,
        }

        if not options_data:
            return analysis

        # Process each options transaction/activity
        total_volume = 0
        call_volume = 0
        put_volume = 0
        call_premium = 0
        put_premium = 0
        sweep_threshold = 10000  # Minimum premium for sweep

        for activity in options_data:
            volume = activity.get("volume", 0)
            premium = activity.get("premium", 0)
            option_type = activity.get("option_type", "").lower()
            is_sweep = activity.get("is_sweep", False)

            total_volume += volume

            if option_type == "call":
                call_volume += volume
                call_premium += premium
                if is_sweep and premium > sweep_threshold:
                    analysis["call_sweeps"] += 1
            elif option_type == "put":
                put_volume += volume
                put_premium += premium
                if is_sweep and premium > sweep_threshold:
                    analysis["put_sweeps"] += 1

        analysis["total_volume"] = total_volume
        analysis["call_volume"] = call_volume
        analysis["put_volume"] = put_volume

        # Calculate volume ratio (vs average)
        avg_volume = sum(a.get("avg_volume", 0) for a in options_data[:1])  # Get from first record
        if avg_volume > 0:
            analysis["volume_ratio"] = total_volume / avg_volume

        # Calculate flow pressure (call premium - put premium normalized)
        total_premium = call_premium + put_premium
        if total_premium > 0:
            analysis["flow_pressure"] = (call_premium - put_premium) / total_premium

        # Check for gamma squeeze setup (simplified)
        # High call volume at specific strikes near current price
        strike_concentrations = {}
        for activity in options_data:
            strike = activity.get("strike", 0)
            if strike > 0 and option_type == "call":
                strike_concentrations[strike] = strike_concentrations.get(strike, 0) + volume

        # If >50% of call volume is concentrated in 2-3 strikes, potential gamma squeeze
        if strike_concentrations and call_volume > 0:
            top_strikes = sorted(strike_concentrations.values(), reverse=True)[:3]
            concentration = sum(top_strikes) / call_volume
            if concentration > 0.5:
                analysis["gamma_squeeze_potential"] = True

        return analysis

    async def get_scanner_metadata(self) -> dict[str, Any]:
        """Get metadata about this scanner."""
        return {
            "scanner_name": "Options Scanner",
            "scanner_type": "options_activity",
            "description": "Detects catalyst signals through unusual options activity and volatility patterns",
            "signal_types": [
                "high_iv_rank",
                "unusual_options_activity",
                "gamma_squeeze_setup",
                "options_flow_pressure",
                "call_sweeps",
                "put_sweeps",
                "straddle_buying",
            ],
            "thresholds": {
                "min_signal_strength": self.min_signal_strength,
                "iv_threshold": self.iv_threshold,
                "volume_threshold": self.volume_threshold,
                "flow_threshold": self.flow_threshold,
            },
            "last_updated": datetime.now(UTC).isoformat(),
        }

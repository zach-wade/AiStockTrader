# File: ai_trader/scanners/sector_scanner.py
"""
Sector Scanner - detects catalyst signals through sector rotation and momentum analysis.
Uses repository pattern for efficient access to sector data and historical patterns.

V3 Enhancement: Sector rotation-based catalyst signal detection
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


class SectorScanner(CatalystScannerBase):
    """
    Scanner for detecting sector rotation-based catalyst signals.
    Identifies sector momentum shifts that create individual stock opportunities.

    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access sector performance data and historical rotation patterns.
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
        Initializes the SectorScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "SectorScanner", config, repository, event_bus, metrics_collector, cache_manager
        )

        # Scanner thresholds
        self.params = self.config.get("scanners.sector", {})
        self.min_signal_strength = self.params.get("min_signal_strength", 0.3)  # 0-1 scale
        self.momentum_threshold = self.params.get("momentum_threshold", 3.0)  # 3% momentum
        self.rotation_threshold = self.params.get(
            "rotation_threshold", 2.0
        )  # 2% relative performance
        self.relative_strength_threshold = self.params.get(
            "relative_strength_threshold", 1.0
        )  # 1% vs SPY
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
        Scan for sector rotation-based catalyst signals.

        Uses repository pattern for efficient sector data access with hot storage
        for recent performance and cold storage for historical patterns.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List of ScanAlert objects
        """
        if not self._initialized:
            await self.initialize()

        with timer() as t:
            logger.info(f"🔄 Sector Scanner: Analyzing {len(symbols)} symbols...")

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"sector_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_days}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name, "batch", cache_key
                    )
                    if cached_alerts is not None:
                        logger.info("Using cached results for sector scan")
                        return cached_alerts

                # Build query filter for sector data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(UTC) - timedelta(days=self.lookback_days),
                    end_date=datetime.now(UTC),
                )

                # Get sector performance data from repository
                # This will use hot storage for recent data, cold for historical
                sector_data = await self.repository.get_sector_data(
                    symbols=symbols, query_filter=query_filter
                )

                # Calculate sector metrics
                sector_metrics = await self._calculate_sector_metrics(sector_data)

                # Process symbols using concurrent batch processing
                async def process_single_symbol(symbol: str) -> list[ScanAlert]:
                    if symbol not in sector_data:
                        return []

                    # Process sector data for this symbol
                    return await self._process_symbol_sector_data(
                        symbol, sector_data[symbol], sector_metrics
                    )

                # Process all symbols concurrently and flatten results
                symbol_results = await self.process_symbols_individually(
                    symbols,
                    process_single_symbol,
                    max_concurrent=self.config.get("max_concurrent_symbols", 20),
                )

                # Flatten the list of lists into a single list of alerts
                alerts = []
                for symbol_alerts in symbol_results:
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
                        ttl_seconds=900,  # 15 minute TTL for sector data
                    )

                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)

                logger.info(
                    f"✅ Sector Scanner: Found {len(alerts)} signals " f"in {t.elapsed_ms:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(self.name, t.elapsed_ms, len(symbols))

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Sector Scanner: {e}", exc_info=True)

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
                "signal_type": "sector",
                "metadata": {
                    "dominant_signal": alert.metadata.get("dominant_signal", ""),
                    "signal_quality": alert.metadata.get("signal_quality", "low"),
                    "sector_momentum": alert.metadata.get("sector_momentum", 0.0),
                    "relative_strength": alert.metadata.get("relative_strength", 0.0),
                    "economic_cycle_alignment": alert.metadata.get(
                        "economic_cycle_alignment", False
                    ),
                },
            }
            catalyst_signals[alert.symbol].append(signal)

        return dict(catalyst_signals)

    async def _calculate_sector_metrics(self, sector_data: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate aggregate sector metrics from raw data.
        """
        metrics = {
            "sector_momentum": {},
            "rotation_signals": {},
            "relative_performance": {},
            "sector_leaders": [],
            "rotation_type": "neutral",
        }

        # Group symbols by sector
        sector_groups = defaultdict(list)
        for symbol, data in sector_data.items():
            if data and "sector" in data:
                sector = data["sector"]
                sector_groups[sector].append((symbol, data))

        # Calculate metrics for each sector
        for sector, symbols_data in sector_groups.items():
            # Calculate average sector performance
            sector_returns = []
            for symbol, data in symbols_data:
                if "returns" in data:
                    sector_returns.extend(data["returns"])

            if sector_returns:
                # Calculate momentum metrics
                recent_return = sum(sector_returns[-5:]) / min(5, len(sector_returns[-5:]))
                medium_return = sum(sector_returns[-20:]) / min(20, len(sector_returns[-20:]))

                metrics["sector_momentum"][sector] = {
                    "short_momentum": recent_return * 100,  # Convert to percentage
                    "medium_momentum": medium_return * 100,
                    "acceleration": (recent_return - medium_return) * 100,
                }

                # Track sector leaders
                if medium_return * 100 > self.momentum_threshold:
                    metrics["sector_leaders"].append(
                        {
                            "sector": sector,
                            "momentum": medium_return * 100,
                            "symbol_count": len(symbols_data),
                        }
                    )

        # Sort sector leaders
        metrics["sector_leaders"].sort(key=lambda x: x["momentum"], reverse=True)

        # Determine rotation type based on leading sectors
        if metrics["sector_leaders"]:
            top_sectors = [leader["sector"] for leader in metrics["sector_leaders"][:3]]
            metrics["rotation_type"] = self._determine_rotation_type(top_sectors)

        return metrics

    async def _process_symbol_sector_data(
        self, symbol: str, symbol_data: dict[str, Any], sector_metrics: dict[str, Any]
    ) -> list[ScanAlert]:
        """
        Process sector data for a symbol and generate alerts.

        Args:
            symbol: Stock symbol
            symbol_data: Symbol's sector data
            sector_metrics: Aggregate sector metrics

        Returns:
            List of alerts for this symbol
        """
        alerts = []

        # Get symbol's sector
        sector = symbol_data.get("sector")
        if not sector:
            return alerts

        # Calculate signal score
        score = 0.0
        reasons = []
        dominant_signal = None

        # 1. Sector Leadership Signal
        sector_momentum = sector_metrics["sector_momentum"].get(sector, {})
        medium_momentum = sector_momentum.get("medium_momentum", 0)
        acceleration = sector_momentum.get("acceleration", 0)

        if medium_momentum > self.momentum_threshold:
            score += 0.3
            reasons.append(f"{sector} sector momentum: {medium_momentum:.1f}%")
            dominant_signal = "sector_leadership"

            # Acceleration bonus
            if acceleration > 2.0:
                score += 0.1
                reasons.append(f"{sector} accelerating momentum")

        # 2. Rotation Alignment Signal
        rotation_type = sector_metrics.get("rotation_type", "neutral")
        if rotation_type != "neutral" and self._is_sector_aligned_with_rotation(
            sector, rotation_type
        ):
            score += 0.2
            reasons.append(f"Aligned with {rotation_type} rotation")
            if not dominant_signal:
                dominant_signal = "rotation_alignment"

        # 3. Relative Strength Signal
        symbol_returns = symbol_data.get("returns", [])
        if symbol_returns and len(symbol_returns) > 5:
            recent_return = sum(symbol_returns[-5:]) / 5
            if recent_return * 100 > self.relative_strength_threshold:
                score += 0.2
                reasons.append(f"Outperforming sector ({recent_return*100:.1f}%)")
                if not dominant_signal:
                    dominant_signal = "relative_strength"

        # 4. Sector Opportunity Signal
        if sector in [leader["sector"] for leader in sector_metrics["sector_leaders"][:2]]:
            score += 0.2
            reasons.append(f"{sector} sector leading market")
            if not dominant_signal:
                dominant_signal = "sector_opportunity"

        # Only create alert if score meets threshold
        if score >= self.min_signal_strength and reasons:
            # Determine signal quality
            if dominant_signal in ["sector_leadership", "sector_opportunity"]:
                signal_quality = "high"
            elif dominant_signal in ["rotation_alignment", "relative_strength"]:
                signal_quality = "medium"
            else:
                signal_quality = "low"

            alert = self.create_alert(
                symbol=symbol,
                alert_type=AlertType.SECTOR_ROTATION,
                score=score,
                metadata={
                    "sector": sector,
                    "sector_momentum": medium_momentum,
                    "relative_strength": recent_return * 100 if "recent_return" in locals() else 0,
                    "rotation_type": rotation_type,
                    "dominant_signal": dominant_signal,
                    "signal_quality": signal_quality,
                    "economic_cycle_alignment": rotation_type != "neutral",
                    "reason": f"Sector signal: {'; '.join(reasons)}",
                    "raw_score": score * 5.0,  # Legacy scale
                },
            )
            alerts.append(alert)

            # Record metric
            if self.metrics:
                self.metrics.record_alert_generated(
                    self.name, AlertType.SECTOR_ROTATION, symbol, score
                )

        return alerts

    def _determine_rotation_type(self, top_sectors: list[str]) -> str:
        """Determine market rotation type based on leading sectors."""
        # Simplified rotation detection
        early_cycle_sectors = {"Financials", "Industrials", "Consumer Discretionary"}
        mid_cycle_sectors = {"Technology", "Materials", "Energy"}
        late_cycle_sectors = {"Energy", "Materials", "Financials"}
        defensive_sectors = {"Utilities", "Consumer Staples", "Healthcare"}

        # Count sector types in top performers
        early_count = sum(1 for s in top_sectors if s in early_cycle_sectors)
        mid_count = sum(1 for s in top_sectors if s in mid_cycle_sectors)
        late_count = sum(1 for s in top_sectors if s in late_cycle_sectors)
        defensive_count = sum(1 for s in top_sectors if s in defensive_sectors)

        # Determine dominant rotation
        if early_count >= 2:
            return "early_cycle"
        elif mid_count >= 2:
            return "mid_cycle"
        elif late_count >= 2:
            return "late_cycle"
        elif defensive_count >= 2:
            return "recession"
        else:
            return "neutral"

    def _is_sector_aligned_with_rotation(self, sector: str, rotation_type: str) -> bool:
        """Check if sector is aligned with rotation type."""
        rotation_favorable_sectors = {
            "early_cycle": ["Financials", "Industrials", "Consumer Discretionary"],
            "mid_cycle": ["Technology", "Materials", "Energy"],
            "late_cycle": ["Energy", "Materials", "Financials"],
            "recession": ["Utilities", "Consumer Staples", "Healthcare"],
        }

        return sector in rotation_favorable_sectors.get(rotation_type, [])

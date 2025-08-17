# File: ai_trader/scanners/insider_scanner.py

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


class InsiderScanner(CatalystScannerBase):
    """
    Scans for catalyst signals based on unusual insider trading patterns.

    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access insider transaction data and historical patterns.
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
        Initializes the InsiderScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "InsiderScanner", config, repository, event_bus, metrics_collector, cache_manager
        )
        # Scanner-specific parameters
        self.params = self.config.get("scanners.insider", {})
        self.min_signal_strength = self.params.get("min_signal_strength", 0.3)  # 0-1 scale
        self.cluster_significance_usd = self.params.get("cluster_significance_usd", 500000)
        self.lookback_days = self.params.get("lookback_days", 90)
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
        Scan for insider trading-based catalyst signals.

        Uses repository pattern for efficient insider data access with hot storage
        for recent transactions and cold storage for historical patterns.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List of ScanAlert objects
        """
        if not self._initialized:
            await self.initialize()

        with timer() as t:
            logger.info(
                f"👤 Insider Scanner: Analyzing {len(symbols)} symbols for insider activity..."
            )

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = (
                        f"insider_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_days}"
                    )
                    cached_alerts = await self.cache.get_cached_result(
                        self.name, "batch", cache_key
                    )
                    if cached_alerts is not None:
                        logger.info("Using cached results for insider scan")
                        return cached_alerts

                # Build query filter for insider data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(UTC) - timedelta(days=self.lookback_days),
                    end_date=datetime.now(UTC),
                )

                # Get insider transaction data from repository
                # This will use hot storage for recent data, cold for historical
                insider_data = await self.repository.get_insider_data(
                    symbols=symbols, query_filter=query_filter
                )

                alerts = []
                for symbol, transactions in insider_data.items():
                    if not transactions:
                        continue

                    # Process insider transactions for this symbol
                    symbol_alerts = await self._process_symbol_insider_data(symbol, transactions)
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
                        ttl_seconds=3600,  # 1 hour TTL for insider data
                    )

                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)

                logger.info(
                    f"✅ Insider Scanner: Found {len(alerts)} high-conviction signals "
                    f"in {t.elapsed_ms:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(self.name, t.elapsed_ms, len(symbols))

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Insider Scanner: {e}", exc_info=True)

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
            A dictionary mapping symbols to their catalyst signal data.
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
                "signal_type": "insider",
                "metadata": {
                    "net_buying_value": alert.metadata.get("net_buying_value"),
                    "buying_clusters": alert.metadata.get("buying_clusters"),
                    "unusual_patterns": alert.metadata.get("unusual_patterns"),
                },
            }
            catalyst_signals[alert.symbol].append(signal)

        return dict(catalyst_signals)

    async def _process_symbol_insider_data(
        self, symbol: str, transactions: list[dict[str, Any]]
    ) -> list[ScanAlert]:
        """
        Process insider transactions for a symbol and generate alerts.

        Args:
            symbol: Stock symbol
            transactions: List of insider transaction data

        Returns:
            List of alerts for this symbol
        """
        alerts = []

        # Analyze transaction patterns
        analysis = self._analyze_transactions(transactions)

        # Generate alerts based on analysis
        if analysis["net_buying_value"] > 0:
            # Calculate score based on various factors
            score = 0.0
            reasons = []

            # Factor 1: Net buying value
            net_value = analysis["net_buying_value"]
            # Score based on dollar value, normalized to 0-1
            value_score = min(net_value / 1_000_000, 1.0)  # Max at $1M
            score += value_score * 0.4  # 40% weight
            reasons.append(f"Net Buying: ${net_value:,.0f}")

            # Factor 2: Buying clusters
            if analysis["buying_clusters"]:
                cluster_score = min(len(analysis["buying_clusters"]) / 3, 1.0)  # Max at 3 clusters
                score += cluster_score * 0.3  # 30% weight
                reasons.append(f"{len(analysis['buying_clusters'])}x Buying Clusters")

            # Factor 3: Multiple insiders
            if analysis["unique_insiders"] > 1:
                insider_score = min(analysis["unique_insiders"] / 5, 1.0)  # Max at 5 insiders
                score += insider_score * 0.3  # 30% weight
                reasons.append(f"{analysis['unique_insiders']} Insiders Buying")

            # Only create alert if score meets threshold
            if score >= self.min_signal_strength:
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.INSIDER_BUYING,
                    score=score,
                    metadata={
                        "net_buying_value": net_value,
                        "buying_clusters": len(analysis["buying_clusters"]),
                        "unique_insiders": analysis["unique_insiders"],
                        "recent_transactions": analysis["recent_transaction_count"],
                        "reason": "; ".join(reasons),
                        "raw_score": score * 5.0,  # Convert to 0-5 scale for legacy
                    },
                )
                alerts.append(alert)

                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name, AlertType.INSIDER_BUYING, symbol, score
                    )

        return alerts

    def _analyze_transactions(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze insider transactions to extract patterns and signals."""
        analysis = {
            "net_buying_value": 0,
            "net_selling_value": 0,
            "buying_clusters": [],
            "unique_insiders": 0,
            "recent_transaction_count": 0,
        }

        if not transactions:
            return analysis

        # Group by transaction type
        buying_value = 0
        selling_value = 0
        insider_names = set()

        # Sort transactions by date
        sorted_transactions = sorted(
            transactions, key=lambda x: x.get("transaction_date", datetime.min), reverse=True
        )

        # Analyze transactions
        current_cluster = []
        last_date = None

        for txn in sorted_transactions:
            # Track unique insiders
            insider_name = txn.get("insider_name", "")
            if insider_name:
                insider_names.add(insider_name)

            # Track recent transactions (last 30 days)
            txn_date = txn.get("transaction_date")
            if isinstance(txn_date, str):
                txn_date = datetime.fromisoformat(txn_date.replace("Z", "+00:00"))

            if txn_date and (datetime.now(UTC) - txn_date).days <= 30:
                analysis["recent_transaction_count"] += 1

            # Calculate net values
            txn_type = txn.get("transaction_type", "").lower()
            value = txn.get("total_value", 0)

            if "buy" in txn_type or "acquisition" in txn_type:
                buying_value += value

                # Detect clusters (transactions within 7 days)
                if last_date and txn_date and (last_date - txn_date).days <= 7:
                    current_cluster.append(txn)
                else:
                    if len(current_cluster) >= 2:
                        cluster_value = sum(t.get("total_value", 0) for t in current_cluster)
                        if cluster_value >= self.cluster_significance_usd:
                            analysis["buying_clusters"].append(
                                {
                                    "count": len(current_cluster),
                                    "total_value": cluster_value,
                                    "start_date": current_cluster[-1].get("transaction_date"),
                                    "end_date": current_cluster[0].get("transaction_date"),
                                }
                            )
                    current_cluster = [txn]

                last_date = txn_date

            elif "sell" in txn_type or "disposition" in txn_type:
                selling_value += value

        # Check last cluster
        if len(current_cluster) >= 2:
            cluster_value = sum(t.get("total_value", 0) for t in current_cluster)
            if cluster_value >= self.cluster_significance_usd:
                analysis["buying_clusters"].append(
                    {
                        "count": len(current_cluster),
                        "total_value": cluster_value,
                        "start_date": current_cluster[-1].get("transaction_date"),
                        "end_date": current_cluster[0].get("transaction_date"),
                    }
                )

        analysis["net_buying_value"] = buying_value - selling_value
        analysis["net_selling_value"] = selling_value
        analysis["unique_insiders"] = len(insider_names)

        return analysis

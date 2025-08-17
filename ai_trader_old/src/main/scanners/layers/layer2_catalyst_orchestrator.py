# File: ai_trader/scanners/layer2_catalyst_orchestrator.py (Final Code)

# Standard library imports
import asyncio
from collections import defaultdict
from datetime import datetime
import logging
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.repositories.repository_factory import get_repository_factory
from main.data_pipeline.storage.storage_router import StorageRouter
from main.data_pipeline.validation.core.validation_pipeline import ValidationPipeline

# Import event system
from main.events.core import EventBusFactory
from main.events.handlers import ScannerFeatureBridge
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.events.types import AlertType, ScanAlert
from main.interfaces.events import IEventBus, ScannerAlertEvent

# Import scanner factory and dependencies
from main.scanners.scanner_factory_v2 import ScannerFactoryV2
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector

logger = logging.getLogger(__name__)


class Layer2CatalystOrchestrator:
    """
    Orchestrates a suite of specialized catalyst scanners, aggregates their signals,
    and produces a unified catalyst score for each symbol.
    """

    def __init__(self, config: DictConfig, db_adapter=None, event_bus: IEventBus = None):
        """
        Initializes the orchestrator and all its sub-scanners.
        Dependencies for each scanner (like analytics calculators) would be injected here.

        Args:
            config: Configuration object
            db_adapter: Optional database adapter to reuse (for testing)
            event_bus: Optional event bus for publishing layer qualification events
        """
        self.config = config
        self.min_final_score = self.config.get("scanners.layer2.min_final_score", 3.0)
        self._provided_db_adapter = db_adapter

        # Initialize dependencies
        self._initialize_dependencies()

        # Initialize all the specialist scanners using factory
        self._initialize_scanners()

        # Initialize event system components
        self.event_bus = event_bus or (
            EventBusFactory.create(self.config)
            if "event_bus" in self.config
            else EventBusFactory.create_test_instance()
        )
        self.scanner_bridge = ScannerFeatureBridge(self.event_bus)

        # Initialize event publisher and company repository
        self.event_publisher = ScannerEventPublisher(self.event_bus) if self.event_bus else None
        # Local imports
        from main.data_pipeline.storage.repositories import get_repository_factory

        repo_factory = get_repository_factory()
        self.company_repository = repo_factory.create_company_repository(self.db_adapter)

        # Initialize validation pipeline
        self.validation_pipeline = ValidationPipeline()

        logger.info(
            "Layer2CatalystOrchestrator initialized with scanner factory and validation pipeline."
        )

    async def run(self, input_universe: list[str]) -> dict[str, Any]:
        """
        Runs all configured catalyst scanners on the input universe in parallel.

        Args:
            input_universe: The list of symbols passed from Layer 1.5.

        Returns:
            A dictionary containing the qualified symbols and their catalyst details.
        """
        if not input_universe:
            return {"symbols": [], "catalyst_details": {}}

        logger.info(f"🔥 Starting Layer 2 Catalyst Scan for {len(input_universe)} symbols...")

        # Start the event bus
        await self.event_bus.start()

        # Ensure scanners are initialized
        await self._async_initialize_scanners()

        # 1. Run all sub-scanners concurrently
        tasks = [scanner.scan(input_universe) for scanner in self.scanners]
        scanner_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 2. Aggregate the signals from all scanners
        aggregated_signals = self._aggregate_scanner_results(scanner_results)

        # 3. Filter for symbols that meet the minimum final score
        qualified_symbols = {
            symbol: data
            for symbol, data in aggregated_signals.items()
            if data.get("final_score", 0) >= self.min_final_score
        }

        # 4. Sort by score and prepare final output
        sorted_symbols = sorted(
            qualified_symbols.items(), key=lambda item: item[1]["final_score"], reverse=True
        )

        # 5. Publish events for detected catalysts
        await self._publish_catalyst_events(sorted_symbols)

        final_output = {
            "symbols": [item[0] for item in sorted_symbols],
            "catalyst_details": {item[0]: item[1] for item in sorted_symbols},
        }

        logger.info(
            f"✅ Layer 2 Scan complete. Found {len(final_output['symbols'])} symbols with significant catalysts."
        )

        # 6. Update company qualifications in database
        await self._update_company_qualifications(
            qualified_symbols=[item[0] for item in sorted_symbols],
            catalyst_scores={item[0]: item[1]["final_score"] for item in sorted_symbols},
            all_input_symbols=input_universe,
        )

        # Stop the event bus
        await self.event_bus.stop()

        return final_output

    def _aggregate_scanner_results(self, scanner_results: list[Any]) -> dict[str, Any]:
        """
        Aggregates results from all sub-scanners into a single score per symbol.
        Handles both legacy dictionary format and new ScanAlert format.
        """
        aggregated = defaultdict(lambda: {"final_score": 0.0, "reasons": [], "alerts": []})

        for idx, result_set in enumerate(scanner_results):
            if isinstance(result_set, Exception):
                logger.error(f"Scanner {idx} failed: {result_set}")
                continue

            # Handle result format - scanners should return List[ScanAlert]
            if isinstance(result_set, list):
                alerts = result_set
            else:
                # If not a list, skip this result
                logger.warning(f"Scanner {idx} returned unexpected result type: {type(result_set)}")
                continue

            # Aggregate alerts by symbol
            for alert in alerts:
                # Add to the symbol's total score
                alert_score = (
                    (
                        alert.score
                        * self.config.get(
                            "orchestrator.catalyst_scoring.scaling.alert_score_multiplier", 5.0
                        )
                    )
                    if alert.score
                    else 1.0
                )  # Convert 0-1 to 0-5 scale
                aggregated[alert.symbol]["final_score"] += alert_score

                # Add reason if available
                reason = alert.metadata.get("reason") if alert.metadata else None
                if reason:
                    aggregated[alert.symbol]["reasons"].append(reason)

                # Store the alert
                aggregated[alert.symbol]["alerts"].append(alert)

                # Store detailed info by alert type
                aggregated[alert.symbol][f"{alert.alert_type}_details"] = {
                    "score": alert_score,
                    "metadata": alert.metadata,
                    "timestamp": alert.timestamp,
                }

        return dict(aggregated)

    async def _publish_catalyst_events(self, sorted_symbols: list[tuple]):
        """
        Publish events for detected catalysts to trigger feature computation.

        Args:
            sorted_symbols: List of (symbol, data) tuples sorted by score
        """
        for symbol, data in sorted_symbols:
            try:
                # Create scanner alert event
                alert_event = ScannerAlertEvent(
                    symbol=symbol,
                    alert_type="catalyst_detected",
                    score=data["final_score"]
                    / self.config.get(
                        "orchestrator.catalyst_scoring.scaling.score_normalization_divisor", 10.0
                    ),  # Normalize to 0-1
                    scanner_name="layer2_catalyst_orchestrator",
                    metadata={
                        "reasons": data.get("reasons", []),
                        "sub_scores": {
                            k: v
                            for k, v in data.items()
                            if k.endswith("_details") or k == "final_score"
                        },
                    },
                )

                # Publish to event bus
                await self.event_bus.publish(alert_event)

                # Also create ScanAlert for backward compatibility
                scan_alert = ScanAlert(
                    symbol=symbol,
                    alert_type="catalyst_detected",
                    timestamp=alert_event.timestamp,
                    score=data["final_score"] / 10.0,
                    message=f"Catalyst detected for {symbol} with score {data['final_score']:.1f}",
                    metadata=alert_event.metadata,
                    source_scanner="layer2_catalyst_orchestrator",
                )

                # Process through scanner bridge
                await self.scanner_bridge.process_scan_alert(scan_alert)

                logger.debug(
                    f"Published catalyst event for {symbol} (score: {data['final_score']})"
                )

            except Exception as e:
                logger.error(f"Failed to publish event for {symbol}: {e}")

    async def process_alerts(self, layer1_alerts: list[ScanAlert]) -> list[ScanAlert]:
        """
        Process alerts from Layer 1 scanners and generate Layer 2 catalyst alerts.

        This method takes the preliminary alerts from Layer 1 (volume, technical patterns, etc.)
        and performs deeper catalyst analysis on the symbols, generating more refined alerts.

        Args:
            layer1_alerts: List of ScanAlert objects from Layer 1 scanners

        Returns:
            List of ScanAlert objects with catalyst-based insights
        """
        if not layer1_alerts:
            return []

        logger.info(
            f"Processing {len(layer1_alerts)} Layer 1 alerts through Layer 2 catalyst analysis"
        )

        # Validate layer1_alerts before processing
        try:
            # Convert alerts to a pseudo-DataFrame for validation
            alerts_data = []
            for alert in layer1_alerts:
                alerts_data.append(
                    {
                        "symbol": alert.symbol,
                        "timestamp": alert.timestamp,
                        "alert_type": alert.alert_type,
                        "score": alert.score or 0.0,
                    }
                )

            if alerts_data:
                # Third-party imports
                import pandas as pd

                alerts_df = pd.DataFrame(alerts_data)

                validation_result = await self.validation_pipeline.validate_feature_ready(
                    data=alerts_df,
                    data_type="scanner_alerts",
                    required_columns=["symbol", "timestamp", "alert_type", "score"],
                )

                if not validation_result.passed:
                    logger.error(f"Layer 1 alerts validation failed: {validation_result.errors}")
                    return []

                if validation_result.has_warnings:
                    logger.warning(
                        f"Layer 1 alerts validation warnings: {validation_result.warnings}"
                    )

        except Exception as e:
            logger.error(f"Failed to validate Layer 1 alerts: {e}")
            # Continue processing despite validation failure

        # Extract unique symbols from Layer 1 alerts
        symbols = list(set(alert.symbol for alert in layer1_alerts))

        # Group Layer 1 alerts by symbol for context
        layer1_context = defaultdict(list)
        for alert in layer1_alerts:
            layer1_context[alert.symbol].append(
                {
                    "alert_type": alert.alert_type,
                    "score": alert.score,
                    "metadata": alert.metadata,
                    "timestamp": alert.timestamp,
                }
            )

        # Run catalyst analysis on these symbols
        catalyst_results = await self.run(symbols)

        # Convert catalyst results to ScanAlert objects
        layer2_alerts = []

        for symbol in catalyst_results["symbols"]:
            catalyst_data = catalyst_results["catalyst_details"].get(symbol, {})

            # Determine the primary catalyst type based on sub-signals
            primary_catalyst = self._determine_primary_catalyst(catalyst_data)

            # Create a unified alert
            alert = ScanAlert(
                symbol=symbol,
                alert_type=primary_catalyst,
                timestamp=datetime.now(),
                score=min(catalyst_data.get("final_score", 0) / 10.0, 1.0),  # Normalize to 0-1
                message=f"{primary_catalyst} detected for {symbol}",
                metadata={
                    "layer1_alerts": layer1_context.get(symbol, []),
                    "catalyst_reasons": catalyst_data.get("reasons", []),
                    "sub_scores": {
                        k: v
                        for k, v in catalyst_data.items()
                        if k.endswith("_details") or k == "final_score"
                    },
                    "catalyst_count": len(catalyst_data.get("reasons", [])),
                    "combined_confidence": self._calculate_combined_confidence(
                        catalyst_data, layer1_context.get(symbol, [])
                    ),
                },
                source_scanner="layer2_catalyst_orchestrator",
            )
            layer2_alerts.append(alert)

        # Also check for emerging catalysts on symbols that didn't make the main cut
        emerging_alerts = await self._check_emerging_catalysts(
            symbols, catalyst_results, layer1_context
        )
        layer2_alerts.extend(emerging_alerts)

        logger.info(f"Generated {len(layer2_alerts)} Layer 2 catalyst alerts")
        return layer2_alerts

    def _determine_primary_catalyst(self, catalyst_data: dict[str, Any]) -> str:
        """
        Determine the primary catalyst type based on sub-scanner results.

        Args:
            catalyst_data: Aggregated catalyst data for a symbol

        Returns:
            Primary AlertType constant
        """
        # Check for specific catalyst patterns in order of priority
        reasons = catalyst_data.get("reasons", [])

        # Insider activity takes precedence
        if any("insider" in reason.lower() for reason in reasons):
            if any("buying" in reason.lower() for reason in reasons):
                return AlertType.INSIDER_BUYING
            else:
                return AlertType.INSIDER_SELLING

        # Options activity
        if any(
            "options" in reason.lower() or "unusual flow" in reason.lower() for reason in reasons
        ):
            return AlertType.UNUSUAL_OPTIONS_ACTIVITY

        # News/Sentiment
        if any("news" in reason.lower() or "sentiment" in reason.lower() for reason in reasons):
            if any("breaking" in reason.lower() or "just" in reason.lower() for reason in reasons):
                return AlertType.BREAKING_NEWS
            else:
                return AlertType.SENTIMENT_SURGE

        # Sector/Market structure
        if any("sector" in reason.lower() for reason in reasons):
            return AlertType.SECTOR_ROTATION

        if any(
            "coordinated" in reason.lower() or "correlated" in reason.lower() for reason in reasons
        ):
            return AlertType.COORDINATED_ACTIVITY

        # Default to momentum if we have strong signals
        if catalyst_data.get("final_score", 0) >= self.config.get(
            "orchestrator.catalyst_scoring.thresholds.strong_signal_threshold", 5.0
        ):
            return AlertType.MOMENTUM

        # Otherwise, potential breakout
        return AlertType.BREAKOUT

    def _calculate_combined_confidence(
        self, catalyst_data: dict[str, Any], layer1_alerts: list[dict[str, Any]]
    ) -> float:
        """
        Calculate a combined confidence score based on Layer 1 and Layer 2 signals.

        Args:
            catalyst_data: Layer 2 catalyst analysis results
            layer1_alerts: Layer 1 alerts for this symbol

        Returns:
            Combined confidence score (0-1)
        """
        # Base confidence from Layer 2 catalyst score
        layer2_confidence = min(
            catalyst_data.get("final_score", 0)
            / self.config.get(
                "orchestrator.catalyst_scoring.scaling.score_normalization_divisor", 10.0
            ),
            1.0,
        )

        # Average Layer 1 scores
        if layer1_alerts:
            layer1_scores = [
                alert.get("score", 0.5) for alert in layer1_alerts if alert.get("score")
            ]
            layer1_confidence = sum(layer1_scores) / len(layer1_scores) if layer1_scores else 0.5
        else:
            layer1_confidence = 0.5

        # Weight Layer 2 more heavily (70/30 split)
        combined = (
            self.config.get("orchestrator.catalyst_scoring.weighting.layer2_confidence_weight", 0.7)
            * layer2_confidence
        ) + (
            self.config.get("orchestrator.catalyst_scoring.weighting.layer1_confidence_weight", 0.3)
            * layer1_confidence
        )

        # Boost for multiple confirming signals
        signal_count = len(catalyst_data.get("reasons", [])) + len(layer1_alerts)
        if signal_count >= self.config.get(
            "orchestrator.catalyst_scoring.thresholds.signal_boost_threshold_high", 5
        ):
            combined = min(
                combined
                * self.config.get(
                    "orchestrator.catalyst_scoring.weighting.high_signal_boost_factor", 1.2
                ),
                1.0,
            )
        elif signal_count >= self.config.get(
            "orchestrator.catalyst_scoring.thresholds.signal_boost_threshold_medium", 3
        ):
            combined = min(
                combined
                * self.config.get(
                    "orchestrator.catalyst_scoring.weighting.medium_signal_boost_factor", 1.1
                ),
                1.0,
            )

        return round(combined, 3)

    async def _check_emerging_catalysts(
        self,
        symbols: list[str],
        catalyst_results: dict[str, Any],
        layer1_context: dict[str, list[dict[str, Any]]],
    ) -> list[ScanAlert]:
        """
        Check for emerging catalysts on symbols that didn't meet the main threshold.

        This helps identify potential opportunities that are building but not yet confirmed.

        Args:
            symbols: All symbols analyzed
            catalyst_results: Results from catalyst analysis
            layer1_context: Layer 1 alerts grouped by symbol

        Returns:
            List of emerging catalyst alerts
        """
        emerging_alerts = []
        qualified_symbols = set(catalyst_results["symbols"])

        # Check symbols that had Layer 1 alerts but didn't qualify for main catalysts
        for symbol in symbols:
            if symbol in qualified_symbols:
                continue

            catalyst_data = catalyst_results.get("catalyst_details", {}).get(symbol, {})
            layer1_alerts = layer1_context.get(symbol, [])

            # Look for emerging patterns
            if catalyst_data.get("final_score", 0) >= self.config.get(
                "orchestrator.catalyst_scoring.thresholds.emerging_catalyst_threshold", 1.5
            ) and len(  # Some catalyst activity
                layer1_alerts
            ) >= self.config.get(
                "orchestrator.catalyst_scoring.thresholds.min_layer1_signals", 2
            ):  # Multiple Layer 1 signals

                alert = ScanAlert(
                    symbol=symbol,
                    alert_type=AlertType.CONSOLIDATION,  # Emerging catalyst
                    timestamp=datetime.now(),
                    score=self.config.get(
                        "orchestrator.catalyst_scoring.scaling.emerging_score_base", 0.4
                    )
                    + (
                        catalyst_data.get("final_score", 0)
                        / self.config.get(
                            "orchestrator.catalyst_scoring.scaling.emerging_score_divisor", 20.0
                        )
                    ),  # Lower scores
                    message=f"Emerging catalyst pattern detected for {symbol}",
                    metadata={
                        "alert_subtype": "emerging_catalyst",
                        "layer1_alerts": layer1_alerts,
                        "catalyst_score": catalyst_data.get("final_score", 0),
                        "reasons": catalyst_data.get("reasons", []),
                        "watch_list": True,
                    },
                    source_scanner="layer2_catalyst_orchestrator",
                )
                emerging_alerts.append(alert)

        return emerging_alerts

    def _initialize_dependencies(self):
        """Initialize core dependencies for scanners."""
        # Check if db_adapter was provided
        if self._provided_db_adapter:
            self.db_adapter = self._provided_db_adapter
            logger.info("Using provided database adapter")
        else:
            # Create database adapter
            logger.info("Creating database adapter for scanners...")
            db_factory = DatabaseFactory()
            self.db_adapter = db_factory.create_async_database(self.config)
            logger.info("Database adapter created")

        # Create storage router and archive
        logger.info("Creating storage router...")
        self.storage_router = StorageRouter(self.config)
        logger.info("Creating data archive...")
        self.archive = DataArchive(self.config.get("storage.archive", {}))
        logger.info("Archive created")

        # Create metrics and cache managers
        self.metrics_collector = (
            ScannerMetricsCollector()
            if self.config.get("scanners.global.enable_metrics", True)
            else None
        )
        self.cache_manager = (
            ScannerCacheManager(
                default_ttl_seconds=self.config.get("scanners.global.cache_ttl_seconds", 60)
            )
            if self.config.get("scanners.global.enable_caching", True)
            else None
        )

        # Create repository factory
        self.repo_factory = get_repository_factory(
            db_adapter=self.db_adapter,
            cold_storage=self.archive,
            event_bus=None,  # Will use scanner factory's event bus
            config=self.config,
        )

    def _initialize_scanners(self):
        """Initialize all catalyst scanners using the factory."""
        # Create scanner factory with dependencies
        self.scanner_factory = ScannerFactoryV2(
            config=self.config,
            db_adapter=self.db_adapter,
            event_bus=None,  # Will create event bus later
            metrics_collector=self.metrics_collector,
            cache_manager=self.cache_manager,
        )

        # Define catalyst scanner types to initialize
        catalyst_types = [
            "news",
            "technical",
            "volume",
            "earnings",
            "insider",
            "options",
            "social",
            "sector",
            "intermarket",
            "market_validation",
            "coordinated_activity",
            "advanced_sentiment",
        ]

        # Initialize scanners - will be populated asynchronously
        self.scanners = []
        self.catalyst_types = catalyst_types

        logger.info(f"Scanner factory initialized, will create {len(catalyst_types)} scanners")

    async def _async_initialize_scanners(self):
        """Asynchronously initialize all scanners."""
        if self.scanners:  # Already initialized
            return

        for scanner_type in self.catalyst_types:
            # Check if scanner is enabled in config
            scanner_config = self.config.get(f"scanners.{scanner_type}", {})
            if scanner_config.get("enabled", True):
                try:
                    scanner = await self.scanner_factory.create_scanner(scanner_type)
                    self.scanners.append(scanner)
                    logger.info(f"Initialized {scanner_type} scanner")
                except Exception as e:
                    logger.warning(f"Failed to create {scanner_type} scanner: {e}")

        logger.info(f"Successfully initialized {len(self.scanners)} scanners")

        # Sort scanners by priority if configured
        if self.scanners and self.config.get("scanners.global.use_priority_ordering", True):
            self.scanners.sort(
                key=lambda s: self.config.get(
                    f'scanners.{s.name.lower().replace("scanner", "")}.priority', 5
                ),
                reverse=True,
            )

    async def cleanup(self):
        """Clean up resources."""
        # Clean up scanner factory
        if hasattr(self, "scanner_factory"):
            await self.scanner_factory.cleanup()

        # Clean up database adapter
        if hasattr(self, "db_adapter"):
            await self.db_adapter.close()

    async def _update_company_qualifications(
        self,
        qualified_symbols: list[str],
        catalyst_scores: dict[str, float],
        all_input_symbols: list[str],
    ):
        """
        Update company layer status to Layer 2 (CATALYST) in database.

        Args:
            qualified_symbols: Symbols that passed Layer 2
            catalyst_scores: Dict mapping symbol to catalyst score
            all_input_symbols: All symbols that were evaluated
        """
        try:
            # Update qualified symbols to Layer 2
            qualified_count = 0
            promoted_count = 0

            for symbol in qualified_symbols:
                # Get current layer for the symbol
                current = await self.company_repository.get_by_symbol(symbol)
                if current:
                    current_layer = current.get("layer", 0)

                    # Update to Layer 2 (CATALYST)
                    result = await self.company_repository.update_layer(
                        symbol=symbol,
                        layer=DataLayer.CATALYST,
                        metadata={
                            "catalyst_score": catalyst_scores.get(symbol, 0.0),
                            "source": "layer2_scanner",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                    if result.success:
                        qualified_count += 1

                        # Publish event based on whether this is qualification or promotion
                        if self.event_publisher:
                            if current_layer < 2:
                                # This is a promotion to Layer 2
                                await self.event_publisher.publish_symbol_promoted(
                                    symbol=symbol,
                                    from_layer=DataLayer(current_layer),
                                    to_layer=DataLayer.CATALYST,
                                    promotion_reason="Symbol has active catalysts",
                                    metrics={
                                        "catalyst_score": catalyst_scores.get(symbol, 0.0),
                                        "catalyst_count": 1,  # Can be enhanced to track actual catalyst count
                                    },
                                )
                                promoted_count += 1
                            else:
                                # Re-qualification at same layer
                                await self.event_publisher.publish_symbol_qualified(
                                    symbol=symbol,
                                    layer=DataLayer.CATALYST,
                                    qualification_reason="Symbol has active catalysts",
                                    metrics={"catalyst_score": catalyst_scores.get(symbol, 0.0)},
                                )
                    else:
                        logger.warning(f"Failed to update layer for {symbol}: {result.errors}")

            # Optionally downgrade non-qualified symbols
            non_qualified = set(all_input_symbols) - set(qualified_symbols)
            for symbol in non_qualified:
                current = await self.company_repository.get_by_symbol(symbol)
                if current and current.get("layer", 0) == 2:
                    # Downgrade from Layer 2 if no longer qualified
                    await self.company_repository.update_layer(
                        symbol=symbol,
                        layer=DataLayer.LIQUID,  # Downgrade to Layer 1
                        metadata={"reason": "No active catalysts detected"},
                    )

            logger.info(
                f"✅ Updated Layer 2 qualifications: "
                f"{qualified_count} qualified ({promoted_count} promoted), "
                f"{len(non_qualified)} non-qualified"
            )

        except Exception as e:
            logger.error(f"Error updating company qualifications: {e}", exc_info=True)
            # Don't fail the entire scan if qualification update fails
            # The scan results are still valid

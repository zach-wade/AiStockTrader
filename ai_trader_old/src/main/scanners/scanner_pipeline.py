"""
Scanner Pipeline Orchestrator

Orchestrates the execution of all scanner layers from Layer 0 to Layer 3,
providing a complete end-to-end scanning workflow.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.config.config_manager import get_config

# Infrastructure imports
from main.data_pipeline.storage.database_factory import DatabaseFactory

# Removed HistoricalManager import - not needed for scanner pipeline
# Layer imports
from main.scanners.layers.layer0_static_universe import Layer0StaticUniverseScanner
from main.scanners.layers.layer1_5_strategy_affinity import Layer1_5_StrategyAffinityFilter
from main.scanners.layers.layer1_liquidity_filter import Layer1LiquidityFilter
from main.scanners.layers.layer2_catalyst_orchestrator import Layer2CatalystOrchestrator
from main.scanners.layers.layer3_premarket_scanner import Layer3PreMarketScanner
from main.scanners.layers.layer3_realtime_scanner import Layer3RealtimeScanner
from main.utils.resilience import get_global_recovery_manager

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    """Result from a scanner layer execution."""

    layer_name: str
    layer_number: str
    input_count: int
    output_count: int
    symbols: list[str]
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    start_time: datetime
    end_time: datetime
    total_duration: float
    layer_results: list[LayerResult]
    final_symbols: list[str]
    final_opportunities: list[dict[str, Any]]
    success: bool
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScannerPipeline:
    """
    Orchestrates the complete scanner pipeline from Layer 0 to Layer 3.

    Features:
    - Sequential layer execution with data flow
    - Comprehensive error handling
    - Performance tracking
    - Result aggregation and reporting
    - Test mode support
    """

    def __init__(self, config: DictConfig | None = None, test_mode: bool = False):
        """
        Initialize the scanner pipeline.

        Args:
            config: Configuration object. If None, loads from default config.
            test_mode: If True, runs in test mode with smaller datasets
        """
        self.config = config or get_config()
        self.test_mode = test_mode

        # Initialize database
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(self.config)

        # Initialize components
        self.recovery_manager = get_global_recovery_manager()
        # Note: HistoricalManager initialization has been removed as it requires
        # many dependencies not needed for the scanner pipeline

        # Pipeline configuration
        pipeline_config = self.config.get("scanner_pipeline", {})
        self.save_intermediate_results = pipeline_config.get("save_intermediate_results", True)
        self.output_dir = Path(pipeline_config.get("output_dir", "data/scanner_pipeline"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test mode configuration
        if self.test_mode:
            self.test_symbols_limit = pipeline_config.get("test_mode", {}).get("symbols_limit", 100)
            logger.info(
                f"Running in TEST MODE - limiting to {self.test_symbols_limit} symbols per layer"
            )

        # Layer results tracking
        self.layer_results = []

        logger.info("Scanner Pipeline initialized")

    async def run(self) -> PipelineResult:
        """
        Run the complete scanner pipeline.

        Returns:
            PipelineResult with comprehensive execution details
        """
        logger.info("=" * 80)
        logger.info("Starting Scanner Pipeline Execution")
        logger.info("=" * 80)

        start_time = datetime.now(UTC)
        errors = []
        final_symbols = []
        final_opportunities = []

        try:
            # Layer 0: Static Universe
            layer0_result = await self._run_layer0()
            self.layer_results.append(layer0_result)

            if not layer0_result.symbols:
                raise ValueError("Layer 0 returned no symbols")

            # Layer 1: Liquidity Filter
            layer1_result = await self._run_layer1(layer0_result.symbols)
            self.layer_results.append(layer1_result)

            if not layer1_result.symbols:
                raise ValueError("Layer 1 returned no symbols")

            # Layer 1.5: Strategy Affinity
            layer1_5_result = await self._run_layer1_5(layer1_result.symbols)
            self.layer_results.append(layer1_5_result)

            if not layer1_5_result.symbols:
                logger.warning("Layer 1.5 returned no symbols - using Layer 1 output")
                layer1_5_result.symbols = layer1_result.symbols[:500]  # Fallback

            # Layer 2: Catalyst Orchestrator
            layer2_result = await self._run_layer2(layer1_5_result.symbols)
            self.layer_results.append(layer2_result)

            if not layer2_result.symbols:
                logger.warning("Layer 2 returned no symbols with catalysts")

            # Layer 3: Real-time Scanner (Pre-market or Real-time based on time)
            layer3_result = await self._run_layer3(layer2_result.symbols)
            self.layer_results.append(layer3_result)

            final_symbols = layer3_result.symbols
            final_opportunities = layer3_result.metadata.get("opportunities", [])

            # Generate pipeline report
            end_time = datetime.now(UTC)
            total_duration = (end_time - start_time).total_seconds()

            result = PipelineResult(
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                layer_results=self.layer_results,
                final_symbols=final_symbols,
                final_opportunities=final_opportunities,
                success=True,
                errors=errors,
                metadata={
                    "test_mode": self.test_mode,
                    "market_hours": self._get_market_status(),
                    "funnel_reduction": self._calculate_funnel_metrics(),
                },
            )

            # Save results
            await self._save_results(result)

            # Log summary
            self._log_summary(result)

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            errors.append(str(e))

            end_time = datetime.now(UTC)
            return PipelineResult(
                start_time=start_time,
                end_time=end_time,
                total_duration=(end_time - start_time).total_seconds(),
                layer_results=self.layer_results,
                final_symbols=final_symbols,
                final_opportunities=final_opportunities,
                success=False,
                errors=errors,
            )

    async def _run_layer0(self) -> LayerResult:
        """Run Layer 0: Static Universe Scanner."""
        logger.info("\n" + "-" * 60)
        logger.info("Running Layer 0: Static Universe Scanner")
        logger.info("-" * 60)

        start_time = datetime.now(UTC)
        errors = []

        try:
            scanner = Layer0StaticUniverseScanner(self.config)
            symbols = await scanner.run()

            # Apply test mode limit
            if self.test_mode and len(symbols) > self.test_symbols_limit:
                logger.info(
                    f"Test mode: Limiting Layer 0 output from {len(symbols)} to {self.test_symbols_limit}"
                )
                symbols = symbols[: self.test_symbols_limit]

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return LayerResult(
                layer_name="Static Universe Scanner",
                layer_number="0",
                input_count=0,  # No input for Layer 0
                output_count=len(symbols),
                symbols=symbols,
                execution_time=execution_time,
                metadata={
                    "excluded_exchanges": scanner.excluded_exchanges,
                    "excluded_patterns": scanner.excluded_patterns,
                },
            )

        except Exception as e:
            logger.error(f"Layer 0 failed: {e}")
            errors.append(str(e))

            return LayerResult(
                layer_name="Static Universe Scanner",
                layer_number="0",
                input_count=0,
                output_count=0,
                symbols=[],
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                errors=errors,
            )

    async def _run_layer1(self, input_symbols: list[str]) -> LayerResult:
        """Run Layer 1: Liquidity Filter."""
        logger.info("\n" + "-" * 60)
        logger.info("Running Layer 1: Liquidity Filter")
        logger.info("-" * 60)

        start_time = datetime.now(UTC)
        errors = []

        try:
            # Initialize and run Layer 1 liquidity filter
            scanner = Layer1LiquidityFilter(self.config, self.db_adapter)
            symbols = await scanner.run(input_symbols)

            # Apply test mode limit
            if self.test_mode and len(symbols) > self.test_symbols_limit:
                logger.info(
                    f"Test mode: Limiting Layer 1 output from {len(symbols)} to {self.test_symbols_limit}"
                )
                symbols = symbols[: self.test_symbols_limit]

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return LayerResult(
                layer_name="Liquidity Filter",
                layer_number="1",
                input_count=len(input_symbols),
                output_count=len(symbols),
                symbols=symbols,
                execution_time=execution_time,
                metadata={
                    "min_avg_dollar_volume": scanner.min_avg_dollar_volume,
                    "min_price": scanner.min_price,
                    "max_price": scanner.max_price,
                    "lookback_days": scanner.lookback_days,
                },
            )

        except Exception as e:
            logger.error(f"Layer 1 failed: {e}")
            errors.append(str(e))

            # Fallback: return top symbols by default
            fallback_symbols = input_symbols[: min(1500, len(input_symbols))]

            return LayerResult(
                layer_name="Liquidity Filter",
                layer_number="1",
                input_count=len(input_symbols),
                output_count=len(fallback_symbols),
                symbols=fallback_symbols,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                errors=errors,
            )

    async def _run_layer1_5(self, input_symbols: list[str]) -> LayerResult:
        """Run Layer 1.5: Strategy Affinity Filter."""
        logger.info("\n" + "-" * 60)
        logger.info("Running Layer 1.5: Strategy Affinity Filter")
        logger.info("-" * 60)

        start_time = datetime.now(UTC)
        errors = []

        try:
            scanner = Layer1_5_StrategyAffinityFilter(self.config, self.db_adapter)
            symbols = await scanner.run(input_symbols)

            # Apply test mode limit
            if self.test_mode and len(symbols) > self.test_symbols_limit:
                logger.info(
                    f"Test mode: Limiting Layer 1.5 output from {len(symbols)} to {self.test_symbols_limit}"
                )
                symbols = symbols[: self.test_symbols_limit]

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Get current market regime
            current_regime = await scanner._get_current_market_regime()

            return LayerResult(
                layer_name="Strategy Affinity Filter",
                layer_number="1.5",
                input_count=len(input_symbols),
                output_count=len(symbols),
                symbols=symbols,
                execution_time=execution_time,
                metadata={
                    "market_regime": current_regime,
                    "min_affinity_score": scanner.min_affinity_score,
                    "regime_filtering_enabled": scanner.enable_regime_filtering,
                },
            )

        except Exception as e:
            logger.error(f"Layer 1.5 failed: {e}")
            errors.append(str(e))

            # Fallback: return top symbols
            fallback_symbols = input_symbols[: min(500, len(input_symbols))]

            return LayerResult(
                layer_name="Strategy Affinity Filter",
                layer_number="1.5",
                input_count=len(input_symbols),
                output_count=len(fallback_symbols),
                symbols=fallback_symbols,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                errors=errors,
            )

    async def _run_layer2(self, input_symbols: list[str]) -> LayerResult:
        """Run Layer 2: Catalyst Orchestrator."""
        logger.info("\n" + "-" * 60)
        logger.info("Running Layer 2: Catalyst Orchestrator")
        logger.info("-" * 60)

        start_time = datetime.now(UTC)
        errors = []

        try:
            scanner = Layer2CatalystOrchestrator(self.config)

            # Initialize sub-scanners based on configuration
            await self._initialize_layer2_scanners(scanner)

            result = await scanner.run(input_symbols)

            symbols = result.get("symbols", [])
            catalyst_details = result.get("catalyst_details", {})

            # Apply test mode limit
            if self.test_mode and len(symbols) > self.test_symbols_limit:
                logger.info(
                    f"Test mode: Limiting Layer 2 output from {len(symbols)} to {self.test_symbols_limit}"
                )
                symbols = symbols[: self.test_symbols_limit]

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return LayerResult(
                layer_name="Catalyst Orchestrator",
                layer_number="2",
                input_count=len(input_symbols),
                output_count=len(symbols),
                symbols=symbols,
                execution_time=execution_time,
                metadata={
                    "catalyst_details": catalyst_details,
                    "sub_scanner_count": len(scanner.scanners),
                    "min_final_score": scanner.min_final_score,
                },
            )

        except Exception as e:
            logger.error(f"Layer 2 failed: {e}")
            errors.append(str(e))

            # Fallback: return top symbols
            fallback_symbols = input_symbols[: min(200, len(input_symbols))]

            return LayerResult(
                layer_name="Catalyst Orchestrator",
                layer_number="2",
                input_count=len(input_symbols),
                output_count=len(fallback_symbols),
                symbols=fallback_symbols,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                errors=errors,
            )

    async def _run_layer3(self, input_symbols: list[str]) -> LayerResult:
        """Run Layer 3: Real-time Scanner (Pre-market or Real-time based on time)."""
        logger.info("\n" + "-" * 60)

        # Determine which Layer 3 scanner to use
        is_premarket = self._is_premarket_hours()
        scanner_type = "Pre-Market" if is_premarket else "Real-time"

        logger.info(f"Running Layer 3: {scanner_type} Scanner")
        logger.info("-" * 60)

        start_time = datetime.now(UTC)
        errors = []

        try:
            if is_premarket:
                result = await self._run_premarket_scanner(input_symbols)
            else:
                result = await self._run_realtime_scanner(input_symbols)

            return result

        except Exception as e:
            logger.error(f"Layer 3 failed: {e}")
            errors.append(str(e))

            # Fallback: return top symbols
            fallback_symbols = input_symbols[: min(30, len(input_symbols))]

            return LayerResult(
                layer_name=f"{scanner_type} Scanner",
                layer_number="3",
                input_count=len(input_symbols),
                output_count=len(fallback_symbols),
                symbols=fallback_symbols,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                errors=errors,
            )

    async def _run_premarket_scanner(self, input_symbols: list[str]) -> LayerResult:
        """Run the pre-market scanner."""
        start_time = datetime.now(UTC)

        scanner = Layer3PreMarketScanner(self.config)
        result = await scanner.scan_premarket(input_symbols)

        # Extract top opportunities
        top_opportunities = result.get("top_opportunities", [])
        symbols = [opp["symbol"] for opp in top_opportunities]

        # Apply test mode limit
        if self.test_mode and len(symbols) > 30:
            logger.info(f"Test mode: Limiting Layer 3 output from {len(symbols)} to 30")
            symbols = symbols[:30]
            top_opportunities = top_opportunities[:30]

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        return LayerResult(
            layer_name="Pre-Market Scanner",
            layer_number="3",
            input_count=len(input_symbols),
            output_count=len(symbols),
            symbols=symbols,
            execution_time=execution_time,
            metadata={
                "opportunities": top_opportunities,
                "statistics": result.get("statistics", {}),
                "market_time": result.get("market_time", "unknown"),
            },
        )

    async def _run_realtime_scanner(self, input_symbols: list[str]) -> LayerResult:
        """Run the real-time scanner."""
        start_time = datetime.now(UTC)

        scanner = Layer3RealtimeScanner(self.config)
        await scanner.initialize()

        opportunities = await scanner.scan(input_symbols)

        # Convert opportunities to dict format
        opp_dicts = []
        for opp in opportunities:
            opp_dicts.append(
                {
                    "symbol": opp.symbol,
                    "score": opp.score,
                    "rvol": opp.rvol,
                    "price_change_pct": opp.price_change_pct,
                    "current_price": opp.current_price,
                    "volume": opp.volume,
                    "spread_bps": opp.spread_bps,
                    "momentum_score": opp.momentum_score,
                    "catalyst_score": opp.catalyst_score,
                }
            )

        symbols = [opp.symbol for opp in opportunities]

        # Apply test mode limit
        if self.test_mode and len(symbols) > 30:
            logger.info(f"Test mode: Limiting Layer 3 output from {len(symbols)} to 30")
            symbols = symbols[:30]
            opp_dicts = opp_dicts[:30]

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        # Get scanner stats
        stats = await scanner.get_stats()

        # Clean up
        await scanner.close()

        return LayerResult(
            layer_name="Real-time Scanner",
            layer_number="3",
            input_count=len(input_symbols),
            output_count=len(symbols),
            symbols=symbols,
            execution_time=execution_time,
            metadata={
                "opportunities": opp_dicts,
                "scanner_stats": stats,
                "use_websocket": scanner.use_websocket,
            },
        )

    async def _initialize_layer2_scanners(self, orchestrator: Layer2CatalystOrchestrator):
        """Initialize Layer 2 sub-scanners."""
        # Import scanner implementations

        # Import scanner factory V2 (clean architecture)
        # Local imports
        from main.scanners.scanner_factory_v2 import ScannerFactoryV2

        # Create scanner factory with clean dependencies
        scanner_factory = ScannerFactoryV2(
            config=self.config, db_adapter=self.db_adapter, event_bus=orchestrator.event_bus
        )

        # Initialize scanners based on configuration
        scanner_config = self.config.get("scanners", {})

        enabled_scanners = []

        # Add scanners that are enabled in config
        if scanner_config.get("volume", {}).get("enabled", True):
            scanner = await scanner_factory.create_scanner("volume")
            enabled_scanners.append(scanner)

        if scanner_config.get("technical", {}).get("enabled", True):
            scanner = await scanner_factory.create_scanner("technical")
            enabled_scanners.append(scanner)

        if scanner_config.get("news", {}).get("enabled", True):
            scanner = await scanner_factory.create_scanner("news")
            enabled_scanners.append(scanner)

        if scanner_config.get("earnings", {}).get("enabled", True):
            scanner = await scanner_factory.create_scanner("earnings")
            enabled_scanners.append(scanner)

        if scanner_config.get("social", {}).get("enabled", True):
            scanner = await scanner_factory.create_scanner("social")
            enabled_scanners.append(scanner)

        # Set the scanners
        orchestrator.scanners = enabled_scanners

        logger.info(f"Initialized {len(enabled_scanners)} sub-scanners for Layer 2")

    def _is_premarket_hours(self) -> bool:
        """Check if currently in pre-market hours."""
        now = datetime.now(UTC)
        et_offset = timedelta(hours=-5)  # ET timezone
        et_now = now + et_offset

        # Pre-market: 4:00 AM - 9:30 AM ET
        hour = et_now.hour
        minute = et_now.minute

        if hour == 4 and minute >= 0 or 5 <= hour <= 8 or hour == 9 and minute < 30:
            return True

        return False

    def _get_market_status(self) -> str:
        """Get current market status."""
        if self._is_premarket_hours():
            return "pre-market"

        now = datetime.now(UTC)
        et_offset = timedelta(hours=-5)
        et_now = now + et_offset
        hour = et_now.hour
        minute = et_now.minute

        # Market hours: 9:30 AM - 4:00 PM ET
        if hour == 9 and minute >= 30 or 10 <= hour <= 15 or hour == 16 and minute == 0:
            return "market-hours"
        elif hour == 16 and minute > 0:
            return "after-hours"
        elif hour > 16 or hour < 4:
            return "closed"

        return "closed"

    def _calculate_funnel_metrics(self) -> dict[str, Any]:
        """Calculate funnel reduction metrics."""
        metrics = {"layer_reductions": [], "total_reduction": 0.0, "final_selection_rate": 0.0}

        if not self.layer_results:
            return metrics

        # Calculate reduction at each layer
        for i, result in enumerate(self.layer_results):
            if i == 0:
                # Layer 0 has no input
                reduction = 0.0
            else:
                prev_count = self.layer_results[i - 1].output_count
                if prev_count > 0:
                    reduction = 1 - (result.output_count / prev_count)
                else:
                    reduction = 0.0

            metrics["layer_reductions"].append(
                {
                    "layer": result.layer_number,
                    "name": result.layer_name,
                    "input": result.input_count,
                    "output": result.output_count,
                    "reduction_rate": reduction,
                }
            )

        # Calculate total reduction
        if self.layer_results[0].output_count > 0:
            metrics["total_reduction"] = 1 - (
                self.layer_results[-1].output_count / self.layer_results[0].output_count
            )
            metrics["final_selection_rate"] = (
                self.layer_results[-1].output_count / self.layer_results[0].output_count
            )

        return metrics

    async def _save_results(self, result: PipelineResult):
        """Save pipeline results to file."""
        if not self.save_intermediate_results:
            return

        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")

        # Convert result to dict
        result_dict = {
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_duration": result.total_duration,
            "success": result.success,
            "errors": result.errors,
            "metadata": result.metadata,
            "final_symbols": result.final_symbols,
            "final_opportunities": result.final_opportunities,
            "layer_results": [],
        }

        # Add layer results
        for layer in result.layer_results:
            result_dict["layer_results"].append(
                {
                    "layer_name": layer.layer_name,
                    "layer_number": layer.layer_number,
                    "input_count": layer.input_count,
                    "output_count": layer.output_count,
                    "execution_time": layer.execution_time,
                    "metadata": layer.metadata,
                    "errors": layer.errors,
                    "symbols": (
                        layer.symbols[:10] if self.test_mode else layer.symbols
                    ),  # Limit symbols in test mode
                }
            )

        # Save to JSON
        output_file = self.output_dir / f"pipeline_result_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

        # Also save a summary file
        summary_file = self.output_dir / f"pipeline_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(self._generate_summary_text(result))

        logger.info(f"Summary saved to: {summary_file}")

    def _generate_summary_text(self, result: PipelineResult) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("Scanner Pipeline Execution Summary")
        lines.append("=" * 80)
        lines.append(f"Start Time: {result.start_time}")
        lines.append(f"End Time: {result.end_time}")
        lines.append(f"Total Duration: {result.total_duration:.2f} seconds")
        lines.append(f"Success: {result.success}")
        lines.append(f"Test Mode: {result.metadata.get('test_mode', False)}")
        lines.append(f"Market Status: {result.metadata.get('market_hours', 'unknown')}")
        lines.append("")

        lines.append("Layer Results:")
        lines.append("-" * 80)

        for layer in result.layer_results:
            lines.append(f"\nLayer {layer.layer_number}: {layer.layer_name}")
            lines.append(f"  Input: {layer.input_count:,} symbols")
            lines.append(f"  Output: {layer.output_count:,} symbols")
            lines.append(f"  Execution Time: {layer.execution_time:.2f}s")

            if layer.input_count > 0:
                reduction = (1 - layer.output_count / layer.input_count) * 100
                lines.append(f"  Reduction: {reduction:.1f}%")

            if layer.errors:
                lines.append(f"  Errors: {', '.join(layer.errors)}")

        lines.append("")
        lines.append("Funnel Metrics:")
        lines.append("-" * 80)

        funnel = result.metadata.get("funnel_reduction", {})
        lines.append(f"Total Reduction: {funnel.get('total_reduction', 0) * 100:.1f}%")
        lines.append(f"Final Selection Rate: {funnel.get('final_selection_rate', 0) * 100:.3f}%")

        lines.append("")
        lines.append(f"Final Output: {len(result.final_symbols)} symbols")

        if result.final_opportunities:
            lines.append("")
            lines.append("Top 10 Opportunities:")
            lines.append("-" * 80)

            for i, opp in enumerate(result.final_opportunities[:10], 1):
                symbol = opp.get("symbol", "N/A")
                score = opp.get("score", 0)
                rvol = opp.get("rvol", 0)
                price_change = opp.get("price_change_pct", 0)

                lines.append(
                    f"{i:2d}. {symbol:<6} Score: {score:6.2f} "
                    f"RVOL: {rvol:5.1f}x Price: {price_change:+6.2f}%"
                )

        if result.errors:
            lines.append("")
            lines.append("Errors:")
            lines.append("-" * 80)
            for error in result.errors:
                lines.append(f"- {error}")

        return "\n".join(lines)

    def _log_summary(self, result: PipelineResult):
        """Log execution summary."""
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Execution Complete")
        logger.info("=" * 80)

        logger.info(f"Total Duration: {result.total_duration:.2f} seconds")
        logger.info(f"Success: {result.success}")

        # Log funnel progression
        logger.info("\nFunnel Progression:")
        for layer in result.layer_results:
            logger.info(
                f"  Layer {layer.layer_number}: {layer.input_count:,} → "
                f"{layer.output_count:,} symbols ({layer.layer_name})"
            )

        # Log final results
        logger.info(f"\nFinal Output: {len(result.final_symbols)} trading opportunities")

        if result.final_opportunities:
            logger.info("\nTop 5 Opportunities:")
            for i, opp in enumerate(result.final_opportunities[:5], 1):
                logger.info(
                    f"  {i}. {opp.get('symbol', 'N/A')}: "
                    f"Score={opp.get('score', 0):.2f}, "
                    f"RVOL={opp.get('rvol', 0):.1f}x"
                )


async def main():
    """Main entry point for testing the scanner pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run in test mode by default
    pipeline = ScannerPipeline(test_mode=True)
    result = await pipeline.run()

    if result.success:
        logger.info("\nPipeline completed successfully!")
    else:
        logger.error("\nPipeline failed!")
        for error in result.errors:
            logger.error(f"  - {error}")

    return result


if __name__ == "__main__":
    asyncio.run(main())

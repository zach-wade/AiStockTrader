#!/usr/bin/env python3
"""
Event-Driven Trading Engine

Orchestrates all real-time, event-driven strategies. Connects to data streams,
enriches events, and dispatches them to the appropriate strategy handlers.
"""

# Standard library imports
import asyncio
from datetime import datetime
import sys
from typing import Any

# Local imports
from main.data_pipeline.sources.news_client import NewsClient
from main.models.event_driven.base_event_strategy import BaseEventStrategy
from main.models.event_driven.news_analytics import NewsAnalyticsStrategy
from main.models.hft.microstructure_alpha import MicrostructureAlphaStrategy
from main.utils import (
    CLIAppConfig,
    ErrorHandlingMixin,
    StandardCLIHandler,
    create_event_driven_app,
    ensure_utc,
    get_logger,
    managed_app_context,
    record_metric,
    setup_logging,
    timer,
)
from main.utils.resilience import (
    NETWORK_RETRY_CONFIG,
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorRecoveryManager,
    RetryConfig,
    get_circuit_breaker,
)

# from main.execution.execution_client import ExecutionClient

# Setup standardized logging
setup_logging()
logger = get_logger(__name__)


class EventDrivenEngine(ErrorHandlingMixin):
    """
    Orchestrates all real-time, event-driven strategies with standardized error handling.

    Connects to data streams, enriches events, and dispatches them to the appropriate
    strategy handlers with proper monitoring and error recovery.
    """

    def __init__(self, config: dict):
        """
        Initialize EventDrivenEngine with standardized error handling.

        Args:
            config: System configuration dictionary
        """
        super().__init__()
        self.config = config
        self.engine_config = config.get("event_driven_engine", {})
        self.logger = get_logger(f"{__name__}.EventDrivenEngine")

        # Engine state tracking
        self.is_running = False
        self.startup_time = datetime.now()
        self.events_processed = 0
        self.active_tasks: list[asyncio.Task] = []

        # Initialize data clients with error handling
        self.news_client = None
        self.strategies: list[BaseEventStrategy] = []

        # Initialize resilience components
        self.circuit_breakers = {}
        self.stream_recovery_manager = ErrorRecoveryManager(NETWORK_RETRY_CONFIG)
        self.connection_recovery_manager = ErrorRecoveryManager(
            RetryConfig(
                max_attempts=5, base_delay=2.0, max_delay=30.0, strategy="exponential_backoff"
            )
        )

        # Register error callback for monitoring
        self.register_error_callback(
            "engine_monitoring",
            lambda error, context: record_metric(
                "event_engine_error",
                1,
                tags={"context": context, "error_type": type(error).__name__},
            ),
        )

        # Initialize components
        self._initialize_clients()
        self._initialize_strategies()

    def _initialize_clients(self):
        """Initialize data clients with error handling."""
        try:
            self.news_client = NewsClient(self.config)
            # self.orderbook_client = OrderBookClient(self.config) # For HFT strategies
            # self.execution_client = ExecutionClient(self.config)
            self.logger.info("Data clients initialized successfully")
        except Exception as e:
            self.handle_error(e, "initializing data clients")
            raise

    def _initialize_strategies(self):
        """Initialize all event-driven strategies defined in the config with error handling."""
        try:
            enabled_strategies = self.engine_config.get("enabled_strategies", {})

            if enabled_strategies.get("news_analytics"):
                strat_config = self.config.get("strategies", {}).get("news_analytics", {})
                strategy = NewsAnalyticsStrategy(self.config, strat_config)
                self.strategies.append(strategy)
                self.logger.info("NewsAnalyticsStrategy loaded into Event-Driven Engine")

            if enabled_strategies.get("microstructure_alpha"):
                strat_config = self.config.get("strategies", {}).get("microstructure_alpha", {})
                strategy = MicrostructureAlphaStrategy(self.config, strat_config)
                self.strategies.append(strategy)
                self.logger.info("MicrostructureAlphaStrategy loaded into Event-Driven Engine")

            self.logger.info(f"Initialized {len(self.strategies)} event-driven strategies")

        except Exception as e:
            self.handle_error(e, "initializing strategies")
            raise

    async def _get_circuit_breaker(self, operation_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for specific operation type."""
        if operation_type not in self.circuit_breakers:
            if operation_type == "streaming":
                config = CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=90.0,
                    success_threshold=3,
                    timeout_seconds=45.0,
                )
            elif operation_type == "connection":
                config = CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=60.0,
                    success_threshold=2,
                    timeout_seconds=30.0,
                )
            elif operation_type == "event_processing":
                config = CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout=30.0,
                    success_threshold=5,
                    timeout_seconds=5.0,
                )
            else:
                config = CircuitBreakerConfig()

            self.circuit_breakers[operation_type] = await get_circuit_breaker(
                f"event_engine_{operation_type}", config
            )

        return self.circuit_breakers[operation_type]

    async def _execute_with_resilience(self, operation_type: str, operation_func, *args, **kwargs):
        """Execute operation with circuit breaker and retry logic."""
        circuit_breaker = await self._get_circuit_breaker(operation_type)

        try:
            # Execute with circuit breaker protection
            if operation_type == "streaming":
                # Use stream recovery manager for streaming operations
                async def retry_wrapper():
                    return await self.stream_recovery_manager.execute_with_retry(
                        operation_func, *args, **kwargs
                    )

                result = await circuit_breaker.call(retry_wrapper)
            elif operation_type == "connection":
                # Use connection recovery manager for connection operations
                async def retry_wrapper():
                    return await self.connection_recovery_manager.execute_with_retry(
                        operation_func, *args, **kwargs
                    )

                result = await circuit_breaker.call(retry_wrapper)
            else:
                # Direct circuit breaker call for other operations
                result = await circuit_breaker.call(operation_func, *args, **kwargs)

            return result
        except Exception as e:
            self.handle_error(e, f"executing {operation_type} operation")
            raise

    async def run(self):
        """The main execution loop for the event-driven engine with proper error handling."""
        if self.is_running:
            self.logger.warning("Engine is already running")
            return

        self.is_running = True
        self.startup_time = datetime.now()

        with timer() as engine_timer:
            try:
                self.logger.info("Starting Event-Driven Engine...")

                if not self.strategies:
                    self.logger.warning("No event-driven strategies loaded. Engine will not start.")
                    return

                # Record engine startup
                record_metric(
                    "event_engine_started", 1, tags={"strategies_count": len(self.strategies)}
                )

                # Check circuit breaker health before starting
                await self._check_circuit_breaker_health()

                # Start all data streams concurrently with circuit breaker protection
                streaming_tasks = []

                if any(isinstance(s, NewsAnalyticsStrategy) for s in self.strategies):
                    if self.news_client:
                        # Create resilient streaming task
                        task = asyncio.create_task(self._start_resilient_news_stream())
                        streaming_tasks.append(task)
                        self.active_tasks.append(task)
                        self.logger.info("Started resilient news stream")

                # Add other streams like order books here...

                if not streaming_tasks:
                    self.logger.warning("No streaming tasks to run")
                    return

                self.logger.info(f"Running {len(streaming_tasks)} resilient streaming tasks")

                # Run all streaming tasks with error handling
                await asyncio.gather(*streaming_tasks, return_exceptions=True)

                # Report circuit breaker status after operations
                await self._report_circuit_breaker_status()

            except Exception as e:
                self.handle_error(e, "running event-driven engine")
                raise

            finally:
                self.is_running = False
                # Record engine metrics
                record_metric(
                    "event_engine_stopped",
                    1,
                    tags={
                        "runtime_seconds": engine_timer.elapsed,
                        "events_processed": self.events_processed,
                    },
                )
                self.logger.info(f"Event-Driven Engine stopped after {engine_timer.elapsed:.2f}s")

    async def _start_resilient_news_stream(self):
        """Start news streaming with circuit breaker protection."""
        try:
            # Create resilient streaming callback
            async def resilient_callback(event_data):
                await self._execute_with_resilience(
                    "event_processing", self.dispatch_news_event, event_data
                )

            # Start stream with circuit breaker protection
            await self._execute_with_resilience(
                "streaming", self.news_client.stream_news, callback=resilient_callback
            )

        except Exception as e:
            self.handle_error(e, "running resilient news stream")
            # Implement exponential backoff for stream restart
            await asyncio.sleep(5.0)
            if self.is_running:
                self.logger.info("Attempting to restart news stream")
                await self._start_resilient_news_stream()

    async def _check_circuit_breaker_health(self):
        """Check circuit breaker health before starting operations."""
        try:
            for operation_type in ["streaming", "connection", "event_processing"]:
                cb = await self._get_circuit_breaker(operation_type)
                if not cb.is_available():
                    self.logger.warning(
                        f"Circuit breaker for {operation_type} is open",
                        "Some operations may be degraded",
                    )
        except Exception as e:
            self.logger.warning(f"Could not check circuit breaker health: {e}")

    async def _report_circuit_breaker_status(self):
        """Report circuit breaker status after operations."""
        try:
            for operation_type, cb in self.circuit_breakers.items():
                metrics = cb.get_metrics()
                self.logger.info(
                    f"Circuit breaker {operation_type}: {metrics['state']}, "
                    f"Success rate: {metrics['success_rate']}%, "
                    f"Requests: {metrics['total_requests']}"
                )
        except Exception as e:
            self.logger.warning(f"Could not report circuit breaker status: {e}")

    async def dispatch_news_event(self, event_data: dict[str, Any]):
        """Enriches and dispatches a news event to all strategies with error handling."""
        try:
            # Enrich event data with timestamp
            event_data.setdefault("timestamp", ensure_utc(datetime.now()).isoformat())
            event_data.setdefault("event_id", f"news_{self.events_processed}")

            # In a full system, the engine enriches the event before dispatching
            # e.g., event_data['sentiment'] = self.sentiment_analyzer.score(event_data['headline'])
            event_data.setdefault("sentiment", 0.0)  # Placeholder

            # Dispatch to all strategies with error handling
            tasks = [strat.on_news_event(event_data) for strat in self.strategies]
            signal_lists = await asyncio.gather(*tasks, return_exceptions=True)

            generated_signals = []
            for i, signals in enumerate(signal_lists):
                if isinstance(signals, Exception):
                    self.handle_error(signals, f"strategy {i} processing news event")
                    continue

                if signals:
                    generated_signals.extend(signals)
                    # await self.execution_client.execute_orders(signals)
                    self.logger.info(f"Generated event-driven signals: {signals}")

            # Update metrics
            self.events_processed += 1
            record_metric(
                "event_processed",
                1,
                tags={"event_type": "news", "signals_generated": len(generated_signals)},
            )

        except Exception as e:
            self.handle_error(e, "dispatching news event")

    async def shutdown(self):
        """Gracefully shutdown the event-driven engine."""
        self.logger.info("Shutting down Event-Driven Engine...")

        # Cancel all active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        self.is_running = False
        self.logger.info("Event-Driven Engine shutdown complete")

    def get_status(self) -> dict[str, Any]:
        """Get current engine status."""
        active_tasks = len([t for t in self.active_tasks if not t.done()])

        # Record status metrics
        record_metric(
            "event_driven_engine.strategies_active", len(self.strategies), metric_type="gauge"
        )
        record_metric("event_driven_engine.tasks_active", active_tasks, metric_type="gauge")
        record_metric(
            "event_driven_engine.total_events_processed", self.events_processed, metric_type="gauge"
        )

        return {
            "is_running": self.is_running,
            "strategies_count": len(self.strategies),
            "events_processed": self.events_processed,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "active_tasks": active_tasks,
        }

    def get_resilience_status(self) -> dict[str, Any]:
        """Get current resilience status."""
        status = {
            "circuit_breakers": {},
            "stream_recovery_metrics": self.stream_recovery_manager.get_metrics(),
            "connection_recovery_metrics": self.connection_recovery_manager.get_metrics(),
        }

        for operation_type, cb in self.circuit_breakers.items():
            status["circuit_breakers"][operation_type] = cb.get_metrics()

        return status


async def main():
    """Main entry point to run the event-driven engine with proper error handling."""
    logger.info("Starting Event-Driven Engine application")

    try:
        # Use managed app context for proper resource management
        async with managed_app_context(
            "event_driven_engine", components=["database", "data_sources"]
        ) as context:

            engine = EventDrivenEngine(context.config)

            # Handle shutdown gracefully
            async def shutdown_handler():
                logger.info("Shutdown signal received")
                await engine.shutdown()

            # Register signal handlers
            # Standard library imports
            import signal

            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda s, f: asyncio.create_task(shutdown_handler()))

            # Run the engine
            await engine.run()

    except Exception as e:
        logger.error(f"Engine failed to start: {e}", exc_info=True)
        sys.exit(1)


# Create CLI app for standalone usage
app = create_event_driven_app("event_engine", "AI Trader Event-Driven Engine")

# Create CLI handler
cli_config = CLIAppConfig(
    name="event_engine",
    description="AI Trader Event-Driven Engine",
    context_components=["database", "data_sources"],
    enable_monitoring=True,
    show_progress=True,
)
cli_handler = StandardCLIHandler(cli_config)


@app.command()
def run():
    """
    Run the event-driven engine with resilience features.
    """
    # Local imports
    from main.utils import async_command

    @async_command(cli_handler, show_progress=True, operation_name="Event-Driven Engine")
    async def _run_engine(handler: StandardCLIHandler):
        try:
            engine = EventDrivenEngine(handler.config)
            await engine.run()

        except Exception as e:
            logger.error(f"Engine failed to run: {e!s}")
            raise

    _run_engine()


@app.command()
def status():
    """
    Show event-driven engine resilience status.
    """
    # Local imports
    from main.utils import async_command

    @async_command(cli_handler, show_progress=False, operation_name="Engine Status")
    async def _show_status(handler: StandardCLIHandler):
        try:
            engine = EventDrivenEngine(handler.config)

            # Get resilience status
            resilience_status = engine.get_resilience_status()
            engine_status = engine.get_status()

            print("=== Event-Driven Engine Status ===")
            print(f"Running: {engine_status['is_running']}")
            print(f"Strategies: {engine_status['strategies_count']}")
            print(f"Events processed: {engine_status['events_processed']}")
            print(f"Active tasks: {engine_status['active_tasks']}")

            # Circuit breaker status
            if resilience_status["circuit_breakers"]:
                print("\nCircuit Breaker Status:")
                for operation_type, cb_metrics in resilience_status["circuit_breakers"].items():
                    print(f"  {operation_type.upper()}:")
                    print(f"    State: {cb_metrics['state']}")
                    print(f"    Success Rate: {cb_metrics['success_rate']}%")
                    print(f"    Total Requests: {cb_metrics['total_requests']}")
                    print(f"    Failed Requests: {cb_metrics['failed_requests']}")

            # Stream recovery metrics
            stream_metrics = resilience_status["stream_recovery_metrics"]
            if stream_metrics["total_attempts"] > 0:
                print("\nStream Recovery Metrics:")
                print(f"  Total Attempts: {stream_metrics['total_attempts']}")
                print(f"  Success Rate: {stream_metrics['success_rate']}%")
                print(f"  Average Delay: {stream_metrics['avg_delay_per_attempt']}s")

            # Connection recovery metrics
            conn_metrics = resilience_status["connection_recovery_metrics"]
            if conn_metrics["total_attempts"] > 0:
                print("\nConnection Recovery Metrics:")
                print(f"  Total Attempts: {conn_metrics['total_attempts']}")
                print(f"  Success Rate: {conn_metrics['success_rate']}%")
                print(f"  Average Delay: {conn_metrics['avg_delay_per_attempt']}s")

            return {"engine_status": engine_status, "resilience_status": resilience_status}

        except Exception as e:
            logger.error(f"Failed to get status: {e!s}")
            return {"error": str(e)}

    _show_status()


if __name__ == "__main__":
    # Standard library imports
    import sys

    if len(sys.argv) > 1:
        # CLI mode
        app()
    else:
        # Direct execution mode
        asyncio.run(main())

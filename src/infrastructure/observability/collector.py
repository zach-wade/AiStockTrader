"""
Observability Collector for AI Trading System

Centralized collection and aggregation of:
- Trading system operations
- Market data processing
- Risk calculations
- Portfolio management
- Order execution flows
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..monitoring.logging import get_correlation_id
from ..monitoring.metrics import get_trading_metrics
from ..monitoring.performance import get_performance_monitor
from ..monitoring.telemetry import get_current_span

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityEvent:
    """Standardized observability event."""

    timestamp: float
    event_type: str  # order, market_data, risk, portfolio, system
    operation: str
    status: str  # success, error, warning
    duration_ms: float | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None

    # Trading-specific fields
    symbol: str | None = None
    order_id: str | None = None
    portfolio_id: str | None = None
    strategy: str | None = None

    # Metrics
    metrics: dict[str, int | float | str] = field(default_factory=dict)

    # Context and metadata
    context: dict[str, Any] = field(default_factory=dict)
    error_details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "operation": self.operation,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "portfolio_id": self.portfolio_id,
            "strategy": self.strategy,
            "metrics": self.metrics,
            "context": self.context,
            "error_details": self.error_details,
        }


@dataclass
class ObservabilityMetadata:
    """Metadata about observability collection."""

    collection_start_time: float
    events_collected: int = 0
    events_processed: int = 0
    events_exported: int = 0
    errors_encountered: int = 0
    last_export_time: float | None = None
    collector_health: str = "healthy"


class EventBuffer:
    """Thread-safe event buffer with automatic flushing."""

    def __init__(self, max_size: int = 10000, flush_interval: float = 30.0) -> None:
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._buffer: deque[ObservabilityEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._flush_callbacks: list[Callable[[list[ObservabilityEvent]], None]] = []
        self._last_flush_time = time.time()

    def add_event(self, event: ObservabilityEvent) -> None:
        """Add event to buffer."""
        with self._lock:
            self._buffer.append(event)

            # Auto-flush if buffer is full or time threshold met
            current_time = time.time()
            if (
                len(self._buffer) >= self.max_size
                or current_time - self._last_flush_time >= self.flush_interval
            ):
                self._flush_internal()

    def add_flush_callback(self, callback: Callable[[list[ObservabilityEvent]], None]) -> None:
        """Add flush callback."""
        self._flush_callbacks.append(callback)

    def flush(self) -> list[ObservabilityEvent]:
        """Manual flush of buffer."""
        with self._lock:
            return self._flush_internal()

    def _flush_internal(self) -> list[ObservabilityEvent]:
        """Internal flush implementation."""
        if not self._buffer:
            return []

        events = list(self._buffer)
        self._buffer.clear()
        self._last_flush_time = time.time()

        # Call flush callbacks
        for callback in self._flush_callbacks:
            try:
                callback(events)
            except Exception as e:
                logger.error(f"Event buffer flush callback failed: {e}")

        return events

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)


class ObservabilityCollector:
    """
    Central observability collector for the trading system.

    Collects and aggregates observability data from:
    - Telemetry (traces and spans)
    - Structured logging
    - Metrics collection
    - Health checks
    - Performance monitoring
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 30.0,
        enable_auto_export: bool = True,
        export_interval: float = 60.0,
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.enable_auto_export = enable_auto_export
        self.export_interval = export_interval

        # Event buffer
        self.event_buffer = EventBuffer(buffer_size, flush_interval)

        # Metadata tracking
        self.metadata = ObservabilityMetadata(collection_start_time=time.time())

        # Exporters
        self._exporters: list[Callable[[list[ObservabilityEvent]], None]] = []

        # Background tasks
        self._export_task: asyncio.Task[None] | None = None
        self._collection_task: asyncio.Task[None] | None = None
        self._stop_background_tasks = False

        # Event aggregation
        self._event_aggregates: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_aggregate_reset = time.time()

        # Setup event buffer callback
        self.event_buffer.add_flush_callback(self._process_events)

        logger.info("Observability collector initialized")

    def collect_order_event(
        self,
        operation: str,
        status: str,
        order_id: str,
        symbol: str,
        duration_ms: float | None = None,
        **context: Any,
    ) -> None:
        """Collect order-related event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="order",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            order_id=order_id,
            symbol=symbol,
            context=context,
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def collect_market_data_event(
        self,
        operation: str,
        status: str,
        symbol: str,
        duration_ms: float | None = None,
        **context: Any,
    ) -> None:
        """Collect market data event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="market_data",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            symbol=symbol,
            context=context,
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def collect_risk_event(
        self,
        operation: str,
        status: str,
        portfolio_id: str | None = None,
        duration_ms: float | None = None,
        **context: Any,
    ) -> None:
        """Collect risk calculation event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="risk",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            portfolio_id=portfolio_id,
            context=context,
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def collect_portfolio_event(
        self,
        operation: str,
        status: str,
        portfolio_id: str,
        duration_ms: float | None = None,
        **context: Any,
    ) -> None:
        """Collect portfolio management event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="portfolio",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            portfolio_id=portfolio_id,
            context=context,
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def collect_system_event(
        self, operation: str, status: str, duration_ms: float | None = None, **context: Any
    ) -> None:
        """Collect system-level event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="system",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            context=context,
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def collect_error_event(
        self,
        event_type: str,
        operation: str,
        error: Exception,
        duration_ms: float | None = None,
        **context: Any,
    ) -> None:
        """Collect error event."""
        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type=event_type,
            operation=operation,
            status="error",
            duration_ms=duration_ms,
            correlation_id=get_correlation_id(),
            error_details=str(error),
            context={**context, "error_type": type(error).__name__, "error_message": str(error)},
        )

        self._add_trace_context(event)
        self._collect_event(event)

    def _add_trace_context(self, event: ObservabilityEvent) -> None:
        """Add tracing context to event."""
        try:
            span = get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                event.trace_id = (
                    format(span_context.trace_id, "032x") if span_context.trace_id else None
                )
                event.span_id = (
                    format(span_context.span_id, "016x") if span_context.span_id else None
                )
        except Exception:
            pass  # Tracing not available

    def _collect_event(self, event: ObservabilityEvent) -> None:
        """Collect event into buffer."""
        self.event_buffer.add_event(event)
        self.metadata.events_collected += 1

        # Update aggregates
        self._update_aggregates(event)

    def _update_aggregates(self, event: ObservabilityEvent) -> None:
        """Update event aggregates."""
        # Reset aggregates periodically
        current_time = time.time()
        if current_time - self._last_aggregate_reset > 3600:  # 1 hour
            self._event_aggregates.clear()
            self._last_aggregate_reset = current_time

        # Update counts
        key = f"{event.event_type}:{event.operation}"
        self._event_aggregates[key][event.status] += 1
        self._event_aggregates[key]["total"] += 1

        if event.duration_ms:
            self._event_aggregates[key]["total_duration"] += int(event.duration_ms)
            self._event_aggregates[key]["avg_duration"] = int(
                self._event_aggregates[key]["total_duration"] / self._event_aggregates[key]["total"]
            )

    def _process_events(self, events: list[ObservabilityEvent]) -> None:
        """Process flushed events."""
        self.metadata.events_processed += len(events)
        logger.debug(f"Processed {len(events)} observability events")

    def add_exporter(self, exporter: Callable[[list[ObservabilityEvent]], None]) -> None:
        """Add event exporter."""
        self._exporters.append(exporter)
        logger.info("Added observability exporter")

    def export_events(self, events: list[ObservabilityEvent] | None = None) -> None:
        """Export events to all configured exporters."""
        if events is None:
            events = self.event_buffer.flush()

        if not events:
            return

        for exporter in self._exporters:
            try:
                exporter(events)
                self.metadata.events_exported += len(events)
            except Exception as e:
                self.metadata.errors_encountered += 1
                logger.error(f"Event export failed: {e}")

        self.metadata.last_export_time = time.time()

    def get_event_aggregates(self) -> dict[str, dict[str, Any]]:
        """Get event aggregates."""
        return dict(self._event_aggregates)

    def get_health_summary(self) -> dict[str, Any]:
        """Get collector health summary."""
        current_time = time.time()

        health_status = "healthy"
        if self.metadata.errors_encountered > 10:
            health_status = "degraded"
        if self.event_buffer.size() >= self.buffer_size * 0.9:
            health_status = "degraded"
        if current_time - (self.metadata.last_export_time or 0) > self.export_interval * 3:
            health_status = "unhealthy"

        return {
            "status": health_status,
            "buffer_size": self.event_buffer.size(),
            "buffer_capacity": self.buffer_size,
            "events_collected": self.metadata.events_collected,
            "events_processed": self.metadata.events_processed,
            "events_exported": self.metadata.events_exported,
            "errors_encountered": self.metadata.errors_encountered,
            "last_export_time": self.metadata.last_export_time,
            "collection_uptime": current_time - self.metadata.collection_start_time,
        }

    def collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics."""
        try:
            # Get trading metrics
            trading_metrics = get_trading_metrics()
            all_metrics = trading_metrics.collect_all_metrics()

            for name, value in all_metrics.items():
                self.collect_system_event(
                    operation="metric_collection",
                    status="success",
                    context={"metric_name": name, "metric_value": value},
                )

        except Exception as e:
            self.collect_error_event(event_type="system", operation="metric_collection", error=e)

    def collect_health_status(self) -> None:
        """Collect health check status."""
        try:
            # This would integrate with health checker
            self.collect_system_event(
                operation="health_check",
                status="success",
                context={"component": "observability_collector"},
            )

        except Exception as e:
            self.collect_error_event(event_type="system", operation="health_check", error=e)

    def collect_performance_data(self) -> None:
        """Collect performance monitoring data."""
        try:
            performance_monitor = get_performance_monitor()
            bottlenecks = performance_monitor.get_bottlenecks()

            for bottleneck in bottlenecks:
                self.collect_system_event(
                    operation="performance_analysis",
                    status="warning" if bottleneck["severity"] == "high" else "info",
                    context=bottleneck,
                )

        except Exception as e:
            self.collect_error_event(event_type="system", operation="performance_analysis", error=e)

    async def start_background_collection(self) -> None:
        """Start background collection tasks."""
        if self.enable_auto_export:
            self._export_task = asyncio.create_task(self._export_loop())

        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info("Started background observability collection")

    async def stop_background_collection(self) -> None:
        """Stop background collection tasks."""
        self._stop_background_tasks = True

        tasks = [self._export_task, self._collection_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Final export
        self.export_events()

        logger.info("Stopped background observability collection")

    async def _export_loop(self) -> None:
        """Background export loop."""
        while not self._stop_background_tasks:
            try:
                self.export_events()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Export loop error: {e}")
                await asyncio.sleep(5)

    async def _collection_loop(self) -> None:
        """Background collection loop."""
        while not self._stop_background_tasks:
            try:
                # Collect system-wide observability data
                self.collect_system_metrics()
                self.collect_health_status()
                self.collect_performance_data()

                await asyncio.sleep(60)  # Collect every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(5)

    def create_trading_span_hook(self) -> Callable[..., Any]:
        """Create hook for trading span events."""

        def span_hook(span_data: Any) -> None:
            """Hook function for span events."""
            try:
                # Extract trading attributes from span
                attributes = getattr(span_data, "attributes", {})

                event_type = "system"
                symbol = attributes.get("trading.symbol")
                order_id = attributes.get("trading.order_id")
                portfolio_id = attributes.get("trading.portfolio_id")

                # Determine event type based on attributes
                if order_id:
                    event_type = "order"
                elif symbol and not order_id:
                    event_type = "market_data"
                elif portfolio_id:
                    event_type = "portfolio"

                # Determine operation and status
                operation = getattr(span_data, "name", "unknown")
                status = "success" if getattr(span_data, "status", None) != "ERROR" else "error"

                # Calculate duration
                start_time = getattr(span_data, "start_time", 0)
                end_time = getattr(span_data, "end_time", 0)
                duration_ms = (
                    (end_time - start_time) / 1000000 if end_time > start_time else None
                )  # ns to ms

                # Create event
                event = ObservabilityEvent(
                    timestamp=end_time / 1000000000 if end_time else time.time(),  # ns to seconds
                    event_type=event_type,
                    operation=operation,
                    status=status,
                    duration_ms=duration_ms,
                    symbol=symbol,
                    order_id=order_id,
                    portfolio_id=portfolio_id,
                    context=dict(attributes) if attributes else {},
                )

                self._collect_event(event)

            except Exception as e:
                logger.warning(f"Span hook failed: {e}")

        return span_hook


# Global collector instance
_observability_collector: ObservabilityCollector | None = None


def initialize_observability_collector(**kwargs: Any) -> ObservabilityCollector:
    """Initialize global observability collector."""
    global _observability_collector
    _observability_collector = ObservabilityCollector(**kwargs)
    return _observability_collector


def get_observability_collector() -> ObservabilityCollector:
    """Get global observability collector."""
    if not _observability_collector:
        raise RuntimeError(
            "Observability collector not initialized. Call initialize_observability_collector() first."
        )
    return _observability_collector


# Convenience functions for event collection
def collect_order_event(
    operation: str, status: str, order_id: str, symbol: str, **kwargs: Any
) -> None:
    """Convenience function to collect order event."""
    collector = get_observability_collector()
    collector.collect_order_event(operation, status, order_id, symbol, **kwargs)


def collect_market_data_event(operation: str, status: str, symbol: str, **kwargs: Any) -> None:
    """Convenience function to collect market data event."""
    collector = get_observability_collector()
    collector.collect_market_data_event(operation, status, symbol, **kwargs)


def collect_risk_event(
    operation: str, status: str, portfolio_id: str | None = None, **kwargs: Any
) -> None:
    """Convenience function to collect risk event."""
    collector = get_observability_collector()
    collector.collect_risk_event(operation, status, portfolio_id, **kwargs)


def collect_portfolio_event(operation: str, status: str, portfolio_id: str, **kwargs: Any) -> None:
    """Convenience function to collect portfolio event."""
    collector = get_observability_collector()
    collector.collect_portfolio_event(operation, status, portfolio_id, **kwargs)


def collect_system_event(operation: str, status: str, **kwargs: Any) -> None:
    """Convenience function to collect system event."""
    collector = get_observability_collector()
    collector.collect_system_event(operation, status, **kwargs)


def collect_error_event(event_type: str, operation: str, error: Exception, **kwargs: Any) -> None:
    """Convenience function to collect error event."""
    collector = get_observability_collector()
    collector.collect_error_event(event_type, operation, error, **kwargs)

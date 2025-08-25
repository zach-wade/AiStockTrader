"""
OpenTelemetry Instrumentation for AI Trading System

Comprehensive distributed tracing implementation with trading-specific spans,
context propagation across services, and performance metrics collection.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from typing import Any

from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# AsyncIOInstrumentor is not available in all versions
try:
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor as AsyncIOInstrumentor

    HAS_ASYNCIO_INSTRUMENTOR = True
except ImportError:
    HAS_ASYNCIO_INSTRUMENTOR = False
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

try:
    from opentelemetry.propagators.b3 import B3MultiFormat

    HAS_B3_FORMAT = True
except ImportError:
    HAS_B3_FORMAT = False
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)


@dataclass
class TradingSpanAttributes:
    """Trading-specific span attributes."""

    TRADING_SYMBOL = "trading.symbol"
    TRADING_ORDER_ID = "trading.order_id"
    TRADING_ORDER_TYPE = "trading.order_type"
    TRADING_ORDER_SIDE = "trading.order_side"
    TRADING_QUANTITY = "trading.quantity"
    TRADING_PRICE = "trading.price"
    TRADING_VALUE = "trading.value"
    TRADING_BROKER = "trading.broker"
    TRADING_ACCOUNT = "trading.account"
    TRADING_PORTFOLIO_ID = "trading.portfolio_id"
    TRADING_POSITION_ID = "trading.position_id"
    TRADING_STRATEGY = "trading.strategy"
    TRADING_RISK_LEVEL = "trading.risk_level"
    TRADING_MARKET_HOURS = "trading.market_hours"
    TRADING_EXECUTION_VENUE = "trading.execution_venue"
    TRADING_LATENCY_CATEGORY = "trading.latency_category"


class TradingTelemetry:
    """
    Trading system telemetry configuration and management.

    Provides comprehensive distributed tracing for trading operations with:
    - Trading-specific span attributes
    - Context propagation across async operations
    - Performance metrics collection
    - Custom instrumentation for trading flows
    """

    def __init__(
        self,
        service_name: str = "ai-trading-system",
        service_version: str = "1.0.0",
        endpoint: str | None = None,
        enable_db_instrumentation: bool = True,
        enable_http_instrumentation: bool = True,
        enable_asyncio_instrumentation: bool = True,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.endpoint = endpoint
        self.enable_db_instrumentation = enable_db_instrumentation
        self.enable_http_instrumentation = enable_http_instrumentation
        self.enable_asyncio_instrumentation = enable_asyncio_instrumentation

        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._tracer: trace.Tracer | None = None
        self._meter: metrics.Meter | None = None

        # Performance counters
        self._span_counters: dict[str, int] = {}
        self._span_durations: dict[str, float] = {}

        self._setup_telemetry()

    def _setup_telemetry(self) -> None:
        """Initialize OpenTelemetry configuration."""
        try:
            # Create resource
            resource = Resource.create(
                {
                    SERVICE_NAME: self.service_name,
                    SERVICE_VERSION: self.service_version,
                    "service.type": "trading_system",
                    "service.environment": "production",
                }
            )

            # Setup trace provider
            self._setup_tracing(resource)

            # Setup metrics provider
            self._setup_metrics(resource)

            # Setup propagation
            if HAS_B3_FORMAT:
                propagate.set_global_textmap(B3MultiFormat())

            # Setup automatic instrumentation
            self._setup_auto_instrumentation()

            logger.info(f"Trading telemetry initialized for {self.service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            raise

    def _setup_tracing(self, resource: Resource) -> None:
        """Setup distributed tracing."""
        self._tracer_provider = TracerProvider(resource=resource)

        if self.endpoint:
            # OTLP exporter for production
            otlp_exporter = OTLPSpanExporter(endpoint=self.endpoint, timeout=10)
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=512,
                export_timeout_millis=30000,
                max_export_batch_size=512,
            )
        else:
            # Console exporter for development
            from opentelemetry.exporter.console import ConsoleSpanExporter

            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)

        self._tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(self._tracer_provider)

        self._tracer = trace.get_tracer(__name__, self.service_version)

    def _setup_metrics(self, resource: Resource) -> None:
        """Setup metrics collection."""
        if self.endpoint:
            # OTLP metric exporter
            metric_exporter = OTLPMetricExporter(
                endpoint=self.endpoint.replace("traces", "metrics"), timeout=10
            )
        else:
            # Console metric exporter for development
            from opentelemetry.exporter.console import ConsoleMetricExporter

            metric_exporter = ConsoleMetricExporter()

        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=60000,  # Export every minute
        )

        self._meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(self._meter_provider)

        self._meter = metrics.get_meter(__name__, self.service_version)

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation."""
        try:
            if self.enable_asyncio_instrumentation:
                if HAS_ASYNCIO_INSTRUMENTOR:
                    AsyncIOInstrumentor().instrument()
                logger.debug("AsyncIO instrumentation enabled")

            if self.enable_http_instrumentation:
                HTTPXClientInstrumentor().instrument()
                logger.debug("HTTPX instrumentation enabled")

            if self.enable_db_instrumentation:
                Psycopg2Instrumentor().instrument()
                logger.debug("PostgreSQL instrumentation enabled")

        except Exception as e:
            logger.warning(f"Failed to setup auto-instrumentation: {e}")

    def get_tracer(self) -> trace.Tracer:
        """Get the configured tracer."""
        if not self._tracer:
            raise RuntimeError("Tracer not initialized")
        return self._tracer

    def get_meter(self) -> metrics.Meter:
        """Get the configured meter."""
        if not self._meter:
            raise RuntimeError("Meter not initialized")
        return self._meter

    def shutdown(self) -> None:
        """Shutdown telemetry providers."""
        try:
            if self._tracer_provider:
                self._tracer_provider.shutdown()
            if self._meter_provider:
                self._meter_provider.shutdown()
            logger.info("Telemetry providers shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down telemetry: {e}")


# Global telemetry instance
_trading_telemetry: TradingTelemetry | None = None


def initialize_trading_telemetry(
    service_name: str = "ai-trading-system",
    service_version: str = "1.0.0",
    endpoint: str | None = None,
    **kwargs: Any,
) -> TradingTelemetry:
    """Initialize global trading telemetry."""
    global _trading_telemetry
    _trading_telemetry = TradingTelemetry(
        service_name=service_name, service_version=service_version, endpoint=endpoint, **kwargs
    )
    return _trading_telemetry


def get_trading_telemetry() -> TradingTelemetry:
    """Get global trading telemetry instance."""
    if not _trading_telemetry:
        raise RuntimeError(
            "Trading telemetry not initialized. Call initialize_trading_telemetry() first."
        )
    return _trading_telemetry


def trading_tracer() -> trace.Tracer:
    """Get trading system tracer."""
    return get_trading_telemetry().get_tracer()


def get_current_span() -> trace.Span:
    """Get current active span."""
    return trace.get_current_span()


def add_trading_attributes(
    span: trace.Span,
    symbol: str | None = None,
    order_id: str | None = None,
    order_type: str | None = None,
    order_side: str | None = None,
    quantity: int | float | Decimal | None = None,
    price: float | Decimal | None = None,
    value: float | Decimal | None = None,
    broker: str | None = None,
    portfolio_id: str | None = None,
    position_id: str | None = None,
    strategy: str | None = None,
    risk_level: str | None = None,
    **kwargs: Any,
) -> None:
    """Add trading-specific attributes to a span."""
    attributes: dict[str, AttributeValue] = {}

    if symbol:
        attributes[TradingSpanAttributes.TRADING_SYMBOL] = symbol
    if order_id:
        attributes[TradingSpanAttributes.TRADING_ORDER_ID] = order_id
    if order_type:
        attributes[TradingSpanAttributes.TRADING_ORDER_TYPE] = order_type
    if order_side:
        attributes[TradingSpanAttributes.TRADING_ORDER_SIDE] = order_side
    if quantity is not None:
        attributes[TradingSpanAttributes.TRADING_QUANTITY] = float(quantity)
    if price is not None:
        attributes[TradingSpanAttributes.TRADING_PRICE] = float(price)
    if value is not None:
        attributes[TradingSpanAttributes.TRADING_VALUE] = float(value)
    if broker:
        attributes[TradingSpanAttributes.TRADING_BROKER] = broker
    if portfolio_id:
        attributes[TradingSpanAttributes.TRADING_PORTFOLIO_ID] = portfolio_id
    if position_id:
        attributes[TradingSpanAttributes.TRADING_POSITION_ID] = position_id
    if strategy:
        attributes[TradingSpanAttributes.TRADING_STRATEGY] = strategy
    if risk_level:
        attributes[TradingSpanAttributes.TRADING_RISK_LEVEL] = risk_level

    # Add any additional custom attributes
    for key, value in kwargs.items():
        if key.startswith("trading."):
            # Convert Decimal to float for OpenTelemetry compatibility
            if isinstance(value, Decimal):
                attributes[key] = float(value)
            elif value is not None:
                attributes[key] = value

    if attributes:
        span.set_attributes(attributes)


@contextmanager
def trading_span(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    set_status_on_exception: bool = True,
    **span_attributes: Any,
) -> Any:
    """
    Context manager for creating trading spans with automatic attribute setting.

    Args:
        name: Span name
        kind: Span kind
        set_status_on_exception: Whether to set error status on exceptions
        **span_attributes: Trading-specific attributes

    Yields:
        Active span
    """
    tracer = trading_tracer()

    with tracer.start_as_current_span(name, kind=kind) as span:
        try:
            # Add trading attributes
            add_trading_attributes(span, **span_attributes)

            yield span

        except Exception as e:
            if set_status_on_exception:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


@asynccontextmanager
async def async_trading_span(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    set_status_on_exception: bool = True,
    **span_attributes: Any,
) -> AsyncGenerator[trace.Span, None]:
    """
    Async context manager for trading spans.

    Args:
        name: Span name
        kind: Span kind
        set_status_on_exception: Whether to set error status on exceptions
        **span_attributes: Trading-specific attributes

    Yields:
        Active span
    """
    tracer = trading_tracer()

    with tracer.start_as_current_span(name, kind=kind) as span:
        try:
            # Add trading attributes
            add_trading_attributes(span, **span_attributes)

            yield span

        except Exception as e:
            if set_status_on_exception:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def trace_trading_operation(
    operation_name: str | None = None,
    span_kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    record_exception: bool = True,
    **span_attributes: Any,
) -> Any:
    """
    Decorator for tracing trading operations.

    Args:
        operation_name: Custom operation name (defaults to function name)
        span_kind: Type of span
        record_exception: Whether to record exceptions in span
        **span_attributes: Trading-specific attributes

    Returns:
        Decorated function
    """

    def decorator(func: Any) -> Any:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"

            with trading_span(
                name, kind=span_kind, set_status_on_exception=record_exception, **span_attributes
            ) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result_type", type(result).__name__)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    span.set_attribute("function.duration", duration)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"

            async with async_trading_span(
                name, kind=span_kind, set_status_on_exception=record_exception, **span_attributes
            ) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.result_type", type(result).__name__)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    span.set_attribute("function.duration", duration)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def trace_order_execution(func: Any) -> Any:
    """Decorator specifically for order execution tracing."""
    return trace_trading_operation(
        span_kind=trace.SpanKind.CLIENT, operation_name=f"order_execution.{func.__name__}"
    )(func)


def trace_market_data_operation(func: Any) -> Any:
    """Decorator for market data operation tracing."""
    return trace_trading_operation(
        span_kind=trace.SpanKind.CLIENT, operation_name=f"market_data.{func.__name__}"
    )(func)


def trace_risk_calculation(func: Any) -> Any:
    """Decorator for risk calculation tracing."""
    return trace_trading_operation(
        span_kind=trace.SpanKind.INTERNAL, operation_name=f"risk_calculation.{func.__name__}"
    )(func)


def trace_portfolio_operation(func: Any) -> Any:
    """Decorator for portfolio operation tracing."""
    return trace_trading_operation(
        span_kind=trace.SpanKind.INTERNAL, operation_name=f"portfolio.{func.__name__}"
    )(func)


class LatencyTracker:
    """Track operation latency for trading system."""

    def __init__(self) -> None:
        self.tracer = trading_tracer()
        self.meter = get_trading_telemetry().get_meter()

        # Create latency histogram
        self.latency_histogram = self.meter.create_histogram(
            name="trading_operation_latency",
            description="Latency of trading operations in milliseconds",
            unit="ms",
        )

        # Create operation counter
        self.operation_counter = self.meter.create_counter(
            name="trading_operations_total", description="Total number of trading operations"
        )

    @contextmanager
    def track_latency(self, operation: str, **labels: Any) -> Any:
        """Track latency for an operation."""
        start_time = time.perf_counter()
        span = get_current_span()

        try:
            yield

            # Record success
            self.operation_counter.add(1, {"operation": operation, "status": "success", **labels})

        except Exception as e:
            # Record failure
            self.operation_counter.add(
                1,
                {
                    "operation": operation,
                    "status": "error",
                    "error_type": type(e).__name__,
                    **labels,
                },
            )

            if span.is_recording():
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

            raise

        finally:
            # Record latency
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.latency_histogram.record(duration_ms, {"operation": operation, **labels})

            if span.is_recording():
                span.set_attribute("latency_ms", duration_ms)


# Global latency tracker
latency_tracker = LatencyTracker() if _trading_telemetry else None


def get_latency_tracker() -> LatencyTracker:
    """Get global latency tracker."""
    global latency_tracker
    if not latency_tracker:
        latency_tracker = LatencyTracker()
    return latency_tracker

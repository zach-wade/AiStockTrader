"""
Comprehensive unit tests for telemetry module.

Tests OpenTelemetry instrumentation, distributed tracing, context propagation,
and trading-specific spans.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from src.infrastructure.monitoring.telemetry import (
    LatencyTracker,
    TradingSpanAttributes,
    TradingTelemetry,
    add_trading_attributes,
    async_trading_span,
    get_current_span,
    get_latency_tracker,
    get_trading_telemetry,
    initialize_trading_telemetry,
    trace_market_data_operation,
    trace_order_execution,
    trace_portfolio_operation,
    trace_risk_calculation,
    trace_trading_operation,
    trading_span,
    trading_tracer,
)


class TestTradingSpanAttributes:
    """Test TradingSpanAttributes constants."""

    def test_span_attributes(self):
        """Test trading-specific span attributes."""
        assert TradingSpanAttributes.TRADING_SYMBOL == "trading.symbol"
        assert TradingSpanAttributes.TRADING_ORDER_ID == "trading.order_id"
        assert TradingSpanAttributes.TRADING_ORDER_TYPE == "trading.order_type"
        assert TradingSpanAttributes.TRADING_ORDER_SIDE == "trading.order_side"
        assert TradingSpanAttributes.TRADING_QUANTITY == "trading.quantity"
        assert TradingSpanAttributes.TRADING_PRICE == "trading.price"
        assert TradingSpanAttributes.TRADING_VALUE == "trading"
        assert TradingSpanAttributes.TRADING_BROKER == "trading.broker"
        assert TradingSpanAttributes.TRADING_ACCOUNT == "trading.account"
        assert TradingSpanAttributes.TRADING_PORTFOLIO_ID == "trading.portfolio_id"
        assert TradingSpanAttributes.TRADING_POSITION_ID == "trading.position_id"
        assert TradingSpanAttributes.TRADING_STRATEGY == "trading.strategy"
        assert TradingSpanAttributes.TRADING_RISK_LEVEL == "trading.risk_level"
        assert TradingSpanAttributes.TRADING_MARKET_HOURS == "trading.market_hours"
        assert TradingSpanAttributes.TRADING_EXECUTION_VENUE == "trading.execution_venue"
        assert TradingSpanAttributes.TRADING_LATENCY_CATEGORY == "trading.latency_category"


class TestTradingTelemetry:
    """Test TradingTelemetry functionality."""

    @patch("src.infrastructure.monitoring.telemetry.TracerProvider")
    @patch("src.infrastructure.monitoring.telemetry.MeterProvider")
    def test_initialization(self, mock_meter_provider, mock_tracer_provider):
        """Test TradingTelemetry initialization."""
        telemetry = TradingTelemetry(
            service_name="test-service", service_version="1.0.0", endpoint="http://localhost:4317"
        )

        assert telemetry.service_name == "test-service"
        assert telemetry.service_version == "1.0.0"
        assert telemetry.endpoint == "http://localhost:4317"
        assert telemetry.enable_db_instrumentation is True
        assert telemetry.enable_http_instrumentation is True
        assert telemetry.enable_asyncio_instrumentation is True

    @patch("src.infrastructure.monitoring.telemetry.Resource.create")
    @patch("src.infrastructure.monitoring.telemetry.TracerProvider")
    def test_setup_telemetry(self, mock_tracer_provider_class, mock_resource_create):
        """Test telemetry setup."""
        mock_resource = Mock()
        mock_resource_create.return_value = mock_resource

        telemetry = TradingTelemetry(service_name="trading-system", service_version="2.0.0")

        mock_resource_create.assert_called_once()
        call_args = mock_resource_create.call_args[0][0]
        assert call_args["service.name"] == "trading-system"
        assert call_args["service.version"] == "2.0.0"
        assert call_args["service.type"] == "trading_system"

    @patch("src.infrastructure.monitoring.telemetry.OTLPSpanExporter")
    @patch("src.infrastructure.monitoring.telemetry.BatchSpanProcessor")
    @patch("src.infrastructure.monitoring.telemetry.TracerProvider")
    @patch("src.infrastructure.monitoring.telemetry.trace.set_tracer_provider")
    @patch("src.infrastructure.monitoring.telemetry.trace.get_tracer")
    def test_setup_tracing_with_endpoint(
        self,
        mock_get_tracer,
        mock_set_provider,
        mock_tracer_provider_class,
        mock_batch_processor,
        mock_otlp_exporter,
    ):
        """Test tracing setup with OTLP endpoint."""
        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_tracer_provider_class.return_value = mock_tracer_provider
        mock_exporter = Mock()
        mock_otlp_exporter.return_value = mock_exporter
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor

        telemetry = TradingTelemetry(endpoint="http://localhost:4317")
        telemetry._setup_tracing(mock_resource)

        mock_otlp_exporter.assert_called_once_with(
            endpoint="http://localhost:4317", command_timeout=10
        )
        mock_batch_processor.assert_called_once_with(
            mock_exporter,
            max_queue_size=512,
            export_timeout_millis=30000,
            max_export_batch_size=512,
        )
        mock_tracer_provider.add_span_processor.assert_called_once_with(mock_processor)
        mock_set_provider.assert_called_once_with(mock_tracer_provider)

    @patch("src.infrastructure.monitoring.telemetry.ConsoleSpanExporter")
    @patch("src.infrastructure.monitoring.telemetry.BatchSpanProcessor")
    @patch("src.infrastructure.monitoring.telemetry.TracerProvider")
    def test_setup_tracing_without_endpoint(
        self, mock_tracer_provider_class, mock_batch_processor, mock_console_exporter
    ):
        """Test tracing setup with console exporter."""
        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_tracer_provider_class.return_value = mock_tracer_provider
        mock_exporter = Mock()
        mock_console_exporter.return_value = mock_exporter
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor

        telemetry = TradingTelemetry(endpoint=None)
        telemetry._setup_tracing(mock_resource)

        mock_console_exporter.assert_called_once()
        mock_batch_processor.assert_called_once_with(mock_exporter)

    @patch("src.infrastructure.monitoring.telemetry.OTLPMetricExporter")
    @patch("src.infrastructure.monitoring.telemetry.PeriodicExportingMetricReader")
    @patch("src.infrastructure.monitoring.telemetry.MeterProvider")
    @patch("src.infrastructure.monitoring.telemetry.metrics.set_meter_provider")
    @patch("src.infrastructure.monitoring.telemetry.metrics.get_meter")
    def test_setup_metrics_with_endpoint(
        self,
        mock_get_meter,
        mock_set_provider,
        mock_meter_provider_class,
        mock_reader_class,
        mock_otlp_exporter,
    ):
        """Test metrics setup with OTLP endpoint."""
        mock_resource = Mock()
        mock_meter_provider = Mock()
        mock_meter_provider_class.return_value = mock_meter_provider
        mock_exporter = Mock()
        mock_otlp_exporter.return_value = mock_exporter
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader

        telemetry = TradingTelemetry(endpoint="http://localhost:4317/traces")
        telemetry._setup_metrics(mock_resource)

        mock_otlp_exporter.assert_called_once_with(
            endpoint="http://localhost:4317/metrics", command_timeout=10
        )
        mock_reader_class.assert_called_once_with(
            exporter=mock_exporter, export_interval_millis=60000
        )

    @patch("src.infrastructure.monitoring.telemetry.HTTPXClientInstrumentor")
    @patch("src.infrastructure.monitoring.telemetry.Psycopg2Instrumentor")
    def test_setup_auto_instrumentation(self, mock_psycopg2, mock_httpx):
        """Test automatic instrumentation setup."""
        mock_httpx_inst = Mock()
        mock_httpx.return_value = mock_httpx_inst
        mock_psycopg2_inst = Mock()
        mock_psycopg2.return_value = mock_psycopg2_inst

        telemetry = TradingTelemetry()

        with patch("src.infrastructure.monitoring.telemetry.HAS_ASYNCIO_INSTRUMENTOR", False):
            telemetry._setup_auto_instrumentation()

        mock_httpx_inst.instrument.assert_called_once()
        mock_psycopg2_inst.instrument.assert_called_once()

    def test_get_tracer(self):
        """Test getting tracer."""
        telemetry = TradingTelemetry()
        telemetry._tracer = Mock()

        tracer = telemetry.get_tracer()
        assert tracer == telemetry._tracer

    def test_get_tracer_not_initialized(self):
        """Test getting tracer when not initialized."""
        telemetry = TradingTelemetry()
        telemetry._tracer = None

        with pytest.raises(RuntimeError, match="Tracer not initialized"):
            telemetry.get_tracer()

    def test_get_meter(self):
        """Test getting meter."""
        telemetry = TradingTelemetry()
        telemetry._meter = Mock()

        meter = telemetry.get_meter()
        assert meter == telemetry._meter

    def test_get_meter_not_initialized(self):
        """Test getting meter when not initialized."""
        telemetry = TradingTelemetry()
        telemetry._meter = None

        with pytest.raises(RuntimeError, match="Meter not initialized"):
            telemetry.get_meter()

    def test_shutdown(self):
        """Test telemetry shutdown."""
        telemetry = TradingTelemetry()
        telemetry._tracer_provider = Mock()
        telemetry._meter_provider = Mock()

        telemetry.shutdown()

        telemetry._tracer_provider.shutdown.assert_called_once()
        telemetry._meter_provider.shutdown.assert_called_once()

    @patch("logging.Logger.error")
    def test_shutdown_error_handling(self, mock_logger):
        """Test error handling during shutdown."""
        telemetry = TradingTelemetry()
        telemetry._tracer_provider = Mock()
        telemetry._tracer_provider.shutdown.side_effect = Exception("Shutdown error")

        telemetry.shutdown()

        mock_logger.assert_called_once()


class TestGlobalTelemetryFunctions:
    """Test global telemetry functions."""

    def test_initialize_trading_telemetry(self):
        """Test initializing global telemetry."""
        telemetry = initialize_trading_telemetry(
            service_name="test-service", service_version="1.0.0", endpoint="http://localhost:4317"
        )

        assert telemetry is not None
        assert telemetry.service_name == "test-service"

    def test_get_trading_telemetry(self):
        """Test getting global telemetry."""
        # Initialize first
        original = initialize_trading_telemetry()

        retrieved = get_trading_telemetry()
        assert retrieved is original

    def test_get_trading_telemetry_not_initialized(self):
        """Test getting telemetry when not initialized."""
        # Reset global
        import src.infrastructure.monitoring.telemetry as telemetry_module

        telemetry_module._trading_telemetry = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_trading_telemetry()

    @patch("src.infrastructure.monitoring.telemetry.get_trading_telemetry")
    def test_trading_tracer(self, mock_get_telemetry):
        """Test getting trading tracer."""
        mock_telemetry = Mock()
        mock_tracer = Mock()
        mock_telemetry.get_tracer.return_value = mock_tracer
        mock_get_telemetry.return_value = mock_telemetry

        tracer = trading_tracer()

        assert tracer == mock_tracer
        mock_telemetry.get_tracer.assert_called_once()

    @patch("src.infrastructure.monitoring.telemetry.trace.get_current_span")
    def test_get_current_span(self, mock_trace_get_span):
        """Test getting current span."""
        mock_span = Mock()
        mock_trace_get_span.return_value = mock_span

        span = get_current_span()

        assert span == mock_span


class TestAddTradingAttributes:
    """Test add_trading_attributes function."""

    def test_add_basic_attributes(self):
        """Test adding basic trading attributes."""
        mock_span = Mock()

        add_trading_attributes(
            mock_span,
            symbol="AAPL",
            order_id="ORD123",
            order_type="LIMIT",
            order_side="BUY",
            quantity=100,
            price=150.50,
        )

        expected_attrs = {
            TradingSpanAttributes.TRADING_SYMBOL: "AAPL",
            TradingSpanAttributes.TRADING_ORDER_ID: "ORD123",
            TradingSpanAttributes.TRADING_ORDER_TYPE: "LIMIT",
            TradingSpanAttributes.TRADING_ORDER_SIDE: "BUY",
            TradingSpanAttributes.TRADING_QUANTITY: 100.0,
            TradingSpanAttributes.TRADING_PRICE: 150.50,
        }
        mock_span.set_attributes.assert_called_once_with(expected_attrs)

    def test_add_portfolio_attributes(self):
        """Test adding portfolio-related attributes."""
        mock_span = Mock()

        add_trading_attributes(
            mock_span,
            portfolio_id="PORT123",
            position_id="POS456",
            strategy="momentum",
            risk_level="medium",
        )

        expected_attrs = {
            TradingSpanAttributes.TRADING_PORTFOLIO_ID: "PORT123",
            TradingSpanAttributes.TRADING_POSITION_ID: "POS456",
            TradingSpanAttributes.TRADING_STRATEGY: "momentum",
            TradingSpanAttributes.TRADING_RISK_LEVEL: "medium",
        }
        mock_span.set_attributes.assert_called_once_with(expected_attrs)

    def test_add_decimal_attributes(self):
        """Test adding Decimal type attributes."""
        mock_span = Mock()

        add_trading_attributes(
            mock_span,
            quantity=Decimal("100.5"),
            price=Decimal("150.75"),
            value=Decimal("15112.875"),
        )

        expected_attrs = {
            TradingSpanAttributes.TRADING_QUANTITY: 100.5,
            TradingSpanAttributes.TRADING_PRICE: 150.75,
            TradingSpanAttributes.TRADING_VALUE: 15112.875,
        }
        mock_span.set_attributes.assert_called_once_with(expected_attrs)

    def test_add_custom_attributes(self):
        """Test adding custom trading attributes."""
        mock_span = Mock()

        add_trading_attributes(
            mock_span,
            symbol="AAPL",
            **{
                "trading.custom_field": "custom_value",
                "trading.another_field": 42,
                "non_trading_field": "ignored",
            },
        )

        call_args = mock_span.set_attributes.call_args[0][0]
        assert TradingSpanAttributes.TRADING_SYMBOL in call_args
        assert "trading.custom_field" in call_args
        assert "trading.another_field" in call_args
        assert "non_trading_field" not in call_args

    def test_add_no_attributes(self):
        """Test when no attributes are provided."""
        mock_span = Mock()

        add_trading_attributes(mock_span)

        mock_span.set_attributes.assert_not_called()


class TestTradingSpanContextManager:
    """Test trading_span context manager."""

    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_trading_span_success(self, mock_tracer_func):
        """Test successful span creation."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context
        mock_tracer_func.return_value = mock_tracer

        with trading_span("test_operation", symbol="AAPL", order_id="ORD123") as span:
            assert span == mock_span

        mock_tracer.start_as_current_span.assert_called_once_with(
            "test_operation", kind=SpanKind.INTERNAL
        )
        # Verify attributes were set
        mock_span.set_attributes.assert_called()

    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_trading_span_with_exception(self, mock_tracer_func):
        """Test span with exception handling."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context
        mock_tracer_func.return_value = mock_tracer

        with pytest.raises(ValueError):
            with trading_span("test_operation", set_status_on_exception=True) as span:
                raise ValueError("Test error")

        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        mock_span.record_exception.assert_called_once()

    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_trading_span_custom_kind(self, mock_tracer_func):
        """Test span with custom kind."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context
        mock_tracer_func.return_value = mock_tracer

        with trading_span("client_operation", kind=SpanKind.CLIENT):
            pass

        mock_tracer.start_as_current_span.assert_called_once_with(
            "client_operation", kind=SpanKind.CLIENT
        )


class TestAsyncTradingSpanContextManager:
    """Test async_trading_span context manager."""

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    async def test_async_trading_span_success(self, mock_tracer_func):
        """Test successful async span creation."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context
        mock_tracer_func.return_value = mock_tracer

        async with async_trading_span("async_operation", portfolio_id="PORT123") as span:
            assert span == mock_span
            await asyncio.sleep(0.01)

        mock_span.set_attributes.assert_called()

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    async def test_async_trading_span_with_exception(self, mock_tracer_func):
        """Test async span with exception handling."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context
        mock_tracer_func.return_value = mock_tracer

        with pytest.raises(RuntimeError):
            async with async_trading_span("async_operation", set_status_on_exception=True):
                await asyncio.sleep(0.01)
                raise RuntimeError("Async error")

        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once()


class TestTraceTradingOperationDecorator:
    """Test trace_trading_operation decorator."""

    @patch("src.infrastructure.monitoring.telemetry.trading_span")
    def test_trace_sync_function(self, mock_trading_span):
        """Test tracing synchronous function."""
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_trading_span.return_value = mock_context

        @trace_trading_operation("custom_operation", symbol="AAPL")
        def test_func(x, y):
            return x + y

        result = test_func(5, 3)

        assert result == 8
        mock_trading_span.assert_called_once()
        call_args = mock_trading_span.call_args
        assert call_args[0][0] == "custom_operation"
        assert call_args[1]["symbol"] == "AAPL"
        mock_span.set_attribute.assert_any_call("function.name", "test_func")
        mock_span.set_attribute.assert_any_call("function.result_type", "int")

    @patch("src.infrastructure.monitoring.telemetry.trading_span")
    def test_trace_sync_function_default_name(self, mock_trading_span):
        """Test tracing with default operation name."""
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_trading_span.return_value = mock_context

        @trace_trading_operation()
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        call_args = mock_trading_span.call_args
        assert "test_func" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.telemetry.async_trading_span")
    async def test_trace_async_function(self, mock_async_trading_span):
        """Test tracing asynchronous function."""
        mock_span = Mock()

        # Create async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_span
        mock_async_trading_span.return_value = mock_context

        @trace_trading_operation("async_op", portfolio_id="PORT123")
        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_func(5)

        assert result == 10
        mock_async_trading_span.assert_called_once()
        mock_span.set_attribute.assert_any_call("function.name", "async_func")
        mock_span.set_attribute.assert_any_call("function.result_type", "int")

    @patch("src.infrastructure.monitoring.telemetry.trading_span")
    def test_trace_function_with_exception(self, mock_trading_span):
        """Test tracing function that raises exception."""
        mock_span = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_trading_span.return_value = mock_context

        @trace_trading_operation(record_exception=True)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        # Duration should still be recorded
        duration_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "function.duration"
        ]
        assert len(duration_calls) > 0


class TestSpecializedTraceDecorators:
    """Test specialized trace decorators."""

    @patch("src.infrastructure.monitoring.telemetry.trace_trading_operation")
    def test_trace_order_execution(self, mock_trace_operation):
        """Test order execution tracing."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_trace_operation.return_value = mock_decorator

        @trace_order_execution
        def execute_order():
            return "executed"

        result = execute_order()

        assert result == "executed"
        mock_trace_operation.assert_called_once_with(
            span_kind=SpanKind.CLIENT, operation_name="order_execution.execute_order"
        )

    @patch("src.infrastructure.monitoring.telemetry.trace_trading_operation")
    def test_trace_market_data_operation(self, mock_trace_operation):
        """Test market data operation tracing."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_trace_operation.return_value = mock_decorator

        @trace_market_data_operation
        def fetch_quotes():
            return [100.0, 101.0]

        result = fetch_quotes()

        assert result == [100.0, 101.0]
        mock_trace_operation.assert_called_once_with(
            span_kind=SpanKind.CLIENT, operation_name="market_data.fetch_quotes"
        )

    @patch("src.infrastructure.monitoring.telemetry.trace_trading_operation")
    def test_trace_risk_calculation(self, mock_trace_operation):
        """Test risk calculation tracing."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_trace_operation.return_value = mock_decorator

        @trace_risk_calculation
        def calculate_var():
            return 5000.0

        result = calculate_var()

        assert result == 5000.0
        mock_trace_operation.assert_called_once_with(
            span_kind=SpanKind.INTERNAL, operation_name="risk_calculation.calculate_var"
        )

    @patch("src.infrastructure.monitoring.telemetry.trace_trading_operation")
    def test_trace_portfolio_operation(self, mock_trace_operation):
        """Test portfolio operation tracing."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_trace_operation.return_value = mock_decorator

        @trace_portfolio_operation
        def rebalance():
            return "rebalanced"

        result = rebalance()

        assert result == "rebalanced"
        mock_trace_operation.assert_called_once_with(
            span_kind=SpanKind.INTERNAL, operation_name="portfolio.rebalance"
        )


class TestLatencyTracker:
    """Test LatencyTracker functionality."""

    @patch("src.infrastructure.monitoring.telemetry.get_trading_telemetry")
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_initialization(self, mock_tracer_func, mock_get_telemetry):
        """Test LatencyTracker initialization."""
        mock_telemetry = Mock()
        mock_meter = Mock()
        mock_telemetry.get_meter.return_value = mock_meter
        mock_get_telemetry.return_value = mock_telemetry

        mock_tracer = Mock()
        mock_tracer_func.return_value = mock_tracer

        tracker = LatencyTracker()

        assert tracker.tracer == mock_tracer
        assert tracker.meter == mock_meter
        mock_meter.create_histogram.assert_called_once()
        mock_meter.create_counter.assert_called_once()

    @patch("src.infrastructure.monitoring.telemetry.get_current_span")
    @patch("src.infrastructure.monitoring.telemetry.get_trading_telemetry")
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_track_latency_success(self, mock_tracer_func, mock_get_telemetry, mock_get_span):
        """Test tracking latency for successful operation."""
        # Setup mocks
        mock_telemetry = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_counter = Mock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_telemetry.get_meter.return_value = mock_meter
        mock_get_telemetry.return_value = mock_telemetry

        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        tracker = LatencyTracker()

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            with tracker.track_latency("test_op", service="test"):
                pass

        # Verify counter incremented with success
        mock_counter.add.assert_called_once_with(
            1, {"operation": "test_op", "status": "success", "service": "test"}
        )

        # Verify latency recorded (100ms)
        mock_histogram.record.assert_called_once()
        histogram_call = mock_histogram.record.call_args
        assert histogram_call[0][0] == 100.0  # 0.1 seconds = 100ms
        assert histogram_call[0][1]["operation"] == "test_op"

        # Verify span attribute set
        mock_span.set_attribute.assert_called_once_with("latency_ms", 100.0)

    @patch("src.infrastructure.monitoring.telemetry.get_current_span")
    @patch("src.infrastructure.monitoring.telemetry.get_trading_telemetry")
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_track_latency_error(self, mock_tracer_func, mock_get_telemetry, mock_get_span):
        """Test tracking latency for failed operation."""
        # Setup mocks
        mock_telemetry = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_counter = Mock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_telemetry.get_meter.return_value = mock_meter
        mock_get_telemetry.return_value = mock_telemetry

        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        tracker = LatencyTracker()

        with patch("time.perf_counter", side_effect=[0.0, 0.05]):
            with pytest.raises(RuntimeError):
                with tracker.track_latency("failing_op"):
                    raise RuntimeError("Test error")

        # Verify counter incremented with error
        mock_counter.add.assert_called_once()
        counter_call = mock_counter.add.call_args
        assert counter_call[0][1]["operation"] == "failing_op"
        assert counter_call[0][1]["status"] == "error"
        assert counter_call[0][1]["error_type"] == "RuntimeError"

        # Verify latency still recorded
        mock_histogram.record.assert_called_once()
        assert mock_histogram.record.call_args[0][0] == 50.0  # 50ms

        # Verify span error recording
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()

    @patch("src.infrastructure.monitoring.telemetry.get_current_span")
    @patch("src.infrastructure.monitoring.telemetry.get_trading_telemetry")
    @patch("src.infrastructure.monitoring.telemetry.trading_tracer")
    def test_track_latency_span_not_recording(
        self, mock_tracer_func, mock_get_telemetry, mock_get_span
    ):
        """Test tracking latency when span is not recording."""
        # Setup mocks
        mock_telemetry = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_counter = Mock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_telemetry.get_meter.return_value = mock_meter
        mock_get_telemetry.return_value = mock_telemetry

        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        tracker = LatencyTracker()

        with tracker.track_latency("test_op"):
            pass

        # Metrics should still be recorded
        mock_counter.add.assert_called_once()
        mock_histogram.record.assert_called_once()

        # But span attributes should not be set
        mock_span.set_attribute.assert_not_called()
        mock_span.record_exception.assert_not_called()


class TestGetLatencyTracker:
    """Test get_latency_tracker function."""

    @patch("src.infrastructure.monitoring.telemetry.LatencyTracker")
    def test_get_latency_tracker_creates_new(self, mock_tracker_class):
        """Test creating new latency tracker."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker

        # Reset global
        import src.infrastructure.monitoring.telemetry as telemetry_module

        telemetry_module.latency_tracker = None
        telemetry_module._trading_telemetry = Mock()  # Ensure telemetry exists

        tracker = get_latency_tracker()

        assert tracker == mock_tracker
        mock_tracker_class.assert_called_once()

    def test_get_latency_tracker_returns_existing(self):
        """Test returning existing latency tracker."""
        import src.infrastructure.monitoring.telemetry as telemetry_module

        # Set up existing tracker
        existing_tracker = Mock()
        telemetry_module.latency_tracker = existing_tracker

        tracker = get_latency_tracker()

        assert tracker is existing_tracker

"""
Comprehensive unit tests for structured logging module.

Tests structured logging, log levels, sensitive data masking,
correlation tracking, and trading-specific logging features.
"""

import asyncio
import json
import logging
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.monitoring.logging import (
    LogSampler,
    LogSamplingConfig,
    SensitiveDataConfig,
    SensitiveDataMasker,
    TradingJSONFormatter,
    TradingLogFilter,
    TradingLogRecord,
    correlation_context,
    correlation_id_var,
    generate_correlation_id,
    get_correlation_id,
    get_trading_logger,
    log_order_filled,
    log_order_submitted,
    log_risk_breach,
    log_trading_operation,
    mask_sensitive_data,
    session_id_var,
    set_correlation_id,
    set_user_context,
    setup_structured_logging,
    user_context,
    user_id_var,
)


class TestLogSamplingConfig:
    """Test LogSamplingConfig configuration."""

    def test_default_config(self):
        """Test default sampling configuration."""
        config = LogSamplingConfig()

        assert config.price_update_sample_rate == 0.01
        assert config.market_data_sample_rate == 0.05
        assert config.heartbeat_sample_rate == 0.001
        assert "order_execution" in config.critical_operations
        assert "risk_breach" in config.critical_operations
        assert config.level_sample_rates["DEBUG"] == 0.1
        assert config.level_sample_rates["INFO"] == 1.0
        assert config.level_sample_rates["ERROR"] == 1.0

    def test_custom_config(self):
        """Test custom sampling configuration."""
        config = LogSamplingConfig()
        config.price_update_sample_rate = 0.1
        config.critical_operations.add("custom_critical")

        assert config.price_update_sample_rate == 0.1
        assert "custom_critical" in config.critical_operations


class TestSensitiveDataConfig:
    """Test SensitiveDataConfig configuration."""

    def test_default_patterns(self):
        """Test default sensitive data patterns."""
        config = SensitiveDataConfig()

        assert "api_key" in config.api_key_patterns[0]
        assert "secret" in config.api_key_patterns[1]
        assert "account_number" in config.financial_patterns[0]
        assert "ssn" in config.financial_patterns[2]
        assert "account_id" in config.trading_patterns[0]
        assert config.mask_replacement == "***MASKED***"
        assert "password" in config.excluded_fields

    def test_custom_patterns(self):
        """Test custom sensitive data patterns."""
        config = SensitiveDataConfig()
        config.api_key_patterns.append(r"custom_key")
        config.excluded_fields.add("custom_secret")

        assert r"custom_key" in config.api_key_patterns
        assert "custom_secret" in config.excluded_fields


class TestTradingLogRecord:
    """Test TradingLogRecord functionality."""

    def test_correlation_tracking(self):
        """Test correlation ID tracking in log records."""
        correlation_id = "test-correlation-123"
        correlation_id_var.set(correlation_id)

        record = TradingLogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None, "test_func"
        )

        assert record.correlation_id == correlation_id
        correlation_id_var.set(None)

    def test_user_context_tracking(self):
        """Test user context tracking in log records."""
        user_id_var.set("user-123")
        session_id_var.set("session-456")

        record = TradingLogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None, "test_func"
        )

        assert record.user_id == "user-123"
        assert record.session_id == "session-456"

        user_id_var.set(None)
        session_id_var.set(None)

    @patch("src.infrastructure.monitoring.logging.trace.get_current_span")
    def test_tracing_context(self, mock_get_span):
        """Test OpenTelemetry tracing context."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        mock_context = Mock()
        mock_context.trace_id = 123456789
        mock_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_context

        mock_get_span.return_value = mock_span

        record = TradingLogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None, "test_func"
        )

        assert record.trace_id == format(123456789, "032x")
        assert record.span_id == format(987654321, "016x")

    def test_trading_fields(self):
        """Test trading-specific fields."""
        record = TradingLogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None, "test_func"
        )

        record.symbol = "AAPL"
        record.order_id = "order-123"
        record.portfolio_id = "portfolio-456"
        record.strategy = "momentum"
        record.operation_type = "order_execution"

        assert record.symbol == "AAPL"
        assert record.order_id == "order-123"
        assert record.portfolio_id == "portfolio-456"
        assert record.strategy == "momentum"
        assert record.operation_type == "order_execution"


class TestSensitiveDataMasker:
    """Test SensitiveDataMasker functionality."""

    def test_mask_api_keys(self):
        """Test masking API keys."""
        config = SensitiveDataConfig()
        masker = SensitiveDataMasker(config)

        message = "Request with api_key=secret123 and authorization=Bearer token456"
        masked = masker.mask_message(message)

        assert "secret123" not in masked
        assert "token456" not in masked
        assert config.mask_replacement in masked

    def test_mask_financial_data(self):
        """Test masking financial data."""
        config = SensitiveDataConfig()
        masker = SensitiveDataMasker(config)

        message = "Account account_number=1234567890 with routing_number=987654321"
        masked = masker.mask_message(message)

        assert "1234567890" not in masked
        assert "987654321" not in masked
        assert config.mask_replacement in masked

    def test_mask_trading_data(self):
        """Test masking trading account data."""
        config = SensitiveDataConfig()
        masker = SensitiveDataMasker(config)

        message = "Trading account_id=TRD123456 for brokerage_account=BRK789"
        masked = masker.mask_message(message)

        assert "TRD123456" not in masked
        assert "BRK789" not in masked

    def test_mask_extra_fields(self):
        """Test masking extra log fields."""
        config = SensitiveDataConfig()
        masker = SensitiveDataMasker(config)

        extra = {
            "user_id": "user123",
            "api_key": "secret_key_value",
            "password": "should_be_excluded",
            "account_number": "1234567890",
            "normal_field": "normal_value",
            "nested": {"secret_key": "nested_secret"},
        }

        masked = masker.mask_extra_fields(extra)

        assert "password" not in masked  # Excluded entirely
        assert masked["api_key"] == config.mask_replacement
        assert masked["account_number"] == config.mask_replacement
        assert masked["normal_field"] == "normal_value"
        assert isinstance(masked["nested"], dict)
        assert masked["nested"]["secret_key"] == config.mask_replacement

    def test_empty_data(self):
        """Test masking empty data."""
        config = SensitiveDataConfig()
        masker = SensitiveDataMasker(config)

        assert masker.mask_message("") == ""
        assert masker.mask_extra_fields({}) == {}
        assert masker.mask_extra_fields(None) is None


class TestLogSampler:
    """Test LogSampler functionality."""

    def test_critical_operations_always_logged(self):
        """Test critical operations are never sampled."""
        config = LogSamplingConfig()
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "order_execution"
        record.levelno = logging.INFO
        record.levelname = "INFO"

        # Should always log critical operations
        for i in range(100):
            assert sampler.should_log(record) is True

    def test_error_level_always_logged(self):
        """Test ERROR and above always logged."""
        config = LogSamplingConfig()
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "some_operation"
        record.levelno = logging.ERROR
        record.levelname = "ERROR"

        # Should always log errors
        for i in range(100):
            assert sampler.should_log(record) is True

    def test_debug_level_sampling(self):
        """Test DEBUG level sampling."""
        config = LogSamplingConfig()
        config.level_sample_rates["DEBUG"] = 0.1  # 10%
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "debug_operation"
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"

        # Should sample approximately 10%
        logged_count = sum(1 for _ in range(100) if sampler.should_log(record))
        assert 5 <= logged_count <= 15  # Allow some variance

    def test_operation_specific_sampling(self):
        """Test operation-specific sampling rates."""
        config = LogSamplingConfig()
        config.price_update_sample_rate = 0.01  # 1%
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "price_update"
        record.levelno = logging.INFO
        record.levelname = "INFO"

        # Should sample approximately 1%
        logged_count = sum(1 for _ in range(1000) if sampler.should_log(record))
        assert 5 <= logged_count <= 15  # Allow variance around 10

    def test_market_data_sampling(self):
        """Test market data operation sampling."""
        config = LogSamplingConfig()
        config.market_data_sample_rate = 0.05  # 5%
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "market_data"
        record.levelno = logging.INFO
        record.levelname = "INFO"

        # Should sample approximately 5%
        logged_count = sum(1 for _ in range(200) if sampler.should_log(record))
        assert 5 <= logged_count <= 15  # Allow variance around 10

    def test_heartbeat_sampling(self):
        """Test heartbeat operation sampling."""
        config = LogSamplingConfig()
        config.heartbeat_sample_rate = 0.001  # 0.1%
        sampler = LogSampler(config)

        record = Mock()
        record.operation_type = "heartbeat"
        record.levelno = logging.INFO
        record.levelname = "INFO"

        # Should sample approximately 0.1%
        logged_count = sum(1 for _ in range(10000) if sampler.should_log(record))
        assert 5 <= logged_count <= 15  # Allow variance around 10


class TestTradingJSONFormatter:
    """Test TradingJSONFormatter functionality."""

    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        formatter = TradingJSONFormatter()

        record = logging.LogRecord(
            "test.logger",
            logging.INFO,
            "/path/test.py",
            10,
            "Test message",
            (),
            None,
            func="test_func",
        )

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test.logger"
        assert log_entry["message"] == "Test message"
        assert log_entry["module"] == "test"
        assert log_entry["function"] == "test_func"
        assert log_entry["line"] == 10

    def test_correlation_tracking_formatting(self):
        """Test correlation tracking in formatted output."""
        formatter = TradingJSONFormatter()

        record = TradingLogRecord(
            "test.logger",
            logging.INFO,
            "/path/test.py",
            10,
            "Test message",
            (),
            None,
            func="test_func",
        )
        record.correlation_id = "corr-123"
        record.user_id = "user-456"
        record.session_id = "session-789"

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert log_entry["correlation_id"] == "corr-123"
        assert log_entry["user_id"] == "user-456"
        assert log_entry["session_id"] == "session-789"

    def test_trading_fields_formatting(self):
        """Test trading-specific fields in formatted output."""
        formatter = TradingJSONFormatter()

        record = TradingLogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Order executed", (), None
        )
        record.symbol = "AAPL"
        record.order_id = "ORD123"
        record.portfolio_id = "PORT456"
        record.strategy = "momentum"
        record.operation_type = "order_execution"

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert "trading" in log_entry
        assert log_entry["trading"]["symbol"] == "AAPL"
        assert log_entry["trading"]["order_id"] == "ORD123"
        assert log_entry["trading"]["portfolio_id"] == "PORT456"
        assert log_entry["trading"]["strategy"] == "momentum"
        assert log_entry["trading"]["operation_type"] == "order_execution"

    def test_exception_formatting(self):
        """Test exception information formatting."""
        formatter = TradingJSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "test.logger",
                logging.ERROR,
                "/path/test.py",
                10,
                "Error occurred",
                (),
                sys.exc_info(),
            )

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert "exception" in log_entry
        assert log_entry["exception"]["type"] == "ValueError"
        assert log_entry["exception"]["message"] == "Test exception"
        assert "traceback" in log_entry["exception"]

    def test_extra_fields_formatting(self):
        """Test extra fields formatting."""
        formatter = TradingJSONFormatter()

        record = logging.LogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None
        )
        record.custom_field = "custom_value"
        record.numeric_field = 42
        record.decimal_field = Decimal("123.45")

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert "extra" in log_entry
        assert log_entry["extra"]["custom_field"] == "custom_value"
        assert log_entry["extra"]["numeric_field"] == 42
        assert log_entry["extra"]["decimal_field"] == 123.45

    def test_sensitive_data_masking(self):
        """Test sensitive data masking in formatter."""
        config = SensitiveDataConfig()
        formatter = TradingJSONFormatter(sensitive_data_config=config)

        record = logging.LogRecord(
            "test.logger",
            logging.INFO,
            "/path/test.py",
            10,
            "API call with api_key=secret123",
            (),
            None,
        )
        record.api_key = "another_secret"
        record.normal_field = "normal_value"

        formatted = formatter.format(record)
        log_entry = json.loads(formatted)

        assert "secret123" not in log_entry["message"]
        assert config.mask_replacement in log_entry["message"]
        assert log_entry["extra"]["api_key"] == config.mask_replacement
        assert log_entry["extra"]["normal_field"] == "normal_value"


class TestTradingLogFilter:
    """Test TradingLogFilter functionality."""

    def test_filter_with_sampling(self):
        """Test filter with log sampling."""
        config = LogSamplingConfig()
        config.price_update_sample_rate = 0.5
        sampler = LogSampler(config)
        filter = TradingLogFilter(sampler)

        record = Mock(spec=logging.LogRecord)
        record.operation_type = "price_update"
        record.levelno = logging.INFO
        record.levelname = "INFO"

        # Should filter approximately 50%
        passed_count = sum(1 for _ in range(100) if filter.filter(record))
        assert 40 <= passed_count <= 60

    def test_filter_without_sampling(self):
        """Test filter without sampling."""
        filter = TradingLogFilter(sampler=None)

        record = Mock(spec=logging.LogRecord)

        # Should always pass without sampler
        for _ in range(10):
            assert filter.filter(record) is True

    def test_convert_to_trading_record(self):
        """Test conversion to TradingLogRecord."""
        filter = TradingLogFilter()

        record = logging.LogRecord(
            "test.logger", logging.INFO, "/path/test.py", 10, "Test message", (), None
        )
        record.custom_attr = "custom_value"

        # The filter should handle conversion internally
        result = filter.filter(record)
        assert result is True


class TestCorrelationManagement:
    """Test correlation ID management functions."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        assert corr_id is not None
        assert isinstance(corr_id, str)
        assert len(corr_id) == 36  # UUID format

        # Should generate unique IDs
        corr_id2 = generate_correlation_id()
        assert corr_id != corr_id2

    def test_set_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

        # Clean up
        correlation_id_var.set(None)

    def test_correlation_context(self):
        """Test correlation context manager."""
        with correlation_context("test-context-123") as corr_id:
            assert corr_id == "test-context-123"
            assert get_correlation_id() == "test-context-123"

        # Should be cleared after context
        assert get_correlation_id() is None

    def test_correlation_context_auto_generate(self):
        """Test correlation context with auto-generated ID."""
        with correlation_context() as corr_id:
            assert corr_id is not None
            assert get_correlation_id() == corr_id

        assert get_correlation_id() is None


class TestUserContextManagement:
    """Test user context management functions."""

    def test_set_user_context(self):
        """Test setting user context."""
        set_user_context("user-123", "session-456")
        assert user_id_var.get() == "user-123"
        assert session_id_var.get() == "session-456"

        # Clean up
        user_id_var.set(None)
        session_id_var.set(None)

    def test_user_context_manager(self):
        """Test user context manager."""
        with user_context("user-789", "session-012"):
            assert user_id_var.get() == "user-789"
            assert session_id_var.get() == "session-012"

        # Should be cleared after context
        assert user_id_var.get() is None
        assert session_id_var.get() is None

    def test_user_context_without_session(self):
        """Test user context without session ID."""
        with user_context("user-999"):
            assert user_id_var.get() == "user-999"
            assert session_id_var.get() is None

        assert user_id_var.get() is None


class TestTradingLogger:
    """Test trading-specific logger functions."""

    def test_get_trading_logger(self):
        """Test getting a trading logger."""
        logger = get_trading_logger("test.module")

        assert logger.name == "test.module"
        assert hasattr(logger, "log_order_event")
        assert hasattr(logger, "log_market_data_event")
        assert hasattr(logger, "log_risk_event")
        assert hasattr(logger, "log_portfolio_event")

    @patch("logging.Logger.log")
    def test_log_order_event(self, mock_log):
        """Test logging order events."""
        logger = get_trading_logger("test.module")

        logger.log_order_event(
            logging.INFO, "Order placed", order_id="ORD123", symbol="AAPL", quantity=100
        )

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == "Order placed"
        assert call_args[1]["extra"]["operation_type"] == "order_execution"
        assert call_args[1]["extra"]["order_id"] == "ORD123"
        assert call_args[1]["extra"]["symbol"] == "AAPL"
        assert call_args[1]["extra"]["quantity"] == 100

    @patch("logging.Logger.log")
    def test_log_market_data_event(self, mock_log):
        """Test logging market data events."""
        logger = get_trading_logger("test.module")

        logger.log_market_data_event(logging.INFO, "Price update", symbol="AAPL", price=150.50)

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]["extra"]["operation_type"] == "market_data"
        assert call_args[1]["extra"]["symbol"] == "AAPL"
        assert call_args[1]["extra"]["price"] == 150.50

    @patch("logging.Logger.log")
    def test_log_risk_event(self, mock_log):
        """Test logging risk events."""
        logger = get_trading_logger("test.module")

        logger.log_risk_event(
            logging.WARNING, "Risk limit approaching", portfolio_id="PORT123", risk_level=0.85
        )

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]["extra"]["operation_type"] == "risk_calculation"
        assert call_args[1]["extra"]["portfolio_id"] == "PORT123"
        assert call_args[1]["extra"]["risk_level"] == 0.85

    @patch("logging.Logger.log")
    def test_log_portfolio_event(self, mock_log):
        """Test logging portfolio events."""
        logger = get_trading_logger("test.module")

        logger.log_portfolio_event(logging.INFO, "Portfolio rebalanced", portfolio_id="PORT456")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]["extra"]["operation_type"] == "portfolio_operation"
        assert call_args[1]["extra"]["portfolio_id"] == "PORT456"


class TestLogTradingOperationDecorator:
    """Test log_trading_operation decorator."""

    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    def test_sync_function_success(self, mock_get_logger):
        """Test decorator on successful sync function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @log_trading_operation("test_operation", symbol="AAPL")
        def test_func(x, y):
            return x + y

        result = test_func(1, 2)

        assert result == 3
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert "completed successfully" in call_args[0][1]
        assert call_args[1]["extra"]["operation_type"] == "test_operation"
        assert call_args[1]["extra"]["symbol"] == "AAPL"
        assert call_args[1]["extra"]["status"] == "success"

    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    def test_sync_function_error(self, mock_get_logger):
        """Test decorator on sync function error."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @log_trading_operation("test_operation")
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_func()

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert "failed" in call_args[0][1]
        assert call_args[1]["extra"]["status"] == "error"
        assert call_args[1]["extra"]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    async def test_async_function_success(self, mock_get_logger):
        """Test decorator on successful async function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @log_trading_operation("async_operation")
        @pytest.mark.asyncio
        async def test_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await test_func(5)

        assert result == 10
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[1]["extra"]["status"] == "success"
        assert call_args[1]["extra"]["duration_ms"] > 0

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    async def test_async_function_error(self, mock_get_logger):
        """Test decorator on async function error."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @log_trading_operation("async_operation")
        @pytest.mark.asyncio
        async def test_func():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await test_func()

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[1]["extra"]["status"] == "error"
        assert call_args[1]["extra"]["error_type"] == "RuntimeError"


class TestMaskSensitiveData:
    """Test mask_sensitive_data utility function."""

    def test_mask_string(self):
        """Test masking sensitive data in string."""
        data = "API key is api_key=secret123"
        masked = mask_sensitive_data(data)
        assert "secret123" not in masked
        assert "***MASKED***" in masked

    def test_mask_dict(self):
        """Test masking sensitive data in dictionary."""
        data = {"user": "john", "api_key": "secret456", "account_number": "1234567890"}
        masked = mask_sensitive_data(data)
        assert masked["user"] == "john"
        assert masked["api_key"] == "***MASKED***"
        assert masked["account_number"] == "***MASKED***"

    def test_mask_with_custom_config(self):
        """Test masking with custom configuration."""
        config = SensitiveDataConfig()
        config.mask_replacement = "[REDACTED]"

        data = "secret_key=value123"
        masked = mask_sensitive_data(data, config)
        assert "[REDACTED]" in masked
        assert "value123" not in masked


class TestSetupStructuredLogging:
    """Test setup_structured_logging function."""

    @patch("logging.setLogRecordFactory")
    @patch("logging.getLogger")
    def test_setup_json_format(self, mock_get_logger, mock_set_factory):
        """Test setting up JSON formatted logging."""
        mock_root_logger = Mock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger

        setup_structured_logging(level="INFO", format_type="json", enable_sampling=False)

        mock_set_factory.assert_called_once_with(TradingLogRecord)
        mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
        assert len(mock_root_logger.addHandler.call_args_list) == 1

    @patch("logging.getLogger")
    def test_setup_text_format(self, mock_get_logger):
        """Test setting up text formatted logging."""
        mock_root_logger = Mock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger

        setup_structured_logging(level="DEBUG", format_type="text", enable_sampling=False)

        mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)

    @patch("logging.FileHandler")
    @patch("logging.getLogger")
    def test_setup_with_file(self, mock_get_logger, mock_file_handler):
        """Test setting up logging with file output."""
        mock_root_logger = Mock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger

        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler

        setup_structured_logging(level="INFO", format_type="json", log_file="/tmp/test.log")

        mock_file_handler.assert_called_once_with("/tmp/test.log")
        # Should add both console and file handlers
        assert mock_root_logger.addHandler.call_count == 2

    @patch("logging.getLogger")
    def test_setup_with_sampling(self, mock_get_logger):
        """Test setting up logging with sampling."""
        mock_root_logger = Mock()
        mock_root_logger.handlers = []
        mock_get_logger.return_value = mock_root_logger

        sampling_config = LogSamplingConfig()
        sampling_config.price_update_sample_rate = 0.001

        setup_structured_logging(
            level="INFO", enable_sampling=True, sampling_config=sampling_config
        )

        # Handler should have filter attached
        handler_calls = mock_root_logger.addHandler.call_args_list
        assert len(handler_calls) > 0


class TestConvenienceFunctions:
    """Test convenience logging functions."""

    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    def test_log_order_submitted(self, mock_get_logger):
        """Test log_order_submitted function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_order_submitted(
            order_id="ORD123", symbol="AAPL", side="BUY", quantity=100, price=150.50
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Order submitted"
        extra = call_args[1]["extra"]
        assert extra["operation_type"] == "order_execution"
        assert extra["order_id"] == "ORD123"
        assert extra["symbol"] == "AAPL"
        assert extra["order_side"] == "BUY"
        assert extra["quantity"] == 100
        assert extra["price"] == 150.50

    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    def test_log_order_filled(self, mock_get_logger):
        """Test log_order_filled function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_order_filled(
            order_id="ORD456", symbol="GOOGL", quantity=50, price=2500.00, commission=1.50
        )

        mock_logger.info.assert_called_once()
        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["fill_status"] == "filled"
        assert extra["commission"] == 1.50

    @patch("src.infrastructure.monitoring.logging.get_trading_logger")
    def test_log_risk_breach(self, mock_get_logger):
        """Test log_risk_breach function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        log_risk_breach(
            risk_type="position_size", current_value=0.25, limit_value=0.20, portfolio_id="PORT789"
        )

        mock_logger.warning.assert_called_once()
        extra = mock_logger.warning.call_args[1]["extra"]
        assert extra["operation_type"] == "risk_breach"
        assert extra["risk_type"] == "position_size"
        assert extra["current_value"] == 0.25
        assert extra["limit_value"] == 0.20
        assert extra["severity"] == "high"
        assert extra["portfolio_id"] == "PORT789"

"""
Structured Logging for AI Trading System

Production-grade logging with JSON structured logs, correlation IDs,
trading-specific log fields, sensitive data masking, and log sampling
for high-frequency operations.
"""

import asyncio
import json
import logging
import logging.config
import re
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import wraps
from typing import Any

from opentelemetry import trace

# Context variables for correlation tracking
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)


@dataclass
class LogSamplingConfig:
    """Configuration for log sampling."""

    # High frequency operations sampling
    price_update_sample_rate: float = 0.01  # Sample 1% of price updates
    market_data_sample_rate: float = 0.05  # Sample 5% of market data logs
    heartbeat_sample_rate: float = 0.001  # Sample 0.1% of heartbeats

    # Critical operations (never sampled)
    critical_operations: set[str] = field(
        default_factory=lambda: {
            "order_execution",
            "risk_breach",
            "error",
            "exception",
            "authentication",
            "authorization",
            "security_event",
        }
    )

    # Sampling thresholds by log level
    level_sample_rates: dict[str, float] = field(
        default_factory=lambda: {
            "DEBUG": 0.1,
            "INFO": 1.0,
            "WARNING": 1.0,
            "ERROR": 1.0,
            "CRITICAL": 1.0,
        }
    )


@dataclass
class SensitiveDataConfig:
    """Configuration for sensitive data masking."""

    # API keys and secrets
    api_key_patterns: list[str] = field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"secret[_-]?key",
            r"access[_-]?token",
            r"bearer[_-]?token",
            r"authorization",
            r"x-api-key",
        ]
    )

    # Financial data patterns
    financial_patterns: list[str] = field(
        default_factory=lambda: [
            r"account[_-]?number",
            r"routing[_-]?number",
            r"ssn",
            r"social[_-]?security",
            r"credit[_-]?card",
            r"card[_-]?number",
        ]
    )

    # Trading specific patterns
    trading_patterns: list[str] = field(
        default_factory=lambda: [
            r"account[_-]?id",
            r"brokerage[_-]?account",
            r"trading[_-]?account",
        ]
    )

    # Replacement text
    mask_replacement: str = "***MASKED***"

    # Fields to completely exclude from logs
    excluded_fields: set[str] = field(
        default_factory=lambda: {"password", "passwd", "secret", "private_key", "token"}
    )


class TradingLogRecord(logging.LogRecord):
    """Enhanced log record with trading-specific fields."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Add correlation tracking
        self.correlation_id = correlation_id_var.get()
        self.user_id = user_id_var.get()
        self.session_id = session_id_var.get()

        # Add tracing context
        span = trace.get_current_span()
        if span.is_recording():
            span_context = span.get_span_context()
            self.trace_id = format(span_context.trace_id, "032x") if span_context.trace_id else None
            self.span_id = format(span_context.span_id, "016x") if span_context.span_id else None
        else:
            self.trace_id = None
            self.span_id = None

        # Trading-specific fields (set by formatters)
        self.symbol = getattr(self, "symbol", None)
        self.order_id = getattr(self, "order_id", None)
        self.portfolio_id = getattr(self, "portfolio_id", None)
        self.strategy = getattr(self, "strategy", None)
        self.operation_type = getattr(self, "operation_type", None)


class SensitiveDataMasker:
    """Masks sensitive data in log messages and extra fields."""

    def __init__(self, config: SensitiveDataConfig) -> None:
        self.config = config
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[re.Pattern[str]]:
        """Compile all sensitive data patterns."""
        all_patterns = (
            self.config.api_key_patterns
            + self.config.financial_patterns
            + self.config.trading_patterns
        )

        compiled = []
        for pattern in all_patterns:
            try:
                # Create pattern that matches key:value or key=value pairs
                full_pattern = rf'("{pattern}":\s*"[^"]*"|{pattern}=\S+|{pattern}:\s*\S+)'
                compiled.append(re.compile(full_pattern, re.IGNORECASE))
            except re.error as e:
                logging.getLogger(__name__).warning(f"Invalid regex pattern '{pattern}': {e}")

        return compiled

    def mask_message(self, message: str) -> str:
        """Mask sensitive data in log message."""
        masked_message = message

        for pattern in self._compiled_patterns:
            masked_message = pattern.sub(lambda m: self._replace_value(m.group(0)), masked_message)

        return masked_message

    def mask_extra_fields(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive data in extra log fields."""
        if not extra:
            return extra

        masked_extra = {}

        for key, value in extra.items():
            # Exclude sensitive fields entirely
            if key.lower() in self.config.excluded_fields:
                continue

            # Mask sensitive field values
            if self._is_sensitive_field(key):
                masked_extra[key] = self.config.mask_replacement
            elif isinstance(value, str):
                masked_extra[key] = self.mask_message(value)
            elif isinstance(value, dict):
                masked_extra[key] = self.mask_extra_fields(value)  # type: ignore[assignment]
            else:
                masked_extra[key] = value

        return masked_extra

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data."""
        field_lower = field_name.lower()

        all_patterns = (
            self.config.api_key_patterns
            + self.config.financial_patterns
            + self.config.trading_patterns
        )

        for pattern in all_patterns:
            if re.search(pattern, field_lower):
                return True

        return False

    def _replace_value(self, match: str) -> str:
        """Replace matched value with mask."""
        if ":" in match:
            key_part = match.split(":", 1)[0]
            return f'{key_part}: "{self.config.mask_replacement}"'
        elif "=" in match:
            key_part = match.split("=", 1)[0]
            return f"{key_part}={self.config.mask_replacement}"
        else:
            return self.config.mask_replacement


class LogSampler:
    """Implements log sampling for high-frequency operations."""

    def __init__(self, config: LogSamplingConfig) -> None:
        self.config = config
        self._counters: dict[str, int] = {}

    def should_log(self, record: logging.LogRecord) -> bool:
        """Determine if a log record should be emitted."""

        # Never sample critical operations
        operation_type = getattr(record, "operation_type", "")
        if operation_type in self.config.critical_operations:
            return True

        # Never sample ERROR and above
        if record.levelno >= logging.ERROR:
            return True

        # Apply level-based sampling
        level_name = record.levelname
        level_sample_rate = self.config.level_sample_rates.get(level_name, 1.0)

        if level_sample_rate < 1.0:
            counter_key = f"{level_name}:{operation_type}"
            count = self._counters.get(counter_key, 0) + 1
            self._counters[counter_key] = count

            # Sample based on counter
            return (count % int(1 / level_sample_rate)) == 0

        # Apply operation-specific sampling
        if operation_type == "price_update":
            return self._should_sample("price_update", self.config.price_update_sample_rate)
        elif operation_type == "market_data":
            return self._should_sample("market_data", self.config.market_data_sample_rate)
        elif operation_type == "heartbeat":
            return self._should_sample("heartbeat", self.config.heartbeat_sample_rate)

        return True

    def _should_sample(self, operation: str, sample_rate: float) -> bool:
        """Check if operation should be sampled."""
        if sample_rate >= 1.0:
            return True

        count = self._counters.get(operation, 0) + 1
        self._counters[operation] = count

        return (count % int(1 / sample_rate)) == 0


class TradingJSONFormatter(logging.Formatter):
    """JSON formatter for structured trading logs."""

    def __init__(
        self,
        sensitive_data_config: SensitiveDataConfig | None = None,
        include_extra: bool = True,
        sort_keys: bool = True,
    ):
        super().__init__()
        self.include_extra = include_extra
        self.sort_keys = sort_keys
        self.masker = SensitiveDataMasker(sensitive_data_config or SensitiveDataConfig())

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self.masker.mask_message(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add correlation tracking
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "user_id") and record.user_id:
            log_entry["user_id"] = record.user_id
        if hasattr(record, "session_id") and record.session_id:
            log_entry["session_id"] = record.session_id

        # Add distributed tracing context
        if hasattr(record, "trace_id") and record.trace_id:
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "span_id") and record.span_id:
            log_entry["span_id"] = record.span_id

        # Add trading-specific fields
        trading_fields = {}
        if hasattr(record, "symbol") and record.symbol:
            trading_fields["symbol"] = record.symbol
        if hasattr(record, "order_id") and record.order_id:
            trading_fields["order_id"] = record.order_id
        if hasattr(record, "portfolio_id") and record.portfolio_id:
            trading_fields["portfolio_id"] = record.portfolio_id
        if hasattr(record, "strategy") and record.strategy:
            trading_fields["strategy"] = record.strategy
        if hasattr(record, "operation_type") and record.operation_type:
            trading_fields["operation_type"] = record.operation_type

        if trading_fields:
            log_entry["trading"] = trading_fields

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields (custom data)
        if self.include_extra and hasattr(record, "__dict__"):
            # Filter out standard fields and add custom fields
            standard_fields = {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "correlation_id",
                "user_id",
                "session_id",
                "trace_id",
                "span_id",
                "symbol",
                "order_id",
                "portfolio_id",
                "strategy",
                "operation_type",
            }

            extra = {
                key: self._serialize_value(value)
                for key, value in record.__dict__.items()
                if key not in standard_fields and not key.startswith("_")
            }

            if extra:
                # Mask sensitive data in extra fields
                extra = self.masker.mask_extra_fields(extra)
                log_entry["extra"] = extra

        return json.dumps(log_entry, sort_keys=self.sort_keys, default=self._serialize_value)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize complex values for JSON output."""
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (set, frozenset)):
            return list(value)
        elif hasattr(value, "__dict__"):
            return str(value)
        return value


class TradingLogFilter(logging.Filter):
    """Filter for trading-specific log processing."""

    def __init__(self, sampler: LogSampler | None = None) -> None:
        super().__init__()
        self.sampler = sampler

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on sampling and other criteria."""

        # Convert to TradingLogRecord if not already
        if not isinstance(record, TradingLogRecord):
            # Copy attributes to new TradingLogRecord
            trading_record = TradingLogRecord(
                record.name,
                record.levelno,
                record.pathname,
                record.lineno,
                record.msg,
                record.args,
                record.exc_info,
                record.funcName,
                record.stack_info,
            )

            # Copy all attributes
            for attr_name in dir(record):
                if not attr_name.startswith("_") and hasattr(record, attr_name):
                    setattr(trading_record, attr_name, getattr(record, attr_name))

            record = trading_record

        # Apply sampling
        if self.sampler and not self.sampler.should_log(record):
            return False

        return True


# Correlation ID management
def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get current correlation ID."""
    return correlation_id_var.get()


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Generator[str, None, None]:
    """Context manager for correlation ID scope."""
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    token = correlation_id_var.set(correlation_id)
    try:
        yield correlation_id
    finally:
        correlation_id_var.reset(token)


# User context management
def set_user_context(user_id: str, session_id: str | None = None) -> None:
    """Set user context for current request."""
    user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


@contextmanager
def user_context(user_id: str, session_id: str | None = None) -> Generator[None, None, None]:
    """Context manager for user context scope."""
    user_token = user_id_var.set(user_id)
    session_token = None
    if session_id:
        session_token = session_id_var.set(session_id)

    try:
        yield
    finally:
        user_id_var.reset(user_token)
        if session_token:
            session_id_var.reset(session_token)


# Trading-specific logging functions
def get_trading_logger(name: str) -> logging.Logger:
    """Get a logger configured for trading operations."""
    logger = logging.getLogger(name)

    # Add trading-specific methods
    def log_order_event(
        level: int,
        msg: str,
        order_id: str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        extra = {
            "operation_type": "order_execution",
            "order_id": order_id,
            "symbol": symbol,
            **kwargs,
        }
        logger.log(level, msg, extra=extra)

    def log_market_data_event(
        level: int, msg: str, symbol: str | None = None, **kwargs: Any
    ) -> None:
        extra = {"operation_type": "market_data", "symbol": symbol, **kwargs}
        logger.log(level, msg, extra=extra)

    def log_risk_event(
        level: int, msg: str, portfolio_id: str | None = None, **kwargs: Any
    ) -> None:
        extra = {"operation_type": "risk_calculation", "portfolio_id": portfolio_id, **kwargs}
        logger.log(level, msg, extra=extra)

    def log_portfolio_event(
        level: int, msg: str, portfolio_id: str | None = None, **kwargs: Any
    ) -> None:
        extra = {"operation_type": "portfolio_operation", "portfolio_id": portfolio_id, **kwargs}
        logger.log(level, msg, extra=extra)

    # Attach methods to logger
    logger.log_order_event = log_order_event  # type: ignore[attr-defined]
    logger.log_market_data_event = log_market_data_event  # type: ignore[attr-defined]
    logger.log_risk_event = log_risk_event  # type: ignore[attr-defined]
    logger.log_portfolio_event = log_portfolio_event  # type: ignore[attr-defined]

    return logger


def log_trading_operation(
    operation_type: str, level: int = logging.INFO, **trading_fields: Any
) -> Any:
    """Decorator for logging trading operations."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_trading_logger(func.__module__)

            extra = {"operation_type": operation_type, **trading_fields}

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                logger.log(
                    level,
                    f"Trading operation {operation_type} completed successfully",
                    extra={**extra, "duration_ms": duration * 1000, "status": "success"},
                )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time

                logger.log(
                    logging.ERROR,
                    f"Trading operation {operation_type} failed: {e}",
                    extra={
                        **extra,
                        "duration_ms": duration * 1000,
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_trading_logger(func.__module__)

            extra = {"operation_type": operation_type, **trading_fields}

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                logger.log(
                    level,
                    f"Trading operation {operation_type} completed successfully",
                    extra={**extra, "duration_ms": duration * 1000, "status": "success"},
                )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time

                logger.log(
                    logging.ERROR,
                    f"Trading operation {operation_type} failed: {e}",
                    extra={
                        **extra,
                        "duration_ms": duration * 1000,
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def mask_sensitive_data(
    data: str | dict[str, Any], config: SensitiveDataConfig | None = None
) -> str | dict[str, Any]:
    """Utility function to mask sensitive data."""
    masker = SensitiveDataMasker(config or SensitiveDataConfig())

    if isinstance(data, str):
        return masker.mask_message(data)
    else:  # data is Dict[str, Any]
        return masker.mask_extra_fields(data)


def setup_structured_logging(
    level: str = "INFO",
    format_type: str = "json",
    enable_sampling: bool = True,
    sampling_config: LogSamplingConfig | None = None,
    sensitive_data_config: SensitiveDataConfig | None = None,
    log_file: str | None = None,
) -> None:
    """
    Setup structured logging for the trading system.

    Args:
        level: Logging level
        format_type: Formatter type ('json' or 'text')
        enable_sampling: Whether to enable log sampling
        sampling_config: Log sampling configuration
        sensitive_data_config: Sensitive data masking configuration
        log_file: Optional log file path
    """

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter: TradingJSONFormatter | logging.Formatter
    if format_type == "json":
        formatter = TradingJSONFormatter(sensitive_data_config)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create filter with sampling
    log_filter = None
    if enable_sampling:
        sampler = LogSampler(sampling_config or LogSamplingConfig())
        log_filter = TradingLogFilter(sampler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if log_filter:
        console_handler.addFilter(log_filter)

    # File handler (if specified)
    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        if log_filter:
            file_handler.addFilter(log_filter)

    # Configure root logger
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)

    if file_handler:
        root_logger.addHandler(file_handler)

    # Set record factory to use TradingLogRecord
    logging.setLogRecordFactory(TradingLogRecord)

    logging.info("Structured logging configured successfully")


# Convenience functions for common trading log patterns
def log_order_submitted(
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float | None = None,
    **kwargs: Any,
) -> None:
    """Log order submission."""
    logger = get_trading_logger(__name__)
    extra = {
        "operation_type": "order_execution",
        "order_id": order_id,
        "symbol": symbol,
        "order_side": side,
        "quantity": quantity,
        **(kwargs if kwargs else {}),
    }
    if price:
        extra["price"] = price

    logger.info("Order submitted", extra=extra)


def log_order_filled(
    order_id: str, symbol: str, quantity: float, price: float, **kwargs: Any
) -> None:
    """Log order fill."""
    logger = get_trading_logger(__name__)
    extra = {
        "operation_type": "order_execution",
        "order_id": order_id,
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "fill_status": "filled",
        **(kwargs if kwargs else {}),
    }

    logger.info("Order filled", extra=extra)


def log_risk_breach(
    risk_type: str, current_value: float, limit_value: float, **kwargs: Any
) -> None:
    """Log risk limit breach."""
    logger = get_trading_logger(__name__)
    extra = {
        "operation_type": "risk_breach",
        "risk_type": risk_type,
        "current_value": current_value,
        "limit_value": limit_value,
        "severity": "high",
        **(kwargs if kwargs else {}),
    }

    logger.warning("Risk limit breached", extra=extra)

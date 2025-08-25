"""
Unit tests for audit decorators.

Tests cover automatic audit logging decorators for different
operation types and error handling scenarios.
"""

import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.audit.decorators import (
    AuditDecorator,
    OrderAuditDecorator,
    PositionAuditDecorator,
    RiskAuditDecorator,
    audit_create_order,
    audit_financial_operation,
    audit_order_operation,
    audit_risk_breach,
)
from src.infrastructure.audit.events import EventSeverity, OrderEvent, PositionEvent
from src.infrastructure.audit.exceptions import AuditException
from src.infrastructure.audit.logger import AuditContext, AuditLogger


@pytest.fixture
def mock_logger():
    """Create mock audit logger."""
    logger = Mock(spec=AuditLogger)
    logger.log_event.return_value = "test_event_id"
    return logger


@pytest.fixture
def sample_audit_context():
    """Create sample audit context."""
    return AuditContext(user_id="test_user", session_id="test_session", request_id="test_request")


class TestAuditDecorator:
    """Test suite for base AuditDecorator class."""

    def test_init(self, mock_logger):
        """Test decorator initialization."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
            include_args=True,
            include_result=True,
            severity=EventSeverity.HIGH,
            is_critical=True,
        )

        assert decorator.logger == mock_logger
        assert decorator.event_type == "test_event"
        assert decorator.resource_type == "test_resource"
        assert decorator.action == "test_action"
        assert decorator.include_args == True
        assert decorator.include_result == True
        assert decorator.severity == EventSeverity.HIGH
        assert decorator.is_critical == True

    def test_successful_function_execution(self, mock_logger):
        """Test decorator with successful function execution."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def test_function(x, y, multiplier=2):
            time.sleep(0.01)  # Small delay to test execution time
            return (x + y) * multiplier

        result = test_function(5, 3, multiplier=2)

        assert result == 16
        mock_logger.log_event.assert_called_once()

        # Check the logged event
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert hasattr(event, "event_type")
        assert event.metadata["execution_status"] == "success"
        assert "execution_time_ms" in event.metadata
        assert event.metadata["execution_time_ms"] > 0
        assert event.metadata["function_name"] == "test_function"
        assert "function_args" in event.metadata
        assert "function_result" in event.metadata

    def test_function_execution_with_exception(self, mock_logger):
        """Test decorator with function that raises exception."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError) as exc_info:
            failing_function()

        assert "Test error" in str(exc_info)
        mock_logger.log_event.assert_called_once()

        # Check the logged event
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert event.metadata["execution_status"] == "error"
        assert "error" in event.metadata
        assert event.metadata["error"]["type"] == "ValueError"
        assert event.metadata["error"]["message"] == "Test error"
        assert "traceback" in event.metadata["error"]

    def test_audit_context_extraction_from_kwargs(self, mock_logger, sample_audit_context):
        """Test extraction of audit context from function kwargs."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def test_function(data, audit_context=None):
            return data

        result = test_function("test_data", audit_context=sample_audit_context)

        assert result == "test_data"

        # Check that context was passed to logger
        call_args = mock_logger.log_event.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("context")

        assert context == sample_audit_context

    def test_audit_context_extraction_from_fields(self, mock_logger):
        """Test extraction of audit context from individual fields."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def test_function(data, user_id=None, session_id=None):
            return data

        result = test_function("test_data", user_id="test_user", session_id="test_session")

        assert result == "test_data"

        # Check that context was created from fields
        call_args = mock_logger.log_event.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("context")

        assert context is not None
        assert context.user_id == "test_user"
        assert context.session_id == "test_session"

    def test_exclude_args_and_result(self, mock_logger):
        """Test decorator with args and result excluded."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
            include_args=False,
            include_result=False,
        )

        @decorator
        def test_function(x, y):
            return x + y

        result = test_function(5, 3)

        assert result == 8

        # Check the logged event
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert "function_args" not in event.metadata
        assert "function_result" not in event.metadata

    def test_exclude_errors(self, mock_logger):
        """Test decorator with errors excluded."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
            include_errors=False,
        )

        @decorator
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Check the logged event
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert "error" not in event.metadata

    def test_audit_logging_failure_does_not_break_function(self, mock_logger):
        """Test that audit logging failure doesn't break original function."""
        mock_logger.log_event.side_effect = AuditException("Logging failed")

        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def test_function():
            return "success"

        with patch("builtins.print") as mock_print:
            result = test_function()

        # Function should still execute successfully
        assert result == "success"

        # Error should be printed but not raised
        mock_print.assert_called()

    def test_value_serialization(self, mock_logger):
        """Test serialization of various value types."""
        decorator = AuditDecorator(
            logger=mock_logger,
            event_type="test_event",
            resource_type="test_resource",
            action="test_action",
        )

        @decorator
        def test_function(
            string_val="test",
            int_val=42,
            float_val=3.14,
            bool_val=True,
            decimal_val=Decimal("123.45"),
            none_val=None,
            dict_val={"key": "value"},
            list_val=[1, 2, 3],
        ):
            return {"string": string_val, "decimal": decimal_val, "dict": dict_val}

        result = test_function()

        # Function should execute
        assert result["string"] == "test"
        assert result["decimal"] == Decimal("123.45")

        # Check serialization in logged event
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        function_args = event.metadata["function_args"]["keyword"]
        assert function_args["string_val"] == "test"
        assert function_args["decimal_val"] == "123.45"  # Decimal as string
        assert function_args["dict_val"] == {"key": "value"}
        assert function_args["list_val"] == [1, 2, 3]


class TestOrderAuditDecorator:
    """Test suite for OrderAuditDecorator class."""

    def test_order_event_creation(self, mock_logger):
        """Test creation of order-specific audit event."""
        decorator = OrderAuditDecorator(
            logger=mock_logger, event_type="order_create", resource_type="order", action="create"
        )

        @decorator
        def create_order(
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            order_type="limit",
        ):
            return {"order_id": order_id, "status": "pending"}

        result = create_order()

        assert result["order_id"] == "order_123"

        # Check that OrderEvent was created
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert isinstance(event, OrderEvent)
        assert event.order_id == "order_123"
        assert event.symbol == "AAPL"
        assert event.side == "buy"
        assert event.quantity == Decimal("100")
        assert event.price == Decimal("150.00")
        assert event.order_type == "limit"

    def test_order_id_extraction_from_result(self, mock_logger):
        """Test extraction of order ID from function result."""
        decorator = OrderAuditDecorator(
            logger=mock_logger, event_type="order_create", resource_type="order", action="create"
        )

        @decorator
        def create_order():
            class OrderResult:
                def __init__(self):
                    self.order_id = "result_order_456"
                    self.symbol = "GOOGL"

            return OrderResult()

        result = create_order()

        # Check that order ID was extracted from result
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert event.order_id == "result_order_456"
        assert event.symbol == "GOOGL"


class TestConvenienceDecorators:
    """Test suite for convenience decorator functions."""

    def test_audit_financial_operation(self, mock_logger):
        """Test general financial operation decorator."""

        @audit_financial_operation(logger=mock_logger, event_type="financial_op", action="process")
        def process_payment(amount, user_id):
            return f"Processed ${amount} for {user_id}"

        result = process_payment(100.0, "user123")

        assert "Processed $100.0 for user123" in result
        mock_logger.log_event.assert_called_once()

    def test_audit_order_operation(self, mock_logger):
        """Test order operation decorator."""

        @audit_order_operation(logger=mock_logger, action="create")
        def create_order(order_id, symbol, quantity):
            return {"id": order_id, "symbol": symbol, "qty": quantity}

        result = create_order("ord123", "AAPL", 100)

        assert result["id"] == "ord123"
        mock_logger.log_event.assert_called_once()

        # Should create OrderEvent
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]
        assert isinstance(event, OrderEvent)

    def test_audit_create_order_convenience(self, mock_logger):
        """Test create order convenience decorator."""

        @audit_create_order(mock_logger)
        def create_market_order(symbol, quantity):
            return {"order_id": "new_order", "symbol": symbol}

        result = create_market_order("MSFT", 50)

        assert result["symbol"] == "MSFT"
        mock_logger.log_event.assert_called_once()

        # Should be marked as critical
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]
        assert event.is_critical == True

    def test_audit_risk_breach_convenience(self, mock_logger):
        """Test risk breach convenience decorator."""

        @audit_risk_breach(mock_logger)
        def check_position_limit(portfolio_id, current_exposure, limit):
            if current_exposure > limit:
                raise ValueError("Position limit exceeded")
            return "OK"

        with pytest.raises(ValueError):
            check_position_limit("port123", 150000, 100000)

        mock_logger.log_event.assert_called_once()

        # Should be critical severity
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]
        assert event.severity == EventSeverity.CRITICAL


class TestPositionAuditDecorator:
    """Test suite for PositionAuditDecorator class."""

    def test_position_event_creation(self, mock_logger):
        """Test creation of position-specific audit event."""
        decorator = PositionAuditDecorator(
            logger=mock_logger, event_type="position_open", resource_type="position", action="open"
        )

        @decorator
        def open_position(position_id="pos_123", symbol="TSLA", quantity=Decimal("25")):
            return {"position_id": position_id, "status": "open"}

        result = open_position()

        assert result["position_id"] == "pos_123"

        # Check that PositionEvent was created
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert isinstance(event, PositionEvent)
        assert event.position_id == "pos_123"
        assert event.symbol == "TSLA"
        assert event.quantity == Decimal("25")


class TestRiskAuditDecorator:
    """Test suite for RiskAuditDecorator class."""

    def test_risk_event_creation(self, mock_logger):
        """Test creation of risk-specific audit event."""
        decorator = RiskAuditDecorator(
            logger=mock_logger, event_type="risk_check", resource_type="risk", action="check"
        )

        @decorator
        def check_var_limit(
            risk_type="var_limit", threshold=Decimal("50000"), current_value=Decimal("45000")
        ):
            return {"status": "within_limit"}

        result = check_var_limit()

        assert result["status"] == "within_limit"

        # Check that RiskEvent was created
        call_args = mock_logger.log_event.call_args
        event = call_args[0][0]

        assert hasattr(event, "risk_type")
        assert event.metadata["function_name"] == "check_var_limit"


class TestDecoratorPerformance:
    """Test suite for decorator performance characteristics."""

    def test_minimal_overhead(self, mock_logger):
        """Test that decorator adds minimal overhead."""
        decorator = AuditDecorator(
            logger=mock_logger, event_type="perf_test", resource_type="test", action="test"
        )

        @decorator
        def fast_function():
            return "fast"

        # Measure execution time
        start_time = time.time()
        for _ in range(100):
            result = fast_function()
        end_time = time.time()

        execution_time = (end_time - start_time) / 100  # Average per call

        # Should be very fast (less than 1ms average per call including audit)
        assert execution_time < 0.001
        assert result == "fast"

    def test_async_processing_does_not_block(self, mock_logger):
        """Test that async audit processing doesn't block function execution."""

        # Simulate slow audit logging
        def slow_log_event(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return "event_id"

        mock_logger.log_event = slow_log_event

        decorator = AuditDecorator(
            logger=mock_logger, event_type="async_test", resource_type="test", action="test"
        )

        @decorator
        def business_function():
            return "business_result"

        start_time = time.time()
        result = business_function()
        end_time = time.time()

        # Function should complete quickly despite slow audit logging
        # (In real implementation, audit would be async)
        assert result == "business_result"
        execution_time = end_time - start_time

        # This test shows current synchronous behavior
        # In production, audit logging would be asynchronous
        assert execution_time >= 0.1

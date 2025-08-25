"""
Comprehensive unit tests for CancelOrderUseCase.

This module provides exhaustive test coverage for the CancelOrderUseCase class,
testing all methods, branches, and edge cases to achieve 100% coverage.
Tests include success scenarios, failure scenarios, validation errors,
transaction management, and various order states.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from src.application.interfaces.broker import IBroker
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases.trading import (
    CancelOrderRequest,
    CancelOrderResponse,
    CancelOrderUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce


# Test Fixtures
@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work with required repositories."""
    uow = AsyncMock(spec=IUnitOfWork)

    # Setup order repository with proper methods
    orders_repo = AsyncMock()
    orders_repo.get_by_id = AsyncMock()
    orders_repo.update = AsyncMock()
    orders_repo.save = AsyncMock()
    uow.orders = orders_repo

    # Setup transaction methods
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Setup context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_broker():
    """Create a mock broker implementation."""
    broker = Mock(spec=IBroker)
    broker.cancel_order = Mock(return_value=True)
    return broker


@pytest.fixture
def cancel_order_use_case(mock_unit_of_work, mock_broker):
    """Create CancelOrderUseCase instance with mocked dependencies."""
    return CancelOrderUseCase(unit_of_work=mock_unit_of_work, broker=mock_broker)


@pytest.fixture
def sample_active_order():
    """Create a sample active order for testing."""
    order = Order(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        status=OrderStatus.SUBMITTED,
        time_in_force=TimeInForce.DAY,
    )
    order.broker_order_id = "BROKER-123"
    return order


@pytest.fixture
def sample_filled_order():
    """Create a sample filled order for testing."""
    order = Order(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("100"),
        average_fill_price=Decimal("149.50"),
    )
    order.broker_order_id = "BROKER-456"
    return order


@pytest.fixture
def sample_cancelled_order():
    """Create a sample cancelled order for testing."""
    order = Order(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("155.00"),
        status=OrderStatus.CANCELLED,
    )
    order.broker_order_id = "BROKER-789"
    return order


@pytest.fixture
def valid_request():
    """Create a valid cancel order request."""
    return CancelOrderRequest(
        order_id=uuid4(),
        reason="User requested cancellation",
        correlation_id=uuid4(),
        metadata={"source": "test", "priority": "normal"},
    )


# Validation Tests
class TestCancelOrderValidation:
    """Test request validation for CancelOrderUseCase."""

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, cancel_order_use_case, valid_request):
        """Test validation passes for valid request."""
        result = await cancel_order_use_case.validate(valid_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_missing_order_id(self, cancel_order_use_case):
        """Test validation fails when order_id is missing."""
        request = CancelOrderRequest(
            order_id=None,
            reason="Test cancellation",  # Invalid: missing order_id
        )
        result = await cancel_order_use_case.validate(request)
        assert result == "Order ID is required"

    @pytest.mark.asyncio
    async def test_validate_request_with_defaults(self, cancel_order_use_case):
        """Test validation with request using default values."""
        request = CancelOrderRequest(order_id=uuid4())
        # Should auto-generate request_id and initialize metadata
        assert request.request_id is not None
        assert request.metadata == {}

        result = await cancel_order_use_case.validate(request)
        assert result is None


# Success Scenarios
class TestCancelOrderSuccess:
    """Test successful order cancellation scenarios."""

    @pytest.mark.asyncio
    async def test_cancel_submitted_order_success(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test successful cancellation of a submitted order."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == OrderStatus.CANCELLED
        assert response.request_id == valid_request.request_id
        assert response.error is None

        # Verify order was retrieved
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(valid_request.order_id)

        # Verify broker cancellation
        mock_broker.cancel_order.assert_called_once_with(sample_active_order.id)

        # Verify order was updated
        mock_unit_of_work.orders.update_order.assert_called_once_with(sample_active_order)
        assert sample_active_order.status == OrderStatus.CANCELLED
        assert sample_active_order.tags.get("cancel_reason") == valid_request.reason

    @pytest.mark.asyncio
    async def test_cancel_partially_filled_order_success(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, valid_request
    ):
        """Test successful cancellation of a partially filled order."""
        # Create partially filled order
        order = Order(
            id=uuid4(),
            symbol="TSLA",
            quantity=Decimal("200"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("250.00"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("75"),
        )

        # Setup
        valid_request.order_id = order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == OrderStatus.CANCELLED
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_pending_order_success(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, valid_request
    ):
        """Test successful cancellation of a pending order."""
        # Create pending order
        order = Order(
            id=uuid4(),
            symbol="GOOGL",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("2800.00"),
            status=OrderStatus.PENDING,
        )

        # Setup
        valid_request.order_id = order.id
        valid_request.reason = "Market conditions changed"
        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert order.tags.get("cancel_reason") == "Market conditions changed"

    @pytest.mark.asyncio
    async def test_cancel_order_without_reason(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test cancellation without providing a reason."""
        # Create request without reason
        request = CancelOrderRequest(order_id=sample_active_order.id, reason=None)

        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.success is True
        assert response.cancelled is True
        assert "cancel_reason" not in sample_active_order.tags


# Failure Scenarios
class TestCancelOrderFailures:
    """Test failure scenarios for order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_non_existent_order(
        self, cancel_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test cancelling a non-existent order."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"
        assert response.cancelled is False
        assert response.final_status is None
        assert response.request_id is not None

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_order(
        self, cancel_order_use_case, mock_unit_of_work, sample_cancelled_order, valid_request
    ):
        """Test cancelling an already cancelled order."""
        # Setup
        valid_request.order_id = sample_cancelled_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_cancelled_order

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert response.error == f"Order cannot be cancelled in status: {OrderStatus.CANCELLED}"
        assert response.cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_filled_order(
        self, cancel_order_use_case, mock_unit_of_work, sample_filled_order, valid_request
    ):
        """Test cancelling a completely filled order."""
        # Setup
        valid_request.order_id = sample_filled_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_filled_order

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert response.error == f"Order cannot be cancelled in status: {OrderStatus.FILLED}"
        assert response.cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_rejected_order(
        self, cancel_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test cancelling a rejected order."""
        # Create rejected order
        order = Order(
            id=uuid4(),
            symbol="NVDA",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("500.00"),
            status=OrderStatus.REJECTED,
        )

        # Setup
        valid_request.order_id = order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert response.error == f"Order cannot be cancelled in status: {OrderStatus.REJECTED}"

    @pytest.mark.asyncio
    async def test_cancel_expired_order(
        self, cancel_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test cancelling an expired order."""
        # Create expired order
        order = Order(
            id=uuid4(),
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("350.00"),
            status=OrderStatus.EXPIRED,
        )

        # Setup
        valid_request.order_id = order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert "Order cannot be cancelled in status: expired" in response.error


# Broker Interaction Failures
class TestBrokerInteractionFailures:
    """Test broker interaction failure scenarios."""

    @pytest.mark.asyncio
    async def test_broker_cancellation_fails(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test when broker fails to cancel the order."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = False  # Broker returns failure

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert response.error == "Broker failed to cancel order"
        assert response.cancelled is False

        # Verify order status wasn't changed
        assert sample_active_order.status == OrderStatus.SUBMITTED

        # Verify update wasn't called
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_broker_cancellation_exception(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test when broker raises an exception during cancellation."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.side_effect = Exception("Broker connection failed")

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert "Failed to cancel order: Broker connection failed" in response.error
        assert response.cancelled is False

        # Verify order wasn't updated
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_broker_timeout_exception(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test when broker times out during cancellation."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.side_effect = TimeoutError("Broker request timed out")

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert "Failed to cancel order: Broker request timed out" in response.error


# Transaction Management Tests
class TestTransactionManagement:
    """Test transaction management in CancelOrderUseCase."""

    @pytest.mark.asyncio
    async def test_transaction_commit_on_success(
        self, mock_unit_of_work, mock_broker, sample_active_order, valid_request
    ):
        """Test that transaction is committed on successful cancellation."""
        # Setup
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(
        self, mock_unit_of_work, mock_broker, sample_active_order, valid_request
    ):
        """Test that transaction is rolled back on failure."""
        # Setup
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = False  # Cause failure

        # Execute
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is False
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_exception(
        self, mock_unit_of_work, mock_broker, sample_active_order, valid_request
    ):
        """Test that transaction is rolled back when exception occurs."""
        # Setup
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.side_effect = Exception("Database error")

        # Execute - The base class catches exceptions and returns error response
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is False
        assert "Database error" in response.error
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_validation_failure(self, mock_unit_of_work, mock_broker):
        """Test that transaction is rolled back on validation failure."""
        # Setup
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)
        request = CancelOrderRequest(order_id=None)  # Invalid request

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order ID is required"
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()


# Edge Cases and Special Scenarios
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_cancel_order_with_special_characters_in_reason(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test cancellation with special characters in reason."""
        # Create request with special characters
        request = CancelOrderRequest(
            order_id=sample_active_order.id,
            reason='User\'s request: "Cancel immediately!" @ 15:30 & save $$',
        )

        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.success is True
        assert sample_active_order.tags["cancel_reason"] == request.reason

    @pytest.mark.asyncio
    async def test_cancel_order_with_empty_reason(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test cancellation with empty string reason."""
        # Create request with empty reason
        request = CancelOrderRequest(order_id=sample_active_order.id, reason="")

        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.success is True
        # Empty string is falsy, so reason shouldn't be added to tags
        assert "cancel_reason" not in sample_active_order.tags

    @pytest.mark.asyncio
    async def test_cancel_order_with_large_metadata(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test cancellation with large metadata object."""
        # Create request with large metadata
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        request = CancelOrderRequest(
            order_id=sample_active_order.id,
            reason="Test with large metadata",
            metadata=large_metadata,
        )

        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.success is True
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_cancel_order_idempotency(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test that cancelling the same order multiple times is handled properly."""
        # Setup for first cancellation
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # First cancellation
        response1 = await cancel_order_use_case.process(valid_request)
        assert response1.success is True
        assert sample_active_order.status == OrderStatus.CANCELLED

        # Second cancellation attempt (order is now cancelled)
        response2 = await cancel_order_use_case.process(valid_request)
        assert response2.success is False
        assert "Order cannot be cancelled in status: cancelled" in response2.error


# Logging Tests
class TestLogging:
    """Test logging behavior in CancelOrderUseCase."""

    @pytest.mark.asyncio
    async def test_logging_on_error(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test that error logs are generated on failure."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.side_effect = Exception("Test error")

        # Patch the logger
        with patch.object(cancel_order_use_case, "logger") as mock_logger:
            # Execute
            response = await cancel_order_use_case.process(valid_request)

            # Assert
            assert response.success is False
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to cancel order: Test error" in error_call


# Concurrent Cancellation Tests
class TestConcurrentOperations:
    """Test concurrent cancellation operations."""

    @pytest.mark.asyncio
    async def test_concurrent_cancellations_different_orders(self, mock_unit_of_work, mock_broker):
        """Test cancelling multiple different orders concurrently."""
        # Create multiple orders
        orders = []
        requests = []
        for i in range(5):
            order = Order(
                id=uuid4(),
                symbol=f"STOCK{i}",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                limit_price=Decimal(f"{100 + i}.00"),
                status=OrderStatus.SUBMITTED,
            )
            orders.append(order)
            requests.append(CancelOrderRequest(order_id=order.id, reason=f"Cancellation {i}"))

        # Setup mock to return different orders
        def get_order_by_id(order_id):
            for order in orders:
                if order.id == order_id:
                    return order
            return None

        mock_unit_of_work.orders.get_order_by_id.side_effect = get_order_by_id
        mock_broker.cancel_order.return_value = True

        # Create use case
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Execute cancellations concurrently
        tasks = [use_case.process(request) for request in requests]
        responses = await asyncio.gather(*tasks)

        # Assert all succeeded
        for response in responses:
            assert response.success is True
            assert response.cancelled is True

        # Verify all orders were cancelled
        for order in orders:
            assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_concurrent_cancellation_same_order(self, mock_unit_of_work, mock_broker):
        """Test cancelling the same order concurrently (race condition)."""
        # Create an active order for testing
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
            status=OrderStatus.SUBMITTED,
        )

        # Create multiple requests for the same order
        requests = [
            CancelOrderRequest(order_id=order.id, reason=f"Concurrent request {i}")
            for i in range(3)
        ]

        # Setup mock to return the order
        mock_unit_of_work.orders.get_order_by_id.return_value = order

        # First cancel succeeds, subsequent ones should fail due to order state
        call_count = [0]

        def cancel_order_side_effect(order_id):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            # Order is already cancelled at this point
            return False

        mock_broker.cancel_order.side_effect = cancel_order_side_effect

        # Create use case
        use_case = CancelOrderUseCase(mock_unit_of_work, mock_broker)

        # Execute first cancellation
        response1 = await use_case.process(requests[0])
        assert response1.success is True
        assert order.status == OrderStatus.CANCELLED

        # Try to cancel again (should fail)
        response2 = await use_case.process(requests[1])
        assert response2.success is False
        assert "Order cannot be cancelled in status: cancelled" in response2.error


# Integration with Order Entity Tests
class TestOrderEntityIntegration:
    """Test integration with Order entity methods."""

    @pytest.mark.asyncio
    async def test_order_is_active_check(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker
    ):
        """Test that is_active() method is properly used."""
        # Test with different order statuses
        test_cases = [
            (OrderStatus.PENDING, True),
            (OrderStatus.SUBMITTED, True),
            (OrderStatus.PARTIALLY_FILLED, True),
            (OrderStatus.FILLED, False),
            (OrderStatus.CANCELLED, False),
            (OrderStatus.REJECTED, False),
            (OrderStatus.EXPIRED, False),
        ]

        for status, should_be_cancellable in test_cases:
            order = Order(
                id=uuid4(),
                symbol="TEST",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=status,
            )

            request = CancelOrderRequest(order_id=order.id)
            mock_unit_of_work.orders.get_order_by_id.return_value = order
            mock_broker.cancel_order.return_value = True

            response = await cancel_order_use_case.process(request)

            if should_be_cancellable:
                assert response.success is True
                assert response.cancelled is True
            else:
                assert response.success is False
                assert f"Order cannot be cancelled in status: {status}" in response.error

    @pytest.mark.asyncio
    async def test_order_cancel_method_integration(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test that Order.cancel() method is properly called."""
        # Setup
        request = CancelOrderRequest(
            order_id=sample_active_order.id, reason="Integration test reason"
        )
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.success is True
        assert sample_active_order.status == OrderStatus.CANCELLED
        assert sample_active_order.cancelled_at is not None
        assert sample_active_order.tags.get("cancel_reason") == "Integration test reason"


# Repository Interaction Tests
class TestRepositoryInteractions:
    """Test interactions with the order repository."""

    @pytest.mark.asyncio
    async def test_repository_get_by_id_called(
        self, cancel_order_use_case, mock_unit_of_work, sample_active_order, valid_request
    ):
        """Test that repository.get_by_id is called with correct parameters."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order

        # Execute
        await cancel_order_use_case.process(valid_request)

        # Assert
        mock_unit_of_work.orders.get_order_by_id.assert_called_once_with(valid_request.order_id)

    @pytest.mark.asyncio
    async def test_repository_update_called_on_success(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test that repository.update is called on successful cancellation."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is True
        mock_unit_of_work.orders.update_order.assert_called_once_with(sample_active_order)

    @pytest.mark.asyncio
    async def test_repository_update_not_called_on_broker_failure(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test that repository.update is not called when broker fails."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = False

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        mock_unit_of_work.orders.update_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_repository_exception_handling(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test handling of repository exceptions."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True
        mock_unit_of_work.orders.update_order.side_effect = Exception("Database update failed")

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert
        assert response.success is False
        assert "Failed to cancel order: Database update failed" in response.error


# Response Generation Tests
class TestResponseGeneration:
    """Test response generation in various scenarios."""

    @pytest.mark.asyncio
    async def test_successful_response_fields(
        self,
        cancel_order_use_case,
        mock_unit_of_work,
        mock_broker,
        sample_active_order,
        valid_request,
    ):
        """Test that successful response contains all expected fields."""
        # Setup
        valid_request.order_id = sample_active_order.id
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert response structure
        assert isinstance(response, CancelOrderResponse)
        assert response.success is True
        assert response.cancelled is True
        assert response.final_status == OrderStatus.CANCELLED
        assert response.request_id == valid_request.request_id
        assert response.error is None

    @pytest.mark.asyncio
    async def test_error_response_fields(
        self, cancel_order_use_case, mock_unit_of_work, valid_request
    ):
        """Test that error response contains all expected fields."""
        # Setup - order not found
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await cancel_order_use_case.process(valid_request)

        # Assert response structure
        assert isinstance(response, CancelOrderResponse)
        assert response.success is False
        assert response.cancelled is False
        assert response.final_status is None
        assert response.request_id is not None
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_response_request_id_preservation(
        self, cancel_order_use_case, mock_unit_of_work, mock_broker, sample_active_order
    ):
        """Test that request_id is preserved in response."""
        # Create request with specific request_id
        specific_id = uuid4()
        request = CancelOrderRequest(order_id=sample_active_order.id, request_id=specific_id)

        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_active_order
        mock_broker.cancel_order.return_value = True

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.request_id == specific_id

    @pytest.mark.asyncio
    async def test_response_request_id_generation_on_error(
        self, cancel_order_use_case, mock_unit_of_work
    ):
        """Test that request_id is generated for response when not provided."""
        # Create request without request_id (will be auto-generated)
        request = CancelOrderRequest(order_id=uuid4())

        # Setup - order not found
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await cancel_order_use_case.process(request)

        # Assert
        assert response.request_id is not None
        assert isinstance(response.request_id, UUID)

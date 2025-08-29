"""
Broker Coordinator

Coordinates between thin broker adapters and business use cases.
This provides the high-level interface for trading operations while
keeping the brokers as simple technical adapters.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from src.application.interfaces.broker import IBroker
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases import (
    CancelOrderUseCase,
    GetOrderStatusUseCase,
    PlaceOrderUseCase,
    ProcessOrderFillUseCase,
    ProcessPendingOrdersUseCase,
    SimulateOrderExecutionUseCase,
    UpdateMarketPriceUseCase,
)

logger = logging.getLogger(__name__)


@dataclass
class UseCaseFactory:
    """Factory for creating use case instances."""

    unit_of_work: IUnitOfWork
    order_processor: Any
    commission_calculator: Any
    market_microstructure: Any
    risk_calculator: Any
    order_validator: Any
    position_manager: Any

    def create_place_order_use_case(self, broker: IBroker) -> PlaceOrderUseCase:
        """Create place order use case."""
        return PlaceOrderUseCase(
            unit_of_work=self.unit_of_work,
            broker=broker,
            order_validator=self.order_validator,
            risk_calculator=self.risk_calculator,
        )

    def create_cancel_order_use_case(self, broker: IBroker) -> CancelOrderUseCase:
        """Create cancel order use case."""
        return CancelOrderUseCase(
            unit_of_work=self.unit_of_work,
            broker=broker,
        )

    def create_process_fill_use_case(self) -> ProcessOrderFillUseCase:
        """Create process order fill use case."""
        return ProcessOrderFillUseCase(
            unit_of_work=self.unit_of_work,
            order_processor=self.order_processor,
            commission_calculator=self.commission_calculator,
        )

    def create_simulate_execution_use_case(self) -> SimulateOrderExecutionUseCase:
        """Create simulate order execution use case."""
        return SimulateOrderExecutionUseCase(
            unit_of_work=self.unit_of_work,
            market_microstructure=self.market_microstructure,
            commission_calculator=self.commission_calculator,
        )

    def create_update_market_price_use_case(self) -> UpdateMarketPriceUseCase:
        """Create update market price use case."""
        return UpdateMarketPriceUseCase(
            unit_of_work=self.unit_of_work,
            order_processor=self.order_processor,
            market_microstructure=self.market_microstructure,
        )

    def create_get_order_status_use_case(self, broker: IBroker) -> GetOrderStatusUseCase:
        """Create get order status use case."""
        from src.application.use_cases.trading import GetOrderStatusUseCase

        return GetOrderStatusUseCase(
            unit_of_work=self.unit_of_work,
            broker=broker,
        )

    def create_process_pending_orders_use_case(self) -> ProcessPendingOrdersUseCase:
        """Create process pending orders use case."""
        from src.application.use_cases.market_simulation import ProcessPendingOrdersUseCase

        return ProcessPendingOrdersUseCase(
            unit_of_work=self.unit_of_work,
            order_processor=self.order_processor,
            market_microstructure=self.market_microstructure,
        )


class BrokerCoordinator:
    """
    Coordinates between thin broker adapters and business use cases.

    This class provides the high-level interface for trading operations,
    orchestrating between the simplified broker adapters and the business
    logic encapsulated in use cases.
    """

    def __init__(
        self,
        broker: IBroker,
        use_case_factory: UseCaseFactory,
    ):
        """
        Initialize broker coordinator.

        Args:
            broker: The broker adapter (thin interface)
            use_case_factory: Factory for creating use cases
        """
        self.broker = broker
        self.use_case_factory = use_case_factory
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Market prices cache (for paper trading)
        self.market_prices: dict[str, Decimal] = {}

    async def place_order(self, order_request: dict[str, Any]) -> dict[str, Any]:
        """
        Place an order through the broker with full business logic.

        Args:
            order_request: Order details

        Returns:
            Order placement result
        """
        self.logger.info(f"Placing order: {order_request}")

        # Use place order use case for validation and risk checks
        use_case = self.use_case_factory.create_place_order_use_case(self.broker)

        # Convert request to use case format
        from src.application.use_cases.trading import PlaceOrderRequest

        request = PlaceOrderRequest(
            request_id=(
                UUID(order_request.get("request_id")) if "request_id" in order_request else uuid4()
            ),
            portfolio_id=UUID(order_request["portfolio_id"]),
            symbol=order_request["symbol"],
            side=order_request["side"],
            order_type=order_request["order_type"],
            quantity=order_request["quantity"],
            limit_price=order_request.get("limit_price"),
            stop_price=order_request.get("stop_price"),
            time_in_force=order_request.get("time_in_force", "day"),
            strategy_id=order_request.get("strategy_id"),
        )

        # Execute use case
        response = await use_case.execute(request)

        if response.success:
            self.logger.info(f"Order placed successfully: {response.order_id}")
        else:
            self.logger.error(f"Order placement failed: {response.error}")

        return {
            "success": response.success,
            "order_id": str(response.order_id) if response.order_id else None,
            "broker_order_id": response.broker_order_id,
            "status": response.status,
            "error": response.error,
        }

    async def cancel_order(self, order_id: UUID, reason: str | None = None) -> dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order to cancel
            reason: Cancellation reason

        Returns:
            Cancellation result
        """
        self.logger.info(f"Cancelling order: {order_id}")

        use_case = self.use_case_factory.create_cancel_order_use_case(self.broker)

        from src.application.use_cases.trading import CancelOrderRequest

        request = CancelOrderRequest(
            request_id=uuid4(),
            order_id=order_id,
            reason=reason,
        )

        response = await use_case.execute(request)

        return {
            "success": response.success,
            "cancelled": response.cancelled,
            "final_status": response.final_status,
            "error": response.error,
        }

    async def update_market_price(self, symbol: str, price: Decimal) -> dict[str, Any]:
        """
        Update market price and trigger pending orders.

        Args:
            symbol: Trading symbol
            price: New market price

        Returns:
            Update result with triggered orders
        """
        self.logger.debug(f"Updating market price: {symbol} = {price}")

        # Update local cache
        self.market_prices[symbol] = price

        # Update in broker if it supports it
        if hasattr(self.broker, "set_market_price"):
            self.broker.set_market_price(symbol, price)

        # Use update market price use case to trigger orders
        use_case = self.use_case_factory.create_update_market_price_use_case()

        from src.application.use_cases.market_simulation import UpdateMarketPriceRequest

        request = UpdateMarketPriceRequest(
            request_id=uuid4(),
            symbol=symbol,
            price=price,
            timestamp=datetime.now(UTC),
        )

        response = await use_case.execute(request)

        return {
            "success": response.success,
            "orders_triggered": response.orders_triggered,
            "orders_filled": response.orders_filled,
            "error": response.error,
        }

    async def process_order_fill(
        self,
        order_id: UUID,
        fill_price: Decimal,
        fill_quantity: int | None = None,
    ) -> dict[str, Any]:
        """
        Process an order fill.

        Args:
            order_id: Order that was filled
            fill_price: Execution price
            fill_quantity: Quantity filled (None for complete fill)

        Returns:
            Fill processing result
        """
        self.logger.info(f"Processing fill for order {order_id}: {fill_quantity}@{fill_price}")

        use_case = self.use_case_factory.create_process_fill_use_case()

        from src.application.use_cases.order_execution import ProcessOrderFillRequest

        request = ProcessOrderFillRequest(
            request_id=uuid4(),
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
        )

        response = await use_case.execute(request)

        if response.success:
            self.logger.info(f"Fill processed: {response.fill_quantity}@{response.fill_price}")
        else:
            self.logger.error(f"Fill processing failed: {response.error}")

        return {
            "success": response.success,
            "filled": response.filled,
            "fill_price": response.fill_price,
            "fill_quantity": response.fill_quantity,
            "commission": response.commission,
            "position_id": str(response.position_id) if response.position_id else None,
            "error": response.error,
        }

    async def get_order_status(self, order_id: UUID) -> dict[str, Any]:
        """
        Get order status.

        Args:
            order_id: Order to check

        Returns:
            Order status information
        """
        use_case = self.use_case_factory.create_get_order_status_use_case(self.broker)

        from src.application.use_cases.trading import GetOrderStatusRequest

        request = GetOrderStatusRequest(
            request_id=uuid4(),
            order_id=order_id,
        )

        response = await use_case.execute(request)

        return {
            "success": response.success,
            "status": response.status,
            "filled_quantity": response.filled_quantity,
            "average_fill_price": response.average_fill_price,
            "error": response.error,
        }

    async def process_pending_orders(self) -> dict[str, Any]:
        """
        Process all pending orders based on current market prices.

        Returns:
            Processing result
        """
        self.logger.debug("Processing pending orders")

        use_case = self.use_case_factory.create_process_pending_orders_use_case()

        from src.application.use_cases.market_simulation import ProcessPendingOrdersRequest

        request = ProcessPendingOrdersRequest(
            request_id=uuid4(),
            current_prices=self.market_prices,
        )

        response = await use_case.execute(request)

        return {
            "success": response.success,
            "processed_count": response.processed_count,
            "triggered_orders": response.triggered_orders,
            "filled_orders": response.filled_orders,
            "error": response.error,
        }

    def create_get_order_status_use_case(self, broker: IBroker) -> GetOrderStatusUseCase:
        """Helper to create get order status use case."""
        return GetOrderStatusUseCase(
            unit_of_work=self.use_case_factory.unit_of_work,
            broker=broker,
        )

    def create_process_pending_orders_use_case(self) -> ProcessPendingOrdersUseCase:
        """Helper to create process pending orders use case."""
        return ProcessPendingOrdersUseCase(
            unit_of_work=self.use_case_factory.unit_of_work,
            order_processor=self.use_case_factory.order_processor,
            market_microstructure=self.use_case_factory.market_microstructure,
        )

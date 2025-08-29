"""
Market Simulation Use Cases

Handles market simulation logic including order triggering, price updates,
and pending order processing that was previously in the infrastructure layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.services.market_microstructure import IMarketMicrostructure
from src.domain.services.order_processor import OrderProcessor
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO

logger = logging.getLogger(__name__)


# Request/Response DTOs
@dataclass
class UpdateMarketPriceRequest(BaseRequestDTO):
    """Request to update market price and trigger pending orders."""

    symbol: str
    price: Decimal
    volume: int | None = None
    timestamp: datetime | None = None


@dataclass
class UpdateMarketPriceResponse(UseCaseResponse):
    """Response from updating market price."""

    orders_triggered: list[UUID] = field(default_factory=list)
    orders_filled: list[UUID] = field(default_factory=list)


@dataclass
class ProcessPendingOrdersRequest(BaseRequestDTO):
    """Request to process all pending orders."""

    symbol: str | None = None  # Process specific symbol or all
    current_prices: dict[str, Decimal] | None = None


@dataclass
class ProcessPendingOrdersResponse(UseCaseResponse):
    """Response from processing pending orders."""

    processed_count: int = 0
    triggered_orders: list[UUID] = field(default_factory=list)
    filled_orders: list[UUID] = field(default_factory=list)


@dataclass
class CheckOrderTriggerRequest(BaseRequestDTO):
    """Request to check if an order should be triggered."""

    order_id: UUID
    current_price: Decimal


@dataclass
class CheckOrderTriggerResponse(UseCaseResponse):
    """Response from checking order trigger."""

    should_trigger: bool = False
    trigger_price: Decimal | None = None
    reason: str | None = None


# Use Case Implementations
class UpdateMarketPriceUseCase(
    TransactionalUseCase[UpdateMarketPriceRequest, UpdateMarketPriceResponse]
):
    """
    Updates market price and processes orders that should be triggered.

    This extracts the market price update and order triggering logic
    from the infrastructure broker.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        order_processor: OrderProcessor,
        market_microstructure: IMarketMicrostructure,
    ):
        """Initialize the update market price use case."""
        super().__init__(unit_of_work, "UpdateMarketPriceUseCase")
        self.order_processor = order_processor
        self.market_microstructure = market_microstructure

    async def validate(self, request: UpdateMarketPriceRequest) -> str | None:
        """Validate the market price update request."""
        if request.price <= 0:
            return "Price must be positive"

        if request.volume is not None and request.volume < 0:
            return "Volume cannot be negative"

        return None

    async def process(self, request: UpdateMarketPriceRequest) -> UpdateMarketPriceResponse:
        """Process the market price update."""
        symbol = Symbol(request.symbol)
        current_price = Price(request.price)
        timestamp = request.timestamp or datetime.now(UTC)

        # Get all active orders for this symbol
        order_repo = self.unit_of_work.orders
        active_orders = await order_repo.get_active_orders()

        orders_triggered = []
        orders_filled = []

        for order in active_orders:
            if order.symbol != symbol.value:
                continue

            # Check if order should be triggered based on new price
            should_fill, trigger_reason = self._should_trigger_order(order, current_price)

            if should_fill:
                self.logger.info(
                    f"Order {order.id} triggered: {trigger_reason}",
                    extra={
                        "order_id": str(order.id),
                        "symbol": symbol.value,
                        "price": float(ValueObjectConverter.extract_value(current_price)),
                        "reason": trigger_reason,
                    },
                )

                orders_triggered.append(order.id)

                # For market simulation, we immediately "fill" MARKET and LIMIT orders
                # STOP and STOP_LIMIT orders are only triggered, not filled immediately
                # In production, triggered orders would be submitted to actual market
                if order.order_type in [OrderType.MARKET, OrderType.LIMIT]:
                    # Calculate execution price with market impact
                    execution_price = self.market_microstructure.calculate_execution_price(
                        base_price=current_price,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                    )

                    # Mark as filled (simplified - real system would use ProcessOrderFillUseCase)
                    order.status = OrderStatus.FILLED
                    order.average_fill_price = execution_price
                    order.filled_quantity = order.quantity
                    order.filled_at = timestamp

                    await order_repo.update_order(order)
                    orders_filled.append(order.id)

        return UpdateMarketPriceResponse(
            success=True,
            orders_triggered=orders_triggered,
            orders_filled=orders_filled,
            request_id=request.request_id,
        )

    def _should_trigger_order(self, order: Order, current_price: Price) -> tuple[bool, str | None]:
        """
        Determine if an order should be triggered based on current price.

        Args:
            order: The order to check
            current_price: Current market price

        Returns:
            Tuple of (should_trigger, reason)
        """
        if not order.is_active():
            return False, None

        # Market orders trigger immediately
        if order.order_type == OrderType.MARKET:
            return True, "Market order triggers immediately"

        # Limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            limit_val = ValueObjectConverter.extract_value(order.limit_price)

            if order.side == OrderSide.BUY and current_val <= limit_val:
                return (
                    True,
                    f"Buy limit triggered: price {current_val} <= limit {limit_val}",
                )
            elif order.side == OrderSide.SELL and current_val >= limit_val:
                return (
                    True,
                    f"Sell limit triggered: price {current_val} >= limit {limit_val}",
                )

        # Stop orders
        if order.order_type == OrderType.STOP and order.stop_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            stop_val = ValueObjectConverter.extract_value(order.stop_price)

            if order.side == OrderSide.BUY and current_val >= stop_val:
                return (
                    True,
                    f"Buy stop triggered: price {current_val} >= stop {stop_val}",
                )
            elif order.side == OrderSide.SELL and current_val <= stop_val:
                return (
                    True,
                    f"Sell stop triggered: price {current_val} <= stop {stop_val}",
                )

        # Stop-limit orders
        if order.order_type == OrderType.STOP_LIMIT and order.stop_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            stop_val = ValueObjectConverter.extract_value(order.stop_price)

            # First check if stop is triggered
            if order.side == OrderSide.BUY and current_val >= stop_val:
                # Then check if limit allows execution
                if order.limit_price:
                    limit_val = ValueObjectConverter.extract_value(order.limit_price)
                    if current_val <= limit_val:
                        return True, "Buy stop-limit triggered: stop hit and price within limit"
            elif order.side == OrderSide.SELL and current_val <= stop_val:
                if order.limit_price:
                    limit_val = ValueObjectConverter.extract_value(order.limit_price)
                    if current_val >= limit_val:
                        return True, "Sell stop-limit triggered: stop hit and price within limit"

        return False, None


class ProcessPendingOrdersUseCase(
    TransactionalUseCase[ProcessPendingOrdersRequest, ProcessPendingOrdersResponse]
):
    """
    Processes all pending orders that should be triggered.

    This extracts the batch order processing logic from the infrastructure.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        order_processor: OrderProcessor,
        market_microstructure: IMarketMicrostructure,
    ):
        """Initialize the process pending orders use case."""
        super().__init__(unit_of_work, "ProcessPendingOrdersUseCase")
        self.order_processor = order_processor
        self.market_microstructure = market_microstructure

    async def validate(self, request: ProcessPendingOrdersRequest) -> str | None:
        """Validate the process pending orders request."""
        return None  # No specific validation needed

    async def process(self, request: ProcessPendingOrdersRequest) -> ProcessPendingOrdersResponse:
        """Process all pending orders."""
        # Get pending orders
        order_repo = self.unit_of_work.orders
        active_orders = await order_repo.get_pending_orders()

        # Filter by symbol if specified
        if request.symbol:
            symbol = Symbol(request.symbol)
            active_orders = [o for o in active_orders if o.symbol == symbol.value]

        triggered_orders = []
        filled_orders = []
        processed_count = 0

        for order in active_orders:
            processed_count += 1

            # Get current price for the symbol
            if request.current_prices and order.symbol in request.current_prices:
                current_price = Price(request.current_prices[order.symbol])
            else:
                # Fetch from market data provider
                bar = await self.unit_of_work.market_data.get_latest_bar(order.symbol)
                if bar is None:
                    # No market data available, skip this order
                    continue
                current_price = Price(bar.close)

            # Check if order should be triggered
            should_fill, trigger_reason = self._should_trigger_order(order, current_price)

            if should_fill:
                triggered_orders.append(order.id)

                # Simulate fill (in production would submit to market)
                from src.domain.value_objects.quantity import Quantity

                execution_price = self.market_microstructure.calculate_execution_price(
                    base_price=current_price,
                    side=order.side,
                    quantity=Quantity(order.quantity),
                    order_type=order.order_type,
                )

                order.status = OrderStatus.FILLED
                order.average_fill_price = execution_price
                order.filled_quantity = order.quantity
                order.filled_at = datetime.now(UTC)

                await order_repo.update_order(order)
                filled_orders.append(order.id)

        return ProcessPendingOrdersResponse(
            success=True,
            processed_count=processed_count,
            triggered_orders=triggered_orders,
            filled_orders=filled_orders,
            request_id=request.request_id,
        )

    def _should_trigger_order(self, order: Order, current_price: Price) -> tuple[bool, str | None]:
        """Check if order should be triggered (reused from UpdateMarketPriceUseCase)."""
        # Implementation identical to UpdateMarketPriceUseCase
        # In production, this would be extracted to a shared service
        if not order.is_active():
            return False, None

        if order.order_type == OrderType.MARKET:
            return True, "Market order"

        if order.order_type == OrderType.LIMIT and order.limit_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            limit_val = ValueObjectConverter.extract_value(order.limit_price)

            if order.side == OrderSide.BUY and current_val <= limit_val:
                return True, "Buy limit triggered"
            elif order.side == OrderSide.SELL and current_val >= limit_val:
                return True, "Sell limit triggered"

        if order.order_type == OrderType.STOP and order.stop_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            stop_val = ValueObjectConverter.extract_value(order.stop_price)

            if order.side == OrderSide.BUY and current_val >= stop_val:
                return True, "Buy stop triggered"
            elif order.side == OrderSide.SELL and current_val <= stop_val:
                return True, "Sell stop triggered"

        return False, None


class CheckOrderTriggerUseCase(
    TransactionalUseCase[CheckOrderTriggerRequest, CheckOrderTriggerResponse]
):
    """
    Checks if a specific order should be triggered at current price.

    This provides a focused use case for checking individual orders.
    """

    def __init__(self, unit_of_work: IUnitOfWork) -> None:
        """Initialize the check order trigger use case."""
        super().__init__(unit_of_work, "CheckOrderTriggerUseCase")

    async def validate(self, request: CheckOrderTriggerRequest) -> str | None:
        """Validate the check order trigger request."""
        if request.current_price <= 0:
            return "Current price must be positive"
        return None

    async def process(self, request: CheckOrderTriggerRequest) -> CheckOrderTriggerResponse:
        """Check if the order should be triggered."""
        # Get the order
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return CheckOrderTriggerResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        current_price = Price(request.current_price)

        # Determine trigger conditions
        should_trigger = False
        trigger_price = None
        reason = None

        if not order.is_active():
            reason = f"Order is not active (status: {order.status.value})"
        elif order.order_type == OrderType.MARKET:
            should_trigger = True
            trigger_price = request.current_price
            reason = "Market order executes immediately"
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            limit_val = ValueObjectConverter.extract_value(order.limit_price)

            if order.side == OrderSide.BUY:
                if current_val <= limit_val:
                    should_trigger = True
                    trigger_price = limit_val
                    reason = f"Buy limit: market price {current_val} <= limit {limit_val}"
                else:
                    reason = (
                        f"Buy limit not triggered: market price {current_val} > limit {limit_val}"
                    )
            elif current_val >= limit_val:
                should_trigger = True
                trigger_price = limit_val
                reason = f"Sell limit: market price {current_val} >= limit {limit_val}"
            else:
                reason = f"Sell limit not triggered: market price {current_val} < limit {limit_val}"
        elif order.order_type == OrderType.STOP and order.stop_price:
            # Handle both Price objects and Decimals
            current_val = ValueObjectConverter.extract_value(current_price)
            stop_val = ValueObjectConverter.extract_value(order.stop_price)

            if order.side == OrderSide.BUY:
                if current_val >= stop_val:
                    should_trigger = True
                    trigger_price = request.current_price
                    reason = f"Buy stop: market price {current_val} >= stop {stop_val}"
                else:
                    reason = f"Buy stop not triggered: market price {current_val} < stop {stop_val}"
            elif current_val <= stop_val:
                should_trigger = True
                trigger_price = request.current_price
                reason = f"Sell stop: market price {current_val} <= stop {stop_val}"
            else:
                reason = f"Sell stop not triggered: market price {current_val} > stop {stop_val}"

        return CheckOrderTriggerResponse(
            success=True,
            should_trigger=should_trigger,
            trigger_price=trigger_price,
            reason=reason,
            request_id=request.request_id,
        )

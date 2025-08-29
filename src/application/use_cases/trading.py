"""
Trading Use Cases

Implements business logic for order management and trade execution.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from src.application.interfaces.broker import IBroker
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.entities.order import Order, OrderSide, OrderType, TimeInForce
from src.domain.services.order_validator import OrderValidator
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO


# Request/Response DTOs
@dataclass
class PlaceOrderRequest(BaseRequestDTO):
    """Request to place a new order."""

    portfolio_id: UUID
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop", "stop_limit"
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"
    strategy_id: str | None = None


@dataclass
class PlaceOrderResponse(UseCaseResponse):
    """Response from placing an order."""

    order_id: UUID | None = None
    broker_order_id: str | None = None
    status: str | None = None


@dataclass
class CancelOrderRequest(BaseRequestDTO):
    """Request to cancel an order."""

    order_id: UUID
    reason: str | None = None


@dataclass
class CancelOrderResponse(UseCaseResponse):
    """Response from cancelling an order."""

    cancelled: bool = False
    final_status: str | None = None


@dataclass
class ModifyOrderRequest(BaseRequestDTO):
    """Request to modify an existing order."""

    order_id: UUID
    new_quantity: int | None = None
    new_limit_price: float | None = None
    new_stop_price: float | None = None


@dataclass
class ModifyOrderResponse(UseCaseResponse):
    """Response from modifying an order."""

    modified: bool = False
    new_values: dict[str, Any] | None = None


@dataclass
class GetOrderStatusRequest(BaseRequestDTO):
    """Request to get order status."""

    order_id: UUID


@dataclass
class GetOrderStatusResponse(UseCaseResponse):
    """Response with order status."""

    status: str | None = None
    filled_quantity: int | None = None
    average_fill_price: float | None = None


# Use Case Implementations
class PlaceOrderUseCase(TransactionalUseCase[PlaceOrderRequest, PlaceOrderResponse]):
    """
    Use case for placing trading orders.

    Coordinates order validation, risk checks, and broker submission.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        broker: IBroker,
        order_validator: OrderValidator,
        risk_calculator: RiskCalculator,
    ):
        """Initialize place order use case."""
        super().__init__(unit_of_work, "PlaceOrderUseCase")
        self.broker = broker
        self.order_validator = order_validator
        self.risk_calculator = risk_calculator

    async def validate(self, request: PlaceOrderRequest) -> str | None:
        """Validate the place order request."""
        # Validate order type
        valid_order_types = ["market", "limit", "stop", "stop_limit"]
        if request.order_type not in valid_order_types:
            return f"Invalid order type: {request.order_type}"

        # Validate side
        valid_sides = ["buy", "sell"]
        if request.side not in valid_sides:
            return f"Invalid order side: {request.side}"

        # Validate quantity
        if request.quantity <= 0:
            return "Quantity must be positive"

        # Validate limit orders have limit price
        if request.order_type in ["limit", "stop_limit"] and request.limit_price is None:
            return f"{request.order_type} order requires limit price"

        # Validate stop orders have stop price
        if request.order_type in ["stop", "stop_limit"] and request.stop_price is None:
            return f"{request.order_type} order requires stop price"

        return None

    async def process(self, request: PlaceOrderRequest) -> PlaceOrderResponse:
        """Process the place order request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return PlaceOrderResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Create order entity
        order = Order(
            symbol=request.symbol,
            side=OrderSide[request.side.upper()],
            order_type=OrderType[request.order_type.upper()],
            quantity=Quantity(Decimal(str(request.quantity))),
            limit_price=Price(Decimal(str(request.limit_price))) if request.limit_price else None,
            stop_price=Price(Decimal(str(request.stop_price))) if request.stop_price else None,
            time_in_force=TimeInForce[request.time_in_force.upper()],
        )

        # Get current price for validation
        market_data_repo = self.unit_of_work.market_data
        latest_bar = await market_data_repo.get_latest_bar(request.symbol)
        if not latest_bar:
            return PlaceOrderResponse(
                success=False,
                error="Cannot get current market price",
                request_id=request.request_id or uuid4(),
            )
        current_price = Price(latest_bar.close)

        # Validate order with business rules
        validation_result = await self.order_validator.validate_order(
            order, portfolio, current_price
        )
        if not validation_result.is_valid:
            return PlaceOrderResponse(
                success=False,
                error=validation_result.error_message or "Order validation failed",
                request_id=request.request_id or uuid4(),
            )

        # Check risk limits
        risk_violations = self.risk_calculator.check_risk_limits(portfolio, order)
        if risk_violations[0] is False:
            return PlaceOrderResponse(
                success=False,
                error=f"Risk limit violated: {risk_violations[1]}",
                request_id=request.request_id or uuid4(),
            )

        # Submit to broker
        try:
            broker_order_id = self.broker.submit_order(order)
            order.submit(str(broker_order_id))
            order.broker_order_id = str(broker_order_id)

            # Save order to repository
            order_repo = self.unit_of_work.orders
            await order_repo.save_order(order)

            return PlaceOrderResponse(
                success=True,
                order_id=order.id,
                broker_order_id=str(broker_order_id),
                status=order.status.value,
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            return PlaceOrderResponse(
                success=False,
                error=f"Failed to submit order: {e}",
                request_id=request.request_id or uuid4(),
            )


class CancelOrderUseCase(TransactionalUseCase[CancelOrderRequest, CancelOrderResponse]):
    """
    Use case for cancelling orders.

    Handles order cancellation with broker and updates order status.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        broker: IBroker,
    ):
        """Initialize cancel order use case."""
        super().__init__(unit_of_work, "CancelOrderUseCase")
        self.broker = broker

    async def validate(self, request: CancelOrderRequest) -> str | None:
        """Validate the cancel order request."""
        if not request.order_id:
            return "Order ID is required"
        return None

    async def process(self, request: CancelOrderRequest) -> CancelOrderResponse:
        """Process the cancel order request."""
        # Get order from repository
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return CancelOrderResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        # Check if order can be cancelled
        if not order.is_active():
            return CancelOrderResponse(
                success=False,
                error=f"Order cannot be cancelled in status: {order.status.value}",
                request_id=request.request_id or uuid4(),
            )

        # Cancel with broker
        try:
            success = self.broker.cancel_order(order.id)

            if success:
                order.cancel(request.reason)
                await order_repo.update_order(order)

                return CancelOrderResponse(
                    success=True,
                    cancelled=True,
                    final_status=order.status,
                    request_id=request.request_id,
                )
            else:
                return CancelOrderResponse(
                    success=False,
                    error="Broker failed to cancel order",
                    request_id=request.request_id or uuid4(),
                )

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return CancelOrderResponse(
                success=False,
                error=f"Failed to cancel order: {e}",
                request_id=request.request_id or uuid4(),
            )


class ModifyOrderUseCase(TransactionalUseCase[ModifyOrderRequest, ModifyOrderResponse]):
    """
    Use case for modifying existing orders.

    Handles order modification validation and broker updates.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        broker: IBroker,
        order_validator: OrderValidator,
    ):
        """Initialize modify order use case."""
        super().__init__(unit_of_work, "ModifyOrderUseCase")
        self.broker = broker
        self.order_validator = order_validator

    async def validate(self, request: ModifyOrderRequest) -> str | None:
        """Validate the modify order request."""
        if not request.order_id:
            return "Order ID is required"

        if not any([request.new_quantity, request.new_limit_price, request.new_stop_price]):
            return "At least one modification is required"

        if request.new_quantity is not None and request.new_quantity <= 0:
            return "New quantity must be positive"

        return None

    async def process(self, request: ModifyOrderRequest) -> ModifyOrderResponse:
        """Process the modify order request."""
        # Get order from repository
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return ModifyOrderResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        # Validate modification
        validation_result = self.order_validator.validate_modification(
            order,
            new_quantity=(
                Quantity(Decimal(str(request.new_quantity))) if request.new_quantity else None
            ),
            new_limit_price=(
                Price(Decimal(str(request.new_limit_price))) if request.new_limit_price else None
            ),
            new_stop_price=(
                Price(Decimal(str(request.new_stop_price))) if request.new_stop_price else None
            ),
        )

        if not validation_result.is_valid:
            return ModifyOrderResponse(
                success=False,
                error=validation_result.error_message or "Modification validation failed",
                request_id=request.request_id or uuid4(),
            )

        # Update with broker
        try:
            # Update order entity first
            if request.new_quantity:
                order.quantity = Decimal(str(request.new_quantity))
            if request.new_limit_price:
                order.limit_price = Decimal(str(request.new_limit_price))
            if request.new_stop_price:
                order.stop_price = Decimal(str(request.new_stop_price))

            # Send updated order to broker
            updated_order = self.broker.update_order(order)

            if updated_order:
                await order_repo.update_order(order)

                # Build updates dict for response
                updates: dict[str, Any] = {}
                if request.new_quantity:
                    updates["quantity"] = request.new_quantity
                if request.new_limit_price:
                    updates["limit_price"] = request.new_limit_price
                if request.new_stop_price:
                    updates["stop_price"] = request.new_stop_price

                return ModifyOrderResponse(
                    success=True, modified=True, new_values=updates, request_id=request.request_id
                )
            else:
                return ModifyOrderResponse(
                    success=False,
                    error="Broker failed to modify order",
                    request_id=request.request_id or uuid4(),
                )

        except Exception as e:
            self.logger.error(f"Failed to modify order: {e}")
            return ModifyOrderResponse(
                success=False,
                error=f"Failed to modify order: {e}",
                request_id=request.request_id or uuid4(),
            )


class GetOrderStatusUseCase(TransactionalUseCase[GetOrderStatusRequest, GetOrderStatusResponse]):
    """
    Use case for retrieving order status.

    Queries broker for latest status and updates local records.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        broker: IBroker,
    ):
        """Initialize get order status use case."""
        super().__init__(unit_of_work, "GetOrderStatusUseCase")
        self.broker = broker

    async def validate(self, request: GetOrderStatusRequest) -> str | None:
        """Validate the get order status request."""
        if not request.order_id:
            return "Order ID is required"
        return None

    async def process(self, request: GetOrderStatusRequest) -> GetOrderStatusResponse:
        """Process the get order status request."""
        # Get order from repository
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return GetOrderStatusResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        # Get latest status from broker
        try:
            broker_status = self.broker.get_order_status(order.id)

            # Update order if status changed
            if broker_status and broker_status != order.status:
                # Update based on broker status
                # Update the order status directly since broker_status is OrderStatus enum
                order.status = broker_status

                await order_repo.update_order(order)

            return GetOrderStatusResponse(
                success=True,
                status=order.status.value,
                filled_quantity=(
                    int(ValueObjectConverter.extract_value(order.filled_quantity))
                    if order.filled_quantity
                    else 0
                ),
                average_fill_price=(
                    float(ValueObjectConverter.extract_value(order.average_fill_price))
                    if order.average_fill_price
                    else None
                ),
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            # Return cached status if broker query fails
            return GetOrderStatusResponse(
                success=True,
                status=order.status.value,
                filled_quantity=(
                    int(ValueObjectConverter.extract_value(order.filled_quantity))
                    if order.filled_quantity
                    else 0
                ),
                average_fill_price=(
                    float(ValueObjectConverter.extract_value(order.average_fill_price))
                    if order.average_fill_price
                    else None
                ),
                request_id=request.request_id,
            )

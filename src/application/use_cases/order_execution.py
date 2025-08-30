"""
Order Execution Use Cases

Handles the business logic for order execution, fill processing, and commission calculation.
This extracts complex business logic from the infrastructure brokers to maintain clean architecture.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.entities.order import OrderSide, OrderType
from src.domain.services.commission_calculator import ICommissionCalculator
from src.domain.services.market_microstructure import IMarketMicrostructure
from src.domain.services.order_processor import FillDetails, OrderProcessor
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO


# Request/Response DTOs
@dataclass
class ProcessOrderFillRequest(BaseRequestDTO):
    """Request to process an order fill."""

    order_id: UUID
    fill_price: Decimal
    fill_quantity: int | None = None  # None means fill entire order
    timestamp: datetime | None = None


@dataclass
class ProcessOrderFillResponse(UseCaseResponse):
    """Response from processing an order fill."""

    filled: bool = False
    fill_price: Decimal | None = None
    fill_quantity: int | None = None
    commission: Decimal | None = None
    position_id: UUID | None = None


@dataclass
class SimulateOrderExecutionRequest(BaseRequestDTO):
    """Request to simulate order execution with market impact."""

    order_id: UUID
    market_price: Decimal
    available_liquidity: Decimal | None = None


@dataclass
class SimulateOrderExecutionResponse(UseCaseResponse):
    """Response from simulating order execution."""

    execution_price: Decimal | None = None
    slippage: Decimal | None = None
    market_impact: Decimal | None = None
    estimated_commission: Decimal | None = None


@dataclass
class CalculateCommissionRequest(BaseRequestDTO):
    """Request to calculate commission for a trade."""

    quantity: int
    price: Decimal
    order_type: str
    symbol: str | None = None


@dataclass
class CalculateCommissionResponse(UseCaseResponse):
    """Response with calculated commission."""

    commission: Decimal | None = None
    commission_rate: Decimal | None = None


# Use Case Implementations
class ProcessOrderFillUseCase(
    TransactionalUseCase[ProcessOrderFillRequest, ProcessOrderFillResponse]
):
    """
    Processes order fills and updates portfolio positions.

    This use case encapsulates the complex logic of processing fills,
    calculating commissions, and updating positions that was previously
    in the infrastructure broker.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        order_processor: OrderProcessor,
        commission_calculator: ICommissionCalculator,
    ):
        """Initialize the process order fill use case."""
        super().__init__(unit_of_work, "ProcessOrderFillUseCase")
        self.order_processor = order_processor
        self.commission_calculator = commission_calculator

    async def validate(self, request: ProcessOrderFillRequest) -> str | None:
        """Validate the fill request."""
        if request.fill_price <= 0:
            return "Fill price must be positive"

        if request.fill_quantity is not None and request.fill_quantity <= 0:
            return "Fill quantity must be positive"

        return None

    async def process(self, request: ProcessOrderFillRequest) -> ProcessOrderFillResponse:
        """Process the order fill."""
        # Get the order
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return ProcessOrderFillResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        if not order.is_active():
            return ProcessOrderFillResponse(
                success=False,
                error=f"Cannot fill order in status: {order.status.value}",
                request_id=request.request_id or uuid4(),
            )

        # Get the portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(order.portfolio_id)

        if not portfolio:
            return ProcessOrderFillResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Determine fill quantity
        remaining_quantity = order.get_remaining_quantity()
        fill_quantity = min(
            request.fill_quantity or ValueObjectConverter.extract_value(remaining_quantity),
            ValueObjectConverter.extract_value(remaining_quantity),
        )

        if fill_quantity <= 0:
            return ProcessOrderFillResponse(
                success=False, error="No quantity to fill", request_id=request.request_id or uuid4()
            )

        # Calculate commission
        commission = self.commission_calculator.calculate(
            quantity=Quantity(fill_quantity), price=Money(request.fill_price)
        )

        # Process the fill using domain service
        fill_timestamp = request.timestamp or datetime.now(UTC)

        try:
            # Update order with fill
            order.fill(
                filled_quantity=Quantity(fill_quantity),
                fill_price=Price(request.fill_price),
                timestamp=fill_timestamp,
            )

            # Process position update
            fill_details = FillDetails(
                order=order,
                fill_price=Price(request.fill_price),
                fill_quantity=Quantity(Decimal(str(fill_quantity))),
                commission=commission,
                timestamp=fill_timestamp,
            )
            self.order_processor.process_fill(fill_details, portfolio)

            # Save updates
            await order_repo.update_order(order)
            await portfolio_repo.update_portfolio(portfolio)

            # Save updated positions
            position_repo = self.unit_of_work.positions
            for position in portfolio.positions.values():
                await position_repo.persist_position(position)

            return ProcessOrderFillResponse(
                success=True,
                filled=True,
                fill_price=request.fill_price,
                fill_quantity=int(fill_quantity) if fill_quantity is not None else None,
                commission=ValueObjectConverter.extract_amount(commission),
                position_id=None,  # Position ID would require looking up from portfolio
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to process fill: {e}")
            return ProcessOrderFillResponse(
                success=False,
                error=f"Failed to process fill: {e}",
                request_id=request.request_id or uuid4(),
            )


class SimulateOrderExecutionUseCase(
    TransactionalUseCase[SimulateOrderExecutionRequest, SimulateOrderExecutionResponse]
):
    """
    Simulates order execution with market impact and slippage.

    This extracts the market simulation logic from the infrastructure broker.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        market_microstructure: IMarketMicrostructure,
        commission_calculator: ICommissionCalculator,
    ):
        """Initialize the simulate order execution use case."""
        super().__init__(unit_of_work, "SimulateOrderExecutionUseCase")
        self.market_microstructure = market_microstructure
        self.commission_calculator = commission_calculator

    async def validate(self, request: SimulateOrderExecutionRequest) -> str | None:
        """Validate the simulation request."""
        if request.market_price <= 0:
            return "Market price must be positive"

        if request.available_liquidity is not None and request.available_liquidity <= 0:
            return "Available liquidity must be positive"

        return None

    async def process(
        self, request: SimulateOrderExecutionRequest
    ) -> SimulateOrderExecutionResponse:
        """Simulate the order execution."""
        # Get the order
        order_repo = self.unit_of_work.orders
        order = await order_repo.get_order_by_id(request.order_id)

        if not order:
            return SimulateOrderExecutionResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        # Calculate execution price with market impact
        execution_price = self.market_microstructure.calculate_execution_price(
            base_price=Price(request.market_price),
            side=order.side,
            quantity=Quantity(order.quantity),
            order_type=order.order_type,
        )

        # Calculate slippage
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                slippage = (
                    ValueObjectConverter.extract_value(execution_price) - request.market_price
                )
            else:
                slippage = request.market_price - ValueObjectConverter.extract_value(
                    execution_price
                )
        else:
            slippage = Decimal("0")

        # Calculate market impact
        market_impact = self.market_microstructure.calculate_market_impact(
            price=Price(request.market_price),
            quantity=Quantity(order.quantity),
            average_volume=Quantity(Decimal("1000000")),  # Default ADV
        )

        # Estimate commission
        commission = self.commission_calculator.calculate(
            quantity=order.quantity,
            price=Money(ValueObjectConverter.extract_value(execution_price)),
        )

        return SimulateOrderExecutionResponse(
            success=True,
            execution_price=ValueObjectConverter.extract_value(execution_price),
            slippage=slippage,
            market_impact=market_impact,
            estimated_commission=ValueObjectConverter.extract_amount(commission),
            request_id=request.request_id,
        )


class CalculateCommissionUseCase:
    """
    Calculates commission for trades.

    This centralizes commission calculation logic that was spread across brokers.
    """

    def __init__(self, commission_calculator: ICommissionCalculator) -> None:
        """Initialize the calculate commission use case."""
        self.commission_calculator = commission_calculator
        self.logger = logging.getLogger(__name__)

    async def execute(self, request: CalculateCommissionRequest) -> CalculateCommissionResponse:
        """Calculate the commission."""
        try:
            commission = self.commission_calculator.calculate(
                quantity=Quantity(request.quantity), price=Money(request.price)
            )

            # Get the rate (simplified - would need schedule access)
            rate = (
                self.commission_calculator.schedule.rate
                if hasattr(self.commission_calculator, "schedule")
                else None
            )

            return CalculateCommissionResponse(
                success=True,
                commission=ValueObjectConverter.extract_amount(commission),
                commission_rate=rate,
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate commission: {e}")
            return CalculateCommissionResponse(
                success=False,
                error=f"Failed to calculate commission: {e}",
                request_id=request.request_id or uuid4(),
            )

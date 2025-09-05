"""
Portfolio Management Use Cases

Handles portfolio queries, updates, and position management.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO


# Request/Response DTOs
@dataclass
class GetPortfolioRequest(BaseRequestDTO):
    """Request to get portfolio details."""

    portfolio_id: UUID
    include_positions: bool = True
    include_metrics: bool = True


@dataclass
class GetPortfolioResponse(UseCaseResponse):
    """Response with portfolio details."""

    portfolio: dict[str, Any] | None = None
    positions: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class UpdatePortfolioRequest(BaseRequestDTO):
    """Request to update portfolio settings."""

    portfolio_id: UUID
    name: str | None = None
    max_position_size: Decimal | None = None
    max_positions: int | None = None
    max_leverage: Decimal | None = None
    max_portfolio_risk: Decimal | None = None


@dataclass
class UpdatePortfolioResponse(UseCaseResponse):
    """Response from updating portfolio."""

    updated: bool = False


@dataclass
class GetPositionsRequest(BaseRequestDTO):
    """Request to get portfolio positions."""

    portfolio_id: UUID
    only_open: bool = True
    symbol: str | None = None


@dataclass
class GetPositionsResponse(UseCaseResponse):
    """Response with portfolio positions."""

    positions: list[dict[str, Any]] = field(default_factory=list)
    total_value: Decimal | None = None


@dataclass
class ClosePositionRequest(BaseRequestDTO):
    """Request to close a position."""

    position_id: UUID
    exit_price: Decimal
    reason: str | None = None


@dataclass
class ClosePositionResponse(UseCaseResponse):
    """Response from closing a position."""

    closed: bool = False
    realized_pnl: Decimal | None = None
    total_return: Decimal | None = None


# Use Case Implementations
class GetPortfolioUseCase(TransactionalUseCase[GetPortfolioRequest, GetPortfolioResponse]):
    """
    Retrieves portfolio information with optional positions and metrics.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        risk_calculator: RiskCalculator,
    ):
        """Initialize get portfolio use case."""
        super().__init__(unit_of_work, "GetPortfolioUseCase")
        self.risk_calculator = risk_calculator

    async def validate(self, request: GetPortfolioRequest) -> str | None:
        """Validate the get portfolio request."""
        return None

    async def process(self, request: GetPortfolioRequest) -> GetPortfolioResponse:
        """Process the get portfolio request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return GetPortfolioResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Build portfolio data
        # Use sync methods for portfolio calculations
        total_value = portfolio.get_total_value()
        # Handle case where mocked method returns coroutine
        if hasattr(total_value, "__await__"):
            # If it's a coroutine, await it
            total_value = await total_value

        portfolio_data = {
            "id": str(portfolio.id),
            "name": portfolio.name,
            "cash_balance": float(ValueObjectConverter.extract_amount(portfolio.cash_balance)),
            "initial_capital": float(
                ValueObjectConverter.extract_amount(portfolio.initial_capital)
            ),
            "total_value": float(ValueObjectConverter.extract_amount(total_value)),
            "positions_value": float(
                ValueObjectConverter.extract_amount(portfolio.get_positions_value())
            ),
            "unrealized_pnl": float(
                ValueObjectConverter.extract_amount(portfolio.get_unrealized_pnl())
            ),
            "realized_pnl": float(
                ValueObjectConverter.extract_amount(portfolio.total_realized_pnl)
            ),
            "total_return": float(portfolio.get_return_percentage()),
            "open_positions_count": len(portfolio.get_open_positions()),
            "max_position_size": (
                float(ValueObjectConverter.extract_amount(portfolio.max_position_size))
                if portfolio.max_position_size
                else None
            ),
            "max_positions": portfolio.max_positions,
            "max_leverage": float(portfolio.max_leverage) if portfolio.max_leverage else None,
        }

        # Get positions if requested
        positions_data = None
        if request.include_positions:
            positions = portfolio.get_open_positions()
            positions_data = [
                {
                    "id": str(pos.id),
                    "symbol": pos.symbol,
                    "side": "long" if pos.quantity > 0 else "short",
                    "quantity": abs(ValueObjectConverter.extract_value(pos.quantity)),
                    "entry_price": float(
                        ValueObjectConverter.extract_value(pos.average_entry_price)
                    ),
                    "current_price": (
                        float(ValueObjectConverter.extract_value(pos.current_price))
                        if pos.current_price
                        else None
                    ),
                    "unrealized_pnl": (
                        float(ValueObjectConverter.extract_amount(pos.get_unrealized_pnl()))
                        if pos.get_unrealized_pnl()
                        else 0.0
                    ),
                    "return_pct": (
                        float(pos.get_return_percentage()) if pos.get_return_percentage() else 0.0
                    ),
                    "value": (
                        float(ValueObjectConverter.extract_amount(pos.get_position_value()))
                        if pos.get_position_value()
                        else 0.0
                    ),
                }
                for pos in positions
            ]

        # Calculate metrics if requested
        metrics_data = None
        if request.include_metrics:
            # Calculate portfolio metrics
            sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(
                returns=[portfolio.get_return_percentage()]  # Simplified
            )

            risk_metrics = {
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio else None,
                "max_drawdown": 0.0,  # Would need historical data
                "portfolio_beta": 1.0,  # Would need market correlation
                "value_at_risk_95": 0.0,  # Would need return distribution
            }

            metrics_data = risk_metrics

        return GetPortfolioResponse(
            success=True,
            portfolio=portfolio_data,
            positions=positions_data,
            metrics=metrics_data,
            request_id=request.request_id,
        )


class UpdatePortfolioUseCase(TransactionalUseCase[UpdatePortfolioRequest, UpdatePortfolioResponse]):
    """
    Updates portfolio configuration and limits.
    """

    def __init__(self, unit_of_work: IUnitOfWork) -> None:
        """Initialize update portfolio use case."""
        super().__init__(unit_of_work, "UpdatePortfolioUseCase")

    async def validate(self, request: UpdatePortfolioRequest) -> str | None:
        """Validate the update portfolio request."""
        if request.max_position_size is not None and request.max_position_size <= 0:
            return "Max position size must be positive"

        if request.max_positions is not None and request.max_positions <= 0:
            return "Max positions must be positive"

        if request.max_leverage is not None and request.max_leverage < 1:
            return "Max leverage must be at least 1.0"

        if request.max_portfolio_risk is not None:
            if request.max_portfolio_risk <= 0 or request.max_portfolio_risk > 1:
                return "Max portfolio risk must be between 0 and 1"

        return None

    async def process(self, request: UpdatePortfolioRequest) -> UpdatePortfolioResponse:
        """Process the update portfolio request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return UpdatePortfolioResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Update fields
        if request.name is not None:
            portfolio.name = request.name

        if request.max_position_size is not None:
            portfolio.max_position_size = Money(request.max_position_size)

        if request.max_positions is not None:
            portfolio.max_positions = request.max_positions

        if request.max_leverage is not None:
            portfolio.max_leverage = request.max_leverage

        if request.max_portfolio_risk is not None:
            portfolio.max_portfolio_risk = request.max_portfolio_risk

        # Save updates
        await portfolio_repo.update_portfolio(portfolio)

        return UpdatePortfolioResponse(success=True, updated=True, request_id=request.request_id)


class GetPositionsUseCase(TransactionalUseCase[GetPositionsRequest, GetPositionsResponse]):
    """
    Retrieves portfolio positions with optional filtering.
    """

    def __init__(self, unit_of_work: IUnitOfWork) -> None:
        """Initialize get positions use case."""
        super().__init__(unit_of_work, "GetPositionsUseCase")

    async def validate(self, request: GetPositionsRequest) -> str | None:
        """Validate the get positions request."""
        return None

    async def process(self, request: GetPositionsRequest) -> GetPositionsResponse:
        """Process the get positions request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return GetPositionsResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Get positions
        if request.only_open:
            positions = portfolio.get_open_positions()
        else:
            positions = list(portfolio.positions.values())

        # Filter by symbol if specified
        if request.symbol:
            positions = [p for p in positions if p.symbol == request.symbol]

        # Build response data
        positions_data = [
            {
                "id": str(pos.id),
                "symbol": pos.symbol,
                "side": "long" if pos.quantity > 0 else "short",
                "quantity": abs(ValueObjectConverter.extract_value(pos.quantity)),
                "entry_price": float(ValueObjectConverter.extract_value(pos.average_entry_price)),
                "current_price": (
                    float(ValueObjectConverter.extract_value(pos.current_price))
                    if pos.current_price
                    else None
                ),
                "unrealized_pnl": (
                    float(ValueObjectConverter.extract_amount(pos.get_unrealized_pnl()))
                    if pos.get_unrealized_pnl()
                    else 0.0
                ),
                "realized_pnl": (
                    float(ValueObjectConverter.extract_amount(pos.realized_pnl))
                    if pos.realized_pnl is not None
                    else 0.0
                ),
                "return_pct": (
                    float(pos.get_return_percentage()) if pos.get_return_percentage() else 0.0
                ),
                "value": (
                    float(ValueObjectConverter.extract_amount(pos.get_position_value()))
                    if pos.get_position_value()
                    else 0.0
                ),
                "is_open": not pos.is_closed(),
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
            }
            for pos in positions
        ]

        # Calculate total value
        total_value = sum(
            pos.get_position_value() if pos.get_position_value() is not None else Decimal("0")
            for pos in positions
            if not pos.is_closed()
        )

        return GetPositionsResponse(
            success=True,
            positions=positions_data,
            total_value=total_value,
            request_id=request.request_id,
        )


class ClosePositionUseCase(TransactionalUseCase[ClosePositionRequest, ClosePositionResponse]):
    """
    Closes a position and calculates realized P&L.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        position_manager: PositionManager,
    ):
        """Initialize close position use case."""
        super().__init__(unit_of_work, "ClosePositionUseCase")
        self.position_manager = position_manager

    async def validate(self, request: ClosePositionRequest) -> str | None:
        """Validate the close position request."""
        if request.exit_price <= 0:
            return "Exit price must be positive"
        return None

    async def process(self, request: ClosePositionRequest) -> ClosePositionResponse:
        """Process the close position request."""
        # Get position
        position_repo = self.unit_of_work.positions
        position = await position_repo.get_position_by_id(request.position_id)

        if not position:
            return ClosePositionResponse(
                success=False, error="Position not found", request_id=request.request_id or uuid4()
            )

        if position.is_closed():
            return ClosePositionResponse(
                success=False,
                error="Position is already closed",
                request_id=request.request_id or uuid4(),
            )

        # Close position manually (simplified - in production would use order)
        exit_price = Price(request.exit_price)
        from decimal import Decimal

        from src.domain.entities.order import Order, OrderSide, OrderType
        from src.domain.value_objects import Quantity

        # Create a mock closing order
        # Handle both Quantity objects and raw values
        position_quantity = ValueObjectConverter.extract_value(position.quantity)

        closing_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL if position_quantity > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(abs(position_quantity)),
        )
        closing_order.average_fill_price = Price(Decimal(str(request.exit_price)))

        self.position_manager.close_position(
            position=position, order=closing_order, exit_price=exit_price
        )

        # Get realized P&L and return
        realized_pnl = position.realized_pnl
        total_return = position.get_return_percentage()

        # Save updated position
        await position_repo.update_position(position)

        # Update portfolio (simplified - in production would track portfolio_id in position)
        # For now, we skip portfolio update as Position doesn't have portfolio_id
        # This would typically be handled by maintaining the relationship separately
        # TODO: Enable once position-portfolio relationship is established
        # if portfolio_id:
        #     portfolio = await portfolio_repo.get(portfolio_id)
        #     portfolio.cash_balance += realized_pnl
        #     portfolio.total_realized_pnl += realized_pnl
        #     await portfolio_repo.update_portfolio(portfolio)

        # Convert Money to Decimal for response DTO
        realized_pnl_decimal = ValueObjectConverter.extract_amount(realized_pnl)

        return ClosePositionResponse(
            success=True,
            closed=True,
            realized_pnl=realized_pnl_decimal,
            total_return=total_return,
            request_id=request.request_id,
        )

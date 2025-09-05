"""
Open Position Use Case

Handles opening new positions in a portfolio, extracting orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.services.portfolio_service import PortfolioService
from src.domain.entities.portfolio import PositionRequest
from src.domain.value_objects import Money, Price, Quantity

from ..base import TransactionalUseCase, UseCaseResponse
from ..base_request import BaseRequestDTO


@dataclass
class OpenPositionRequest(BaseRequestDTO):
    """Request to open a position in a portfolio."""

    portfolio_id: UUID
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    commission: Decimal = Decimal("0")
    strategy: str | None = None


@dataclass
class OpenPositionResponse(UseCaseResponse):
    """Response from opening a position."""

    position: dict[str, Any] | None = None
    portfolio_updated: bool = False


class OpenPositionUseCase(TransactionalUseCase[OpenPositionRequest, OpenPositionResponse]):
    """
    Opens a new position in a portfolio.

    Extracts the exact orchestration logic from Portfolio.open_position(),
    coordinating between validation, transaction processing, and state updates.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the open position use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "OpenPositionUseCase")

    async def validate(self, request: OpenPositionRequest) -> str | None:
        """Validate the open position request.

        Args:
            request: The open position request

        Returns:
            Error message if validation fails, None otherwise
        """
        if not request.symbol:
            return "Symbol is required"

        if request.quantity == 0:
            return "Quantity must be non-zero"

        if request.entry_price <= 0:
            return "Entry price must be positive"

        if request.commission < 0:
            return "Commission cannot be negative"

        return None

    async def process(self, request: OpenPositionRequest) -> OpenPositionResponse:
        """Process the open position request.

        Extracts the EXACT orchestration logic from Portfolio.open_position():
        1. Get portfolio and validate it exists
        2. Create PositionRequest with value objects
        3. Use PortfolioTransactionService to validate and create position
        4. Use PortfolioStateService to update portfolio state
        5. Save updated portfolio and position

        Args:
            request: The validated request

        Returns:
            Response with position data and update status
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return OpenPositionResponse(
                success=False,
                error="Portfolio not found",
                request_id=request.request_id or request.request_id,
            )

        try:
            # Create PositionRequest with proper value objects (exact same as Portfolio entity)
            position_request = PositionRequest(
                symbol=request.symbol,
                quantity=Quantity(request.quantity),
                entry_price=Price(request.entry_price),
                commission=Money(request.commission),
                strategy=request.strategy,
            )

            # EXACT orchestration logic from Portfolio.open_position()
            # Using PortfolioService which handles all the orchestration
            service = PortfolioService()
            position = service.open_position(portfolio, position_request)

            # Save updated entities
            await portfolio_repo.update_portfolio(portfolio)

            # If it's a new position, also save it
            if (
                position.symbol not in portfolio.positions
                or portfolio.positions[position.symbol] == position
            ):
                position_repo = self.unit_of_work.positions
                if hasattr(position_repo, "add_position"):
                    await position_repo.add_position(position)
                elif hasattr(position_repo, "update_position"):
                    await position_repo.update_position(position)

            # Build response data
            position_data = {
                "id": str(position.id),
                "symbol": position.symbol,
                "quantity": float(position.quantity.value),
                "entry_price": float(position.average_entry_price.value),
                "commission": float(position.commission_paid.amount),
                "strategy": position.strategy,
                "opened_at": position.opened_at.isoformat() if position.opened_at else None,
                "is_open": not position.is_closed(),
            }

            return OpenPositionResponse(
                success=True,
                position=position_data,
                portfolio_updated=True,
                request_id=request.request_id,
            )

        except ValueError as e:
            return OpenPositionResponse(success=False, error=str(e), request_id=request.request_id)
        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error opening position: {e}", exc_info=True)
            return OpenPositionResponse(
                success=False,
                error="Failed to open position due to internal error",
                request_id=request.request_id,
            )

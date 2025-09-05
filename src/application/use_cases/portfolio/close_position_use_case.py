"""
Close Position Use Case

Handles closing positions in a portfolio, extracting orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.
"""

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.services.portfolio_service import PortfolioService
from src.domain.value_objects import Money, Price, Quantity

from ..base import TransactionalUseCase, UseCaseResponse
from ..base_request import BaseRequestDTO


@dataclass
class ClosePositionRequest(BaseRequestDTO):
    """Request to close a position in a portfolio."""

    portfolio_id: UUID
    symbol: str
    exit_price: Decimal
    commission: Decimal = Decimal("0")
    quantity: Decimal | None = None  # For partial closes


@dataclass
class ClosePositionResponse(UseCaseResponse):
    """Response from closing a position."""

    pnl: Decimal | None = None
    net_proceeds: Decimal | None = None
    portfolio_updated: bool = False
    position_closed: bool = False


class ClosePositionUseCase(TransactionalUseCase[ClosePositionRequest, ClosePositionResponse]):
    """
    Closes a position in a portfolio.

    Extracts the exact orchestration logic from Portfolio.close_position(),
    coordinating between validation, transaction processing, and state updates.
    Supports both full and partial position closes.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the close position use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "ClosePositionUseCase")

    async def validate(self, request: ClosePositionRequest) -> str | None:
        """Validate the close position request.

        Args:
            request: The close position request

        Returns:
            Error message if validation fails, None otherwise
        """
        if not request.symbol:
            return "Symbol is required"

        if request.exit_price <= 0:
            return "Exit price must be positive"

        if request.commission < 0:
            return "Commission cannot be negative"

        if request.quantity is not None and request.quantity == 0:
            return "Quantity must be non-zero for partial closes"

        return None

    async def process(self, request: ClosePositionRequest) -> ClosePositionResponse:
        """Process the close position request.

        Extracts the EXACT orchestration logic from Portfolio.close_position():
        1. Get portfolio and validate it exists
        2. Check if position exists and is open
        3. Use PortfolioTransactionService for full or partial close
        4. Use PortfolioStateService to update portfolio cash and statistics
        5. Save updated portfolio

        Args:
            request: The validated request

        Returns:
            Response with P&L, proceeds, and update status
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return ClosePositionResponse(
                success=False, error="Portfolio not found", request_id=request.request_id
            )

        # Check if position exists
        if request.symbol not in portfolio.positions:
            return ClosePositionResponse(
                success=False,
                error=f"No position found for {request.symbol}",
                request_id=request.request_id,
            )

        position = portfolio.positions[request.symbol]
        if position.is_closed():
            return ClosePositionResponse(
                success=False,
                error=f"Position for {request.symbol} is already closed",
                request_id=request.request_id,
            )

        try:
            # Create value objects
            exit_price = Price(request.exit_price)
            commission = Money(request.commission)

            # EXACT orchestration logic from Portfolio.close_position()
            # Using PortfolioService which handles all the orchestration
            quantity = Quantity(request.quantity) if request.quantity is not None else None

            # Close position returns the P&L
            service = PortfolioService()
            pnl = service.close_position(
                portfolio, request.symbol, exit_price, commission, quantity
            )

            # Check if position is now fully closed
            position_closed = position.is_closed()

            # Calculate net proceeds for response
            if position_closed:
                # For full close, net proceeds is the final cash received
                net_proceeds = pnl + (position.quantity * exit_price) - commission
            else:
                # For partial close, calculate based on quantity closed
                closed_qty = quantity if quantity else position.quantity
                net_proceeds = (closed_qty * exit_price) - commission

            # Save updated portfolio
            await portfolio_repo.update_portfolio(portfolio)

            # Update position in repository if needed
            if hasattr(self.unit_of_work.positions, "update_position"):
                await self.unit_of_work.positions.update_position(position)

            return ClosePositionResponse(
                success=True,
                pnl=pnl.amount,
                net_proceeds=net_proceeds.amount,
                portfolio_updated=True,
                position_closed=position_closed,
                request_id=request.request_id,
            )

        except ValueError as e:
            return ClosePositionResponse(success=False, error=str(e), request_id=request.request_id)
        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error closing position: {e}", exc_info=True)
            return ClosePositionResponse(
                success=False,
                error="Failed to close position due to internal error",
                request_id=request.request_id,
            )

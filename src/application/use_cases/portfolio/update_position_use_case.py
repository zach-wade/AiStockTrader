"""
Update Position Use Case

Handles updating position prices in a portfolio, extracting orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.
"""

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.value_objects import Price

from ..base import TransactionalUseCase, UseCaseResponse
from ..base_request import BaseRequestDTO


@dataclass
class UpdatePositionRequest(BaseRequestDTO):
    """Request to update a position's market price."""

    portfolio_id: UUID
    symbol: str
    price: Decimal


@dataclass
class UpdatePositionPricesRequest(BaseRequestDTO):
    """Request to update multiple position prices."""

    portfolio_id: UUID
    prices: dict[str, Decimal]  # symbol -> price mapping


@dataclass
class UpdatePositionResponse(UseCaseResponse):
    """Response from updating position price(s)."""

    updated_symbols: list[str] | None = None
    portfolio_updated: bool = False


class UpdatePositionUseCase(TransactionalUseCase[UpdatePositionRequest, UpdatePositionResponse]):
    """
    Updates a position's market price in a portfolio.

    Extracts the exact orchestration logic from Portfolio.update_position_price(),
    coordinating position price updates and portfolio versioning.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the update position use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "UpdatePositionUseCase")

    async def validate(self, request: UpdatePositionRequest) -> str | None:
        """Validate the update position request.

        Args:
            request: The update position request

        Returns:
            Error message if validation fails, None otherwise
        """
        if not request.symbol:
            return "Symbol is required"

        if request.price <= 0:
            return "Price must be positive"

        return None

    async def process(self, request: UpdatePositionRequest) -> UpdatePositionResponse:
        """Process the update position request.

        Extracts the EXACT orchestration logic from Portfolio.update_position_price():
        1. Get portfolio and validate it exists
        2. Check if position exists and is open
        3. Update position market price
        4. Increment portfolio version
        5. Save updated portfolio

        Args:
            request: The validated request

        Returns:
            Response with update status
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return UpdatePositionResponse(
                success=False, error="Portfolio not found", request_id=request.request_id
            )

        try:
            # EXACT orchestration logic from Portfolio.update_position_price()
            if request.symbol not in portfolio.positions:
                return UpdatePositionResponse(
                    success=False,
                    error=f"No position found for {request.symbol}",
                    request_id=request.request_id,
                )

            price = Price(request.price)

            # Check if position is open before updating
            if not portfolio.positions[request.symbol].is_closed():
                portfolio.positions[request.symbol].update_market_price(price)
                portfolio._increment_version()

                # Save updated portfolio
                await portfolio_repo.update_portfolio(portfolio)

                return UpdatePositionResponse(
                    success=True,
                    updated_symbols=[request.symbol],
                    portfolio_updated=True,
                    request_id=request.request_id,
                )
            else:
                return UpdatePositionResponse(
                    success=False,
                    error=f"Position for {request.symbol} is closed and cannot be updated",
                    request_id=request.request_id,
                )

        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error updating position: {e}", exc_info=True)
            return UpdatePositionResponse(
                success=False,
                error="Failed to update position due to internal error",
                request_id=request.request_id,
            )


class UpdatePositionPricesUseCase(
    TransactionalUseCase[UpdatePositionPricesRequest, UpdatePositionResponse]
):
    """
    Updates multiple position prices in a portfolio.

    Extracts the exact orchestration logic from Portfolio.update_all_prices(),
    coordinating batch price updates and portfolio versioning.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the update position prices use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "UpdatePositionPricesUseCase")

    async def validate(self, request: UpdatePositionPricesRequest) -> str | None:
        """Validate the update position prices request.

        Args:
            request: The update position prices request

        Returns:
            Error message if validation fails, None otherwise
        """
        if not request.prices:
            return "Prices dictionary cannot be empty"

        for symbol, price in request.prices.items():
            if not symbol:
                return "Symbol cannot be empty"
            if price <= 0:
                return f"Price for {symbol} must be positive"

        return None

    async def process(self, request: UpdatePositionPricesRequest) -> UpdatePositionResponse:
        """Process the update position prices request.

        Extracts the EXACT orchestration logic from Portfolio.update_all_prices():
        1. Get portfolio and validate it exists
        2. Iterate through prices and update open positions
        3. Track if any updates occurred
        4. Increment portfolio version if updated
        5. Save updated portfolio

        Args:
            request: The validated request

        Returns:
            Response with update status and updated symbols
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return UpdatePositionResponse(
                success=False, error="Portfolio not found", request_id=request.request_id
            )

        try:
            # EXACT orchestration logic from Portfolio.update_all_prices()
            updated_symbols = []
            updated = False

            for symbol, price_value in request.prices.items():
                if symbol in portfolio.positions and not portfolio.positions[symbol].is_closed():
                    price = Price(price_value)
                    portfolio.positions[symbol].update_market_price(price)
                    updated_symbols.append(symbol)
                    updated = True

            if updated:
                portfolio._increment_version()

                # Save updated portfolio
                await portfolio_repo.update_portfolio(portfolio)

            return UpdatePositionResponse(
                success=True,
                updated_symbols=updated_symbols,
                portfolio_updated=updated,
                request_id=request.request_id,
            )

        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error updating position prices: {e}", exc_info=True)
            return UpdatePositionResponse(
                success=False,
                error="Failed to update position prices due to internal error",
                request_id=request.request_id,
            )

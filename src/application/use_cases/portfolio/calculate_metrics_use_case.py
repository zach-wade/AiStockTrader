"""
Calculate Portfolio Metrics Use Case

Handles portfolio metrics calculations, extracting orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.
"""

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.services.portfolio_calculator import PortfolioCalculator
from src.domain.value_objects import Money, Price

from ..base import TransactionalUseCase, UseCaseResponse
from ..base_request import BaseRequestDTO


@dataclass
class CalculateMetricsRequest(BaseRequestDTO):
    """Request to calculate portfolio metrics."""

    portfolio_id: UUID
    include_advanced: bool = False
    risk_free_rate: Decimal = Decimal("0.02")
    historical_returns: list[Decimal] | None = None
    historical_values: list[Decimal] | None = None
    current_prices: dict[str, Decimal] | None = None  # For real-time calculations


@dataclass
class PortfolioMetrics:
    """Portfolio metrics data structure."""

    # Basic Metrics
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    return_percentage: Decimal
    total_return: Decimal

    # Position Summary
    open_positions_count: int
    closed_positions_count: int

    # Trade Statistics
    win_rate: Decimal | None = None
    profit_factor: Decimal | None = None
    average_win: Decimal | None = None
    average_loss: Decimal | None = None

    # Risk Metrics (Advanced)
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None
    value_at_risk: Decimal | None = None
    portfolio_beta: Decimal | None = None


@dataclass
class CalculateMetricsResponse(UseCaseResponse):
    """Response from calculating portfolio metrics."""

    metrics: PortfolioMetrics | None = None


class CalculatePortfolioMetricsUseCase(
    TransactionalUseCase[CalculateMetricsRequest, CalculateMetricsResponse]
):
    """
    Calculates portfolio metrics and performance indicators.

    Extracts the exact orchestration logic from Portfolio metrics delegation methods,
    coordinating all metrics calculations through the PortfolioCalculator service.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the calculate metrics use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "CalculatePortfolioMetricsUseCase")

    async def validate(self, request: CalculateMetricsRequest) -> str | None:
        """Validate the calculate metrics request.

        Args:
            request: The calculate metrics request

        Returns:
            Error message if validation fails, None otherwise
        """
        if request.risk_free_rate < 0:
            return "Risk-free rate cannot be negative"

        if request.current_prices:
            for symbol, price in request.current_prices.items():
                if price <= 0:
                    return f"Price for {symbol} must be positive"

        return None

    async def process(self, request: CalculateMetricsRequest) -> CalculateMetricsResponse:
        """Process the calculate metrics request.

        Extracts the EXACT orchestration logic from Portfolio metrics delegation methods:
        1. Get portfolio and validate it exists
        2. Optionally update positions with current prices
        3. Calculate basic metrics via PortfolioMetricsCalculator
        4. Calculate advanced metrics if requested
        5. Return comprehensive metrics structure

        Args:
            request: The validated request

        Returns:
            Response with calculated metrics
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return CalculateMetricsResponse(
                success=False, error="Portfolio not found", request_id=request.request_id
            )

        try:
            # Update positions with current prices if provided (from Portfolio.total_portfolio_value)
            if request.current_prices:
                for position in portfolio.get_open_positions():
                    if position.symbol not in request.current_prices:
                        return CalculateMetricsResponse(
                            success=False,
                            error=f"Price not found for symbol: {position.symbol}",
                            request_id=request.request_id,
                        )
                    position.update_market_price(Price(request.current_prices[position.symbol]))

            # EXACT orchestration logic from Portfolio metrics delegation methods
            calculator = PortfolioCalculator

            # Basic Metrics (exact same delegation as Portfolio entity)
            total_value = calculator.get_total_value(portfolio)
            positions_value = calculator.get_positions_value(portfolio)
            unrealized_pnl = calculator.get_unrealized_pnl(portfolio)
            total_pnl = calculator.get_total_pnl(portfolio)
            return_percentage = calculator.get_return_percentage(portfolio)
            total_return = calculator.get_total_return(portfolio)

            # Trade Statistics (exact same delegation as Portfolio entity)
            win_rate = calculator.get_win_rate(portfolio)
            profit_factor = calculator.get_profit_factor(portfolio)
            average_win = calculator.get_average_win(portfolio)
            average_loss = calculator.get_average_loss(portfolio)

            # Position counts
            open_positions_count = len(portfolio.get_open_positions())
            closed_positions_count = len(portfolio.get_closed_positions())

            # Build basic metrics
            metrics = PortfolioMetrics(
                total_value=total_value.amount,
                cash_balance=portfolio.cash_balance.amount,
                positions_value=positions_value.amount,
                unrealized_pnl=unrealized_pnl.amount,
                total_pnl=total_pnl.amount,
                return_percentage=return_percentage,
                total_return=total_return,
                open_positions_count=open_positions_count,
                closed_positions_count=closed_positions_count,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win.amount if average_win else None,
                average_loss=average_loss.amount if average_loss else None,
            )

            # Advanced metrics if requested (exact same delegation as Portfolio entity)
            if request.include_advanced:
                sharpe_ratio = calculator.get_sharpe_ratio(
                    portfolio, request.risk_free_rate, request.historical_returns
                )

                # Convert historical values if provided
                historical_money_values = None
                if request.historical_values:
                    historical_money_values = [Money(value) for value in request.historical_values]

                max_drawdown = calculator.get_max_drawdown(portfolio, historical_money_values)

                # Set advanced metrics
                metrics.sharpe_ratio = sharpe_ratio
                metrics.max_drawdown = max_drawdown

            return CalculateMetricsResponse(
                success=True, metrics=metrics, request_id=request.request_id
            )

        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error calculating metrics: {e}", exc_info=True)
            return CalculateMetricsResponse(
                success=False,
                error="Failed to calculate metrics due to internal error",
                request_id=request.request_id,
            )

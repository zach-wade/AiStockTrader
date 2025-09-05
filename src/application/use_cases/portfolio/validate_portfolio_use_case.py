"""
Validate Portfolio Use Case

Handles portfolio validation logic, extracting orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.services.portfolio_service import PortfolioService
from src.domain.services.portfolio_validator_consolidated import PortfolioValidator
from src.domain.value_objects import Price, Quantity

from ..base import TransactionalUseCase, UseCaseResponse
from ..base_request import BaseRequestDTO


@dataclass
class ValidatePortfolioRequest(BaseRequestDTO):
    """Request to validate a portfolio."""

    portfolio_id: UUID
    check_position_limits: bool = True
    check_risk_limits: bool = True
    check_business_rules: bool = True


@dataclass
class PositionValidationInfo:
    """Information about position validation."""

    symbol: str
    can_open: bool
    reason: str | None = None
    max_quantity: Decimal | None = None


@dataclass
class ValidatePortfolioResponse(UseCaseResponse):
    """Response from portfolio validation."""

    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    position_validations: list[PositionValidationInfo] = field(default_factory=list)
    risk_metrics: dict[str, Any] | None = None


class ValidatePortfolioUseCase(
    TransactionalUseCase[ValidatePortfolioRequest, ValidatePortfolioResponse]
):
    """
    Validates portfolio state and business rules.

    Extracts the exact orchestration logic from Portfolio validation methods,
    coordinating comprehensive portfolio validation through service layers.
    """

    def __init__(self, unit_of_work: IUnitOfWork):
        """Initialize the validate portfolio use case.

        Args:
            unit_of_work: Unit of work for transaction management
        """
        super().__init__(unit_of_work, "ValidatePortfolioUseCase")
        self.portfolio_service = PortfolioService()

    async def validate(self, request: ValidatePortfolioRequest) -> str | None:
        """Validate the validate portfolio request.

        Args:
            request: The validate portfolio request

        Returns:
            Error message if validation fails, None otherwise
        """
        # No additional validation needed for this request
        return None

    async def process(self, request: ValidatePortfolioRequest) -> ValidatePortfolioResponse:
        """Process the validate portfolio request.

        Extracts the EXACT orchestration logic from Portfolio validation methods:
        1. Get portfolio and validate it exists
        2. Use PortfolioValidationService to validate business rules
        3. Check position limits and risk constraints
        4. Validate each position's ability to be opened
        5. Return comprehensive validation results

        Args:
            request: The validated request

        Returns:
            Response with validation results and any errors
        """
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return ValidatePortfolioResponse(
                success=False, error="Portfolio not found", request_id=request.request_id
            )

        try:
            validation_errors = []
            position_validations = []
            is_valid = True

            # EXACT orchestration logic from Portfolio._validate()
            if request.check_business_rules:
                # Validate portfolio risk
                risk_errors = PortfolioValidator.validate_portfolio_risk(portfolio)
                if risk_errors:
                    validation_errors.extend(risk_errors)
                    is_valid = False

            # Check position limits and constraints
            if request.check_position_limits:
                # Check if position limit is reached
                if portfolio.is_position_limit_reached():
                    validation_errors.append(
                        f"Position limit reached: {portfolio.get_position_count()}/{portfolio.max_positions}"
                    )
                    is_valid = False

                # Validate existing positions
                for symbol, position in portfolio.positions.items():
                    if not position.is_closed():
                        # Check if position violates size limits
                        position_value = position.get_position_value()
                        if position_value and position_value > portfolio.max_position_size:
                            validation_errors.append(
                                f"Position {symbol} value {position_value} exceeds limit {portfolio.max_position_size}"
                            )
                            is_valid = False

            # Check risk limits
            if request.check_risk_limits:
                from src.domain.services.portfolio_calculator import PortfolioCalculator

                # Calculate current leverage
                total_value = PortfolioCalculator.get_total_value(portfolio)
                positions_value = PortfolioCalculator.get_positions_value(portfolio)

                if portfolio.cash_balance.amount > 0:
                    current_leverage = positions_value.amount / portfolio.cash_balance.amount
                    if current_leverage > portfolio.max_leverage:
                        validation_errors.append(
                            f"Current leverage {current_leverage:.2f} exceeds limit {portfolio.max_leverage}"
                        )
                        is_valid = False

                # Calculate current portfolio risk
                if total_value.amount > 0:
                    portfolio_risk = positions_value.amount / total_value.amount
                    if portfolio_risk > portfolio.max_portfolio_risk:
                        validation_errors.append(
                            f"Current portfolio risk {portfolio_risk:.2%} exceeds limit {portfolio.max_portfolio_risk:.2%}"
                        )
                        is_valid = False

            # Test position opening capabilities (similar to Portfolio.can_open_position)
            # Test with small amounts to check general capability
            test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]  # Common test symbols
            test_quantity = Quantity(Decimal("1"))  # Small test quantity
            test_price = Price(Decimal("100"))  # Reasonable test price

            for symbol in test_symbols:
                if symbol not in portfolio.positions:  # Only test new positions
                    can_open, reason = self.portfolio_service.can_open_position(
                        portfolio, symbol, test_quantity, test_price
                    )

                    # Calculate max quantity if position can be opened
                    max_quantity = None
                    if can_open:
                        # Calculate max quantity based on available cash and position size limit
                        available_cash = portfolio.cash_balance.amount
                        max_by_cash = available_cash / test_price.value
                        max_by_position_limit = (
                            portfolio.max_position_size.amount / test_price.value
                        )
                        max_quantity = min(max_by_cash, max_by_position_limit)

                    position_validations.append(
                        PositionValidationInfo(
                            symbol=symbol,
                            can_open=can_open,
                            reason=reason,
                            max_quantity=max_quantity,
                        )
                    )

            # Build risk metrics summary
            risk_metrics = {
                "current_leverage": (
                    positions_value.amount / portfolio.cash_balance.amount
                    if portfolio.cash_balance.amount > 0
                    else 0
                ),
                "portfolio_risk": (
                    positions_value.amount / total_value.amount if total_value.amount > 0 else 0
                ),
                "cash_utilization": (
                    (portfolio.initial_capital.amount - portfolio.cash_balance.amount)
                    / portfolio.initial_capital.amount
                    if portfolio.initial_capital.amount > 0
                    else 0
                ),
                "position_count": portfolio.get_position_count(),
                "max_positions": portfolio.max_positions,
                "positions_available": portfolio.max_positions - portfolio.get_position_count(),
            }

            return ValidatePortfolioResponse(
                success=True,
                is_valid=is_valid,
                validation_errors=validation_errors,
                position_validations=position_validations,
                risk_metrics=risk_metrics,
                request_id=request.request_id,
            )

        except Exception as e:
            # Log the error and return generic error message
            self.logger.error(f"Unexpected error validating portfolio: {e}", exc_info=True)
            return ValidatePortfolioResponse(
                success=False,
                error="Failed to validate portfolio due to internal error",
                request_id=request.request_id,
            )

"""
Risk Management Use Cases

Handles risk calculation, validation, and monitoring.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.price import Price

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO


# Request/Response DTOs
@dataclass
class CalculateRiskRequest(BaseRequestDTO):
    """Request to calculate portfolio risk metrics."""

    portfolio_id: UUID
    include_var: bool = True
    include_sharpe: bool = True
    include_drawdown: bool = True
    confidence_level: float = 0.95


@dataclass
class CalculateRiskResponse(UseCaseResponse):
    """Response with risk metrics."""

    value_at_risk: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None
    portfolio_beta: Decimal | None = None
    risk_score: Decimal | None = None


@dataclass
class ValidateOrderRiskRequest(BaseRequestDTO):
    """Request to validate order risk."""

    order_id: UUID
    portfolio_id: UUID
    current_price: Decimal


@dataclass
class ValidateOrderRiskResponse(UseCaseResponse):
    """Response from order risk validation."""

    is_valid: bool = False
    risk_violations: list[str] | None = None
    risk_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.risk_violations is None:
            self.risk_violations = []


@dataclass
class GetRiskMetricsRequest(BaseRequestDTO):
    """Request to get current risk metrics."""

    portfolio_id: UUID


@dataclass
class GetRiskMetricsResponse(UseCaseResponse):
    """Response with current risk metrics."""

    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}


# Use Case Implementations
class CalculateRiskUseCase(TransactionalUseCase[CalculateRiskRequest, CalculateRiskResponse]):
    """
    Calculates comprehensive risk metrics for a portfolio.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        risk_calculator: RiskCalculator,
    ):
        """Initialize calculate risk use case."""
        super().__init__(unit_of_work, "CalculateRiskUseCase")
        self.risk_calculator = risk_calculator

    async def validate(self, request: CalculateRiskRequest) -> str | None:
        """Validate the calculate risk request."""
        if request.confidence_level <= 0 or request.confidence_level >= 1:
            return "Confidence level must be between 0 and 1"
        return None

    async def process(self, request: CalculateRiskRequest) -> CalculateRiskResponse:
        """Process the risk calculation request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return CalculateRiskResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        response = CalculateRiskResponse(success=True, request_id=request.request_id)

        # Calculate VaR if requested
        if request.include_var:
            # Simplified - would need historical returns
            var = self.risk_calculator.calculate_portfolio_var(
                portfolio=portfolio,
                confidence_level=Decimal(str(request.confidence_level)),
                time_horizon=1,
            )
            response.value_at_risk = ValueObjectConverter.to_decimal(var)

        # Calculate Sharpe ratio if requested
        if request.include_sharpe:
            # Simplified - would need return history
            sharpe = self.risk_calculator.calculate_sharpe_ratio(
                returns=[portfolio.get_total_return()], risk_free_rate=Decimal("0.02")
            )
            response.sharpe_ratio = sharpe

        # Calculate max drawdown if requested
        if request.include_drawdown:
            # Simplified - would need equity curve
            drawdown = self.risk_calculator.calculate_max_drawdown(
                portfolio_history=[portfolio.initial_capital, portfolio.get_total_value()]
            )
            response.max_drawdown = drawdown

        # Calculate overall risk score
        risk_score = Decimal("0.5")  # Simplified scoring
        response.risk_score = risk_score

        return response


class ValidateOrderRiskUseCase(
    TransactionalUseCase[ValidateOrderRiskRequest, ValidateOrderRiskResponse]
):
    """
    Validates that an order meets risk requirements.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        risk_calculator: RiskCalculator,
    ):
        """Initialize validate order risk use case."""
        super().__init__(unit_of_work, "ValidateOrderRiskUseCase")
        self.risk_calculator = risk_calculator

    async def validate(self, request: ValidateOrderRiskRequest) -> str | None:
        """Validate the request."""
        if request.current_price <= 0:
            return "Current price must be positive"
        return None

    async def process(self, request: ValidateOrderRiskRequest) -> ValidateOrderRiskResponse:
        """Process the order risk validation."""
        # Get order and portfolio
        order_repo = self.unit_of_work.orders
        portfolio_repo = self.unit_of_work.portfolios

        order = await order_repo.get_order_by_id(request.order_id)
        if not order:
            return ValidateOrderRiskResponse(
                success=False, error="Order not found", request_id=request.request_id or uuid4()
            )

        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)
        if not portfolio:
            return ValidateOrderRiskResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Check risk limits
        current_price = Price(request.current_price)
        risk_check = self.risk_calculator.check_risk_limits(portfolio=portfolio, new_order=order)
        violations = [] if risk_check[0] else [risk_check[1]]

        # Calculate risk metrics
        quantity_value = ValueObjectConverter.extract_value(order.quantity)
        position_value = quantity_value * ValueObjectConverter.extract_value(current_price)
        portfolio_value = ValueObjectConverter.extract_amount(portfolio.get_total_value())
        cash_balance_value = ValueObjectConverter.extract_amount(portfolio.cash_balance)

        risk_metrics = {
            "position_size_pct": float(position_value / portfolio_value * 100),
            "leverage": float(portfolio_value / cash_balance_value),
            "concentration": float(position_value / portfolio_value),
            "max_loss": float(position_value * Decimal("0.1")),  # 10% max loss assumption
        }

        return ValidateOrderRiskResponse(
            success=True,
            is_valid=len(violations) == 0,
            risk_violations=violations,
            risk_metrics=risk_metrics,
            request_id=request.request_id,
        )


class GetRiskMetricsUseCase(TransactionalUseCase[GetRiskMetricsRequest, GetRiskMetricsResponse]):
    """
    Gets current risk metrics for monitoring.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        risk_calculator: RiskCalculator,
    ):
        """Initialize get risk metrics use case."""
        super().__init__(unit_of_work, "GetRiskMetricsUseCase")
        self.risk_calculator = risk_calculator

    async def validate(self, request: GetRiskMetricsRequest) -> str | None:
        """Validate the request."""
        return None

    async def process(self, request: GetRiskMetricsRequest) -> GetRiskMetricsResponse:
        """Process the get risk metrics request."""
        # Get portfolio
        portfolio_repo = self.unit_of_work.portfolios
        portfolio = await portfolio_repo.get_portfolio_by_id(request.portfolio_id)

        if not portfolio:
            return GetRiskMetricsResponse(
                success=False, error="Portfolio not found", request_id=request.request_id or uuid4()
            )

        # Calculate current metrics
        portfolio_value = portfolio.get_total_value()
        positions_value = portfolio.get_positions_value()
        cash_balance = portfolio.cash_balance

        metrics = {
            "portfolio_value": float(ValueObjectConverter.extract_amount(portfolio_value)),
            "positions_value": float(ValueObjectConverter.extract_amount(positions_value)),
            "cash_balance": float(ValueObjectConverter.extract_amount(cash_balance)),
            "leverage": (
                float(
                    ValueObjectConverter.extract_amount(portfolio_value)
                    / ValueObjectConverter.extract_amount(cash_balance)
                )
                if ValueObjectConverter.extract_amount(cash_balance) > 0
                else 0
            ),
            "position_count": len(portfolio.get_open_positions()),
            "concentration": (
                float(
                    ValueObjectConverter.extract_amount(positions_value)
                    / ValueObjectConverter.extract_amount(portfolio_value)
                )
                if ValueObjectConverter.extract_amount(portfolio_value) > 0
                else 0
            ),
            "unrealized_pnl": float(
                ValueObjectConverter.extract_amount(portfolio.get_unrealized_pnl())
            ),
            "realized_pnl": float(
                ValueObjectConverter.extract_amount(portfolio.total_realized_pnl)
            ),
            "total_return_pct": float(
                ValueObjectConverter.extract_amount(portfolio.get_total_return()) * 100
            ),
        }

        return GetRiskMetricsResponse(success=True, metrics=metrics, request_id=request.request_id)

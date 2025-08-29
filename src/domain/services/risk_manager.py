"""Risk Manager domain service as coordinating facade for risk calculations.

This module provides the RiskManager service which coordinates between all specialized
risk calculation services. It implements the Facade pattern to provide a unified
interface for risk calculations while delegating to specialized services.

This service maintains all existing functionality while providing better separation
of concerns and improved maintainability through delegation to focused calculators.
"""

# Standard library imports
from decimal import Decimal

from ..entities import Order, Portfolio, Position
from ..value_objects import Money, Price
from .risk import (
    PerformanceCalculator,
    PortfolioVaRCalculator,
    PositionRiskCalculator,
    PositionSizingCalculator,
    RiskLimitValidator,
)


class RiskManager:
    """Coordinating facade for risk calculation services.

    This service provides a unified interface for all risk calculations by
    delegating to specialized risk calculation services. It implements the
    Facade pattern to maintain backward compatibility while providing better
    separation of concerns.

    The RiskManager coordinates between:
    - PositionRiskCalculator: Individual position risk analysis
    - PortfolioVaRCalculator: Portfolio VaR and drawdown calculations
    - PerformanceCalculator: Risk-adjusted performance metrics
    - PositionSizingCalculator: Kelly criterion and position sizing
    - RiskLimitValidator: Risk limit validation and business rules

    This service is stateless and thread-safe, with all methods operating as pure
    functions by delegating to the appropriate specialized services.
    """

    def __init__(self) -> None:
        """Initialize the RiskManager with specialized calculators."""
        self._position_risk_calculator = PositionRiskCalculator()
        self._portfolio_var_calculator = PortfolioVaRCalculator()
        self._performance_calculator = PerformanceCalculator()
        self._position_sizing_calculator = PositionSizingCalculator()
        self._risk_limit_validator = RiskLimitValidator()

    def calculate_position_risk(
        self, position: Position, current_price: Price
    ) -> dict[str, Money | Decimal | None]:
        """Calculate comprehensive risk metrics for a single position.

        Delegates to PositionRiskCalculator for position-level risk analysis.

        Args:
            position: Position to analyze. Can be open or closed.
            current_price: Current market price for the position's symbol.

        Returns:
            dict[str, Money | Decimal | None]: Dictionary containing position risk metrics.
        """
        return self._position_risk_calculator.calculate_position_risk(position, current_price)

    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Money:
        """Calculate Value at Risk (VaR) for portfolio.

        Delegates to PortfolioVaRCalculator for portfolio-level VaR calculations.

        Args:
            portfolio: Portfolio to analyze.
            confidence_level: Confidence level as decimal (e.g., 0.95 for 95%).
            time_horizon: Time horizon in trading days.

        Returns:
            Money: VaR amount in USD.
        """
        return self._portfolio_var_calculator.calculate_portfolio_var(
            portfolio, confidence_level, time_horizon
        )

    def calculate_max_drawdown(self, portfolio_history: list[Money]) -> Decimal:
        """Calculate maximum drawdown from portfolio value history.

        Delegates to PortfolioVaRCalculator for drawdown analysis.

        Args:
            portfolio_history: List of portfolio values over time, ordered chronologically.

        Returns:
            Decimal: Maximum drawdown as a percentage (0-100).
        """
        return self._portfolio_var_calculator.calculate_max_drawdown(portfolio_history)

    def calculate_sharpe_ratio(
        self, returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal | None:
        """Calculate Sharpe ratio from returns.

        Delegates to PerformanceCalculator for Sharpe ratio calculations.

        Args:
            returns: List of period returns as decimals.
            risk_free_rate: Annual risk-free rate as decimal.

        Returns:
            Decimal | None: Annualized Sharpe ratio or None if insufficient data.
        """
        return self._performance_calculator.calculate_sharpe_ratio(returns, risk_free_rate)

    def calculate_risk_adjusted_return(
        self, portfolio: Portfolio, time_period_days: int = 30
    ) -> dict[str, Decimal | Money | None]:
        """Calculate comprehensive risk-adjusted return metrics.

        Delegates to PerformanceCalculator for comprehensive risk-adjusted analysis.

        Args:
            portfolio: Portfolio to analyze.
            time_period_days: Period for calculations.

        Returns:
            dict[str, Decimal | Money | None]: Dictionary containing risk-adjusted metrics.
        """
        return self._performance_calculator.calculate_risk_adjusted_return(
            portfolio, time_period_days
        )

    def calculate_kelly_criterion(
        self, win_probability: Decimal, win_amount: Money, loss_amount: Money
    ) -> Decimal:
        """Calculate optimal position size using Kelly Criterion.

        Delegates to PositionSizingCalculator for position sizing calculations.

        Args:
            win_probability: Probability of winning as decimal (0-1).
            win_amount: Average win amount.
            loss_amount: Average loss amount.

        Returns:
            Decimal: Optimal fraction of capital to risk (0-0.25).
        """
        return self._position_sizing_calculator.calculate_kelly_criterion(
            win_probability, win_amount, loss_amount
        )

    def calculate_position_risk_reward(
        self, entry_price: Price, stop_loss: Price, take_profit: Price
    ) -> Decimal:
        """Calculate risk/reward ratio for a position.

        Delegates to PositionRiskCalculator for risk/reward analysis.

        Args:
            entry_price: Planned entry price for the position.
            stop_loss: Stop loss price (risk level).
            take_profit: Take profit target price (reward level).

        Returns:
            Decimal: Risk/reward ratio.
        """
        return self._position_risk_calculator.calculate_position_risk_reward(
            entry_price, stop_loss, take_profit
        )

    def check_risk_limits(self, portfolio: Portfolio, new_order: Order) -> tuple[bool, str]:
        """Check if a new order violates portfolio risk limits.

        Delegates to RiskLimitValidator for risk limit validation.

        Args:
            portfolio: Current portfolio state against which to check limits.
            new_order: Proposed new order to validate.

        Returns:
            tuple[bool, str]: Tuple of (within_limits, reason).
        """
        return self._risk_limit_validator.check_risk_limits(portfolio, new_order)

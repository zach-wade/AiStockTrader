"""Risk Calculator domain service for portfolio risk metrics.

This module provides the RiskCalculator service which maintains backward compatibility
while delegating to specialized risk calculation services. The service has been
refactored to follow Single Responsibility Principle by delegating to focused
calculators while preserving all existing functionality.

The RiskCalculator now acts as a facade to the specialized services:
    - PositionRiskCalculator: Position-level risk assessment
    - PortfolioVaRCalculator: Portfolio VaR and maximum drawdown
    - PerformanceCalculator: Sharpe ratio and risk-adjusted returns
    - PositionSizingCalculator: Kelly criterion and position sizing
    - RiskLimitValidator: Risk limit validation for orders

This refactoring improves maintainability, testability, and follows SOLID principles
while maintaining complete backward compatibility.

Example:
    >>> from decimal import Decimal
    >>> from domain.services import RiskCalculator
    >>> from domain.entities import Portfolio, Position
    >>> from domain.value_objects import Price
    >>>
    >>> calculator = RiskCalculator()
    >>> portfolio = Portfolio(cash_balance=Decimal("100000"))
    >>> var = calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0.95"))
    >>> print(f"95% VaR: ${var.amount:.2f}")

Note:
    This service now delegates to specialized calculators. For new code, consider
    using the RiskManager service or the specific calculators directly for better
    separation of concerns.
"""

# Standard library imports
from decimal import Decimal

from ..entities import Order, Portfolio, Position
from ..value_objects import Money, Price
from .risk_manager import RiskManager


class RiskCalculator:
    """Domain service for calculating position and portfolio risk metrics.

    The RiskCalculator maintains backward compatibility while delegating to specialized
    risk calculation services. This refactored service follows the Facade pattern
    to provide a unified interface while improving separation of concerns.

    All existing functionality is preserved through delegation to specialized services:
    - Position risk calculations via PositionRiskCalculator
    - Portfolio VaR and drawdown via PortfolioVaRCalculator
    - Performance metrics via PerformanceCalculator
    - Position sizing via PositionSizingCalculator
    - Risk limit validation via RiskLimitValidator

    This service is stateless and thread-safe, with all methods operating as pure
    functions by delegating to the appropriate specialized services.

    Note:
        This service maintains complete backward compatibility. For new code, consider
        using RiskManager or the specific specialized calculators directly.
    """

    def __init__(self) -> None:
        """Initialize the RiskCalculator with the RiskManager for delegation."""
        self._risk_manager = RiskManager()

    def calculate_position_risk(
        self, position: Position, current_price: Price
    ) -> dict[str, Money | Decimal | None]:
        """Calculate comprehensive risk metrics for a single position.

        Delegates to PositionRiskCalculator for position-level risk analysis.

        Args:
            position: Position to analyze. Can be open or closed.
            current_price: Current market price for the position's symbol.

        Returns:
            dict[str, Money | Decimal]: Dictionary containing position risk metrics.
        """
        return self._risk_manager.calculate_position_risk(position, current_price)

    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Money:
        """Calculate Value at Risk (VaR) for portfolio.

        Delegates to PortfolioVaRCalculator for portfolio VaR calculations.

        Args:
            portfolio: Portfolio to analyze.
            confidence_level: Confidence level as decimal (e.g., 0.95 for 95%).
            time_horizon: Time horizon in trading days.

        Returns:
            Money: VaR amount in USD.
        """
        return self._risk_manager.calculate_portfolio_var(portfolio, confidence_level, time_horizon)

    def calculate_max_drawdown(self, portfolio_history: list[Money]) -> Decimal:
        """Calculate maximum drawdown from portfolio value history.

        Delegates to PortfolioVaRCalculator for drawdown calculations.

        Args:
            portfolio_history: List of portfolio values over time, ordered chronologically.

        Returns:
            Decimal: Maximum drawdown as a percentage (0-100).
        """
        return self._risk_manager.calculate_max_drawdown(portfolio_history)

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
        return self._risk_manager.calculate_sharpe_ratio(returns, risk_free_rate)

    def check_risk_limits(self, portfolio: Portfolio, new_order: Order) -> tuple[bool, str]:
        """Check if a new order violates portfolio risk limits.

        Delegates to RiskLimitValidator for risk limit validation.

        Args:
            portfolio: Current portfolio state against which to check limits.
            new_order: Proposed new order to validate.

        Returns:
            tuple[bool, str]: Tuple of (within_limits, reason).
        """
        return self._risk_manager.check_risk_limits(portfolio, new_order)

    def calculate_position_risk_reward(
        self, entry_price: Price, stop_loss: Price, take_profit: Price
    ) -> Decimal:
        """Calculate risk/reward ratio for a position.

        Delegates to PositionRiskCalculator for risk/reward calculations.

        Args:
            entry_price: Planned entry price for the position.
            stop_loss: Stop loss price (risk level).
            take_profit: Take profit target price (reward level).

        Returns:
            Decimal: Risk/reward ratio.
        """
        return self._risk_manager.calculate_position_risk_reward(
            entry_price, stop_loss, take_profit
        )

    def calculate_kelly_criterion(
        self, win_probability: Decimal, win_amount: Money, loss_amount: Money
    ) -> Decimal:
        """Calculate optimal position size using Kelly Criterion.

        Delegates to PositionSizingCalculator for Kelly criterion calculations.

        Args:
            win_probability: Probability of winning as decimal (0-1).
            win_amount: Average win amount.
            loss_amount: Average loss amount.

        Returns:
            Decimal: Optimal fraction of capital to risk (0-0.25).
        """
        return self._risk_manager.calculate_kelly_criterion(
            win_probability, win_amount, loss_amount
        )

    def calculate_risk_adjusted_return(
        self, portfolio: Portfolio, time_period_days: int = 30
    ) -> dict[str, Decimal | Money | None]:
        """Calculate comprehensive risk-adjusted return metrics.

        Delegates to PerformanceCalculator for risk-adjusted return analysis.

        Args:
            portfolio: Portfolio to analyze.
            time_period_days: Period for calculations.

        Returns:
            dict[str, Decimal | Money | None]: Dictionary containing risk-adjusted metrics.
        """
        return self._risk_manager.calculate_risk_adjusted_return(portfolio, time_period_days)

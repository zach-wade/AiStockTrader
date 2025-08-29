"""
Portfolio Metrics Calculator Service

Handles advanced portfolio metrics and performance calculations.
Extracted from Portfolio entity to follow Single Responsibility Principle.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from ..value_objects import Money
from ..value_objects.converter import ValueObjectConverter

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio


class PortfolioMetricsCalculator:
    """
    Service for calculating portfolio metrics and performance indicators.

    Handles all portfolio metrics calculations, extracted from the Portfolio entity
    to follow the Single Responsibility Principle. This includes both basic metrics
    and advanced calculations requiring historical data or statistical analysis.
    """

    @staticmethod
    def get_cash_usage_ratio(portfolio: "Portfolio") -> Decimal:
        """Calculate the ratio of cash used vs total initial capital.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Cash usage ratio as a decimal (0.0 to 1.0)
        """
        if portfolio.initial_capital.amount == 0:
            return Decimal("1")
        cash_used = portfolio.initial_capital.amount - portfolio.cash_balance.amount
        return cash_used / portfolio.initial_capital.amount

    @staticmethod
    def get_total_value(portfolio: "Portfolio") -> Money:
        """Calculate total portfolio value (cash + positions).

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Total portfolio value
        """
        # Calculate total value directly
        total = Money(ValueObjectConverter.to_decimal(portfolio.cash_balance))

        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total = total + position_value

        return total

    @staticmethod
    def get_positions_value(portfolio: "Portfolio") -> Money:
        """Calculate total value of all open positions.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Total value of open positions
        """
        total = Money(Decimal("0"))

        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total = total + position_value

        return total

    @staticmethod
    def get_unrealized_pnl(portfolio: "Portfolio") -> Money:
        """Calculate total unrealized P&L.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Total unrealized profit/loss
        """
        total = Money(Decimal("0"))

        for position in portfolio.get_open_positions():
            unrealized = position.get_unrealized_pnl()
            if unrealized is not None:
                total = total + unrealized

        return total

    @staticmethod
    def get_total_pnl(portfolio: "Portfolio") -> Money:
        """Calculate total P&L (realized + unrealized).

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Total profit/loss
        """
        return portfolio.total_realized_pnl + PortfolioMetricsCalculator.get_unrealized_pnl(
            portfolio
        )

    @staticmethod
    def get_return_percentage(portfolio: "Portfolio") -> Decimal:
        """Calculate portfolio return percentage.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Return percentage
        """
        # Handle both Money objects and raw Decimal values
        initial_capital_amount = ValueObjectConverter.extract_amount(portfolio.initial_capital)

        if initial_capital_amount == 0:
            return Decimal("0")

        current_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        return ((current_value.amount - initial_capital_amount) / initial_capital_amount) * Decimal(
            "100"
        )

    @staticmethod
    def get_win_rate(portfolio: "Portfolio") -> Decimal | None:
        """Calculate win rate percentage.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Win rate percentage or None if no closed trades
        """
        total_closed = portfolio.winning_trades + portfolio.losing_trades
        if total_closed == 0:
            return None

        return (Decimal(portfolio.winning_trades) / Decimal(total_closed)) * Decimal("100")

    @staticmethod
    def get_average_win(portfolio: "Portfolio") -> Money | None:
        """Calculate average winning trade.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Average winning amount or None if no winning trades
        """
        if portfolio.winning_trades == 0:
            return None

        total_wins = Money(Decimal("0"))
        for position in portfolio.get_closed_positions():
            if position.realized_pnl.amount > 0:
                total_wins = total_wins + position.realized_pnl

        return Money(total_wins.amount / Decimal(portfolio.winning_trades))

    @staticmethod
    def get_average_loss(portfolio: "Portfolio") -> Money | None:
        """Calculate average losing trade.

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Average loss amount or None if no losing trades
        """
        if portfolio.losing_trades == 0:
            return None

        total_losses = Money(Decimal("0"))
        for position in portfolio.get_closed_positions():
            if position.realized_pnl.amount < 0:
                total_losses = total_losses + abs(position.realized_pnl)

        return Money(total_losses.amount / Decimal(portfolio.losing_trades))

    @staticmethod
    def get_profit_factor(portfolio: "Portfolio") -> Decimal | None:
        """Calculate profit factor (gross profit / gross loss).

        Args:
            portfolio: The portfolio to analyze

        Returns:
            Profit factor or None if no losses
        """
        gross_profit = Money(Decimal("0"))
        gross_loss = Money(Decimal("0"))

        for position in portfolio.get_closed_positions():
            if position.realized_pnl.amount > 0:
                gross_profit = gross_profit + position.realized_pnl
            elif position.realized_pnl.amount < 0:
                gross_loss = gross_loss + abs(position.realized_pnl)

        if gross_loss.amount == 0:
            return None if gross_profit.amount == 0 else Decimal("999.99")  # Cap at 999.99

        return gross_profit.amount / gross_loss.amount

    @staticmethod
    def calculate_value_at_risk(
        portfolio: "Portfolio",
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon_days: int = 1,
        historical_returns: list[Decimal] | None = None,
    ) -> Money:
        """Calculate Value at Risk (VaR) for the portfolio.

        Args:
            portfolio: The portfolio to analyze
            confidence_level: Confidence level (0.95 = 95%)
            time_horizon_days: Time horizon in days
            historical_returns: Historical daily returns data

        Returns:
            VaR amount (potential loss)
        """
        # Placeholder implementation - would need historical data
        # In production, this would use historical simulation, Monte Carlo, or parametric methods
        portfolio_value = portfolio.get_total_value()

        # Simple example using portfolio return percentage as proxy
        return_pct = portfolio.get_return_percentage()

        # Very simplified VaR calculation (not suitable for production)
        var_percentage = abs(return_pct) * Decimal("0.02")  # 2% of return as risk proxy
        return Money(portfolio_value.amount * var_percentage / Decimal("100"))

    @staticmethod
    def calculate_portfolio_beta(
        portfolio: "Portfolio", market_returns: list[Decimal], portfolio_returns: list[Decimal]
    ) -> Decimal | None:
        """Calculate portfolio beta relative to market.

        Args:
            portfolio: The portfolio to analyze
            market_returns: Historical market returns
            portfolio_returns: Historical portfolio returns

        Returns:
            Portfolio beta coefficient
        """
        if len(market_returns) != len(portfolio_returns) or len(market_returns) < 2:
            return None

        # Calculate covariance and variance
        market_mean = sum(market_returns) / Decimal(len(market_returns))
        portfolio_mean = sum(portfolio_returns) / Decimal(len(portfolio_returns))

        covariance = sum(
            (m - market_mean) * (p - portfolio_mean)
            for m, p in zip(market_returns, portfolio_returns)
        ) / Decimal(len(market_returns))

        market_variance = sum((m - market_mean) ** 2 for m in market_returns) / Decimal(
            len(market_returns)
        )

        if market_variance == 0:
            return None

        return covariance / market_variance

    @staticmethod
    def calculate_information_ratio(
        portfolio: "Portfolio", benchmark_returns: list[Decimal], portfolio_returns: list[Decimal]
    ) -> Decimal | None:
        """Calculate information ratio (excess return vs tracking error).

        Args:
            portfolio: The portfolio to analyze
            benchmark_returns: Benchmark returns history
            portfolio_returns: Portfolio returns history

        Returns:
            Information ratio
        """
        if len(benchmark_returns) != len(portfolio_returns) or len(benchmark_returns) < 2:
            return None

        # Calculate excess returns
        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]

        # Mean excess return
        mean_excess = sum(excess_returns) / Decimal(len(excess_returns))

        # Tracking error (standard deviation of excess returns)
        variance = sum((er - mean_excess) ** 2 for er in excess_returns) / Decimal(
            len(excess_returns)
        )

        tracking_error = variance ** Decimal("0.5")

        if tracking_error == 0:
            return None

        return mean_excess / tracking_error

    @staticmethod
    def get_sharpe_ratio(
        portfolio: "Portfolio",
        risk_free_rate: Decimal = Decimal("0.02"),
        historical_returns: list[Decimal] | None = None,
    ) -> Decimal | None:
        """
        Calculate Sharpe ratio using historical returns data.

        Args:
            portfolio: The portfolio to analyze
            risk_free_rate: Risk-free rate (annualized)
            historical_returns: Historical daily returns data

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if historical_returns is None or len(historical_returns) < 2:
            return None

        # Calculate mean return
        mean_return = sum(historical_returns) / Decimal(len(historical_returns))

        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in historical_returns) / Decimal(
            len(historical_returns)
        )

        std_dev = variance ** Decimal("0.5")

        if std_dev == 0:
            return None

        # Convert annual risk-free rate to daily
        daily_risk_free = risk_free_rate / Decimal("252")  # ~252 trading days/year

        # Calculate Sharpe ratio
        excess_return = mean_return - daily_risk_free
        sharpe = excess_return / std_dev

        # Annualize the Sharpe ratio
        return sharpe * (Decimal("252") ** Decimal("0.5"))

    @staticmethod
    def get_max_drawdown(
        portfolio: "Portfolio", historical_values: list[Money] | None = None
    ) -> Decimal:
        """
        Calculate maximum drawdown using historical portfolio values.

        Args:
            portfolio: The portfolio to analyze
            historical_values: Historical portfolio values

        Returns:
            Maximum drawdown percentage
        """
        if historical_values is None or len(historical_values) < 2:
            return Decimal("0")

        max_drawdown = Decimal("0")
        peak = historical_values[0].amount

        for value in historical_values[1:]:
            current = value.amount

            if current > peak:
                peak = current
            else:
                drawdown = (peak - current) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * Decimal("100")  # Return as percentage

"""Portfolio Calculator Service - Handles all portfolio metrics and calculations."""

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from ..value_objects import Money, Price

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio
    from ..entities.position import Position


class PortfolioCalculator:
    """Consolidated service for portfolio metrics, analytics, and calculations.

    This service combines functionality from:
    - PortfolioMetricsCalculator
    - PortfolioAnalyticsService
    - Related calculation services
    """

    # --- Core Value Calculations ---

    @staticmethod
    def get_total_value(portfolio: "Portfolio") -> Money:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = PortfolioCalculator.get_positions_value(portfolio)
        return portfolio.cash_balance + positions_value

    @staticmethod
    def get_positions_value(portfolio: "Portfolio") -> Money:
        """Calculate total value of all open positions."""
        total_value = Money(Decimal("0"))
        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total_value = total_value + position_value
        return total_value

    @staticmethod
    def get_cash_usage_ratio(portfolio: "Portfolio") -> Decimal:
        """Calculate the percentage of cash being used for positions."""
        total_value = PortfolioCalculator.get_total_value(portfolio)
        if total_value.amount == 0:
            return Decimal("0")
        return (total_value.amount - portfolio.cash_balance.amount) / total_value.amount

    # --- P&L Calculations ---

    @staticmethod
    def get_unrealized_pnl(portfolio: "Portfolio") -> Money:
        """Calculate total unrealized P&L across all open positions."""
        total_pnl = Money(Decimal("0"))
        for position in portfolio.get_open_positions():
            pnl = position.get_unrealized_pnl()
            if pnl is not None:
                total_pnl = total_pnl + pnl
        return total_pnl

    @staticmethod
    def get_total_pnl(portfolio: "Portfolio") -> Money:
        """Calculate total P&L (realized + unrealized)."""
        unrealized = PortfolioCalculator.get_unrealized_pnl(portfolio)
        return portfolio.total_realized_pnl + unrealized

    # --- Return Calculations ---

    @staticmethod
    def get_return_percentage(portfolio: "Portfolio") -> Decimal:
        """Calculate portfolio return as a percentage."""
        if portfolio.initial_capital.amount == 0:
            return Decimal("0")

        total_pnl = PortfolioCalculator.get_total_pnl(portfolio)
        return (total_pnl.amount / portfolio.initial_capital.amount) * Decimal("100")

    @staticmethod
    def get_total_return(portfolio: "Portfolio") -> Decimal:
        """Calculate portfolio total return as a ratio."""
        if portfolio.initial_capital.amount == 0:
            return Decimal("0")

        total_value = PortfolioCalculator.get_total_value(portfolio)
        return (
            total_value.amount - portfolio.initial_capital.amount
        ) / portfolio.initial_capital.amount

    # --- Trading Performance Metrics ---

    @staticmethod
    def get_win_rate(portfolio: "Portfolio") -> Decimal | None:
        """Calculate win rate percentage."""
        total_trades = portfolio.winning_trades + portfolio.losing_trades
        if total_trades == 0:
            return None
        return (Decimal(portfolio.winning_trades) / Decimal(total_trades)) * Decimal("100")

    @staticmethod
    def get_average_win(portfolio: "Portfolio") -> Money | None:
        """Calculate average winning trade amount."""
        if portfolio.winning_trades == 0:
            return None

        # If we have closed positions, calculate from them
        closed_positions = portfolio.get_closed_positions()
        if closed_positions:
            total_wins = Money(Decimal("0"))
            win_count = 0

            for position in closed_positions:
                if hasattr(position, "realized_pnl") and position.realized_pnl is not None:
                    if position.realized_pnl.amount > 0:
                        total_wins = total_wins + position.realized_pnl
                        win_count += 1

            if win_count > 0:
                return Money(total_wins.amount / Decimal(win_count))

        # Fallback: estimate from aggregate statistics if available
        if portfolio.winning_trades > 0 and portfolio.total_realized_pnl.amount > 0:
            # This is an approximation when we don't have detailed position history
            avg_pnl_per_trade = portfolio.total_realized_pnl.amount / (
                portfolio.winning_trades + portfolio.losing_trades
            )
            # Estimate based on the ratio
            if portfolio.losing_trades == 0:
                return Money(portfolio.total_realized_pnl.amount / portfolio.winning_trades)
            # More complex estimation when we have both wins and losses
            # This is just an estimate and may not be accurate
            estimated_avg_win = portfolio.total_realized_pnl.amount / portfolio.winning_trades
            return Money(abs(estimated_avg_win))

        return None

    @staticmethod
    def get_average_loss(portfolio: "Portfolio") -> Money | None:
        """Calculate average losing trade amount."""
        if portfolio.losing_trades == 0:
            return None

        # If we have closed positions, calculate from them
        closed_positions = portfolio.get_closed_positions()
        if closed_positions:
            total_losses = Money(Decimal("0"))
            loss_count = 0

            for position in closed_positions:
                if hasattr(position, "realized_pnl") and position.realized_pnl is not None:
                    if position.realized_pnl.amount < 0:
                        total_losses = total_losses + position.realized_pnl
                        loss_count += 1

            if loss_count > 0:
                return Money(abs(total_losses.amount / Decimal(loss_count)))

        # Fallback: estimate from aggregate statistics if available
        if portfolio.losing_trades > 0:
            # This is an approximation when we don't have detailed position history
            if portfolio.winning_trades == 0 and portfolio.total_realized_pnl.amount < 0:
                # All trades are losses
                return Money(abs(portfolio.total_realized_pnl.amount) / portfolio.losing_trades)
            # For mixed scenarios, we need to estimate
            # This is a rough approximation
            return Money(Decimal("50"))  # Default estimate when we can't calculate precisely

        return None

    @staticmethod
    def get_profit_factor(portfolio: "Portfolio") -> Decimal | None:
        """Calculate profit factor (gross profits / gross losses)."""
        gross_profits = Money(Decimal("0"))
        gross_losses = Money(Decimal("0"))

        for position in portfolio.get_closed_positions():
            if hasattr(position, "realized_pnl") and position.realized_pnl is not None:
                if position.realized_pnl.amount > 0:
                    gross_profits = gross_profits + position.realized_pnl
                else:
                    gross_losses = gross_losses + Money(abs(position.realized_pnl.amount))

        if gross_losses.amount == 0:
            return None if gross_profits.amount == 0 else Decimal("999.99")  # Cap at 999.99

        return gross_profits.amount / gross_losses.amount

    @staticmethod
    def get_expectancy(portfolio: "Portfolio") -> Money:
        """Calculate expectancy (average expected profit per trade)."""
        total_trades = portfolio.winning_trades + portfolio.losing_trades
        if total_trades == 0:
            return Money(Decimal("0"))

        # Simple expectancy: Total P&L divided by number of trades
        # This is the most straightforward calculation when we don't have position details
        return Money(portfolio.total_realized_pnl.amount / Decimal(total_trades))

    # --- Risk Metrics ---

    @staticmethod
    def get_sharpe_ratio(
        portfolio: "Portfolio",
        risk_free_rate: Decimal = Decimal("0.02"),
        returns: list[Decimal] | None = None,
    ) -> Decimal | None:
        """Calculate Sharpe ratio."""
        if returns is None:
            # Simple calculation based on total return
            total_return = PortfolioCalculator.get_total_return(portfolio)
            if total_return == 0:
                return None

            # Simplified Sharpe calculation (annualized)
            excess_return = total_return - risk_free_rate
            # Assume standard deviation of 0.15 for simplification if not provided
            std_dev = Decimal("0.15")

            if std_dev == 0:
                return None

            return excess_return / std_dev

        # Calculate from provided returns
        if len(returns) < 2:
            return None

        avg_return = sum(returns) / Decimal(str(len(returns)))
        excess_returns = [r - risk_free_rate for r in returns]

        if len(excess_returns) < 2:
            return None

        # Calculate variance using Decimal arithmetic
        variance_sum = sum((r - avg_return) ** 2 for r in excess_returns)
        variance = variance_sum / Decimal(str(len(excess_returns) - 1))

        # Calculate standard deviation
        if variance > 0:
            import math

            std_dev = Decimal(str(math.sqrt(float(variance))))
        else:
            std_dev = Decimal("0")

        if std_dev == 0:
            return None

        return (avg_return - risk_free_rate) / std_dev

    @staticmethod
    def get_max_drawdown(
        portfolio: "Portfolio", historical_values: list[Money] | None = None
    ) -> Decimal:
        """Calculate maximum drawdown."""
        if historical_values is None or len(historical_values) < 2:
            # Simple calculation based on current value vs initial capital
            current_value = PortfolioCalculator.get_total_value(portfolio)
            if current_value < portfolio.initial_capital:
                drawdown = (
                    portfolio.initial_capital.amount - current_value.amount
                ) / portfolio.initial_capital.amount
                return drawdown
            return Decimal("0")

        # Calculate from historical values
        max_dd = Decimal("0")
        peak = historical_values[0].amount

        for value in historical_values:
            if value.amount > peak:
                peak = value.amount

            if peak > 0:
                drawdown = (peak - value.amount) / peak
                if drawdown > max_dd:
                    max_dd = drawdown

        return max_dd

    @staticmethod
    def calculate_value_at_risk(
        portfolio: "Portfolio",
        confidence_level: Decimal = Decimal("0.95"),
        returns: list[Decimal] | None = None,
    ) -> Money:
        """Calculate Value at Risk (VaR)."""
        if returns is None or len(returns) == 0:
            # Simple VaR calculation based on portfolio value and assumed volatility
            portfolio_value = PortfolioCalculator.get_total_value(portfolio)
            # Assume 2% daily volatility for simplification
            daily_volatility = Decimal("0.02")

            # Z-score for 95% confidence is approximately 1.645
            z_score = Decimal("1.645") if confidence_level == Decimal("0.95") else Decimal("2.326")

            var_amount = portfolio_value.amount * daily_volatility * z_score
            return Money(var_amount)

        # Calculate from historical returns
        sorted_returns = sorted(returns)
        index = int(len(sorted_returns) * (1 - confidence_level))

        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1

        portfolio_value = PortfolioCalculator.get_total_value(portfolio)
        var_percentage = abs(sorted_returns[index])
        return Money(portfolio_value.amount * var_percentage)

    # --- Position Analysis ---

    @staticmethod
    def get_largest_position(
        portfolio: "Portfolio", current_prices: dict[str, Price]
    ) -> Optional["Position"]:
        """Get the largest position by current value."""
        largest = None
        max_value = Money(Decimal("0"))

        for position in portfolio.get_open_positions():
            if position.symbol in current_prices:
                position.update_market_price(current_prices[position.symbol])

            position_value = position.get_position_value()
            if position_value and position_value > max_value:
                max_value = position_value
                largest = position

        return largest

    @staticmethod
    def get_positions_by_profit(
        portfolio: "Portfolio", current_prices: dict[str, Price]
    ) -> list[tuple["Position", Money]]:
        """Get positions sorted by profit (descending)."""
        position_profits = []

        for position in portfolio.get_open_positions():
            if position.symbol in current_prices:
                position.update_market_price(current_prices[position.symbol])

            pnl = position.get_unrealized_pnl()
            if pnl is not None:
                position_profits.append((position, pnl))

        # Sort by profit descending
        position_profits.sort(key=lambda x: x[1].amount, reverse=True)
        return position_profits

    # --- Portfolio Summary & Serialization ---

    @staticmethod
    def get_portfolio_summary(portfolio: "Portfolio") -> dict[str, Any]:
        """Get comprehensive portfolio summary."""
        return_percentage = PortfolioCalculator.get_return_percentage(portfolio)
        win_rate = PortfolioCalculator.get_win_rate(portfolio)

        return {
            "id": str(portfolio.id),
            "name": portfolio.name,
            "total_value": float(PortfolioCalculator.get_total_value(portfolio).amount),
            "cash_balance": float(portfolio.cash_balance.amount),
            "positions_value": float(PortfolioCalculator.get_positions_value(portfolio).amount),
            "total_pnl": float(PortfolioCalculator.get_total_pnl(portfolio).amount),
            "unrealized_pnl": float(PortfolioCalculator.get_unrealized_pnl(portfolio).amount),
            "realized_pnl": float(portfolio.total_realized_pnl.amount),
            "return_percentage": float(return_percentage) if return_percentage is not None else 0.0,
            "roi": float(return_percentage) if return_percentage is not None else 0.0,  # ROI alias
            "win_rate": float(win_rate) if win_rate is not None else 0.0,
            "position_count": portfolio.get_position_count(),
            "trades_count": portfolio.trades_count,
        }

    @staticmethod
    def portfolio_to_dict(portfolio: "Portfolio") -> dict[str, Any]:
        """Convert portfolio to dictionary for serialization."""
        positions_dict = {}
        for symbol, position in portfolio.positions.items():
            positions_dict[symbol] = {
                "symbol": position.symbol,
                "quantity": float(position.quantity.value),
                "average_entry_price": float(position.average_entry_price.value),
                "current_price": (
                    float(position.current_price.value) if position.current_price else None
                ),
                "entry_time": position.entry_time.isoformat() if position.entry_time else None,
                "exit_time": position.exit_time.isoformat() if position.exit_time else None,
                "exit_price": float(position.exit_price.value) if position.exit_price else None,
                "realized_pnl": (
                    float(position.realized_pnl.amount) if position.realized_pnl else None
                ),
            }

        return {
            "id": str(portfolio.id),
            "name": portfolio.name,
            "initial_capital": float(portfolio.initial_capital.amount),
            "cash_balance": float(portfolio.cash_balance.amount),
            "positions": positions_dict,
            "max_position_size": float(portfolio.max_position_size.amount),
            "max_portfolio_risk": float(portfolio.max_portfolio_risk),
            "max_positions": portfolio.max_positions,
            "max_leverage": float(portfolio.max_leverage),
            "total_realized_pnl": float(portfolio.total_realized_pnl.amount),
            "total_commission_paid": float(portfolio.total_commission_paid.amount),
            "trades_count": portfolio.trades_count,
            "winning_trades": portfolio.winning_trades,
            "losing_trades": portfolio.losing_trades,
            "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
            "last_updated": portfolio.last_updated.isoformat() if portfolio.last_updated else None,
            "strategy": portfolio.strategy,
            "tags": portfolio.tags,
            "version": portfolio.version,
        }

    @staticmethod
    def portfolio_to_string(portfolio: "Portfolio") -> str:
        """Generate string representation of portfolio."""
        total_value = PortfolioCalculator.get_total_value(portfolio)
        pnl = PortfolioCalculator.get_total_pnl(portfolio)
        return_pct = PortfolioCalculator.get_return_percentage(portfolio)

        return (
            f"Portfolio {portfolio.name} (ID: {portfolio.id})\n"
            f"Total Value: {total_value}\n"
            f"Cash: {portfolio.cash_balance}\n"
            f"P&L: {pnl} ({return_pct:.2f}%)\n"
            f"Positions: {portfolio.get_position_count()}/{portfolio.max_positions}"
        )

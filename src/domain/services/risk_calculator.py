"""Risk Calculator domain service for portfolio risk metrics."""

# Standard library imports
from decimal import Decimal

from ..constants import MIN_DATA_POINTS_FOR_STATS
from ..entities import Order, Portfolio, Position
from ..value_objects import Money, Price


class RiskCalculator:
    """Domain service for calculating position and portfolio risk metrics."""

    def calculate_position_risk(
        self, position: Position, current_price: Price
    ) -> dict[str, Decimal]:
        """Calculate risk metrics for a single position.

        Args:
            position: Position to analyze
            current_price: Current market price

        Returns:
            Dictionary of risk metrics
        """
        position.update_market_price(current_price.value)

        metrics = {
            "position_value": Decimal("0"),
            "unrealized_pnl": Decimal("0"),
            "realized_pnl": position.realized_pnl,
            "total_pnl": Decimal("0"),
            "return_pct": Decimal("0"),
            "risk_amount": Decimal("0"),
        }

        if position.is_closed():
            metrics["total_pnl"] = position.realized_pnl
        else:
            # Position value
            position_value = position.get_position_value()
            if position_value:
                metrics["position_value"] = position_value

            # P&L
            unrealized = position.get_unrealized_pnl()
            if unrealized:
                metrics["unrealized_pnl"] = unrealized
                metrics["total_pnl"] = position.realized_pnl + unrealized

            # Return percentage
            return_pct = position.get_return_percentage()
            if return_pct:
                metrics["return_pct"] = return_pct

            # Risk amount (distance to stop loss)
            if position.stop_loss_price:
                risk_per_share = abs(current_price.value - position.stop_loss_price)
                metrics["risk_amount"] = risk_per_share * abs(position.quantity)

        return metrics

    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Money:
        """Calculate Value at Risk for portfolio.

        Simplified VaR calculation based on current positions.
        Production implementation would use historical returns.

        Args:
            portfolio: Portfolio to analyze
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR as Money object
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        # Simplified VaR: use portfolio value * assumed volatility * z-score
        portfolio_value = portfolio.get_total_value()

        # Z-scores for common confidence levels
        z_scores = {
            Decimal("0.90"): Decimal("1.28"),
            Decimal("0.95"): Decimal("1.65"),
            Decimal("0.99"): Decimal("2.33"),
        }

        # Get closest z-score
        z_score = z_scores.get(confidence_level, Decimal("1.65"))

        # Assume 2% daily volatility (would calculate from historical data)
        daily_volatility = Decimal("0.02")

        # Calculate VaR
        var = portfolio_value * daily_volatility * z_score * Decimal(time_horizon).sqrt()

        return Money(var, "USD")

    def calculate_max_drawdown(self, portfolio_history: list[Decimal]) -> Decimal:
        """Calculate maximum drawdown from portfolio value history.

        Args:
            portfolio_history: List of portfolio values over time

        Returns:
            Maximum drawdown as decimal percentage
        """
        if not portfolio_history or len(portfolio_history) < MIN_DATA_POINTS_FOR_STATS:
            return Decimal("0")

        max_value = portfolio_history[0]
        max_drawdown = Decimal("0")

        for value in portfolio_history:
            if value > max_value:
                max_value = value

            drawdown = (max_value - value) / max_value if max_value > 0 else Decimal("0")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown * Decimal("100")

    def calculate_sharpe_ratio(
        self, returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal | None:
        """Calculate Sharpe ratio from returns.

        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if not returns or len(returns) < MIN_DATA_POINTS_FOR_STATS:
            return None

        # Calculate average return
        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((Decimal(str(r)) - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance.sqrt() if hasattr(variance, 'sqrt') else Decimal(str(variance ** Decimal('0.5')))

        if std_dev == 0:
            return None

        # Annualize (assuming daily returns)
        annual_return = avg_return * Decimal("252")
        annual_std = std_dev * (Decimal("252") ** Decimal('0.5'))

        # Calculate Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_std

        return sharpe

    def check_risk_limits(self, portfolio: Portfolio, new_order: Order) -> tuple[bool, str]:
        """Check if a new order violates risk limits.

        Args:
            portfolio: Current portfolio
            new_order: Proposed new order

        Returns:
            Tuple of (is_within_limits, reason_if_not)
        """
        # Check position limits
        can_open, reason = portfolio.can_open_position(
            symbol=new_order.symbol,
            quantity=new_order.quantity,
            price=new_order.limit_price or Decimal("100"),  # Use estimate if market order
        )

        if not can_open:
            return False, reason

        # Check leverage
        if portfolio.max_leverage > 1:
            positions_value = portfolio.get_positions_value()
            order_value = new_order.quantity * (new_order.limit_price or Decimal("100"))
            total_exposure = positions_value + order_value

            leverage = (
                total_exposure / portfolio.cash_balance
                if portfolio.cash_balance > 0
                else Decimal("999")
            )

            if leverage > portfolio.max_leverage:
                return (
                    False,
                    f"Order would exceed leverage limit: {leverage:.2f} > {portfolio.max_leverage}",
                )

        # Check concentration
        max_concentration = Decimal("0.20")  # Max 20% in single position
        portfolio_value = portfolio.get_total_value()

        if portfolio_value > 0:
            order_value = new_order.quantity * (new_order.limit_price or Decimal("100"))
            concentration = order_value / portfolio_value

            if concentration > max_concentration:
                return (
                    False,
                    f"Order would exceed concentration limit: {concentration:.1%} > {max_concentration:.1%}",
                )

        return True, ""

    def calculate_position_risk_reward(
        self, entry_price: Price, stop_loss: Price, take_profit: Price
    ) -> Decimal:
        """Calculate risk/reward ratio for a position.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Risk/reward ratio

        Raises:
            ValueError: If prices are invalid
        """
        risk = abs(entry_price.value - stop_loss.value)
        reward = abs(take_profit.value - entry_price.value)

        if risk == 0:
            raise ValueError("Risk cannot be zero")

        return reward / risk

    def calculate_kelly_criterion(
        self, win_probability: Decimal, win_amount: Decimal, loss_amount: Decimal
    ) -> Decimal:
        """Calculate optimal position size using Kelly Criterion.

        Args:
            win_probability: Probability of winning (0-1)
            win_amount: Average win amount
            loss_amount: Average loss amount (positive)

        Returns:
            Optimal fraction of capital to risk
        """
        if win_probability <= 0 or win_probability >= 1:
            raise ValueError("Win probability must be between 0 and 1")

        if win_amount <= 0 or loss_amount <= 0:
            raise ValueError("Win and loss amounts must be positive")

        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        loss_probability = 1 - win_probability
        win_loss_ratio = win_amount / loss_amount

        kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

        # Cap at 25% for safety (full Kelly can be too aggressive)
        return min(kelly_fraction, Decimal("0.25"))

    def calculate_risk_adjusted_return(
        self, portfolio: Portfolio, _time_period_days: int = 30
    ) -> dict[str, Decimal | None]:
        """Calculate risk-adjusted return metrics.

        Args:
            portfolio: Portfolio to analyze
            time_period_days: Period for calculations

        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {
            "total_return": portfolio.get_return_percentage(),
            "win_rate": portfolio.get_win_rate(),
            "profit_factor": portfolio.get_profit_factor(),
            "average_win": portfolio.get_average_win(),
            "average_loss": portfolio.get_average_loss(),
            "max_drawdown": self.calculate_max_drawdown([portfolio.get_total_value()]),
            "sharpe_ratio": portfolio.get_sharpe_ratio(),
        }

        # Calculate risk-adjusted return
        if metrics["total_return"] and metrics["max_drawdown"]:
            if metrics["max_drawdown"] > 0:
                metrics["calmar_ratio"] = metrics["total_return"] / metrics["max_drawdown"]
            else:
                metrics["calmar_ratio"] = None

        # Calculate expectancy
        if metrics["win_rate"] and metrics["average_win"] and metrics["average_loss"]:
            win_rate = metrics["win_rate"] / Decimal("100")
            loss_rate = 1 - win_rate
            expectancy = (win_rate * metrics["average_win"]) - (loss_rate * metrics["average_loss"])
            metrics["expectancy"] = expectancy

        return metrics

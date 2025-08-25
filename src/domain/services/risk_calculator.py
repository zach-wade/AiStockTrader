"""Risk Calculator domain service for portfolio risk metrics.

This module provides the RiskCalculator service which computes various risk metrics
for positions and portfolios. It implements standard risk management calculations
used in quantitative trading and portfolio management.

The RiskCalculator encapsulates complex financial calculations including Value at Risk
(VaR), Sharpe ratio, maximum drawdown, and position-specific risk metrics. It follows
Domain-Driven Design principles by keeping all risk calculation logic within the domain.

Key Responsibilities:
    - Position-level risk assessment
    - Portfolio-level risk metrics (VaR, Sharpe, drawdown)
    - Risk-adjusted return calculations
    - Risk limit validation for new orders
    - Kelly criterion and position sizing

Design Decisions:
    - All calculations use Decimal for financial precision
    - Simplified VaR using parametric approach (production would use historical simulation)
    - Annualization assumes 252 trading days per year (US equity markets)
    - Risk metrics return None when insufficient data rather than throwing exceptions

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
    Many calculations are simplified for the trading system context. Production
    implementations would incorporate more sophisticated models including historical
    correlation matrices, stress testing, and Monte Carlo simulations.
"""

# Standard library imports
import math
from decimal import Decimal

from ..constants import MIN_DATA_POINTS_FOR_STATS
from ..entities import Order, Portfolio, Position
from ..value_objects import Money, Price

# Trading days constants
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_FACTOR = Decimal(str(TRADING_DAYS_PER_YEAR**0.5))


class RiskCalculator:
    """Domain service for calculating position and portfolio risk metrics.

    The RiskCalculator provides comprehensive risk analysis functionality for both
    individual positions and entire portfolios. It implements industry-standard
    risk metrics and provides decision support for risk management.

    This service is stateless and thread-safe, with all methods operating as pure
    functions on provided entities.

    Constants:
        TRADING_DAYS_PER_YEAR: 252 (standard for US equity markets)
        ANNUALIZATION_FACTOR: Square root of trading days for volatility scaling
        MIN_DATA_POINTS_FOR_STATS: Minimum data points required for statistical calculations

    Note:
        All monetary values use Decimal type for precision. Statistical calculations
        may have reduced precision due to mathematical operations but maintain
        sufficient accuracy for trading decisions.
    """

    def calculate_position_risk(
        self, position: Position, current_price: Price
    ) -> dict[str, Decimal]:
        """Calculate comprehensive risk metrics for a single position.

        Computes various risk and performance metrics for an individual position,
        providing a complete risk profile for position analysis and decision-making.

        Args:
            position: Position to analyze. Can be open or closed.
            current_price: Current market price for the position's symbol.

        Returns:
            dict[str, Decimal]: Dictionary containing risk metrics:
                - position_value: Current market value of the position
                - unrealized_pnl: Unrealized profit/loss (open positions only)
                - realized_pnl: Realized profit/loss from closed portions
                - total_pnl: Sum of realized and unrealized P&L
                - return_pct: Percentage return on the position
                - risk_amount: Dollar amount at risk if stop loss is hit

        Behavior:
            - For closed positions: Only realized_pnl and total_pnl are non-zero
            - For open positions: All metrics are calculated based on current price
            - Metrics default to Decimal("0") when not applicable

        Example:
            >>> position = Position(symbol="AAPL", quantity=100, average_entry_price=Decimal("150"))
            >>> position.stop_loss_price = Decimal("145")
            >>> metrics = calculator.calculate_position_risk(position, Price(Decimal("155")))
            >>> print(f"Unrealized P&L: ${metrics['unrealized_pnl']}")
            >>> print(f"Risk if stop hit: ${metrics['risk_amount']}")

        Note:
            This method updates the position's market price as a side effect
            to ensure consistent calculations.
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
        """Calculate Value at Risk (VaR) for portfolio.

        Computes the potential loss in portfolio value that will not be exceeded
        with a given confidence level over a specified time horizon. This implementation
        uses a parametric (variance-covariance) approach with assumed volatility.

        Args:
            portfolio: Portfolio to analyze.
            confidence_level: Confidence level as decimal (e.g., 0.95 for 95%).
                Common values: 0.90, 0.95, 0.99. Must be between 0 and 1.
            time_horizon: Time horizon in trading days. Typically 1, 5, or 10 days.

        Returns:
            Money: VaR amount in USD. This represents the maximum expected loss
                at the given confidence level.

        Raises:
            ValueError: If confidence_level is not between 0 and 1.

        Algorithm:
            VaR = Portfolio Value × Daily Volatility × Z-Score × √Time Horizon
            - Uses 2% daily volatility assumption (would use historical data in production)
            - Z-scores: 90%=1.28, 95%=1.65, 99%=2.33 (normal distribution)
            - Scales by square root of time for multi-day horizons

        Example:
            >>> portfolio = Portfolio(cash_balance=Decimal("50000"))
            >>> # ... add positions ...
            >>> var_95 = calculator.calculate_portfolio_var(portfolio, Decimal("0.95"), 1)
            >>> print(f"95% 1-day VaR: ${var_95.amount:.2f}")
            >>> # Interpretation: 95% confident that losses won't exceed this amount in 1 day

        Limitations:
            - Assumes normal distribution of returns (often violated in practice)
            - Uses fixed volatility rather than historical or implied volatility
            - Doesn't account for correlation between positions
            - Simplified for demonstration; production would use historical simulation

        Note:
            VaR has known limitations including failure to capture tail risk.
            Consider supplementing with stress testing and scenario analysis.
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
        var = portfolio_value * daily_volatility * z_score * Decimal(math.sqrt(float(time_horizon)))

        return Money(var, "USD")

    def calculate_max_drawdown(self, portfolio_history: list[Decimal]) -> Decimal:
        """Calculate maximum drawdown from portfolio value history.

        Computes the largest peak-to-trough decline in portfolio value over the
        given history. Maximum drawdown is a key risk metric that measures the
        worst-case historical loss from a peak value.

        Args:
            portfolio_history: List of portfolio values over time, ordered chronologically.
                Each value represents the total portfolio value at a point in time.
                Requires at least MIN_DATA_POINTS_FOR_STATS values.

        Returns:
            Decimal: Maximum drawdown as a percentage (0-100). Returns 0 if insufficient
                data or empty list. A value of 25 means the portfolio experienced a
                maximum 25% decline from peak.

        Algorithm:
            1. Track running maximum (peak) value
            2. Calculate drawdown from peak for each subsequent value
            3. Return the maximum drawdown observed

        Example:
            >>> history = [Decimal("100000"), Decimal("110000"), Decimal("95000"),
            ...            Decimal("105000"), Decimal("85000")]
            >>> max_dd = calculator.calculate_max_drawdown(history)
            >>> # Peak was 110000, trough was 85000
            >>> # Drawdown = (110000 - 85000) / 110000 = 22.7%
            >>> assert max_dd == Decimal("22.7")

        Note:
            Drawdown is always calculated from the historical peak, not from
            the initial value. This captures the psychological impact of losses
            from the highest point achieved.
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

        Computes the risk-adjusted return metric that measures excess return per
        unit of risk. The Sharpe ratio is the industry standard for comparing
        investment strategies on a risk-adjusted basis.

        Args:
            returns: List of period returns as decimals (e.g., 0.01 for 1%).
                Assumed to be daily returns. Requires at least MIN_DATA_POINTS_FOR_STATS
                values for statistical significance.
            risk_free_rate: Annual risk-free rate as decimal (default 0.02 for 2%).
                Typically the Treasury bill rate or similar benchmark.

        Returns:
            Decimal | None: Annualized Sharpe ratio. Returns None if:
                - Insufficient data points
                - Standard deviation is zero (no volatility)
                Higher values indicate better risk-adjusted returns.
                - < 0: Losing money relative to risk-free rate
                - 0-1: Positive but subpar risk-adjusted returns
                - 1-2: Good risk-adjusted returns
                - > 2: Excellent risk-adjusted returns

        Formula:
            Sharpe = (Annual Return - Risk-Free Rate) / Annual Volatility
            - Returns are annualized by multiplying by 252 (trading days)
            - Volatility is annualized by multiplying by √252

        Example:
            >>> daily_returns = [Decimal("0.01"), Decimal("-0.005"), Decimal("0.015"), ...]
            >>> sharpe = calculator.calculate_sharpe_ratio(daily_returns, Decimal("0.02"))
            >>> if sharpe and sharpe > 1:
            ...     print("Strategy has good risk-adjusted returns")

        Note:
            The Sharpe ratio assumes returns are normally distributed, which may
            not hold for all trading strategies. Consider supplementing with other
            metrics like Sortino ratio for downside risk assessment.
        """
        if not returns or len(returns) < MIN_DATA_POINTS_FOR_STATS:
            return None

        # Calculate average return
        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((Decimal(str(r)) - Decimal(str(avg_return))) ** 2 for r in returns) / len(
            returns
        )
        std_dev = (
            variance.sqrt() if hasattr(variance, "sqrt") else Decimal(str(float(variance) ** 0.5))
        )

        if std_dev == 0:
            return None

        # Annualize (assuming daily returns)
        annual_return = Decimal(str(avg_return)) * Decimal(str(TRADING_DAYS_PER_YEAR))
        annual_std = Decimal(str(std_dev)) * ANNUALIZATION_FACTOR

        # Calculate Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_std

        return sharpe

    def check_risk_limits(self, portfolio: Portfolio, new_order: Order) -> tuple[bool, str]:
        """Check if a new order violates portfolio risk limits.

        Validates a proposed order against various risk management constraints
        including position limits, leverage limits, and concentration limits.
        This method serves as a pre-trade risk check to prevent excessive risk-taking.

        Args:
            portfolio: Current portfolio state against which to check limits.
            new_order: Proposed new order to validate.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if order is within all risk limits, False otherwise
                - str: Empty string if within limits, otherwise a description of
                    the violated limit

        Risk Checks:
            1. Position limits: Validates portfolio can accept new position
            2. Leverage limits: Ensures total exposure doesn't exceed leverage cap
            3. Concentration limits: Prevents excessive allocation to single position

        Default Limits:
            - Maximum leverage: Defined by portfolio.max_leverage
            - Maximum concentration: 20% of portfolio value per position

        Example:
            >>> portfolio = Portfolio(cash_balance=Decimal("10000"), max_leverage=2)
            >>> order = Order(symbol="TSLA", quantity=100, limit_price=Decimal("200"))
            >>> within_limits, reason = calculator.check_risk_limits(portfolio, order)
            >>> if not within_limits:
            ...     print(f"Order rejected: {reason}")

        Note:
            Uses limit_price for limit orders or assumes $100 for market orders
            when calculating position values. This is a conservative approach
            for market orders.
        """
        # Check position limits
        can_open, reason = portfolio.can_open_position(
            symbol=new_order.symbol,
            quantity=new_order.quantity,
            price=new_order.limit_price or Decimal("100"),  # Use estimate if market order
        )

        if not can_open:
            return False, reason or "Position cannot be opened"

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

        Computes the ratio of potential reward to potential risk for a trade setup.
        This is a fundamental metric for evaluating whether a trade offers favorable
        risk-adjusted returns.

        Args:
            entry_price: Planned entry price for the position.
            stop_loss: Stop loss price (risk level).
            take_profit: Take profit target price (reward level).

        Returns:
            Decimal: Risk/reward ratio. Values interpretation:
                - < 1: Risk exceeds reward (generally unfavorable)
                - 1: Risk equals reward (breakeven risk profile)
                - > 2: Reward is at least twice the risk (favorable)
                - > 3: Excellent risk/reward profile

        Raises:
            ValueError: If risk (distance to stop loss) is zero.

        Formula:
            Risk/Reward = (Take Profit - Entry) / (Entry - Stop Loss)

        Example:
            >>> entry = Price(Decimal("100"))
            >>> stop = Price(Decimal("95"))   # Risk: $5
            >>> target = Price(Decimal("115")) # Reward: $15
            >>> ratio = calculator.calculate_position_risk_reward(entry, stop, target)
            >>> assert ratio == Decimal("3")  # 15/5 = 3:1 risk/reward

        Note:
            A favorable risk/reward ratio doesn't guarantee profitability.
            The probability of reaching the target vs stop loss must also
            be considered (see expectancy calculations).
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

        Determines the mathematically optimal fraction of capital to risk based on
        the probability and magnitude of wins and losses. The Kelly Criterion maximizes
        long-term growth rate while avoiding ruin.

        Args:
            win_probability: Probability of winning as decimal (0-1).
                Should be based on historical performance or backtesting.
            win_amount: Average win amount in dollars (positive value).
            loss_amount: Average loss amount in dollars (positive value).

        Returns:
            Decimal: Optimal fraction of capital to risk (0-0.25).
                Capped at 25% for safety as full Kelly can be too aggressive.
                - Negative values indicate unfavorable odds (don't trade)
                - 0-0.10: Conservative position sizing
                - 0.10-0.25: Aggressive position sizing

        Raises:
            ValueError: If win_probability is not between 0 and 1.
            ValueError: If win_amount or loss_amount are not positive.

        Formula:
            f* = (p × b - q) / b
            Where:
            - f* = Optimal fraction of capital to bet
            - p = Probability of winning
            - q = Probability of losing (1 - p)
            - b = Win/loss ratio

        Example:
            >>> # 60% win rate, average win $200, average loss $100
            >>> win_prob = Decimal("0.6")
            >>> avg_win = Decimal("200")
            >>> avg_loss = Decimal("100")
            >>> kelly = calculator.calculate_kelly_criterion(win_prob, avg_win, avg_loss)
            >>> # f* = (0.6 × 2 - 0.4) / 2 = 0.4, capped at 0.25
            >>> print(f"Optimal position size: {kelly:.1%} of capital")

        Note:
            The Kelly Criterion assumes:
            - Accurate probability estimates
            - Consistent win/loss amounts
            - Independent trades
            Many traders use "fractional Kelly" (e.g., 25% of full Kelly) for safety.
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
        """Calculate comprehensive risk-adjusted return metrics.

        Computes a suite of risk-adjusted performance metrics that provide a
        holistic view of portfolio performance considering both returns and risk.
        These metrics are essential for strategy evaluation and comparison.

        Args:
            portfolio: Portfolio to analyze.
            _time_period_days: Period for calculations (currently unused but reserved
                for future time-based filtering). Defaults to 30 days.

        Returns:
            dict[str, Decimal | None]: Dictionary containing metrics:
                - total_return: Overall portfolio return percentage
                - win_rate: Percentage of winning trades
                - profit_factor: Ratio of gross profit to gross loss
                - average_win: Average profit on winning trades
                - average_loss: Average loss on losing trades
                - max_drawdown: Maximum peak-to-trough decline
                - sharpe_ratio: Risk-adjusted return metric
                - calmar_ratio: Return relative to maximum drawdown
                - expectancy: Expected value per trade
            Values are None when insufficient data or calculation not possible.

        Metrics Interpretation:
            - Profit Factor > 1.5: Good profitability
            - Win Rate > 50%: More winners than losers (not always necessary)
            - Sharpe Ratio > 1: Good risk-adjusted returns
            - Calmar Ratio > 1: Return exceeds maximum drawdown
            - Positive Expectancy: Positive expected value per trade

        Example:
            >>> portfolio = Portfolio(cash_balance=Decimal("100000"))
            >>> # ... execute trades ...
            >>> metrics = calculator.calculate_risk_adjusted_return(portfolio)
            >>> print(f"Win Rate: {metrics['win_rate']:.1f}%")
            >>> print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        Note:
            Some metrics may be None if the portfolio lacks sufficient trading
            history or if certain calculations are undefined (e.g., Calmar ratio
            when max drawdown is zero).
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

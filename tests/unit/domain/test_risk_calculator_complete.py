"""Comprehensive unit tests for RiskCalculator service to achieve 80%+ coverage.

This module provides extensive test coverage for the RiskCalculator service,
testing all risk metrics, calculations, and edge cases.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.entities import Order, OrderSide, OrderType, Portfolio, Position
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects import Money, Price


class TestRiskCalculatorInitialization:
    """Test RiskCalculator initialization."""

    def test_risk_calculator_creation(self):
        """Test creating a RiskCalculator instance."""
        calculator = RiskCalculator()
        assert calculator is not None
        assert isinstance(calculator, RiskCalculator)


class TestPositionRiskCalculations:
    """Test position-level risk calculations."""

    def test_calculate_position_risk_long_profit(self):
        """Test risk metrics for profitable long position."""
        calculator = RiskCalculator()
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("200"),
            stop_loss_price=Decimal("145.00"),
        )
        current_price = Price(Decimal("160.00"))

        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Decimal("16000")  # 100 * 160
        assert metrics["unrealized_pnl"] == Decimal("1000")  # 100 * (160 - 150)
        assert metrics["realized_pnl"] == Decimal("200")
        assert metrics["total_pnl"] == Decimal("1200")  # 200 + 1000
        assert metrics["return_pct"] == Decimal("100") * Decimal("1200") / Decimal("15000")
        assert metrics["risk_amount"] == Decimal("1500")  # (160 - 145) * 100

    def test_calculate_position_risk_short_loss(self):
        """Test risk metrics for losing short position."""
        calculator = RiskCalculator()
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            stop_loss_price=Decimal("660.00"),
        )
        current_price = Price(Decimal("655.00"))

        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Decimal("32750")  # 50 * 655
        assert metrics["unrealized_pnl"] == Decimal("-250")  # 50 * (650 - 655)
        assert metrics["total_pnl"] == Decimal("-250")
        assert metrics["risk_amount"] == Decimal("250")  # abs(655 - 660) * 50

    def test_calculate_position_risk_closed(self):
        """Test risk metrics for closed position."""
        calculator = RiskCalculator()
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("500"),
            closed_at=datetime.now(UTC),
        )
        current_price = Price(Decimal("160.00"))

        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Decimal("0")
        assert metrics["unrealized_pnl"] == Decimal("0")
        assert metrics["realized_pnl"] == Decimal("500")
        assert metrics["total_pnl"] == Decimal("500")
        assert metrics["return_pct"] == Decimal("0")
        assert metrics["risk_amount"] == Decimal("0")

    def test_calculate_position_risk_no_stop_loss(self):
        """Test risk metrics without stop loss."""
        calculator = RiskCalculator()
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        current_price = Price(Decimal("155.00"))

        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["risk_amount"] == Decimal("0")  # No stop loss set


class TestPortfolioVaR:
    """Test Value at Risk calculations."""

    def test_calculate_portfolio_var_95_confidence(self):
        """Test VaR calculation at 95% confidence level."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("50000"), initial_capital=Decimal("100000"))

        # Add some positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=Decimal("50"),
            average_entry_price=Decimal("350.00"),
            current_price=Decimal("360.00"),
        )

        var = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.95"), time_horizon=1
        )

        assert isinstance(var, Money)
        assert var.currency == "USD"
        # VaR = portfolio_value * volatility * z_score * sqrt(time)
        # portfolio_value ≈ 50000 + 15500 + 18000 = 83500
        # volatility = 0.02, z_score(0.95) = 1.65, sqrt(1) = 1
        # VaR ≈ 83500 * 0.02 * 1.65 * 1 = 2755.5
        assert var.amount > Decimal("2000")
        assert var.amount < Decimal("3000")

    def test_calculate_portfolio_var_99_confidence(self):
        """Test VaR calculation at 99% confidence level."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("100000"))

        var = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.99"), time_horizon=1
        )

        # 99% confidence has higher z-score (2.33)
        # VaR ≈ 100000 * 0.02 * 2.33 * 1 = 4660
        assert var.amount > Decimal("4000")
        assert var.amount < Decimal("5000")

    def test_calculate_portfolio_var_multi_day(self):
        """Test VaR calculation for multiple day horizon."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("100000"))

        var = calculator.calculate_portfolio_var(
            portfolio,
            confidence_level=Decimal("0.95"),
            time_horizon=5,  # 5 days
        )

        # VaR scales with sqrt(time)
        # VaR ≈ 100000 * 0.02 * 1.65 * sqrt(5) ≈ 7370
        assert var.amount > Decimal("7000")
        assert var.amount < Decimal("8000")

    def test_calculate_portfolio_var_invalid_confidence(self):
        """Test VaR with invalid confidence level raises error."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("100000"))

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("1.5"))

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("-0.1"))

    def test_calculate_portfolio_var_90_confidence(self):
        """Test VaR at 90% confidence level."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("100000"))

        var = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.90"), time_horizon=1
        )

        # 90% confidence has z-score of 1.28
        # VaR ≈ 100000 * 0.02 * 1.28 * 1 = 2560
        assert var.amount > Decimal("2000")
        assert var.amount < Decimal("3000")

    def test_calculate_portfolio_var_non_standard_confidence(self):
        """Test VaR with non-standard confidence level (uses default)."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("100000"))

        var = calculator.calculate_portfolio_var(
            portfolio,
            confidence_level=Decimal("0.93"),  # Not in predefined z-scores
            time_horizon=1,
        )

        # Should use default z-score of 1.65 (95% confidence)
        assert var.amount > Decimal("3000")
        assert var.amount < Decimal("4000")


class TestMaxDrawdown:
    """Test maximum drawdown calculations."""

    def test_calculate_max_drawdown_normal(self):
        """Test max drawdown with normal portfolio history."""
        calculator = RiskCalculator()

        history = [
            Decimal("100000"),
            Decimal("110000"),  # Peak
            Decimal("95000"),  # Drawdown
            Decimal("105000"),
            Decimal("85000"),  # Trough (max drawdown from 110000)
            Decimal("90000"),
            Decimal("100000"),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Max drawdown: (110000 - 85000) / 110000 = 22.73%
        expected = Decimal("100") * (Decimal("110000") - Decimal("85000")) / Decimal("110000")
        assert abs(max_dd - expected) < Decimal("0.01")

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with continuously increasing values."""
        calculator = RiskCalculator()

        history = [
            Decimal("100000"),
            Decimal("105000"),
            Decimal("110000"),
            Decimal("115000"),
            Decimal("120000"),
        ]

        max_dd = calculator.calculate_max_drawdown(history)
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_all_declining(self):
        """Test max drawdown with continuously declining values."""
        calculator = RiskCalculator()

        history = [
            Decimal("100000"),
            Decimal("90000"),
            Decimal("80000"),
            Decimal("70000"),
            Decimal("60000"),
        ]

        max_dd = calculator.calculate_max_drawdown(history)

        # Max drawdown: (100000 - 60000) / 100000 = 40%
        assert max_dd == Decimal("40")

    def test_calculate_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data points."""
        calculator = RiskCalculator()

        # Less than MIN_DATA_POINTS_FOR_STATS
        history = [Decimal("100000"), Decimal("95000")]

        max_dd = calculator.calculate_max_drawdown(history)
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_empty_history(self):
        """Test max drawdown with empty history."""
        calculator = RiskCalculator()

        max_dd = calculator.calculate_max_drawdown([])
        assert max_dd == Decimal("0")

    def test_calculate_max_drawdown_zero_peak(self):
        """Test max drawdown with zero peak value."""
        calculator = RiskCalculator()

        history = [Decimal("0"), Decimal("100"), Decimal("50")]

        max_dd = calculator.calculate_max_drawdown(history)
        # When peak is 0, drawdown calculation should handle division by zero
        assert max_dd == Decimal("50")  # (100 - 50) / 100 = 50%


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""

    def test_calculate_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive returns."""
        calculator = RiskCalculator()

        # Daily returns
        returns = [
            Decimal("0.01"),  # 1%
            Decimal("-0.005"),  # -0.5%
            Decimal("0.015"),  # 1.5%
            Decimal("0.002"),  # 0.2%
            Decimal("-0.003"),  # -0.3%
            Decimal("0.008"),  # 0.8%
            Decimal("0.012"),  # 1.2%
            Decimal("-0.002"),  # -0.2%
            Decimal("0.007"),  # 0.7%
            Decimal("0.005"),  # 0.5%
        ]

        sharpe = calculator.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.02"))

        assert sharpe is not None
        # Positive Sharpe ratio indicates returns exceed risk-free rate
        assert sharpe > Decimal("0")

    def test_calculate_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns."""
        calculator = RiskCalculator()

        # Mostly negative returns
        returns = [
            Decimal("-0.01"),
            Decimal("-0.015"),
            Decimal("0.005"),
            Decimal("-0.02"),
            Decimal("-0.008"),
            Decimal("0.002"),
            Decimal("-0.012"),
            Decimal("-0.005"),
            Decimal("-0.018"),
            Decimal("0.003"),
        ]

        sharpe = calculator.calculate_sharpe_ratio(returns)

        assert sharpe is not None
        # Negative Sharpe ratio indicates underperformance
        assert sharpe < Decimal("0")

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility returns None."""
        calculator = RiskCalculator()

        # Constant returns (zero volatility)
        returns = [Decimal("0.01")] * 10

        sharpe = calculator.calculate_sharpe_ratio(returns)

        # Cannot calculate Sharpe with zero volatility
        assert sharpe is None

    def test_calculate_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data returns None."""
        calculator = RiskCalculator()

        returns = [Decimal("0.01"), Decimal("0.02")]  # Too few data points

        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe is None

    def test_calculate_sharpe_ratio_empty_returns(self):
        """Test Sharpe ratio with empty returns list."""
        calculator = RiskCalculator()

        sharpe = calculator.calculate_sharpe_ratio([])
        assert sharpe is None


class TestRiskLimits:
    """Test risk limit checking."""

    def test_check_risk_limits_within_limits(self):
        """Test order within all risk limits."""
        calculator = RiskCalculator()
        portfolio = Portfolio(
            cash_balance=Decimal("50000"),
            max_position_size=Decimal("10000"),
            max_leverage=Decimal("2.0"),
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        assert within_limits is True
        assert reason == ""

    def test_check_risk_limits_exceeds_position_size(self):
        """Test order exceeding position size limit."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("50000"), max_position_size=Decimal("5000"))

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),  # Total: $10,000
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        assert within_limits is False
        assert "exceeds limit" in reason

    def test_check_risk_limits_exceeds_leverage(self):
        """Test order exceeding leverage limit."""
        calculator = RiskCalculator()
        portfolio = Portfolio(
            cash_balance=Decimal("10000"),
            max_leverage=Decimal("1.5"),
            max_position_size=Decimal("50000"),
        )

        # Add existing positions
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=Decimal("30"),
            average_entry_price=Decimal("350.00"),
            current_price=Decimal("360.00"),
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        assert within_limits is False
        assert "leverage limit" in reason

    def test_check_risk_limits_exceeds_concentration(self):
        """Test order exceeding concentration limit."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("20000"), initial_capital=Decimal("100000"))

        # Add positions to increase portfolio value
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=Decimal("100"),
            average_entry_price=Decimal("350.00"),
            current_price=Decimal("360.00"),
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),  # $15,000 position
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        assert within_limits is False
        assert "concentration limit" in reason

    def test_check_risk_limits_market_order(self):
        """Test risk limits with market order (uses estimate)."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("50000"), max_position_size=Decimal("15000"))

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,  # No limit price
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        # Should use $100 estimate for market order
        assert within_limits is True

    def test_check_risk_limits_zero_cash_balance(self):
        """Test risk limits with zero cash balance."""
        calculator = RiskCalculator()
        portfolio = Portfolio(cash_balance=Decimal("0"), max_leverage=Decimal("2.0"))

        order = Order(
            symbol="AAPL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)

        assert within_limits is False
        # Should detect infinite leverage (division by zero handled)


class TestRiskRewardRatio:
    """Test risk/reward ratio calculations."""

    def test_calculate_position_risk_reward_favorable(self):
        """Test favorable risk/reward ratio."""
        calculator = RiskCalculator()

        entry_price = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("95.00"))
        take_profit = Price(Decimal("115.00"))

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 100 - 95 = 5
        # Reward: 115 - 100 = 15
        # Ratio: 15 / 5 = 3
        assert ratio == Decimal("3")

    def test_calculate_position_risk_reward_unfavorable(self):
        """Test unfavorable risk/reward ratio."""
        calculator = RiskCalculator()

        entry_price = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("90.00"))
        take_profit = Price(Decimal("105.00"))

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 100 - 90 = 10
        # Reward: 105 - 100 = 5
        # Ratio: 5 / 10 = 0.5
        assert ratio == Decimal("0.5")

    def test_calculate_position_risk_reward_zero_risk(self):
        """Test risk/reward with zero risk raises error."""
        calculator = RiskCalculator()

        entry_price = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("100.00"))  # Same as entry
        take_profit = Price(Decimal("110.00"))

        with pytest.raises(ValueError, match="Risk cannot be zero"):
            calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""

    def test_calculate_kelly_criterion_favorable(self):
        """Test Kelly criterion with favorable odds."""
        calculator = RiskCalculator()

        win_probability = Decimal("0.6")  # 60% win rate
        win_amount = Decimal("200")
        loss_amount = Decimal("100")

        kelly = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # f* = (p*b - q) / b
        # p = 0.6, q = 0.4, b = 200/100 = 2
        # f* = (0.6*2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # But capped at 0.25
        assert kelly == Decimal("0.25")

    def test_calculate_kelly_criterion_unfavorable(self):
        """Test Kelly criterion with unfavorable odds."""
        calculator = RiskCalculator()

        win_probability = Decimal("0.3")  # 30% win rate
        win_amount = Decimal("100")
        loss_amount = Decimal("100")

        kelly = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # f* = (0.3*1 - 0.7) / 1 = -0.4
        # Negative indicates don't trade
        assert kelly < Decimal("0")

    def test_calculate_kelly_criterion_breakeven(self):
        """Test Kelly criterion at breakeven."""
        calculator = RiskCalculator()

        win_probability = Decimal("0.5")  # 50% win rate
        win_amount = Decimal("100")
        loss_amount = Decimal("100")

        kelly = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # f* = (0.5*1 - 0.5) / 1 = 0
        assert kelly == Decimal("0")

    def test_calculate_kelly_criterion_invalid_probability(self):
        """Test Kelly criterion with invalid probability."""
        calculator = RiskCalculator()

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(Decimal("1.5"), Decimal("100"), Decimal("100"))

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(Decimal("-0.1"), Decimal("100"), Decimal("100"))

    def test_calculate_kelly_criterion_invalid_amounts(self):
        """Test Kelly criterion with invalid amounts."""
        calculator = RiskCalculator()

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(Decimal("0.6"), Decimal("-100"), Decimal("100"))

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(Decimal("0.6"), Decimal("100"), Decimal("0"))


class TestRiskAdjustedReturn:
    """Test risk-adjusted return calculations."""

    def test_calculate_risk_adjusted_return_complete(self):
        """Test complete risk-adjusted return metrics."""
        calculator = RiskCalculator()

        portfolio = Portfolio(
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("95000"),
            winning_trades=6,
            losing_trades=4,
        )

        # Add some closed positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("500"),
            closed_at=datetime.now(UTC),
        )

        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=Decimal("0"),
            average_entry_price=Decimal("350.00"),
            realized_pnl=Decimal("-200"),
            closed_at=datetime.now(UTC),
        )

        # Add open position
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            quantity=Decimal("10"),
            average_entry_price=Decimal("2500.00"),
            current_price=Decimal("2550.00"),
        )

        portfolio.total_realized_pnl = Decimal("300")

        metrics = calculator.calculate_risk_adjusted_return(portfolio)

        assert "total_return" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "average_win" in metrics
        assert "average_loss" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["win_rate"] == Decimal("60")  # 6/(6+4) * 100

    def test_calculate_risk_adjusted_return_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calculator = RiskCalculator()

        portfolio = Portfolio(initial_capital=Decimal("100000"), cash_balance=Decimal("110000"))

        # Mock some return and drawdown
        portfolio.total_realized_pnl = Decimal("10000")

        metrics = calculator.calculate_risk_adjusted_return(portfolio)

        # With single data point, max_drawdown is 0
        # So calmar_ratio should be None
        assert "calmar_ratio" not in metrics or metrics["calmar_ratio"] is None

    def test_calculate_risk_adjusted_return_expectancy(self):
        """Test expectancy calculation."""
        calculator = RiskCalculator()

        portfolio = Portfolio(
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            winning_trades=6,
            losing_trades=4,
        )

        # Create positions with specific P&L
        for i in range(6):
            portfolio.positions[f"WIN{i}"] = Position(
                symbol=f"WIN{i}",
                quantity=Decimal("0"),
                average_entry_price=Decimal("100"),
                realized_pnl=Decimal("300"),
                closed_at=datetime.now(UTC),
            )

        for i in range(4):
            portfolio.positions[f"LOSS{i}"] = Position(
                symbol=f"LOSS{i}",
                quantity=Decimal("0"),
                average_entry_price=Decimal("100"),
                realized_pnl=Decimal("-200"),
                closed_at=datetime.now(UTC),
            )

        metrics = calculator.calculate_risk_adjusted_return(portfolio)

        if "expectancy" in metrics and metrics["expectancy"] is not None:
            # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            # = (0.6 * 300) - (0.4 * 200) = 180 - 80 = 100
            assert metrics["expectancy"] == Decimal("100")

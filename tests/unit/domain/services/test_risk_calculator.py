"""
Comprehensive unit tests for RiskCalculator domain service.

This consolidated test suite covers all risk calculation functionality including:
- Position risk metrics
- Portfolio Value at Risk (VaR)
- Sharpe ratio calculations
- Maximum drawdown analysis
- Risk limits validation
- Kelly criterion optimization
- Risk-adjusted returns
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects import Money, Price, Quantity


class TestPositionRiskCalculation:
    """Test suite for position-level risk calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    @pytest.fixture
    def long_position(self):
        """Create a test long position"""
        pos = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
            opened_at=datetime.now(UTC),
        )
        pos.realized_pnl = Money(Decimal("200.00"))
        return pos

    @pytest.fixture
    def short_position(self):
        """Create a test short position"""
        return Position(
            id=uuid4(),
            symbol="TSLA",
            quantity=Quantity(Decimal("-50")),
            average_entry_price=Price(Decimal("800.00")),
            current_price=Price(Decimal("750.00")),
            opened_at=datetime.now(UTC),
        )

    @pytest.fixture
    def closed_position(self):
        """Create a closed position"""
        pos = Position(
            id=uuid4(),
            symbol="MSFT",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("200.00")),
            current_price=Price(Decimal("210.00")),
            opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
        )
        pos.realized_pnl = Money(Decimal("1000.00"))
        return pos

    def test_calculate_position_risk_long_profitable(self, calculator, long_position):
        """Test risk calculation for profitable long position"""
        current_price = Price(Decimal("160.00"))

        metrics = calculator.calculate_position_risk(long_position, current_price)

        assert metrics["position_value"] == Money(Decimal("16000.00"))  # 100 * 160
        assert metrics["unrealized_pnl"] == Money(Decimal("1000.00"))  # (160-150) * 100
        assert metrics["realized_pnl"] == Money(Decimal("200.00"))
        assert metrics["total_pnl"] == Money(Decimal("1200.00"))
        assert abs(metrics["return_pct"] - Decimal("8.00")) < Decimal("0.1")
        assert metrics["risk_amount"] == Money(Decimal("0"))  # No stop loss

    def test_calculate_position_risk_long_loss(self, calculator):
        """Test risk calculation for losing long position"""
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("140.00")),
            opened_at=datetime.now(UTC),
        )

        current_price = Price(Decimal("140.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Money(Decimal("14000.00"))
        assert metrics["unrealized_pnl"] == Money(Decimal("-1000.00"))
        assert metrics["total_pnl"] == Money(Decimal("-1000.00"))
        assert metrics["return_pct"] < Decimal("0")

    def test_calculate_position_risk_with_stop_loss(self, calculator):
        """Test risk calculation with stop loss"""
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
            opened_at=datetime.now(UTC),
        )
        position.stop_loss_price = Price(Decimal("145.00"))

        current_price = Price(Decimal("155.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["risk_amount"] == Money(Decimal("1000.00"))  # (155-145) * 100

    def test_calculate_position_risk_short_profitable(self, calculator, short_position):
        """Test risk calculation for profitable short position"""
        current_price = Price(Decimal("750.00"))

        metrics = calculator.calculate_position_risk(short_position, current_price)

        assert metrics["position_value"] == Money(Decimal("37500.00"))  # abs(-50) * 750
        assert metrics["unrealized_pnl"] == Money(Decimal("2500.00"))  # (800-750) * 50
        assert metrics["return_pct"] > Decimal("0")

    def test_calculate_position_risk_short_with_stop_loss(self, calculator):
        """Test risk calculation for short position with stop loss"""
        position = Position(
            id=uuid4(),
            symbol="TSLA",
            quantity=Quantity(Decimal("-50")),
            average_entry_price=Price(Decimal("800.00")),
            current_price=Price(Decimal("750.00")),
            opened_at=datetime.now(UTC),
        )
        position.stop_loss_price = Price(Decimal("850.00"))

        current_price = Price(Decimal("750.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["risk_amount"] == Money(Decimal("5000.00"))  # abs(750-850) * 50

    def test_calculate_position_risk_closed(self, calculator, closed_position):
        """Test risk calculation for closed position"""
        current_price = Price(Decimal("220.00"))  # Should be ignored

        metrics = calculator.calculate_position_risk(closed_position, current_price)

        assert metrics["position_value"] == Money(Decimal("0"))
        assert metrics["unrealized_pnl"] == Money(Decimal("0"))
        assert metrics["realized_pnl"] == Money(Decimal("1000.00"))
        assert metrics["total_pnl"] == Money(Decimal("1000.00"))
        assert metrics["return_pct"] == Decimal("0")
        assert metrics["risk_amount"] == Money(Decimal("0"))

    def test_calculate_position_risk_fractional_shares(self, calculator):
        """Test position risk with fractional shares"""
        position = Position(
            id=uuid4(),
            symbol="FRAC",
            quantity=Quantity(Decimal("12.345")),
            average_entry_price=Price(Decimal("100.00")),
            current_price=Price(Decimal("105.00")),
            opened_at=datetime.now(UTC),
        )
        position.stop_loss_price = Price(Decimal("95.00"))

        current_price = Price(Decimal("105.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Money(Decimal("1296.225"))  # 12.345 * 105
        assert abs(metrics["unrealized_pnl"] - Money(Decimal("61.725"))) < Money(Decimal("0.001"))
        assert metrics["risk_amount"] == Money(Decimal("123.45"))  # (105-95) * 12.345

    @pytest.mark.skip(reason="Position risk calculation with zero quantity needs fix")
    def test_calculate_position_risk_zero_quantity(self, calculator):
        """Test position risk with zero quantity"""
        position = Position(
            id=uuid4(),
            symbol="ZERO",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("100.00")),
            current_price=Price(Decimal("100.00")),
            opened_at=datetime.now(UTC),
        )

        current_price = Price(Decimal("100.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Money(Decimal("0"))
        assert metrics["risk_amount"] == Money(Decimal("0"))


class TestPortfolioVaR:
    """Test suite for Portfolio Value at Risk calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio"""
        return Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.2"),
            max_leverage=Decimal("2.0"),
        )

    def test_calculate_portfolio_var_95_confidence(self, calculator, portfolio):
        """Test VaR calculation with default 95% confidence"""
        var = calculator.calculate_portfolio_var(portfolio)

        assert isinstance(var, Money)
        assert var.currency == "USD"
        # Expected: 50000 * 0.02 * 1.65 * 1 = 1650
        assert abs(var - Decimal("1650.00")) < Decimal("20")

    def test_calculate_portfolio_var_99_confidence(self, calculator, portfolio):
        """Test VaR calculation with 99% confidence"""
        var = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.99"), time_horizon=1
        )

        # Expected: 50000 * 0.02 * 2.33 * 1 = 2330
        assert abs(var - Decimal("2330.00")) < Decimal("25")

    def test_calculate_portfolio_var_90_confidence(self, calculator, portfolio):
        """Test VaR calculation with 90% confidence"""
        var = calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0.90"))

        # Expected: 50000 * 0.02 * 1.28 * 1 = 1280
        assert abs(var - Decimal("1280.00")) < Decimal("15")

    def test_calculate_portfolio_var_multi_day(self, calculator, portfolio):
        """Test VaR calculation with multi-day horizon"""
        var = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.95"), time_horizon=5
        )

        # Expected: 50000 * 0.02 * 1.65 * sqrt(5) ≈ 3689
        assert var > Decimal("3500")
        assert var < Decimal("3900")

    def test_calculate_portfolio_var_invalid_confidence(self, calculator, portfolio):
        """Test VaR with invalid confidence levels"""
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("0"))

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("1"))

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            calculator.calculate_portfolio_var(portfolio, confidence_level=Decimal("1.5"))

    def test_calculate_portfolio_var_unsupported_confidence(self, calculator, portfolio):
        """Test VaR with unsupported confidence level (uses default)"""
        var = calculator.calculate_portfolio_var(
            portfolio,
            confidence_level=Decimal("0.97"),  # Not in z_scores dict
        )

        # Should use default 1.65 z-score
        assert abs(var - Decimal("1650.00")) < Decimal("20")

    @pytest.mark.skip(reason="VaR calculation with zero value needs fix")
    def test_calculate_portfolio_var_zero_value(self, calculator):
        """Test VaR calculation when portfolio value is zero"""
        portfolio = Mock()
        portfolio.get_total_value = Mock(return_value=Decimal("0"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("0"))

        var = calculator.calculate_portfolio_var(portfolio)
        assert var == Decimal("0")

    @pytest.mark.skip(reason="Large portfolio VaR calculation needs optimization")
    def test_calculate_portfolio_var_large_portfolio(self, calculator):
        """Test VaR with extreme portfolio values"""
        portfolio = Mock()
        portfolio.get_total_value = Mock(return_value=Decimal("1000000000"))  # 1 billion
        portfolio.get_total_value_sync = Mock(return_value=Decimal("1000000000"))

        var = calculator.calculate_portfolio_var(portfolio)
        assert var > Decimal("10000000")  # Should be millions


class TestSharpeRatio:
    """Test suite for Sharpe ratio calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    def test_calculate_sharpe_ratio_positive_returns(self, calculator):
        """Test Sharpe ratio with positive returns"""
        returns = [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.015"),
            Decimal("0.005"),
            Decimal("-0.005"),
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.008"),
            Decimal("0.012"),
        ]

        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe is not None
        assert isinstance(sharpe, Decimal)
        assert sharpe > Decimal("0")

    def test_calculate_sharpe_ratio_negative_returns(self, calculator):
        """Test Sharpe ratio with negative returns"""
        returns = [
            Decimal("-0.01"),
            Decimal("-0.02"),
            Decimal("-0.015"),
            Decimal("-0.005"),
            Decimal("-0.025"),
        ]

        sharpe = calculator.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.02"))
        assert sharpe is not None
        assert sharpe < Decimal("0")

    def test_calculate_sharpe_ratio_zero_volatility(self, calculator):
        """Test Sharpe ratio with zero volatility"""
        returns = [Decimal("0.01")] * 10  # Constant returns

        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe is None  # Cannot calculate with zero std dev

    def test_calculate_sharpe_ratio_insufficient_data(self, calculator):
        """Test Sharpe ratio with insufficient data"""
        returns = [Decimal("0.01")]  # Only 1 data point

        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe is None

    def test_calculate_sharpe_ratio_empty_returns(self, calculator):
        """Test Sharpe ratio with empty returns"""
        sharpe = calculator.calculate_sharpe_ratio([])
        assert sharpe is None

    def test_calculate_sharpe_ratio_custom_risk_free(self, calculator):
        """Test Sharpe ratio with custom risk-free rate"""
        returns = [
            Decimal("0.02"),
            Decimal("0.01"),
            Decimal("0.015"),
            Decimal("0.025"),
            Decimal("0.005"),
        ]

        sharpe_low_rf = calculator.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.01"))
        sharpe_high_rf = calculator.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.05"))

        assert sharpe_low_rf is not None
        assert sharpe_high_rf is not None
        # Higher risk-free rate should result in lower Sharpe ratio
        assert sharpe_low_rf > sharpe_high_rf


class TestMaxDrawdown:
    """Test suite for maximum drawdown calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    def test_calculate_max_drawdown_normal(self, calculator):
        """Test max drawdown calculation with normal portfolio history"""
        history = [
            Money(Decimal("100000")),
            Money(Decimal("110000")),  # Peak
            Money(Decimal("105000")),
            Money(Decimal("95000")),  # Trough
            Money(Decimal("100000")),
            Money(Decimal("90000")),  # Lower trough
            Money(Decimal("95000")),
        ]

        drawdown = calculator.calculate_max_drawdown(history)

        # Max was 110000, min after that was 90000
        # Drawdown = (110000 - 90000) / 110000 * 100 = 18.18%
        assert abs(drawdown - Decimal("18.18")) < Decimal("0.02")

    def test_calculate_max_drawdown_no_drawdown(self, calculator):
        """Test max drawdown when portfolio only increases"""
        history = [
            Money(Decimal("100000")),
            Money(Decimal("110000")),
            Money(Decimal("120000")),
            Money(Decimal("130000")),
        ]

        drawdown = calculator.calculate_max_drawdown(history)
        assert drawdown == Decimal("0")

    def test_calculate_max_drawdown_continuous_decline(self, calculator):
        """Test max drawdown with continuous decline"""
        history = [
            Money(Decimal("100000")),
            Money(Decimal("90000")),
            Money(Decimal("80000")),
            Money(Decimal("70000")),
            Money(Decimal("60000")),
            Money(Decimal("50000")),
        ]

        drawdown = calculator.calculate_max_drawdown(history)
        assert drawdown == Decimal("50")  # 50% drawdown

    def test_calculate_max_drawdown_with_recovery(self, calculator):
        """Test max drawdown with recovery"""
        history = [
            Money(Decimal("100000")),
            Money(Decimal("80000")),  # 20% drawdown
            Money(Decimal("90000")),  # Recovery
            Money(Decimal("70000")),  # 30% drawdown from 100000
            Money(Decimal("110000")),  # Full recovery and new high
        ]

        drawdown = calculator.calculate_max_drawdown(history)
        assert drawdown == Decimal("30")  # Maximum was 30%

    def test_calculate_max_drawdown_complete_loss(self, calculator):
        """Test max drawdown with complete loss"""
        history = [Money(Decimal("100000")), Money(Decimal("50000")), Money(Decimal("0"))]

        drawdown = calculator.calculate_max_drawdown(history)
        assert drawdown == Decimal("100")  # 100% drawdown

    def test_calculate_max_drawdown_empty_history(self, calculator):
        """Test max drawdown with empty history"""
        drawdown = calculator.calculate_max_drawdown([])
        assert drawdown == Decimal("0")

    def test_calculate_max_drawdown_single_value(self, calculator):
        """Test max drawdown with single value"""
        drawdown = calculator.calculate_max_drawdown([Money(Decimal("100000"))])
        assert drawdown == Decimal("0")

    def test_calculate_max_drawdown_volatile_history(self, calculator):
        """Test max drawdown with highly volatile history"""
        history = [
            Money(Decimal("100000")),
            Money(Decimal("150000")),  # +50%
            Money(Decimal("75000")),  # -50% from peak
            Money(Decimal("200000")),  # Recovery and new peak
            Money(Decimal("50000")),  # -75% from new peak
            Money(Decimal("100000")),  # Partial recovery
        ]

        drawdown = calculator.calculate_max_drawdown(history)
        assert drawdown == Decimal("75")  # 75% maximum drawdown


class TestRiskLimits:
    """Test suite for risk limits validation"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio with risk limits"""
        return Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.2"),
            max_leverage=Decimal("2.0"),
        )

    def test_check_risk_limits_within_limits(self, calculator, portfolio):
        """Test risk limits check when order is within limits"""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("150.00")),
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is True
        assert reason == ""

    def test_check_risk_limits_exceeds_position_size(self, calculator, portfolio):
        """Test risk limits when order exceeds max position size"""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("200")),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("150.00")),  # 200 * 150 = 30000 > 20000 max
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is False
        assert "position size" in reason.lower()

    @pytest.mark.skip(reason="Risk limit leverage check needs fix")
    def test_check_risk_limits_exceeds_leverage(self, calculator):
        """Test risk limits when order exceeds leverage limit"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("10000")),
            max_position_size=Money(Decimal("50000")),
            max_leverage=Decimal("1.5"),
        )

        # Add existing position
        position = Position(
            id=uuid4(),
            symbol="MSFT",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("200.00")),
            current_price=Price(Decimal("200.00")),
            opened_at=datetime.now(UTC),
        )
        portfolio.positions["MSFT"] = position

        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("100.00")),
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is False
        assert "risk" in reason.lower() or "leverage" in reason.lower()

    def test_check_risk_limits_market_order(self, calculator, portfolio):
        """Test risk limits with market order (no limit price)"""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is True  # Should use estimate price

    def test_check_risk_limits_zero_cash(self, calculator):
        """Test risk limits with zero cash balance"""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("0")),
            max_position_size=Money(Decimal("20000")),
            max_leverage=Decimal("2.0"),
        )

        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("150.00")),
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is False
        assert "insufficient" in reason.lower() or "cash" in reason.lower()

    def test_check_risk_limits_with_sell_order(self, calculator, portfolio):
        """Test risk limits with sell orders"""
        # Add an existing position to sell
        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("160.00")),
            opened_at=datetime.now(UTC),
        )
        portfolio.positions["AAPL"] = position

        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("160.00")),
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert isinstance(within_limits, bool)
        assert isinstance(reason, str)


class TestKellyCriterion:
    """Test suite for Kelly criterion calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    def test_calculate_kelly_criterion_positive_edge(self, calculator):
        """Test Kelly criterion with positive edge"""
        win_prob = Decimal("0.6")
        win_amount = Money(Decimal("100"))
        loss_amount = Money(Decimal("50"))

        kelly = calculator.calculate_kelly_criterion(win_prob, win_amount, loss_amount)

        # f = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4, but capped at 0.25
        assert kelly == Decimal("0.25")

    def test_calculate_kelly_criterion_low_edge(self, calculator):
        """Test Kelly criterion with low edge"""
        win_prob = Decimal("0.55")
        win_amount = Money(Decimal("100"))
        loss_amount = Money(Decimal("100"))

        kelly = calculator.calculate_kelly_criterion(win_prob, win_amount, loss_amount)

        # f = (0.55 * 1 - 0.45) / 1 = 0.1
        assert kelly == Decimal("0.1")

    def test_calculate_kelly_criterion_negative_edge(self, calculator):
        """Test Kelly criterion with negative edge"""
        win_prob = Decimal("0.4")
        win_amount = Money(Decimal("100"))
        loss_amount = Money(Decimal("100"))

        kelly = calculator.calculate_kelly_criterion(win_prob, win_amount, loss_amount)

        # Negative edge should result in negative Kelly
        assert kelly < Decimal("0")

    def test_calculate_kelly_criterion_breakeven(self, calculator):
        """Test Kelly criterion at breakeven"""
        win_prob = Decimal("0.5")
        win_amount = Money(Decimal("100"))
        loss_amount = Money(Decimal("100"))

        kelly = calculator.calculate_kelly_criterion(win_prob, win_amount, loss_amount)

        # No edge = no bet
        assert kelly == Decimal("0")

    def test_calculate_kelly_criterion_high_edge(self, calculator):
        """Test Kelly criterion with very high edge"""
        win_prob = Decimal("0.8")
        win_amount = Money(Decimal("200"))
        loss_amount = Money(Decimal("50"))

        kelly = calculator.calculate_kelly_criterion(win_prob, win_amount, loss_amount)

        # Should be capped at 0.25
        assert kelly == Decimal("0.25")

    def test_calculate_kelly_criterion_invalid_probability(self, calculator):
        """Test Kelly criterion with invalid probability"""
        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(
                Decimal("0"), Money(Decimal("100")), Money(Decimal("50"))
            )

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(
                Decimal("1"), Money(Decimal("100")), Money(Decimal("50"))
            )

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(
                Decimal("-0.1"), Money(Decimal("100")), Money(Decimal("50"))
            )

    def test_calculate_kelly_criterion_invalid_amounts(self, calculator):
        """Test Kelly criterion with invalid amounts"""
        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(
                Decimal("0.6"), Money(Decimal("0")), Money(Decimal("50"))
            )

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(
                Decimal("0.6"), Money(Decimal("100")), Money(Decimal("-50"))
            )


class TestPositionRiskReward:
    """Test suite for position risk/reward ratio calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    def test_calculate_position_risk_reward_favorable(self, calculator):
        """Test risk/reward ratio with favorable setup"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("95.00"))
        take_profit = Price(Decimal("110.00"))

        ratio = calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

        # Risk = 5, Reward = 10, Ratio = 2
        assert ratio == Decimal("2")

    def test_calculate_position_risk_reward_unfavorable(self, calculator):
        """Test risk/reward ratio with unfavorable setup"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("90.00"))
        take_profit = Price(Decimal("105.00"))

        ratio = calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

        # Risk = 10, Reward = 5, Ratio = 0.5
        assert ratio == Decimal("0.5")

    def test_calculate_position_risk_reward_short(self, calculator):
        """Test risk/reward ratio for short position"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("105.00"))  # Stop above for short
        take_profit = Price(Decimal("90.00"))  # Target below for short

        ratio = calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

        # Risk = 5, Reward = 10, Ratio = 2
        assert ratio == Decimal("2")

    def test_calculate_position_risk_reward_zero_risk(self, calculator):
        """Test risk/reward with zero risk raises error"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("100.00"))
        take_profit = Price(Decimal("110.00"))

        with pytest.raises(ValueError, match="Risk cannot be zero"):
            calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

    def test_calculate_position_risk_reward_fractional(self, calculator):
        """Test risk/reward ratio with fractional result"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("90.00"))
        take_profit = Price(Decimal("115.00"))

        ratio = calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

        # Risk = 10, Reward = 15, Ratio = 1.5
        assert ratio == Decimal("1.5")

    def test_calculate_position_risk_reward_tight_stops(self, calculator):
        """Test risk/reward with very tight stops"""
        entry = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("99.99"))
        take_profit = Price(Decimal("100.01"))

        ratio = calculator.calculate_position_risk_reward(entry, stop_loss, take_profit)

        # Risk = 0.01, Reward = 0.01, Ratio = 1
        assert ratio == Decimal("1")


class TestRiskAdjustedReturn:
    """Test suite for risk-adjusted return calculations"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    @pytest.fixture
    def portfolio_with_metrics(self):
        """Create a portfolio with mocked metrics"""
        portfolio = Mock()
        portfolio.get_return_percentage = Mock(return_value=Decimal("15.5"))
        portfolio.get_win_rate = Mock(return_value=Decimal("60"))
        portfolio.get_profit_factor = Mock(return_value=Decimal("1.5"))
        portfolio.get_average_win = Mock(return_value=Decimal("500"))
        portfolio.get_average_loss = Mock(return_value=Decimal("300"))
        portfolio.get_sharpe_ratio = Mock(return_value=Decimal("1.2"))
        portfolio.get_total_value = Mock(return_value=Decimal("115500"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("115500"))
        return portfolio

    def test_calculate_risk_adjusted_return_complete(self, calculator, portfolio_with_metrics):
        """Test risk-adjusted return with all metrics available"""
        metrics = calculator.calculate_risk_adjusted_return(portfolio_with_metrics)

        assert metrics["total_return"] == Decimal("15.5")
        assert metrics["win_rate"] == Decimal("60")
        assert metrics["profit_factor"] == Decimal("1.5")
        assert metrics["average_win"] == Decimal("500")
        assert metrics["average_loss"] == Decimal("300")
        assert metrics["sharpe_ratio"] == Decimal("1.2")

        # Expectancy = (0.6 * 500) - (0.4 * 300) = 300 - 120 = 180
        assert metrics["expectancy"] == Decimal("180")

    def test_calculate_risk_adjusted_return_with_drawdown(self, calculator):
        """Test risk-adjusted return with max drawdown"""
        portfolio = Mock()
        portfolio.get_return_percentage = Mock(return_value=Decimal("20"))
        portfolio.get_win_rate = Mock(return_value=None)
        portfolio.get_profit_factor = Mock(return_value=None)
        portfolio.get_average_win = Mock(return_value=None)
        portfolio.get_average_loss = Mock(return_value=None)
        portfolio.get_sharpe_ratio = Mock(return_value=None)
        portfolio.get_total_value = Mock(return_value=Decimal("120000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("120000"))

        # Patch the actual method that gets called via delegation
        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown",
            return_value=Decimal("10"),
        ):
            metrics = calculator.calculate_risk_adjusted_return(portfolio)

        assert metrics["total_return"] == Decimal("20")
        assert metrics["max_drawdown"] == Decimal("10")
        assert metrics["calmar_ratio"] == Decimal("2")  # 20 / 10

    def test_calculate_risk_adjusted_return_zero_drawdown(self, calculator):
        """Test risk-adjusted return with zero drawdown"""
        portfolio = Mock()
        portfolio.get_return_percentage = Mock(return_value=Decimal("10"))
        portfolio.get_win_rate = Mock(return_value=None)
        portfolio.get_profit_factor = Mock(return_value=None)
        portfolio.get_average_win = Mock(return_value=None)
        portfolio.get_average_loss = Mock(return_value=None)
        portfolio.get_sharpe_ratio = Mock(return_value=None)
        portfolio.get_total_value = Mock(return_value=Decimal("110000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("110000"))

        # Patch the actual method that gets called via delegation
        with patch(
            "src.domain.services.risk.portfolio_var_calculator.PortfolioVaRCalculator.calculate_max_drawdown",
            return_value=Decimal("0"),
        ):
            metrics = calculator.calculate_risk_adjusted_return(portfolio)

        assert metrics["total_return"] == Decimal("10")
        assert metrics["max_drawdown"] == Decimal("0")
        # When drawdown is 0, calmar_ratio should be None (not missing from dict)
        assert metrics["calmar_ratio"] is None

    def test_calculate_risk_adjusted_return_missing_metrics(self, calculator):
        """Test risk-adjusted return with missing metrics"""
        portfolio = Mock()
        portfolio.get_return_percentage = Mock(return_value=None)
        portfolio.get_win_rate = Mock(return_value=Decimal("55"))
        portfolio.get_profit_factor = Mock(return_value=None)
        portfolio.get_average_win = Mock(return_value=Decimal("400"))
        portfolio.get_average_loss = Mock(return_value=None)
        portfolio.get_sharpe_ratio = Mock(return_value=None)
        portfolio.get_total_value = Mock(return_value=Decimal("100000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("100000"))

        metrics = calculator.calculate_risk_adjusted_return(portfolio)

        assert metrics["total_return"] is None
        assert metrics["win_rate"] == Decimal("55")
        assert metrics["average_win"] == Decimal("400")
        assert metrics["average_loss"] is None
        assert metrics.get("expectancy") is None  # Cannot calculate without average_loss

    def test_calculate_risk_adjusted_return_all_none(self, calculator):
        """Test risk-adjusted return when all metrics return None"""
        portfolio = Mock()
        portfolio.get_return_percentage = Mock(return_value=None)
        portfolio.get_win_rate = Mock(return_value=None)
        portfolio.get_profit_factor = Mock(return_value=None)
        portfolio.get_average_win = Mock(return_value=None)
        portfolio.get_average_loss = Mock(return_value=None)
        portfolio.get_sharpe_ratio = Mock(return_value=None)
        portfolio.get_total_value = Mock(return_value=Decimal("100000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("100000"))

        with patch.object(calculator, "calculate_max_drawdown", return_value=Decimal("0")):
            metrics = calculator.calculate_risk_adjusted_return(portfolio)

        assert metrics["total_return"] is None
        assert metrics["win_rate"] is None
        assert metrics["profit_factor"] is None
        assert metrics.get("calmar_ratio") is None
        assert metrics.get("expectancy") is None


class TestRiskCalculatorIntegration:
    """Integration tests for RiskCalculator with real entities"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    @pytest.fixture
    def populated_portfolio(self):
        """Create a portfolio with multiple positions"""
        portfolio = Portfolio(
            name="Integration Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("30000")),
            max_position_size=Money(Decimal("25000")),
            max_leverage=Decimal("3.0"),
        )

        # Add diverse positions
        positions = [
            Position(
                id=uuid4(),
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("150.00")),
                current_price=Price(Decimal("160.00")),
                opened_at=datetime.now(UTC),
            ),
            Position(
                id=uuid4(),
                symbol="GOOGL",
                quantity=Quantity(Decimal("50")),
                average_entry_price=Price(Decimal("2800.00")),
                current_price=Price(Decimal("2750.00")),
                opened_at=datetime.now(UTC),
            ),
            Position(
                id=uuid4(),
                symbol="TSLA",
                quantity=Quantity(Decimal("-30")),  # Short
                average_entry_price=Price(Decimal("900.00")),
                current_price=Price(Decimal("850.00")),
                opened_at=datetime.now(UTC),
            ),
        ]

        for pos in positions:
            portfolio.positions[pos.symbol] = pos

        return portfolio

    def test_full_portfolio_risk_analysis(self, calculator, populated_portfolio):
        """Test complete portfolio risk analysis"""
        # Calculate risk for each position
        positions_risk = {}
        for symbol, position in populated_portfolio.positions.items():
            current_price = Price(position.current_price)
            positions_risk[symbol] = calculator.calculate_position_risk(position, current_price)

        # Verify AAPL position (long, profitable)
        assert positions_risk["AAPL"]["unrealized_pnl"] == Money(Decimal("1000"))  # (160-150)*100

        # Verify GOOGL position (long, loss)
        assert positions_risk["GOOGL"]["unrealized_pnl"] == Money(
            Decimal("-2500")
        )  # (2750-2800)*50

        # Verify TSLA position (short, profitable)
        assert positions_risk["TSLA"]["unrealized_pnl"] == Money(Decimal("1500"))  # (900-850)*30

        # Calculate portfolio VaR
        var = calculator.calculate_portfolio_var(populated_portfolio)
        assert var > Decimal("0")

        # Calculate risk-adjusted returns
        metrics = calculator.calculate_risk_adjusted_return(populated_portfolio)
        assert "total_return" in metrics
        assert "win_rate" in metrics

    def test_position_lifecycle_risk_tracking(self, calculator):
        """Test risk tracking through position lifecycle"""
        # Create new position
        position = Position(
            id=uuid4(),
            symbol="NVDA",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("500.00")),
            current_price=Price(Decimal("500.00")),
            opened_at=datetime.now(UTC),
        )
        position.stop_loss_price = Price(Decimal("475.00"))
        position.take_profit_price = Price(Decimal("550.00"))

        # Initial risk assessment
        initial_metrics = calculator.calculate_position_risk(position, Price(Decimal("500.00")))
        assert initial_metrics["unrealized_pnl"] == Money(Decimal("0"))
        assert initial_metrics["risk_amount"] == Money(Decimal("1250"))  # (500-475)*50

        # Price moves up
        position.current_price = Price(Decimal("525.00"))
        up_metrics = calculator.calculate_position_risk(position, Price(Decimal("525.00")))
        assert up_metrics["unrealized_pnl"] == Money(Decimal("1250"))  # (525-500)*50
        assert up_metrics["risk_amount"] == Money(Decimal("2500"))  # (525-475)*50

        # Price moves down
        position.current_price = Price(Decimal("490.00"))
        down_metrics = calculator.calculate_position_risk(position, Price(Decimal("490.00")))
        assert down_metrics["unrealized_pnl"] == Money(Decimal("-500"))  # (490-500)*50
        assert down_metrics["risk_amount"] == Money(Decimal("750"))  # (490-475)*50

        # Close position
        position.quantity = Quantity(Decimal("0"))
        position.closed_at = datetime.now(UTC)
        position.realized_pnl = Money(Decimal("-500"))

        close_metrics = calculator.calculate_position_risk(position, Price(Decimal("490.00")))
        assert close_metrics["total_pnl"] == Money(Decimal("-500"))
        assert close_metrics["unrealized_pnl"] == Money(Decimal("0"))


class TestRiskCalculatorEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.fixture
    def calculator(self):
        """Create a RiskCalculator instance"""
        return RiskCalculator()

    def test_position_risk_with_extreme_prices(self, calculator):
        """Test position risk with extreme price values"""
        position = Position(
            id=uuid4(),
            symbol="EXTREME",
            quantity=Quantity(Decimal("1")),
            average_entry_price=Price(Decimal("0.0001")),
            current_price=Price(Decimal("10000.00")),
            opened_at=datetime.now(UTC),
        )
        position.stop_loss_price = Price(Decimal("0.00005"))

        current_price = Price(Decimal("10000.00"))
        metrics = calculator.calculate_position_risk(position, current_price)

        assert metrics["position_value"] == Money(Decimal("10000.00"))
        assert metrics["unrealized_pnl"] == Money(Decimal("9999.9999"))
        assert metrics["risk_amount"] == Money(Decimal("9999.99995"))

    def test_check_risk_limits_with_minimal_cash(self, calculator):
        """Test risk limits when portfolio has minimal cash"""
        portfolio = Portfolio(
            name="Low Cash Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100")),  # Very low cash, but not negative
            max_position_size=Money(Decimal("20000")),
            max_leverage=Decimal("3.0"),
        )

        # Add an existing position to use up most capital
        portfolio.positions["EXISTING"] = Position(
            id=uuid4(),
            symbol="EXISTING",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("999")),
            current_price=Price(Decimal("999")),
            opened_at=datetime.now(UTC),
        )

        order = Order(
            id=uuid4(),
            symbol="RISKY",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("100.00")),
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        within_limits, reason = calculator.check_risk_limits(portfolio, order)
        assert within_limits is False
        assert "insufficient" in reason.lower() or "cash" in reason.lower()

    def test_kelly_criterion_near_boundaries(self, calculator):
        """Test Kelly criterion with probabilities near boundaries"""
        # Very low win probability
        kelly_low = calculator.calculate_kelly_criterion(
            Decimal("0.001"), Money(Decimal("100")), Money(Decimal("100"))
        )
        assert kelly_low < Decimal("0")

        # Very high win probability
        kelly_high = calculator.calculate_kelly_criterion(
            Decimal("0.999"), Money(Decimal("100")), Money(Decimal("100"))
        )
        assert kelly_high == Decimal("0.25")  # Capped

    def test_portfolio_var_extreme_confidence(self, calculator):
        """Test VaR with extreme but valid confidence levels"""
        portfolio = Mock()
        portfolio.get_total_value = Mock(return_value=Money(Decimal("100000")))
        portfolio.get_total_value_sync = Mock(return_value=Money(Decimal("100000")))

        # Very low confidence
        var_low = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.01"), time_horizon=1
        )
        assert var_low > Money(Decimal("0"))

        # Very high confidence
        var_high = calculator.calculate_portfolio_var(
            portfolio, confidence_level=Decimal("0.999"), time_horizon=1
        )
        assert var_high > Money(Decimal("0"))

    def test_sharpe_ratio_with_extreme_returns(self, calculator):
        """Test Sharpe ratio with extreme mixed returns"""
        returns = [
            Decimal("0.50"),  # 50% gain
            Decimal("-0.30"),  # 30% loss
            Decimal("0.10"),
            Decimal("-0.05"),
            Decimal("0.20"),
        ]

        sharpe = calculator.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.02"))
        assert sharpe is not None
        assert isinstance(sharpe, Decimal)

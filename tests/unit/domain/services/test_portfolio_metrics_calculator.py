"""
Comprehensive Tests for Portfolio Metrics Calculator Service
==========================================================

Tests for the PortfolioMetricsCalculator domain service that handles
all portfolio financial calculations and performance metrics.
"""

from decimal import Decimal
from unittest.mock import Mock

from src.domain.entities.portfolio import Portfolio
from src.domain.services.portfolio_metrics_calculator import PortfolioMetricsCalculator
from src.domain.value_objects import Money


class TestPortfolioMetricsCalculatorBasic:
    """Test basic portfolio metrics calculations."""

    def test_get_total_value_cash_only(self):
        """Test total value calculation with cash only."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        assert total_value == Money(Decimal("50000"))

    def test_get_total_value_with_positions(self):
        """Test total value calculation with positions."""
        portfolio = Portfolio(cash_balance=Money(Decimal("30000")))

        # Create mock positions
        position1 = Mock()
        position1.get_position_value.return_value = Money(Decimal("10000"))
        position1.is_closed.return_value = False

        position2 = Mock()
        position2.get_position_value.return_value = Money(Decimal("15000"))
        position2.is_closed.return_value = False

        portfolio.positions = {"AAPL": position1, "MSFT": position2}

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        # Cash: 30000 + Positions: 10000 + 15000 = 55000
        assert total_value == Money(Decimal("55000"))

    def test_get_total_value_with_decimal_cash_balance(self):
        """Test total value with cash balance as Decimal."""
        portfolio = Mock()
        portfolio.cash_balance = Decimal("40000")  # Raw Decimal instead of Money
        portfolio.get_open_positions.return_value = []
        portfolio._calculate_simple_total_value.return_value = Money(Decimal("40000"))

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        assert total_value == Money(Decimal("40000"))

    def test_get_total_value_ignores_closed_positions(self):
        """Test that closed positions are ignored in total value."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        # Open position
        open_position = Mock()
        open_position.get_position_value.return_value = Money(Decimal("10000"))
        open_position.is_closed.return_value = False

        # Closed position
        closed_position = Mock()
        closed_position.get_position_value.return_value = Money(Decimal("5000"))
        closed_position.is_closed.return_value = True

        portfolio.positions = {"AAPL": open_position, "MSFT": closed_position}

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        # Only cash + open position: 50000 + 10000 = 60000
        assert total_value == Money(Decimal("60000"))

    def test_get_total_value_position_value_none(self):
        """Test total value when position value is None."""
        portfolio = Portfolio(cash_balance=Money(Decimal("25000")))

        position = Mock()
        position.get_position_value.return_value = None  # No current price
        position.is_closed.return_value = False

        portfolio.positions = {"AAPL": position}

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        # Only cash value: 25000
        assert total_value == Money(Decimal("25000"))


class TestPositionsValue:
    """Test positions value calculations."""

    def test_get_positions_value_empty(self):
        """Test positions value with no positions."""
        portfolio = Portfolio()

        positions_value = PortfolioMetricsCalculator.get_positions_value(portfolio)
        assert positions_value == Money(Decimal("0"))

    def test_get_positions_value_multiple_positions(self):
        """Test positions value with multiple positions."""
        portfolio = Portfolio()

        position1 = Mock()
        position1.get_position_value.return_value = Money(Decimal("12000"))
        position1.is_closed.return_value = False

        position2 = Mock()
        position2.get_position_value.return_value = Money(Decimal("8000"))
        position2.is_closed.return_value = False

        position3 = Mock()
        position3.get_position_value.return_value = Money(Decimal("5000"))
        position3.is_closed.return_value = False

        portfolio.positions = {"AAPL": position1, "MSFT": position2, "GOOGL": position3}

        positions_value = PortfolioMetricsCalculator.get_positions_value(portfolio)
        assert positions_value == Money(Decimal("25000"))

    def test_get_positions_value_with_none_values(self):
        """Test positions value when some positions have None values."""
        portfolio = Portfolio()

        position1 = Mock()
        position1.get_position_value.return_value = Money(Decimal("10000"))
        position1.is_closed.return_value = False

        position2 = Mock()
        position2.get_position_value.return_value = None
        position2.is_closed.return_value = False

        portfolio.positions = {"AAPL": position1, "MSFT": position2}

        positions_value = PortfolioMetricsCalculator.get_positions_value(portfolio)
        assert positions_value == Money(Decimal("10000"))


class TestUnrealizedPnL:
    """Test unrealized P&L calculations."""

    def test_get_unrealized_pnl_empty(self):
        """Test unrealized P&L with no positions."""
        portfolio = Portfolio()

        unrealized_pnl = PortfolioMetricsCalculator.get_unrealized_pnl(portfolio)
        assert unrealized_pnl == Money(Decimal("0"))

    def test_get_unrealized_pnl_mixed(self):
        """Test unrealized P&L with mixed profit/loss positions."""
        portfolio = Portfolio()

        # Profitable position
        position1 = Mock()
        position1.get_unrealized_pnl.return_value = Money(Decimal("1500"))
        position1.is_closed.return_value = False

        # Losing position
        position2 = Mock()
        position2.get_unrealized_pnl.return_value = Money(Decimal("-800"))
        position2.is_closed.return_value = False

        # Breakeven position
        position3 = Mock()
        position3.get_unrealized_pnl.return_value = Money(Decimal("0"))
        position3.is_closed.return_value = False

        portfolio.positions = {"AAPL": position1, "MSFT": position2, "GOOGL": position3}

        unrealized_pnl = PortfolioMetricsCalculator.get_unrealized_pnl(portfolio)
        # 1500 - 800 + 0 = 700
        assert unrealized_pnl == Money(Decimal("700"))

    def test_get_unrealized_pnl_with_none_values(self):
        """Test unrealized P&L when some positions have None values."""
        portfolio = Portfolio()

        position1 = Mock()
        position1.get_unrealized_pnl.return_value = Money(Decimal("500"))
        position1.is_closed.return_value = False

        position2 = Mock()
        position2.get_unrealized_pnl.return_value = None  # No current price
        position2.is_closed.return_value = False

        portfolio.positions = {"AAPL": position1, "MSFT": position2}

        unrealized_pnl = PortfolioMetricsCalculator.get_unrealized_pnl(portfolio)
        assert unrealized_pnl == Money(Decimal("500"))


class TestTotalPnL:
    """Test total P&L calculations."""

    def test_get_total_pnl_basic(self):
        """Test total P&L calculation."""
        portfolio = Portfolio(total_realized_pnl=Money(Decimal("2000")))

        position = Mock()
        position.get_unrealized_pnl.return_value = Money(Decimal("800"))
        position.is_closed.return_value = False

        portfolio.positions = {"AAPL": position}

        total_pnl = PortfolioMetricsCalculator.get_total_pnl(portfolio)
        # Realized: 2000 + Unrealized: 800 = 2800
        assert total_pnl == Money(Decimal("2800"))

    def test_get_total_pnl_negative(self):
        """Test total P&L with losses."""
        portfolio = Portfolio(total_realized_pnl=Money(Decimal("-500")))

        position = Mock()
        position.get_unrealized_pnl.return_value = Money(Decimal("-300"))
        position.is_closed.return_value = False

        portfolio.positions = {"AAPL": position}

        total_pnl = PortfolioMetricsCalculator.get_total_pnl(portfolio)
        # Realized: -500 + Unrealized: -300 = -800
        assert total_pnl == Money(Decimal("-800"))


class TestReturnPercentage:
    """Test return percentage calculations."""

    def test_get_return_percentage_positive(self):
        """Test positive return percentage."""
        portfolio = Mock()
        portfolio.initial_capital = Money(Decimal("100000"))
        portfolio.get_open_positions.return_value = []
        portfolio.cash_balance = Money(Decimal("110000"))  # 10% gain
        portfolio._calculate_simple_total_value.return_value = Money(Decimal("110000"))

        return_pct = PortfolioMetricsCalculator.get_return_percentage(portfolio)
        assert return_pct == Decimal("10")

    def test_get_return_percentage_negative(self):
        """Test negative return percentage."""
        portfolio = Mock()
        portfolio.initial_capital = Money(Decimal("100000"))
        portfolio.get_open_positions.return_value = []
        portfolio.cash_balance = Money(Decimal("85000"))  # 15% loss
        portfolio._calculate_simple_total_value.return_value = Money(Decimal("85000"))

        return_pct = PortfolioMetricsCalculator.get_return_percentage(portfolio)
        assert return_pct == Decimal("-15")

    def test_get_return_percentage_zero_capital(self):
        """Test return percentage with zero initial capital."""
        portfolio = Mock()
        portfolio.initial_capital = Money(Decimal("0"))

        return_pct = PortfolioMetricsCalculator.get_return_percentage(portfolio)
        assert return_pct == Decimal("0")

    def test_get_return_percentage_with_positions(self):
        """Test return percentage including position values."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("80000")),  # Some cash used for positions
        )

        position = Mock()
        position.get_position_value.return_value = Money(Decimal("25000"))
        position.is_closed.return_value = False

        portfolio.positions = {"AAPL": position}

        return_pct = PortfolioMetricsCalculator.get_return_percentage(portfolio)
        # Total value: 80000 + 25000 = 105000
        # Return: (105000 - 100000) / 100000 * 100 = 5%
        assert return_pct == Decimal("5")


class TestWinRate:
    """Test win rate calculations."""

    def test_get_win_rate_no_trades(self):
        """Test win rate with no completed trades."""
        portfolio = Portfolio()

        win_rate = PortfolioMetricsCalculator.get_win_rate(portfolio)
        assert win_rate is None

    def test_get_win_rate_all_wins(self):
        """Test win rate with all winning trades."""
        portfolio = Portfolio(winning_trades=5, losing_trades=0)

        win_rate = PortfolioMetricsCalculator.get_win_rate(portfolio)
        assert win_rate == Decimal("100")

    def test_get_win_rate_all_losses(self):
        """Test win rate with all losing trades."""
        portfolio = Portfolio(winning_trades=0, losing_trades=3)

        win_rate = PortfolioMetricsCalculator.get_win_rate(portfolio)
        assert win_rate == Decimal("0")

    def test_get_win_rate_mixed(self):
        """Test win rate with mixed results."""
        portfolio = Portfolio(winning_trades=7, losing_trades=3)

        win_rate = PortfolioMetricsCalculator.get_win_rate(portfolio)
        # 7 / (7 + 3) * 100 = 70%
        assert win_rate == Decimal("70")

    def test_get_win_rate_fractional(self):
        """Test win rate with fractional result."""
        portfolio = Portfolio(winning_trades=2, losing_trades=1)

        win_rate = PortfolioMetricsCalculator.get_win_rate(portfolio)
        # 2 / 3 * 100 = 66.6666...
        expected = (Decimal("2") / Decimal("3")) * Decimal("100")
        assert win_rate == expected


class TestAverageWin:
    """Test average win calculations."""

    def test_get_average_win_no_wins(self):
        """Test average win with no winning trades."""
        portfolio = Portfolio(winning_trades=0)

        average_win = PortfolioMetricsCalculator.get_average_win(portfolio)
        assert average_win is None

    def test_get_average_win_single_win(self):
        """Test average win with single winning trade."""
        portfolio = Portfolio(winning_trades=1)

        position = Mock()
        position.realized_pnl = Money(Decimal("500"))
        position.is_closed.return_value = True

        portfolio.positions = {"AAPL": position}

        average_win = PortfolioMetricsCalculator.get_average_win(portfolio)
        assert average_win == Money(Decimal("500"))

    def test_get_average_win_multiple_wins(self):
        """Test average win with multiple winning trades."""
        portfolio = Portfolio(winning_trades=3)

        # Winning positions
        position1 = Mock()
        position1.realized_pnl = Money(Decimal("600"))
        position1.is_closed.return_value = True

        position2 = Mock()
        position2.realized_pnl = Money(Decimal("900"))
        position2.is_closed.return_value = True

        position3 = Mock()
        position3.realized_pnl = Money(Decimal("1200"))
        position3.is_closed.return_value = True

        # Losing position (should be ignored)
        position4 = Mock()
        position4.realized_pnl = Money(Decimal("-300"))
        position4.is_closed.return_value = True

        portfolio.positions = {
            "AAPL": position1,
            "MSFT": position2,
            "GOOGL": position3,
            "TSLA": position4,
        }

        average_win = PortfolioMetricsCalculator.get_average_win(portfolio)
        # (600 + 900 + 1200) / 3 = 900
        assert average_win == Money(Decimal("900"))

    def test_get_average_win_ignores_losses(self):
        """Test that average win ignores losing trades."""
        portfolio = Portfolio(winning_trades=2)

        win_position = Mock()
        win_position.realized_pnl = Money(Decimal("1000"))
        win_position.is_closed.return_value = True

        loss_position = Mock()
        loss_position.realized_pnl = Money(Decimal("-500"))
        loss_position.is_closed.return_value = True

        breakeven_position = Mock()
        breakeven_position.realized_pnl = Money(Decimal("0"))
        breakeven_position.is_closed.return_value = True

        another_win_position = Mock()
        another_win_position.realized_pnl = Money(Decimal("500"))
        another_win_position.is_closed.return_value = True

        portfolio.positions = {
            "AAPL": win_position,
            "MSFT": loss_position,
            "GOOGL": breakeven_position,
            "TSLA": another_win_position,
        }

        average_win = PortfolioMetricsCalculator.get_average_win(portfolio)
        # (1000 + 500) / 2 = 750
        assert average_win == Money(Decimal("750"))


class TestAverageLoss:
    """Test average loss calculations."""

    def test_get_average_loss_no_losses(self):
        """Test average loss with no losing trades."""
        portfolio = Portfolio(losing_trades=0)

        average_loss = PortfolioMetricsCalculator.get_average_loss(portfolio)
        assert average_loss is None

    def test_get_average_loss_single_loss(self):
        """Test average loss with single losing trade."""
        portfolio = Portfolio(losing_trades=1)

        position = Mock()
        position.realized_pnl = Money(Decimal("-300"))
        position.is_closed.return_value = True

        portfolio.positions = {"AAPL": position}

        average_loss = PortfolioMetricsCalculator.get_average_loss(portfolio)
        assert average_loss == Money(Decimal("300"))  # Absolute value

    def test_get_average_loss_multiple_losses(self):
        """Test average loss with multiple losing trades."""
        portfolio = Portfolio(losing_trades=2)

        loss1 = Mock()
        loss1.realized_pnl = Money(Decimal("-400"))
        loss1.is_closed.return_value = True

        loss2 = Mock()
        loss2.realized_pnl = Money(Decimal("-600"))
        loss2.is_closed.return_value = True

        # Winning position (should be ignored)
        win = Mock()
        win.realized_pnl = Money(Decimal("500"))
        win.is_closed.return_value = True

        portfolio.positions = {"AAPL": loss1, "MSFT": loss2, "GOOGL": win}

        average_loss = PortfolioMetricsCalculator.get_average_loss(portfolio)
        # (400 + 600) / 2 = 500
        assert average_loss == Money(Decimal("500"))


class TestProfitFactor:
    """Test profit factor calculations."""

    def test_get_profit_factor_no_trades(self):
        """Test profit factor with no trades."""
        portfolio = Portfolio()

        profit_factor = PortfolioMetricsCalculator.get_profit_factor(portfolio)
        assert profit_factor is None

    def test_get_profit_factor_only_wins(self):
        """Test profit factor with only winning trades."""
        portfolio = Portfolio()

        position = Mock()
        position.realized_pnl = Money(Decimal("500"))
        position.is_closed.return_value = True

        portfolio.positions = {"AAPL": position}

        profit_factor = PortfolioMetricsCalculator.get_profit_factor(portfolio)
        # No losses, should return max value
        assert profit_factor == Decimal("999.99")

    def test_get_profit_factor_only_losses(self):
        """Test profit factor with only losing trades."""
        portfolio = Portfolio()

        position = Mock()
        position.realized_pnl = Money(Decimal("-300"))
        position.is_closed.return_value = True

        portfolio.positions = {"AAPL": position}

        profit_factor = PortfolioMetricsCalculator.get_profit_factor(portfolio)
        # No profits, should return 0
        assert profit_factor == Decimal("0")

    def test_get_profit_factor_mixed_trades(self):
        """Test profit factor with mixed trades."""
        portfolio = Portfolio()

        # Wins: 1000 + 500 = 1500
        win1 = Mock()
        win1.realized_pnl = Money(Decimal("1000"))
        win1.is_closed.return_value = True

        win2 = Mock()
        win2.realized_pnl = Money(Decimal("500"))
        win2.is_closed.return_value = True

        # Losses: 300 + 200 = 500
        loss1 = Mock()
        loss1.realized_pnl = Money(Decimal("-300"))
        loss1.is_closed.return_value = True

        loss2 = Mock()
        loss2.realized_pnl = Money(Decimal("-200"))
        loss2.is_closed.return_value = True

        # Breakeven (should be ignored)
        breakeven = Mock()
        breakeven.realized_pnl = Money(Decimal("0"))
        breakeven.is_closed.return_value = True

        portfolio.positions = {
            "AAPL": win1,
            "MSFT": win2,
            "GOOGL": loss1,
            "TSLA": loss2,
            "NVDA": breakeven,
        }

        profit_factor = PortfolioMetricsCalculator.get_profit_factor(portfolio)
        # 1500 / 500 = 3.0
        assert profit_factor == Decimal("3.0")

    def test_get_profit_factor_zero_gross_profit_and_loss(self):
        """Test profit factor when both profits and losses are zero."""
        portfolio = Portfolio()

        position = Mock()
        position.realized_pnl = Money(Decimal("0"))
        position.is_closed.return_value = True

        portfolio.positions = {"AAPL": position}

        profit_factor = PortfolioMetricsCalculator.get_profit_factor(portfolio)
        assert profit_factor is None


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""

    def test_get_sharpe_ratio_placeholder(self):
        """Test Sharpe ratio placeholder implementation."""
        portfolio = Portfolio()

        sharpe_ratio = PortfolioMetricsCalculator.get_sharpe_ratio(portfolio)
        # Current implementation is placeholder
        assert sharpe_ratio is None

    def test_get_sharpe_ratio_with_custom_risk_free_rate(self):
        """Test Sharpe ratio with custom risk-free rate."""
        portfolio = Portfolio()

        sharpe_ratio = PortfolioMetricsCalculator.get_sharpe_ratio(
            portfolio, risk_free_rate=Decimal("0.05")
        )
        # Current implementation is placeholder
        assert sharpe_ratio is None


class TestMaxDrawdown:
    """Test maximum drawdown calculations."""

    def test_get_max_drawdown_placeholder(self):
        """Test max drawdown placeholder implementation."""
        portfolio = Portfolio()

        max_drawdown = PortfolioMetricsCalculator.get_max_drawdown(portfolio)
        # Current implementation is placeholder
        assert max_drawdown == Decimal("0")


class TestPortfolioMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_calculations_with_empty_portfolio(self):
        """Test all calculations with empty portfolio."""
        portfolio = Portfolio()

        assert PortfolioMetricsCalculator.get_total_value(portfolio) == Money(Decimal("100000"))
        assert PortfolioMetricsCalculator.get_positions_value(portfolio) == Money(Decimal("0"))
        assert PortfolioMetricsCalculator.get_unrealized_pnl(portfolio) == Money(Decimal("0"))
        assert PortfolioMetricsCalculator.get_total_pnl(portfolio) == Money(Decimal("0"))
        assert PortfolioMetricsCalculator.get_return_percentage(portfolio) == Decimal("0")
        assert PortfolioMetricsCalculator.get_win_rate(portfolio) is None
        assert PortfolioMetricsCalculator.get_average_win(portfolio) is None
        assert PortfolioMetricsCalculator.get_average_loss(portfolio) is None
        assert PortfolioMetricsCalculator.get_profit_factor(portfolio) is None

    def test_calculations_with_extreme_values(self):
        """Test calculations with extreme values."""
        large_amount = Decimal("999999999999.99")
        portfolio = Portfolio(
            initial_capital=Money(large_amount),
            cash_balance=Money(large_amount),
            total_realized_pnl=Money(large_amount),
        )

        # Should handle large values without overflow
        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        assert total_value.amount == large_amount

        total_pnl = PortfolioMetricsCalculator.get_total_pnl(portfolio)
        assert total_pnl.amount == large_amount

    def test_precision_maintenance(self):
        """Test that decimal precision is maintained throughout calculations."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000.123456")),
            cash_balance=Money(Decimal("98765.654321")),
            total_realized_pnl=Money(Decimal("1234.567890")),
        )

        return_pct = PortfolioMetricsCalculator.get_return_percentage(portfolio)
        # Should maintain precision in calculation
        expected = (
            (Decimal("98765.654321") - Decimal("100000.123456")) / Decimal("100000.123456")
        ) * Decimal("100")
        assert return_pct == expected

    def test_calculations_with_mixed_position_states(self):
        """Test calculations with mix of open/closed positions."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        # Open position
        open_pos = Mock()
        open_pos.get_position_value.return_value = Money(Decimal("10000"))
        open_pos.get_unrealized_pnl.return_value = Money(Decimal("500"))
        open_pos.is_closed.return_value = False
        open_pos.realized_pnl = Money(Decimal("0"))

        # Closed position
        closed_pos = Mock()
        closed_pos.get_position_value.return_value = (
            None  # Closed positions don't have current value
        )
        closed_pos.get_unrealized_pnl.return_value = None
        closed_pos.is_closed.return_value = True
        closed_pos.realized_pnl = Money(Decimal("1000"))

        portfolio.positions = {"AAPL": open_pos, "MSFT": closed_pos}

        # Verify correct handling
        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        assert total_value == Money(Decimal("60000"))  # Cash + open position only

        positions_value = PortfolioMetricsCalculator.get_positions_value(portfolio)
        assert positions_value == Money(Decimal("10000"))  # Only open position

        unrealized_pnl = PortfolioMetricsCalculator.get_unrealized_pnl(portfolio)
        assert unrealized_pnl == Money(Decimal("500"))  # Only open position

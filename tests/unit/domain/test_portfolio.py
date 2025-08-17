"""
Unit tests for Portfolio entity
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

# Third-party imports
import pytest

# Local imports
from src.domain.entities import Portfolio, Position


@pytest.fixture
def default_portfolio():
    """Create a default portfolio for testing"""
    return Portfolio(
        name="Test Portfolio",
        initial_capital=Decimal("100000"),
        cash_balance=Decimal("100000"),
        max_position_size=Decimal("10000"),
        max_portfolio_risk=Decimal("0.02"),
        max_positions=10,
        strategy="test_strategy",
    )


@pytest.fixture
def portfolio_with_positions():
    """Create a portfolio with some existing positions"""
    portfolio = Portfolio(initial_capital=Decimal("100000"), cash_balance=Decimal("80000"))

    # Add an open winning position (current but not realized)
    position1 = Position.open_position(
        symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
    )
    portfolio.positions["AAPL"] = position1

    # Add a closed winning position (realized)
    position2 = Position.open_position(
        symbol="MSFT", quantity=Decimal("50"), entry_price=Decimal("100.00")
    )
    position2.realized_pnl = Decimal("500")
    position2.closed_at = datetime.now(UTC)
    position2.quantity = Decimal("0")
    portfolio.positions["MSFT"] = position2
    portfolio.winning_trades = 1

    # Add a closed losing position
    position3 = Position.open_position(
        symbol="GOOGL", quantity=Decimal("50"), entry_price=Decimal("100.00")
    )
    position3.realized_pnl = Decimal("-200")
    position3.closed_at = datetime.now(UTC)
    position3.quantity = Decimal("0")
    portfolio.positions["GOOGL"] = position3
    portfolio.losing_trades = 1

    portfolio.total_realized_pnl = Decimal("300")
    portfolio.trades_count = 3

    return portfolio


class TestPortfolioCreation:
    """Test Portfolio creation and validation"""

    def test_create_portfolio_with_defaults(self):
        """Test creating a portfolio with default values"""
        portfolio = Portfolio()

        assert portfolio.name == "Default Portfolio"
        assert portfolio.initial_capital == Decimal("100000")
        assert portfolio.cash_balance == Decimal("100000")
        assert portfolio.max_position_size == Decimal("10000")
        assert portfolio.max_portfolio_risk == Decimal("0.02")
        assert portfolio.max_positions == 10
        assert portfolio.max_leverage == Decimal("1.0")
        assert isinstance(portfolio.id, UUID)
        assert portfolio.created_at is not None
        assert portfolio.last_updated is None
        assert len(portfolio.positions) == 0

    def test_create_portfolio_with_custom_values(self):
        """Test creating a portfolio with custom values"""
        portfolio = Portfolio(
            name="Custom Portfolio",
            initial_capital=Decimal("50000"),
            cash_balance=Decimal("45000"),
            max_position_size=Decimal("5000"),
            max_portfolio_risk=Decimal("0.05"),
            max_positions=20,
            max_leverage=Decimal("2.0"),
            strategy="momentum",
        )

        assert portfolio.name == "Custom Portfolio"
        assert portfolio.initial_capital == Decimal("50000")
        assert portfolio.cash_balance == Decimal("45000")
        assert portfolio.max_position_size == Decimal("5000")
        assert portfolio.max_portfolio_risk == Decimal("0.05")
        assert portfolio.max_positions == 20
        assert portfolio.max_leverage == Decimal("2.0")
        assert portfolio.strategy == "momentum"


class TestPortfolioValidation:
    """Test Portfolio validation rules"""

    def test_negative_initial_capital_raises_error(self):
        """Test that negative initial capital raises ValueError"""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("-1000"))

    def test_zero_initial_capital_raises_error(self):
        """Test that zero initial capital raises ValueError"""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("0"))

    def test_negative_cash_balance_raises_error(self):
        """Test that negative cash balance raises ValueError"""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Portfolio(cash_balance=Decimal("-100"))

    def test_negative_max_position_size_raises_error(self):
        """Test that negative max position size raises ValueError"""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Decimal("-1000"))

    def test_zero_max_position_size_raises_error(self):
        """Test that zero max position size raises ValueError"""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Decimal("0"))

    def test_invalid_max_portfolio_risk_raises_error(self):
        """Test that invalid max portfolio risk raises ValueError"""
        # Test negative risk
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("-0.01"))

        # Test zero risk
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("0"))

        # Test risk > 1
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("1.5"))

    def test_invalid_max_positions_raises_error(self):
        """Test that invalid max positions raises ValueError"""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=0)

        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=-5)

    def test_invalid_max_leverage_raises_error(self):
        """Test that invalid max leverage raises ValueError"""
        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            Portfolio(max_leverage=Decimal("0.5"))


class TestPositionOpening:
    """Test opening positions in portfolio"""

    def test_can_open_position_success(self, default_portfolio):
        """Test successful position opening check"""
        can_open, reason = default_portfolio.can_open_position(
            symbol="AAPL", quantity=Decimal("10"), price=Decimal("150.00")
        )

        assert can_open is True
        assert reason is None

    def test_cannot_open_existing_position(self, default_portfolio):
        """Test cannot open position for symbol with existing open position"""
        # Open initial position
        default_portfolio.open_position(
            symbol="AAPL", quantity=Decimal("10"), entry_price=Decimal("150.00")
        )

        # Try to open another position for same symbol
        can_open, reason = default_portfolio.can_open_position(
            symbol="AAPL", quantity=Decimal("10"), price=Decimal("160.00")
        )

        assert can_open is False
        assert "Position already exists for AAPL" in reason

    def test_cannot_exceed_max_positions(self, default_portfolio):
        """Test cannot exceed maximum positions limit"""
        default_portfolio.max_positions = 2

        # Open two positions
        default_portfolio.open_position("AAPL", Decimal("10"), Decimal("150.00"))
        default_portfolio.open_position("GOOGL", Decimal("10"), Decimal("100.00"))

        # Try to open third position
        can_open, reason = default_portfolio.can_open_position(
            symbol="MSFT", quantity=Decimal("10"), price=Decimal("300.00")
        )

        assert can_open is False
        assert "Maximum positions limit reached" in reason

    def test_cannot_exceed_position_size_limit(self, default_portfolio):
        """Test cannot exceed position size limit"""
        can_open, reason = default_portfolio.can_open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150.00"),  # Total: $15,000 > $10,000 limit
        )

        assert can_open is False
        assert "Position size $15000.00 exceeds limit $10000.00" in reason

    def test_cannot_open_with_insufficient_cash(self, default_portfolio):
        """Test cannot open position with insufficient cash"""
        default_portfolio.cash_balance = Decimal("1000")

        can_open, reason = default_portfolio.can_open_position(
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("150.00"),  # Requires $1,500
        )

        assert can_open is False
        assert "Insufficient cash" in reason
        assert "$1000.00 available" in reason
        assert "$1500.00 required" in reason

    def test_cannot_exceed_portfolio_risk_limit(self, default_portfolio):
        """Test cannot exceed portfolio risk limit"""
        default_portfolio.max_portfolio_risk = Decimal("0.01")  # 1% risk limit

        can_open, reason = default_portfolio.can_open_position(
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("200.00"),  # $2,000 position = 2% of $100,000
        )

        assert can_open is False
        assert "Position risk 2.0% exceeds portfolio limit 1.0%" in reason

    def test_open_position_success(self, default_portfolio):
        """Test successfully opening a position"""
        initial_cash = default_portfolio.cash_balance

        position = default_portfolio.open_position(
            symbol="AAPL",
            quantity=Decimal("10"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="momentum",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("10")
        assert position.average_entry_price == Decimal("150.00")
        assert position.strategy == "momentum"

        # Check portfolio updates
        assert "AAPL" in default_portfolio.positions
        assert default_portfolio.positions["AAPL"] == position
        assert default_portfolio.cash_balance == initial_cash - Decimal("1505.00")  # 10*150 + 5
        assert default_portfolio.total_commission_paid == Decimal("5.00")
        assert default_portfolio.trades_count == 1
        assert default_portfolio.last_updated is not None

    def test_open_position_inherits_strategy(self, default_portfolio):
        """Test position inherits portfolio strategy if not specified"""
        position = default_portfolio.open_position(
            symbol="AAPL", quantity=Decimal("10"), entry_price=Decimal("150.00")
        )

        assert position.strategy == "test_strategy"

    def test_open_position_validation_failure(self, default_portfolio):
        """Test opening position with validation failure"""
        default_portfolio.cash_balance = Decimal("500")

        with pytest.raises(ValueError, match="Cannot open position: Insufficient cash"):
            default_portfolio.open_position(
                symbol="AAPL", quantity=Decimal("10"), entry_price=Decimal("150.00")
            )


class TestPositionClosing:
    """Test closing positions in portfolio"""

    def test_close_position_with_profit(self, default_portfolio):
        """Test closing a position with profit"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        # Open position
        default_portfolio.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
        )

        initial_cash = default_portfolio.cash_balance

        # Close with profit
        pnl = default_portfolio.close_position(
            symbol="AAPL", exit_price=Decimal("160.00"), commission=Decimal("5.00")
        )

        # P&L: 100 * (160 - 150) - 5 = 995
        assert pnl == Decimal("995.00")
        assert default_portfolio.total_realized_pnl == Decimal("995.00")
        assert default_portfolio.cash_balance == initial_cash + Decimal("15995.00")  # 100*160 - 5
        assert default_portfolio.total_commission_paid == Decimal("10.00")
        assert default_portfolio.winning_trades == 1
        assert default_portfolio.losing_trades == 0
        assert default_portfolio.last_updated is not None

    def test_close_position_with_loss(self, default_portfolio):
        """Test closing a position with loss"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        # Open position
        default_portfolio.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
        )

        # Close with loss
        pnl = default_portfolio.close_position(
            symbol="AAPL", exit_price=Decimal("140.00"), commission=Decimal("5.00")
        )

        # P&L: 100 * (140 - 150) - 5 = -1005
        assert pnl == Decimal("-1005.00")
        assert default_portfolio.total_realized_pnl == Decimal("-1005.00")
        assert default_portfolio.winning_trades == 0
        assert default_portfolio.losing_trades == 1

    def test_close_position_breakeven(self, default_portfolio):
        """Test closing a position at breakeven"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        # Open position
        default_portfolio.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
        )

        # Close at same price (loss due to commission)
        pnl = default_portfolio.close_position(
            symbol="AAPL", exit_price=Decimal("150.00"), commission=Decimal("5.00")
        )

        # P&L: 0 - 5 = -5 (commission only)
        assert pnl == Decimal("-5.00")
        assert default_portfolio.losing_trades == 1  # Commission makes it a loss

    def test_close_nonexistent_position(self, default_portfolio):
        """Test closing a position that doesn't exist"""
        with pytest.raises(ValueError, match="No position found for AAPL"):
            default_portfolio.close_position("AAPL", Decimal("150.00"))

    def test_close_already_closed_position(self, default_portfolio):
        """Test closing an already closed position"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        # Open and close position
        default_portfolio.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        default_portfolio.close_position("AAPL", Decimal("160.00"))

        # Try to close again
        with pytest.raises(ValueError, match="Position for AAPL is already closed"):
            default_portfolio.close_position("AAPL", Decimal("165.00"))


class TestPositionUpdates:
    """Test updating positions in portfolio"""

    def test_update_single_position_price(self, default_portfolio):
        """Test updating price for a single position"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        default_portfolio.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        default_portfolio.update_position_price("AAPL", Decimal("160.00"))

        position = default_portfolio.get_position("AAPL")
        assert position.current_price == Decimal("160.00")
        assert default_portfolio.last_updated is not None

    def test_update_nonexistent_position_price(self, default_portfolio):
        """Test updating price for nonexistent position"""
        with pytest.raises(ValueError, match="No position found for AAPL"):
            default_portfolio.update_position_price("AAPL", Decimal("150.00"))

    def test_update_all_prices(self, default_portfolio):
        """Test updating prices for multiple positions"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("25000")
        default_portfolio.max_portfolio_risk = Decimal("0.50")  # 50%

        # Open multiple positions
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        default_portfolio.open_position("GOOGL", Decimal("50"), Decimal("100.00"))
        default_portfolio.open_position("MSFT", Decimal("75"), Decimal("300.00"))

        # Close one position
        default_portfolio.close_position("MSFT", Decimal("310.00"))

        # Update all prices
        prices = {
            "AAPL": Decimal("160.00"),
            "GOOGL": Decimal("110.00"),
            "MSFT": Decimal("320.00"),  # Should be ignored (closed)
            "TSLA": Decimal("200.00"),  # Should be ignored (no position)
        }

        default_portfolio.update_all_prices(prices)

        # Check updates
        assert default_portfolio.positions["AAPL"].current_price == Decimal("160.00")
        assert default_portfolio.positions["GOOGL"].current_price == Decimal("110.00")
        assert default_portfolio.positions["MSFT"].current_price != Decimal("320.00")  # Not updated
        assert default_portfolio.last_updated is not None


class TestPositionQueries:
    """Test querying positions from portfolio"""

    def test_get_position(self, portfolio_with_positions):
        """Test getting a specific position"""
        position = portfolio_with_positions.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"

        # Nonexistent position
        assert portfolio_with_positions.get_position("TSLA") is None

    def test_get_open_positions(self, portfolio_with_positions):
        """Test getting all open positions"""
        open_positions = portfolio_with_positions.get_open_positions()

        assert len(open_positions) == 1
        assert open_positions[0].symbol == "AAPL"

    def test_get_closed_positions(self, portfolio_with_positions):
        """Test getting all closed positions"""
        closed_positions = portfolio_with_positions.get_closed_positions()

        assert len(closed_positions) == 2
        closed_symbols = {pos.symbol for pos in closed_positions}
        assert closed_symbols == {"GOOGL", "MSFT"}


class TestPortfolioValueCalculations:
    """Test portfolio value calculations"""

    def test_get_total_value_cash_only(self, default_portfolio):
        """Test total value with cash only"""
        assert default_portfolio.get_total_value() == Decimal("100000")

    def test_get_total_value_with_positions(self, default_portfolio):
        """Test total value with open positions"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # Open positions
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        default_portfolio.open_position("GOOGL", Decimal("50"), Decimal("100.00"))

        # Update prices
        default_portfolio.update_position_price("AAPL", Decimal("160.00"))
        default_portfolio.update_position_price("GOOGL", Decimal("110.00"))

        # Cash after positions: 100000 - 15000 - 5000 = 80000
        # Position values: 100*160 + 50*110 = 16000 + 5500 = 21500
        # Total: 80000 + 21500 = 101500
        assert default_portfolio.get_total_value() == Decimal("101500")

    def test_get_positions_value(self, default_portfolio):
        """Test calculating total positions value"""
        # No positions
        assert default_portfolio.get_positions_value() == Decimal("0")

        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # Open positions
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        default_portfolio.open_position("GOOGL", Decimal("50"), Decimal("100.00"))

        # Without current prices
        assert default_portfolio.get_positions_value() == Decimal("0")

        # With current prices
        default_portfolio.update_position_price("AAPL", Decimal("160.00"))
        default_portfolio.update_position_price("GOOGL", Decimal("110.00"))

        # 100*160 + 50*110 = 21500
        assert default_portfolio.get_positions_value() == Decimal("21500")

    def test_get_return_percentage(self, default_portfolio):
        """Test calculating portfolio return percentage"""
        # No change
        assert default_portfolio.get_return_percentage() == Decimal("0")

        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # With profit
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        default_portfolio.update_position_price("AAPL", Decimal("165.00"))

        # Current value: 85000 (cash) + 16500 (position) = 101500
        # Return: (101500 - 100000) / 100000 * 100 = 1.5%
        assert default_portfolio.get_return_percentage() == Decimal("1.5")

        # With loss
        default_portfolio.update_position_price("AAPL", Decimal("140.00"))

        # Current value: 85000 + 14000 = 99000
        # Return: (99000 - 100000) / 100000 * 100 = -1%
        assert default_portfolio.get_return_percentage() == Decimal("-1")

    def test_get_return_percentage_zero_initial_capital(self):
        """Test return percentage with zero initial capital"""
        portfolio = Portfolio(
            initial_capital=Decimal("0.01"),  # Can't be zero due to validation
            cash_balance=Decimal("0"),
        )
        portfolio.initial_capital = Decimal("0")  # Override after creation

        assert portfolio.get_return_percentage() == Decimal("0")


class TestPnLCalculations:
    """Test P&L calculation methods"""

    def test_get_unrealized_pnl(self, default_portfolio):
        """Test calculating unrealized P&L"""
        # No positions
        assert default_portfolio.get_unrealized_pnl() == Decimal("0")

        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # Open positions
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        default_portfolio.open_position("GOOGL", Decimal("50"), Decimal("100.00"))

        # Without current prices
        assert default_portfolio.get_unrealized_pnl() == Decimal("0")

        # With current prices
        default_portfolio.update_position_price("AAPL", Decimal("160.00"))  # +1000
        default_portfolio.update_position_price("GOOGL", Decimal("95.00"))  # -250

        assert default_portfolio.get_unrealized_pnl() == Decimal("750")

    def test_get_total_pnl(self, default_portfolio):
        """Test calculating total P&L (realized + unrealized)"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # Open and partially close position
        default_portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))

        # Partially close with profit
        position = default_portfolio.positions["AAPL"]
        position.reduce_position(Decimal("50"), Decimal("160.00"))  # +500 realized
        default_portfolio.total_realized_pnl = Decimal("500")

        # Update current price
        # +750 unrealized on remaining 50
        default_portfolio.update_position_price("AAPL", Decimal("165.00"))

        # Total: 500 (realized) + 750 (unrealized) = 1250
        assert default_portfolio.get_total_pnl() == Decimal("1250")


class TestPerformanceMetrics:
    """Test portfolio performance metrics"""

    def test_get_win_rate(self, portfolio_with_positions):
        """Test calculating win rate"""
        # 1 win, 1 loss = 50% win rate
        assert portfolio_with_positions.get_win_rate() == Decimal("50")

        # Add another win
        portfolio_with_positions.winning_trades = 2
        # 2 wins, 1 loss = 66.67% win rate
        expected = (Decimal("2") / Decimal("3")) * Decimal("100")
        assert portfolio_with_positions.get_win_rate() == expected

    def test_get_win_rate_no_trades(self, default_portfolio):
        """Test win rate with no closed trades"""
        assert default_portfolio.get_win_rate() is None

    def test_get_win_rate_all_wins(self, default_portfolio):
        """Test win rate with all winning trades"""
        default_portfolio.winning_trades = 5
        default_portfolio.losing_trades = 0
        assert default_portfolio.get_win_rate() == Decimal("100")

    def test_get_average_win(self, portfolio_with_positions):
        """Test calculating average winning trade"""
        avg_win = portfolio_with_positions.get_average_win()
        assert avg_win == Decimal("500")  # One win of 500

    def test_get_average_win_no_wins(self, default_portfolio):
        """Test average win with no winning trades"""
        assert default_portfolio.get_average_win() is None

    def test_get_average_loss(self, portfolio_with_positions):
        """Test calculating average losing trade"""
        avg_loss = portfolio_with_positions.get_average_loss()
        assert avg_loss == Decimal("200")  # One loss of -200

    def test_get_average_loss_no_losses(self, default_portfolio):
        """Test average loss with no losing trades"""
        assert default_portfolio.get_average_loss() is None

    def test_get_profit_factor(self):
        """Test calculating profit factor"""
        portfolio = Portfolio()

        # No trades
        assert portfolio.get_profit_factor() is None

        # Create winning position
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("100"))
        pos1.close_position(Decimal("110"))  # +1000 profit
        portfolio.positions["AAPL"] = pos1

        # Create losing position
        pos2 = Position.open_position("GOOGL", Decimal("100"), Decimal("100"))
        pos2.close_position(Decimal("95"))  # -500 loss
        portfolio.positions["GOOGL"] = pos2

        # Profit factor: 1000 / 500 = 2.0
        assert portfolio.get_profit_factor() == Decimal("2")

    def test_get_profit_factor_no_losses(self):
        """Test profit factor with no losses"""
        portfolio = Portfolio()

        # Only winning trades
        pos = Position.open_position("AAPL", Decimal("100"), Decimal("100"))
        pos.close_position(Decimal("110"))
        portfolio.positions["AAPL"] = pos

        # Should return capped value
        assert portfolio.get_profit_factor() == Decimal("999.99")

    def test_get_profit_factor_no_profits(self):
        """Test profit factor with no profits"""
        portfolio = Portfolio()

        # Only losing trades
        pos = Position.open_position("AAPL", Decimal("100"), Decimal("100"))
        pos.close_position(Decimal("90"))
        portfolio.positions["AAPL"] = pos

        # Should return 0
        assert portfolio.get_profit_factor() == Decimal("0")

    def test_get_sharpe_ratio(self, default_portfolio):
        """Test Sharpe ratio calculation (placeholder)"""
        # Currently returns None (placeholder implementation)
        assert default_portfolio.get_sharpe_ratio() is None

    def test_get_max_drawdown(self, default_portfolio):
        """Test max drawdown calculation (placeholder)"""
        # Currently returns 0 (placeholder implementation)
        assert default_portfolio.get_max_drawdown() == Decimal("0")


class TestPortfolioSerialization:
    """Test portfolio serialization"""

    def test_to_dict_basic(self, default_portfolio):
        """Test converting portfolio to dictionary"""
        result = default_portfolio.to_dict()

        assert result["name"] == "Test Portfolio"
        assert result["cash_balance"] == 100000.0
        assert result["total_value"] == 100000.0
        assert result["positions_value"] == 0.0
        assert result["unrealized_pnl"] == 0.0
        assert result["realized_pnl"] == 0.0
        assert result["total_pnl"] == 0.0
        assert result["return_pct"] == 0.0
        assert result["open_positions"] == 0
        assert result["total_trades"] == 0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["win_rate"] is None
        assert result["commission_paid"] == 0.0

    def test_to_dict_with_positions(self):
        """Test converting portfolio with positions to dictionary"""
        portfolio = Portfolio(initial_capital=Decimal("100000"), cash_balance=Decimal("85000"))

        # Temporarily increase limits for this test
        portfolio.max_position_size = Decimal("20000")
        portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        # Open position with profit
        portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("5"))
        portfolio.update_position_price("AAPL", Decimal("160.00"))

        # Add some closed trades
        portfolio.winning_trades = 3
        portfolio.losing_trades = 2
        portfolio.total_realized_pnl = Decimal("2500")
        portfolio.trades_count = 6  # 5 closed + 1 open

        result = portfolio.to_dict()

        assert result["cash_balance"] == 85000.0 - 15005.0  # After opening position
        assert result["positions_value"] == 16000.0  # 100 * 160
        assert result["unrealized_pnl"] == 1000.0  # 100 * (160-150)
        assert result["realized_pnl"] == 2500.0
        assert result["total_pnl"] == 3500.0  # 2500 + 1000
        assert result["open_positions"] == 1
        assert result["total_trades"] == 6
        assert result["win_rate"] == 60.0  # 3/5 * 100
        assert result["commission_paid"] == 5.0


class TestPortfolioStringRepresentation:
    """Test portfolio string representation"""

    def test_string_representation_basic(self, default_portfolio):
        """Test basic string representation"""
        str_repr = str(default_portfolio)

        assert "Test Portfolio" in str_repr
        assert "Value=$100000.00" in str_repr
        assert "Cash=$100000.00" in str_repr
        assert "Positions=0" in str_repr
        assert "P&L=$0.00" in str_repr
        assert "Return=0.00%" in str_repr

    def test_string_representation_with_positions(self):
        """Test string representation with positions"""
        portfolio = Portfolio(name="Active Portfolio")

        # Temporarily increase limits for this test
        portfolio.max_position_size = Decimal("20000")
        portfolio.max_portfolio_risk = Decimal("0.30")  # 30%

        portfolio.open_position("AAPL", Decimal("100"), Decimal("150.00"))
        portfolio.update_position_price("AAPL", Decimal("160.00"))

        str_repr = str(portfolio)

        assert "Active Portfolio" in str_repr
        assert "Positions=1" in str_repr
        # Should show unrealized profit
        assert "P&L=$1000.00" in str_repr


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.parametrize(
        "quantity,price,expected_size",
        [
            (Decimal("1"), Decimal("10000"), Decimal("10000")),  # Exactly at limit
            (Decimal("0.5"), Decimal("20000"), Decimal("10000")),  # Exactly at limit
            (Decimal("-100"), Decimal("100"), Decimal("10000")),  # Short position at limit
        ],
    )
    def test_position_size_at_limit(self, default_portfolio, quantity, price, expected_size):
        """Test position sizes exactly at the limit"""
        # Temporarily increase risk limit to allow max position size
        default_portfolio.max_portfolio_risk = Decimal(
            "0.15"
        )  # 15% - allows 10k position in 100k portfolio

        can_open, reason = default_portfolio.can_open_position("TEST", quantity, price)
        assert can_open is True
        assert reason is None

    def test_multiple_positions_interaction(self, default_portfolio):
        """Test interaction between multiple positions"""
        # Open multiple positions
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        for i, symbol in enumerate(symbols):
            default_portfolio.open_position(
                symbol=symbol,
                quantity=Decimal("10"),
                entry_price=Decimal(str(100 + i * 10)),
                commission=Decimal("1"),
            )

        # Update all prices
        prices = {symbol: Decimal(str(110 + i * 10)) for i, symbol in enumerate(symbols)}
        default_portfolio.update_all_prices(prices)

        # Close some positions
        default_portfolio.close_position("AAPL", Decimal("120"), Decimal("1"))  # Profit
        default_portfolio.close_position("MSFT", Decimal("115"), Decimal("1"))  # Loss

        # Verify state
        assert len(default_portfolio.get_open_positions()) == 3
        assert len(default_portfolio.get_closed_positions()) == 2
        assert default_portfolio.winning_trades == 1
        assert default_portfolio.losing_trades == 1

    def test_zero_commission_trades(self, default_portfolio):
        """Test trades with zero commission"""
        # Temporarily increase limits for this test
        default_portfolio.max_position_size = Decimal("20000")
        default_portfolio.max_portfolio_risk = Decimal("0.20")  # 20%

        default_portfolio.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("0"),
        )

        assert default_portfolio.total_commission_paid == Decimal("0")

        pnl = default_portfolio.close_position(
            symbol="AAPL", exit_price=Decimal("160.00"), commission=Decimal("0")
        )

        # Full profit without commission
        assert pnl == Decimal("1000.00")
        assert default_portfolio.total_commission_paid == Decimal("0")

    def test_very_small_positions(self, default_portfolio):
        """Test handling very small position sizes"""
        # Small fractional shares
        position = default_portfolio.open_position(
            symbol="BRK.A",
            quantity=Decimal("0.001"),
            entry_price=Decimal("500000.00"),  # $500 position
        )

        assert position.quantity == Decimal("0.001")
        assert default_portfolio.cash_balance == Decimal("99500")

    def test_very_large_numbers(self):
        """Test handling very large numbers"""
        portfolio = Portfolio(
            initial_capital=Decimal("1000000000"),  # 1 billion
            cash_balance=Decimal("1000000000"),
            max_position_size=Decimal("100000000"),  # 100 million
            max_portfolio_risk=Decimal("0.15"),  # 15% to allow large positions
        )

        portfolio.open_position(
            symbol="MEGA", quantity=Decimal("1000000"), entry_price=Decimal("100")
        )

        assert portfolio.cash_balance == Decimal("900000000")

        # Large profit
        pnl = portfolio.close_position("MEGA", Decimal("200"))
        assert pnl == Decimal("100000000")  # 100 million profit

    def test_precision_in_calculations(self, default_portfolio):
        """Test precision in financial calculations"""
        # Use prices that would cause rounding issues with floats
        default_portfolio.open_position(
            symbol="AAPL", quantity=Decimal("3"), entry_price=Decimal("149.99")
        )

        # Price that creates repeating decimal
        default_portfolio.update_position_price("AAPL", Decimal("150.01"))

        # Should maintain precision
        unrealized = default_portfolio.get_unrealized_pnl()
        assert unrealized == Decimal("0.06")  # 3 * (150.01 - 149.99)

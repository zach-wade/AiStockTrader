"""
Comprehensive test suite for Portfolio entity - achieving full coverage.
Tests portfolio management, position tracking, risk limits, and performance metrics.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest


class TestPortfolioInitialization:
    """Test Portfolio initialization and validation."""

    def test_portfolio_creation_with_defaults(self):
        """Test creating a portfolio with default values."""
        portfolio = Portfolio()

        assert isinstance(portfolio.id, UUID)
        assert portfolio.name == "Default Portfolio"
        assert portfolio.initial_capital == Decimal("100000")
        assert portfolio.cash_balance == Decimal("100000")
        assert portfolio.positions == {}
        assert portfolio.max_position_size == Decimal("10000")
        assert portfolio.max_portfolio_risk == Decimal("0.02")
        assert portfolio.max_positions == 10
        assert portfolio.max_leverage == Decimal("1.0")
        assert portfolio.total_realized_pnl == Decimal("0")
        assert portfolio.total_commission_paid == Decimal("0")
        assert portfolio.trades_count == 0
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0
        assert isinstance(portfolio.created_at, datetime)
        assert portfolio.last_updated is None
        assert portfolio.strategy is None
        assert portfolio.tags == {}

    def test_portfolio_creation_with_custom_values(self):
        """Test creating a portfolio with custom values."""
        portfolio = Portfolio(
            name="Aggressive Growth",
            initial_capital=Decimal("50000"),
            cash_balance=Decimal("45000"),
            max_position_size=Decimal("5000"),
            max_portfolio_risk=Decimal("0.05"),
            max_positions=20,
            max_leverage=Decimal("2.0"),
            strategy="momentum",
            tags={"risk_level": "high", "sector_focus": "tech"},
        )

        assert portfolio.name == "Aggressive Growth"
        assert portfolio.initial_capital == Decimal("50000")
        assert portfolio.cash_balance == Decimal("45000")
        assert portfolio.max_position_size == Decimal("5000")
        assert portfolio.max_portfolio_risk == Decimal("0.05")
        assert portfolio.max_positions == 20
        assert portfolio.max_leverage == Decimal("2.0")
        assert portfolio.strategy == "momentum"
        assert portfolio.tags == {"risk_level": "high", "sector_focus": "tech"}

    def test_portfolio_validation_negative_initial_capital(self):
        """Test that negative initial capital raises ValueError."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("-10000"))

    def test_portfolio_validation_zero_initial_capital(self):
        """Test that zero initial capital raises ValueError."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("0"))

    def test_portfolio_validation_negative_cash_balance(self):
        """Test that negative cash balance raises ValueError."""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Portfolio(cash_balance=Decimal("-1000"))

    def test_portfolio_validation_negative_max_position_size(self):
        """Test that negative max position size raises ValueError."""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Decimal("-5000"))

    def test_portfolio_validation_zero_max_position_size(self):
        """Test that zero max position size raises ValueError."""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Decimal("0"))

    def test_portfolio_validation_invalid_portfolio_risk(self):
        """Test that invalid portfolio risk raises ValueError."""
        # Zero risk
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("0"))

        # Risk > 1
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("1.5"))

    def test_portfolio_validation_negative_max_positions(self):
        """Test that negative max positions raises ValueError."""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=-5)

    def test_portfolio_validation_zero_max_positions(self):
        """Test that zero max positions raises ValueError."""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=0)

    def test_portfolio_validation_invalid_leverage(self):
        """Test that leverage less than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            Portfolio(max_leverage=Decimal("0.5"))


class TestPositionRequest:
    """Test PositionRequest dataclass."""

    def test_position_request_creation(self):
        """Test creating a position request."""
        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="momentum",
        )

        assert request.symbol == "AAPL"
        assert request.quantity == Decimal("100")
        assert request.entry_price == Decimal("150.00")
        assert request.commission == Decimal("5.00")
        assert request.strategy == "momentum"

    def test_position_request_defaults(self):
        """Test position request with default values."""
        request = PositionRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            entry_price=Decimal("350.00"),
        )

        assert request.commission == Decimal("0")
        assert request.strategy is None


class TestPositionManagement:
    """Test portfolio position management."""

    def test_can_open_position_valid(self):
        """Test checking if a position can be opened - valid case."""
        portfolio = Portfolio(
            cash_balance=Decimal("50000"),
            max_portfolio_risk=Decimal("0.20"),  # 20% to allow the position
        )

        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Decimal("50"),
            Decimal("150.00"),  # 7500 / 50000 = 15% risk
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_existing_position(self):
        """Test that existing position prevents opening new one."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("0.10"))  # 10% to allow positions

        # Open initial position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("50"),
            entry_price=Decimal("150.00"),
        )
        portfolio.open_position(request)

        # Try to open another position for same symbol
        can_open, reason = portfolio.can_open_position("AAPL", Decimal("50"), Decimal("155.00"))

        assert can_open is False
        assert "Position already exists for AAPL" in reason

    def test_can_open_position_max_positions_reached(self):
        """Test max positions limit check."""
        portfolio = Portfolio(
            max_positions=2,
            cash_balance=Decimal("100000"),
            max_portfolio_risk=Decimal("0.10"),  # Allow positions
        )

        # Open two positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("10"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("10"), Decimal("350.00")))

        # Try to open third position
        can_open, reason = portfolio.can_open_position("TSLA", Decimal("10"), Decimal("250.00"))

        assert can_open is False
        assert "Maximum positions limit reached (2)" in reason

    def test_can_open_position_exceeds_size_limit(self):
        """Test position size limit check."""
        portfolio = Portfolio(max_position_size=Decimal("5000"))

        # Try to open position exceeding size limit
        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Decimal("100"),
            Decimal("60.00"),  # 6000 > 5000
        )

        assert can_open is False
        assert "Position size $6000.00 exceeds limit $5000.00" in reason

    def test_can_open_position_insufficient_cash(self):
        """Test insufficient cash check."""
        portfolio = Portfolio(cash_balance=Decimal("5000"))

        # Try to open position requiring more cash
        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Decimal("50"),
            Decimal("150.00"),  # 7500 > 5000
        )

        assert can_open is False
        assert "Insufficient cash: $5000.00 available, $7500.00 required" in reason

    def test_can_open_position_exceeds_risk_limit(self):
        """Test portfolio risk limit check."""
        portfolio = Portfolio(
            cash_balance=Decimal("10000"),
            max_portfolio_risk=Decimal("0.20"),  # 20% max risk
        )

        # Try to open position exceeding risk limit
        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Decimal("20"),
            Decimal("150.00"),  # 3000 / 10000 = 30% > 20%
        )

        assert can_open is False
        assert "Position risk 30.0% exceeds portfolio limit 20.0%" in reason

    def test_can_open_position_closed_position_same_symbol(self):
        """Test that closed position allows opening new one for same symbol."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("0.10"))  # 10% risk limit

        # Open and close a position
        request = PositionRequest("AAPL", Decimal("50"), Decimal("150.00"))
        portfolio.open_position(request)
        portfolio.close_position("AAPL", Decimal("155.00"))

        # Should be able to open new position for same symbol
        can_open, reason = portfolio.can_open_position("AAPL", Decimal("30"), Decimal("160.00"))

        assert can_open is True
        assert reason is None

    def test_open_position_success(self):
        """Test successfully opening a position."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
            max_position_size=Decimal("20000"),  # Allow larger positions
        )
        initial_cash = portfolio.cash_balance

        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="momentum",
        )

        position = portfolio.open_position(request)

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.strategy == "momentum"

        # Check portfolio updates
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"] == position
        expected_cash = initial_cash - Decimal("15000") - Decimal("5")
        assert portfolio.cash_balance == expected_cash
        assert portfolio.total_commission_paid == Decimal("5.00")
        assert portfolio.trades_count == 1
        assert portfolio.last_updated is not None

    def test_open_position_uses_portfolio_strategy(self):
        """Test that position inherits portfolio strategy if not specified."""
        portfolio = Portfolio(
            strategy="value_investing",
            max_portfolio_risk=Decimal("0.20"),
            max_position_size=Decimal("20000"),
        )

        request = PositionRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            entry_price=Decimal("350.00"),
        )

        position = portfolio.open_position(request)
        assert position.strategy == "value_investing"

    def test_open_position_validation_failure(self):
        """Test that validation failure prevents opening position."""
        portfolio = Portfolio(
            cash_balance=Decimal("1000"),
            max_portfolio_risk=Decimal("1.0"),  # Allow any risk for this test
            max_position_size=Decimal("20000"),  # Allow large positions
        )

        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),  # Requires 15000, have 1000
        )

        with pytest.raises(ValueError, match="Cannot open position"):
            portfolio.open_position(request)

    def test_close_position_success(self):
        """Test successfully closing a position."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
        )
        portfolio.open_position(request)
        initial_cash = portfolio.cash_balance

        # Close with profit
        pnl = portfolio.close_position("AAPL", Decimal("160.00"), Decimal("5.00"))

        # P&L = 100 * (160 - 150) - 5 = 1000 - 5 = 995
        assert pnl == Decimal("995.00")

        # Check portfolio updates
        position = portfolio.positions["AAPL"]
        assert position.is_closed() is True
        expected_cash = initial_cash + Decimal("16000") - Decimal("5")
        assert portfolio.cash_balance == expected_cash
        assert portfolio.total_realized_pnl == Decimal("995.00")
        assert portfolio.total_commission_paid == Decimal("10.00")
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 0

    def test_close_position_with_loss(self):
        """Test closing a position with loss."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("15000")
        )

        # Open position
        request = PositionRequest(
            symbol="TSLA",
            quantity=Decimal("50"),
            entry_price=Decimal("250.00"),
        )
        portfolio.open_position(request)

        # Close with loss
        pnl = portfolio.close_position("TSLA", Decimal("240.00"), Decimal("3.00"))

        # P&L = 50 * (240 - 250) - 3 = -500 - 3 = -503
        assert pnl == Decimal("-503.00")
        assert portfolio.total_realized_pnl == Decimal("-503.00")
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1

    def test_close_position_breakeven(self):
        """Test closing a position at breakeven."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("15000")
        )

        # Open position
        request = PositionRequest(
            symbol="MSFT",
            quantity=Decimal("30"),
            entry_price=Decimal("350.00"),
            commission=Decimal("5.00"),
        )
        portfolio.open_position(request)

        # Close at same price with commission
        pnl = portfolio.close_position("MSFT", Decimal("350.00"), Decimal("5.00"))

        # P&L = 0 - 5 = -5 (only commission loss)
        assert pnl == Decimal("-5.00")
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1  # Loss due to commission

    def test_close_position_not_found(self):
        """Test closing non-existent position raises ValueError."""
        portfolio = Portfolio()

        with pytest.raises(ValueError, match="No position found for AAPL"):
            portfolio.close_position("AAPL", Decimal("150.00"))

    def test_close_position_already_closed(self):
        """Test closing already closed position raises ValueError."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open and close position
        request = PositionRequest("AAPL", Decimal("100"), Decimal("150.00"))
        portfolio.open_position(request)
        portfolio.close_position("AAPL", Decimal("155.00"))

        # Try to close again
        with pytest.raises(ValueError, match="Position for AAPL is already closed"):
            portfolio.close_position("AAPL", Decimal("160.00"))


class TestPriceUpdates:
    """Test portfolio price update functionality."""

    def test_update_position_price(self):
        """Test updating single position price."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open position
        request = PositionRequest("AAPL", Decimal("100"), Decimal("150.00"))
        portfolio.open_position(request)

        # Update price
        portfolio.update_position_price("AAPL", Decimal("155.00"))

        position = portfolio.positions["AAPL"]
        assert position.current_price == Decimal("155.00")
        assert position.last_updated is not None
        assert portfolio.last_updated is not None

    def test_update_position_price_not_found(self):
        """Test updating price for non-existent position raises ValueError."""
        portfolio = Portfolio()

        with pytest.raises(ValueError, match="No position found for AAPL"):
            portfolio.update_position_price("AAPL", Decimal("150.00"))

    def test_update_all_prices(self):
        """Test updating prices for multiple positions."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.25"),  # Slightly higher to allow all positions
            max_position_size=Decimal("20000"),
        )

        # Open multiple positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))

        # Close one position
        portfolio.close_position("TSLA", Decimal("255.00"))

        # Update all prices
        prices = {
            "AAPL": Decimal("155.00"),
            "MSFT": Decimal("355.00"),
            "TSLA": Decimal("260.00"),  # Should be ignored (closed)
            "NVDA": Decimal("450.00"),  # Should be ignored (no position)
        }

        portfolio.update_all_prices(prices)

        # Check open positions updated
        assert portfolio.positions["AAPL"].current_price == Decimal("155.00")
        assert portfolio.positions["MSFT"].current_price == Decimal("355.00")

        # Check closed position not updated
        assert portfolio.positions["TSLA"].current_price != Decimal("260.00")

        assert portfolio.last_updated is not None


class TestPositionQueries:
    """Test portfolio position query methods."""

    def test_get_position(self):
        """Test getting a specific position."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open position
        request = PositionRequest("AAPL", Decimal("100"), Decimal("150.00"))
        position = portfolio.open_position(request)

        # Get position
        retrieved = portfolio.get_position("AAPL")
        assert retrieved == position

        # Get non-existent position
        assert portfolio.get_position("MSFT") is None

    def test_get_open_positions(self):
        """Test getting all open positions."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.25"),  # Higher to allow all positions
            max_position_size=Decimal("20000"),
        )

        # Open multiple positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))

        # Close one
        portfolio.close_position("MSFT", Decimal("355.00"))

        open_positions = portfolio.get_open_positions()
        assert len(open_positions) == 2

        symbols = [p.symbol for p in open_positions]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "MSFT" not in symbols

    def test_get_closed_positions(self):
        """Test getting all closed positions."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.25"),  # Higher to allow all positions
            max_position_size=Decimal("20000"),
        )

        # Open multiple positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))

        # Close two
        portfolio.close_position("AAPL", Decimal("155.00"))
        portfolio.close_position("TSLA", Decimal("245.00"))

        closed_positions = portfolio.get_closed_positions()
        assert len(closed_positions) == 2

        symbols = [p.symbol for p in closed_positions]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "MSFT" not in symbols


class TestPortfolioValueCalculations:
    """Test portfolio value and P&L calculations."""

    def test_get_total_value(self):
        """Test calculating total portfolio value."""
        portfolio = Portfolio(
            cash_balance=Decimal("50000"),
            max_portfolio_risk=Decimal("0.50"),
            max_position_size=Decimal("20000"),
        )

        # Open positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))

        # Update prices
        portfolio.update_position_price("AAPL", Decimal("155.00"))
        portfolio.update_position_price("MSFT", Decimal("360.00"))

        total_value = portfolio.get_total_value()

        # Cash after positions: 50000 - 15000 - 17500 = 17500
        # AAPL value: 100 * 155 = 15500
        # MSFT value: 50 * 360 = 18000
        # Total: 17500 + 15500 + 18000 = 51000
        assert total_value == Decimal("51000.00")

    def test_get_total_value_no_prices(self):
        """Test total value when positions have no current prices."""
        portfolio = Portfolio(
            cash_balance=Decimal("50000"),
            max_portfolio_risk=Decimal("0.50"),
            max_position_size=Decimal("20000"),
        )

        # Open positions without updating prices
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))

        total_value = portfolio.get_total_value()

        # Only cash since position value is None
        assert total_value == Decimal("35000.00")  # 50000 - 15000

    def test_get_positions_value(self):
        """Test calculating total value of all positions."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.50"),  # Higher to allow both positions
            max_position_size=Decimal("20000"),
        )

        # Open positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))

        # Update prices
        portfolio.update_position_price("AAPL", Decimal("155.00"))
        portfolio.update_position_price("MSFT", Decimal("360.00"))

        positions_value = portfolio.get_positions_value()

        # AAPL: 100 * 155 = 15500
        # MSFT: 50 * 360 = 18000
        # Total: 33500
        assert positions_value == Decimal("33500.00")

    def test_get_unrealized_pnl(self):
        """Test calculating total unrealized P&L."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.50"),  # Higher to allow both positions
            max_position_size=Decimal("20000"),
        )

        # Open positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))

        # Update prices (AAPL up, MSFT down)
        portfolio.update_position_price("AAPL", Decimal("160.00"))
        portfolio.update_position_price("MSFT", Decimal("340.00"))

        unrealized = portfolio.get_unrealized_pnl()

        # AAPL: 100 * (160 - 150) = 1000
        # MSFT: 50 * (340 - 350) = -500
        # Total: 500
        assert unrealized == Decimal("500.00")

    def test_get_total_pnl(self):
        """Test calculating total P&L (realized + unrealized)."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open and close position for realized P&L
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))
        portfolio.close_position("TSLA", Decimal("260.00"))  # Realized: 250

        # Open positions for unrealized P&L
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.update_position_price("AAPL", Decimal("155.00"))  # Unrealized: 500

        total_pnl = portfolio.get_total_pnl()

        # Total: 250 + 500 = 750
        assert total_pnl == Decimal("750.00")

    def test_get_return_percentage(self):
        """Test calculating portfolio return percentage."""
        portfolio = Portfolio(
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            max_portfolio_risk=Decimal("0.50"),
            max_position_size=Decimal("20000"),
        )

        # Open positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))

        # Update prices for gains
        portfolio.update_position_price("AAPL", Decimal("165.00"))  # +1500
        portfolio.update_position_price("MSFT", Decimal("370.00"))  # +1000

        return_pct = portfolio.get_return_percentage()

        # Current value: cash (67500) + positions (16500 + 18500) = 102500
        # Return: (102500 - 100000) / 100000 * 100 = 2.5%
        assert return_pct == Decimal("2.5")

    def test_get_return_percentage_zero_capital(self):
        """Test return percentage with zero initial capital."""
        portfolio = Portfolio(
            initial_capital=Decimal("0"),
            cash_balance=Decimal("10000"),
        )

        # Override validation for testing
        portfolio.initial_capital = Decimal("0")

        assert portfolio.get_return_percentage() == Decimal("0")


class TestPerformanceMetrics:
    """Test portfolio performance metrics."""

    def test_get_win_rate(self):
        """Test calculating win rate percentage."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Open and close positions with mixed results
        # Win 1
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("155.00"))

        # Win 2
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.close_position("MSFT", Decimal("360.00"))

        # Loss 1
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))
        portfolio.close_position("TSLA", Decimal("240.00"))

        win_rate = portfolio.get_win_rate()

        # 2 wins out of 3 trades = 66.67%
        assert win_rate == Decimal("200") / Decimal("3")

    def test_get_win_rate_no_trades(self):
        """Test win rate with no closed trades."""
        portfolio = Portfolio()

        assert portfolio.get_win_rate() is None

    def test_get_average_win(self):
        """Test calculating average winning trade."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Win 1: +500
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("155.00"))

        # Win 2: +1000
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.close_position("MSFT", Decimal("370.00"))

        # Loss (should be ignored)
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))
        portfolio.close_position("TSLA", Decimal("240.00"))

        avg_win = portfolio.get_average_win()

        # (500 + 1000) / 2 = 750
        assert avg_win == Decimal("750.00")

    def test_get_average_win_no_wins(self):
        """Test average win with no winning trades."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Only losses
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("145.00"))

        assert portfolio.get_average_win() is None

    def test_get_average_loss(self):
        """Test calculating average losing trade."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Loss 1: -500
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("145.00"))

        # Loss 2: -250
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))
        portfolio.close_position("TSLA", Decimal("240.00"))

        # Win (should be ignored)
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.close_position("MSFT", Decimal("360.00"))

        avg_loss = portfolio.get_average_loss()

        # (500 + 250) / 2 = 375
        assert avg_loss == Decimal("375.00")

    def test_get_average_loss_no_losses(self):
        """Test average loss with no losing trades."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Only wins
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("155.00"))

        assert portfolio.get_average_loss() is None

    def test_get_profit_factor(self):
        """Test calculating profit factor."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Wins: +500, +1000 = 1500 gross profit
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("155.00"))

        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.close_position("MSFT", Decimal("370.00"))

        # Losses: -250, -300 = 550 gross loss
        portfolio.open_position(PositionRequest("TSLA", Decimal("25"), Decimal("250.00")))
        portfolio.close_position("TSLA", Decimal("240.00"))

        portfolio.open_position(PositionRequest("NVDA", Decimal("10"), Decimal("450.00")))
        portfolio.close_position("NVDA", Decimal("420.00"))

        profit_factor = portfolio.get_profit_factor()

        # 1500 / 550 = 2.727...
        assert profit_factor == Decimal("1500") / Decimal("550")

    def test_get_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        portfolio = Portfolio(
            max_portfolio_risk=Decimal("0.20"), max_position_size=Decimal("20000")
        )

        # Only wins
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.close_position("AAPL", Decimal("155.00"))

        profit_factor = portfolio.get_profit_factor()

        # Capped at 999.99
        assert profit_factor == Decimal("999.99")

    def test_get_profit_factor_no_trades(self):
        """Test profit factor with no trades."""
        portfolio = Portfolio()

        assert portfolio.get_profit_factor() is None

    def test_get_sharpe_ratio(self):
        """Test Sharpe ratio calculation (placeholder)."""
        portfolio = Portfolio()

        # Currently returns None (placeholder implementation)
        assert portfolio.get_sharpe_ratio() is None

    def test_get_max_drawdown(self):
        """Test max drawdown calculation (placeholder)."""
        portfolio = Portfolio()

        # Currently returns 0 (placeholder implementation)
        assert portfolio.get_max_drawdown() == Decimal("0")


class TestPortfolioSerialization:
    """Test portfolio serialization."""

    def test_to_dict(self):
        """Test converting portfolio to dictionary."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            max_portfolio_risk=Decimal("0.20"),
            max_position_size=Decimal("20000"),
        )

        # Add some activity
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.update_position_price("AAPL", Decimal("155.00"))

        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.close_position("MSFT", Decimal("360.00"))

        data = portfolio.to_dict()

        assert data["name"] == "Test Portfolio"
        assert isinstance(data["id"], str)
        assert isinstance(data["cash_balance"], float)
        assert isinstance(data["total_value"], float)
        assert isinstance(data["positions_value"], float)
        assert isinstance(data["unrealized_pnl"], float)
        assert isinstance(data["realized_pnl"], float)
        assert isinstance(data["total_pnl"], float)
        assert isinstance(data["return_pct"], float)
        assert data["open_positions"] == 1
        assert data["total_trades"] == 2
        assert data["winning_trades"] == 1
        assert data["losing_trades"] == 0
        assert isinstance(data["win_rate"], float)
        assert isinstance(data["commission_paid"], float)

    def test_to_dict_no_win_rate(self):
        """Test to_dict when win rate is None."""
        portfolio = Portfolio()

        data = portfolio.to_dict()

        assert data["win_rate"] is None


class TestPortfolioStringRepresentation:
    """Test Portfolio string representation."""

    def test_portfolio_string(self):
        """Test string representation of portfolio."""
        portfolio = Portfolio(
            name="Growth Portfolio",
            cash_balance=Decimal("75000"),
            max_portfolio_risk=Decimal("0.20"),
            max_position_size=Decimal("20000"),
        )

        # Add positions
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))
        portfolio.open_position(PositionRequest("MSFT", Decimal("50"), Decimal("350.00")))
        portfolio.update_position_price("AAPL", Decimal("155.00"))
        portfolio.update_position_price("MSFT", Decimal("360.00"))

        portfolio_str = str(portfolio)

        assert "Growth Portfolio" in portfolio_str
        assert "Value=" in portfolio_str
        assert "Cash=" in portfolio_str
        assert "Positions=2" in portfolio_str
        assert "P&L=" in portfolio_str
        assert "Return=" in portfolio_str


class TestPortfolioEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_complex_portfolio_lifecycle(self):
        """Test complex portfolio lifecycle with multiple operations."""
        portfolio = Portfolio(
            name="Active Trading",
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size=Decimal("20000"),
            strategy="day_trading",
        )

        # Open multiple positions
        positions_opened = []
        for symbol, qty, price in [
            ("AAPL", Decimal("100"), Decimal("150.00")),
            ("MSFT", Decimal("50"), Decimal("350.00")),
            ("TSLA", Decimal("40"), Decimal("250.00")),
            ("NVDA", Decimal("20"), Decimal("450.00")),
        ]:
            request = PositionRequest(symbol, qty, price, commission=Decimal("5.00"))
            position = portfolio.open_position(request)
            positions_opened.append(position)

        # Update all prices
        new_prices = {
            "AAPL": Decimal("155.00"),  # +500
            "MSFT": Decimal("345.00"),  # -250
            "TSLA": Decimal("260.00"),  # +400
            "NVDA": Decimal("445.00"),  # -100
        }
        portfolio.update_all_prices(new_prices)

        # Close some positions
        portfolio.close_position("AAPL", Decimal("156.00"), Decimal("5.00"))  # Win
        portfolio.close_position("MSFT", Decimal("344.00"), Decimal("5.00"))  # Loss

        # Check metrics
        assert len(portfolio.get_open_positions()) == 2
        assert portfolio.trades_count == 4
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 1

        # Verify cash flow
        # Initial: 100000
        # Positions: -15000 -17500 -10000 -9000 = -51500
        # Commissions: -20
        # Close AAPL: +15600 -5
        # Close MSFT: +17200 -5
        # Final cash should reflect all these transactions

        # Check performance metrics
        assert portfolio.get_win_rate() == Decimal("50")
        assert portfolio.get_average_win() is not None
        assert portfolio.get_average_loss() is not None
        assert portfolio.get_profit_factor() is not None

    def test_portfolio_with_leverage(self):
        """Test portfolio with leverage settings."""
        portfolio = Portfolio(
            cash_balance=Decimal("50000"),
            max_leverage=Decimal("2.0"),  # 2x leverage allowed
        )

        # Should be able to open positions worth more than cash with leverage
        # This is mainly for validation - actual leverage implementation
        # would be in broker integration
        assert portfolio.max_leverage == Decimal("2.0")

    def test_portfolio_risk_management(self):
        """Test portfolio risk management features."""
        portfolio = Portfolio(
            cash_balance=Decimal("100000"),
            max_portfolio_risk=Decimal("0.10"),  # 10% max risk
            max_position_size=Decimal("15000"),
        )

        # Test risk limits are enforced
        can_open, reason = portfolio.can_open_position(
            "RISKY",
            Decimal("100"),
            Decimal("200.00"),  # 20000 position
        )

        assert can_open is False
        assert "exceeds" in reason.lower()

    def test_portfolio_with_tags_and_metadata(self):
        """Test portfolio with custom tags and metadata."""
        portfolio = Portfolio(
            name="Quantitative Strategy",
            strategy="mean_reversion",
            tags={
                "model": "ML_v2.3",
                "risk_score": 0.65,
                "sector_allocation": {"tech": 0.6, "finance": 0.4},
            },
        )

        assert portfolio.strategy == "mean_reversion"
        assert portfolio.tags["model"] == "ML_v2.3"
        assert portfolio.tags["risk_score"] == 0.65
        assert portfolio.tags["sector_allocation"]["tech"] == 0.6

    def test_portfolio_timestamp_tracking(self):
        """Test portfolio timestamp tracking."""
        before = datetime.now(UTC)
        portfolio = Portfolio()
        after = datetime.now(UTC)

        assert before <= portfolio.created_at <= after
        assert portfolio.created_at.tzinfo is not None
        assert portfolio.last_updated is None

        # Trigger update
        portfolio.open_position(PositionRequest("AAPL", Decimal("100"), Decimal("150.00")))

        assert portfolio.last_updated is not None
        assert portfolio.last_updated.tzinfo is not None
        assert portfolio.last_updated >= portfolio.created_at

    def test_zero_portfolio_value_edge_case(self):
        """Test edge case with zero portfolio value."""
        portfolio = Portfolio(
            initial_capital=Decimal("10000"),
            cash_balance=Decimal("0"),
        )

        # Open position using all cash
        portfolio.cash_balance = Decimal("10000")  # Reset for test
        portfolio.open_position(PositionRequest("AAPL", Decimal("66.67"), Decimal("150.00")))

        # Cash should be near zero
        assert portfolio.cash_balance < Decimal("1")

        # Risk calculation should still work
        can_open, reason = portfolio.can_open_position("MSFT", Decimal("10"), Decimal("350.00"))
        assert can_open is False  # No cash left

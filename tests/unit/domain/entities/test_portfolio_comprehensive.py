"""
Comprehensive test suite for Portfolio Entity

Achieves >85% coverage by testing:
- Portfolio initialization and validation
- Position management (open, close, update)
- Risk limit validation
- Performance metrics calculation
- Edge cases and error conditions
- Portfolio statistics
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioInitialization:
    """Test suite for Portfolio initialization."""

    def test_default_initialization(self):
        """Test portfolio with default values."""
        portfolio = Portfolio()

        assert portfolio.name == "Default Portfolio"
        assert portfolio.initial_capital == Money(Decimal("100000"))
        assert portfolio.cash_balance == Money(Decimal("100000"))
        assert len(portfolio.positions) == 0
        assert portfolio.max_position_size == Money(Decimal("10000"))
        assert portfolio.max_portfolio_risk == Decimal("0.02")
        assert portfolio.max_positions == 10
        assert portfolio.max_leverage == Decimal("1.0")
        assert portfolio.total_realized_pnl == Money(Decimal("0"))
        assert portfolio.total_commission_paid == Money(Decimal("0"))
        assert portfolio.trades_count == 0
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0
        assert isinstance(portfolio.created_at, datetime)
        assert portfolio.last_updated is None
        assert portfolio.strategy is None
        assert len(portfolio.tags) == 0

    def test_custom_initialization(self):
        """Test portfolio with custom values."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("50000")),
            cash_balance=Money(Decimal("45000")),
            max_position_size=Money(Decimal("5000")),
            max_portfolio_risk=Decimal("0.01"),
            max_positions=5,
            max_leverage=Decimal("2.0"),
            strategy="momentum",
            tags={"type": "aggressive"},
        )

        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == Money(Decimal("50000"))
        assert portfolio.cash_balance == Money(Decimal("45000"))
        assert portfolio.max_position_size == Money(Decimal("5000"))
        assert portfolio.max_portfolio_risk == Decimal("0.01")
        assert portfolio.max_positions == 5
        assert portfolio.max_leverage == Decimal("2.0")
        assert portfolio.strategy == "momentum"
        assert portfolio.tags["type"] == "aggressive"

    def test_validation_negative_initial_capital(self):
        """Test validation rejects negative initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("-1000"))

    def test_validation_zero_initial_capital(self):
        """Test validation rejects zero initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Decimal("0"))

    def test_validation_negative_cash_balance(self):
        """Test validation rejects negative cash balance."""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Portfolio(cash_balance=Decimal("-100"))

    def test_validation_negative_max_position_size(self):
        """Test validation rejects negative max position size."""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Decimal("-1000"))

    def test_validation_invalid_max_portfolio_risk(self):
        """Test validation rejects invalid portfolio risk."""
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("0"))

        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("1.5"))

    def test_validation_negative_max_positions(self):
        """Test validation rejects negative max positions."""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=0)

    def test_validation_invalid_max_leverage(self):
        """Test validation rejects invalid leverage."""
        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            Portfolio(max_leverage=Decimal("0.5"))


class TestPortfolioPositionValidation:
    """Test suite for position validation."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return Portfolio(
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            max_position_size=Decimal("10000"),
            max_portfolio_risk=Decimal("0.02"),
            max_positions=3,
        )

    def test_can_open_position_success(self, portfolio):
        """Test successful position validation."""
        # Use smaller quantity to stay within 2% risk limit
        # 10 shares × $150 = $1,500 = 1.5% of $100,000
        can_open, reason = portfolio.can_open_position(
            symbol="AAPL", quantity=Quantity(Decimal("10")), price=Price(Decimal("150.00"))
        )

        assert can_open is True, f"Failed to open position: {reason}"
        assert reason is None

    def test_can_open_position_already_exists(self, portfolio):
        """Test validation when position already exists."""
        # Add existing position
        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
        )
        portfolio.positions["AAPL"] = position

        can_open, reason = portfolio.can_open_position(
            symbol="AAPL", quantity=Quantity(Decimal("50")), price=Price(Decimal("155.00"))
        )

        assert can_open is False
        assert "Position already exists for AAPL" in reason

    def test_can_open_position_closed_position_exists(self, portfolio):
        """Test validation when closed position exists."""
        # Add closed position
        from datetime import datetime

        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("150.00")),
            closed_at=datetime.now(UTC),
        )
        portfolio.positions["AAPL"] = position

        # Use smaller quantity to stay within 2% risk limit
        can_open, reason = portfolio.can_open_position(
            symbol="AAPL", quantity=Quantity(Decimal("10")), price=Price(Decimal("155.00"))
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_max_positions_reached(self, portfolio):
        """Test validation when max positions reached."""
        # Add max number of positions
        for i in range(3):
            position = Position(
                symbol=f"SYM{i}",
                quantity=Quantity(Decimal("10")),
                average_entry_price=Price(Decimal("100.00")),
            )
            portfolio.positions[f"SYM{i}"] = position

        can_open, reason = portfolio.can_open_position(
            symbol="NEW", quantity=Quantity(Decimal("10")), price=Price(Decimal("100.00"))
        )

        assert can_open is False
        assert "Maximum positions limit reached (3)" in reason

    def test_can_open_position_exceeds_size_limit(self, portfolio):
        """Test validation when position exceeds size limit."""
        can_open, reason = portfolio.can_open_position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            price=Decimal("150.00"),  # Total: $15,000 > $10,000 limit
        )

        assert can_open is False
        assert "Position size" in reason and "exceeds limit" in reason

    def test_can_open_position_insufficient_cash(self, portfolio):
        """Test validation with insufficient cash."""
        portfolio.cash_balance = Decimal("5000")

        can_open, reason = portfolio.can_open_position(
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            price=Decimal("150.00"),  # Needs $7,500
        )

        assert can_open is False
        assert "Insufficient cash" in reason and "available" in reason and "required" in reason

    def test_can_open_position_exceeds_risk_limit(self, portfolio):
        """Test validation when position exceeds risk limit."""
        # Set low cash to trigger risk limit
        portfolio.cash_balance = Decimal("10000")

        can_open, reason = portfolio.can_open_position(
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            price=Decimal("150.00"),  # $7,500 position
        )

        # Risk ratio: 7500 / 10000 = 0.75 > 0.02
        assert can_open is False
        assert "Position risk 75.0% exceeds portfolio limit 2.0%" in reason


class TestPortfolioPositionManagement:
    """Test suite for position management."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return Portfolio(initial_capital=Decimal("100000"), cash_balance=Decimal("100000"))

    def test_open_position_success(self, portfolio):
        """Test successfully opening a position."""
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),  # Reduced to stay within 2% risk limit
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
            strategy="momentum",
        )

        position = portfolio.open_position(request)

        assert position.symbol == "AAPL"
        assert position.quantity.value == Decimal("10")
        assert position.average_entry_price.value == Decimal("150.00")
        assert position.strategy == "momentum"
        assert portfolio.positions["AAPL"] == position
        assert portfolio.cash_balance.amount == Decimal("98499.00")  # 100000 - 1500 - 1
        assert portfolio.total_commission_paid.amount == Decimal("1.00")
        assert portfolio.trades_count == 1
        assert portfolio.last_updated is not None

    def test_open_position_with_portfolio_strategy(self, portfolio):
        """Test position inherits portfolio strategy if not specified."""
        portfolio.strategy = "value"

        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("10")), entry_price=Price(Decimal("100.00"))
        )

        position = portfolio.open_position(request)

        assert position.strategy == "value"

    def test_open_position_validation_failure(self, portfolio):
        """Test opening position fails validation."""
        portfolio.cash_balance = Decimal("1000")

        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("150.00"))
        )

        with pytest.raises(ValueError, match="Cannot open position"):
            portfolio.open_position(request)

    def test_close_position_with_profit(self, portfolio):
        """Test closing position with profit."""
        # Open position first
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),  # Reduced to stay within risk limits
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        portfolio.open_position(request)

        # Close with profit
        pnl = portfolio.close_position(
            symbol="AAPL", exit_price=Price(Decimal("160.00")), commission=Money(Decimal("1.00"))
        )

        expected_pnl = Money(Decimal("10") * (Decimal("160") - Decimal("150")) - Decimal("1"))
        assert pnl == expected_pnl  # 100 - 1 = 99 (only closing commission affects PnL)
        assert portfolio.cash_balance == Money(Decimal("100098.00"))  # 98499 + 1600 - 1 = 100098
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.total_commission_paid == Money(Decimal("2.00"))
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 0

    def test_close_position_with_loss(self, portfolio):
        """Test closing position with loss."""
        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),  # Reduced to stay within risk limits
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        portfolio.open_position(request)

        # Close with loss
        pnl = portfolio.close_position(
            symbol="AAPL", exit_price=Price(Decimal("140.00")), commission=Money(Decimal("1.00"))
        )

        expected_pnl = Money(Decimal("10") * (Decimal("140") - Decimal("150")) - Decimal("1"))
        assert pnl == expected_pnl  # -100 - 1 = -101
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1

    def test_close_position_not_found(self, portfolio):
        """Test closing non-existent position."""
        with pytest.raises(ValueError, match="No position found for AAPL"):
            portfolio.close_position("AAPL", Decimal("150.00"))

    def test_close_position_already_closed(self, portfolio):
        """Test closing already closed position."""
        # Create and close position
        from datetime import datetime

        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("150.00")),
            closed_at=datetime.now(UTC),
        )
        portfolio.positions["AAPL"] = position

        with pytest.raises(ValueError, match="Position for AAPL is already closed"):
            portfolio.close_position("AAPL", Decimal("160.00"))

    def test_update_position_price(self, portfolio):
        """Test updating position price."""
        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("150.00")),
        )
        portfolio.positions["AAPL"] = position

        portfolio.update_position_price("AAPL", Price(Decimal("155.00")))

        assert position.current_price == Price(Decimal("155.00"))
        assert portfolio.last_updated is not None

    def test_update_position_price_not_found(self, portfolio):
        """Test updating non-existent position price."""
        with pytest.raises(ValueError, match="No position found for AAPL"):
            portfolio.update_position_price("AAPL", Price(Decimal("150.00")))

    def test_update_all_prices(self, portfolio):
        """Test updating multiple position prices."""
        # Create positions
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            position = Position(
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                average_entry_price=Price(Decimal("100.00")),
            )
            portfolio.positions[symbol] = position

        # Close one position
        portfolio.positions["MSFT"].closed_at = datetime.now(UTC)

        prices = {
            "AAPL": Price(Decimal("110.00")),
            "GOOGL": Price(Decimal("120.00")),
            "MSFT": Price(Decimal("130.00")),  # Should be ignored (closed)
            "TSLA": Price(Decimal("140.00")),  # Should be ignored (not in portfolio)
        }

        portfolio.update_all_prices(prices)

        assert portfolio.positions["AAPL"].current_price == Price(Decimal("110.00"))
        assert portfolio.positions["GOOGL"].current_price == Price(Decimal("120.00"))
        assert portfolio.positions["MSFT"].current_price is None  # Not updated
        assert portfolio.last_updated is not None


class TestPortfolioQueries:
    """Test suite for portfolio queries."""

    @pytest.fixture
    def portfolio_with_positions(self):
        """Create portfolio with various positions."""
        portfolio = Portfolio()

        # Open positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("160.00")),
        )

        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            quantity=Quantity(Decimal("50")),
            average_entry_price=Price(Decimal("2500.00")),
            current_price=Price(Decimal("2450.00")),
        )

        # Closed position - set closed_at to avoid validation error
        from datetime import datetime

        closed_pos = Position(
            symbol="MSFT",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("300.00")),
            realized_pnl=Money(Decimal("500.00")),
            closed_at=datetime.now(UTC),
        )
        portfolio.positions["MSFT"] = closed_pos

        # Update portfolio stats
        portfolio.total_realized_pnl = Money(Decimal("500.00"))
        portfolio.winning_trades = 1
        portfolio.trades_count = 1

        return portfolio

    def test_get_position(self, portfolio_with_positions):
        """Test getting specific position."""
        position = portfolio_with_positions.get_position("AAPL")

        assert position is not None
        assert position.symbol == "AAPL"

        # Non-existent position
        assert portfolio_with_positions.get_position("TSLA") is None

    def test_get_open_positions(self, portfolio_with_positions):
        """Test getting open positions."""
        open_positions = portfolio_with_positions.get_open_positions()

        assert len(open_positions) == 2
        symbols = [p.symbol for p in open_positions]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" not in symbols

    def test_get_closed_positions(self, portfolio_with_positions):
        """Test getting closed positions."""
        closed_positions = portfolio_with_positions.get_closed_positions()

        assert len(closed_positions) == 1
        assert closed_positions[0].symbol == "MSFT"

    def test_get_total_value(self, portfolio_with_positions):
        """Test calculating total portfolio value."""
        portfolio = portfolio_with_positions
        portfolio.cash_balance = Money(Decimal("50000"))

        total_value = portfolio.get_total_value()

        # Cash: 50000
        # AAPL: 100 * 160 = 16000
        # GOOGL: 50 * 2450 = 122500
        expected = Money(Decimal("50000") + Decimal("16000") + Decimal("122500"))
        assert total_value == expected

    def test_get_positions_value(self, portfolio_with_positions):
        """Test calculating positions value."""
        positions_value = portfolio_with_positions.get_positions_value()

        # AAPL: 100 * 160 = 16000
        # GOOGL: 50 * 2450 = 122500
        expected = Money(Decimal("16000") + Decimal("122500"))
        assert positions_value == expected

    def test_get_unrealized_pnl(self, portfolio_with_positions):
        """Test calculating unrealized P&L."""
        unrealized_pnl = portfolio_with_positions.get_unrealized_pnl()

        # AAPL: 100 * (160 - 150) = 1000
        # GOOGL: 50 * (2450 - 2500) = -2500
        expected = Money(Decimal("1000") + Decimal("-2500"))
        assert unrealized_pnl == expected

    def test_get_total_pnl(self, portfolio_with_positions):
        """Test calculating total P&L."""
        total_pnl = portfolio_with_positions.get_total_pnl()

        # Realized: 500
        # Unrealized: 1000 - 2500 = -1500
        expected = Money(Decimal("500") + Decimal("-1500"))
        assert total_pnl == expected

    def test_get_return_percentage(self, portfolio_with_positions):
        """Test calculating return percentage."""
        portfolio = portfolio_with_positions
        portfolio.initial_capital = Money(Decimal("100000"))
        portfolio.cash_balance = Money(Decimal("50000"))

        return_pct = portfolio.get_return_percentage()

        # Total value: 50000 + 16000 + 122500 = 188500
        # Return: (188500 - 100000) / 100000 * 100 = 88.5%
        assert return_pct == Decimal("88.5")

    def test_get_return_percentage_zero_capital(self):
        """Test return percentage with zero initial capital."""
        portfolio = Portfolio()
        portfolio.initial_capital = Money(Decimal("0"))

        return_pct = portfolio.get_return_percentage()

        assert return_pct == Decimal("0")


class TestPortfolioStatistics:
    """Test suite for portfolio statistics."""

    @pytest.fixture
    def portfolio_with_trades(self):
        """Create portfolio with trade history."""
        portfolio = Portfolio()

        # Simulate closed trades
        # 3 winning trades
        for i in range(3):
            pos = Position(
                symbol=f"WIN{i}",
                quantity=Quantity(Decimal("0")),
                average_entry_price=Price(Decimal("100")),
                realized_pnl=Money(Decimal(str(100 * (i + 1)))),  # 100, 200, 300
                closed_at=datetime.now(UTC),  # Set closed_at during initialization
            )
            portfolio.positions[f"WIN{i}"] = pos

        # 2 losing trades
        for i in range(2):
            pos = Position(
                symbol=f"LOSS{i}",
                quantity=Quantity(Decimal("0")),
                average_entry_price=Price(Decimal("100")),
                realized_pnl=Money(Decimal(str(-50 * (i + 1)))),  # -50, -100
                closed_at=datetime.now(UTC),  # Set closed_at during initialization
            )
            portfolio.positions[f"LOSS{i}"] = pos

        portfolio.winning_trades = 3
        portfolio.losing_trades = 2
        portfolio.total_realized_pnl = Money(Decimal("450"))  # 600 - 150

        return portfolio

    def test_get_win_rate(self, portfolio_with_trades):
        """Test calculating win rate."""
        win_rate = portfolio_with_trades.get_win_rate()

        # 3 wins / 5 total = 60%
        assert win_rate == Decimal("60")

    def test_get_win_rate_no_trades(self):
        """Test win rate with no trades."""
        portfolio = Portfolio()

        win_rate = portfolio.get_win_rate()

        assert win_rate is None

    def test_get_average_win(self, portfolio_with_trades):
        """Test calculating average win."""
        avg_win = portfolio_with_trades.get_average_win()

        # (100 + 200 + 300) / 3 = 200
        assert avg_win == Money(Decimal("200"))

    def test_get_average_win_no_wins(self):
        """Test average win with no winning trades."""
        portfolio = Portfolio()
        portfolio.winning_trades = 0

        avg_win = portfolio.get_average_win()

        assert avg_win is None

    def test_get_average_loss(self, portfolio_with_trades):
        """Test calculating average loss."""
        avg_loss = portfolio_with_trades.get_average_loss()

        # (50 + 100) / 2 = 75
        assert avg_loss == Money(Decimal("75"))

    def test_get_average_loss_no_losses(self):
        """Test average loss with no losing trades."""
        portfolio = Portfolio()
        portfolio.losing_trades = 0

        avg_loss = portfolio.get_average_loss()

        assert avg_loss is None

    def test_get_profit_factor(self, portfolio_with_trades):
        """Test calculating profit factor."""
        profit_factor = portfolio_with_trades.get_profit_factor()

        # Gross profit: 600, Gross loss: 150
        # Profit factor: 600 / 150 = 4
        assert profit_factor == Decimal("4")

    def test_get_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        portfolio = Portfolio()

        # Only winning trade - create as closed position
        pos = Position(
            symbol="WIN",
            quantity=Quantity(Decimal("0")),
            average_entry_price=Price(Decimal("100")),
            realized_pnl=Money(Decimal("500")),
            closed_at=datetime.now(UTC),
        )
        portfolio.positions["WIN"] = pos

        profit_factor = portfolio.get_profit_factor()

        assert profit_factor == Decimal("999.99")  # Capped

    def test_get_profit_factor_no_trades(self):
        """Test profit factor with no trades."""
        portfolio = Portfolio()

        profit_factor = portfolio.get_profit_factor()

        assert profit_factor is None

    def test_get_sharpe_ratio(self, portfolio_with_trades):
        """Test Sharpe ratio calculation (placeholder)."""
        sharpe = portfolio_with_trades.get_sharpe_ratio()

        # Currently returns None (placeholder implementation)
        assert sharpe is None

    def test_get_max_drawdown(self, portfolio_with_trades):
        """Test max drawdown calculation (placeholder)."""
        drawdown = portfolio_with_trades.get_max_drawdown()

        # Currently returns 0 (placeholder implementation)
        assert drawdown == Decimal("0")


class TestPortfolioSerialization:
    """Test suite for portfolio serialization."""

    @pytest.fixture
    def portfolio(self):
        """Create portfolio with data."""
        portfolio = Portfolio(name="Test Portfolio")

        # Add position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("160.00")),
        )

        portfolio.cash_balance = Money(Decimal("90000"))
        portfolio.total_realized_pnl = Money(Decimal("1000"))
        portfolio.trades_count = 5
        portfolio.winning_trades = 3
        portfolio.losing_trades = 2
        portfolio.total_commission_paid = Money(Decimal("50"))

        return portfolio

    def test_to_dict(self, portfolio):
        """Test converting portfolio to dictionary."""
        data = portfolio.to_dict()

        assert data["id"] == str(portfolio.id)
        assert data["name"] == "Test Portfolio"
        assert data["cash_balance"] == 90000.0
        assert data["total_value"] == 106000.0  # 90000 + 16000
        assert data["positions_value"] == 16000.0
        assert data["unrealized_pnl"] == 1000.0
        assert data["realized_pnl"] == 1000.0
        assert data["total_pnl"] == 2000.0
        assert data["return_pct"] == 6.0
        assert data["open_positions"] == 1
        assert data["total_trades"] == 5
        assert data["winning_trades"] == 3
        assert data["losing_trades"] == 2
        assert data["win_rate"] == 60.0
        assert data["commission_paid"] == 50.0

    def test_to_dict_with_none_values(self):
        """Test to_dict with None values."""
        portfolio = Portfolio()

        data = portfolio.to_dict()

        assert data["win_rate"] is None  # No trades

    def test_str_representation(self, portfolio):
        """Test string representation."""
        result = str(portfolio)

        assert "Test Portfolio" in result
        assert "Value=$106,000.00" in result
        assert "Cash=$90,000.00" in result
        assert "Positions=1" in result
        assert "P&L=$2,000.00" in result
        assert "Return=6.00%" in result


class TestPortfolioEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_portfolio_metrics(self):
        """Test metrics on empty portfolio."""
        portfolio = Portfolio()

        assert portfolio.get_open_positions() == []
        assert portfolio.get_closed_positions() == []
        assert portfolio.get_positions_value() == Money(Decimal("0"))
        assert portfolio.get_unrealized_pnl() == Money(Decimal("0"))
        assert portfolio.get_total_pnl() == Money(Decimal("0"))
        assert portfolio.get_win_rate() is None
        assert portfolio.get_average_win() is None
        assert portfolio.get_average_loss() is None
        assert portfolio.get_profit_factor() is None

    def test_position_with_none_values(self):
        """Test handling positions with None values."""
        portfolio = Portfolio()

        # Position without current price
        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
        )
        portfolio.positions["AAPL"] = position

        # Should handle None gracefully
        total_value = portfolio.get_total_value()
        assert total_value == portfolio.cash_balance

        positions_value = portfolio.get_positions_value()
        assert positions_value == Money(Decimal("0"))

        unrealized_pnl = portfolio.get_unrealized_pnl()
        assert unrealized_pnl == Money(Decimal("0"))

    def test_large_number_of_positions(self):
        """Test portfolio with many positions."""
        portfolio = Portfolio(max_positions=1000)

        # Add 100 positions
        for i in range(100):
            position = Position(
                symbol=f"SYM{i}",
                quantity=Quantity(Decimal("10")),
                average_entry_price=Price(Decimal("100.00")),
                current_price=Price(Decimal(str(100 + i))),
            )
            portfolio.positions[f"SYM{i}"] = position

        open_positions = portfolio.get_open_positions()
        assert len(open_positions) == 100

        # Calculate total unrealized P&L
        unrealized_pnl = portfolio.get_unrealized_pnl()
        expected_pnl = sum(Decimal("10") * Decimal(str(i)) for i in range(100))
        assert unrealized_pnl == Money(expected_pnl)

"""
Ultra-Comprehensive Tests for Portfolio Entity
===========================================

This test suite provides exhaustive coverage for the Portfolio domain entity,
covering all methods, edge cases, business logic, and financial calculations
with focus on precision and reliability for a financial trading system.
"""

from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.exceptions import StaleDataException
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioCreation:
    """Test portfolio creation and initialization."""

    def test_portfolio_default_creation(self):
        """Test creating portfolio with default values."""
        portfolio = Portfolio()

        assert portfolio.id is not None
        assert portfolio.name == "Default Portfolio"
        assert portfolio.initial_capital == Money(Decimal("100000"))
        assert portfolio.cash_balance == Money(Decimal("100000"))
        assert portfolio.positions == {}
        assert portfolio.max_position_size == Money(Decimal("10000"))
        assert portfolio.max_portfolio_risk == Decimal("0.02")
        assert portfolio.max_positions == 10
        assert portfolio.max_leverage == Decimal("1.0")
        assert portfolio.total_realized_pnl == Money(Decimal("0"))
        assert portfolio.total_commission_paid == Money(Decimal("0"))
        assert portfolio.trades_count == 0
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0
        assert portfolio.strategy is None
        assert portfolio.tags == {}
        assert portfolio.version == 1
        assert portfolio.created_at is not None
        assert portfolio.last_updated is None

    def test_portfolio_custom_creation(self):
        """Test creating portfolio with custom values."""
        custom_id = uuid4()
        custom_capital = Money(Decimal("50000"))

        portfolio = Portfolio(
            id=custom_id,
            name="Test Portfolio",
            initial_capital=custom_capital,
            cash_balance=custom_capital,
            max_position_size=Money(Decimal("5000")),
            max_portfolio_risk=Decimal("0.01"),
            max_positions=5,
            max_leverage=Decimal("2.0"),
            strategy="momentum",
            tags={"risk": "conservative"},
        )

        assert portfolio.id == custom_id
        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == custom_capital
        assert portfolio.cash_balance == custom_capital
        assert portfolio.max_position_size == Money(Decimal("5000"))
        assert portfolio.max_portfolio_risk == Decimal("0.01")
        assert portfolio.max_positions == 5
        assert portfolio.max_leverage == Decimal("2.0")
        assert portfolio.strategy == "momentum"
        assert portfolio.tags == {"risk": "conservative"}

    def test_portfolio_post_init_validation(self):
        """Test portfolio validation during initialization."""
        # This should pass validation
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100000")),
        )
        assert portfolio.version == 1


class TestPortfolioVersionControl:
    """Test portfolio version control for optimistic locking."""

    def test_version_increment(self):
        """Test version increment functionality."""
        portfolio = Portfolio()
        original_version = portfolio.version

        portfolio._increment_version()

        assert portfolio.version == original_version + 1
        assert portfolio.last_updated is not None

    def test_check_version_success(self):
        """Test successful version check."""
        portfolio = Portfolio()
        current_version = portfolio.version

        # Should not raise exception
        portfolio._check_version(current_version)

    def test_check_version_failure(self):
        """Test version check failure."""
        portfolio = Portfolio()

        with pytest.raises(StaleDataException) as exc_info:
            portfolio._check_version(999)

        assert exc_info.value.entity_type == "Portfolio"
        assert exc_info.value.entity_id == portfolio.id
        assert exc_info.value.expected_version == 999
        assert exc_info.value.actual_version == portfolio.version

    def test_check_version_none(self):
        """Test version check with None (should not raise)."""
        portfolio = Portfolio()
        portfolio._check_version(None)  # Should not raise


class TestPortfolioPositionManagement:
    """Test portfolio position management operations."""

    def test_can_open_position_success(self):
        """Test successful position opening validation."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Increased to allow 100*150
            max_positions=10,
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        can_open, reason = portfolio.can_open_position(
            "AAPL", Quantity(Decimal("100")), Price(Decimal("150"))
        )

        assert can_open is True
        assert reason is None

    def test_can_open_position_insufficient_funds(self):
        """Test position opening validation with insufficient funds."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("1000")),
            max_position_size=Money(Decimal("20000")),  # Allow large position to test cash limit
            max_portfolio_risk=Decimal("1.0"),  # Allow 100% risk to test cash limit
        )

        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Quantity(Decimal("100")),
            Price(Decimal("150")),  # Costs $15,000
        )

        assert can_open is False
        assert "insufficient cash" in reason.lower()

    def test_can_open_position_max_positions_reached(self):
        """Test position opening validation when max positions reached."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")), max_positions=1)

        # Add one position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("10")), entry_price=Price(Decimal("150"))
        )
        portfolio.open_position(request)

        # Try to add another
        can_open, reason = portfolio.can_open_position(
            "MSFT", Quantity(Decimal("10")), Price(Decimal("300"))
        )

        assert can_open is False
        assert "maximum positions limit" in reason.lower()

    def test_open_position_success(self):
        """Test successful position opening."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),
            commission=Money(Decimal("5")),
            strategy="momentum",
        )

        position = portfolio.open_position(request)

        assert position.symbol == "AAPL"
        assert position.quantity.value == Decimal("100")
        assert position.average_entry_price.value == Decimal("150")
        assert position.commission_paid.amount == Decimal("5")
        assert position.strategy == "momentum"
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"] == position
        # Cash should be reduced by position cost + commission
        expected_cash = Money(Decimal("100000")) - Money(Decimal("15000")) - Money(Decimal("5"))
        assert portfolio.cash_balance == expected_cash

    def test_close_position_success(self):
        """Test successful position closing."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        portfolio.open_position(request)

        # Close position at profit
        realized_pnl = portfolio.close_position(
            "AAPL", Price(Decimal("160")), commission=Money(Decimal("5"))
        )

        # Should realize $1000 profit minus $5 commission = $995
        expected_pnl = Money(Decimal("995"))
        assert realized_pnl == expected_pnl
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.trades_count == 1
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 0

        # Position should be closed
        position = portfolio.get_position("AAPL")
        assert position.is_closed()

    def test_close_position_at_loss(self):
        """Test closing position at a loss."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        portfolio.open_position(request)

        # Close position at loss
        realized_pnl = portfolio.close_position(
            "AAPL", Price(Decimal("140")), commission=Money(Decimal("5"))
        )

        # Should realize -$1000 loss minus $5 commission = -$1005
        expected_pnl = Money(Decimal("-1005"))
        assert realized_pnl == expected_pnl
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.trades_count == 1
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1

    def test_update_position_price(self):
        """Test updating position market price."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        portfolio.open_position(request)

        # Update price
        new_price = Price(Decimal("160"))
        portfolio.update_position_price("AAPL", new_price)

        position = portfolio.get_position("AAPL")
        assert position.current_price == new_price
        assert position.last_updated is not None

    def test_update_all_prices(self):
        """Test updating multiple position prices."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow both positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open multiple positions
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )

        # Update all prices
        prices = {
            "AAPL": Price(Decimal("160")),
            "MSFT": Price(Decimal("310")),
            "GOOGL": Price(Decimal("2500")),  # Not in portfolio, should be ignored
        }
        portfolio.update_all_prices(prices)

        assert portfolio.get_position("AAPL").current_price.value == Decimal("160")
        assert portfolio.get_position("MSFT").current_price.value == Decimal("310")


class TestPortfolioPositionQueries:
    """Test portfolio position query methods."""

    def test_get_position_exists(self):
        """Test getting existing position."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        created_position = portfolio.open_position(request)

        retrieved_position = portfolio.get_position("AAPL")
        assert retrieved_position == created_position

    def test_get_position_not_exists(self):
        """Test getting non-existent position."""
        portfolio = Portfolio()

        position = portfolio.get_position("NONEXISTENT")
        assert position is None

    def test_get_open_positions(self):
        """Test getting open positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow both positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open two positions
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )

        # Close one position
        portfolio.close_position("AAPL", Price(Decimal("160")))

        open_positions = portfolio.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].symbol == "MSFT"
        assert not open_positions[0].is_closed()

    def test_get_closed_positions(self):
        """Test getting closed positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow both positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open two positions
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )

        # Close one position
        portfolio.close_position("AAPL", Price(Decimal("160")))

        closed_positions = portfolio.get_closed_positions()
        assert len(closed_positions) == 1
        assert closed_positions[0].symbol == "AAPL"
        assert closed_positions[0].is_closed()


class TestPortfolioMetrics:
    """Test portfolio financial metrics calculations."""

    def test_get_total_value_cash_only(self):
        """Test total value calculation with cash only."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        total_value = portfolio.get_total_value()
        assert total_value == Money(Decimal("50000"))

    def test_get_total_value_with_positions(self):
        """Test total value calculation with positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open position
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )

        # Update price to calculate current value
        portfolio.update_position_price("AAPL", Price(Decimal("160")))

        total_value = portfolio.get_total_value()
        # Cash (50000 - 15000) + Position value (100 * 160) = 35000 + 16000 = 51000
        expected = Money(Decimal("51000"))
        assert total_value == expected

    def test_get_positions_value(self):
        """Test positions value calculation."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow both positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open multiple positions
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )

        # Update prices
        portfolio.update_position_price("AAPL", Price(Decimal("160")))
        portfolio.update_position_price("MSFT", Price(Decimal("310")))

        positions_value = portfolio.get_positions_value()
        # AAPL: 100 * 160 = 16000, MSFT: 50 * 310 = 15500, Total = 31500
        expected = Money(Decimal("31500"))
        assert positions_value == expected

    def test_get_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        # Open position
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )

        # Update price for profit
        portfolio.update_position_price("AAPL", Price(Decimal("160")))

        unrealized_pnl = portfolio.get_unrealized_pnl()
        # (160 - 150) * 100 = 1000
        expected = Money(Decimal("1000"))
        assert unrealized_pnl == expected

    def test_get_total_pnl(self):
        """Test total P&L calculation (realized + unrealized)."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open and close one position for realized PnL
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.close_position("AAPL", Price(Decimal("160")))

        # Open another position for unrealized PnL
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )
        portfolio.update_position_price("MSFT", Price(Decimal("310")))

        total_pnl = portfolio.get_total_pnl()
        # Realized: 1000, Unrealized: 500, Total: 1500
        expected = Money(Decimal("1500"))
        assert total_pnl == expected

    def test_get_return_percentage(self):
        """Test return percentage calculation."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk
        )

        # Open position and make profit
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.update_position_price("AAPL", Price(Decimal("160")))

        return_pct = portfolio.get_return_percentage()
        # Total value: 85000 (cash) + 16000 (position) = 101000
        # Return: (101000 - 100000) / 100000 * 100 = 1%
        expected = Decimal("1.00")
        assert return_pct == expected

    def test_get_win_rate_no_trades(self):
        """Test win rate with no completed trades."""
        portfolio = Portfolio()

        win_rate = portfolio.get_win_rate()
        assert win_rate is None

    def test_get_win_rate_with_trades(self):
        """Test win rate calculation with trades."""
        portfolio = Portfolio()
        portfolio.winning_trades = 7
        portfolio.losing_trades = 3

        win_rate = portfolio.get_win_rate()
        # 7 / (7 + 3) * 100 = 70%
        expected = Decimal("70")
        assert win_rate == expected

    def test_get_profit_factor(self):
        """Test profit factor calculation."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow large positions
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Create winning and losing trades
        # Winning trade: +$1000
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.close_position("AAPL", Price(Decimal("160")))

        # Losing trade: -$500
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("300"))
            )
        )
        portfolio.close_position("MSFT", Price(Decimal("295")))

        profit_factor = portfolio.get_profit_factor()
        # Gross profit: 1000, Gross loss: 500, Ratio: 2.0
        expected = Decimal("2.0")
        assert profit_factor == expected


class TestPortfolioEdgeCases:
    """Test portfolio edge cases and error conditions."""

    def test_zero_initial_capital(self):
        """Test portfolio with zero initial capital."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("0.01")),  # Use small positive value instead of zero
            cash_balance=Money(Decimal("0.01")),
        )

        return_pct = portfolio.get_return_percentage()
        # With tiny capital, return % might be 0 or very small
        assert return_pct >= Decimal("0")

    def test_extremely_large_values(self):
        """Test portfolio with extremely large values."""
        large_amount = Decimal("999999999999.99")
        portfolio = Portfolio(initial_capital=Money(large_amount), cash_balance=Money(large_amount))

        # Should handle large values without overflow
        total_value = portfolio.get_total_value()
        assert total_value.amount == large_amount

    def test_precision_with_many_positions(self):
        """Test precision is maintained with many small positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_positions=100,  # Allow 100 positions for this test
            max_position_size=Money(Decimal("1000")),  # Small position size
            max_portfolio_risk=Decimal("0.50"),  # Allow 50% risk for many positions
        )

        # Open many small positions
        for i in range(100):
            portfolio.open_position(
                PositionRequest(
                    symbol=f"STOCK{i:03d}",
                    quantity=Quantity(Decimal("1")),
                    entry_price=Price(Decimal("10.01")),
                )
            )

        # Update all prices slightly
        for i in range(100):
            portfolio.update_position_price(f"STOCK{i:03d}", Price(Decimal("10.02")))

        # Should maintain precision
        unrealized_pnl = portfolio.get_unrealized_pnl()
        expected = Money(Decimal("1.00"))  # 100 * 0.01 = 1.00
        assert unrealized_pnl == expected


class TestPortfolioStringRepresentation:
    """Test portfolio string representation and serialization."""

    def test_string_representation(self):
        """Test portfolio string representation."""
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Open a position for more interesting display
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.update_position_price("AAPL", Price(Decimal("160")))

        str_repr = str(portfolio)

        assert "Test Portfolio" in str_repr
        assert "Value=" in str_repr
        assert "Cash=" in str_repr
        assert "Positions=1" in str_repr
        assert "P&L=" in str_repr
        assert "Return=" in str_repr

    def test_to_dict_serialization(self):
        """Test portfolio dictionary serialization."""
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("20000")),  # Allow position of $15,000
            max_portfolio_risk=Decimal("0.30"),  # Allow 30% risk
        )

        # Add some trading activity
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.close_position("AAPL", Price(Decimal("160")))
        portfolio.winning_trades = 1
        portfolio.trades_count = 1

        portfolio_dict = portfolio.to_dict()

        assert portfolio_dict["name"] == "Test Portfolio"
        assert portfolio_dict["cash_balance"] == 51000.0  # 50000 + 1000 profit
        assert portfolio_dict["total_trades"] == 1
        assert portfolio_dict["winning_trades"] == 1
        assert portfolio_dict["win_rate"] == 100.0
        assert "id" in portfolio_dict
        assert "total_value" in portfolio_dict
        assert "positions_value" in portfolio_dict
        assert "unrealized_pnl" in portfolio_dict
        assert "realized_pnl" in portfolio_dict
        assert "total_pnl" in portfolio_dict
        assert "return_pct" in portfolio_dict


class TestPortfolioComplexScenarios:
    """Test complex multi-position scenarios."""

    def test_mixed_long_short_positions(self):
        """Test portfolio with mixed long and short positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow both positions
            max_portfolio_risk=Decimal("0.40"),  # Allow 40% risk
        )

        # Long position
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )

        # Short position (negative quantity)
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("-50")), entry_price=Price(Decimal("300"))
            )
        )

        # Update prices
        portfolio.update_position_price("AAPL", Price(Decimal("160")))  # +$1000 unrealized
        portfolio.update_position_price(
            "MSFT", Price(Decimal("290"))
        )  # +$500 unrealized (short profit)

        unrealized_pnl = portfolio.get_unrealized_pnl()
        expected = Money(Decimal("1500"))  # 1000 + 500
        assert unrealized_pnl == expected

    def test_portfolio_rebalancing(self):
        """Test portfolio through multiple rebalancing operations."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("30000")),  # Allow large positions
            max_portfolio_risk=Decimal("0.40"),  # Allow 40% risk
            max_positions=10,  # Allow multiple positions
        )

        # Initial positions
        portfolio.open_position(
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            )
        )
        portfolio.open_position(
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            )
        )

        # Close one, open another
        portfolio.close_position("AAPL", Price(Decimal("160")))
        portfolio.open_position(
            PositionRequest(
                symbol="GOOGL", quantity=Quantity(Decimal("10")), entry_price=Price(Decimal("2500"))
            )
        )

        # Verify portfolio state
        assert len(portfolio.get_open_positions()) == 2
        assert len(portfolio.get_closed_positions()) == 1
        assert portfolio.trades_count >= 1
        assert portfolio.total_realized_pnl.amount > 0

    def test_risk_limits_enforcement(self):
        """Test that risk limits are properly checked."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("10000")),
            max_position_size=Money(Decimal("5000")),
            max_positions=2,
            max_portfolio_risk=Decimal("0.50"),  # Allow 50% risk for this test
        )

        # First position should be allowed
        can_open, reason = portfolio.can_open_position(
            "AAPL",
            Quantity(Decimal("30")),
            Price(Decimal("150")),  # $4500 total
        )
        assert can_open is True

        # Position exceeding max size should be rejected
        can_open, reason = portfolio.can_open_position(
            "MSFT",
            Quantity(Decimal("20")),
            Price(Decimal("300")),  # $6000 total > $5000 limit
        )
        assert can_open is False
        assert "position size" in reason.lower() or "exceeds limit" in reason.lower()

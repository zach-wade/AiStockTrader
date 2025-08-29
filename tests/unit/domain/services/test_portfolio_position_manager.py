"""
Comprehensive Tests for Portfolio Position Manager Service
========================================================

Tests for the PortfolioPositionManager domain service that handles
all position lifecycle management operations.
"""

from decimal import Decimal

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.services.portfolio_position_manager import PortfolioPositionManager
from src.domain.value_objects import Money, Price, Quantity


class TestPositionOpening:
    """Test position opening operations."""

    def test_open_position_success(self):
        """Test successful position opening."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),
            commission=Money(Decimal("5")),
            strategy="momentum",
        )

        position = PortfolioPositionManager.open_position(portfolio, request)

        # Verify position creation
        assert position.symbol == "AAPL"
        assert position.quantity.value == Decimal("100")
        assert position.average_entry_price.value == Decimal("150")
        assert position.commission_paid.amount == Decimal("5")
        assert position.strategy == "momentum"

        # Verify portfolio updates
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"] == position
        expected_cash = Money(Decimal("100000")) - Money(Decimal("15000")) - Money(Decimal("5"))
        assert portfolio.cash_balance == expected_cash
        assert portfolio.total_commission_paid.amount == Decimal("5")
        assert portfolio.trades_count == 1
        assert portfolio.version == 2  # Should increment

    def test_open_position_uses_portfolio_strategy(self):
        """Test opening position uses portfolio strategy when not specified."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")), strategy="value_investing")

        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position = PortfolioPositionManager.open_position(portfolio, request)
        assert position.strategy == "value_investing"

    def test_open_position_validation_failure(self):
        """Test position opening with validation failure."""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000")))  # Insufficient cash

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),  # Costs $15,000
        )

        with pytest.raises(ValueError, match="Cannot open position"):
            PortfolioPositionManager.open_position(portfolio, request)

    def test_open_position_already_exists(self):
        """Test opening position when symbol already exists."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # First position
        request1 = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request1)

        # Try to open another for same symbol
        request2 = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("160"))
        )

        with pytest.raises(ValueError, match="Position already exists for AAPL"):
            PortfolioPositionManager.open_position(portfolio, request2)

    def test_open_position_insufficient_cash(self):
        """Test opening position with insufficient cash."""
        portfolio = Portfolio(cash_balance=Money(Decimal("5000")))

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),
            commission=Money(Decimal("10")),
        )

        with pytest.raises(ValueError, match="Insufficient cash"):
            PortfolioPositionManager.open_position(portfolio, request)

    def test_open_position_handles_value_objects(self):
        """Test opening position correctly handles value object attributes."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        request = PositionRequest(
            symbol="MSFT",
            quantity=Quantity(Decimal("50")),
            entry_price=Price(Decimal("300")),
            commission=Money(Decimal("7.50")),
        )

        position = PortfolioPositionManager.open_position(portfolio, request)

        # Verify value object attributes are handled correctly
        assert isinstance(position.quantity, Quantity)
        assert isinstance(position.average_entry_price, Price)
        assert isinstance(position.commission_paid, Money)

        # Verify cash calculation with value objects
        expected_cash = Money(Decimal("50000")) - Money(Decimal("15000")) - Money(Decimal("7.50"))
        assert portfolio.cash_balance == expected_cash


class TestPositionClosing:
    """Test position closing operations."""

    def test_close_position_success_profit(self):
        """Test successful position closing at profit."""
        portfolio = Portfolio(cash_balance=Money(Decimal("50000")))

        # Open position first
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        # Close at profit
        realized_pnl = PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("160")), commission=Money(Decimal("5"))
        )

        # Verify P&L calculation: 100 * (160 - 150) - 5 = 995
        expected_pnl = Money(Decimal("995"))
        assert realized_pnl == expected_pnl

        # Verify portfolio updates
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 0
        assert portfolio.total_commission_paid.amount == Decimal("5")  # Commission from closing

        # Verify cash balance: original - position cost + proceeds - commission
        # 50000 - 15000 + 16000 - 5 = 51000 - 5 = 50995
        expected_cash = Money(Decimal("50995"))
        assert portfolio.cash_balance == expected_cash

        # Position should be closed
        position = portfolio.get_position("AAPL")
        assert position.is_closed()

    def test_close_position_success_loss(self):
        """Test successful position closing at loss."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        # Close at loss
        realized_pnl = PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("140")), commission=Money(Decimal("5"))
        )

        # Verify P&L: 100 * (140 - 150) - 5 = -1005
        expected_pnl = Money(Decimal("-1005"))
        assert realized_pnl == expected_pnl

        # Verify portfolio updates
        assert portfolio.total_realized_pnl == expected_pnl
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1

    def test_close_position_breakeven(self):
        """Test closing position at breakeven."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        # Close at entry price with commission
        realized_pnl = PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("150")), commission=Money(Decimal("10"))
        )

        # Only commission loss: -10
        expected_pnl = Money(Decimal("-10"))
        assert realized_pnl == expected_pnl

        # Should count as loss due to commission
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 1

    def test_close_position_exactly_zero_pnl(self):
        """Test closing position with exactly zero P&L."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        # Close at same price with no commission
        realized_pnl = PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("150"))
        )

        assert realized_pnl.amount == Decimal("0")
        # Zero P&L shouldn't increment either win or loss count
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0

    def test_close_position_not_found(self):
        """Test closing non-existent position."""
        portfolio = Portfolio()

        with pytest.raises(ValueError, match="No position found for NONEXISTENT"):
            PortfolioPositionManager.close_position(portfolio, "NONEXISTENT", Price(Decimal("100")))

    def test_close_position_already_closed(self):
        """Test closing already closed position."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open and close position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)
        PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("160")))

        # Try to close again
        with pytest.raises(ValueError, match="Position for AAPL is already closed"):
            PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("170")))

    def test_close_short_position(self):
        """Test closing short position."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open short position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        # Cover at profit (price went down)
        realized_pnl = PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("140")), commission=Money(Decimal("5"))
        )

        # Short profit: 100 * (150 - 140) - 5 = 995
        expected_pnl = Money(Decimal("995"))
        assert realized_pnl == expected_pnl
        assert portfolio.winning_trades == 1


class TestPriceUpdating:
    """Test position price updating operations."""

    def test_update_position_price_success(self):
        """Test successful position price update."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        request = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request)

        original_version = portfolio.version

        # Update price
        new_price = Price(Decimal("160"))
        PortfolioPositionManager.update_position_price(portfolio, "AAPL", new_price)

        # Verify update
        position = portfolio.get_position("AAPL")
        assert position.current_price == new_price
        assert position.last_updated is not None
        assert portfolio.version == original_version + 1

    def test_update_position_price_not_found(self):
        """Test updating price for non-existent position."""
        portfolio = Portfolio()

        with pytest.raises(ValueError, match="No position found for NONEXISTENT"):
            PortfolioPositionManager.update_position_price(
                portfolio, "NONEXISTENT", Price(Decimal("100"))
            )

    def test_update_all_prices_success(self):
        """Test updating multiple position prices."""
        portfolio = Portfolio(cash_balance=Money(Decimal("200000")))

        # Open multiple positions
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            ),
        )
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            ),
        )

        original_version = portfolio.version

        # Update all prices
        prices = {
            "AAPL": Price(Decimal("160")),
            "MSFT": Price(Decimal("310")),
            "GOOGL": Price(Decimal("2500")),  # Not in portfolio
        }
        PortfolioPositionManager.update_all_prices(portfolio, prices)

        # Verify updates
        assert portfolio.get_position("AAPL").current_price.value == Decimal("160")
        assert portfolio.get_position("MSFT").current_price.value == Decimal("310")
        assert portfolio.version == original_version + 1

    def test_update_all_prices_no_positions(self):
        """Test updating prices when no positions exist."""
        portfolio = Portfolio()
        original_version = portfolio.version

        prices = {"AAPL": Price(Decimal("160"))}
        PortfolioPositionManager.update_all_prices(portfolio, prices)

        # Version should not change
        assert portfolio.version == original_version

    def test_update_all_prices_empty_dict(self):
        """Test updating with empty price dictionary."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            ),
        )

        original_version = portfolio.version

        # Empty prices
        PortfolioPositionManager.update_all_prices(portfolio, {})

        # Version should not change
        assert portfolio.version == original_version

    def test_update_all_prices_ignores_closed_positions(self):
        """Test that price updates ignore closed positions."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            ),
        )

        # Close position
        PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("160")))

        original_version = portfolio.version
        closed_position = portfolio.get_position("AAPL")
        original_price = closed_position.current_price

        # Try to update price of closed position
        prices = {"AAPL": Price(Decimal("170"))}
        PortfolioPositionManager.update_all_prices(portfolio, prices)

        # Price should not be updated for closed position, version unchanged
        assert closed_position.current_price == original_price
        assert portfolio.version == original_version

    def test_update_all_prices_partial_match(self):
        """Test updating prices where only some symbols match."""
        portfolio = Portfolio(cash_balance=Money(Decimal("200000")))

        # Open positions
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            ),
        )
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="MSFT", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("300"))
            ),
        )

        original_version = portfolio.version

        # Only update one position
        prices = {
            "AAPL": Price(Decimal("155")),
            "GOOGL": Price(Decimal("2500")),  # Not in portfolio
            "TSLA": Price(Decimal("800")),  # Not in portfolio
        }
        PortfolioPositionManager.update_all_prices(portfolio, prices)

        # Only AAPL should be updated
        assert portfolio.get_position("AAPL").current_price.value == Decimal("155")
        assert portfolio.get_position("MSFT").current_price is None  # Not updated
        assert portfolio.version == original_version + 1


class TestComplexScenarios:
    """Test complex position management scenarios."""

    def test_multiple_position_lifecycle(self):
        """Test complete lifecycle with multiple positions."""
        portfolio = Portfolio(cash_balance=Money(Decimal("500000")))

        # Open multiple positions
        positions_data = [
            ("AAPL", Decimal("100"), Decimal("150")),
            ("MSFT", Decimal("50"), Decimal("300")),
            ("GOOGL", Decimal("10"), Decimal("2500")),
        ]

        for symbol, qty, price in positions_data:
            PortfolioPositionManager.open_position(
                portfolio,
                PositionRequest(
                    symbol=symbol,
                    quantity=Quantity(qty),
                    entry_price=Price(price),
                    commission=Money(Decimal("5")),
                ),
            )

        assert len(portfolio.get_open_positions()) == 3
        assert portfolio.trades_count == 3
        assert portfolio.total_commission_paid.amount == Decimal("15")

        # Update all prices
        prices = {
            "AAPL": Price(Decimal("160")),
            "MSFT": Price(Decimal("320")),
            "GOOGL": Price(Decimal("2600")),
        }
        PortfolioPositionManager.update_all_prices(portfolio, prices)

        # Close some positions
        PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("165")))
        PortfolioPositionManager.close_position(portfolio, "MSFT", Price(Decimal("310")))

        assert len(portfolio.get_open_positions()) == 1
        assert len(portfolio.get_closed_positions()) == 2
        assert portfolio.winning_trades == 2
        assert portfolio.losing_trades == 0

    def test_position_reopening_after_close(self):
        """Test opening position for same symbol after closing previous one."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open and close first position
        request1 = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        PortfolioPositionManager.open_position(portfolio, request1)
        PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("160")))

        # Should be able to open new position for same symbol
        request2 = PositionRequest(
            symbol="AAPL", quantity=Quantity(Decimal("50")), entry_price=Price(Decimal("170"))
        )
        new_position = PortfolioPositionManager.open_position(portfolio, request2)

        # Should be new position object
        old_position = portfolio.positions["AAPL"]
        # The old position should still exist but be closed
        assert old_position.is_closed()
        # But the new position should now be in the portfolio
        assert new_position.quantity.value == Decimal("50")
        assert new_position.average_entry_price.value == Decimal("170")
        assert not new_position.is_closed()

    def test_commission_tracking_accuracy(self):
        """Test accurate commission tracking across operations."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position with commission
        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150")),
                commission=Money(Decimal("7.50")),
            ),
        )

        assert portfolio.total_commission_paid.amount == Decimal("7.50")

        # Close position with commission
        PortfolioPositionManager.close_position(
            portfolio, "AAPL", Price(Decimal("160")), commission=Money(Decimal("8.25"))
        )

        # Total commission should be sum of both
        assert portfolio.total_commission_paid.amount == Decimal("15.75")

        # Position should also track its commission
        position = portfolio.get_position("AAPL")
        assert position.commission_paid.amount == Decimal("15.75")

    def test_fractional_shares_handling(self):
        """Test handling of fractional share quantities."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        # Open position with fractional shares
        request = PositionRequest(
            symbol="BRK.A", quantity=Quantity(Decimal("0.5")), entry_price=Price(Decimal("500000"))
        )

        position = PortfolioPositionManager.open_position(portfolio, request)

        assert position.quantity.value == Decimal("0.5")
        assert position.average_entry_price.value == Decimal("500000")

        # Cash should be reduced by 0.5 * 500000 = 250000
        expected_cash = Money(Decimal("100000")) - Money(Decimal("250000"))
        assert portfolio.cash_balance == expected_cash

    def test_high_precision_calculations(self):
        """Test calculations maintain decimal precision."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000.123456")))

        request = PositionRequest(
            symbol="PRECISE",
            quantity=Quantity(Decimal("123.456789")),
            entry_price=Price(Decimal("98.765432")),
            commission=Money(Decimal("3.141593")),
        )

        position = PortfolioPositionManager.open_position(portfolio, request)

        # Verify precision is maintained
        position_cost = Decimal("123.456789") * Decimal("98.765432")
        total_cost = position_cost + Decimal("3.141593")
        expected_cash = Money(Decimal("100000.123456") - total_cost)

        assert portfolio.cash_balance == expected_cash
        assert position.commission_paid.amount == Decimal("3.141593")

    def test_version_increment_consistency(self):
        """Test that version increments are consistent across operations."""
        portfolio = Portfolio()
        initial_version = portfolio.version

        # Each operation should increment version
        portfolio.cash_balance = Money(Decimal("100000"))

        PortfolioPositionManager.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
            ),
        )
        assert portfolio.version == initial_version + 1

        PortfolioPositionManager.update_position_price(portfolio, "AAPL", Price(Decimal("155")))
        assert portfolio.version == initial_version + 2

        PortfolioPositionManager.close_position(portfolio, "AAPL", Price(Decimal("160")))
        assert portfolio.version == initial_version + 3

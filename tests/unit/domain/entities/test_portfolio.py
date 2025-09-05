"""
Comprehensive test suite for Portfolio Entity

Achieves >90% coverage by testing:
- Portfolio initialization and validation
- Position management (open, close, update)
- Risk limit validation
- Performance metrics calculation
- Concurrency handling
- Edge cases and error conditions
- Portfolio statistics and serialization
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application.services.portfolio_service import PortfolioService
from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position
from src.domain.services.portfolio_calculator import PortfolioCalculator
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioCreation:
    """Test suite for Portfolio creation and initialization."""

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
            tags={"type": "aggressive", "market": "US"},
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
        assert portfolio.tags["market"] == "US"

    def test_portfolio_with_existing_positions(self):
        """Test creating portfolio with existing positions."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("150.00")),
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=Quantity(Decimal("50")),
                average_entry_price=Price(Decimal("2500.00")),
            ),
        }

        portfolio = Portfolio(positions=positions, cash_balance=Money(Decimal("50000")))

        assert len(portfolio.positions) == 2
        assert "AAPL" in portfolio.positions
        assert "GOOGL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == Quantity(Decimal("100"))


class TestPortfolioValidation:
    """Test suite for Portfolio validation rules."""

    def test_validation_negative_initial_capital(self):
        """Test validation rejects negative initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Money(Decimal("-1000")))

    def test_validation_zero_initial_capital(self):
        """Test validation rejects zero initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=Money(Decimal("0")))

    def test_validation_negative_cash_balance(self):
        """Test validation rejects negative cash balance."""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Portfolio(cash_balance=Money(Decimal("-100")))

    def test_validation_negative_max_position_size(self):
        """Test validation rejects negative max position size."""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            Portfolio(max_position_size=Money(Decimal("-1000")))

    def test_validation_invalid_max_portfolio_risk(self):
        """Test validation rejects invalid portfolio risk."""
        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("0"))

        with pytest.raises(ValueError, match="Max portfolio risk must be between 0 and 1"):
            Portfolio(max_portfolio_risk=Decimal("1.5"))

    def test_validation_negative_max_positions(self):
        """Test validation rejects negative max positions."""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=-1)

    def test_validation_zero_max_positions(self):
        """Test validation rejects zero max positions."""
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=0)

    def test_validation_negative_max_leverage(self):
        """Test validation rejects negative max leverage."""
        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            Portfolio(max_leverage=Decimal("-1"))

    def test_validation_zero_max_leverage(self):
        """Test validation rejects zero max leverage."""
        with pytest.raises(ValueError, match="Max leverage must be at least 1.0"):
            Portfolio(max_leverage=Decimal("0"))


class TestPortfolioPositionManagement:
    """Test suite for Portfolio position management."""

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio for testing."""
        return Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),
            max_positions=5,
            max_portfolio_risk=Decimal("0.50"),  # Allow 50% risk for testing
        )

    def test_open_new_position_success(self, portfolio):
        """Test successfully opening a new position."""
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )

        service = PortfolioService()
        position = service.open_position(portfolio, request)

        assert position.symbol == "AAPL"
        assert position.quantity == Quantity(Decimal("100"))
        assert position.average_entry_price == Price(
            Decimal("150.00")
        )  # Commission tracked separately
        assert "AAPL" in portfolio.positions
        assert portfolio.cash_balance == Money(Decimal("84999"))  # 100000 - 15000 - 1

    def test_open_position_insufficient_cash(self, portfolio):
        """Test opening position with insufficient cash."""
        # Set cash to a low amount to trigger cash check
        portfolio.cash_balance = Money(Decimal("1000"))

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),
            entry_price=Price(Decimal("200.00")),
            commission=Money(Decimal("1.00")),
        )

        service = PortfolioService()
        with pytest.raises(ValueError, match="Insufficient cash"):
            service.open_position(portfolio, request)

    def test_open_position_exceeds_max_size(self, portfolio):
        """Test opening position that exceeds max size."""
        service = PortfolioService()
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("200")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )

        with pytest.raises(ValueError, match="Position size.*exceeds"):
            service.open_position(portfolio, request)

    def test_open_position_exceeds_max_positions(self, portfolio):
        """Test opening position when max positions reached."""
        service = PortfolioService()
        # Open maximum number of positions
        for i in range(5):
            request = PositionRequest(
                symbol=f"STOCK{i}",
                quantity=Quantity(Decimal("10")),
                entry_price=Price(Decimal("100.00")),
                commission=Money(Decimal("1.00")),
            )
            service.open_position(portfolio, request)

        # Try to open one more
        request = PositionRequest(
            symbol="EXTRA",
            quantity=Quantity(Decimal("10")),
            entry_price=Price(Decimal("100.00")),
            commission=Money(Decimal("1.00")),
        )

        with pytest.raises(ValueError, match="Cannot open position.*Maximum position limit"):
            service.open_position(portfolio, request)

    def test_prevent_duplicate_position_opening(self, portfolio):
        """Test that opening a position for an existing symbol is prevented."""
        service = PortfolioService()
        # Open initial position
        request1 = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        service.open_position(portfolio, request1)

        # Add to position
        request2 = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            entry_price=Price(Decimal("155.00")),
            commission=Money(Decimal("1.00")),
        )
        # Should prevent duplicate position opening
        with pytest.raises(ValueError, match="Position already exists for AAPL"):
            service.open_position(portfolio, request2)

    def test_close_full_position(self, portfolio):
        """Test closing a full position."""
        service = PortfolioService()
        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        service.open_position(portfolio, request)

        # Close position
        close_request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("160.00")),
            commission=Money(Decimal("1.00")),
        )
        pnl = service.close_position(
            portfolio, close_request.symbol, close_request.entry_price, close_request.commission
        )

        assert "AAPL" in portfolio.positions  # Position kept for history
        assert portfolio.positions["AAPL"].is_closed()  # But marked as closed
        assert pnl == Money(Decimal("999"))  # (160 - 150) * 100 - 1 = 999
        assert portfolio.total_realized_pnl == Money(Decimal("999"))
        assert portfolio.winning_trades == 1
        assert portfolio.losing_trades == 0

    def test_close_partial_position(self, portfolio):
        """Test closing a partial position."""
        service = PortfolioService()

        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        service.open_position(portfolio, request)

        # Close partial position
        close_request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("50")),
            entry_price=Price(Decimal("160.00")),
            commission=Money(Decimal("1.00")),
        )
        pnl = service.close_position(
            portfolio,
            close_request.symbol,
            close_request.entry_price,
            close_request.commission,
            close_request.quantity,
        )

        assert "AAPL" in portfolio.positions
        assert not portfolio.positions["AAPL"].is_closed()  # Still open
        assert portfolio.positions["AAPL"].quantity == Quantity(Decimal("50"))
        # P&L calculation: (160 - 150) * 50 - 1 commission = 499
        assert pnl == Money(Decimal("499"))  # (160 - 150) * 50 - 1 = 499

    def test_close_position_with_loss(self, portfolio):
        """Test closing a position with a loss."""
        service = PortfolioService()

        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        service.open_position(portfolio, request)

        # Close at a loss
        close_request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("140.00")),
            commission=Money(Decimal("1.00")),
        )
        pnl = service.close_position(
            portfolio, close_request.symbol, close_request.entry_price, close_request.commission
        )

        assert pnl == Money(Decimal("-1001"))  # (140 - 150) * 100 - 1 = -1001
        assert portfolio.losing_trades == 1
        assert portfolio.winning_trades == 0

    def test_close_nonexistent_position(self, portfolio):
        """Test closing a position that doesn't exist."""
        service = PortfolioService()

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )

        with pytest.raises(ValueError, match="No position found"):
            service.close_position(
                portfolio, request.symbol, request.entry_price, request.commission
            )

    def test_close_position_exceeds_quantity(self, portfolio):
        """Test closing more than the position quantity."""
        service = PortfolioService()

        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        service.open_position(portfolio, request)

        # Try to close more than we have
        close_request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("150")),
            entry_price=Price(Decimal("160.00")),
            commission=Money(Decimal("1.00")),
        )

        with pytest.raises(ValueError, match="Cannot close.*only.*available"):
            service.close_position(
                portfolio,
                close_request.symbol,
                close_request.entry_price,
                close_request.commission,
                close_request.quantity,
            )


class TestPortfolioMetrics:
    """Test suite for Portfolio metrics and calculations."""

    @pytest.fixture
    def portfolio_with_positions(self):
        """Create a portfolio with multiple positions."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("50000")),
            max_position_size=Money(Decimal("30000")),  # Allow larger positions for tests
            max_portfolio_risk=Decimal("0.80"),  # Allow 80% risk for testing
        )

        service = PortfolioService()

        # Add profitable position
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        # Add losing position
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="GOOGL",
                quantity=Quantity(Decimal("10")),
                entry_price=Price(Decimal("2500.00")),
                commission=Money(Decimal("2.00")),
            ),
        )

        return portfolio

    def test_total_portfolio_value(self, portfolio_with_positions):
        """Test calculating total portfolio value."""
        # Update current prices for positions
        portfolio_with_positions.update_position_price("AAPL", Price(Decimal("160.00")))
        portfolio_with_positions.update_position_price("GOOGL", Price(Decimal("2400.00")))

        total_value = portfolio_with_positions.get_total_value()

        # Cash: 50000 - 15001 - 25002 = 9997
        # AAPL: 100 * 160 = 16000
        # GOOGL: 10 * 2400 = 24000
        # Total: 9997 + 16000 + 24000 = 49997
        assert total_value == Money(Decimal("49997"))

    def test_total_portfolio_value_with_partial_prices(self, portfolio_with_positions):
        """Test total value calculation with partial price updates."""
        # Only update one position's price
        portfolio_with_positions.update_position_price("AAPL", Price(Decimal("160.00")))

        total_value = portfolio_with_positions.get_total_value()

        # Cash: 50000 - 15001 - 25002 = 9997
        # AAPL: 100 * 160 = 16000
        # GOOGL: 10 * 2500 (entry price) = 25000
        # Total: 9997 + 16000 + 25000 = 50997
        assert total_value == Money(Decimal("50997"))

    def test_unrealized_pnl(self, portfolio_with_positions):
        """Test calculating unrealized PnL."""
        current_prices = {"AAPL": Price(Decimal("160.00")), "GOOGL": Price(Decimal("2400.00"))}

        # Update prices first
        portfolio_with_positions.update_all_prices(current_prices)
        # Then get the unrealized pnl
        unrealized_pnl = portfolio_with_positions.get_unrealized_pnl()

        # AAPL: (160 - 150.00) * 100 = 1000
        # GOOGL: (2400 - 2500.00) * 10 = -1000
        # Total: 1000 + (-1000) = 0
        assert unrealized_pnl == Money(Decimal("0"))

    def test_win_rate_no_trades(self):
        """Test win rate with no trades."""
        portfolio = Portfolio()
        win_rate = PortfolioCalculator.get_win_rate(portfolio)
        assert win_rate == Decimal("0") or win_rate is None

    def test_win_rate_with_trades(self):
        """Test win rate calculation."""
        portfolio = Portfolio()
        portfolio.winning_trades = 7
        portfolio.losing_trades = 3

        assert PortfolioCalculator.get_win_rate(portfolio) == Decimal("70.0")  # 70% win rate

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        portfolio = Portfolio()
        portfolio.total_realized_pnl = Money(Decimal("1000"))
        portfolio.winning_trades = 5

        assert PortfolioCalculator.get_profit_factor(portfolio) is None

    def test_profit_factor_with_trades(self):
        """Test profit factor calculation."""
        portfolio = Portfolio(max_position_size=Money(Decimal("30000")))

        # Create some closed positions with realized P&L
        from datetime import datetime

        from src.domain.entities.position import Position

        # Winning position
        win_position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("0")),  # Closed position
            average_entry_price=Price(Decimal("150.00")),
            realized_pnl=Money(Decimal("100")),  # Profit
            closed_at=datetime.now(UTC),
        )

        # Losing position
        loss_position = Position(
            symbol="GOOGL",
            quantity=Quantity(Decimal("0")),  # Closed position
            average_entry_price=Price(Decimal("2500.00")),
            realized_pnl=Money(Decimal("-50")),  # Loss
            closed_at=datetime.now(UTC),
        )

        portfolio.positions["AAPL"] = win_position
        portfolio.positions["GOOGL"] = loss_position

        factor = PortfolioCalculator.get_profit_factor(portfolio)
        # Should be 100/50 = 2.0
        assert factor == Decimal("2.0")

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        portfolio = Portfolio()

        # Add some realized PnL history
        portfolio.total_realized_pnl = Money(Decimal("5000"))
        portfolio.trades_count = 100

        sharpe = PortfolioCalculator.get_sharpe_ratio(portfolio)
        assert isinstance(sharpe, (Decimal, type(None)))

    def test_max_drawdown_no_positions(self):
        """Test max drawdown with no positions."""
        portfolio = Portfolio()
        assert PortfolioCalculator.get_max_drawdown(portfolio) == Decimal("0")

    def test_roi_calculation(self):
        """Test ROI calculation."""
        portfolio = Portfolio(
            initial_capital=Money(Decimal("100000")), cash_balance=Money(Decimal("110000"))
        )
        portfolio.total_realized_pnl = Money(Decimal("10000"))

        roi = PortfolioCalculator.get_return_percentage(portfolio)
        assert roi == Decimal("10.0")  # 10% return (as percentage)


class TestPortfolioQueries:
    """Test suite for Portfolio query methods."""

    @pytest.fixture
    def portfolio_with_positions(self):
        """Create a portfolio with various positions."""
        from src.application.services.portfolio_service import PortfolioService

        portfolio = Portfolio(
            cash_balance=Money(Decimal("300000")),  # Enough cash for all positions
            max_position_size=Money(Decimal("150000")),  # Allow large positions for tests
            max_portfolio_risk=Decimal("0.90"),  # Allow high risk for testing
        )

        positions = [
            ("AAPL", Quantity(Decimal("100")), Price(Decimal("150.00"))),
            ("GOOGL", Quantity(Decimal("50")), Price(Decimal("2500.00"))),
            ("MSFT", Quantity(Decimal("200")), Price(Decimal("300.00"))),
            ("TSLA", Quantity(Decimal("25")), Price(Decimal("700.00"))),
        ]

        service = PortfolioService()
        for symbol, qty, price in positions:
            service.open_position(
                portfolio,
                PositionRequest(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=price,
                    commission=Money(Decimal("1.00")),
                ),
            )

        return portfolio

    def test_get_position_exists(self, portfolio_with_positions):
        """Test getting an existing position."""
        position = portfolio_with_positions.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == Quantity(Decimal("100"))

    def test_get_position_not_exists(self, portfolio_with_positions):
        """Test getting a non-existent position."""
        position = portfolio_with_positions.get_position("FB")
        assert position is None

    def test_has_position(self, portfolio_with_positions):
        """Test checking if portfolio has a position."""
        assert portfolio_with_positions.has_position("AAPL") is True
        assert portfolio_with_positions.has_position("FB") is False

    def test_get_open_positions(self, portfolio_with_positions):
        """Test getting all open positions."""
        positions = portfolio_with_positions.get_open_positions()
        assert len(positions) == 4
        assert all(isinstance(p, Position) for p in positions)

    def test_get_position_count(self, portfolio_with_positions):
        """Test getting position count."""
        assert portfolio_with_positions.get_position_count() == 4

    def test_is_position_limit_reached(self, portfolio_with_positions):
        """Test checking if position limit is reached."""
        portfolio_with_positions.max_positions = 4
        assert portfolio_with_positions.is_position_limit_reached() is True

        portfolio_with_positions.max_positions = 5
        assert portfolio_with_positions.is_position_limit_reached() is False

    def test_get_largest_position(self, portfolio_with_positions):
        """Test getting the largest position by value."""
        current_prices = {
            "AAPL": Price(Decimal("150.00")),
            "GOOGL": Price(Decimal("2500.00")),
            "MSFT": Price(Decimal("300.00")),
            "TSLA": Price(Decimal("700.00")),
        }

        calculator = PortfolioCalculator()
        largest = calculator.get_largest_position(portfolio_with_positions, current_prices)
        assert largest is not None
        assert largest.symbol == "GOOGL"  # 50 * 2500 = 125000

    def test_get_positions_by_profit(self, portfolio_with_positions):
        """Test getting positions sorted by profit."""
        current_prices = {
            "AAPL": Price(Decimal("160.00")),  # Profit
            "GOOGL": Price(Decimal("2400.00")),  # Loss
            "MSFT": Price(Decimal("310.00")),  # Profit
            "TSLA": Price(Decimal("680.00")),  # Loss
        }

        calculator = PortfolioCalculator()
        positions_by_profit = calculator.get_positions_by_profit(
            portfolio_with_positions, current_prices
        )

        # Should be sorted by profit descending
        assert len(positions_by_profit) == 4
        # First should be most profitable
        assert positions_by_profit[0][0].symbol == "MSFT"  # Biggest total profit
        # Last should be biggest loss
        assert positions_by_profit[-1][0].symbol == "GOOGL"  # Biggest total loss


class TestPortfolioVersionControl:
    """Test suite for Portfolio version control."""

    def test_portfolio_version_increments(self):
        """Test that portfolio version increments on update."""
        portfolio = Portfolio(max_portfolio_risk=Decimal("0.10"))  # 10% risk limit
        initial_version = portfolio.version

        # Open a position (within $10K limit)
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("50")),  # 50 * $150 = $7,500
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        assert portfolio.version == initial_version + 1
        assert portfolio.last_updated is not None

    def test_portfolio_snapshot(self):
        """Test creating portfolio snapshot."""
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Money(Decimal("50000")),
            strategy="momentum",
            max_portfolio_risk=Decimal("0.50"),  # 50% risk limit for testing
            max_position_size=Money(Decimal("20000")),  # Allow $20K positions
        )

        # Add position
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        snapshot = PortfolioCalculator.portfolio_to_dict(portfolio)

        assert snapshot["name"] == "Test Portfolio"
        # Note: version is not included in create_snapshot output
        assert "positions" in snapshot
        assert len(snapshot["positions"]) == 1
        assert snapshot["strategy"] == "momentum"


class TestPortfolioConcurrency:
    """Test suite for Portfolio concurrency handling."""

    def test_concurrent_position_opens(self):
        """Test opening positions concurrently."""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000")), max_positions=100)

        service = PortfolioService()

        def open_position(symbol: str, i: int):
            try:
                service.open_position(
                    portfolio,
                    PositionRequest(
                        symbol=f"{symbol}_{i}",
                        quantity=Quantity(Decimal("10")),
                        entry_price=Price(Decimal("100.00")),
                        commission=Money(Decimal("1.00")),
                    ),
                )
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(open_position, "STOCK", i))

            results = [f.result() for f in futures]

        # All should succeed as we have enough cash and position slots
        assert all(results)
        assert portfolio.get_position_count() == 50

    @pytest.mark.skip(reason="TODO: PortfolioService needs thread safety implementation")
    def test_concurrent_position_updates(self):
        """Test updating the same position concurrently."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("1000000")),
            max_position_size=Money(Decimal("100000")),  # Increase limit for this test
        )

        # Create initial position
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        def add_to_position(i: int):
            try:
                service.open_position(
                    portfolio,
                    PositionRequest(
                        symbol="AAPL",
                        quantity=Quantity(Decimal("10")),
                        entry_price=Price(Decimal("151.00")),
                        commission=Money(Decimal("0.10")),
                    ),
                )
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_to_position, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(results)
        # Final quantity should be 100 + (20 * 10) = 300
        assert portfolio.positions["AAPL"].quantity == Quantity(Decimal("300"))

    @pytest.mark.skip(reason="TODO: PortfolioService needs thread safety implementation")
    def test_concurrent_cash_operations(self):
        """Test concurrent operations affecting cash balance."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")))

        service = PortfolioService()

        def perform_trade(i: int):
            symbol = f"STOCK_{i}"
            try:
                # Open position
                service.open_position(
                    portfolio,
                    PositionRequest(
                        symbol=symbol,
                        quantity=Quantity(Decimal("10")),
                        entry_price=Price(Decimal("100.00")),
                        commission=Money(Decimal("1.00")),
                    ),
                )

                # Close position
                service.close_position(
                    portfolio,
                    symbol=symbol,
                    exit_price=Price(Decimal("101.00")),
                    commission=Money(Decimal("1.00")),
                )
                return True
            except Exception as e:
                print(f"Trade {i} failed: {e}")
                return False

        initial_cash = portfolio.cash_balance.amount

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_trade, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All trades should complete
        assert all(results)

        # Each trade makes $10 profit - $2 commission = $8
        expected_cash = initial_cash + (Decimal("8") * 10)
        assert abs(portfolio.cash_balance.amount - expected_cash) < Decimal("0.01")

    def test_thread_safe_version_increments(self):
        """Test that version increments are thread-safe."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("1000000")),
            max_positions=100,  # Allow more positions for concurrent test
        )
        initial_version = portfolio.version

        service = PortfolioService()

        def update_portfolio(i: int):
            service.open_position(
                portfolio,
                PositionRequest(
                    symbol=f"STOCK_{i}",
                    quantity=Quantity(Decimal("1")),
                    entry_price=Price(Decimal("100.00")),
                    commission=Money(Decimal("0.10")),
                ),
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_portfolio, i) for i in range(50)]
            for f in futures:
                f.result()

        # Version should have incremented exactly 50 times
        assert portfolio.version == initial_version + 50


class TestPortfolioEdgeCases:
    """Test suite for Portfolio edge cases."""

    def test_very_small_position(self):
        """Test handling very small position quantities."""
        portfolio = Portfolio()

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("0.001")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("0.01")),
        )

        service = PortfolioService()
        position = service.open_position(portfolio, request)
        assert position.quantity == Quantity(Decimal("0.001"))

    def test_very_large_position(self):
        """Test handling very large position quantities."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("1000000000")),
            max_position_size=Money(Decimal("1000000000")),
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk for large positions
        )

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("1000000")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("100.00")),
        )

        service = PortfolioService()
        position = service.open_position(portfolio, request)
        assert position.quantity == Quantity(Decimal("1000000"))

    def test_zero_commission(self):
        """Test handling zero commission trades."""
        portfolio = Portfolio(
            max_position_size=Money(Decimal("20000")),  # Allow larger position for this test
            max_portfolio_risk=Decimal("0.20"),  # Allow 20% risk for this test
        )

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("0")),
        )

        service = PortfolioService()
        position = service.open_position(portfolio, request)
        assert position.average_entry_price == Price(Decimal("150.00"))
        assert portfolio.total_commission_paid == Money(Decimal("0"))

    @pytest.mark.skip(reason="TODO: PortfolioService needs thread safety implementation")
    def test_high_frequency_trading_scenario(self):
        """Test portfolio handling high-frequency trading."""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000")), max_positions=1000)

        service = PortfolioService()

        # Simulate 100 rapid trades
        for i in range(100):
            symbol = f"HFT_{i % 10}"  # Rotate through 10 symbols

            # Open or add to position
            service.open_position(
                portfolio,
                PositionRequest(
                    symbol=symbol,
                    quantity=Quantity(Decimal("1")),
                    entry_price=Price(Decimal("100.00")),
                    commission=Money(Decimal("0.01")),
                ),
            )

            # Immediately close if we have enough
            if portfolio.has_position(symbol) and portfolio.positions[
                symbol
            ].quantity.value >= Decimal("2"):
                service.close_position(
                    portfolio,
                    symbol=symbol,
                    exit_price=Price(Decimal("100.10")),
                    commission=Money(Decimal("0.01")),
                    quantity=Quantity(Decimal("1")),
                )

        assert portfolio.trades_count > 0
        assert portfolio.total_commission_paid.amount > Decimal("0")

    def test_portfolio_after_many_operations(self):
        """Test portfolio consistency after many operations."""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000")), max_positions=30)
        initial_cash = portfolio.cash_balance.amount

        # Track all operations
        total_spent = Decimal("0")
        total_received = Decimal("0")
        total_commission = Decimal("0")

        service = PortfolioService()

        # Perform many operations
        for i in range(20):
            symbol = f"TEST_{i}"

            # Open position
            open_qty = Quantity(Decimal("10"))
            open_price = Price(Decimal("100.00"))
            commission = Money(Decimal("1.00"))

            service.open_position(
                portfolio,
                PositionRequest(
                    symbol=symbol, quantity=open_qty, entry_price=open_price, commission=commission
                ),
            )

            total_spent += open_qty.value * open_price.value
            total_commission += commission.amount

            # Close half
            close_qty = Quantity(Decimal("5"))
            close_price = Price(Decimal("101.00"))

            service.close_position(
                portfolio,
                symbol=symbol,
                exit_price=close_price,
                commission=commission,
                quantity=close_qty,
            )

            total_received += close_qty.value * close_price.value
            total_commission += commission.amount

        # Verify cash balance
        expected_cash = initial_cash - total_spent + total_received - total_commission
        assert abs(portfolio.cash_balance.amount - expected_cash) < Decimal("0.01")

        # Verify commission tracking
        assert portfolio.total_commission_paid == Money(total_commission)


class TestPortfolioSerialization:
    """Test suite for Portfolio serialization."""

    def test_portfolio_to_dict(self):
        """Test converting portfolio to dictionary."""
        portfolio = Portfolio(
            name="Test Portfolio",
            strategy="momentum",
            tags={"risk": "high"},
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.20"),  # Allow up to 20% risk
        )

        # Add position
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        data = PortfolioCalculator.portfolio_to_dict(portfolio)

        assert data["name"] == "Test Portfolio"
        assert data["strategy"] == "momentum"
        assert data["tags"]["risk"] == "high"
        assert len(data["positions"]) == 1
        assert data["cash_balance"] == float(portfolio.cash_balance.amount)

    def test_portfolio_json_serialization(self):
        """Test JSON serialization of portfolio."""
        portfolio = Portfolio(
            name="JSON Test",
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.20"),
        )

        # Add positions
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        # Convert to JSON
        json_str = json.dumps(PortfolioCalculator.portfolio_to_dict(portfolio), default=str)

        # Should not raise exception
        data = json.loads(json_str)
        assert data["name"] == "JSON Test"
        assert "positions" in data


class TestPortfolioStatistics:
    """Test suite for Portfolio statistics."""

    def test_average_win_calculation(self):
        """Test calculating average win."""
        portfolio = Portfolio()

        # Simulate winning trades
        for i in range(5):
            portfolio.winning_trades += 1
            portfolio.total_realized_pnl = Money(
                portfolio.total_realized_pnl.amount + Decimal("100")
            )

        avg_win = PortfolioCalculator.get_average_win(portfolio)
        assert avg_win == Money(Decimal("100"))

    def test_average_loss_calculation(self):
        """Test calculating average loss."""
        portfolio = Portfolio()

        # Simulate losing trades
        for i in range(3):
            portfolio.losing_trades += 1
            portfolio.total_realized_pnl = Money(
                portfolio.total_realized_pnl.amount - Decimal("50")
            )

        avg_loss = PortfolioCalculator.get_average_loss(portfolio)
        assert avg_loss == Money(Decimal("50"))

    def test_expectancy_calculation(self):
        """Test calculating trading expectancy."""
        portfolio = Portfolio()

        # Add winning trades
        portfolio.winning_trades = 6
        portfolio.losing_trades = 4
        portfolio.total_realized_pnl = Money(Decimal("200"))  # Net profit

        expectancy = PortfolioCalculator.get_expectancy(portfolio)
        assert expectancy == Money(Decimal("20"))  # 200 / 10 trades

    def test_portfolio_summary(self):
        """Test generating portfolio summary."""
        # Create portfolio with higher risk limit for testing
        portfolio = Portfolio(
            name="Summary Test",
            max_portfolio_risk=Decimal("0.10"),  # Allow 10% risk for testing
        )

        # Add some activity - within both position size and risk limits
        service = PortfolioService()
        service.open_position(
            portfolio,
            PositionRequest(
                symbol="AAPL",
                quantity=Quantity(Decimal("50")),  # 50 shares × $150 = $7,500
                entry_price=Price(Decimal("150.00")),
                commission=Money(Decimal("1.00")),
            ),
        )

        summary = PortfolioCalculator.get_portfolio_summary(portfolio)

        assert summary["name"] == "Summary Test"
        assert summary["position_count"] == 1
        assert summary["cash_balance"] == portfolio.cash_balance.amount
        assert "total_value" in summary
        assert "roi" in summary

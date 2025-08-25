"""
End-to-End Trading Workflow Integration Tests

Comprehensive tests validating complete trading workflows including:
- Complete trading lifecycle from portfolio creation to P&L calculation
- Risk management workflows with limit validation
- Market simulation with limit order triggering
- Portfolio management across multiple positions

These tests use actual PostgreSQL database connections and test the full stack.
"""

# Standard library imports
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

# Third-party imports
import pytest

# Local imports
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position, PositionSide
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.price import Price
from src.infrastructure.brokers.paper_broker import PaperBroker
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import ConnectionFactory, DatabaseConfig
from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWork

# Skip integration tests if environment variable not set
SKIP_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
pytestmark = pytest.mark.skipif(
    SKIP_INTEGRATION, reason="Integration tests require database and RUN_INTEGRATION_TESTS=true"
)


@pytest.fixture(scope="session")
async def database_config():
    """Database configuration for integration tests."""
    return DatabaseConfig(
        host=os.getenv("TEST_DATABASE_HOST", "localhost"),
        port=int(os.getenv("TEST_DATABASE_PORT", "5432")),
        database=os.getenv("TEST_DATABASE_NAME", "ai_trader"),
        user=os.getenv("TEST_DATABASE_USER", "zachwade"),
        password=os.getenv("TEST_DATABASE_PASSWORD", ""),
        min_pool_size=2,
        max_pool_size=10,
        command_timeout=30.0,
    )


@pytest.fixture(scope="session")
async def database_adapter(database_config) -> AsyncGenerator[PostgreSQLAdapter, None]:
    """Create database adapter for the test session."""
    adapter = None
    try:
        adapter = await ConnectionFactory.create_connection(database_config)

        # Verify connection
        health_ok = await adapter.health_check()
        if not health_ok:
            pytest.skip("Database health check failed")

        yield adapter
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {e}")
    finally:
        if adapter and hasattr(adapter, "_pool"):
            await adapter._pool.close()


@pytest.fixture
async def clean_database(database_adapter):
    """Clean database before and after each test."""
    adapter = database_adapter

    # Clean up test data before test
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'E2E_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'E2E_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'E2E_%'")

    yield adapter

    # Clean up test data after test
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'E2E_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'E2E_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'E2E_%'")


@pytest.fixture
async def unit_of_work(clean_database):
    """Create unit of work for transaction management."""
    return PostgreSQLUnitOfWork(clean_database)


@pytest.fixture
def paper_broker():
    """Create paper broker for simulated trading."""
    broker = PaperBroker(initial_capital=Decimal("1000000"))
    broker.connect()
    return broker


@pytest.fixture
def risk_calculator():
    """Create risk calculator service."""
    return RiskCalculator()


@pytest.fixture
def position_manager():
    """Create position manager service."""
    return PositionManager()


class TestCompleteTradingLifecycle:
    """Test complete trading lifecycle from portfolio creation to P&L calculation."""

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, unit_of_work, paper_broker, risk_calculator):
        """
        Test complete trading lifecycle:
        1. Create portfolio
        2. Place buy order
        3. Process order fill
        4. Update market price
        5. Place sell order
        6. Process sell fill
        7. Calculate P&L
        8. Verify portfolio state
        """
        # 1. Create portfolio
        portfolio = Portfolio(
            id=uuid4(),
            name="E2E_Test_Portfolio",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            max_position_size=Decimal("20000"),
            max_portfolio_risk=Decimal("0.05"),
            strategy="momentum",
        )

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.portfolios.save(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 2. Place buy order
        buy_order = Order.create_market_order(
            symbol="E2E_AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            reason="Test buy order",
        )

        # Submit order through broker
        submitted_order = paper_broker.submit_order(buy_order)

        # Save order to database
        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.orders.save(submitted_order)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 3. Process order fill (simulate market execution)
        fill_price = Decimal("150.00")
        paper_broker.set_market_price("E2E_AAPL", fill_price)
        paper_broker.process_market_order(submitted_order.id, fill_price)

        # Get updated order
        filled_order = paper_broker.get_order(submitted_order.id)
        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.average_fill_price == fill_price

        # Update order in database
        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.orders.update(filled_order)

            # Create position from filled order
            position_request = PositionRequest(
                symbol="E2E_AAPL",
                quantity=filled_order.filled_quantity,
                entry_price=filled_order.average_fill_price,
                commission=Decimal("1.00"),
                strategy="momentum",
            )
            position = await portfolio.open_position(position_request)

            # Save position
            await unit_of_work.positions.save(position)

            # Update portfolio cash balance
            cost = filled_order.filled_quantity * filled_order.average_fill_price + Decimal("1.00")
            portfolio.cash_balance -= cost
            portfolio.total_commission_paid += Decimal("1.00")
            await unit_of_work.portfolios.update(portfolio)

            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 4. Update market price (price moves up)
        new_market_price = Decimal("155.00")
        paper_broker.set_market_price("E2E_AAPL", new_market_price)

        # Get position and calculate unrealized P&L
        position = await unit_of_work.positions.get_by_symbol("E2E_AAPL")
        assert position is not None

        position_risk = risk_calculator.calculate_position_risk(position, Price(new_market_price))
        unrealized_pnl = position_risk["unrealized_pnl"]
        expected_unrealized = (new_market_price - fill_price) * Decimal("100")
        assert unrealized_pnl == expected_unrealized

        # 5. Place sell order to close position
        sell_order = Order.create_market_order(
            symbol="E2E_AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            reason="Close position with profit",
        )

        # Submit sell order
        submitted_sell = paper_broker.submit_order(sell_order)

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.orders.save(submitted_sell)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 6. Process sell fill
        paper_broker.process_market_order(submitted_sell.id, new_market_price)
        filled_sell = paper_broker.get_order(submitted_sell.id)
        assert filled_sell.status == OrderStatus.FILLED

        # Update database with filled sell order
        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.orders.update(filled_sell)

            # Close position
            position = await unit_of_work.positions.get_by_symbol("E2E_AAPL")
            await portfolio.close_position(
                symbol="E2E_AAPL",
                exit_price=new_market_price,
                commission=Decimal("1.00"),
            )

            # Update portfolio with proceeds
            proceeds = filled_sell.filled_quantity * new_market_price - Decimal("1.00")
            portfolio.cash_balance += proceeds
            portfolio.total_commission_paid += Decimal("1.00")
            portfolio.trades_count += 1

            # Calculate and record P&L
            trade_pnl = (new_market_price - fill_price) * Decimal("100") - Decimal(
                "2.00"
            )  # Minus commissions
            portfolio.total_realized_pnl += trade_pnl

            if trade_pnl > 0:
                portfolio.winning_trades += 1
            else:
                portfolio.losing_trades += 1

            await unit_of_work.portfolios.update(portfolio)
            await unit_of_work.positions.update(position)

            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 7. Verify final portfolio state
        final_portfolio = await unit_of_work.portfolios.get(portfolio.id)
        assert final_portfolio is not None

        # Check P&L calculation
        expected_gross_pnl = (new_market_price - fill_price) * Decimal("100")
        expected_net_pnl = expected_gross_pnl - Decimal("2.00")  # Minus commissions
        assert final_portfolio.total_realized_pnl == expected_net_pnl

        # Check cash balance
        expected_cash = Decimal("100000") - Decimal("2.00")  # Initial minus commissions
        assert abs(final_portfolio.cash_balance - expected_cash) < Decimal("0.01")

        # Check trade statistics
        assert final_portfolio.trades_count == 1
        assert final_portfolio.winning_trades == 1
        assert final_portfolio.losing_trades == 0
        assert final_portfolio.total_commission_paid == Decimal("2.00")

        # Verify position is closed
        closed_position = await unit_of_work.positions.get_by_symbol("E2E_AAPL")
        assert closed_position is not None
        assert closed_position.is_closed()


class TestRiskManagementWorkflow:
    """Test risk management workflows with limit validation."""

    @pytest.mark.asyncio
    async def test_risk_limits_enforcement(self, unit_of_work, paper_broker, risk_calculator):
        """
        Test risk management workflow:
        1. Create portfolio with risk limits
        2. Place orders that test risk limits
        3. Validate risk calculations
        4. Test position sizing based on risk
        """
        # 1. Create portfolio with strict risk limits
        portfolio = Portfolio(
            id=uuid4(),
            name="E2E_Risk_Portfolio",
            initial_capital=Decimal("50000"),
            cash_balance=Decimal("50000"),
            max_position_size=Decimal("5000"),  # Max $5000 per position
            max_portfolio_risk=Decimal("0.02"),  # Max 2% portfolio risk
            max_positions=3,  # Max 3 concurrent positions
            strategy="conservative",
        )

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.portfolios.save(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 2. Test position size limit
        # This order should be rejected (exceeds max_position_size)
        can_open, reason = portfolio.can_open_position(
            symbol="E2E_TSLA",
            quantity=Decimal("100"),
            price=Decimal("200"),  # Total value: $20,000 > $5,000 limit
        )
        assert not can_open
        assert "exceeds limit" in reason.lower()

        # This order should be accepted
        can_open, reason = portfolio.can_open_position(
            symbol="E2E_TSLA",
            quantity=Decimal("20"),
            price=Decimal("200"),  # Total value: $4,000 < $5,000 limit
        )
        assert can_open

        # 3. Place order within limits
        order = Order.create_limit_order(
            symbol="E2E_TSLA",
            quantity=Decimal("20"),
            side=OrderSide.BUY,
            limit_price=Decimal("200"),
            reason="Risk-managed entry",
        )

        submitted = paper_broker.submit_order(order)

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.orders.save(submitted)

            # Simulate fill
            paper_broker.set_market_price("E2E_TSLA", Decimal("199"))
            filled_order = paper_broker.get_order(submitted.id)
            filled_order.fill(Decimal("20"), Decimal("199"))

            await unit_of_work.orders.update(filled_order)

            # Open position
            position_request = PositionRequest(
                symbol="E2E_TSLA",
                quantity=Decimal("20"),
                entry_price=Decimal("199"),
                commission=Decimal("1.00"),
                strategy="conservative",
            )
            position = await portfolio.open_position(position_request)

            await unit_of_work.positions.save(position)

            # Update portfolio
            portfolio.cash_balance -= Decimal("20") * Decimal("199") + Decimal("1.00")
            await unit_of_work.portfolios.update(portfolio)

            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 4. Test portfolio risk calculation
        position = await unit_of_work.positions.get_by_symbol("E2E_TSLA")
        assert position is not None

        # Calculate position risk with 5% stop loss
        stop_price = Decimal("189.05")  # 5% below entry
        potential_loss = (Decimal("199") - stop_price) * Decimal("20")
        portfolio_value = portfolio.get_total_value_sync()
        risk_percentage = potential_loss / portfolio_value

        assert risk_percentage < portfolio.max_portfolio_risk

        # 5. Test max positions limit
        # Open second position
        can_open, _ = portfolio.can_open_position(
            symbol="E2E_GOOGL",
            quantity=Decimal("2"),
            price=Decimal("2000"),
        )
        assert can_open  # Should be allowed (1 of 3 positions used)

        # Simulate opening two more positions
        portfolio.positions["E2E_GOOGL"] = Position(
            symbol="E2E_GOOGL",
            quantity=Decimal("2"),
            side=PositionSide.LONG,
            average_price=Decimal("2000"),
            current_price=Decimal("2000"),
            opened_at=datetime.now(UTC),
        )

        portfolio.positions["E2E_MSFT"] = Position(
            symbol="E2E_MSFT",
            quantity=Decimal("10"),
            side=PositionSide.LONG,
            average_price=Decimal("300"),
            current_price=Decimal("300"),
            opened_at=datetime.now(UTC),
        )

        # Try to open fourth position (should be rejected)
        can_open, reason = portfolio.can_open_position(
            symbol="E2E_NVDA",
            quantity=Decimal("5"),
            price=Decimal("500"),
        )
        assert not can_open
        assert "maximum positions limit" in reason.lower()

        # 6. Test position sizing based on Kelly Criterion
        win_rate = Decimal("0.6")  # 60% win rate
        avg_win = Decimal("1.5")  # Average win is 1.5x average loss
        avg_loss = Decimal("1.0")

        kelly_fraction = risk_calculator.calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        # Kelly should suggest reasonable position sizing
        assert Decimal("0") < kelly_fraction < Decimal("1")

        # Calculate position size based on Kelly
        portfolio_value = Decimal("50000")
        position_size = portfolio_value * kelly_fraction

        # Ensure position size respects portfolio limits
        if position_size > portfolio.max_position_size:
            position_size = portfolio.max_position_size


class TestMarketSimulationWorkflow:
    """Test market simulation with limit order triggering."""

    @pytest.mark.asyncio
    async def test_limit_order_execution(self, unit_of_work, paper_broker):
        """
        Test market simulation workflow:
        1. Place multiple limit orders
        2. Simulate market price movements
        3. Trigger limit orders
        4. Process fills
        5. Verify execution logic
        """
        # 1. Create portfolio
        portfolio = Portfolio(
            id=uuid4(),
            name="E2E_Market_Sim_Portfolio",
            initial_capital=Decimal("200000"),
            cash_balance=Decimal("200000"),
            strategy="market_making",
        )

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.portfolios.save(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 2. Place multiple limit orders at different price levels
        orders = []

        # Buy limit orders (below current market)
        buy_limits = [
            ("E2E_SPY", Decimal("50"), Decimal("440")),  # Buy 50 shares at $440
            ("E2E_SPY", Decimal("100"), Decimal("435")),  # Buy 100 shares at $435
            ("E2E_SPY", Decimal("150"), Decimal("430")),  # Buy 150 shares at $430
        ]

        for symbol, quantity, limit_price in buy_limits:
            order = Order.create_limit_order(
                symbol=symbol,
                quantity=quantity,
                side=OrderSide.BUY,
                limit_price=limit_price,
                reason=f"Limit buy at {limit_price}",
            )
            submitted = paper_broker.submit_order(order)
            orders.append(submitted)

        # Sell limit orders (above current market)
        current_position_qty = Decimal("200")  # Assume we have 200 shares
        sell_limits = [
            ("E2E_SPY", Decimal("50"), Decimal("455")),  # Sell 50 shares at $455
            ("E2E_SPY", Decimal("75"), Decimal("460")),  # Sell 75 shares at $460
            ("E2E_SPY", Decimal("75"), Decimal("465")),  # Sell 75 shares at $465
        ]

        for symbol, quantity, limit_price in sell_limits:
            order = Order.create_limit_order(
                symbol=symbol,
                quantity=quantity,
                side=OrderSide.SELL,
                limit_price=limit_price,
                reason=f"Limit sell at {limit_price}",
            )
            submitted = paper_broker.submit_order(order)
            orders.append(submitted)

        # Save all orders to database
        await unit_of_work.begin_transaction()
        try:
            for order in orders:
                await unit_of_work.orders.save(order)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 3. Simulate market price movements
        # Initial market price
        paper_broker.set_market_price("E2E_SPY", Decimal("445"))

        # Market drops - should trigger buy orders
        price_sequence = [
            Decimal("442"),  # No triggers yet
            Decimal("439"),  # Should trigger $440 buy
            Decimal("434"),  # Should trigger $435 buy
            Decimal("428"),  # Should trigger $430 buy
        ]

        filled_orders = []

        for market_price in price_sequence:
            paper_broker.set_market_price("E2E_SPY", market_price)

            # Check each buy order
            for order in orders[:3]:  # First 3 are buy orders
                if order.status == OrderStatus.PENDING:
                    if order.order_type == OrderType.LIMIT and order.side == OrderSide.BUY:
                        # Buy limit triggers when market <= limit price
                        if market_price <= order.limit_price:
                            # Fill the order
                            paper_broker.process_limit_order(order.id, market_price)
                            filled = paper_broker.get_order(order.id)
                            filled_orders.append(filled)

        # Verify buy orders filled correctly
        assert len(filled_orders) == 3
        for filled in filled_orders:
            assert filled.status == OrderStatus.FILLED
            assert filled.average_fill_price <= filled.limit_price

        # 4. Market rises - should trigger sell orders
        price_sequence_up = [
            Decimal("450"),  # No sell triggers yet
            Decimal("456"),  # Should trigger $455 sell
            Decimal("461"),  # Should trigger $460 sell
            Decimal("467"),  # Should trigger $465 sell
        ]

        for market_price in price_sequence_up:
            paper_broker.set_market_price("E2E_SPY", market_price)

            # Check each sell order
            for order in orders[3:]:  # Last 3 are sell orders
                if order.status == OrderStatus.PENDING:
                    if order.order_type == OrderType.LIMIT and order.side == OrderSide.SELL:
                        # Sell limit triggers when market >= limit price
                        if market_price >= order.limit_price:
                            # Fill the order
                            paper_broker.process_limit_order(order.id, market_price)
                            filled = paper_broker.get_order(order.id)
                            filled_orders.append(filled)

        # 5. Verify all orders executed correctly
        await unit_of_work.begin_transaction()
        try:
            for filled in filled_orders:
                await unit_of_work.orders.update(filled)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # Verify execution logic
        buy_fills = filled_orders[:3]
        sell_fills = filled_orders[3:]

        # All buy orders should have filled at or below limit
        for order in buy_fills:
            assert order.average_fill_price <= order.limit_price
            assert order.status == OrderStatus.FILLED

        # All sell orders should have filled at or above limit
        for order in sell_fills:
            assert order.average_fill_price >= order.limit_price
            assert order.status == OrderStatus.FILLED

        # Calculate total execution costs
        total_bought = sum(o.filled_quantity * o.average_fill_price for o in buy_fills)
        total_sold = sum(o.filled_quantity * o.average_fill_price for o in sell_fills)

        # Net P&L from limit order execution
        net_pnl = total_sold - total_bought
        print(f"Market simulation P&L: ${net_pnl:.2f}")


class TestPortfolioManagementWorkflow:
    """Test portfolio management across multiple positions."""

    @pytest.mark.asyncio
    async def test_multi_position_portfolio(self, unit_of_work, paper_broker, risk_calculator):
        """
        Test portfolio management workflow:
        1. Create multiple positions
        2. Calculate portfolio metrics
        3. Close positions
        4. Verify final state
        """
        # 1. Create diversified portfolio
        portfolio = Portfolio(
            id=uuid4(),
            name="E2E_Diversified_Portfolio",
            initial_capital=Decimal("500000"),
            cash_balance=Decimal("500000"),
            max_position_size=Decimal("100000"),
            max_positions=10,
            strategy="diversified",
        )

        await unit_of_work.begin_transaction()
        try:
            await unit_of_work.portfolios.save(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 2. Create multiple positions across sectors
        positions_data = [
            # Tech stocks
            ("E2E_AAPL", Decimal("200"), Decimal("150"), "tech"),
            ("E2E_MSFT", Decimal("150"), Decimal("300"), "tech"),
            ("E2E_GOOGL", Decimal("50"), Decimal("2500"), "tech"),
            # Financial stocks
            ("E2E_JPM", Decimal("300"), Decimal("140"), "finance"),
            ("E2E_BAC", Decimal("500"), Decimal("35"), "finance"),
            # Healthcare stocks
            ("E2E_JNJ", Decimal("100"), Decimal("160"), "healthcare"),
            ("E2E_PFE", Decimal("400"), Decimal("40"), "healthcare"),
        ]

        total_invested = Decimal("0")
        positions_created = []

        await unit_of_work.begin_transaction()
        try:
            for symbol, quantity, entry_price, sector in positions_data:
                # Create and submit order
                order = Order.create_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=OrderSide.BUY,
                    reason=f"Diversification - {sector}",
                )

                submitted = paper_broker.submit_order(order)
                paper_broker.set_market_price(symbol, entry_price)
                paper_broker.process_market_order(submitted.id, entry_price)

                await unit_of_work.orders.save(submitted)

                # Create position
                position_request = PositionRequest(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    commission=Decimal("1.00"),
                    strategy=sector,
                )
                position = await portfolio.open_position(position_request)

                await unit_of_work.positions.save(position)
                positions_created.append(position)

                # Update portfolio cash
                cost = quantity * entry_price + Decimal("1.00")
                portfolio.cash_balance -= cost
                total_invested += cost

            await unit_of_work.portfolios.update(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 3. Simulate market movements and calculate portfolio metrics
        market_changes = {
            "E2E_AAPL": Decimal("155"),  # +3.33%
            "E2E_MSFT": Decimal("310"),  # +3.33%
            "E2E_GOOGL": Decimal("2450"),  # -2%
            "E2E_JPM": Decimal("145"),  # +3.57%
            "E2E_BAC": Decimal("34"),  # -2.86%
            "E2E_JNJ": Decimal("165"),  # +3.125%
            "E2E_PFE": Decimal("42"),  # +5%
        }

        for symbol, new_price in market_changes.items():
            paper_broker.set_market_price(symbol, new_price)

        # Calculate portfolio metrics
        portfolio_value = portfolio.cash_balance
        total_unrealized_pnl = Decimal("0")
        position_values = []

        for position in positions_created:
            current_price = market_changes[position.symbol]
            position.current_price = current_price

            risk_metrics = risk_calculator.calculate_position_risk(position, Price(current_price))

            position_value = risk_metrics["position_value"]
            unrealized_pnl = risk_metrics["unrealized_pnl"]

            portfolio_value += position_value
            total_unrealized_pnl += unrealized_pnl
            position_values.append((position.symbol, position_value, unrealized_pnl))

        # Calculate portfolio-level metrics
        initial_portfolio_value = portfolio.initial_capital
        total_return = (portfolio_value - initial_portfolio_value) / initial_portfolio_value

        # Calculate diversification metrics
        position_weights = [pv / portfolio_value for _, pv, _ in position_values]

        # Herfindahl-Hirschman Index for concentration
        hhi = sum(w**2 for w in position_weights)
        effective_n = 1 / hhi if hhi > 0 else 1

        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Unrealized P&L: ${total_unrealized_pnl:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Effective Number of Positions: {effective_n:.2f}")

        # 4. Close winning positions (take profits)
        positions_to_close = []

        for position in positions_created:
            current_price = market_changes[position.symbol]
            pnl_pct = (current_price - position.average_price) / position.average_price

            # Close positions with >3% gain
            if pnl_pct > Decimal("0.03"):
                positions_to_close.append((position, current_price))

        await unit_of_work.begin_transaction()
        try:
            for position, exit_price in positions_to_close:
                # Create sell order
                sell_order = Order.create_market_order(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    side=OrderSide.SELL,
                    reason="Take profit - 3% gain",
                )

                submitted_sell = paper_broker.submit_order(sell_order)
                paper_broker.process_market_order(submitted_sell.id, exit_price)

                await unit_of_work.orders.save(submitted_sell)

                # Close position
                await portfolio.close_position(
                    symbol=position.symbol,
                    exit_price=exit_price,
                    commission=Decimal("1.00"),
                )

                # Update portfolio
                proceeds = position.quantity * exit_price - Decimal("1.00")
                portfolio.cash_balance += proceeds

                # Calculate realized P&L
                realized_pnl = (exit_price - position.average_price) * position.quantity - Decimal(
                    "2.00"
                )
                portfolio.total_realized_pnl += realized_pnl
                portfolio.trades_count += 1

                if realized_pnl > 0:
                    portfolio.winning_trades += 1
                else:
                    portfolio.losing_trades += 1

                # Mark position as closed
                position.close(exit_price, datetime.now(UTC))
                await unit_of_work.positions.update(position)

            await unit_of_work.portfolios.update(portfolio)
            await unit_of_work.commit()
        except Exception:
            await unit_of_work.rollback()
            raise

        # 5. Verify final portfolio state
        final_portfolio = await unit_of_work.portfolios.get(portfolio.id)
        assert final_portfolio is not None

        # Check positions closed correctly
        for position, _ in positions_to_close:
            db_position = await unit_of_work.positions.get(position.id)
            assert db_position is not None
            assert db_position.is_closed()

        # Verify portfolio statistics
        assert final_portfolio.trades_count == len(positions_to_close)
        assert final_portfolio.winning_trades == len(positions_to_close)  # All were profitable
        assert final_portfolio.total_realized_pnl > 0

        # Calculate final metrics
        open_positions = [p for p in positions_created if not p.is_closed()]
        final_portfolio_value = final_portfolio.cash_balance

        for position in open_positions:
            if not position.is_closed():
                current_price = market_changes[position.symbol]
                final_portfolio_value += position.quantity * current_price

        final_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

        print("\nFinal Portfolio Summary:")
        print(f"Cash Balance: ${final_portfolio.cash_balance:.2f}")
        print(f"Open Positions: {len(open_positions)}")
        print(f"Closed Positions: {len(positions_to_close)}")
        print(f"Total Realized P&L: ${final_portfolio.total_realized_pnl:.2f}")
        print(f"Win Rate: {final_portfolio.winning_trades}/{final_portfolio.trades_count}")
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Final Return: {final_return:.2%}")

        # Assert reasonable performance
        assert final_return > Decimal("-0.1")  # No worse than -10%
        assert final_portfolio.cash_balance > Decimal("0")  # Still have cash
        assert len(open_positions) + len(positions_to_close) == len(positions_created)


# Helper methods for PaperBroker to simulate market
def set_market_price(self, symbol: str, price: Decimal) -> None:
    """Set market price for a symbol."""
    with self._lock:
        if self.state.market_prices is None:
            self.state.market_prices = {}
        self.state.market_prices[symbol] = price


def process_market_order(self, order_id: UUID, fill_price: Decimal) -> None:
    """Process a market order fill."""
    with self._lock:
        if self.state.orders is None:
            return

        if order_id in self.state.orders:
            order = self.state.orders[order_id]
            if order.status == OrderStatus.PENDING and order.order_type == OrderType.MARKET:
                order.fill(order.quantity, fill_price)


def process_limit_order(self, order_id: UUID, market_price: Decimal) -> None:
    """Process a limit order if market conditions are met."""
    with self._lock:
        if self.state.orders is None:
            return

        if order_id in self.state.orders:
            order = self.state.orders[order_id]
            if order.status == OrderStatus.PENDING and order.order_type == OrderType.LIMIT:
                should_fill = False

                if (
                    order.side == OrderSide.BUY
                    and market_price <= order.limit_price
                    or order.side == OrderSide.SELL
                    and market_price >= order.limit_price
                ):
                    should_fill = True

                if should_fill:
                    # Fill at the limit price (conservative) or market price
                    fill_price = order.limit_price  # Conservative fill
                    order.fill(order.quantity, fill_price)


# Monkey-patch PaperBroker with helper methods for testing
PaperBroker.set_market_price = set_market_price
PaperBroker.process_market_order = process_market_order
PaperBroker.process_limit_order = process_limit_order


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

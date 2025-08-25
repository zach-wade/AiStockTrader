"""
Tests for application interface examples.

Tests the TradingService example class to ensure interface usage patterns work correctly.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.interfaces.examples import TradingService, example_usage_patterns
from src.application.interfaces.exceptions import OrderNotFoundError, PositionNotFoundError
from src.domain.entities.order import Order, OrderRequest, OrderSide, OrderStatus
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position


class TestTradingService:
    """Test the TradingService example class."""

    @pytest.fixture
    def mock_uow(self):
        """Create a mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.positions = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        return uow

    @pytest.fixture
    def mock_tx_manager(self):
        """Create a mock transaction manager."""
        tx_manager = AsyncMock()
        return tx_manager

    @pytest.fixture
    def trading_service(self, mock_uow, mock_tx_manager):
        """Create a TradingService instance with mocked dependencies."""
        return TradingService(mock_uow, mock_tx_manager)

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        request = OrderRequest(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, reason="Test order"
        )
        order = Order.create_market_order(request)
        # Submit the order so it can be filled
        order.submit("test_broker_id")
        return order

    @pytest.fixture
    def sample_position(self):
        """Create a sample position for testing."""
        return Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        portfolio_id = uuid4()
        return Portfolio(
            id=portfolio_id,
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
        )

    async def test_place_market_order_success(self, trading_service, mock_uow):
        """Test successful market order placement."""
        # Setup
        saved_order = MagicMock()
        mock_uow.orders.save_order.return_value = saved_order

        # Execute
        result = await trading_service.place_market_order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, reason="Test order"
        )

        # Verify
        assert result == saved_order
        mock_uow.orders.save_order.assert_called_once()
        mock_uow.commit.assert_called_once()
        mock_uow.__aenter__.assert_called_once()
        mock_uow.__aexit__.assert_called_once()

        # Verify the order was created correctly
        call_args = mock_uow.orders.save_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.quantity == Decimal("100")
        assert call_args.side == OrderSide.BUY

    async def test_place_market_order_without_reason(self, trading_service, mock_uow):
        """Test market order placement without reason."""
        # Setup
        saved_order = MagicMock()
        mock_uow.orders.save_order.return_value = saved_order

        # Execute
        result = await trading_service.place_market_order(
            symbol="TSLA", quantity=Decimal("50"), side=OrderSide.SELL
        )

        # Verify
        assert result == saved_order
        call_args = mock_uow.orders.save_order.call_args[0][0]
        assert call_args.symbol == "TSLA"
        assert call_args.quantity == Decimal("50")
        assert call_args.side == OrderSide.SELL

    async def test_execute_trade_with_new_position(
        self, trading_service, mock_tx_manager, sample_order
    ):
        """Test trade execution when creating a new position."""
        # Setup
        order_id = sample_order.id
        fill_price = Decimal("150.50")
        fill_quantity = Decimal("100")

        updated_order = MagicMock()
        updated_position = MagicMock()

        async def mock_trade_operation(operation):
            # Mock the unit of work within the operation
            mock_inner_uow = AsyncMock()
            mock_inner_uow.orders.get_order_by_id.return_value = sample_order
            mock_inner_uow.positions.get_position_by_symbol.return_value = None
            mock_inner_uow.orders.update_order.return_value = updated_order
            mock_inner_uow.positions.persist_position.return_value = updated_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_trade_operation

        # Execute
        result_order, result_position = await trading_service.execute_trade(
            order_id, fill_price, fill_quantity
        )

        # Verify
        assert result_order == updated_order
        assert result_position == updated_position
        mock_tx_manager.execute_in_transaction.assert_called_once()

    async def test_execute_trade_with_existing_buy_position(
        self, trading_service, mock_tx_manager, sample_order, sample_position
    ):
        """Test trade execution with existing position for buy order."""
        # Setup
        order_id = sample_order.id
        fill_price = Decimal("160.00")
        fill_quantity = Decimal("50")

        updated_order = MagicMock()
        updated_position = MagicMock()

        async def mock_trade_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.orders.get_order_by_id.return_value = sample_order
            mock_inner_uow.positions.get_position_by_symbol.return_value = sample_position
            mock_inner_uow.orders.update_order.return_value = updated_order
            mock_inner_uow.positions.persist_position.return_value = updated_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_trade_operation

        # Execute
        result_order, result_position = await trading_service.execute_trade(
            order_id, fill_price, fill_quantity
        )

        # Verify
        assert result_order == updated_order
        assert result_position == updated_position

    async def test_execute_trade_with_existing_sell_position(
        self, trading_service, mock_tx_manager, sample_position
    ):
        """Test trade execution with existing position for sell order."""
        # Setup
        sell_request = OrderRequest(symbol="AAPL", quantity=Decimal("50"), side=OrderSide.SELL)
        sell_order = Order.create_market_order(sell_request)
        sell_order.submit("test_broker_id")
        order_id = sell_order.id
        fill_price = Decimal("160.00")
        fill_quantity = Decimal("50")

        updated_order = MagicMock()
        updated_position = MagicMock()

        async def mock_trade_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.orders.get_order_by_id.return_value = sell_order
            mock_inner_uow.positions.get_position_by_symbol.return_value = sample_position
            mock_inner_uow.orders.update_order.return_value = updated_order
            mock_inner_uow.positions.persist_position.return_value = updated_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_trade_operation

        # Execute
        result_order, result_position = await trading_service.execute_trade(
            order_id, fill_price, fill_quantity
        )

        # Verify
        assert result_order == updated_order
        assert result_position == updated_position

    async def test_execute_trade_order_not_found(self, trading_service, mock_tx_manager):
        """Test trade execution when order is not found."""
        # Setup
        order_id = uuid4()

        async def mock_trade_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.orders.get_order_by_id.return_value = None
            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_trade_operation

        # Execute & Verify
        with pytest.raises(OrderNotFoundError):
            await trading_service.execute_trade(order_id, Decimal("150.00"), Decimal("100"))

    async def test_close_position_completely_long_position(
        self, trading_service, mock_tx_manager, sample_position
    ):
        """Test closing a long position completely."""
        # Setup
        exit_price = Decimal("160.00")
        symbol = "AAPL"

        saved_order = MagicMock()
        updated_position = MagicMock()

        async def mock_close_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.positions.get_position_by_symbol.return_value = sample_position
            mock_inner_uow.orders.save_order.return_value = saved_order
            mock_inner_uow.positions.update_position.return_value = updated_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_close_operation

        # Execute
        result_position, result_order = await trading_service.close_position_completely(
            symbol, exit_price
        )

        # Verify
        assert result_position == updated_position
        assert result_order == saved_order
        mock_tx_manager.execute_in_transaction.assert_called_once()

    async def test_close_position_completely_short_position(self, trading_service, mock_tx_manager):
        """Test closing a short position completely."""
        # Setup
        short_position = Position.open_position(
            symbol="AAPL", quantity=Decimal("-100"), entry_price=Decimal("150.00")
        )
        exit_price = Decimal("140.00")
        symbol = "AAPL"

        saved_order = MagicMock()
        updated_position = MagicMock()

        async def mock_close_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.positions.get_position_by_symbol.return_value = short_position
            mock_inner_uow.orders.save_order.return_value = saved_order
            mock_inner_uow.positions.update_position.return_value = updated_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_close_operation

        # Execute
        result_position, result_order = await trading_service.close_position_completely(
            symbol, exit_price
        )

        # Verify
        assert result_position == updated_position
        assert result_order == saved_order

    async def test_close_position_not_found(self, trading_service, mock_tx_manager):
        """Test closing position when position is not found."""
        # Setup
        symbol = "AAPL"
        exit_price = Decimal("160.00")

        async def mock_close_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.positions.get_position_by_symbol.return_value = None
            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_close_operation

        # Execute & Verify
        with pytest.raises(PositionNotFoundError):
            await trading_service.close_position_completely(symbol, exit_price)

    async def test_close_position_already_closed(self, trading_service, mock_tx_manager):
        """Test closing position when position is already closed."""
        # Setup
        closed_position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        closed_position.close_position(Decimal("160.00"))

        symbol = "AAPL"
        exit_price = Decimal("160.00")

        async def mock_close_operation(operation):
            mock_inner_uow = AsyncMock()
            mock_inner_uow.positions.get_position_by_symbol.return_value = closed_position
            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_close_operation

        # Execute & Verify
        with pytest.raises(PositionNotFoundError):
            await trading_service.close_position_completely(symbol, exit_price)

    @pytest.mark.skip(reason="implementation needs update")
    async def test_get_portfolio_summary_success(
        self, trading_service, mock_uow, sample_portfolio, sample_position, sample_order
    ):
        """Test getting portfolio summary successfully."""
        # Setup
        portfolio_id = sample_portfolio.id

        active_positions = [sample_position]
        recent_orders = [sample_order]

        mock_uow.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_uow.positions.get_active_positions.return_value = active_positions
        mock_uow.orders.get_orders_by_date_range.return_value = recent_orders

        # Execute
        result = await trading_service.get_portfolio_summary(portfolio_id)

        # Verify structure
        assert "portfolio" in result
        assert "active_positions" in result
        assert "recent_orders" in result
        assert "summary" in result

        # Verify portfolio data
        assert result["portfolio"] == sample_portfolio.to_dict()

        # Verify positions data
        assert len(result["active_positions"]) == 1
        pos_data = result["active_positions"][0]
        assert pos_data["symbol"] == "AAPL"
        assert pos_data["quantity"] == 100.0

        # Verify orders data
        assert len(result["recent_orders"]) == 1
        order_data = result["recent_orders"][0]
        assert order_data["symbol"] == "AAPL"
        assert order_data["side"] == OrderSide.BUY

        # Verify summary data
        summary = result["summary"]
        assert "total_value" in summary
        assert "cash_balance" in summary

    async def test_get_portfolio_summary_portfolio_not_found(self, trading_service, mock_uow):
        """Test getting portfolio summary when portfolio is not found."""
        # Setup
        portfolio_id = uuid4()
        mock_uow.portfolios.get_portfolio_by_id.return_value = None

        # Execute & Verify
        with pytest.raises(PositionNotFoundError):
            await trading_service.get_portfolio_summary(portfolio_id)

    async def test_bulk_update_positions_success(
        self, trading_service, mock_tx_manager, sample_position
    ):
        """Test bulk position price updates."""
        # Setup
        price_updates = {"AAPL": Decimal("160.00"), "TSLA": Decimal("250.00")}

        updated_positions = [sample_position]

        async def mock_bulk_operation(operation):
            mock_inner_uow = AsyncMock()

            # Mock position retrieval - AAPL exists, TSLA doesn't
            def mock_get_position(symbol):
                if symbol == "AAPL":
                    return sample_position
                return None

            mock_inner_uow.positions.get_position_by_symbol.side_effect = mock_get_position
            mock_inner_uow.positions.update_position.return_value = sample_position

            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_bulk_operation

        # Execute
        result = await trading_service.bulk_update_positions(price_updates)

        # Verify
        assert len(result) == 1
        assert result[0] == sample_position
        mock_tx_manager.execute_in_transaction.assert_called_once()

    async def test_bulk_update_positions_empty_updates(self, trading_service, mock_tx_manager):
        """Test bulk position updates with empty price updates."""
        # Setup
        price_updates = {}

        async def mock_bulk_operation(operation):
            mock_inner_uow = AsyncMock()
            return await operation(mock_inner_uow)

        mock_tx_manager.execute_in_transaction.side_effect = mock_bulk_operation

        # Execute
        result = await trading_service.bulk_update_positions(price_updates)

        # Verify
        assert result == []

    async def test_get_trading_metrics_success(
        self, trading_service, mock_uow, sample_order, sample_position
    ):
        """Test getting trading metrics for a symbol."""
        # Setup
        symbol = "AAPL"

        # Create different order statuses
        filled_order = sample_order
        filled_order.status = OrderStatus.FILLED

        cancelled_request = OrderRequest(symbol="AAPL", quantity=Decimal("50"), side=OrderSide.SELL)
        cancelled_order = Order.create_market_order(cancelled_request)
        cancelled_order.submit("test_broker_id")
        cancelled_order.cancel("Test cancellation")

        all_orders = [filled_order, cancelled_order]

        # Create closed position with profit
        closed_position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("100.00")
        )
        closed_position.close_position(Decimal("120.00"))
        closed_position._realized_pnl = Decimal("2000.00")  # Mock positive PnL

        all_positions = [sample_position, closed_position]

        mock_uow.orders.get_orders_by_symbol.return_value = all_orders
        mock_uow.positions.get_positions_by_symbol.return_value = all_positions

        # Execute
        result = await trading_service.get_trading_metrics(symbol)

        # Verify structure
        assert result["symbol"] == symbol
        assert "order_stats" in result
        assert "position_stats" in result
        assert "pnl_stats" in result

        # Verify order stats
        order_stats = result["order_stats"]
        assert order_stats["total_orders"] == 2
        assert order_stats["filled_orders"] == 1
        assert order_stats["cancelled_orders"] == 1
        assert order_stats["fill_rate"] == 0.5

        # Verify position stats
        position_stats = result["position_stats"]
        assert position_stats["total_positions"] == 2
        assert position_stats["closed_positions"] == 1
        assert position_stats["winning_positions"] == 1
        assert position_stats["win_rate"] == 1.0

        # Verify PnL stats
        pnl_stats = result["pnl_stats"]
        assert pnl_stats["total_realized_pnl"] == 2000.0
        assert pnl_stats["average_win"] == 2000.0
        assert pnl_stats["average_loss"] == 0.0

    async def test_get_trading_metrics_no_data(self, trading_service, mock_uow):
        """Test getting trading metrics when no data exists."""
        # Setup
        symbol = "AAPL"
        mock_uow.orders.get_orders_by_symbol.return_value = []
        mock_uow.positions.get_positions_by_symbol.return_value = []

        # Execute
        result = await trading_service.get_trading_metrics(symbol)

        # Verify
        assert result["symbol"] == symbol
        assert result["order_stats"]["total_orders"] == 0
        assert result["order_stats"]["fill_rate"] == 0
        assert result["position_stats"]["total_positions"] == 0
        assert result["position_stats"]["win_rate"] == 0

    async def test_get_trading_metrics_with_losses(self, trading_service, mock_uow):
        """Test trading metrics calculation with losing positions."""
        # Setup
        symbol = "AAPL"

        # Create positions with losses
        losing_position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        losing_position.close_position(Decimal("120.00"))
        losing_position._realized_pnl = Decimal("-3000.00")  # Mock negative PnL

        winning_position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("100.00")
        )
        winning_position.close_position(Decimal("130.00"))
        winning_position._realized_pnl = Decimal("3000.00")  # Mock positive PnL

        all_positions = [losing_position, winning_position]

        mock_uow.orders.get_orders_by_symbol.return_value = []
        mock_uow.positions.get_positions_by_symbol.return_value = all_positions

        # Execute
        result = await trading_service.get_trading_metrics(symbol)

        # Verify PnL calculations
        pnl_stats = result["pnl_stats"]
        assert pnl_stats["total_realized_pnl"] == 0.0  # 3000 - 3000
        assert pnl_stats["average_win"] == 3000.0
        assert pnl_stats["average_loss"] == -3000.0

        # Verify position stats
        position_stats = result["position_stats"]
        assert position_stats["winning_positions"] == 1
        assert position_stats["win_rate"] == 0.5


class TestExampleUsagePatterns:
    """Test the example usage patterns function."""

    async def test_example_usage_patterns_runs_without_error(self):
        """Test that the example usage patterns function runs without error."""
        # This is a documentation function that should not raise exceptions
        await example_usage_patterns()
        # If we get here without exception, the test passes
        assert True

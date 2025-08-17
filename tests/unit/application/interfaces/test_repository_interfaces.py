"""
Unit tests for repository interface contracts.

Tests the repository interfaces to ensure they define proper contracts
and that implementations satisfy the expected behaviors.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import (
    OrderNotFoundError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    RepositoryError,
)
from src.application.interfaces.repositories import (
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position, PositionSide


class MockOrderRepository:
    """Mock implementation of IOrderRepository for interface testing."""

    def __init__(self):
        self.orders = {}
        self.call_log = []

    async def save_order(self, order: Order) -> Order:
        self.call_log.append(("save_order", order.id))
        self.orders[order.id] = order
        return order

    async def get_order_by_id(self, order_id) -> Order | None:
        self.call_log.append(("get_order_by_id", order_id))
        return self.orders.get(order_id)

    async def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        self.call_log.append(("get_orders_by_symbol", symbol))
        return [order for order in self.orders.values() if order.symbol.value == symbol]

    async def get_orders_by_status(self, status: OrderStatus) -> list[Order]:
        self.call_log.append(("get_orders_by_status", status))
        return [order for order in self.orders.values() if order.status == status]

    async def get_active_orders(self) -> list[Order]:
        self.call_log.append(("get_active_orders",))
        return [order for order in self.orders.values() if order.is_active()]

    async def get_orders_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[Order]:
        self.call_log.append(("get_orders_by_date_range", start_date, end_date))
        return [
            order for order in self.orders.values() if start_date <= order.created_at <= end_date
        ]

    async def update_order(self, order: Order) -> Order:
        self.call_log.append(("update_order", order.id))
        if order.id not in self.orders:
            raise OrderNotFoundError(order.id)
        self.orders[order.id] = order
        return order

    async def delete_order(self, order_id) -> bool:
        self.call_log.append(("delete_order", order_id))
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    async def get_orders_by_broker_id(self, broker_order_id: str) -> list[Order]:
        self.call_log.append(("get_orders_by_broker_id", broker_order_id))
        return [order for order in self.orders.values() if order.broker_order_id == broker_order_id]


class MockPositionRepository:
    """Mock implementation of IPositionRepository for interface testing."""

    def __init__(self):
        self.positions = {}
        self.call_log = []

    async def save_position(self, position: Position) -> Position:
        self.call_log.append(("save_position", position.id))
        self.positions[position.id] = position
        return position

    async def get_position_by_id(self, position_id) -> Position | None:
        self.call_log.append(("get_position_by_id", position_id))
        return self.positions.get(position_id)

    async def get_position_by_symbol(self, symbol: str) -> Position | None:
        self.call_log.append(("get_position_by_symbol", symbol))
        for position in self.positions.values():
            if position.symbol == symbol and not position.is_closed():
                return position
        return None

    async def get_positions_by_symbol(self, symbol: str) -> list[Position]:
        self.call_log.append(("get_positions_by_symbol", symbol))
        return [pos for pos in self.positions.values() if pos.symbol == symbol]

    async def get_active_positions(self) -> list[Position]:
        self.call_log.append(("get_active_positions",))
        return [pos for pos in self.positions.values() if not pos.is_closed()]

    async def get_closed_positions(self) -> list[Position]:
        self.call_log.append(("get_closed_positions",))
        return [pos for pos in self.positions.values() if pos.is_closed()]

    async def get_positions_by_strategy(self, strategy: str) -> list[Position]:
        self.call_log.append(("get_positions_by_strategy", strategy))
        return [pos for pos in self.positions.values() if pos.strategy == strategy]

    async def get_positions_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[Position]:
        self.call_log.append(("get_positions_by_date_range", start_date, end_date))
        return [pos for pos in self.positions.values() if start_date <= pos.opened_at <= end_date]

    async def update_position(self, position: Position) -> Position:
        self.call_log.append(("update_position", position.id))
        if position.id not in self.positions:
            raise PositionNotFoundError(position.id)
        self.positions[position.id] = position
        return position

    async def close_position(self, position_id) -> bool:
        self.call_log.append(("close_position", position_id))
        if position_id in self.positions:
            position = self.positions[position_id]
            position.close(Decimal("0"), datetime.now(UTC))
            return True
        return False

    async def delete_position(self, position_id) -> bool:
        self.call_log.append(("delete_position", position_id))
        if position_id in self.positions:
            del self.positions[position_id]
            return True
        return False


class MockPortfolioRepository:
    """Mock implementation of IPortfolioRepository for interface testing."""

    def __init__(self):
        self.portfolios = {}
        self.call_log = []

    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        self.call_log.append(("save_portfolio", portfolio.id))
        self.portfolios[portfolio.id] = portfolio
        return portfolio

    async def get_portfolio_by_id(self, portfolio_id) -> Portfolio | None:
        self.call_log.append(("get_portfolio_by_id", portfolio_id))
        return self.portfolios.get(portfolio_id)

    async def get_portfolio_by_name(self, name: str) -> Portfolio | None:
        self.call_log.append(("get_portfolio_by_name", name))
        for portfolio in self.portfolios.values():
            if portfolio.name == name:
                return portfolio
        return None

    async def get_current_portfolio(self) -> Portfolio | None:
        self.call_log.append(("get_current_portfolio",))
        # Return first portfolio as "current" for testing
        return next(iter(self.portfolios.values()), None)

    async def get_all_portfolios(self) -> list[Portfolio]:
        self.call_log.append(("get_all_portfolios",))
        return list(self.portfolios.values())

    async def get_portfolios_by_strategy(self, strategy: str) -> list[Portfolio]:
        self.call_log.append(("get_portfolios_by_strategy", strategy))
        return [p for p in self.portfolios.values() if p.strategy == strategy]

    async def get_portfolio_history(
        self, portfolio_id, start_date: datetime, end_date: datetime
    ) -> list[Portfolio]:
        self.call_log.append(("get_portfolio_history", portfolio_id, start_date, end_date))
        return []  # Simplified for testing

    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        self.call_log.append(("update_portfolio", portfolio.id))
        if portfolio.id not in self.portfolios:
            raise PortfolioNotFoundError(portfolio.id)
        self.portfolios[portfolio.id] = portfolio
        return portfolio

    async def delete_portfolio(self, portfolio_id) -> bool:
        self.call_log.append(("delete_portfolio", portfolio_id))
        if portfolio_id in self.portfolios:
            del self.portfolios[portfolio_id]
            return True
        return False

    async def create_portfolio_snapshot(self, portfolio: Portfolio) -> Portfolio:
        self.call_log.append(("create_portfolio_snapshot", portfolio.id))
        snapshot = Portfolio(
            id=uuid4(),
            name=f"{portfolio.name}_snapshot",
            cash_balance=portfolio.cash_balance,
            strategy=portfolio.strategy,
            created_at=datetime.now(UTC),
        )
        self.portfolios[snapshot.id] = snapshot
        return snapshot


@pytest.mark.unit
class TestOrderRepositoryInterface:
    """Test IOrderRepository interface contract."""

    @pytest.fixture
    def repository(self):
        return MockOrderRepository()

    @pytest.fixture
    def sample_order(self):
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.symbol import Symbol
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        request = OrderRequest(
            symbol=Symbol("AAPL"),
            quantity=Quantity("100"),
            side=OrderSide.BUY,
            limit_price=Price("150.00"),
            reason="Test order",
        )
        return Order.create_limit_order(request)

    async def test_save_order_success(self, repository, sample_order):
        """Test successful order save."""
        result = await repository.save_order(sample_order)

        assert result == sample_order
        assert ("save_order", sample_order.id) in repository.call_log
        assert sample_order.id in repository.orders

    async def test_get_order_by_id_found(self, repository, sample_order):
        """Test get order by ID when order exists."""
        await repository.save_order(sample_order)

        result = await repository.get_order_by_id(sample_order.id)

        assert result == sample_order
        assert ("get_order_by_id", sample_order.id) in repository.call_log

    async def test_get_order_by_id_not_found(self, repository):
        """Test get order by ID when order doesn't exist."""
        non_existent_id = uuid4()

        result = await repository.get_order_by_id(non_existent_id)

        assert result is None
        assert ("get_order_by_id", non_existent_id) in repository.call_log

    async def test_get_orders_by_symbol(self, repository, sample_order):
        """Test get orders by symbol."""
        await repository.save_order(sample_order)

        # Create another order with different symbol
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.symbol import Symbol
        from src.domain.value_objects.quantity import Quantity
        
        other_request = OrderRequest(
            symbol=Symbol("GOOGL"),
            quantity=Quantity("50"),
            side=OrderSide.SELL,
        )
        other_order = Order.create_market_order(other_request)
        await repository.save_order(other_order)

        result = await repository.get_orders_by_symbol("AAPL")

        assert len(result) == 1
        assert result[0] == sample_order
        assert ("get_orders_by_symbol", "AAPL") in repository.call_log

    async def test_get_orders_by_status(self, repository, sample_order):
        """Test get orders by status."""
        await repository.save_order(sample_order)

        result = await repository.get_orders_by_status(OrderStatus.PENDING)

        assert len(result) == 1
        assert result[0] == sample_order
        assert ("get_orders_by_status", OrderStatus.PENDING) in repository.call_log

    async def test_get_active_orders(self, repository, sample_order):
        """Test get active orders."""
        await repository.save_order(sample_order)

        result = await repository.get_active_orders()

        assert len(result) == 1
        assert result[0] == sample_order
        assert ("get_active_orders",) in repository.call_log

    async def test_update_order_success(self, repository, sample_order):
        """Test successful order update."""
        await repository.save_order(sample_order)
        sample_order.submit("broker123")

        result = await repository.update_order(sample_order)

        assert result == sample_order
        assert repository.orders[sample_order.id].status == OrderStatus.SUBMITTED
        assert ("update_order", sample_order.id) in repository.call_log

    async def test_update_order_not_found(self, repository, sample_order):
        """Test update order when order doesn't exist."""
        with pytest.raises(OrderNotFoundError):
            await repository.update_order(sample_order)

    async def test_delete_order_success(self, repository, sample_order):
        """Test successful order deletion."""
        await repository.save_order(sample_order)

        result = await repository.delete_order(sample_order.id)

        assert result is True
        assert sample_order.id not in repository.orders
        assert ("delete_order", sample_order.id) in repository.call_log

    async def test_delete_order_not_found(self, repository):
        """Test delete order when order doesn't exist."""
        non_existent_id = uuid4()

        result = await repository.delete_order(non_existent_id)

        assert result is False
        assert ("delete_order", non_existent_id) in repository.call_log

    async def test_get_orders_by_broker_id(self, repository, sample_order):
        """Test get orders by broker ID."""
        sample_order.submit("broker123")
        await repository.save_order(sample_order)

        result = await repository.get_orders_by_broker_id("broker123")

        assert len(result) == 1
        assert result[0] == sample_order
        assert ("get_orders_by_broker_id", "broker123") in repository.call_log


@pytest.mark.unit
class TestPositionRepositoryInterface:
    """Test IPositionRepository interface contract."""

    @pytest.fixture
    def repository(self):
        return MockPositionRepository()

    @pytest.fixture
    def sample_position(self):
        return Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),  # Positive for long
            average_entry_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="test_strategy",
        )

    async def test_save_position_success(self, repository, sample_position):
        """Test successful position save."""
        result = await repository.save_position(sample_position)

        assert result == sample_position
        assert ("save_position", sample_position.id) in repository.call_log
        assert sample_position.id in repository.positions

    async def test_get_position_by_id_found(self, repository, sample_position):
        """Test get position by ID when position exists."""
        await repository.save_position(sample_position)

        result = await repository.get_position_by_id(sample_position.id)

        assert result == sample_position
        assert ("get_position_by_id", sample_position.id) in repository.call_log

    async def test_get_position_by_symbol(self, repository, sample_position):
        """Test get current position by symbol."""
        await repository.save_position(sample_position)

        result = await repository.get_position_by_symbol("AAPL")

        assert result == sample_position
        assert ("get_position_by_symbol", "AAPL") in repository.call_log

    async def test_get_active_positions(self, repository, sample_position):
        """Test get active positions."""
        await repository.save_position(sample_position)

        result = await repository.get_active_positions()

        assert len(result) == 1
        assert result[0] == sample_position
        assert ("get_active_positions",) in repository.call_log

    async def test_close_position_success(self, repository, sample_position):
        """Test successful position closure."""
        await repository.save_position(sample_position)

        result = await repository.close_position(sample_position.id)

        assert result is True
        assert repository.positions[sample_position.id].is_closed()
        assert ("close_position", sample_position.id) in repository.call_log


@pytest.mark.unit
class TestPortfolioRepositoryInterface:
    """Test IPortfolioRepository interface contract."""

    @pytest.fixture
    def repository(self):
        return MockPortfolioRepository()

    @pytest.fixture
    def sample_portfolio(self):
        return Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="test_strategy",
            created_at=datetime.now(UTC),
        )

    async def test_save_portfolio_success(self, repository, sample_portfolio):
        """Test successful portfolio save."""
        result = await repository.save_portfolio(sample_portfolio)

        assert result == sample_portfolio
        assert ("save_portfolio", sample_portfolio.id) in repository.call_log
        assert sample_portfolio.id in repository.portfolios

    async def test_get_portfolio_by_name(self, repository, sample_portfolio):
        """Test get portfolio by name."""
        await repository.save_portfolio(sample_portfolio)

        result = await repository.get_portfolio_by_name("Test Portfolio")

        assert result == sample_portfolio
        assert ("get_portfolio_by_name", "Test Portfolio") in repository.call_log

    async def test_get_all_portfolios(self, repository, sample_portfolio):
        """Test get all portfolios."""
        await repository.save_portfolio(sample_portfolio)

        result = await repository.get_all_portfolios()

        assert len(result) == 1
        assert result[0] == sample_portfolio
        assert ("get_all_portfolios",) in repository.call_log

    async def test_create_portfolio_snapshot(self, repository, sample_portfolio):
        """Test create portfolio snapshot."""
        await repository.save_portfolio(sample_portfolio)

        snapshot = await repository.create_portfolio_snapshot(sample_portfolio)

        assert snapshot.name == "Test Portfolio_snapshot"
        assert snapshot.cash_balance == sample_portfolio.cash_balance
        assert snapshot.id != sample_portfolio.id
        assert ("create_portfolio_snapshot", sample_portfolio.id) in repository.call_log


@pytest.mark.unit
class TestRepositoryErrorHandling:
    """Test repository interface error handling."""

    def test_repository_error_inheritance(self):
        """Test that repository errors inherit properly."""
        error = RepositoryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_entity_not_found_errors(self):
        """Test entity-specific not found errors."""
        order_id = uuid4()
        position_id = uuid4()
        portfolio_id = uuid4()

        order_error = OrderNotFoundError(order_id)
        position_error = PositionNotFoundError(position_id)
        portfolio_error = PortfolioNotFoundError(portfolio_id)

        assert str(order_id) in str(order_error)
        assert str(position_id) in str(position_error)
        assert str(portfolio_id) in str(portfolio_error)

        assert isinstance(order_error, RepositoryError)
        assert isinstance(position_error, RepositoryError)
        assert isinstance(portfolio_error, RepositoryError)


@pytest.mark.unit
class TestRepositoryTypeAnnotations:
    """Test repository interface type annotations."""

    def test_order_repository_annotations(self):
        """Test IOrderRepository has correct type annotations."""
        # This test ensures the interface defines proper type hints

        # Check that the protocol has the expected methods
        expected_methods = [
            "save_order",
            "get_order_by_id",
            "get_orders_by_symbol",
            "get_orders_by_status",
            "get_active_orders",
            "get_orders_by_date_range",
            "update_order",
            "delete_order",
            "get_orders_by_broker_id",
        ]

        for method in expected_methods:
            assert hasattr(IOrderRepository, method)

    def test_position_repository_annotations(self):
        """Test IPositionRepository has correct type annotations."""

        expected_methods = [
            "save_position",
            "get_position_by_id",
            "get_position_by_symbol",
            "get_positions_by_symbol",
            "get_active_positions",
            "get_closed_positions",
            "get_positions_by_strategy",
            "get_positions_by_date_range",
            "update_position",
            "close_position",
            "delete_position",
        ]

        for method in expected_methods:
            assert hasattr(IPositionRepository, method)

    def test_portfolio_repository_annotations(self):
        """Test IPortfolioRepository has correct type annotations."""

        expected_methods = [
            "save_portfolio",
            "get_portfolio_by_id",
            "get_portfolio_by_name",
            "get_current_portfolio",
            "get_all_portfolios",
            "get_portfolios_by_strategy",
            "get_portfolio_history",
            "update_portfolio",
            "delete_portfolio",
            "create_portfolio_snapshot",
        ]

        for method in expected_methods:
            assert hasattr(IPortfolioRepository, method)

"""Mock fixtures for testing."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.interfaces.broker import IBroker
from src.application.interfaces.repositories import (
    IMarketDataRepository,
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity

# ============================================================================
# VALUE OBJECT FACTORIES
# ============================================================================


def money(value: str | int | float | Decimal) -> Money:
    """Create a Money value object from various input types."""
    return Money(Decimal(str(value)))


def price(value: str | int | float | Decimal) -> Price:
    """Create a Price value object from various input types."""
    return Price(Decimal(str(value)))


def quantity(value: str | int | float | Decimal) -> Quantity:
    """Create a Quantity value object from various input types."""
    return Quantity(Decimal(str(value)))


# ============================================================================
# ENTITY BUILDERS
# ============================================================================


def create_test_order(
    symbol: str = "AAPL",
    qty: str | int | float | Decimal = 100,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    limit_price: str | int | float | Decimal | None = None,
    stop_price: str | int | float | Decimal | None = None,
    status: OrderStatus = OrderStatus.PENDING,
    time_in_force: TimeInForce = TimeInForce.DAY,
    reason: str = "Test order",
    **kwargs: Any,
) -> Order:
    """Create a test Order entity with proper value objects."""
    order_data = {
        "symbol": symbol,
        "quantity": quantity(qty),
        "side": side,
        "order_type": order_type,
        "status": status,
        "time_in_force": time_in_force,
        "reason": reason,
    }

    if limit_price is not None:
        order_data["limit_price"] = price(limit_price)

    if stop_price is not None:
        order_data["stop_price"] = price(stop_price)

    # Apply any additional overrides
    order_data.update(kwargs)

    return Order(**order_data)


def create_test_portfolio(
    name: str = "test_portfolio",
    initial_capital: str | int | float | Decimal = 100000,
    cash_balance: str | int | float | Decimal = 100000,
    max_position_size: str | int | float | Decimal = 10000,
    max_portfolio_risk: str | float | Decimal = 0.02,
    max_positions: int = 10,
    max_leverage: str | float | Decimal = 1.0,
    **kwargs: Any,
) -> Portfolio:
    """Create a test Portfolio entity with proper value objects."""
    portfolio_data = {
        "name": name,
        "initial_capital": money(initial_capital),
        "cash_balance": money(cash_balance),
        "max_position_size": money(max_position_size),
        "max_portfolio_risk": Decimal(str(max_portfolio_risk)),
        "max_positions": max_positions,
        "max_leverage": Decimal(str(max_leverage)),
    }

    # Apply any additional overrides
    portfolio_data.update(kwargs)

    return Portfolio(**portfolio_data)


def create_test_position(
    symbol: str = "AAPL",
    qty: str | int | float | Decimal = 100,
    entry_price: str | int | float | Decimal = 150.00,
    current_price: str | int | float | Decimal = 155.00,
    stop_loss: str | int | float | Decimal | None = None,
    take_profit: str | int | float | Decimal | None = None,
    **kwargs: Any,
) -> Position:
    """Create a test Position entity with proper value objects."""
    position_data = {
        "symbol": symbol,
        "quantity": quantity(qty),
        "entry_price": price(entry_price),
        "current_price": price(current_price),
    }

    if stop_loss is not None:
        position_data["stop_loss"] = price(stop_loss)

    if take_profit is not None:
        position_data["take_profit"] = price(take_profit)

    # Apply any additional overrides
    position_data.update(kwargs)

    return Position(**position_data)


# ============================================================================
# MOCK REPOSITORIES
# ============================================================================


def create_mock_order_repository(**overrides: Any) -> IOrderRepository:
    """Create a mock order repository with sensible defaults."""
    mock = AsyncMock(spec=IOrderRepository)

    # Default behaviors
    mock.save_order.return_value = None
    mock.get_order_by_id.return_value = None
    mock.get_orders_by_symbol.return_value = []
    mock.get_orders_by_status.return_value = []
    mock.get_active_orders.return_value = []
    mock.update_order.return_value = None
    mock.delete_order.return_value = None
    mock.get_orders_by_broker_id.return_value = []

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock


def create_mock_position_repository(**overrides: Any) -> IPositionRepository:
    """Create a mock position repository with sensible defaults."""
    mock = AsyncMock(spec=IPositionRepository)

    # Default behaviors
    mock.persist_position.return_value = None
    mock.get_position_by_id.return_value = None
    mock.get_position_by_symbol.return_value = None
    mock.get_active_positions.return_value = []
    mock.close_position.return_value = None
    mock.update_position.return_value = None

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock


def create_mock_portfolio_repository(**overrides: Any) -> IPortfolioRepository:
    """Create a mock portfolio repository with sensible defaults."""
    mock = AsyncMock(spec=IPortfolioRepository)

    # Default behaviors
    mock.save_portfolio.return_value = None
    mock.get_portfolio_by_id.return_value = None
    mock.get_portfolio_by_name.return_value = None
    mock.get_all_portfolios.return_value = []
    mock.update_portfolio.return_value = None
    mock.delete_portfolio.return_value = None
    mock.create_portfolio_snapshot.return_value = None

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock


def create_mock_market_data_repository(**overrides: Any) -> IMarketDataRepository:
    """Create a mock market data repository with sensible defaults."""
    mock = AsyncMock(spec=IMarketDataRepository)

    # Default behaviors
    mock.get_current_price.return_value = Price(Decimal("150.00"))
    mock.get_historical_prices.return_value = []
    mock.get_market_hours.return_value = {
        "open": datetime.now(UTC).replace(hour=9, minute=30),
        "close": datetime.now(UTC).replace(hour=16, minute=0),
    }

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock


def create_mock_broker(**overrides: Any) -> IBroker:
    """Create a mock broker with sensible defaults."""
    mock = AsyncMock(spec=IBroker)

    # Default behaviors
    mock.submit_order.return_value = str(uuid4())
    mock.cancel_order.return_value = True
    mock.get_order_status.return_value = OrderStatus.PENDING
    mock.get_account_balance.return_value = Money(Decimal("100000.00"))
    mock.get_positions.return_value = []
    mock.get_market_status.return_value = "open"
    mock.validate_order.return_value = True

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock


def create_mock_database_adapter(**overrides: Any) -> Any:
    """Create a mock database adapter for infrastructure tests."""
    mock = AsyncMock()

    # Default behaviors
    mock.execute.return_value = None
    mock.fetch_one.return_value = None
    mock.fetch_all.return_value = []
    mock.fetch_value.return_value = None
    mock.fetch_values.return_value = []
    mock.begin_transaction.return_value = AsyncMock()

    # Transaction context manager
    transaction = AsyncMock()
    transaction.__aenter__.return_value = transaction
    transaction.__aexit__.return_value = None
    transaction.commit.return_value = None
    transaction.rollback.return_value = None
    mock.transaction.return_value = transaction

    # Apply overrides
    for attr, value in overrides.items():
        setattr(mock, attr, value)

    return mock

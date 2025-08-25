"""Mock fixtures for testing."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.interfaces.brokers import IBroker
from src.application.interfaces.repositories import (
    IMarketDataRepository,
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.domain.entities.order import OrderStatus
from src.domain.value_objects import Money, Price


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

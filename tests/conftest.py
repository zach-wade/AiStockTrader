"""Global pytest configuration and fixtures."""

# Standard library imports
import asyncio
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

# Load test environment variables
from dotenv import load_dotenv

test_env_path = Path(__file__).parent.parent / ".env.test"
if test_env_path.exists():
    load_dotenv(test_env_path, override=True)

# Third-party imports
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
# Import domain entities for fixtures
from src.domain.entities.order import Order, OrderSide
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position, PositionSide


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Provides mock configuration for tests."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "test_user",
            "password": "test_password",
        },
        "api_keys": {
            "polygon": "test_polygon_key",
            "alpaca": "test_alpaca_key",
        },
        "risk": {
            "max_position_size": Decimal("10000"),
            "max_portfolio_risk": Decimal("0.02"),
            "stop_loss_percent": Decimal("0.05"),
        },
        "trading": {
            "mode": "paper",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "strategies": ["momentum", "mean_reversion"],
        },
    }


@pytest.fixture
def mock_db_adapter() -> AsyncMock:
    """Provides mock database adapter."""
    adapter = AsyncMock()
    adapter.fetch_one.return_value = {"id": 1, "symbol": "AAPL", "price": 150.0}
    adapter.fetch_all.return_value = [
        {"id": 1, "symbol": "AAPL", "price": 150.0},
        {"id": 2, "symbol": "GOOGL", "price": 2800.0},
    ]
    adapter.execute.return_value = None
    adapter.close.return_value = None
    return adapter


@pytest.fixture
def mock_market_data() -> dict[str, Any]:
    """Provides mock market data."""
    return {
        "symbol": "AAPL",
        "price": Decimal("150.00"),
        "volume": 1000000,
        "bid": Decimal("149.95"),
        "ask": Decimal("150.05"),
        "timestamp": "2025-08-16T10:00:00Z",
        "open": Decimal("148.00"),
        "high": Decimal("151.00"),
        "low": Decimal("147.50"),
        "close": Decimal("150.00"),
    }


@pytest.fixture
def mock_order() -> dict[str, Any]:
    """Provides mock order data."""
    return {
        "id": "order_123",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": Decimal("150.00"),
        "status": "pending",
        "created_at": "2025-08-16T10:00:00Z",
    }


@pytest.fixture
def mock_position() -> dict[str, Any]:
    """Provides mock position data."""
    return {
        "symbol": "AAPL",
        "quantity": 100,
        "avg_price": Decimal("145.00"),
        "current_price": Decimal("150.00"),
        "unrealized_pnl": Decimal("500.00"),
        "realized_pnl": Decimal("0.00"),
    }


@pytest.fixture
def mock_broker() -> AsyncMock:
    """Provides mock broker interface."""
    broker = AsyncMock()
    broker.submit_order.return_value = {"order_id": "order_123", "status": "submitted"}
    broker.cancel_order.return_value = {"order_id": "order_123", "status": "cancelled"}
    broker.get_positions.return_value = []
    broker.get_account.return_value = {
        "cash": Decimal("100000.00"),
        "buying_power": Decimal("200000.00"),
    }
    return broker


@pytest.fixture
def mock_risk_engine() -> Mock:
    """Provides mock risk engine."""
    engine = Mock()
    engine.check_order.return_value = (True, None)
    engine.check_position_size.return_value = (True, None)
    engine.check_portfolio_risk.return_value = (True, None)
    engine.calculate_position_size.return_value = 100
    return engine


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Provides mock event bus."""
    bus = AsyncMock()
    bus.publish.return_value = None
    bus.subscribe.return_value = None
    bus.unsubscribe.return_value = None
    return bus


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton reset logic here
    yield


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_api_response() -> dict[str, Any]:
    """Provides mock API response."""
    return {
        "status": "success",
        "data": {
            "symbol": "AAPL",
            "bars": [
                {
                    "t": "2025-08-16T10:00:00Z",
                    "o": 148.0,
                    "h": 151.0,
                    "l": 147.5,
                    "c": 150.0,
                    "v": 1000000,
                }
            ],
        },
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_db: Tests requiring database")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API")


# Repository and database test fixtures


@pytest.fixture
def test_database_config():
    """Test database configuration."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "ai_trader_test",
        "user": "zachwade",
        "password": "",
        "min_pool_size": 1,
        "max_pool_size": 5,
    }


@pytest.fixture
def mock_connection_pool():
    """Mock psycopg3 connection pool."""
    pool = AsyncMock()
    pool.max_size = 10
    pool.min_size = 1
    pool.closed = False
    return pool


@pytest.fixture
def mock_repository_adapter():
    """Mock database adapter for repository tests."""
    adapter = AsyncMock()
    adapter.execute_query.return_value = "EXECUTE 1"
    adapter.fetch_one.return_value = None
    adapter.fetch_all.return_value = []
    adapter.fetch_values.return_value = []
    adapter.execute_batch.return_value = None
    adapter.begin_transaction.return_value = None
    adapter.commit_transaction.return_value = None
    adapter.rollback_transaction.return_value = None
    adapter.has_active_transaction = False
    adapter.health_check.return_value = True
    return adapter


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order.create_limit_order(
        symbol="AAPL",
        quantity=Decimal("100"),
        side=OrderSide.BUY,
        limit_price=Decimal("150.00"),
        reason="Test order",
    )


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position(
        id=uuid4(),
        symbol="AAPL",
        quantity=Decimal("100"),
        side=PositionSide.LONG,
        average_price=Decimal("145.00"),
        current_price=Decimal("150.00"),
        opened_at=datetime.now(UTC),
        strategy="test_strategy",
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        cash_balance=Decimal("100000.00"),
        strategy="test_strategy",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def order_db_record():
    """Sample order database record."""
    return {
        "id": uuid4(),
        "symbol": "AAPL",
        "side": "buy",
        "order_type": "limit",
        "status": "pending",
        "quantity": Decimal("100"),
        "limit_price": Decimal("150.00"),
        "stop_price": None,
        "time_in_force": "day",
        "broker_order_id": None,
        "filled_quantity": Decimal("0"),
        "average_fill_price": None,
        "created_at": datetime.now(UTC),
        "submitted_at": None,
        "filled_at": None,
        "cancelled_at": None,
        "reason": "Test order",
        "tags": {},
    }


@pytest.fixture
def position_db_record():
    """Sample position database record."""
    return {
        "id": uuid4(),
        "symbol": "AAPL",
        "quantity": Decimal("100"),
        "side": "long",
        "average_price": Decimal("145.00"),
        "current_price": Decimal("150.00"),
        "unrealized_pnl": Decimal("500.00"),
        "realized_pnl": Decimal("0.00"),
        "opened_at": datetime.now(UTC),
        "closed_at": None,
        "strategy": "test_strategy",
        "tags": {},
    }


@pytest.fixture
def portfolio_db_record():
    """Sample portfolio database record."""
    return {
        "id": uuid4(),
        "name": "Test Portfolio",
        "cash_balance": Decimal("100000.00"),
        "total_value": Decimal("110000.00"),
        "unrealized_pnl": Decimal("10000.00"),
        "realized_pnl": Decimal("0.00"),
        "strategy": "test_strategy",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "tags": {},
    }


@pytest.fixture
def test_transaction_manager():
    """Mock transaction manager for testing."""
    manager = AsyncMock()
    manager.execute_in_transaction.return_value = "success"
    manager.execute_with_retry.return_value = "success"
    manager.execute_batch.return_value = ["success", "success"]
    return manager


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["DATABASE_URL"] = "postgresql://zachwade@localhost:5432/ai_trader_test"
    yield
    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("DATABASE_URL", None)

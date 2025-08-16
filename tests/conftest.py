"""Global pytest configuration and fixtures."""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> Dict[str, Any]:
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
def mock_market_data() -> Dict[str, Any]:
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
def mock_order() -> Dict[str, Any]:
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
def mock_position() -> Dict[str, Any]:
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
def mock_api_response() -> Dict[str, Any]:
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


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    import os

    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    # Cleanup
    os.environ.pop("TESTING", None)
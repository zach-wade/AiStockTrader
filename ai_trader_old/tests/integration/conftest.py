"""
Integration Test Configuration and Fixtures

Provides shared fixtures and configuration for integration tests.
"""

# Standard library imports
import asyncio
import os
from pathlib import Path
import shutil

# Add the src directory to the path
import sys
import tempfile

# Third-party imports
import pytest
import pytest_asyncio
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Local imports
from main.config.config_manager import get_config_manager
from main.utils.database import DatabasePool

# from main.data_pipeline.orchestrator import DataPipelineOrchestrator  # Module doesn't exist
# from main.trading_engine.core.execution_engine import ExecutionEngine
# from main.trading_engine.brokers.mock_broker import MockBroker
# from main.models.training.training_orchestrator import ModelTrainingOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config_dir():
    """Create a temporary config directory for tests."""
    temp_dir = tempfile.mkdtemp()

    # Create test configuration
    test_config = {
        "app": {"name": "AI Trader Test", "environment": "test"},
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "ai_trader_test",
            "user": "test_user",
            "password": "test_password",
        },
        "trading": {"mode": "paper", "max_position_size": 1000, "max_portfolio_value": 10000},
        "data_pipeline": {"batch_size": 10, "parallel_workers": 2},
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }

    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config(test_config_dir):
    """Load test configuration."""
    manager = get_config_manager(config_dir=test_config_dir)
    config = manager.load_config(config_name="test_config", env="test")
    return config


@pytest_asyncio.fixture(scope="function")
async def test_database(test_config):
    """Create test database connection pool."""
    db_config = test_config.get("database", {})

    pool = DatabasePool(
        host=db_config.get("host", "localhost"),
        port=db_config.get("port", 5432),
        database=db_config.get("name", "ai_trader_test"),
        user=db_config.get("user", "test_user"),
        password=db_config.get("password", "test_password"),
        min_size=1,
        max_size=5,
    )

    await pool.initialize()

    yield pool

    await pool.close()


@pytest.fixture(scope="function")
def mock_broker():
    """Create a mock broker for testing."""
    broker = MockBroker(config={"initial_cash": 100000, "commission": 0.001, "slippage": 0.0001})

    # Set up some default behavior
    broker.set_response(
        "get_account_info",
        {"cash": 100000, "portfolio_value": 100000, "buying_power": 200000, "margin_used": 0},
    )

    broker.set_response("get_positions", [])
    broker.set_response("get_orders", [])

    return broker


@pytest_asyncio.fixture(scope="function")
async def data_pipeline(test_config):
    """Create test data pipeline orchestrator."""
    # This is a simplified setup - in real tests would need proper mocking
    # Local imports
    from main.data_pipeline.config import DataPipelineConfig

    pipeline_config = DataPipelineConfig(test_config.get("data_pipeline", {}))

    # Create mock components (would need proper implementations)
    # For now, just return None to show the structure


@pytest_asyncio.fixture(scope="function")
async def execution_engine(test_config, mock_broker):
    """Create test execution engine."""
    engine = ExecutionEngine(config=test_config)

    # Initialize with mock broker
    await engine.initialize()

    yield engine

    await engine.shutdown()


@pytest.fixture(scope="function")
def test_symbols():
    """Common test symbols."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


@pytest.fixture(scope="function")
def test_timeframe():
    """Common test timeframe."""
    # Standard library imports
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    return {"start_date": start_date, "end_date": end_date, "lookback_days": 30}


@pytest.fixture(scope="function")
def sample_market_data():
    """Sample market data for testing."""
    # Standard library imports
    from datetime import datetime

    # Third-party imports
    import numpy as np
    import pandas as pd

    # Generate sample OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

    data = []
    for date in dates:
        base_price = 100 + np.random.randn() * 10
        data.append(
            {
                "timestamp": date,
                "open": base_price + np.random.randn(),
                "high": base_price + abs(np.random.randn()) * 2,
                "low": base_price - abs(np.random.randn()) * 2,
                "close": base_price + np.random.randn(),
                "volume": int(1000000 + np.random.randn() * 100000),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def performance_benchmark():
    """Performance benchmarking fixture."""
    # Standard library imports
    import time

    class PerformanceBenchmark:
        def __init__(self):
            self.timings = {}

        def start(self, name: str):
            self.timings[name] = {"start": time.time()}

        def end(self, name: str):
            if name in self.timings:
                self.timings[name]["end"] = time.time()
                self.timings[name]["duration"] = (
                    self.timings[name]["end"] - self.timings[name]["start"]
                )

        def get_timing(self, name: str) -> float | None:
            if name in self.timings and "duration" in self.timings[name]:
                return self.timings[name]["duration"]
            return None

        def report(self):
            for name, timing in self.timings.items():
                if "duration" in timing:
                    print(f"{name}: {timing['duration']:.3f} seconds")

    return PerformanceBenchmark()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup code here


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_db: mark test as requiring database")
    config.addinivalue_line("markers", "requires_market_data: mark test as requiring market data")

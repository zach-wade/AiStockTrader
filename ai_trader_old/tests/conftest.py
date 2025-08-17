"""
Global pytest configuration and shared fixtures for AI Trader test suite.

This file contains fixtures and configuration that are available to all tests
across the entire project.
"""

# Standard library imports
import asyncio
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import pytest

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set test environment variables
os.environ["AI_TRADER_ENV"] = "test"
os.environ["TESTING"] = "1"
os.environ["PYTHONPATH"] = str(src_path)


# Global pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up test environment
    os.environ["AI_TRADER_TEST_MODE"] = "1"


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Auto-mark based on file path
        test_path = str(item.fspath)

        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Module-specific markers
        if "/events/" in test_path:
            item.add_marker(pytest.mark.events)
        elif "/data_pipeline/" in test_path:
            item.add_marker(pytest.mark.data_pipeline)
        elif "/risk/" in test_path:
            item.add_marker(pytest.mark.risk)
        elif "/trading/" in test_path:
            item.add_marker(pytest.mark.trading)
        elif "/monitoring/" in test_path:
            item.add_marker(pytest.mark.monitoring)


# ============================================================================
# EVENT LOOP FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_context():
    """Provide async context for tests."""
    async with asyncio.TaskGroup() as tg:
        yield tg


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        "environment": "test",
        "logging": {"level": "DEBUG", "format": "json"},
        "database": {"url": "sqlite:///:memory:", "echo": False},
        "redis": {"url": "redis://localhost:6379/1"},
        "events": {"batch_size": 10, "batch_interval_seconds": 0.1, "max_queue_size": 1000},
        "testing": {"mock_external_apis": True, "use_in_memory_db": True, "fast_mode": True},
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock()
    config.get.return_value = {}
    config.__getitem__ = lambda self, key: {}
    config.__contains__ = lambda self, key: True
    return config


# ============================================================================
# DATABASE FIXTURES
# ============================================================================


@pytest.fixture
async def mock_db_pool():
    """Mock database connection pool."""
    pool = AsyncMock()

    # Mock connection context manager
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)

    pool.acquire = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return pool


@pytest.fixture
def temp_database_url():
    """Temporary database URL for testing."""
    return "sqlite:///:memory:"


# ============================================================================
# FILE SYSTEM FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create temporary configuration file."""
    config_content = """
# Test configuration
environment: test
database:
  url: sqlite:///:memory:
  echo: false
events:
  batch_size: 5
  batch_interval_seconds: 0.1
"""
    config_file = temp_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


# ============================================================================
# MOCK SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_broker():
    """Mock broker service for trading tests."""
    broker = AsyncMock()
    broker.submit_order = AsyncMock(return_value="ORDER_123")
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_order_status = AsyncMock(return_value="filled")
    broker.get_positions = AsyncMock(return_value=[])
    broker.get_account = AsyncMock(return_value={"cash": 10000})
    return broker


@pytest.fixture
def mock_market_data():
    """Mock market data service."""
    service = AsyncMock()
    service.get_latest_price = AsyncMock(return_value=100.0)
    service.get_historical_data = AsyncMock(return_value=[])
    service.subscribe_to_updates = AsyncMock()
    service.unsubscribe_from_updates = AsyncMock()
    return service


@pytest.fixture
def mock_feature_service():
    """Mock feature computation service."""
    service = AsyncMock()

    async def mock_compute_features(symbols, features, **kwargs):
        return {
            symbol: {
                feature: {"value": 1.0, "timestamp": "2023-01-01T00:00:00Z"} for feature in features
            }
            for symbol in symbols
        }

    service.compute_features = mock_compute_features
    return service


# ============================================================================
# TIME AND DATA FIXTURES
# ============================================================================


@pytest.fixture
def freezer():
    """Freeze time for deterministic testing."""
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = "2023-01-01T12:00:00Z"
        mock_datetime.utcnow.return_value = "2023-01-01T12:00:00Z"
        yield mock_datetime


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "AAPL": {
            "price": 150.0,
            "volume": 1000000,
            "bid": 149.98,
            "ask": 150.02,
            "timestamp": "2023-01-01T12:00:00Z",
        },
        "MSFT": {
            "price": 300.0,
            "volume": 500000,
            "bid": 299.95,
            "ask": 300.05,
            "timestamp": "2023-01-01T12:00:00Z",
        },
    }


@pytest.fixture
def sample_orders():
    """Sample orders for testing."""
    return [
        {
            "id": "ORDER_1",
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "type": "market",
            "status": "pending",
        },
        {
            "id": "ORDER_2",
            "symbol": "MSFT",
            "quantity": 50,
            "side": "sell",
            "type": "limit",
            "price": 300.0,
            "status": "filled",
        },
    ]


# ============================================================================
# ASYNC UTILITIES
# ============================================================================


@pytest.fixture
async def async_mock_manager():
    """Manager for async mocks that need cleanup."""
    mocks = []

    def create_async_mock(*args, **kwargs):
        mock = AsyncMock(*args, **kwargs)
        mocks.append(mock)
        return mock

    yield create_async_mock

    # Cleanup
    for mock in mocks:
        if hasattr(mock, "close"):
            try:
                await mock.close()
            except:
                pass


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================


@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    # Standard library imports
    import os
    import time

    # Third-party imports
    import psutil

    class PerformanceTracker:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
            self.metrics = {}

        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        def stop(self):
            if self.start_time is None:
                return

            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            self.metrics = {
                "duration": end_time - self.start_time,
                "memory_start_mb": self.start_memory,
                "memory_end_mb": end_memory,
                "memory_delta_mb": end_memory - self.start_memory,
                "cpu_percent": self.process.cpu_percent(),
            }

            return self.metrics

    return PerformanceTracker()


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically cleanup environment after each test."""
    yield

    # Reset environment variables that tests might have modified
    test_vars = [key for key in os.environ.keys() if key.startswith("TEST_")]
    for var in test_vars:
        os.environ.pop(var, None)

    # Close any remaining async resources
    try:
        loop = asyncio.get_running_loop()
        # Cancel any remaining tasks
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in tasks:
            task.cancel()
    except RuntimeError:
        pass  # No running loop


# ============================================================================
# PYTEST HOOKS
# ============================================================================


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Add test name to environment for debugging
    os.environ["CURRENT_TEST"] = item.name


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test run."""
    # Clean up test-specific environment
    os.environ.pop("CURRENT_TEST", None)


def pytest_sessionstart(session):
    """Start of test session."""
    print("\n🚀 Starting AI Trader test session")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}...")


def pytest_sessionfinish(session, exitstatus):
    """End of test session."""
    if exitstatus == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests completed with exit status: {exitstatus}")


# ============================================================================
# CUSTOM MARKERS AND SKIP CONDITIONS
# ============================================================================


def pytest_runtest_setup(item):
    """Custom test setup with conditional skipping."""
    # Skip tests requiring external services in CI
    if item.get_closest_marker("requires_api") and os.getenv("CI"):
        pytest.skip("Skipping API tests in CI environment")

    if item.get_closest_marker("requires_network") and os.getenv("SKIP_NETWORK_TESTS"):
        pytest.skip("Skipping network tests")

    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow") and not item.config.getoption("--runslow", default=False):
        pytest.skip("Skipping slow test (use --runslow to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--integration-only", action="store_true", default=False, help="Run only integration tests"
    )
    parser.addoption("--unit-only", action="store_true", default=False, help="Run only unit tests")

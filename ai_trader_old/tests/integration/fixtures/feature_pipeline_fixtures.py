# tests/integration/fixtures/feature_pipeline_fixtures.py

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Mock External Dependencies


class MockRedisClient:
    """Mock Redis client for caching simulation."""

    def __init__(self):
        self._data = {}
        self._expiry = {}

    def get(self, key: str) -> str | None:
        """Get value from mock Redis."""
        if key in self._data:
            # Check expiry
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._data[key]
                del self._expiry[key]
                return None
            return self._data[key]
        return None

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Set value in mock Redis."""
        self._data[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True

    def delete(self, key: str) -> int:
        """Delete key from mock Redis."""
        deleted = 0
        if key in self._data:
            del self._data[key]
            deleted += 1
        if key in self._expiry:
            del self._expiry[key]
        return deleted

    def exists(self, key: str) -> int:
        """Check if key exists in mock Redis."""
        return 1 if key in self._data else 0

    def flushall(self) -> bool:
        """Clear all data from mock Redis."""
        self._data.clear()
        self._expiry.clear()
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern."""
        if pattern == "*":
            return list(self._data.keys())
        # Simple pattern matching for tests
        # Standard library imports
        import fnmatch

        return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]


class MockPostgreSQLAdapter:
    """Mock PostgreSQL database adapter."""

    def __init__(self):
        self._tables = {}
        self._connected = False

    async def create_connection(self):
        """Mock connection creation."""
        self._connected = True
        return True

    async def close_connection(self):
        """Mock connection closure."""
        self._connected = False
        return True

    async def execute_query(self, sql: str, params: dict[str, Any] = None) -> Any:
        """Mock query execution."""
        if not self._connected:
            raise Exception("Database not connected")

        # Simulate successful execution
        result = MagicMock()
        result.success = True
        result.errors = []
        result.rowcount = 1

        # Store data for retrieval if it's an INSERT/UPDATE
        if params and "INSERT" in sql.upper():
            table_name = "features"  # Simplified
            if table_name not in self._tables:
                self._tables[table_name] = []
            self._tables[table_name].append(params.copy())

        return result

    async def fetch_query(self, sql: str, params: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Mock query fetching."""
        if not self._connected:
            raise Exception("Database not connected")

        # Return mock data based on query
        if "SELECT" in sql.upper():
            table_name = "features"
            if table_name in self._tables:
                # Return stored data with some filtering simulation
                return self._tables[table_name][-10:]  # Last 10 records

        return []

    async def execute_batch(self, sql: str, params_list: list[dict[str, Any]]) -> bool:
        """Mock batch execution."""
        for params in params_list:
            await self.execute_query(sql, params)
        return True

    def get_table_data(self, table_name: str) -> list[dict[str, Any]]:
        """Get stored table data for testing."""
        return self._tables.get(table_name, [])

    def clear_table_data(self, table_name: str):
        """Clear table data for testing."""
        if table_name in self._tables:
            del self._tables[table_name]


class MockDataArchive:
    """Mock Data Lake archive storage."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path(tempfile.mkdtemp())
        self._stored_files = {}
        self._metadata = {}

    async def save_dataframe(
        self, key: str, dataframe: pd.DataFrame, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Mock saving DataFrame to archive."""
        # Store in memory for testing
        self._stored_files[key] = dataframe.copy()
        if metadata:
            self._metadata[key] = metadata.copy()

        # Also save to temp file for realistic behavior
        file_path = self.base_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(file_path)

        return True

    async def load_dataframe(self, key: str) -> pd.DataFrame:
        """Mock loading DataFrame from archive."""
        if key in self._stored_files:
            return self._stored_files[key].copy()

        # Try to load from file
        file_path = self.base_path / key
        if file_path.exists():
            return pd.read_parquet(file_path)

        raise FileNotFoundError(f"Key not found: {key}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in archive."""
        return key in self._stored_files or (self.base_path / key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with optional prefix filter."""
        keys = list(self._stored_files.keys())
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    async def delete_key(self, key: str) -> bool:
        """Delete key from archive."""
        deleted = False
        if key in self._stored_files:
            del self._stored_files[key]
            deleted = True
        if key in self._metadata:
            del self._metadata[key]

        file_path = self.base_path / key
        if file_path.exists():
            file_path.unlink()
            deleted = True

        return deleted

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for key."""
        return self._metadata.get(key)

    def cleanup(self):
        """Clean up temporary files."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)


class MockAPIClient:
    """Mock external API client (Alpaca, Polygon, etc.)."""

    def __init__(self, client_name: str):
        self.client_name = client_name
        self._call_count = 0
        self._rate_limit_calls = 0
        self._max_calls_per_minute = 60
        self._should_fail = False
        self._response_delay = 0.0

    async def get_market_data(
        self, symbols: list[str], start_time: datetime, end_time: datetime, **kwargs
    ) -> pd.DataFrame:
        """Mock market data retrieval."""
        await self._simulate_api_call()

        if self._should_fail:
            raise Exception(f"{self.client_name} API temporarily unavailable")

        # Generate mock data
        date_range = pd.date_range(start_time, end_time, freq="D", tz=UTC)
        data_frames = []

        for symbol in symbols:
            base_price = hash(symbol) % 200 + 50  # Deterministic price based on symbol

            prices = [base_price]
            for _ in range(len(date_range) - 1):
                change = secure_numpy_normal(0, 0.02)
                prices.append(prices[-1] * (1 + change))

            symbol_data = pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": date_range,
                    "close": prices,
                    "volume": np.secure_randint(10000, 100000, len(date_range)),
                }
            )
            data_frames.append(symbol_data)

        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

    async def _simulate_api_call(self):
        """Simulate API call with rate limiting and delays."""
        self._call_count += 1
        self._rate_limit_calls += 1

        # Simulate rate limiting
        if self._rate_limit_calls > self._max_calls_per_minute:
            raise Exception(f"{self.client_name} API rate limit exceeded")

        # Simulate network delay
        if self._response_delay > 0:
            await asyncio.sleep(self._response_delay)

    def set_failure_mode(self, should_fail: bool):
        """Control whether API calls should fail."""
        self._should_fail = should_fail

    def set_response_delay(self, delay_seconds: float):
        """Set artificial delay for API responses."""
        self._response_delay = delay_seconds

    def reset_rate_limit(self):
        """Reset rate limit counter."""
        self._rate_limit_calls = 0

    def get_call_count(self) -> int:
        """Get total number of API calls made."""
        return self._call_count


class MockEventBus:
    """Mock event bus for testing."""

    def __init__(self):
        self._subscribers = {}
        self._published_events = []
        self._processing_delay = 0.0

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: callable):
        """Unsubscribe from event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event):
        """Publish event to subscribers."""
        self._published_events.append(event)

        # Simulate processing delay
        if self._processing_delay > 0:
            await asyncio.sleep(self._processing_delay)

        # Call subscribers
        event_type = getattr(event, "event_type", type(event).__name__)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"Event handler error: {e}")

    def get_published_events(self) -> list[Any]:
        """Get all published events."""
        return self._published_events.copy()

    def set_processing_delay(self, delay_seconds: float):
        """Set artificial delay for event processing."""
        self._processing_delay = delay_seconds

    def clear_events(self):
        """Clear published events history."""
        self._published_events.clear()


# Test Data Generators


class MarketDataGenerator:
    """Generate realistic market data for testing."""

    @staticmethod
    def generate_intraday_data(
        symbols: list[str], date: datetime, freq: str = "1H", include_issues: bool = False
    ) -> pd.DataFrame:
        """Generate intraday market data."""
        if freq == "1H":
            periods = 24
        elif freq == "15T":
            periods = 96
        else:
            periods = 24

        date_range = pd.date_range(
            date.replace(hour=0, minute=0, second=0), periods=periods, freq=freq, tz=UTC
        )

        data_frames = []
        np.random.seed(42)

        for symbol in symbols:
            base_price = hash(symbol) % 200 + 50

            # Generate price movements with some realism
            returns = secure_numpy_normal(0, 0.001, len(date_range))  # Hourly returns
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Generate OHLC
            opens = prices[:-1] + [prices[-1]]
            closes = prices
            highs = [
                max(o, c) * (1 + abs(secure_numpy_normal(0, 0.002))) for o, c in zip(opens, closes)
            ]
            lows = [
                min(o, c) * (1 - abs(secure_numpy_normal(0, 0.002))) for o, c in zip(opens, closes)
            ]

            # Generate volume with pattern
            base_volume = 50000
            volume_pattern = (
                np.sin(np.arange(len(date_range)) * 2 * np.pi / 24) * 20000 + base_volume
            )
            volumes = volume_pattern + secure_numpy_normal(0, 5000, len(date_range))
            volumes = np.maximum(volumes, 1000)  # Minimum volume

            symbol_data = pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": date_range,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes.astype(int),
                }
            )

            # Introduce data quality issues if requested
            if include_issues:
                # Random missing values
                if len(symbol_data) > 10:
                    missing_idx = np.secure_choice(len(symbol_data), size=2, replace=False)
                    symbol_data.loc[missing_idx, "close"] = np.nan

                # Invalid OHLC relationship
                if len(symbol_data) > 5:
                    bad_idx = len(symbol_data) // 2
                    symbol_data.loc[bad_idx, "high"] = symbol_data.loc[bad_idx, "low"] - 1

            data_frames.append(symbol_data)

        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

    @staticmethod
    def generate_feature_data(
        symbols: list[str], start_date: datetime, end_date: datetime, feature_sets: list[str] = None
    ) -> dict[str, pd.DataFrame]:
        """Generate feature data for symbols."""
        if feature_sets is None:
            feature_sets = ["technical", "volume", "sentiment"]

        date_range = pd.date_range(start_date, end_date, freq="D", tz=UTC)
        features_dict = {}

        np.random.seed(42)

        for symbol in symbols:
            features = pd.DataFrame({"timestamp": date_range, "symbol": symbol})

            # Technical features
            if "technical" in feature_sets:
                features["sma_10"] = np.secure_uniform(95, 105, len(date_range))
                features["sma_20"] = np.secure_uniform(95, 105, len(date_range))
                features["rsi"] = np.secure_uniform(20, 80, len(date_range))
                features["macd"] = secure_numpy_normal(0, 1, len(date_range))
                features["bollinger_upper"] = np.secure_uniform(100, 110, len(date_range))
                features["bollinger_lower"] = np.secure_uniform(90, 100, len(date_range))

            # Volume features
            if "volume" in feature_sets:
                features["volume_sma"] = np.secure_uniform(40000, 60000, len(date_range))
                features["volume_ratio"] = np.secure_uniform(0.5, 2.0, len(date_range))
                features["volume_weighted_price"] = np.secure_uniform(95, 105, len(date_range))

            # Sentiment features
            if "sentiment" in feature_sets:
                features["news_sentiment"] = np.secure_uniform(-1, 1, len(date_range))
                features["social_sentiment"] = np.secure_uniform(-1, 1, len(date_range))
                features["sentiment_momentum"] = secure_numpy_normal(0, 0.1, len(date_range))

            # Volatility features
            if "volatility" in feature_sets:
                features["realized_volatility"] = np.secure_uniform(0.1, 0.5, len(date_range))
                features["garch_volatility"] = np.secure_uniform(0.1, 0.4, len(date_range))
                features["volatility_skew"] = secure_numpy_normal(0, 0.5, len(date_range))

            features_dict[symbol] = features

        return features_dict


# Pytest Fixtures


@pytest.fixture
def mock_redis_client():
    """Provide mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def mock_postgresql_adapter():
    """Provide mock PostgreSQL adapter."""
    return MockPostgreSQLAdapter()


@pytest.fixture
def mock_data_archive():
    """Provide mock data archive."""
    archive = MockDataArchive()
    yield archive
    archive.cleanup()


@pytest.fixture
def mock_api_clients():
    """Provide mock API clients."""
    return {
        "alpaca": MockAPIClient("Alpaca"),
        "polygon": MockAPIClient("Polygon"),
        "alpha_vantage": MockAPIClient("AlphaVantage"),
    }


@pytest.fixture
def mock_event_bus():
    """Provide mock event bus."""
    return MockEventBus()


@pytest.fixture
def market_data_generator():
    """Provide market data generator."""
    return MarketDataGenerator()


@pytest.fixture
def integration_test_symbols():
    """Standard symbols for integration testing."""
    return ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "BRK.B"]


@pytest.fixture
def integration_test_timeframe():
    """Standard timeframe for integration testing."""
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=30)
    return start_time, end_time


@pytest.fixture
def feature_pipeline_config():
    """Standard configuration for feature pipeline integration tests."""
    return {
        "orchestrator": {
            "features": {
                "cache": {"ttl_seconds": 300, "max_size_mb": 100},
                "parallel_processing": {"max_workers": 4, "max_concurrent_tasks": 10},
                "batch_processing": {
                    "default_batch_size": 50,
                    "interval_seconds": 5,
                    "processing_delay_seconds": 1,
                },
            },
            "lookback_periods": {"feature_calculation_days": 30},
        },
        "feature_store": {"hot_storage_days": 7, "batch_size": 1000, "parallel_writes": True},
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_features",
            "user": "test_user",
            "password": "test_password",
        },
        "redis": {"host": "localhost", "port": 6379, "db": 1},
    }


@pytest.fixture
def temp_test_environment():
    """Create temporary test environment."""
    base_dir = Path(tempfile.mkdtemp(prefix="feature_integration_test_"))

    env = {
        "base_dir": base_dir,
        "features_dir": base_dir / "features",
        "archive_dir": base_dir / "archive",
        "cache_dir": base_dir / "cache",
        "logs_dir": base_dir / "logs",
    }

    # Create directories
    for dir_path in env.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)

    yield env

    # Cleanup
    if base_dir.exists():
        shutil.rmtree(base_dir)

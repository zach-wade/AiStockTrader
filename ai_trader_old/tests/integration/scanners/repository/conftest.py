"""
Configuration and fixtures for scanner repository integration tests.
"""

# Standard library imports
import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

# Third-party imports
from omegaconf import DictConfig, OmegaConf
import pytest

# Local imports
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.data_pipeline.storage.repositories.scanner_data_repository import ScannerDataRepository
from main.data_pipeline.storage.storage_router import StorageRouter
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerRepository
from main.scanners.scanner_factory import ScannerFactory


@pytest.fixture
def test_scanner_config():
    """Configuration for scanner testing."""
    return OmegaConf.create(
        {
            "scanners": {
                "volume": {
                    "enabled": True,
                    "priority": 8,
                    "timeout_seconds": 30.0,
                    "lookback_days": 30,
                    "min_volume_ratio": 2.0,
                    "use_cache": False,  # Disable for testing
                },
                "technical": {
                    "enabled": True,
                    "priority": 7,
                    "timeout_seconds": 30.0,
                    "lookback_days": 60,
                    "use_cache": False,
                },
                "news": {
                    "enabled": True,
                    "priority": 9,
                    "timeout_seconds": 45.0,
                    "lookback_hours": 24,
                    "use_cache": False,
                },
                "social": {
                    "enabled": True,
                    "priority": 6,
                    "timeout_seconds": 30.0,
                    "lookback_hours": 12,
                    "use_cache": False,
                },
                "earnings": {
                    "enabled": True,
                    "priority": 10,
                    "timeout_seconds": 30.0,
                    "lookback_days": 90,
                    "use_cache": False,
                },
            },
            "repository": {
                "hot_storage_threshold_days": 30,
                "cache_enabled": False,
                "query_timeout_seconds": 30.0,
            },
        }
    )


@pytest.fixture
async def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = Mock(spec=IEventBus)
    event_bus.publish = AsyncMock()
    event_bus.subscribe = Mock()
    event_bus.unsubscribe = Mock()
    return event_bus


@pytest.fixture
async def mock_storage_router():
    """Mock storage router for testing."""
    router = Mock(spec=StorageRouter)

    # Mock routing decisions
    async def mock_route_query(query_filter, query_type):
        # Route based on date range for testing
        days_back = (datetime.now(UTC) - query_filter.start_date).days
        if days_back <= 30:
            return "hot"
        else:
            return "cold"

    router.route_query = mock_route_query
    return router


@pytest.fixture
async def scanner_repository(
    real_database: IAsyncDatabase, mock_storage_router, mock_event_bus
) -> AsyncGenerator[IScannerRepository, None]:
    """Create scanner data repository for testing."""
    repository = ScannerDataRepository(
        db_adapter=real_database, storage_router=mock_storage_router, event_bus=mock_event_bus
    )

    # Initialize if needed
    if hasattr(repository, "initialize"):
        await repository.initialize()

    yield repository

    # Cleanup
    if hasattr(repository, "cleanup"):
        await repository.cleanup()


@pytest.fixture
async def scanner_factory(
    test_scanner_config: DictConfig, real_database: IAsyncDatabase, mock_event_bus: IEventBus
) -> ScannerFactory:
    """Create scanner factory for testing."""
    factory = ScannerFactory(
        config=test_scanner_config, db_adapter=real_database, event_bus=mock_event_bus
    )
    return factory


@pytest.fixture
def test_symbols():
    """Standard set of test symbols."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


@pytest.fixture
def recent_date_range():
    """Recent date range for hot storage testing."""
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=7)
    return QueryFilter(start_date=start_date, end_date=end_date)


@pytest.fixture
def historical_date_range():
    """Historical date range for cold storage testing."""
    end_date = datetime.now(UTC) - timedelta(days=60)
    start_date = end_date - timedelta(days=30)
    return QueryFilter(start_date=start_date, end_date=end_date)


@pytest.fixture
def mixed_date_range():
    """Date range spanning hot and cold storage."""
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=45)  # Spans 30-day threshold
    return QueryFilter(start_date=start_date, end_date=end_date)


@pytest.fixture
async def sample_market_data():
    """Sample market data for testing."""
    base_date = datetime.now(UTC) - timedelta(days=1)

    data = []
    symbols = ["AAPL", "GOOGL", "MSFT"]

    for i, symbol in enumerate(symbols):
        for day_offset in range(10):  # 10 days of data
            date = base_date - timedelta(days=day_offset)

            # Generate realistic-looking data
            base_price = 150 + i * 50  # Different base prices per symbol
            price_variation = day_offset * 2  # Some price movement

            record = {
                "symbol": symbol,
                "date": date,
                "open": base_price + price_variation,
                "high": base_price + price_variation + 5,
                "low": base_price + price_variation - 3,
                "close": base_price + price_variation + 1,
                "volume": 1000000 + day_offset * 100000,
                "returns": 0.01 if day_offset % 2 == 0 else -0.005,
            }
            data.append(record)

    return data


@pytest.fixture
async def sample_news_data():
    """Sample news data for testing."""
    base_date = datetime.now(UTC) - timedelta(hours=2)

    return [
        {
            "symbol": "AAPL",
            "headline": "Apple announces new product launch",
            "content": "Apple Inc. announced today a revolutionary new product that will change the market.",
            "sentiment_score": 0.8,
            "timestamp": base_date,
            "source": "Reuters",
            "relevance_score": 0.9,
        },
        {
            "symbol": "GOOGL",
            "headline": "Google reports strong earnings",
            "content": "Alphabet Inc. exceeded earnings expectations in the latest quarter.",
            "sentiment_score": 0.75,
            "timestamp": base_date - timedelta(hours=1),
            "source": "Bloomberg",
            "relevance_score": 0.85,
        },
        {
            "symbol": "MSFT",
            "headline": "Microsoft cloud growth continues",
            "content": "Microsoft Azure continues to show strong growth in cloud services.",
            "sentiment_score": 0.7,
            "timestamp": base_date - timedelta(hours=3),
            "source": "CNBC",
            "relevance_score": 0.8,
        },
    ]


@pytest.fixture
async def sample_social_data():
    """Sample social media data for testing."""
    base_date = datetime.now(UTC) - timedelta(minutes=30)

    return {
        "AAPL": [
            {
                "author": "trader123",
                "content": "AAPL looking strong today! Great momentum.",
                "timestamp": base_date,
                "platform": "twitter",
                "sentiment_score": 0.8,
                "engagement_score": 150,
            },
            {
                "author": "investor456",
                "content": "Apple earnings next week should be good",
                "timestamp": base_date - timedelta(minutes=15),
                "platform": "reddit",
                "sentiment_score": 0.7,
                "engagement_score": 85,
            },
        ],
        "GOOGL": [
            {
                "author": "tech_analyst",
                "content": "Google search revenue concerns overblown",
                "timestamp": base_date - timedelta(minutes=10),
                "platform": "stocktwits",
                "sentiment_score": 0.6,
                "engagement_score": 120,
            }
        ],
    }


@pytest.fixture
async def sample_earnings_data():
    """Sample earnings data for testing."""
    base_date = datetime.now(UTC) - timedelta(days=1)

    return [
        {
            "symbol": "AAPL",
            "report_date": base_date,
            "period_ending": base_date - timedelta(days=30),
            "eps_actual": 2.50,
            "eps_estimate": 2.45,
            "revenue_actual": 95000000000,
            "revenue_estimate": 94000000000,
            "surprise_percent": 2.04,
        },
        {
            "symbol": "GOOGL",
            "report_date": base_date - timedelta(days=2),
            "period_ending": base_date - timedelta(days=32),
            "eps_actual": 1.85,
            "eps_estimate": 1.80,
            "revenue_actual": 75000000000,
            "revenue_estimate": 74500000000,
            "surprise_percent": 2.78,
        },
    ]


@pytest.fixture
async def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "repository": {
            "query_time_ms": 1000,  # Max 1 second per query
            "large_query_time_ms": 5000,  # Max 5 seconds for large queries
            "concurrent_queries": 10,  # Support 10 concurrent queries
        },
        "scanner": {
            "scan_time_ms": 30000,  # Max 30 seconds per scan
            "alert_generation_ms": 5000,  # Max 5 seconds for alert generation
        },
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async testing."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Clean up test data after each test
@pytest.fixture(autouse=True)
async def cleanup_test_data(real_database: IAsyncDatabase):
    """Clean up test data after each test."""
    yield  # Run the test

    # Cleanup after test
    try:
        # Clean up any test data that might have been inserted
        cleanup_queries = [
            "DELETE FROM market_data WHERE symbol LIKE 'TEST%'",
            "DELETE FROM news_data WHERE symbol LIKE 'TEST%'",
            "DELETE FROM social_data WHERE symbol LIKE 'TEST%'",
            "DELETE FROM earnings_data WHERE symbol LIKE 'TEST%'",
        ]

        for query in cleanup_queries:
            try:
                await real_database.execute(query)
            except Exception:
                pass  # Ignore errors during cleanup

    except Exception:
        # Don't fail tests due to cleanup issues
        pass

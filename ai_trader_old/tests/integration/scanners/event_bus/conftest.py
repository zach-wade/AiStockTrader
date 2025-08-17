"""
Configuration and fixtures for scanner event bus integration tests.
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
from main.events.core.event_bus_factory import EventBusFactory
from main.events.types.event_types import AlertType, EventType, ScannerAlertEvent
from main.interfaces.events import Event, IEventBus
from main.interfaces.scanners import IScanner, IScannerRepository
from main.scanners.types import ScanAlert
from main.utils.datetime_utils import ensure_utc


@pytest.fixture
def test_event_bus_config():
    """Configuration for event bus testing."""
    return OmegaConf.create(
        {
            "event_bus": {
                "max_queue_size": 1000,
                "enable_persistence": False,
                "enable_dead_letter_queue": True,
                "batch_size": 10,
                "flush_interval_seconds": 1.0,
                "retry_attempts": 2,
                "enable_metrics": True,
            }
        }
    )


@pytest.fixture
async def test_event_bus(test_event_bus_config) -> AsyncGenerator[IEventBus, None]:
    """Create test event bus instance."""
    event_bus = EventBusFactory.create_test_instance(test_event_bus_config)

    # Initialize if needed
    if hasattr(event_bus, "initialize"):
        await event_bus.initialize()

    yield event_bus

    # Cleanup
    if hasattr(event_bus, "cleanup"):
        await event_bus.cleanup()


@pytest.fixture
async def mock_scanner_repository():
    """Mock scanner repository for testing."""
    repository = Mock(spec=IScannerRepository)

    # Mock basic data methods
    repository.get_market_data = AsyncMock(return_value={})
    repository.get_news_data = AsyncMock(return_value=[])
    repository.get_social_sentiment = AsyncMock(return_value={})
    repository.get_earnings_data = AsyncMock(return_value=[])
    repository.get_insider_data = AsyncMock(return_value=[])
    repository.get_options_data = AsyncMock(return_value=[])
    repository.get_sector_data = AsyncMock(return_value={})

    return repository


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
                    "publish_events": True,
                },
                "technical": {
                    "enabled": True,
                    "priority": 7,
                    "timeout_seconds": 30.0,
                    "lookback_days": 60,
                    "publish_events": True,
                },
                "news": {
                    "enabled": True,
                    "priority": 9,
                    "timeout_seconds": 45.0,
                    "lookback_hours": 24,
                    "publish_events": True,
                },
            }
        }
    )


@pytest.fixture
async def volume_scanner(
    test_scanner_config: DictConfig,
    mock_scanner_repository: IScannerRepository,
    test_event_bus: IEventBus,
) -> IScanner:
    """Create volume scanner for testing."""
    # Local imports
    from main.scanners.volume_scanner import VolumeScanner

    scanner = VolumeScanner(
        config=test_scanner_config.scanners.volume,
        repository=mock_scanner_repository,
        event_bus=test_event_bus,
    )
    return scanner


@pytest.fixture
async def technical_scanner(
    test_scanner_config: DictConfig,
    mock_scanner_repository: IScannerRepository,
    test_event_bus: IEventBus,
) -> IScanner:
    """Create technical scanner for testing."""
    # Local imports
    from main.scanners.technical_scanner import TechnicalScanner

    scanner = TechnicalScanner(
        config=test_scanner_config.scanners.technical,
        repository=mock_scanner_repository,
        event_bus=test_event_bus,
    )
    return scanner


@pytest.fixture
async def news_scanner(
    test_scanner_config: DictConfig,
    mock_scanner_repository: IScannerRepository,
    test_event_bus: IEventBus,
) -> IScanner:
    """Create news scanner for testing."""
    # Local imports
    from main.scanners.news_scanner import NewsScanner

    scanner = NewsScanner(
        config=test_scanner_config.scanners.news,
        repository=mock_scanner_repository,
        event_bus=test_event_bus,
    )
    return scanner


@pytest.fixture
def test_symbols():
    """Standard set of test symbols."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


@pytest.fixture
def sample_scan_alerts():
    """Sample scan alerts for testing."""
    base_time = datetime.now(UTC)

    return [
        ScanAlert(
            symbol="AAPL",
            alert_type=AlertType.VOLUME_SPIKE,
            score=0.85,
            metadata={
                "current_volume": 150000000,
                "avg_volume": 75000000,
                "relative_volume": 2.0,
                "confidence": 0.9,
            },
            timestamp=base_time - timedelta(minutes=5),
            scanner_name="volume_scanner",
        ),
        ScanAlert(
            symbol="GOOGL",
            alert_type=AlertType.TECHNICAL_BREAKOUT,
            score=0.78,
            metadata={
                "breakout_level": 135.50,
                "current_price": 138.25,
                "volume_confirmation": True,
                "confidence": 0.8,
            },
            timestamp=base_time - timedelta(minutes=10),
            scanner_name="technical_scanner",
        ),
        ScanAlert(
            symbol="MSFT",
            alert_type=AlertType.NEWS_SENTIMENT,
            score=0.92,
            metadata={
                "sentiment_score": 0.85,
                "news_volume": 15,
                "relevance_score": 0.9,
                "confidence": 0.92,
            },
            timestamp=base_time - timedelta(minutes=15),
            scanner_name="news_scanner",
        ),
    ]


@pytest.fixture
async def event_collector():
    """Event collector for testing event publication."""
    collected_events = []

    class EventCollector:
        def __init__(self):
            self.events = []
            self.event_counts = {}
            self.subscribers = {}

        async def publish(self, event: Event):
            """Collect published events."""
            self.events.append(event)
            event_type = getattr(event, "event_type", "unknown")
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        async def subscribe(self, event_type: str, handler):
            """Mock subscription."""
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

        async def unsubscribe(self, event_type: str, handler):
            """Mock unsubscription."""
            if event_type in self.subscribers:
                if handler in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(handler)

        def get_events_by_type(self, event_type: str) -> list[Event]:
            """Get events by type."""
            return [
                event for event in self.events if getattr(event, "event_type", None) == event_type
            ]

        def get_scanner_alerts(self) -> list[ScannerAlertEvent]:
            """Get scanner alert events."""
            return [event for event in self.events if isinstance(event, ScannerAlertEvent)]

        def clear(self):
            """Clear collected events."""
            self.events.clear()
            self.event_counts.clear()

    return EventCollector()


@pytest.fixture
def event_performance_thresholds():
    """Performance thresholds for event testing."""
    return {
        "publish_latency_ms": 50,  # Max 50ms to publish event
        "batch_processing_ms": 200,  # Max 200ms for batch processing
        "event_throughput_per_second": 1000,  # Min 1000 events/sec
        "scanner_scan_time_ms": 5000,  # Max 5 seconds per scan
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async testing."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_volume_data():
    """Sample volume data for volume scanner tests."""
    # Third-party imports
    import pandas as pd

    base_time = datetime.now(UTC)

    return {
        "AAPL": pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "timestamp": base_time - timedelta(minutes=i * 5),
                    "volume": 150000000 if i == 0 else 75000000,  # Volume spike at latest
                    "close": 150.0 + i * 0.1,
                    "returns": 0.02 if i == 0 else 0.01,
                }
                for i in range(20)
            ]
        )
    }


@pytest.fixture
def sample_technical_data():
    """Sample technical data for technical scanner tests."""
    # Third-party imports
    import pandas as pd

    base_time = datetime.now(UTC)

    return {
        "GOOGL": pd.DataFrame(
            [
                {
                    "symbol": "GOOGL",
                    "timestamp": base_time - timedelta(days=i),
                    "open": 130.0 + i * 0.2,
                    "high": 132.0 + i * 0.2,
                    "low": 128.0 + i * 0.2,
                    "close": 131.0 + i * 0.2,
                    "volume": 25000000,
                }
                for i in range(50)
            ]
        )
    }


@pytest.fixture
def sample_news_data():
    """Sample news data for news scanner tests."""
    base_time = datetime.now(UTC)

    return [
        {
            "symbol": "MSFT",
            "headline": "Microsoft reports strong quarterly earnings",
            "content": "Microsoft exceeded expectations with strong cloud growth.",
            "sentiment_score": 0.85,
            "timestamp": base_time - timedelta(hours=1),
            "source": "Reuters",
            "relevance_score": 0.9,
        },
        {
            "symbol": "MSFT",
            "headline": "Microsoft announces new AI partnership",
            "content": "Strategic partnership to accelerate AI development.",
            "sentiment_score": 0.8,
            "timestamp": base_time - timedelta(hours=2),
            "source": "TechCrunch",
            "relevance_score": 0.85,
        },
    ]


# Mock scanner implementations for testing
class MockEventPublishingScanner:
    """Mock scanner that publishes events for testing."""

    def __init__(self, name: str, event_bus: IEventBus):
        self.name = name
        self.event_bus = event_bus
        self.alerts_generated = []

    async def scan(self, symbols: list[str]) -> list[ScanAlert]:
        """Generate mock alerts and publish events."""
        alerts = []

        for symbol in symbols:
            alert = ScanAlert(
                symbol=symbol,
                alert_type=AlertType.VOLUME_SPIKE,
                score=0.8,
                metadata={"test": True},
                timestamp=datetime.now(UTC),
                scanner_name=self.name,
            )
            alerts.append(alert)

        # Publish events
        if self.event_bus:
            await self.publish_alerts_to_event_bus(alerts)

        self.alerts_generated.extend(alerts)
        return alerts

    async def publish_alerts_to_event_bus(self, alerts: list[ScanAlert]):
        """Publish alerts to event bus."""
        for alert in alerts:
            event = ScannerAlertEvent(
                symbol=alert.symbol,
                alert_type=str(alert.alert_type),
                score=alert.score,
                scanner_name=self.name,
                metadata=alert.metadata,
                timestamp=ensure_utc(alert.timestamp),
                event_type=EventType.SCANNER_ALERT,
            )
            await self.event_bus.publish(event)


@pytest.fixture
async def mock_event_scanner(test_event_bus: IEventBus) -> MockEventPublishingScanner:
    """Create mock scanner for event testing."""
    return MockEventPublishingScanner("mock_scanner", test_event_bus)

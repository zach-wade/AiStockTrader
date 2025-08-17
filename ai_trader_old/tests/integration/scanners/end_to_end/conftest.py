"""
Configuration and fixtures for end-to-end scanner integration tests.
"""

# Standard library imports
import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

# Third-party imports
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytest

# Local imports
from main.data_pipeline.storage.repositories.scanner_data_repository import ScannerDataRepository
from main.data_pipeline.storage.storage_router import StorageRouter
from main.events.core.event_bus_factory import EventBusFactory
from main.events.types.event_types import ScannerAlertEvent
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerOrchestrator, IScannerRepository
from main.scanners.scanner_factory import ScannerFactory
from main.scanners.scanner_orchestrator_factory import ScannerOrchestratorFactory


@pytest.fixture
def end_to_end_config():
    """Complete configuration for end-to-end testing."""
    return OmegaConf.create(
        {
            "scanners": {
                "volume": {
                    "enabled": True,
                    "priority": 8,
                    "timeout_seconds": 30.0,
                    "lookback_days": 30,
                    "min_volume_ratio": 2.0,
                    "use_cache": False,
                    "publish_events": True,
                },
                "technical": {
                    "enabled": True,
                    "priority": 7,
                    "timeout_seconds": 30.0,
                    "lookback_days": 60,
                    "breakout_threshold": 0.05,
                    "use_cache": False,
                    "publish_events": True,
                },
                "news": {
                    "enabled": True,
                    "priority": 9,
                    "timeout_seconds": 45.0,
                    "lookback_hours": 24,
                    "min_sentiment_score": 0.7,
                    "min_relevance_score": 0.8,
                    "use_cache": False,
                    "publish_events": True,
                },
                "earnings": {
                    "enabled": True,
                    "priority": 10,
                    "timeout_seconds": 30.0,
                    "lookback_days": 90,
                    "min_surprise_threshold": 5.0,
                    "use_cache": False,
                    "publish_events": True,
                },
                "social": {
                    "enabled": True,
                    "priority": 6,
                    "timeout_seconds": 30.0,
                    "lookback_hours": 12,
                    "min_sentiment_score": 0.75,
                    "min_volume_threshold": 100,
                    "use_cache": False,
                    "publish_events": True,
                },
            },
            "orchestrator": {
                "execution_strategy": "parallel",
                "max_concurrent_scanners": 5,
                "timeout_seconds": 60.0,
                "enable_health_monitoring": True,
                "enable_deduplication": True,
                "publish_events": True,
                "alert_aggregation_window_seconds": 300,
            },
            "event_bus": {
                "max_queue_size": 1000,
                "enable_persistence": False,
                "enable_dead_letter_queue": True,
                "batch_size": 10,
                "flush_interval_seconds": 1.0,
                "retry_attempts": 2,
                "enable_metrics": True,
            },
            "repository": {
                "hot_storage_threshold_days": 30,
                "cache_enabled": False,
                "query_timeout_seconds": 30.0,
                "batch_size": 1000,
            },
        }
    )


@pytest.fixture
async def end_to_end_event_bus(end_to_end_config) -> AsyncGenerator[IEventBus, None]:
    """Create event bus for end-to-end testing."""
    event_bus = EventBusFactory.create_test_instance(end_to_end_config.event_bus)

    if hasattr(event_bus, "initialize"):
        await event_bus.initialize()

    yield event_bus

    if hasattr(event_bus, "cleanup"):
        await event_bus.cleanup()


@pytest.fixture
async def mock_storage_router():
    """Mock storage router for end-to-end testing."""
    router = Mock(spec=StorageRouter)

    async def mock_route_query(query_filter, query_type):
        # Route based on date range
        days_back = (datetime.now(UTC) - query_filter.start_date).days
        if days_back <= 30:
            return "hot"
        else:
            return "cold"

    router.route_query = mock_route_query
    return router


@pytest.fixture
async def end_to_end_repository(
    real_database: IAsyncDatabase, mock_storage_router, end_to_end_event_bus: IEventBus
) -> AsyncGenerator[IScannerRepository, None]:
    """Create repository for end-to-end testing."""
    repository = ScannerDataRepository(
        db_adapter=real_database, storage_router=mock_storage_router, event_bus=end_to_end_event_bus
    )

    if hasattr(repository, "initialize"):
        await repository.initialize()

    yield repository

    if hasattr(repository, "cleanup"):
        await repository.cleanup()


@pytest.fixture
async def end_to_end_scanner_factory(
    end_to_end_config: DictConfig,
    end_to_end_repository: IScannerRepository,
    end_to_end_event_bus: IEventBus,
) -> ScannerFactory:
    """Create scanner factory for end-to-end testing."""
    factory = ScannerFactory(
        config=end_to_end_config, repository=end_to_end_repository, event_bus=end_to_end_event_bus
    )
    return factory


@pytest.fixture
async def end_to_end_orchestrator(
    end_to_end_config: DictConfig,
    end_to_end_repository: IScannerRepository,
    end_to_end_event_bus: IEventBus,
) -> IScannerOrchestrator:
    """Create scanner orchestrator for end-to-end testing."""
    orchestrator = ScannerOrchestratorFactory.create_orchestrator(
        config=end_to_end_config, repository=end_to_end_repository, event_bus=end_to_end_event_bus
    )
    return orchestrator


@pytest.fixture
def comprehensive_test_symbols():
    """Comprehensive set of test symbols for end-to-end testing."""
    return [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",  # Large cap tech
        "NVDA",
        "META",
        "NFLX",
        "CRM",
        "ADBE",  # More tech
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",  # Finance
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "MRK",  # Healthcare
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",  # Energy
    ]


@pytest.fixture
def realistic_market_data():
    """Realistic market data for end-to-end testing."""
    data = {}
    base_time = datetime.now(UTC)

    # Create realistic data for each symbol
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    base_prices = [155.0, 135.0, 280.0, 145.0, 200.0]
    base_volumes = [75000000, 25000000, 30000000, 40000000, 80000000]

    for symbol, base_price, base_volume in zip(symbols, base_prices, base_volumes):
        # Generate 60 days of daily data
        daily_data = []
        for i in range(60):
            date = base_time - timedelta(days=i)

            # Add some realistic price movement
            price_change = (i % 7 - 3) * 0.02 * base_price  # ±6% moves
            current_price = base_price + price_change

            # Add volume variation
            volume_multiplier = 1.0 + (i % 5 - 2) * 0.3  # ±60% volume variation
            if i == 0:  # Today has volume spike
                volume_multiplier = 2.5
            current_volume = int(base_volume * volume_multiplier)

            daily_data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "timestamp": date,
                    "open": current_price - 0.5,
                    "high": current_price + 2.0,
                    "low": current_price - 2.0,
                    "close": current_price,
                    "volume": current_volume,
                    "returns": price_change / base_price if i > 0 else 0.0,
                    "interval": "1day",
                }
            )

        # Generate intraday data for today
        intraday_data = []
        for i in range(78):  # 6.5 hours of 5-minute data
            timestamp = base_time - timedelta(minutes=i * 5)

            # Intraday price movement
            intraday_price = base_price + (i % 10 - 5) * 0.005 * base_price
            intraday_volume = base_volume // 78  # Distribute daily volume

            if i < 10:  # Early trading has higher volume
                intraday_volume *= 2

            intraday_data.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": intraday_price - 0.1,
                    "high": intraday_price + 0.3,
                    "low": intraday_price - 0.3,
                    "close": intraday_price,
                    "volume": intraday_volume,
                    "interval": "5min",
                }
            )

        # Combine daily and intraday data
        all_data = daily_data + intraday_data
        data[symbol] = pd.DataFrame(all_data).sort_values("timestamp")

    return data


@pytest.fixture
def realistic_news_data():
    """Realistic news data for end-to-end testing."""
    base_time = datetime.now(UTC)

    news_data = [
        # AAPL news
        {
            "symbol": "AAPL",
            "headline": "Apple Reports Record Quarterly Revenue of $123.9 Billion",
            "content": "Apple Inc. today announced financial results for its fiscal 2024 first quarter ended December 30, 2023. The Company posted quarterly revenue of $123.9 billion, up 2 percent year over year, and quarterly earnings per diluted share of $2.18, up 16 percent year over year.",
            "sentiment_score": 0.85,
            "relevance_score": 0.95,
            "timestamp": base_time - timedelta(hours=2),
            "source": "Apple Press Release",
            "category": "earnings",
            "breaking_news": True,
        },
        {
            "symbol": "AAPL",
            "headline": "Apple Announces New AI Features Coming to iPhone",
            "content": "Apple today previewed powerful new AI capabilities coming to iPhone with iOS 18. The new features will enhance productivity and creativity.",
            "sentiment_score": 0.8,
            "relevance_score": 0.9,
            "timestamp": base_time - timedelta(hours=4),
            "source": "TechCrunch",
            "category": "product",
            "breaking_news": False,
        },
        # GOOGL news
        {
            "symbol": "GOOGL",
            "headline": "Google Cloud Revenue Surges 35% in Q4, Beats Expectations",
            "content": "Alphabet Inc. reported strong fourth-quarter results with Google Cloud revenue growing 35% year-over-year to $8.03 billion, significantly beating analyst expectations.",
            "sentiment_score": 0.9,
            "relevance_score": 0.92,
            "timestamp": base_time - timedelta(hours=1),
            "source": "Reuters",
            "category": "earnings",
            "breaking_news": True,
        },
        # MSFT news
        {
            "symbol": "MSFT",
            "headline": "Microsoft Azure AI Services See Record Adoption",
            "content": "Microsoft Corporation announced that its Azure AI services have seen unprecedented adoption, with revenue growing 83% year-over-year.",
            "sentiment_score": 0.88,
            "relevance_score": 0.9,
            "timestamp": base_time - timedelta(hours=3),
            "source": "Bloomberg",
            "category": "business",
            "breaking_news": False,
        },
        # Market-wide news
        {
            "symbol": "TSLA",
            "headline": "Tesla Deliveries Beat Q4 Estimates Despite Production Challenges",
            "content": "Tesla Inc. delivered 484,507 vehicles in Q4 2023, beating analyst estimates of 473,000 despite ongoing production challenges.",
            "sentiment_score": 0.75,
            "relevance_score": 0.85,
            "timestamp": base_time - timedelta(hours=6),
            "source": "CNBC",
            "category": "deliveries",
            "breaking_news": False,
        },
    ]

    return news_data


@pytest.fixture
def realistic_social_data():
    """Realistic social media data for end-to-end testing."""
    base_time = datetime.now(UTC)

    social_data = {
        "AAPL": [
            {
                "author": "tech_analyst_pro",
                "content": "AAPL earnings absolutely crushing it! Revenue growth + margin expansion = 🚀🚀🚀 $AAPL",
                "sentiment_score": 0.95,
                "timestamp": base_time - timedelta(minutes=30),
                "platform": "twitter",
                "engagement_score": 2500,
                "follower_count": 500000,
                "verified": True,
                "retweet_count": 850,
                "like_count": 3200,
            },
            {
                "author": "warren_buffett_fan",
                "content": "Apple continues to demonstrate exceptional capital allocation and margin discipline. Long-term hold.",
                "sentiment_score": 0.8,
                "timestamp": base_time - timedelta(hours=1),
                "platform": "reddit",
                "engagement_score": 450,
                "follower_count": 50000,
                "verified": False,
                "upvote_count": 125,
                "comment_count": 35,
            },
            {
                "author": "retail_investor_2024",
                "content": "Just added more $AAPL to my portfolio. These earnings are insane!",
                "sentiment_score": 0.85,
                "timestamp": base_time - timedelta(minutes=45),
                "platform": "stocktwits",
                "engagement_score": 180,
                "follower_count": 5000,
                "verified": False,
            },
        ],
        "GOOGL": [
            {
                "author": "cloud_computing_expert",
                "content": "Google Cloud growth trajectory is incredible. 35% YoY growth at this scale is remarkable. $GOOGL",
                "sentiment_score": 0.9,
                "timestamp": base_time - timedelta(minutes=20),
                "platform": "twitter",
                "engagement_score": 1800,
                "follower_count": 250000,
                "verified": True,
                "retweet_count": 520,
                "like_count": 2100,
            }
        ],
        "TSLA": [
            {
                "author": "ev_enthusiast",
                "content": "Tesla delivery numbers show resilience despite challenges. Q1 2024 will be the real test.",
                "sentiment_score": 0.65,
                "timestamp": base_time - timedelta(hours=2),
                "platform": "reddit",
                "engagement_score": 320,
                "follower_count": 25000,
                "verified": False,
                "upvote_count": 89,
                "comment_count": 42,
            }
        ],
    }

    return social_data


@pytest.fixture
def realistic_earnings_data():
    """Realistic earnings data for end-to-end testing."""
    base_time = datetime.now(UTC)

    earnings_data = [
        {
            "symbol": "AAPL",
            "report_date": base_time - timedelta(hours=2),
            "period_ending": base_time - timedelta(days=30),
            "eps_actual": 2.18,
            "eps_estimate": 2.10,
            "revenue_actual": 123900000000,
            "revenue_estimate": 121000000000,
            "surprise_percent": 3.81,
            "revenue_surprise_percent": 2.40,
            "fiscal_quarter": "Q1",
            "fiscal_year": 2024,
            "guidance_raised": True,
            "conference_call_sentiment": 0.85,
        },
        {
            "symbol": "GOOGL",
            "report_date": base_time - timedelta(hours=1),
            "period_ending": base_time - timedelta(days=31),
            "eps_actual": 1.64,
            "eps_estimate": 1.59,
            "revenue_actual": 86250000000,
            "revenue_estimate": 85300000000,
            "surprise_percent": 3.14,
            "revenue_surprise_percent": 1.11,
            "fiscal_quarter": "Q4",
            "fiscal_year": 2023,
            "guidance_raised": False,
            "conference_call_sentiment": 0.75,
        },
        {
            "symbol": "MSFT",
            "report_date": base_time + timedelta(days=2),  # Upcoming
            "period_ending": base_time - timedelta(days=1),
            "eps_estimate": 3.12,
            "revenue_estimate": 61500000000,
            "fiscal_quarter": "Q2",
            "fiscal_year": 2024,
            "confirmed": True,
            "time_of_day": "after_market",
        },
    ]

    return earnings_data


@pytest.fixture
async def event_collector():
    """Comprehensive event collector for end-to-end testing."""

    class ComprehensiveEventCollector:
        def __init__(self):
            self.all_events = []
            self.scanner_alerts = []
            self.feature_events = []
            self.system_events = []
            self.event_counts = {}
            self.processing_times = []
            self.error_events = []

        async def publish(self, event):
            """Collect all published events."""
            start_time = datetime.now()

            self.all_events.append(
                {"event": event, "timestamp": start_time, "event_id": len(self.all_events)}
            )

            # Categorize events
            if isinstance(event, ScannerAlertEvent):
                self.scanner_alerts.append(event)
                event_type = "scanner_alert"
            elif hasattr(event, "event_type"):
                event_type = event.event_type
                if event_type == "FEATURE_EXTRACTED":
                    self.feature_events.append(event)
                else:
                    self.system_events.append(event)
            else:
                event_type = "unknown"
                self.system_events.append(event)

            # Count events by type
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

            # Simulate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)

        async def subscribe(self, event_type: str, handler):
            """Mock subscription."""
            pass

        async def unsubscribe(self, event_type: str, handler):
            """Mock unsubscription."""
            pass

        def get_events_by_scanner(self, scanner_name: str) -> list[ScannerAlertEvent]:
            """Get events from specific scanner."""
            return [
                event
                for event in self.scanner_alerts
                if getattr(event, "scanner_name", "") == scanner_name
            ]

        def get_events_by_symbol(self, symbol: str) -> list[ScannerAlertEvent]:
            """Get events for specific symbol."""
            return [
                event for event in self.scanner_alerts if getattr(event, "symbol", "") == symbol
            ]

        def get_high_confidence_alerts(self, threshold: float = 0.8) -> list[ScannerAlertEvent]:
            """Get high-confidence alerts."""
            return [
                event for event in self.scanner_alerts if getattr(event, "score", 0) >= threshold
            ]

        def get_performance_metrics(self) -> dict[str, Any]:
            """Get performance metrics."""
            if not self.processing_times:
                return {}

            return {
                "total_events": len(self.all_events),
                "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
                "max_processing_time_ms": max(self.processing_times),
                "min_processing_time_ms": min(self.processing_times),
                "events_per_second": (
                    len(self.all_events) / (max(self.processing_times) / 1000)
                    if self.processing_times
                    else 0
                ),
                "event_type_counts": self.event_counts.copy(),
            }

        def clear(self):
            """Clear all collected data."""
            self.all_events.clear()
            self.scanner_alerts.clear()
            self.feature_events.clear()
            self.system_events.clear()
            self.event_counts.clear()
            self.processing_times.clear()
            self.error_events.clear()

    return ComprehensiveEventCollector()


@pytest.fixture
def end_to_end_performance_thresholds():
    """Performance thresholds for end-to-end testing."""
    return {
        "scan_completion_time_ms": 30000,  # 30 seconds max for full scan
        "alert_generation_time_ms": 5000,  # 5 seconds max per scanner
        "event_publishing_time_ms": 100,  # 100ms max per event
        "orchestrator_overhead_ms": 2000,  # 2 seconds max orchestrator overhead
        "memory_usage_mb": 500,  # 500MB max memory usage
        "alert_accuracy_threshold": 0.7,  # 70% min accuracy for generated alerts
        "system_availability": 0.99,  # 99% system availability
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async testing."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

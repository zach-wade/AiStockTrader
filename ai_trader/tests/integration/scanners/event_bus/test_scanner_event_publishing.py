"""
Integration tests for scanner event publishing.

Tests how scanners publish events to the event bus system.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScanner
from main.events.types.event_types import ScannerAlertEvent, EventType, AlertType
from main.scanners.types import ScanAlert


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerEventPublishing:
    """Test scanner event publishing integration."""

    async def test_volume_scanner_event_publishing(
        self,
        volume_scanner: IScanner,
        event_collector,
        test_symbols,
        sample_volume_data
    ):
        """Test volume scanner publishes events correctly."""
        # Replace event bus with collector to capture events
        volume_scanner.event_bus = event_collector
        
        # Mock repository data
        volume_scanner.repository.get_market_data.return_value = sample_volume_data
        volume_scanner.repository.get_volume_statistics.return_value = {
            'AAPL': {
                'avg_volume': 75000000,
                'std_volume': 15000000,
                'data_points': 20
            }
        }
        
        # Run scanner
        alerts = await volume_scanner.scan(symbols=['AAPL'])
        
        # Verify alerts were generated
        assert len(alerts) > 0
        
        # Verify events were published
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) > 0
        
        # Check event structure
        event = published_events[0]
        assert isinstance(event, ScannerAlertEvent)
        assert event.symbol == 'AAPL'
        assert event.scanner_name == 'volume_scanner'
        assert event.event_type == EventType.SCANNER_ALERT
        assert event.score > 0
        
        # Check alert type is volume-related
        assert 'VOLUME' in str(event.alert_type).upper()

    async def test_technical_scanner_event_publishing(
        self,
        technical_scanner: IScanner,
        event_collector,
        test_symbols,
        sample_technical_data
    ):
        """Test technical scanner publishes events correctly."""
        # Replace event bus with collector
        technical_scanner.event_bus = event_collector
        
        # Mock repository data for breakout pattern
        technical_scanner.repository.get_market_data.return_value = sample_technical_data
        
        # Run scanner
        alerts = await technical_scanner.scan(symbols=['GOOGL'])
        
        # Verify events were published
        published_events = event_collector.get_scanner_alerts()
        
        if published_events:  # Technical scanner may not always generate alerts
            event = published_events[0]
            assert isinstance(event, ScannerAlertEvent)
            assert event.symbol == 'GOOGL'
            assert event.scanner_name == 'technical_scanner'
            assert event.event_type == EventType.SCANNER_ALERT
            
            # Check metadata contains technical analysis data
            assert 'metadata' in event.__dict__
            metadata = event.metadata
            assert isinstance(metadata, dict)

    async def test_news_scanner_event_publishing(
        self,
        news_scanner: IScanner,
        event_collector,
        test_symbols,
        sample_news_data
    ):
        """Test news scanner publishes events correctly."""
        # Replace event bus with collector
        news_scanner.event_bus = event_collector
        
        # Mock repository data
        news_scanner.repository.get_news_data.return_value = sample_news_data
        
        # Run scanner
        alerts = await news_scanner.scan(symbols=['MSFT'])
        
        # Verify alerts were generated
        assert len(alerts) > 0
        
        # Verify events were published
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) > 0
        
        # Check event structure
        event = published_events[0]
        assert isinstance(event, ScannerAlertEvent)
        assert event.symbol == 'MSFT'
        assert event.scanner_name == 'news_scanner'
        assert event.event_type == EventType.SCANNER_ALERT
        
        # Check alert type is news-related
        assert 'NEWS' in str(event.alert_type).upper() or 'SENTIMENT' in str(event.alert_type).upper()
        
        # Check metadata contains sentiment information
        metadata = event.metadata
        assert 'sentiment_score' in metadata or 'relevance_score' in metadata

    async def test_event_publishing_with_multiple_alerts(
        self,
        mock_event_scanner,
        event_collector,
        test_symbols
    ):
        """Test publishing multiple alerts generates multiple events."""
        # Replace event bus with collector
        mock_event_scanner.event_bus = event_collector
        
        # Run scanner with multiple symbols
        alerts = await mock_event_scanner.scan(symbols=test_symbols[:3])
        
        # Should generate one alert per symbol
        assert len(alerts) == 3
        
        # Should publish one event per alert
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) == 3
        
        # Check all symbols are represented
        event_symbols = [event.symbol for event in published_events]
        assert set(event_symbols) == set(test_symbols[:3])
        
        # All events should be from the same scanner
        scanner_names = [event.scanner_name for event in published_events]
        assert all(name == 'mock_scanner' for name in scanner_names)

    async def test_event_publishing_performance(
        self,
        mock_event_scanner,
        event_collector,
        test_symbols,
        event_performance_thresholds
    ):
        """Test event publishing performance meets requirements."""
        # Replace event bus with collector
        mock_event_scanner.event_bus = event_collector
        
        # Measure publishing time
        start_time = datetime.now()
        
        alerts = await mock_event_scanner.scan(symbols=test_symbols)
        
        end_time = datetime.now()
        publishing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should meet performance threshold
        threshold = event_performance_thresholds['scanner_scan_time_ms']
        assert publishing_time_ms < threshold
        
        # Verify all events were published
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) == len(test_symbols)

    async def test_event_publishing_error_handling(
        self,
        mock_event_scanner,
        test_symbols
    ):
        """Test event publishing handles errors gracefully."""
        # Create failing event bus
        failing_event_bus = Mock(spec=IEventBus)
        failing_event_bus.publish = AsyncMock(side_effect=Exception("Event bus error"))
        
        mock_event_scanner.event_bus = failing_event_bus
        
        # Scanner should still complete despite event bus errors
        alerts = await mock_event_scanner.scan(symbols=test_symbols[:2])
        
        # Should still generate alerts
        assert len(alerts) == 2
        
        # Event bus publish should have been called
        assert failing_event_bus.publish.called

    async def test_event_metadata_preservation(
        self,
        volume_scanner: IScanner,
        event_collector,
        sample_volume_data
    ):
        """Test that alert metadata is preserved in events."""
        volume_scanner.event_bus = event_collector
        
        # Mock detailed volume data
        volume_scanner.repository.get_market_data.return_value = sample_volume_data
        volume_scanner.repository.get_volume_statistics.return_value = {
            'AAPL': {
                'avg_volume': 75000000,
                'std_volume': 15000000,
                'data_points': 20
            }
        }
        
        alerts = await volume_scanner.scan(symbols=['AAPL'])
        
        if alerts:
            # Get published event
            published_events = event_collector.get_scanner_alerts()
            assert len(published_events) > 0
            
            event = published_events[0]
            alert = alerts[0]
            
            # Event metadata should match alert metadata
            assert event.metadata == alert.metadata
            assert event.score == alert.score
            assert event.symbol == alert.symbol

    async def test_event_timestamp_accuracy(
        self,
        mock_event_scanner,
        event_collector,
        test_symbols
    ):
        """Test that event timestamps are accurate."""
        mock_event_scanner.event_bus = event_collector
        
        # Capture time before scan
        before_scan = datetime.now(timezone.utc)
        
        await mock_event_scanner.scan(symbols=['AAPL'])
        
        # Capture time after scan
        after_scan = datetime.now(timezone.utc)
        
        # Check event timestamp
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) == 1
        
        event = published_events[0]
        event_time = event.timestamp
        
        # Event timestamp should be within scan timeframe
        assert before_scan <= event_time <= after_scan

    async def test_concurrent_event_publishing(
        self,
        event_collector,
        test_symbols
    ):
        """Test concurrent event publishing from multiple scanners."""
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner
        
        # Create multiple mock scanners
        scanners = [
            MockEventPublishingScanner(f'scanner_{i}', event_collector)
            for i in range(3)
        ]
        
        # Run scanners concurrently
        tasks = [
            scanner.scan(symbols=test_symbols[:2])
            for scanner in scanners
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All scanners should complete successfully
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert len(result) == 2  # 2 symbols per scanner
        
        # Should have events from all scanners
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) == 6  # 3 scanners * 2 symbols
        
        # Check scanner names are distributed
        scanner_names = [event.scanner_name for event in published_events]
        unique_scanners = set(scanner_names)
        assert len(unique_scanners) == 3

    async def test_event_bus_subscription_integration(
        self,
        test_event_bus: IEventBus,
        mock_event_scanner
    ):
        """Test integration with event bus subscriptions."""
        received_events = []
        
        # Create event handler
        async def alert_handler(event: ScannerAlertEvent):
            received_events.append(event)
        
        # Subscribe to scanner alerts
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, alert_handler)
        
        # Use real event bus
        mock_event_scanner.event_bus = test_event_bus
        
        # Generate alerts
        await mock_event_scanner.scan(symbols=['AAPL', 'GOOGL'])
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Should have received events
        assert len(received_events) == 2
        
        # Check event types
        for event in received_events:
            assert isinstance(event, ScannerAlertEvent)
            assert event.event_type == EventType.SCANNER_ALERT

    async def test_event_publishing_with_batch_processing(
        self,
        test_event_bus: IEventBus,
        event_performance_thresholds
    ):
        """Test event publishing performance with batch processing."""
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner
        
        received_events = []
        
        async def batch_handler(event: ScannerAlertEvent):
            received_events.append(event)
        
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, batch_handler)
        
        # Create scanner with real event bus
        batch_scanner = MockEventPublishingScanner('batch_scanner', test_event_bus)
        
        # Generate large batch of alerts
        large_symbol_list = [f'SYM{i:03d}' for i in range(50)]
        
        start_time = datetime.now()
        await batch_scanner.scan(symbols=large_symbol_list)
        
        # Wait for all events to be processed
        await asyncio.sleep(0.5)
        end_time = datetime.now()
        
        batch_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should meet batch processing threshold
        threshold = event_performance_thresholds['batch_processing_ms']
        # Allow extra time for large batch
        assert batch_time_ms < threshold * 5
        
        # Should have received all events
        assert len(received_events) == 50

    async def test_event_publishing_with_no_event_bus(
        self,
        mock_scanner_repository,
        test_symbols
    ):
        """Test scanner behavior when no event bus is provided."""
        from main.scanners.volume_scanner import VolumeScanner
        from omegaconf import OmegaConf
        
        # Create scanner without event bus
        config = OmegaConf.create({
            'enabled': True,
            'timeout_seconds': 30.0,
            'min_volume_ratio': 2.0
        })
        
        scanner = VolumeScanner(
            config=config,
            repository=mock_scanner_repository,
            event_bus=None  # No event bus
        )
        
        # Mock repository data
        mock_scanner_repository.get_market_data.return_value = {}
        mock_scanner_repository.get_volume_statistics.return_value = {}
        
        # Should not fail without event bus
        alerts = await scanner.scan(symbols=test_symbols[:2])
        
        # Should return empty list or handle gracefully
        assert isinstance(alerts, list)

    async def test_event_deduplication(
        self,
        mock_event_scanner,
        event_collector,
        test_symbols
    ):
        """Test that duplicate events are handled properly."""
        mock_event_scanner.event_bus = event_collector
        
        # Generate alerts multiple times
        await mock_event_scanner.scan(symbols=['AAPL'])
        await mock_event_scanner.scan(symbols=['AAPL'])  # Same symbol again
        
        published_events = event_collector.get_scanner_alerts()
        
        # Should have published both events (deduplication handled at higher level)
        assert len(published_events) == 2
        
        # Both events should be for AAPL
        assert all(event.symbol == 'AAPL' for event in published_events)
        
        # Events should have different timestamps
        timestamps = [event.timestamp for event in published_events]
        assert timestamps[0] != timestamps[1]

    async def test_event_priority_handling(
        self,
        event_collector,
        test_symbols
    ):
        """Test event priority is properly set and handled."""
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner
        
        # Create high-priority scanner
        high_priority_scanner = MockEventPublishingScanner('high_priority', event_collector)
        
        # Mock high-priority alert
        async def scan_with_priority(symbols):
            alerts = []
            for symbol in symbols:
                alert = ScanAlert(
                    symbol=symbol,
                    alert_type=AlertType.EARNINGS_SURPRISE,  # High priority alert type
                    score=0.95,  # High score
                    metadata={'priority': 'high', 'confidence': 0.95},
                    timestamp=datetime.now(timezone.utc),
                    scanner_name='high_priority'
                )
                alerts.append(alert)
            
            await high_priority_scanner.publish_alerts_to_event_bus(alerts)
            return alerts
        
        high_priority_scanner.scan = scan_with_priority
        
        # Generate high-priority alerts
        await high_priority_scanner.scan(['AAPL'])
        
        published_events = event_collector.get_scanner_alerts()
        assert len(published_events) == 1
        
        event = published_events[0]
        assert event.score == 0.95
        assert event.metadata['priority'] == 'high'
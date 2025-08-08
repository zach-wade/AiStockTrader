"""
End-to-end integration tests for complete scanner workflows.

Tests the entire scanner pipeline from data ingestion to alert generation
and event publishing.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import psutil
import os

from main.interfaces.scanners import IScannerOrchestrator, IScannerRepository
from main.interfaces.events import IEventBus
from main.events.types.event_types import ScannerAlertEvent, AlertType
from main.scanners.types import ScanAlert


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteScannerWorkflow:
    """Test complete end-to-end scanner workflows."""

    async def test_full_orchestrated_scan_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        end_to_end_repository: IScannerRepository,
        event_collector,
        comprehensive_test_symbols,
        realistic_market_data,
        realistic_news_data,
        realistic_social_data,
        realistic_earnings_data,
        end_to_end_performance_thresholds
    ):
        """Test complete orchestrated scan workflow with all scanners."""
        # Replace event bus with collector
        end_to_end_orchestrator.event_bus = event_collector
        
        # Mock repository data with realistic datasets
        end_to_end_repository.get_market_data = AsyncMock(return_value=realistic_market_data)
        end_to_end_repository.get_news_data = AsyncMock(return_value=realistic_news_data)
        end_to_end_repository.get_social_sentiment = AsyncMock(return_value=realistic_social_data)
        end_to_end_repository.get_earnings_data = AsyncMock(return_value=realistic_earnings_data)
        
        # Mock volume statistics for volume scanner
        volume_stats = {
            symbol: {
                'avg_volume': realistic_market_data[symbol]['volume'].mean(),
                'std_volume': realistic_market_data[symbol]['volume'].std(),
                'data_points': len(realistic_market_data[symbol])
            }
            for symbol in realistic_market_data.keys()
        }
        end_to_end_repository.get_volume_statistics = AsyncMock(return_value=volume_stats)
        
        # Record start time and memory
        start_time = datetime.now()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run orchestrated scan
        results = await end_to_end_orchestrator.run_scan(
            symbols=comprehensive_test_symbols[:10]  # Test with 10 symbols
        )
        
        # Record end time and memory
        end_time = datetime.now()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        memory_usage_mb = end_memory - start_memory
        
        # Verify scan completion
        assert isinstance(results, dict)
        assert len(results) > 0  # Should have results from enabled scanners
        
        # Check performance thresholds
        assert scan_time_ms < end_to_end_performance_thresholds['scan_completion_time_ms']
        assert memory_usage_mb < end_to_end_performance_thresholds['memory_usage_mb']
        
        # Verify events were published
        published_alerts = event_collector.scanner_alerts
        assert len(published_alerts) > 0
        
        # Check event diversity (multiple scanners should have generated alerts)
        scanner_names = set(alert.scanner_name for alert in published_alerts)
        assert len(scanner_names) >= 2  # At least 2 different scanners
        
        # Verify alert quality
        high_confidence_alerts = event_collector.get_high_confidence_alerts(0.7)
        accuracy_ratio = len(high_confidence_alerts) / len(published_alerts) if published_alerts else 0
        assert accuracy_ratio >= end_to_end_performance_thresholds['alert_accuracy_threshold']
        
        # Check performance metrics
        metrics = event_collector.get_performance_metrics()
        assert metrics['total_events'] > 0
        assert metrics['avg_processing_time_ms'] < end_to_end_performance_thresholds['event_publishing_time_ms']

    async def test_volume_scanner_end_to_end_workflow(
        self,
        end_to_end_scanner_factory,
        event_collector,
        realistic_market_data,
        end_to_end_performance_thresholds
    ):
        """Test complete volume scanner workflow from data to alerts."""
        # Create volume scanner
        volume_scanner = await end_to_end_scanner_factory.create_scanner('volume')
        volume_scanner.event_bus = event_collector
        
        # Mock repository with volume spike data
        volume_scanner.repository.get_market_data = AsyncMock(return_value=realistic_market_data)
        
        # Mock volume statistics showing normal baseline
        volume_stats = {}
        for symbol in realistic_market_data.keys():
            df = realistic_market_data[symbol]
            # Exclude today's spike from average
            historical_volumes = df[df['timestamp'] < datetime.now(timezone.utc) - timedelta(hours=1)]['volume']
            
            volume_stats[symbol] = {
                'avg_volume': historical_volumes.mean(),
                'std_volume': historical_volumes.std(),
                'min_volume': historical_volumes.min(),
                'max_volume': historical_volumes.max(),
                'data_points': len(historical_volumes)
            }
        
        volume_scanner.repository.get_volume_statistics = AsyncMock(return_value=volume_stats)
        
        # Run volume scan
        start_time = datetime.now()
        alerts = await volume_scanner.scan(symbols=['AAPL', 'TSLA'])  # These have volume spikes
        end_time = datetime.now()
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance
        assert scan_time_ms < end_to_end_performance_thresholds['alert_generation_time_ms']
        
        # Verify alerts generated
        assert len(alerts) > 0
        
        # Check alert quality
        for alert in alerts:
            assert isinstance(alert, ScanAlert)
            assert alert.symbol in ['AAPL', 'TSLA']
            assert alert.alert_type == AlertType.VOLUME_SPIKE
            assert alert.score > 0.5  # Reasonable confidence
            assert 'relative_volume' in alert.metadata
            assert alert.metadata['relative_volume'] > 1.5  # Significant volume increase
        
        # Verify events published
        published_events = event_collector.get_events_by_scanner('volume_scanner')
        assert len(published_events) == len(alerts)
        
        # Check event-alert consistency
        for alert, event in zip(alerts, published_events):
            assert event.symbol == alert.symbol
            assert event.alert_type == str(alert.alert_type)
            assert event.score == alert.score
            assert event.metadata == alert.metadata

    async def test_news_scanner_end_to_end_workflow(
        self,
        end_to_end_scanner_factory,
        event_collector,
        realistic_news_data,
        end_to_end_performance_thresholds
    ):
        """Test complete news scanner workflow from news data to alerts."""
        # Create news scanner
        news_scanner = await end_to_end_scanner_factory.create_scanner('news')
        news_scanner.event_bus = event_collector
        
        # Mock repository with news data
        news_scanner.repository.get_news_data = AsyncMock(return_value=realistic_news_data)
        
        # Run news scan
        start_time = datetime.now()
        alerts = await news_scanner.scan(symbols=['AAPL', 'GOOGL', 'MSFT'])
        end_time = datetime.now()
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance
        assert scan_time_ms < end_to_end_performance_thresholds['alert_generation_time_ms']
        
        # Verify alerts generated for symbols with significant news
        assert len(alerts) > 0
        
        # Check alert quality
        for alert in alerts:
            assert isinstance(alert, ScanAlert)
            assert alert.symbol in ['AAPL', 'GOOGL', 'MSFT']
            assert alert.alert_type in [AlertType.NEWS_SENTIMENT, AlertType.BREAKING_NEWS]
            assert alert.score >= 0.7  # High-quality news should have high scores
            assert 'sentiment_score' in alert.metadata or 'news_count' in alert.metadata
        
        # Verify breaking news detection
        breaking_news_alerts = [
            alert for alert in alerts 
            if alert.metadata.get('breaking_news', False)
        ]
        assert len(breaking_news_alerts) > 0  # Should detect breaking news
        
        # Verify events published
        published_events = event_collector.get_events_by_scanner('news_scanner')
        assert len(published_events) == len(alerts)

    async def test_earnings_scanner_end_to_end_workflow(
        self,
        end_to_end_scanner_factory,
        event_collector,
        realistic_earnings_data,
        end_to_end_performance_thresholds
    ):
        """Test complete earnings scanner workflow from earnings data to alerts."""
        # Create earnings scanner
        earnings_scanner = await end_to_end_scanner_factory.create_scanner('earnings')
        earnings_scanner.event_bus = event_collector
        
        # Mock repository with earnings data
        earnings_scanner.repository.get_earnings_data = AsyncMock(return_value=realistic_earnings_data)
        
        # Run earnings scan
        start_time = datetime.now()
        alerts = await earnings_scanner.scan(symbols=['AAPL', 'GOOGL'])
        end_time = datetime.now()
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance
        assert scan_time_ms < end_to_end_performance_thresholds['alert_generation_time_ms']
        
        # Verify alerts generated for earnings surprises
        assert len(alerts) > 0
        
        # Check alert quality
        for alert in alerts:
            assert isinstance(alert, ScanAlert)
            assert alert.symbol in ['AAPL', 'GOOGL']
            assert alert.alert_type == AlertType.EARNINGS_SURPRISE
            assert alert.score > 0.7  # Earnings surprises should be high confidence
            assert 'surprise_percent' in alert.metadata
            assert abs(alert.metadata['surprise_percent']) > 3.0  # Significant surprise
        
        # Verify events published
        published_events = event_collector.get_events_by_scanner('earnings_scanner')
        assert len(published_events) == len(alerts)

    async def test_multi_scanner_correlation_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        realistic_market_data,
        realistic_news_data,
        realistic_social_data,
        comprehensive_test_symbols
    ):
        """Test multi-scanner correlation and cross-validation."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector
        
        # Mock repository with correlated data (AAPL has volume spike + positive news + social buzz)
        end_to_end_orchestrator.repository.get_market_data = AsyncMock(return_value=realistic_market_data)
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(return_value=realistic_news_data)
        end_to_end_orchestrator.repository.get_social_sentiment = AsyncMock(return_value=realistic_social_data)
        
        # Mock volume statistics
        volume_stats = {
            'AAPL': {'avg_volume': 50000000, 'std_volume': 10000000, 'data_points': 30}
        }
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value=volume_stats)
        
        # Run orchestrated scan
        results = await end_to_end_orchestrator.run_scan(symbols=['AAPL'])
        
        # Verify multiple scanner types generated alerts for AAPL
        aapl_events = event_collector.get_events_by_symbol('AAPL')
        
        # Should have alerts from multiple scanners (correlated signals)
        scanner_types = set(event.scanner_name for event in aapl_events)
        assert len(scanner_types) >= 2  # At least volume + news or volume + social
        
        # Check for correlation patterns
        volume_alerts = [e for e in aapl_events if 'volume' in e.scanner_name]
        news_alerts = [e for e in aapl_events if 'news' in e.scanner_name]
        social_alerts = [e for e in aapl_events if 'social' in e.scanner_name]
        
        # AAPL should trigger multiple types of alerts (correlated signals)
        total_signal_types = len([alerts for alerts in [volume_alerts, news_alerts, social_alerts] if alerts])
        assert total_signal_types >= 2
        
        # Combined signals should indicate strong bullish sentiment
        combined_score = max(event.score for event in aapl_events)
        assert combined_score > 0.8  # Strong combined signal

    async def test_real_time_scanning_workflow(
        self,
        end_to_end_scanner_factory,
        event_collector,
        comprehensive_test_symbols,
        end_to_end_performance_thresholds
    ):
        """Test real-time scanning workflow with continuous updates."""
        # Create volume scanner for real-time testing
        volume_scanner = await end_to_end_scanner_factory.create_scanner('volume')
        volume_scanner.event_bus = event_collector
        
        # Mock real-time data updates
        real_time_updates = []
        
        async def mock_get_latest_prices(symbols):
            """Mock real-time price updates."""
            return {
                symbol: {
                    'timestamp': datetime.now(timezone.utc),
                    'close': 150.0 + len(real_time_updates) * 0.5,  # Price moving up
                    'volume': 100000000 + len(real_time_updates) * 5000000,  # Volume increasing
                    'returns': 0.01
                }
                for symbol in symbols
            }
        
        volume_scanner.repository.get_latest_prices = mock_get_latest_prices
        volume_scanner.repository.get_volume_statistics = AsyncMock(return_value={
            'AAPL': {'avg_volume': 75000000, 'std_volume': 15000000, 'data_points': 20}
        })
        
        # Simulate real-time scanning (multiple rapid scans)
        all_alerts = []
        scan_times = []
        
        for i in range(5):  # 5 rapid scans
            start_time = datetime.now()
            alerts = await volume_scanner.scan(symbols=['AAPL'])
            end_time = datetime.now()
            
            scan_time_ms = (end_time - start_time).total_seconds() * 1000
            scan_times.append(scan_time_ms)
            all_alerts.extend(alerts)
            real_time_updates.append(i)
            
            await asyncio.sleep(0.1)  # Brief pause between scans
        
        # Verify real-time performance
        avg_scan_time = sum(scan_times) / len(scan_times)
        assert avg_scan_time < end_to_end_performance_thresholds['alert_generation_time_ms']
        
        # Should detect increasing volume pattern
        if all_alerts:
            # Later alerts should have higher volume ratios
            volume_ratios = [
                alert.metadata.get('relative_volume', 1.0) 
                for alert in all_alerts
            ]
            # Volume should be increasing over time
            assert max(volume_ratios) > min(volume_ratios)
        
        # Verify events published in real-time
        published_events = event_collector.get_events_by_scanner('volume_scanner')
        assert len(published_events) == len(all_alerts)

    async def test_error_recovery_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        end_to_end_performance_thresholds
    ):
        """Test error recovery and fault tolerance in complete workflow."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector
        
        # Mock repository with intermittent failures
        failure_count = 0
        
        async def failing_get_market_data(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # First 2 calls fail
                raise Exception("Database connection timeout")
            else:  # Subsequent calls succeed
                return {'AAPL': Mock()}  # Mock success
        
        async def failing_get_news_data(*args, **kwargs):
            nonlocal failure_count
            if failure_count <= 1:  # First call fails
                raise Exception("News API rate limit")
            else:
                return []  # Empty but successful
        
        end_to_end_orchestrator.repository.get_market_data = failing_get_market_data
        end_to_end_orchestrator.repository.get_news_data = failing_get_news_data
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_social_sentiment = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_earnings_data = AsyncMock(return_value=[])
        
        # Run scan with error recovery
        start_time = datetime.now()
        results = await end_to_end_orchestrator.run_scan(symbols=['AAPL'])
        end_time = datetime.now()
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should complete despite initial failures
        assert isinstance(results, dict)
        
        # Should meet availability threshold (system recovered)
        availability_score = 1.0 if results else 0.0  # Simple binary availability
        assert availability_score >= end_to_end_performance_thresholds['system_availability']
        
        # Performance should still be reasonable despite retries
        # Allow extra time for error recovery
        assert scan_time_ms < end_to_end_performance_thresholds['scan_completion_time_ms'] * 2

    async def test_data_quality_validation_workflow(
        self,
        end_to_end_scanner_factory,
        event_collector,
        comprehensive_test_symbols
    ):
        """Test data quality validation in end-to-end workflow."""
        # Create technical scanner for data quality testing
        technical_scanner = await end_to_end_scanner_factory.create_scanner('technical')
        technical_scanner.event_bus = event_collector
        
        # Mock repository with mixed quality data
        good_data = {
            'AAPL': Mock(spec=['iloc', 'empty', '__len__']),
        }
        good_data['AAPL'].empty = False
        good_data['AAPL'].__len__ = lambda: 50  # Sufficient data points
        
        bad_data = {
            'GOOGL': Mock(spec=['empty', '__len__']),
        }
        bad_data['GOOGL'].empty = True  # No data
        bad_data['GOOGL'].__len__ = lambda: 0
        
        async def quality_aware_get_market_data(symbols, *args, **kwargs):
            result = {}
            for symbol in symbols:
                if symbol == 'AAPL':
                    result[symbol] = good_data[symbol]
                elif symbol == 'GOOGL':
                    result[symbol] = bad_data[symbol]
            return result
        
        technical_scanner.repository.get_market_data = quality_aware_get_market_data
        
        # Run scan with mixed quality data
        alerts = await technical_scanner.scan(symbols=['AAPL', 'GOOGL'])
        
        # Should only generate alerts for symbols with good data
        if alerts:
            alert_symbols = [alert.symbol for alert in alerts]
            # Should prefer AAPL (good data) over GOOGL (bad data)
            good_data_alerts = [s for s in alert_symbols if s == 'AAPL']
            bad_data_alerts = [s for s in alert_symbols if s == 'GOOGL']
            
            # Good data should generate more/better alerts
            assert len(good_data_alerts) >= len(bad_data_alerts)

    async def test_complete_feature_pipeline_integration(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        end_to_end_event_bus: IEventBus,
        realistic_market_data,
        realistic_news_data,
        comprehensive_test_symbols
    ):
        """Test complete integration with feature pipeline."""
        # Mock feature pipeline
        extracted_features = []
        
        async def feature_pipeline_handler(event):
            """Mock feature pipeline handler."""
            if isinstance(event, ScannerAlertEvent):
                features = {
                    'symbol': event.symbol,
                    'alert_type': event.alert_type,
                    'score': event.score,
                    'scanner_name': event.scanner_name,
                    'timestamp': event.timestamp,
                    'extracted_at': datetime.now(timezone.utc)
                }
                extracted_features.append(features)
        
        # Subscribe to scanner alerts
        await end_to_end_event_bus.subscribe(
            EventType.SCANNER_ALERT,
            feature_pipeline_handler
        )
        
        # Mock repository data
        end_to_end_orchestrator.repository.get_market_data = AsyncMock(return_value=realistic_market_data)
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(return_value=realistic_news_data)
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={
            'AAPL': {'avg_volume': 50000000, 'std_volume': 10000000, 'data_points': 20}
        })
        
        # Run orchestrated scan
        results = await end_to_end_orchestrator.run_scan(symbols=['AAPL'])
        
        # Wait for feature extraction
        await asyncio.sleep(0.2)
        
        # Verify feature extraction occurred
        assert len(extracted_features) > 0
        
        # Check feature quality
        for features in extracted_features:
            assert features['symbol'] == 'AAPL'
            assert features['score'] > 0
            assert features['scanner_name'] is not None
            assert features['timestamp'] is not None
            assert features['extracted_at'] is not None

    async def test_system_scalability_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        end_to_end_performance_thresholds
    ):
        """Test system scalability with large symbol sets."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector
        
        # Mock repository with scalable responses
        end_to_end_orchestrator.repository.get_market_data = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(return_value=[])
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_social_sentiment = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_earnings_data = AsyncMock(return_value=[])
        
        # Test with large symbol set
        large_symbol_set = comprehensive_test_symbols  # All 25 symbols
        
        # Record performance metrics
        start_time = datetime.now()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        results = await end_to_end_orchestrator.run_scan(symbols=large_symbol_set)
        
        end_time = datetime.now()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        scan_time_ms = (end_time - start_time).total_seconds() * 1000
        memory_usage_mb = end_memory - start_memory
        
        # Verify scalability performance
        # Scale thresholds based on symbol count
        symbol_scale_factor = len(large_symbol_set) / 5  # Base threshold for 5 symbols
        scaled_time_threshold = end_to_end_performance_thresholds['scan_completion_time_ms'] * symbol_scale_factor
        scaled_memory_threshold = end_to_end_performance_thresholds['memory_usage_mb'] * symbol_scale_factor
        
        assert scan_time_ms < scaled_time_threshold
        assert memory_usage_mb < scaled_memory_threshold
        
        # System should handle large loads gracefully
        assert isinstance(results, dict)
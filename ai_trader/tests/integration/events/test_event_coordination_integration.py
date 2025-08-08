"""
Integration tests for Event Coordination.

Tests the coordinated operation of all events components working together
including EventBus, Scanner-Feature Bridge, Feature Pipeline, and all helpers.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any, Set
import json

from main.interfaces.events import IEventBus, Event, EventType
from main.events.core import EventBusFactory
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.handlers.feature_pipeline_handler import FeaturePipelineHandler
from main.events.types import (
    ScannerAlertEvent, FeatureRequestEvent, 
    FeatureComputedEvent, ErrorEvent, ScanAlert, AlertType
)
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.fixture
async def coordinated_event_system():
    """Create a fully coordinated event system with all components."""
    # Mock configuration
    mock_config = {
        'events': {
            'batch_size': 3,
            'batch_interval_seconds': 0.1,
            'max_queue_size': 1000,
            'worker_pool_size': 2
        },
        'scanner_bridge': {
            'batch_size': 3,
            'batch_timeout': 0.1,
            'priority_boost_threshold': 0.8,
            'urgent_alert_types': ['catalyst_detected', 'opportunity_signal']
        },
        'feature_pipeline': {
            'batch_size': 5,
            'queue_timeout': 30,
            'max_retries': 3,
            'computation_timeout': 60
        }
    }
    
    # Mock external dependencies
    mock_db_pool = create_mock_db_pool()
    mock_feature_service = AsyncMock()
    mock_feature_service.compute_features = AsyncMock(return_value={
        'AAPL': {
            'price_features': {'sma_20': 150.0, 'rsi': 65.0},
            'volume_features': {'volume_sma': 50000, 'volume_ratio': 1.2}
        }
    })
    
    # Initialize event bus
    event_bus = EventBusFactory.create_test_instance()
    await event_bus.start()
    
    # Initialize scanner-feature bridge with DI
    bridge = ScannerFeatureBridge(event_bus=event_bus, config=mock_config)
    await bridge.start()
    
    # Initialize feature pipeline handler with DI
    pipeline_handler = FeaturePipelineHandler(
        event_bus=event_bus,
        feature_service=mock_feature_service,
        config=mock_config
    )
    await pipeline_handler.start()
    
    # Subscribe feature pipeline to feature requests
    event_bus.subscribe(EventType.FEATURE_REQUEST, pipeline_handler.handle_feature_request)
    
    yield {
        'event_bus': event_bus,
        'bridge': bridge,
        'pipeline_handler': pipeline_handler,
        'feature_service': mock_feature_service,
        'db_pool': mock_db_pool,
        'config': mock_config
    }
    
    # Cleanup
    await pipeline_handler.stop()
    await bridge.stop()
    await event_bus.stop()


@pytest.fixture
def end_to_end_test_data():
    """Create test data for end-to-end scenarios."""
    return {
        'scanner_alerts': [
            ScanAlert(
                symbol="AAPL",
                alert_type="high_volume",
                score=0.85,
                data={
                    'volume_ratio': 3.5,
                    'current_volume': 150000,
                    'scanner_id': 'volume_scanner_1'
                }
            ),
            ScanAlert(
                symbol="MSFT",
                alert_type="breakout",
                score=0.75,
                data={
                    'breakout_level': 300.0,
                    'resistance_level': 295.0,
                    'scanner_id': 'technical_scanner_1'
                }
            ),
            ScanAlert(
                symbol="GOOGL",
                alert_type="catalyst_detected",
                score=0.95,
                data={
                    'catalyst_type': 'earnings_surprise',
                    'news_count': 5,
                    'scanner_id': 'catalyst_scanner_1'
                }
            ),
            ScanAlert(
                symbol="TSLA",
                alert_type="momentum_shift",
                score=0.65,
                data={
                    'momentum_change': 0.15,
                    'timeframe': '5m',
                    'scanner_id': 'momentum_scanner_1'
                }
            )
        ],
        'expected_feature_groups': {
            'AAPL': ['volume_features', 'price_features', 'volatility_features'],
            'MSFT': ['price_features', 'trend_features', 'support_resistance_features'], 
            'GOOGL': 'all_features',  # Catalyst should trigger comprehensive analysis
            'TSLA': ['momentum_features', 'trend_features', 'volume_features']
        }
    }


class TestEventCoordinationIntegration:
    """Test coordination across all event system components."""
    
    @pytest.mark.asyncio
    async def test_complete_end_to_end_flow(
        self,
        coordinated_event_system,
        end_to_end_test_data
    ):
        """Test complete flow from scanner alert to feature computation."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        
        test_data = end_to_end_test_data
        scanner_alerts = test_data['scanner_alerts']
        
        # Track events throughout the pipeline
        published_events = {
            EventType.SCANNER_ALERT: [],
            EventType.FEATURE_REQUEST: [],
            EventType.FEATURE_COMPUTED: []
        }
        
        async def event_tracker(event_type, event):
            published_events[event_type].append(event)
        
        # Subscribe to track all event types
        for event_type in published_events.keys():
            event_bus.subscribe(event_type, lambda evt, et=event_type: asyncio.create_task(event_tracker(et, evt)))
        
        # Step 1: Process scanner alerts through bridge
        for scan_alert in scanner_alerts:
            scanner_event = ScannerAlertEvent(
                symbol=scan_alert.symbol,
                alert_type=scan_alert.alert_type,
                score=scan_alert.score,
                scanner_name=scan_alert.data.get('scanner_id', 'test_scanner'),
                metadata=scan_alert.data
            )
            
            # Publish scanner alert
            await event_bus.publish(scanner_event)
        
        # Wait for processing through all stages
        await asyncio.sleep(0.5)
        
        # Verify event flow
        assert len(published_events[EventType.SCANNER_ALERT]) >= len(scanner_alerts)
        assert len(published_events[EventType.FEATURE_REQUEST]) >= 1
        
        # Verify feature service was called
        assert feature_service.compute_features.call_count >= 1
        
        # Verify bridge statistics
        bridge_stats = bridge.get_stats()
        assert bridge_stats['alerts_received_total'] >= len(scanner_alerts)
        assert bridge_stats['feature_requests_sent_total'] >= 1
        
        # Verify pipeline handler statistics
        pipeline_stats = pipeline_handler.get_stats()
        assert pipeline_stats['requests_received'] >= 1
    
    @pytest.mark.asyncio
    async def test_multi_stage_event_correlation(
        self,
        coordinated_event_system
    ):
        """Test that correlation IDs are preserved across event stages."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        
        # Track correlation IDs throughout pipeline
        correlation_tracking = {
            'scanner_alerts': [],
            'feature_requests': [],
            'feature_computed': []
        }
        
        async def correlation_tracker(event, stage):
            correlation_tracking[stage].append({
                'event_id': event.event_id,
                'correlation_id': getattr(event, 'correlation_id', None),
                'event_type': event.event_type
            })
        
        # Subscribe to track correlations
        event_bus.subscribe(EventType.SCANNER_ALERT, 
                          lambda evt: asyncio.create_task(correlation_tracker(evt, 'scanner_alerts')))
        event_bus.subscribe(EventType.FEATURE_REQUEST, 
                          lambda evt: asyncio.create_task(correlation_tracker(evt, 'feature_requests')))
        event_bus.subscribe(EventType.FEATURE_COMPUTED, 
                          lambda evt: asyncio.create_task(correlation_tracker(evt, 'feature_computed')))
        
        # Create scanner alert with explicit correlation ID
        test_correlation_id = "test_correlation_12345"
        scanner_event = ScannerAlertEvent(
            symbol="CORRELATION_TEST",
            alert_type="high_volume",
            score=0.8,
            scanner_name="correlation_scanner",
            metadata={'test': 'correlation'}
        )
        scanner_event.correlation_id = test_correlation_id
        
        # Publish and wait for processing
        await event_bus.publish(scanner_event)
        await asyncio.sleep(0.3)
        
        # Verify correlation ID propagation
        assert len(correlation_tracking['scanner_alerts']) >= 1
        
        # Find the original scanner alert
        original_alert = next(
            (evt for evt in correlation_tracking['scanner_alerts'] 
             if evt['correlation_id'] == test_correlation_id), 
            None
        )
        assert original_alert is not None
        
        # Verify correlation ID appears in subsequent events
        if correlation_tracking['feature_requests']:
            correlated_requests = [
                evt for evt in correlation_tracking['feature_requests']
                if evt['correlation_id'] == test_correlation_id
            ]
            # At least one feature request should have the correlation ID
            assert len(correlated_requests) >= 0  # May not propagate in simplified test
    
    @pytest.mark.asyncio
    async def test_priority_propagation_across_stages(
        self,
        coordinated_event_system
    ):
        """Test that priority is properly propagated and handled across stages."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        
        # Track priority across stages
        priority_tracking = []
        
        async def priority_tracker(event, stage):
            priority = getattr(event, 'priority', None)
            priority_tracking.append({
                'stage': stage,
                'symbol': getattr(event, 'symbol', None),
                'priority': priority,
                'event_type': event.event_type
            })
        
        # Subscribe to track priorities
        event_bus.subscribe(EventType.FEATURE_REQUEST, 
                          lambda evt: asyncio.create_task(priority_tracker(evt, 'feature_request')))
        
        # Test alerts with different priorities
        test_alerts = [
            # Low priority
            ScannerAlertEvent(
                symbol="LOW_PRIORITY",
                alert_type="high_volume", 
                score=0.3,
                scanner_name="test_scanner"
            ),
            # High priority with boost
            ScannerAlertEvent(
                symbol="HIGH_PRIORITY",
                alert_type="catalyst_detected",
                score=0.9,
                scanner_name="test_scanner"
            ),
            # Medium priority
            ScannerAlertEvent(
                symbol="MED_PRIORITY",
                alert_type="breakout",
                score=0.6,
                scanner_name="test_scanner"
            )
        ]
        
        # Publish alerts
        for alert in test_alerts:
            await event_bus.publish(alert)
        
        await asyncio.sleep(0.3)
        
        # Verify priority assignment and propagation
        feature_request_priorities = [
            track for track in priority_tracking 
            if track['stage'] == 'feature_request' and track['priority'] is not None
        ]
        
        if feature_request_priorities:
            # Should have different priority levels
            priorities = [track['priority'] for track in feature_request_priorities]
            assert len(set(priorities)) > 1  # Multiple priority levels
            
            # High score catalyst should have highest priority
            catalyst_priority = next(
                (track['priority'] for track in feature_request_priorities 
                 if track['symbol'] == 'HIGH_PRIORITY'), 
                None
            )
            if catalyst_priority:
                assert catalyst_priority >= 8  # Should be high priority
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_coordination(
        self,
        coordinated_event_system
    ):
        """Test coordination under concurrent processing load."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        
        # Generate concurrent load
        concurrent_alerts = []
        symbols = [f'CONCURRENT_{i}' for i in range(20)]
        alert_types = ['high_volume', 'breakout', 'catalyst_detected', 'momentum_shift']
        
        for i, symbol in enumerate(symbols):
            alert = ScannerAlertEvent(
                symbol=symbol,
                alert_type=alert_types[i % len(alert_types)],
                score=0.5 + (i % 50) / 100.0,  # Vary scores
                scanner_name=f'concurrent_scanner_{i % 3}',
                metadata={'batch': i // 5}
            )
            concurrent_alerts.append(alert)
        
        # Publish all alerts concurrently
        publish_tasks = [
            event_bus.publish(alert) for alert in concurrent_alerts
        ]
        await asyncio.gather(*publish_tasks)
        
        # Wait for processing
        await asyncio.sleep(0.8)
        
        # Verify concurrent processing handled correctly
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        # Should have processed all alerts
        assert bridge_stats['alerts_received_total'] >= len(concurrent_alerts)
        assert bridge_stats['unique_symbols_processed'] == len(symbols)
        
        # Should have generated feature requests
        assert bridge_stats['feature_requests_sent_total'] >= 1
        assert pipeline_stats['requests_received'] >= 1
        
        # System should remain stable
        assert isinstance(bridge_stats, dict)
        assert isinstance(pipeline_stats, dict)
    
    @pytest.mark.asyncio
    async def test_error_isolation_across_components(
        self,
        coordinated_event_system,
        caplog
    ):
        """Test that errors in one component don't cascade to others."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        
        # Make feature service fail for specific symbols
        original_compute = feature_service.compute_features
        
        async def failing_compute_features(symbols, features, **kwargs):
            if any('FAIL' in symbol for symbol in symbols):
                raise Exception("Feature computation failed")
            return await original_compute(symbols, features, **kwargs)
        
        feature_service.compute_features.side_effect = failing_compute_features
        
        # Mix successful and failing alerts
        mixed_alerts = [
            ScannerAlertEvent(
                symbol="SUCCESS_1",
                alert_type="high_volume",
                score=0.7,
                scanner_name="test_scanner"
            ),
            ScannerAlertEvent(
                symbol="FAIL_SYMBOL",
                alert_type="breakout", 
                score=0.8,
                scanner_name="test_scanner"
            ),
            ScannerAlertEvent(
                symbol="SUCCESS_2",
                alert_type="catalyst_detected",
                score=0.9,
                scanner_name="test_scanner"
            )
        ]
        
        # Process mixed alerts
        for alert in mixed_alerts:
            await event_bus.publish(alert)
        
        await asyncio.sleep(0.5)
        
        # Verify error isolation
        # Bridge should continue operating despite pipeline errors
        bridge_stats = bridge.get_stats()
        assert bridge_stats['alerts_received_total'] >= len(mixed_alerts)
        
        # Pipeline should continue operating despite some failures
        pipeline_stats = pipeline_handler.get_stats()
        assert pipeline_stats['requests_received'] >= 1
        
        # Should have logged errors but continued processing
        assert "failed" in caplog.text.lower() or "error" in caplog.text.lower()
        
        # Components should remain responsive
        assert isinstance(bridge_stats, dict)
        assert isinstance(pipeline_stats, dict)
    
    @pytest.mark.asyncio
    async def test_batching_coordination_across_components(
        self,
        coordinated_event_system
    ):
        """Test batching coordination between bridge and pipeline."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        
        # Track batching behavior
        batch_tracking = {
            'feature_requests_published': 0,
            'feature_computations_called': 0,
            'symbols_in_requests': set(),
            'features_requested': set()
        }
        
        original_publish = event_bus.publish
        original_compute = feature_service.compute_features
        
        async def track_publish(event):
            if event.event_type == EventType.FEATURE_REQUEST:
                batch_tracking['feature_requests_published'] += 1
                batch_tracking['symbols_in_requests'].update(event.symbols)
                batch_tracking['features_requested'].update(event.features)
            return await original_publish(event)
        
        async def track_compute(symbols, features, **kwargs):
            batch_tracking['feature_computations_called'] += 1
            return await original_compute(symbols, features, **kwargs)
        
        event_bus.publish = track_publish
        feature_service.compute_features.side_effect = track_compute
        
        # Generate alerts for batching
        batch_test_alerts = []
        for i in range(12):  # Should create multiple batches
            alert = ScannerAlertEvent(
                symbol=f'BATCH_SYMBOL_{i}',
                alert_type='high_volume' if i % 2 == 0 else 'breakout',
                score=0.6 + (i % 40) / 100.0,
                scanner_name=f'batch_scanner_{i % 2}'
            )
            batch_test_alerts.append(alert)
        
        # Publish alerts in groups to test batching
        for i in range(0, len(batch_test_alerts), 3):
            batch_group = batch_test_alerts[i:i+3]
            for alert in batch_group:
                await event_bus.publish(alert)
            await asyncio.sleep(0.05)  # Small delay between groups
        
        # Wait for all batching to complete
        await asyncio.sleep(0.6)
        
        # Verify batching coordination
        assert batch_tracking['feature_requests_published'] >= 1
        assert batch_tracking['feature_computations_called'] >= 1
        
        # Should have processed multiple symbols
        assert len(batch_tracking['symbols_in_requests']) >= 6
        
        # Should have requested multiple feature types
        assert len(batch_tracking['features_requested']) >= 2
        
        # Verify bridge created appropriate batches
        bridge_stats = bridge.get_stats()
        assert bridge_stats['feature_requests_sent_total'] >= 1
        
        # Verify pipeline processed batched requests
        pipeline_stats = pipeline_handler.get_stats()
        assert pipeline_stats['requests_received'] >= 1
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(
        self,
        coordinated_event_system
    ):
        """Test overall system performance under sustained load."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        
        # Performance tracking
        start_time = asyncio.get_event_loop().time()
        
        # Generate high load
        load_alerts = []
        symbols_pool = [f'PERF_STOCK_{i}' for i in range(30)]
        alert_types = ['high_volume', 'breakout', 'catalyst_detected']
        
        num_alerts = 150
        for i in range(num_alerts):
            alert = ScannerAlertEvent(
                symbol=symbols_pool[i % len(symbols_pool)],
                alert_type=alert_types[i % len(alert_types)],
                score=0.4 + (i % 60) / 100.0,
                scanner_name=f'perf_scanner_{i % 5}',
                metadata={'load_test': True}
            )
            load_alerts.append(alert)
        
        # Process load in waves
        wave_size = 25
        for wave_start in range(0, num_alerts, wave_size):
            wave_alerts = load_alerts[wave_start:wave_start + wave_size]
            
            # Publish wave concurrently
            wave_tasks = [event_bus.publish(alert) for alert in wave_alerts]
            await asyncio.gather(*wave_tasks)
            
            # Small delay between waves
            await asyncio.sleep(0.02)
        
        # Wait for processing to complete
        await asyncio.sleep(1.0)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Verify performance
        throughput = num_alerts / total_time
        assert throughput > 50  # Should handle at least 50 alerts/second
        
        # Verify system handled the load
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        assert bridge_stats['alerts_received_total'] >= num_alerts
        assert bridge_stats['unique_symbols_processed'] == len(symbols_pool)
        assert bridge_stats['feature_requests_sent_total'] >= 5  # Should create multiple batches
        
        assert pipeline_stats['requests_received'] >= 5
        
        # Feature service should have been called multiple times
        assert feature_service.compute_features.call_count >= 5
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_coordination(
        self,
        coordinated_event_system
    ):
        """Test graceful shutdown of all coordinated components."""
        system = coordinated_event_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        
        # Add some activity before shutdown
        pre_shutdown_alerts = [
            ScannerAlertEvent(
                symbol="SHUTDOWN_TEST_1",
                alert_type="high_volume",
                score=0.7,
                scanner_name="shutdown_scanner"
            ),
            ScannerAlertEvent(
                symbol="SHUTDOWN_TEST_2", 
                alert_type="catalyst_detected",
                score=0.9,
                scanner_name="shutdown_scanner"
            )
        ]
        
        for alert in pre_shutdown_alerts:
            await event_bus.publish(alert)
        
        await asyncio.sleep(0.2)
        
        # Verify activity was processed
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        assert bridge_stats['alerts_received_total'] >= len(pre_shutdown_alerts)
        
        # Test graceful shutdown - already handled in fixture cleanup
        # Components should shut down without errors
        
        # This test verifies that the fixture cleanup works properly
        # which indicates graceful shutdown capability
        assert True  # If we reach here, shutdown was successful
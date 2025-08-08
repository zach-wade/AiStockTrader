"""
Integration tests for Feature Pipeline Helpers.

Tests the coordinated operation of FeatureComputationWorker, FeatureGroupMapper,
and RequestQueueManager with their refactored dependencies.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any
import yaml
from pathlib import Path

from main.interfaces.events import IEventBus, Event, EventType
from main.events.core import EventBusFactory
from main.events.feature_pipeline_helpers.feature_computation_worker import FeatureComputationWorker
from main.events.feature_pipeline_helpers.feature_group_mapper import FeatureGroupMapper
from main.events.feature_pipeline_helpers.request_queue_manager import RequestQueueManager
from main.events.feature_pipeline_helpers.feature_handler_stats_tracker import FeatureHandlerStatsTracker
from main.events.types import (
    FeatureRequestEvent, FeatureComputedEvent
)
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.fixture
async def feature_pipeline_components():
    """Create feature pipeline components with mocked dependencies."""
    # Mock configuration
    mock_config = {
        'feature_pipeline': {
            'batch_size': 10,
            'queue_timeout': 30,
            'max_retries': 3,
            'worker_pool_size': 2
        },
        'features': {
            'default_priority': 5,
            'cache_timeout': 300,
            'computation_timeout': 60
        }
    }
    
    # Mock feature computation service
    mock_feature_service = AsyncMock()
    mock_feature_service.compute_features = AsyncMock(return_value={
        'AAPL': {
            'price_features': {'sma_20': 150.0, 'rsi': 65.0},
            'volume_features': {'volume_sma': 50000, 'volume_ratio': 1.2}
        }
    })
    
    # Create components
    stats_tracker = FeatureHandlerStatsTracker()
    mapper = FeatureGroupMapper()
    queue_manager = RequestQueueManager(max_queue_size=1000)
    worker = FeatureComputationWorker(
        feature_service=mock_feature_service,
        config=mock_config
    )
    
    await queue_manager.start()
    
    yield {
        'stats_tracker': stats_tracker,
        'mapper': mapper,
        'queue_manager': queue_manager,
        'worker': worker,
        'feature_service': mock_feature_service,
        'config': mock_config
    }
    
    await queue_manager.stop()


@pytest.fixture
def sample_feature_requests():
    """Create sample feature request events for testing."""
    return [
        FeatureRequestEvent(
            symbols=['AAPL'],
            features=['price_features'],
            requester='test_scanner_1',
            priority=5
        ),
        FeatureRequestEvent(
            symbols=['MSFT', 'GOOGL'],
            features=['volume_features', 'trend_features'],
            requester='test_scanner_2',
            priority=7
        ),
        FeatureRequestEvent(
            symbols=['TSLA'],
            features=['all_features'],
            requester='urgent_scanner',
            priority=9
        )
    ]


@pytest.fixture
async def feature_event_bus():
    """Create event bus configured for feature pipeline testing."""
    event_bus = EventBusFactory.create_test_instance()
    await event_bus.start()
    
    yield event_bus
    
    await event_bus.stop()


class TestFeaturePipelineHelpersIntegration:
    """Test integrated operation of all feature pipeline helpers."""
    
    @pytest.mark.asyncio
    async def test_complete_feature_request_lifecycle(
        self, 
        feature_pipeline_components, 
        sample_feature_requests,
        feature_event_bus
    ):
        """Test complete feature request from queue to computation."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        stats = components['stats_tracker']
        
        # Subscribe worker to feature requests
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Track computed events
        computed_events = []
        async def track_computed(event):
            computed_events.append(event)
        
        feature_event_bus.subscribe(EventType.FEATURE_COMPUTED, track_computed)
        
        # Process feature requests through queue
        for request in sample_feature_requests:
            await queue_manager.enqueue_request(request.symbols, request.features, request.priority)
            stats.increment_requests_received()
        
        # Wait for queue processing
        await asyncio.sleep(0.2)
        
        # Get queued requests and publish as events
        pending_requests = await queue_manager.get_pending_requests()
        for request_data in pending_requests:
            event = FeatureRequestEvent(
                symbols=request_data['symbols'],
                features=request_data['features'],
                requester=request_data.get('requester', 'queue_manager'),
                priority=request_data.get('priority', 5)
            )
            await feature_event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # Verify feature computation occurred
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count > 0
        
        # Verify computed events were published
        assert len(computed_events) > 0
        
        # Verify stats tracking
        handler_stats = stats.get_stats()
        assert handler_stats['requests_received'] >= len(sample_feature_requests)
    
    @pytest.mark.asyncio
    async def test_feature_group_mapping_integration(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test feature group mapping with actual computation worker."""
        components = feature_pipeline_components
        mapper = components['mapper']
        worker = components['worker']
        
        # Subscribe worker to events
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Test different feature group mappings
        test_cases = [
            {
                'input_features': ['price_features'],
                'expected_groups': ['price_features', 'trend_features']
            },
            {
                'input_features': ['volume_features'],
                'expected_groups': ['volume_features']
            },
            {
                'input_features': ['all_features'],
                'expected_groups': 'all'  # Should expand to all available features
            }
        ]
        
        for case in test_cases:
            # Create feature request
            request = FeatureRequestEvent(
                symbols=['TEST_SYMBOL'],
                features=case['input_features'],
                requester='mapper_test'
            )
            
            # Map features using mapper
            mapped_features = mapper.map_features_to_groups(case['input_features'])
            
            if case['expected_groups'] == 'all':
                # Should contain many feature groups
                assert len(mapped_features) > 5
            else:
                assert set(mapped_features) == set(case['expected_groups'])
            
            # Process through worker
            await feature_event_bus.publish(request)
        
        await asyncio.sleep(0.2)
        
        # Verify all requests were processed
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count >= len(test_cases)
    
    @pytest.mark.asyncio
    async def test_queue_manager_priority_handling(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test that queue manager properly handles priority ordering."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Add requests with different priorities
        priority_requests = [
            {'symbols': ['LOW1'], 'features': ['price_features'], 'priority': 2},
            {'symbols': ['HIGH1'], 'features': ['price_features'], 'priority': 9},
            {'symbols': ['MED1'], 'features': ['price_features'], 'priority': 5},
            {'symbols': ['HIGH2'], 'features': ['volume_features'], 'priority': 8},
            {'symbols': ['LOW2'], 'features': ['trend_features'], 'priority': 1},
        ]
        
        # Enqueue in mixed order
        for req in priority_requests:
            await queue_manager.enqueue_request(
                req['symbols'], 
                req['features'], 
                req['priority']
            )
        
        await asyncio.sleep(0.1)
        
        # Get requests - should be in priority order
        pending = await queue_manager.get_pending_requests(limit=10)
        
        # Verify priority ordering (higher priority first)
        priorities = [req.get('priority', 5) for req in pending]
        assert priorities == sorted(priorities, reverse=True)
        
        # Process high priority requests
        for request_data in pending[:2]:  # Process top 2 highest priority
            event = FeatureRequestEvent(
                symbols=request_data['symbols'],
                features=request_data['features'],
                priority=request_data.get('priority', 5)
            )
            await feature_event_bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Verify processing occurred
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_deduplication_across_components(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test deduplication functionality across pipeline components."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        
        # Track actual processing
        processed_requests = []
        original_process = worker.process_feature_request
        
        async def track_processing(event):
            processed_requests.append(event)
            return await original_process(event)
        
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, track_processing)
        
        # Add duplicate requests
        duplicate_requests = [
            {'symbols': ['AAPL'], 'features': ['price_features'], 'priority': 5},
            {'symbols': ['AAPL'], 'features': ['price_features'], 'priority': 5},  # Duplicate
            {'symbols': ['MSFT'], 'features': ['volume_features'], 'priority': 7},
            {'symbols': ['AAPL'], 'features': ['price_features'], 'priority': 6},  # Similar but different priority
            {'symbols': ['MSFT'], 'features': ['volume_features'], 'priority': 7},  # Exact duplicate
        ]
        
        # Enqueue all requests
        for req in duplicate_requests:
            await queue_manager.enqueue_request(
                req['symbols'], 
                req['features'], 
                req['priority']
            )
        
        await asyncio.sleep(0.1)
        
        # Get pending requests - should be deduplicated
        pending = await queue_manager.get_pending_requests()
        
        # Should have fewer than original due to deduplication
        assert len(pending) < len(duplicate_requests)
        
        # Process requests
        for request_data in pending:
            event = FeatureRequestEvent(
                symbols=request_data['symbols'],
                features=request_data['features'],
                priority=request_data.get('priority', 5)
            )
            await feature_event_bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Verify deduplication worked
        assert len(processed_requests) == len(pending)
    
    @pytest.mark.asyncio
    async def test_error_handling_across_pipeline(
        self,
        feature_pipeline_components,
        feature_event_bus,
        caplog
    ):
        """Test error handling throughout the feature pipeline."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        stats = components['stats_tracker']
        
        # Make feature service fail for specific symbols
        mock_service = components['feature_service']
        
        async def failing_compute_features(symbols, features, **kwargs):
            if 'FAIL' in symbols:
                raise Exception("Feature computation failed")
            return {
                symbol: {feature: {'test_value': 1.0} for feature in features}
                for symbol in symbols if 'FAIL' not in symbol
            }
        
        mock_service.compute_features.side_effect = failing_compute_features
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Mix successful and failing requests
        test_requests = [
            {'symbols': ['AAPL'], 'features': ['price_features'], 'priority': 5},
            {'symbols': ['FAIL_SYMBOL'], 'features': ['price_features'], 'priority': 5},
            {'symbols': ['MSFT'], 'features': ['volume_features'], 'priority': 7},
        ]
        
        # Process requests
        for req in test_requests:
            await queue_manager.enqueue_request(
                req['symbols'], 
                req['features'], 
                req['priority']
            )
            stats.increment_requests_received()
        
        await asyncio.sleep(0.1)
        
        # Get and process requests
        pending = await queue_manager.get_pending_requests()
        for request_data in pending:
            event = FeatureRequestEvent(
                symbols=request_data['symbols'],
                features=request_data['features'],
                priority=request_data.get('priority', 5)
            )
            await feature_event_bus.publish(event)
        
        await asyncio.sleep(0.3)
        
        # Verify error handling
        assert "Feature computation failed" in caplog.text or "Error processing feature request" in caplog.text
        
        # Verify stats include both successful and failed requests
        handler_stats = stats.get_stats()
        assert handler_stats['requests_received'] == len(test_requests)
        
        # Should have attempted processing all requests
        assert mock_service.compute_features.call_count >= len(test_requests)
    
    @pytest.mark.asyncio
    async def test_high_volume_feature_processing(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test pipeline performance under high volume."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        stats = components['stats_tracker']
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Generate high volume of requests
        num_requests = 100
        symbols_pool = [f'STOCK_{i}' for i in range(20)]  # 20 unique symbols
        features_pool = ['price_features', 'volume_features', 'trend_features']
        
        # Enqueue many requests
        for i in range(num_requests):
            symbol = symbols_pool[i % len(symbols_pool)]
            features = [features_pool[i % len(features_pool)]]
            priority = (i % 10) + 1  # Priorities 1-10
            
            await queue_manager.enqueue_request([symbol], features, priority)
            stats.increment_requests_received()
        
        await asyncio.sleep(0.2)
        
        # Process in batches to simulate real usage
        batch_size = 10
        total_processed = 0
        
        while total_processed < num_requests:
            pending = await queue_manager.get_pending_requests(limit=batch_size)
            if not pending:
                break
                
            for request_data in pending:
                event = FeatureRequestEvent(
                    symbols=request_data['symbols'],
                    features=request_data['features'],
                    priority=request_data.get('priority', 5)
                )
                await feature_event_bus.publish(event)
                total_processed += 1
            
            await asyncio.sleep(0.05)  # Small pause between batches
        
        await asyncio.sleep(0.5)  # Final processing time
        
        # Verify high volume processing
        handler_stats = stats.get_stats()
        assert handler_stats['requests_received'] == num_requests
        
        # Should have made many feature computation calls
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count >= 50
        
        # Queue should be processing efficiently
        remaining = await queue_manager.get_queue_size()
        assert remaining < num_requests * 0.2  # Should have processed most requests
    
    @pytest.mark.asyncio
    async def test_configuration_integration(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test that configuration is properly applied across components."""
        components = feature_pipeline_components
        config = components['config']
        queue_manager = components['queue_manager']
        worker = components['worker']
        
        # Verify configuration is applied
        assert queue_manager._max_queue_size == 1000  # From fixture
        
        # Test batch size configuration
        original_batch_size = config['feature_pipeline']['batch_size']
        assert original_batch_size == 10
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Add requests up to batch size
        for i in range(original_batch_size):
            await queue_manager.enqueue_request(
                [f'BATCH_TEST_{i}'], 
                ['price_features'], 
                5
            )
        
        await asyncio.sleep(0.1)
        
        # Should be able to get all requests
        pending = await queue_manager.get_pending_requests(limit=original_batch_size)
        assert len(pending) == original_batch_size
        
        # Process requests to verify worker configuration
        for request_data in pending:
            event = FeatureRequestEvent(
                symbols=request_data['symbols'],
                features=request_data['features'],
                priority=request_data.get('priority', 5)
            )
            await feature_event_bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Verify processing completed
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count >= original_batch_size
    
    @pytest.mark.asyncio
    async def test_memory_management_under_load(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test memory management of components under sustained load."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Generate sustained load in waves
        for wave in range(5):
            # Add batch of requests
            for i in range(50):
                symbol = f'MEMORY_TEST_{wave}_{i}'
                await queue_manager.enqueue_request([symbol], ['price_features'], 5)
            
            # Process batch
            pending = await queue_manager.get_pending_requests(limit=50)
            for request_data in pending:
                event = FeatureRequestEvent(
                    symbols=request_data['symbols'],
                    features=request_data['features'],
                    priority=request_data.get('priority', 5)
                )
                await feature_event_bus.publish(event)
            
            await asyncio.sleep(0.1)
        
        # Final processing time
        await asyncio.sleep(0.5)
        
        # Verify queue is not accumulating requests
        final_queue_size = await queue_manager.get_queue_size()
        assert final_queue_size < 100  # Should not be accumulating
        
        # Verify processing occurred
        mock_service = components['feature_service']
        assert mock_service.compute_features.call_count >= 200
    
    @pytest.mark.asyncio
    async def test_pipeline_shutdown_cleanup(
        self,
        feature_pipeline_components,
        feature_event_bus
    ):
        """Test proper cleanup during pipeline shutdown."""
        components = feature_pipeline_components
        queue_manager = components['queue_manager']
        worker = components['worker']
        stats = components['stats_tracker']
        
        # Subscribe worker
        feature_event_bus.subscribe(EventType.FEATURE_REQUEST, worker.process_feature_request)
        
        # Add some requests
        for i in range(10):
            await queue_manager.enqueue_request([f'SHUTDOWN_TEST_{i}'], ['price_features'], 5)
            stats.increment_requests_received()
        
        await asyncio.sleep(0.1)
        
        # Verify there are pending requests
        pending_count = await queue_manager.get_queue_size()
        assert pending_count > 0
        
        # Verify stats before shutdown
        pre_shutdown_stats = stats.get_stats()
        assert pre_shutdown_stats['requests_received'] == 10
        
        # Shutdown queue manager
        await queue_manager.stop()
        
        # Verify graceful shutdown (no exceptions)
        final_stats = stats.get_stats()
        assert final_stats['requests_received'] == 10  # Should persist stats
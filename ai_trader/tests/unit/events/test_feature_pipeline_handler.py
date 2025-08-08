"""Unit tests for feature_pipeline_handler module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from main.events.handlers.feature_pipeline_handler import FeaturePipelineHandler
from main.events.types import Event, EventType
from main.feature_pipeline.orchestrator import FeatureOrchestrator

from tests.fixtures.events.mock_events import (
    create_feature_request_event,
    create_feature_computed_event,
    create_error_event
)
from tests.fixtures.events.mock_configs import create_feature_pipeline_config


class TestFeaturePipelineHandler:
    """Test FeaturePipelineHandler class."""
    
    @pytest.fixture
    def mock_feature_orchestrator(self):
        """Create mock feature orchestrator."""
        orchestrator = Mock(spec=FeatureOrchestrator)
        orchestrator.compute_features = AsyncMock()
        return orchestrator
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock()
        bus.subscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return create_feature_pipeline_config()
    
    @pytest.fixture
    async def handler(self, mock_feature_orchestrator, mock_event_bus, config):
        """Create FeaturePipelineHandler instance for testing."""
        with patch('main.events.handlers.feature_pipeline_handler.get_event_bus') as mock_get_bus:
            mock_get_bus.return_value = mock_event_bus
            
            handler = FeaturePipelineHandler(
                feature_orchestrator=mock_feature_orchestrator,
                config=config
            )
            return handler
    
    def test_initialization_with_orchestrator(self, mock_feature_orchestrator, mock_event_bus, config):
        """Test initialization with provided orchestrator."""
        with patch('main.events.handlers.feature_pipeline_handler.get_event_bus') as mock_get_bus:
            mock_get_bus.return_value = mock_event_bus
            
            handler = FeaturePipelineHandler(
                feature_orchestrator=mock_feature_orchestrator,
                config=config
            )
            
            assert handler.feature_orchestrator == mock_feature_orchestrator
            assert handler.event_bus == mock_event_bus
            assert handler.config == config
            assert handler._num_workers == 3  # From config
    
    def test_initialization_without_orchestrator(self, mock_event_bus, config):
        """Test initialization creates orchestrator if not provided."""
        with patch('main.events.handlers.feature_pipeline_handler.get_event_bus') as mock_get_bus:
            with patch('main.events.handlers.feature_pipeline_handler.FeatureOrchestrator') as mock_orch_class:
                mock_get_bus.return_value = mock_event_bus
                mock_orchestrator = Mock()
                mock_orch_class.return_value = mock_orchestrator
                
                handler = FeaturePipelineHandler(config=config)
                
                # Should create orchestrator with config
                mock_orch_class.assert_called_once_with(config)
                assert handler.feature_orchestrator == mock_orchestrator
    
    def test_initialization_without_config(self, mock_feature_orchestrator, mock_event_bus):
        """Test initialization gets config if not provided."""
        with patch('main.events.handlers.feature_pipeline_handler.get_event_bus') as mock_get_bus:
            with patch('main.events.handlers.feature_pipeline_handler.get_config') as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_config = create_feature_pipeline_config()
                mock_get_config.return_value = mock_config
                
                handler = FeaturePipelineHandler(
                    feature_orchestrator=mock_feature_orchestrator
                )
                
                # Should get config
                mock_get_config.assert_called_once()
                assert handler.config == mock_config
    
    @pytest.mark.asyncio
    async def test_start(self, handler, mock_event_bus):
        """Test starting the handler."""
        # Start handler
        await handler.start()
        
        # Should subscribe to feature requests
        mock_event_bus.subscribe.assert_called_once()
        call_args = mock_event_bus.subscribe.call_args[0]
        assert call_args[0] == EventType.FEATURE_REQUEST
        
        # Should create worker tasks
        assert len(handler._workers) == handler._num_workers
        for worker in handler._workers:
            assert isinstance(worker, asyncio.Task)
    
    @pytest.mark.asyncio
    async def test_stop(self, handler):
        """Test stopping the handler."""
        # Start first
        await handler.start()
        assert len(handler._workers) > 0
        
        # Stop handler
        await handler.stop()
        
        # Workers should be cancelled
        assert len(handler._workers) == 0
    
    @pytest.mark.asyncio
    async def test_handle_feature_request_event(self, handler):
        """Test handling feature request events."""
        # Create request event
        event = create_feature_request_event(
            symbols=["AAPL", "GOOGL"],
            features=["price_features", "volume_features"]
        )
        
        # Handle event
        await handler._handle_feature_request_event(event)
        
        # Should add to queue
        assert handler._request_queue_manager.get_queue_size() > 0
        
        # Check stats
        stats = handler._stats_tracker.get_stats()
        assert stats["requests_received"] >= 1
    
    @pytest.mark.asyncio
    async def test_worker_loop(self, handler, mock_feature_orchestrator):
        """Test worker loop processing."""
        # Create mock worker
        mock_worker = AsyncMock()
        mock_worker.process_request = AsyncMock()
        
        # Add request to queue
        request = {
            'symbols': ['AAPL'],
            'features': ['price_features'],
            'event': create_feature_request_event()
        }
        
        # Mock queue manager to return request
        handler._request_queue_manager.dequeue_request = AsyncMock(
            side_effect=[Mock(request=request), None]  # Return request then None
        )
        handler._request_queue_manager.complete_request = AsyncMock()
        
        # Run worker loop briefly
        worker_task = asyncio.create_task(handler._run_worker_loop(mock_worker))
        await asyncio.sleep(0.1)
        
        # Cancel task
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        # Worker should have processed request
        mock_worker.process_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_error_handling(self, handler):
        """Test worker handles errors gracefully."""
        # Create failing worker
        mock_worker = AsyncMock()
        mock_worker.process_request = AsyncMock(side_effect=Exception("Processing failed"))
        
        # Add request to queue
        request = {
            'symbols': ['AAPL'],
            'features': ['price_features'],
            'event': create_feature_request_event()
        }
        
        # Mock queue manager
        handler._request_queue_manager.dequeue_request = AsyncMock(
            side_effect=[Mock(request=request, request_id="test_123"), None]
        )
        handler._request_queue_manager.complete_request = AsyncMock()
        
        # Run worker loop
        worker_task = asyncio.create_task(handler._run_worker_loop(mock_worker))
        await asyncio.sleep(0.1)
        
        # Cancel task
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        # Should mark request as failed
        handler._request_queue_manager.complete_request.assert_called_with(
            "test_123", success=False
        )
    
    @pytest.mark.asyncio
    async def test_queue_management(self, handler):
        """Test integration with request queue manager."""
        # Create multiple events
        events = []
        for i in range(5):
            event = create_feature_request_event(
                symbols=[f"STOCK{i}"],
                features=["price_features"]
            )
            events.append(event)
        
        # Handle all events
        for event in events:
            await handler._handle_feature_request_event(event)
        
        # Check queue size
        queue_size = handler._request_queue_manager.get_queue_size()
        assert queue_size == 5
        
        # Check stats
        stats = handler._request_queue_manager.get_stats()
        assert stats.total_queued == 5
    
    @pytest.mark.asyncio
    async def test_feature_group_mapping(self, handler):
        """Test feature group mapping integration."""
        # Mock mapper response
        handler._feature_group_mapper.map_alert_to_features = Mock(
            return_value=Mock(
                feature_groups=["PRICE", "VOLUME"],
                priority=7
            )
        )
        
        # Create event with alert data
        event = Event(
            event_type=EventType.FEATURE_REQUEST,
            source="scanner",
            data={
                "symbols": ["AAPL"],
                "features": ["mapped_features"],
                "alert_data": {"alert_type": "high_volume"}
            }
        )
        
        # Handle event
        await handler._handle_feature_request_event(event)
        
        # Mapper should have been called if alert data present
        # Implementation depends on handler logic
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, handler):
        """Test statistics tracking."""
        # Process some events
        for i in range(3):
            event = create_feature_request_event()
            await handler._handle_feature_request_event(event)
        
        # Get stats
        stats = handler._stats_tracker.get_stats()
        
        assert "requests_received" in stats
        assert stats["requests_received"] >= 3
    
    @pytest.mark.asyncio
    async def test_concurrent_workers(self, handler):
        """Test multiple workers processing concurrently."""
        processed_count = 0
        processing_lock = asyncio.Lock()
        
        # Create mock worker that tracks processing
        async def mock_process(request):
            nonlocal processed_count
            async with processing_lock:
                processed_count += 1
            await asyncio.sleep(0.05)  # Simulate work
        
        # Create workers with tracking
        for i in range(handler._num_workers):
            worker = AsyncMock()
            worker.process_request = mock_process
            handler._workers[i] = worker
        
        # Add multiple requests
        for i in range(10):
            request = {
                'symbols': [f'STOCK{i}'],
                'features': ['price_features'],
                'event': create_feature_request_event()
            }
            await handler._request_queue_manager.enqueue_request(
                request, f"request_{i}"
            )
        
        # Let workers process
        await asyncio.sleep(0.2)
        
        # Should have processed requests concurrently
        assert processed_count > 0
    
    def test_get_stats(self, handler):
        """Test getting handler statistics."""
        # Mock stats from components
        handler._stats_tracker.get_stats = Mock(return_value={
            "requests_received": 100,
            "requests_processed": 95
        })
        
        handler._request_queue_manager.get_stats = Mock(return_value=Mock(
            total_queued=100,
            total_processed=95,
            total_failed=5,
            current_queue_size=0
        ))
        
        # Get combined stats
        stats = handler.get_stats()
        
        assert "handler_stats" in stats
        assert "queue_stats" in stats
        assert stats["handler_stats"]["requests_received"] == 100
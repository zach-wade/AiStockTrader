"""Unit tests for event_history_manager module."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from collections import deque
import threading
import time

from main.events.core.event_bus_helpers.event_history_manager import EventHistoryManager
from main.events.types import Event, EventType

from tests.fixtures.events.mock_events import (
    create_scan_alert,
    create_order_event,
    create_feature_request_event,
    create_error_event
)


class TestEventHistoryManager:
    """Test EventHistoryManager class."""
    
    @pytest.fixture
    def history_manager(self):
        """Create EventHistoryManager instance for testing."""
        return EventHistoryManager(max_history=100)
    
    def test_initialization_default(self):
        """Test initialization with default values."""
        manager = EventHistoryManager()
        
        assert isinstance(manager._history, deque)
        assert manager._history.maxlen == 1000  # Default
        assert len(manager._history) == 0
    
    def test_initialization_custom_max_history(self):
        """Test initialization with custom max history."""
        manager = EventHistoryManager(max_history=50)
        
        assert manager._history.maxlen == 50
        assert len(manager._history) == 0
    
    def test_initialization_zero_max_history(self):
        """Test initialization with zero max history."""
        manager = EventHistoryManager(max_history=0)
        
        # Should use a reasonable minimum
        assert manager._history.maxlen == 0 or manager._history.maxlen == 1
    
    def test_add_event(self, history_manager):
        """Test adding events to history."""
        event1 = create_scan_alert(symbol="AAPL")
        event2 = create_order_event(symbol="GOOGL")
        
        history_manager.add_event(event1)
        history_manager.add_event(event2)
        
        assert len(history_manager._history) == 2
        assert history_manager._history[0] == event1
        assert history_manager._history[1] == event2
    
    def test_add_event_automatic_removal(self):
        """Test automatic removal of oldest events when maxlen reached."""
        manager = EventHistoryManager(max_history=3)
        
        # Add 5 events to a history with maxlen=3
        events = []
        for i in range(5):
            event = create_scan_alert(symbol=f"STOCK{i}")
            events.append(event)
            manager.add_event(event)
        
        # Should only have last 3 events
        assert len(manager._history) == 3
        
        # Should have events 2, 3, 4 (0 and 1 were removed)
        history = list(manager._history)
        assert history[0].data["symbol"] == "STOCK2"
        assert history[1].data["symbol"] == "STOCK3"
        assert history[2].data["symbol"] == "STOCK4"
    
    def test_get_history_all(self, history_manager):
        """Test getting all history."""
        # Add some events
        events = []
        for i in range(5):
            event = create_scan_alert(symbol=f"STOCK{i}")
            events.append(event)
            history_manager.add_event(event)
        
        # Get all history
        history = history_manager.get_history()
        
        assert len(history) == 5
        assert history == events
    
    def test_get_history_with_limit(self, history_manager):
        """Test getting history with limit."""
        # Add 10 events
        for i in range(10):
            event = create_scan_alert(symbol=f"STOCK{i}")
            history_manager.add_event(event)
        
        # Get last 5 events
        history = history_manager.get_history(limit=5)
        
        assert len(history) == 5
        # Should be most recent 5 events
        assert history[0].data["symbol"] == "STOCK5"
        assert history[4].data["symbol"] == "STOCK9"
    
    def test_get_history_limit_exceeds_size(self, history_manager):
        """Test getting history when limit exceeds actual size."""
        # Add 3 events
        for i in range(3):
            event = create_scan_alert(symbol=f"STOCK{i}")
            history_manager.add_event(event)
        
        # Request 10 events
        history = history_manager.get_history(limit=10)
        
        # Should return all 3 events
        assert len(history) == 3
    
    def test_get_history_by_event_type(self, history_manager):
        """Test filtering history by event type."""
        # Add different event types
        history_manager.add_event(create_scan_alert(symbol="AAPL"))
        history_manager.add_event(create_order_event(symbol="AAPL"))
        history_manager.add_event(create_scan_alert(symbol="GOOGL"))
        history_manager.add_event(create_feature_request_event())
        history_manager.add_event(create_order_event(symbol="MSFT"))
        
        # Get only scanner alerts
        alerts = history_manager.get_history(event_type=EventType.SCANNER_ALERT)
        
        assert len(alerts) == 2
        assert all(event.event_type == EventType.SCANNER_ALERT for event in alerts)
        
        # Get only order events
        orders = history_manager.get_history(event_type=EventType.ORDER_PLACED)
        
        assert len(orders) == 2
        assert all(event.event_type == EventType.ORDER_PLACED for event in orders)
    
    def test_get_history_by_time_range(self, history_manager):
        """Test filtering history by time range."""
        now = datetime.now(timezone.utc)
        
        # Add events with different timestamps
        old_event = create_scan_alert(symbol="OLD")
        old_event.timestamp = now - timedelta(hours=2)
        
        recent_event1 = create_scan_alert(symbol="RECENT1")
        recent_event1.timestamp = now - timedelta(minutes=30)
        
        recent_event2 = create_scan_alert(symbol="RECENT2")
        recent_event2.timestamp = now - timedelta(minutes=15)
        
        history_manager.add_event(old_event)
        history_manager.add_event(recent_event1)
        history_manager.add_event(recent_event2)
        
        # Get events from last hour
        one_hour_ago = now - timedelta(hours=1)
        recent_history = history_manager.get_history(since=one_hour_ago)
        
        assert len(recent_history) == 2
        assert recent_history[0].data["symbol"] == "RECENT1"
        assert recent_history[1].data["symbol"] == "RECENT2"
    
    def test_get_history_combined_filters(self, history_manager):
        """Test combining multiple filters."""
        now = datetime.now(timezone.utc)
        
        # Add various events
        for i in range(5):
            alert = create_scan_alert(symbol=f"ALERT{i}")
            alert.timestamp = now - timedelta(minutes=i*10)
            history_manager.add_event(alert)
            
            order = create_order_event(symbol=f"ORDER{i}")
            order.timestamp = now - timedelta(minutes=i*10)
            history_manager.add_event(order)
        
        # Get scanner alerts from last 30 minutes, limit 2
        filtered = history_manager.get_history(
            event_type=EventType.SCANNER_ALERT,
            since=now - timedelta(minutes=30),
            limit=2
        )
        
        assert len(filtered) == 2
        assert all(event.event_type == EventType.SCANNER_ALERT for event in filtered)
        assert all(event.timestamp >= now - timedelta(minutes=30) for event in filtered)
    
    def test_clear_history(self, history_manager):
        """Test clearing history."""
        # Add some events
        for i in range(5):
            history_manager.add_event(create_scan_alert(symbol=f"STOCK{i}"))
        
        assert len(history_manager._history) == 5
        
        # Clear history
        history_manager.clear_history()
        
        assert len(history_manager._history) == 0
        assert history_manager.get_history() == []
    
    def test_get_history_empty(self, history_manager):
        """Test getting history when empty."""
        history = history_manager.get_history()
        
        assert history == []
        assert len(history) == 0
    
    def test_deque_efficiency(self):
        """Test deque efficiency with large number of events."""
        manager = EventHistoryManager(max_history=1000)
        
        # Add many events
        start_time = time.time()
        for i in range(2000):
            event = create_scan_alert(symbol=f"STOCK{i}")
            manager.add_event(event)
        add_time = time.time() - start_time
        
        # Should be fast due to deque O(1) append
        assert add_time < 1.0  # Should take less than 1 second
        
        # Should only have last 1000 events
        assert len(manager._history) == 1000
        
        # Getting history should also be fast
        start_time = time.time()
        history = manager.get_history()
        get_time = time.time() - start_time
        
        assert get_time < 0.1  # Should be very fast
        assert len(history) == 1000
    
    def test_thread_safety(self, history_manager):
        """Test thread-safe operations."""
        events_to_add = 100
        threads_count = 5
        
        def add_events(thread_id):
            for i in range(events_to_add):
                event = create_scan_alert(symbol=f"T{thread_id}_E{i}")
                history_manager.add_event(event)
        
        # Create and start threads
        threads = []
        for i in range(threads_count):
            t = threading.Thread(target=add_events, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Should have all events (up to max_history)
        expected_count = min(
            events_to_add * threads_count,
            history_manager._history.maxlen
        )
        assert len(history_manager._history) == expected_count
    
    def test_event_ordering(self, history_manager):
        """Test that events maintain insertion order."""
        # Add events with specific order
        events = []
        for i in range(10):
            event = create_scan_alert(symbol=f"STOCK{i}")
            events.append(event)
            history_manager.add_event(event)
        
        # Get history
        history = history_manager.get_history()
        
        # Should maintain order
        for i, event in enumerate(history):
            assert event.data["symbol"] == f"STOCK{i}"
    
    def test_get_history_slice(self, history_manager):
        """Test getting a slice of history."""
        # Add 10 events
        for i in range(10):
            event = create_scan_alert(symbol=f"STOCK{i}")
            history_manager.add_event(event)
        
        # If manager supports slicing
        if hasattr(history_manager, 'get_history_slice'):
            # Get middle slice
            slice_history = history_manager.get_history_slice(start=3, end=7)
            
            assert len(slice_history) == 4
            assert slice_history[0].data["symbol"] == "STOCK3"
            assert slice_history[3].data["symbol"] == "STOCK6"
    
    def test_memory_usage(self):
        """Test memory usage with maxlen constraint."""
        import sys
        
        # Create manager with small limit
        manager = EventHistoryManager(max_history=100)
        
        # Add many more events than limit
        for i in range(1000):
            event = create_scan_alert(symbol=f"STOCK{i}")
            manager.add_event(event)
        
        # Should only have 100 events in memory
        assert len(manager._history) == 100
        
        # Memory usage should be bounded
        # (This is more of a sanity check)
        if hasattr(sys, 'getsizeof'):
            size = sys.getsizeof(manager._history)
            # Should be reasonable for 100 events
            assert size < 1_000_000  # Less than 1MB
"""Unit tests for feature_request_batcher module."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from main.events.handlers.scanner_bridge_helpers.feature_request_batcher import (
    FeatureRequestBatcher, FeatureRequestBatch
)


class TestFeatureRequestBatch:
    """Test FeatureRequestBatch dataclass."""
    
    def test_batch_creation_defaults(self):
        """Test creating batch with default values."""
        batch = FeatureRequestBatch()
        
        assert isinstance(batch.symbols, set)
        assert isinstance(batch.features, set)
        assert batch.priority == 5
        assert isinstance(batch.created_at, datetime)
        assert batch.created_at.tzinfo == timezone.utc
        assert isinstance(batch.scanner_sources, set)
        assert isinstance(batch.correlation_ids, set)
        assert len(batch.symbols) == 0
        assert len(batch.features) == 0
    
    def test_batch_creation_with_values(self):
        """Test creating batch with specific values."""
        symbols = {"AAPL", "GOOGL"}
        features = {"price", "volume"}
        priority = 8
        sources = {"scanner1"}
        correlation_ids = {"corr_123"}
        
        batch = FeatureRequestBatch(
            symbols=symbols,
            features=features,
            priority=priority,
            scanner_sources=sources,
            correlation_ids=correlation_ids
        )
        
        assert batch.symbols == symbols
        assert batch.features == features
        assert batch.priority == priority
        assert batch.scanner_sources == sources
        assert batch.correlation_ids == correlation_ids
    
    def test_batch_timestamp(self):
        """Test batch creation timestamp."""
        before = datetime.now(timezone.utc)
        batch = FeatureRequestBatch()
        after = datetime.now(timezone.utc)
        
        assert before <= batch.created_at <= after


class TestFeatureRequestBatcher:
    """Test FeatureRequestBatcher class."""
    
    @pytest.fixture
    def batcher(self):
        """Create FeatureRequestBatcher instance."""
        return FeatureRequestBatcher(batch_size=3)
    
    def test_initialization(self):
        """Test batcher initialization."""
        batcher = FeatureRequestBatcher(batch_size=50)
        
        assert batcher._batch_size == 50
        assert isinstance(batcher._pending_requests, dict)
        assert len(batcher._pending_requests) == 0
    
    def test_add_to_pending_first_symbol(self, batcher):
        """Test adding first symbol to pending batch."""
        result = batcher.add_to_pending(
            symbol="AAPL",
            features=["price", "volume"],
            priority=7,
            scanner_source="scanner1",
            correlation_id="corr_123"
        )
        
        assert result is None  # Batch not full yet
        assert len(batcher._pending_requests) == 1
        
        # Check batch was created correctly
        batch_key = "p7_price_volume"
        assert batch_key in batcher._pending_requests
        
        batch = batcher._pending_requests[batch_key]
        assert "AAPL" in batch.symbols
        assert batch.features == {"price", "volume"}
        assert batch.priority == 7
        assert "scanner1" in batch.scanner_sources
        assert "corr_123" in batch.correlation_ids
    
    def test_add_to_pending_same_batch(self, batcher):
        """Test adding multiple symbols to same batch."""
        # Add first symbol
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        
        # Add second symbol with same features and priority
        result = batcher.add_to_pending("GOOGL", ["price"], 5, "scanner2")
        
        assert result is None  # Not full yet (batch size is 3)
        assert len(batcher._pending_requests) == 1
        
        batch = list(batcher._pending_requests.values())[0]
        assert batch.symbols == {"AAPL", "GOOGL"}
        assert batch.scanner_sources == {"scanner1", "scanner2"}
    
    def test_add_to_pending_batch_full(self, batcher):
        """Test returning full batch when size limit reached."""
        # Add symbols up to batch size (3)
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["price"], 5, "scanner1")
        
        # Third symbol should fill the batch
        result = batcher.add_to_pending("MSFT", ["price"], 5, "scanner1")
        
        assert isinstance(result, FeatureRequestBatch)
        assert result.symbols == {"AAPL", "GOOGL", "MSFT"}
        assert len(batcher._pending_requests) == 0  # Batch was removed
    
    def test_add_to_pending_overflow_batch(self, batcher):
        """Test creating overflow batch when current is full."""
        # Fill a batch
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["price"], 5, "scanner1")
        batcher.add_to_pending("MSFT", ["price"], 5, "scanner1")
        
        # Add one more symbol - should create overflow batch
        result = batcher.add_to_pending("AMZN", ["price"], 5, "scanner1")
        
        assert result is None  # New batch not full
        assert len(batcher._pending_requests) == 1  # New overflow batch
        
        # Check new batch contains only the new symbol
        new_batch = list(batcher._pending_requests.values())[0]
        assert new_batch.symbols == {"AMZN"}
    
    def test_add_to_pending_duplicate_symbol(self, batcher):
        """Test adding duplicate symbol to same batch."""
        # Add symbol
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        
        # Add same symbol again
        result = batcher.add_to_pending("AAPL", ["volume"], 5, "scanner2")
        
        assert result is None
        batch = list(batcher._pending_requests.values())[0]
        
        # Symbol count should not increase
        assert len(batch.symbols) == 1
        assert "AAPL" in batch.symbols
        
        # But features and sources should be updated
        assert batch.features == {"price", "volume"}
        assert batch.scanner_sources == {"scanner1", "scanner2"}
    
    def test_add_to_pending_different_priorities(self, batcher):
        """Test that different priorities create different batches."""
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["price"], 8, "scanner1")
        
        assert len(batcher._pending_requests) == 2  # Different batches
        
        # Check batch keys
        assert "p5_price" in batcher._pending_requests
        assert "p8_price" in batcher._pending_requests
    
    def test_add_to_pending_different_features(self, batcher):
        """Test that different features create different batches."""
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["volume"], 5, "scanner1")
        
        assert len(batcher._pending_requests) == 2  # Different batches
        
        # Check batch keys
        assert "p5_price" in batcher._pending_requests
        assert "p5_volume" in batcher._pending_requests
    
    def test_add_to_pending_features_sorted(self, batcher):
        """Test that features are sorted for consistent batch keys."""
        # Add with features in different order
        batcher.add_to_pending("AAPL", ["volume", "price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["price", "volume"], 5, "scanner1")
        
        # Should be in same batch due to sorting
        assert len(batcher._pending_requests) == 1
        batch_key = "p5_price_volume"
        assert batch_key in batcher._pending_requests
    
    def test_add_to_pending_no_correlation_id(self, batcher):
        """Test adding without correlation ID."""
        result = batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        
        assert result is None
        batch = list(batcher._pending_requests.values())[0]
        assert len(batch.correlation_ids) == 0
    
    def test_get_and_clear_all_pending_batches_empty(self, batcher):
        """Test getting batches when none exist."""
        batches = batcher.get_and_clear_all_pending_batches()
        
        assert batches == []
        assert len(batcher._pending_requests) == 0
    
    def test_get_and_clear_all_pending_batches_single(self, batcher):
        """Test getting single pending batch."""
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        
        batches = batcher.get_and_clear_all_pending_batches()
        
        assert len(batches) == 1
        assert batches[0].symbols == {"AAPL"}
        assert len(batcher._pending_requests) == 0  # Cleared
    
    def test_get_and_clear_all_pending_batches_sorted_by_priority(self, batcher):
        """Test that batches are returned sorted by priority (highest first)."""
        # Add batches with different priorities
        batcher.add_to_pending("LOW", ["price"], 3, "scanner1")
        batcher.add_to_pending("HIGH", ["volume"], 9, "scanner1")
        batcher.add_to_pending("MED", ["momentum"], 5, "scanner1")
        
        batches = batcher.get_and_clear_all_pending_batches()
        
        assert len(batches) == 3
        assert batches[0].priority == 9  # Highest
        assert batches[1].priority == 5
        assert batches[2].priority == 3  # Lowest
        assert len(batcher._pending_requests) == 0
    
    def test_get_pending_counts_empty(self, batcher):
        """Test getting counts when no pending batches."""
        batch_count, symbol_count = batcher.get_pending_counts()
        
        assert batch_count == 0
        assert symbol_count == 0
    
    def test_get_pending_counts_with_data(self, batcher):
        """Test getting counts with multiple batches."""
        # Create multiple batches
        batcher.add_to_pending("AAPL", ["price"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["price"], 5, "scanner1")
        batcher.add_to_pending("MSFT", ["volume"], 7, "scanner1")
        batcher.add_to_pending("AMZN", ["volume"], 7, "scanner1")
        
        batch_count, symbol_count = batcher.get_pending_counts()
        
        assert batch_count == 2  # Two different batches
        assert symbol_count == 4  # Four total symbols
    
    def test_batch_key_generation(self, batcher):
        """Test that batch keys are generated consistently."""
        # Same priority and features should generate same key
        batcher.add_to_pending("AAPL", ["price", "volume"], 5, "scanner1")
        batcher.add_to_pending("GOOGL", ["volume", "price"], 5, "scanner2")
        
        # Should be in same batch
        assert len(batcher._pending_requests) == 1
        assert "p5_price_volume" in batcher._pending_requests
    
    def test_overflow_batch_unique_key(self, batcher):
        """Test that overflow batches get unique keys."""
        # Fill first batch
        for i in range(3):
            batcher.add_to_pending(f"STOCK{i}", ["price"], 5, "scanner1")
        
        # Create overflow
        batcher.add_to_pending("OVERFLOW1", ["price"], 5, "scanner1")
        
        # Fill overflow and create another
        for i in range(3, 6):
            batcher.add_to_pending(f"STOCK{i}", ["price"], 5, "scanner1")
        
        batcher.add_to_pending("OVERFLOW2", ["price"], 5, "scanner1")
        
        # Should have 2 batches with unique keys
        assert len(batcher._pending_requests) == 2
        keys = list(batcher._pending_requests.keys())
        assert keys[0] != keys[1]  # Different keys
        assert all("overflow" in key for key in keys)
    
    def test_large_batch_size(self):
        """Test batcher with large batch size."""
        batcher = FeatureRequestBatcher(batch_size=1000)
        
        # Add many symbols
        for i in range(999):
            result = batcher.add_to_pending(f"STOCK{i}", ["price"], 5, "scanner1")
            assert result is None  # Not full yet
        
        # 1000th symbol should complete the batch
        result = batcher.add_to_pending("STOCK999", ["price"], 5, "scanner1")
        assert isinstance(result, FeatureRequestBatch)
        assert len(result.symbols) == 1000
    
    def test_error_handling_inheritance(self, batcher):
        """Test that batcher inherits from ErrorHandlingMixin."""
        # Verify it has error handling methods
        assert hasattr(batcher, '_handle_error')
    
    def test_concurrent_operations(self, batcher):
        """Test thread safety of batcher operations."""
        import threading
        completed_batches = []
        
        def add_symbols(thread_id):
            for i in range(10):
                result = batcher.add_to_pending(
                    f"T{thread_id}_S{i}", 
                    ["price"], 
                    5, 
                    f"scanner{thread_id}"
                )
                if result:
                    completed_batches.append(result)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_symbols, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Get remaining batches
        remaining = batcher.get_and_clear_all_pending_batches()
        
        # Total symbols should be 50 (5 threads * 10 symbols)
        total_symbols = sum(len(b.symbols) for b in completed_batches + remaining)
        assert total_symbols == 50
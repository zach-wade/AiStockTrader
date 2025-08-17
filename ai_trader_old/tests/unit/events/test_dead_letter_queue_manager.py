"""Unit tests for dead_letter_queue_manager module."""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest

# Local imports
from main.events.core.event_bus_helpers.dead_letter_queue_manager import DeadLetterQueueManager
from main.events.types import EventType
from tests.fixtures.events.mock_database import MockConnection, create_mock_db_pool
from tests.fixtures.events.mock_events import create_error_event, create_scan_alert


class TestDeadLetterQueueManager:
    """Test DeadLetterQueueManager class."""

    @pytest.fixture
    def mock_db_pool(self):
        """Create mock database pool."""
        return create_mock_db_pool()

    @pytest.fixture
    async def dlq_manager(self, mock_db_pool):
        """Create DeadLetterQueueManager instance for testing."""
        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.DatabasePool"
        ) as mock_pool_class:
            mock_pool_class.return_value = mock_db_pool

            manager = DeadLetterQueueManager(db_pool=mock_db_pool, max_retries=3, retention_days=7)

            # Initialize the manager
            await manager.initialize()

            yield manager

            # Cleanup
            await manager.close()

    def test_initialization(self, mock_db_pool):
        """Test DLQ manager initialization."""
        manager = DeadLetterQueueManager(db_pool=mock_db_pool, max_retries=5, retention_days=14)

        assert manager.db_pool == mock_db_pool
        assert manager.max_retries == 5
        assert manager.retention_days == 14

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self, mock_db_pool):
        """Test that initialize creates the DLQ table."""
        manager = DeadLetterQueueManager(db_pool=mock_db_pool)

        # Setup mock connection
        mock_conn = mock_db_pool.get_connection()
        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await manager.initialize()

        # Should execute CREATE TABLE
        mock_conn.execute.assert_called()
        create_table_sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS" in create_table_sql
        assert "event_dlq" in create_table_sql

    @pytest.mark.asyncio
    async def test_add_failed_event(self, dlq_manager):
        """Test adding a failed event to DLQ."""
        event = create_scan_alert(symbol="AAPL")
        error_message = "Processing failed"

        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"
        ) as mock_batch_upsert:
            mock_batch_upsert.return_value = 1

            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
            ) as mock_tx:
                mock_conn = MockConnection()
                mock_tx.return_value.__aenter__.return_value = mock_conn

                await dlq_manager.add_failed_event(event, error_message)

                # Should use batch_upsert
                mock_batch_upsert.assert_called_once()
                call_args = mock_batch_upsert.call_args

                # Check table name
                assert call_args[0][1] == "event_dlq"

                # Check record data
                records = call_args[0][2]
                assert len(records) == 1
                record = records[0]

                assert record["event_id"] == event.event_id
                assert record["event_type"] == event.event_type.value
                assert record["error_message"] == error_message
                assert record["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_add_failed_event_with_retry_count(self, dlq_manager):
        """Test adding failed event with existing retry count."""
        event = create_error_event()
        event.metadata["retry_count"] = 2

        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"
        ) as mock_batch_upsert:
            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
            ) as mock_tx:
                mock_conn = MockConnection()
                mock_tx.return_value.__aenter__.return_value = mock_conn

                await dlq_manager.add_failed_event(event, "Still failing")

                # Check retry count is preserved
                records = mock_batch_upsert.call_args[0][2]
                assert records[0]["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_add_failed_event_records_metrics(self, dlq_manager):
        """Test that adding failed event records metrics."""
        event = create_scan_alert()

        with patch("main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"):
            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
            ) as mock_tx:
                with patch(
                    "main.events.event_bus_helpers.dead_letter_queue_manager.record_metric"
                ) as mock_metric:
                    mock_conn = MockConnection()
                    mock_tx.return_value.__aenter__.return_value = mock_conn

                    await dlq_manager.add_failed_event(event, "Test error")

                    # Should record metric
                    mock_metric.assert_called_with(
                        "dlq.event_added",
                        1,
                        tags={"event_type": event.event_type.value, "retry_count": 0},
                    )

    @pytest.mark.asyncio
    async def test_get_failed_events(self, dlq_manager):
        """Test retrieving failed events from DLQ."""
        # Setup mock data
        mock_rows = [
            {
                "id": 1,
                "event_id": "test_123",
                "event_type": "scanner_alert",
                "event_data": json.dumps({"symbol": "AAPL"}),
                "error_message": "Error 1",
                "retry_count": 0,
                "created_at": datetime.now(UTC),
                "last_retry": None,
            },
            {
                "id": 2,
                "event_id": "test_456",
                "event_type": "order_placed",
                "event_data": json.dumps({"symbol": "GOOGL"}),
                "error_message": "Error 2",
                "retry_count": 1,
                "created_at": datetime.now(UTC),
                "last_retry": datetime.now(UTC),
            },
        ]

        mock_conn = MockConnection()
        mock_conn.fetch.return_value = mock_rows
        dlq_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Get events
        events = await dlq_manager.get_failed_events(limit=10)

        assert len(events) == 2
        assert events[0]["event_id"] == "test_123"
        assert events[1]["event_id"] == "test_456"

        # Check SQL query
        sql = mock_conn.fetch.call_args[0][0]
        assert "SELECT" in sql
        assert "FROM event_dlq" in sql
        assert "LIMIT" in sql

    @pytest.mark.asyncio
    async def test_get_failed_events_by_type(self, dlq_manager):
        """Test filtering failed events by event type."""
        mock_conn = MockConnection()
        mock_conn.fetch.return_value = []
        dlq_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await dlq_manager.get_failed_events(event_type=EventType.SCANNER_ALERT, limit=5)

        # Check SQL includes event type filter
        sql = mock_conn.fetch.call_args[0][0]
        assert "WHERE event_type = $1" in sql
        assert mock_conn.fetch.call_args[0][1] == "scanner_alert"

    @pytest.mark.asyncio
    async def test_get_retryable_events(self, dlq_manager):
        """Test getting events eligible for retry."""
        mock_rows = [
            {
                "id": 1,
                "event_id": "retry_1",
                "event_type": "scanner_alert",
                "event_data": "{}",
                "error_message": "Temporary error",
                "retry_count": 1,
                "created_at": datetime.now(UTC),
                "last_retry": datetime.now(UTC) - timedelta(minutes=10),
            }
        ]

        mock_conn = MockConnection()
        mock_conn.fetch.return_value = mock_rows
        dlq_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        events = await dlq_manager.get_retryable_events()

        assert len(events) == 1

        # Check SQL conditions
        sql = mock_conn.fetch.call_args[0][0]
        assert "retry_count < $1" in sql
        assert "last_retry < $2" in sql or "last_retry IS NULL" in sql

    @pytest.mark.asyncio
    async def test_mark_event_retried(self, dlq_manager):
        """Test marking an event as retried."""
        event_id = "test_123"

        mock_conn = MockConnection()
        dlq_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await dlq_manager.mark_event_retried(event_id)

        # Check UPDATE query
        sql = mock_conn.execute.call_args[0][0]
        assert "UPDATE event_dlq" in sql
        assert "SET retry_count = retry_count + 1" in sql
        assert "WHERE event_id = $1" in sql
        assert mock_conn.execute.call_args[0][1] == event_id

    @pytest.mark.asyncio
    async def test_delete_event(self, dlq_manager):
        """Test deleting an event from DLQ."""
        event_id = "test_123"

        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.execute_with_retry"
        ) as mock_retry:
            mock_retry.return_value = None

            await dlq_manager.delete_event(event_id)

            # Should use execute_with_retry for resilience
            mock_retry.assert_called_once()

            # Check the function passed to execute_with_retry
            delete_func = mock_retry.call_args[0][0]
            assert callable(delete_func)

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, dlq_manager):
        """Test cleaning up old events."""
        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.execute_with_retry"
        ) as mock_retry:
            mock_retry.return_value = 10  # Deleted count

            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.record_metric"
            ) as mock_metric:
                deleted_count = await dlq_manager.cleanup_old_events()

                assert deleted_count == 10

                # Should record metric
                mock_metric.assert_called_with("dlq.events_cleaned_up", 10)

    @pytest.mark.asyncio
    async def test_get_stats(self, dlq_manager):
        """Test getting DLQ statistics."""
        # Setup mock data
        mock_stats = [
            {"event_type": "scanner_alert", "count": 50, "avg_retry_count": 1.5},
            {"event_type": "order_placed", "count": 20, "avg_retry_count": 2.0},
        ]

        mock_conn = MockConnection()
        mock_conn.fetch.return_value = mock_stats
        mock_conn.fetchval.return_value = 70  # Total count
        dlq_manager.db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        stats = await dlq_manager.get_stats()

        assert stats["total_events"] == 70
        assert stats["events_by_type"]["scanner_alert"]["count"] == 50
        assert stats["events_by_type"]["scanner_alert"]["avg_retry_count"] == 1.5
        assert stats["events_by_type"]["order_placed"]["count"] == 20

    @pytest.mark.asyncio
    async def test_close(self, dlq_manager):
        """Test closing the DLQ manager."""
        # Close should not raise errors
        await dlq_manager.close()

        # If manager tracks closure state
        if hasattr(dlq_manager, "_closed"):
            assert dlq_manager._closed is True

    @pytest.mark.asyncio
    async def test_error_handling_in_add_failed_event(self, dlq_manager):
        """Test error handling when adding failed event."""
        event = create_scan_alert()

        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"
        ) as mock_batch:
            mock_batch.side_effect = Exception("Database error")

            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
            ) as mock_tx:
                mock_conn = MockConnection()
                mock_tx.return_value.__aenter__.return_value = mock_conn

                # Should handle error gracefully
                with pytest.raises(Exception):
                    await dlq_manager.add_failed_event(event, "Test error")

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, dlq_manager):
        """Test transaction rollback on error."""
        event = create_scan_alert()

        with patch(
            "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
        ) as mock_tx:
            # Setup context manager that raises in the middle
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = MockConnection()
            mock_context.__aexit__.return_value = None
            mock_tx.return_value = mock_context

            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"
            ) as mock_batch:
                mock_batch.side_effect = Exception("DB error")

                try:
                    await dlq_manager.add_failed_event(event, "Error")
                except:
                    pass

                # Transaction should handle rollback
                mock_context.__aexit__.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, dlq_manager):
        """Test concurrent DLQ operations."""

        async def add_events():
            for i in range(10):
                event = create_scan_alert(symbol=f"STOCK{i}")
                await dlq_manager.add_failed_event(event, f"Error {i}")

        async def get_events():
            for _ in range(5):
                await dlq_manager.get_failed_events(limit=10)
                await asyncio.sleep(0.01)

        with patch("main.events.event_bus_helpers.dead_letter_queue_manager.batch_upsert"):
            with patch(
                "main.events.event_bus_helpers.dead_letter_queue_manager.transaction_context"
            ) as mock_tx:
                mock_conn = MockConnection()
                mock_tx.return_value.__aenter__.return_value = mock_conn

                # Run concurrent operations
                await asyncio.gather(add_events(), get_events(), add_events())

                # Should complete without errors

"""Mock database objects for testing."""

# Standard library imports
from datetime import UTC, datetime
import json
from typing import Any
from unittest.mock import AsyncMock


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self):
        self.execute = AsyncMock()
        self.fetch = AsyncMock()
        self.fetchrow = AsyncMock()
        self.fetchval = AsyncMock()
        self.executemany = AsyncMock()
        self.close = AsyncMock()
        self._transaction_active = False
        self._in_transaction_block = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    def transaction(self):
        """Return a mock transaction context."""
        return MockTransaction(self)


class MockTransaction:
    """Mock database transaction for testing."""

    def __init__(self, connection: MockConnection):
        self.connection = connection
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def __aenter__(self):
        self.connection._transaction_active = True
        self.connection._in_transaction_block = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
        self.connection._transaction_active = False
        self.connection._in_transaction_block = False
        return False


class MockDatabasePool:
    """Mock database pool for testing."""

    def __init__(self):
        self._connections = []
        self.acquire = AsyncMock(return_value=MockConnection())
        self.close = AsyncMock()
        self._dead_letter_events = []
        self._event_history = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    def get_connection(self) -> MockConnection:
        """Get a mock connection."""
        conn = MockConnection()
        self._connections.append(conn)
        return conn

    def setup_dlq_responses(self, events: list[dict[str, Any]]):
        """Setup mock responses for dead letter queue queries."""
        self._dead_letter_events = events

        # Setup execute response for insert
        async def mock_dlq_insert(*args, **kwargs):
            if "INSERT INTO event_dlq" in str(args):
                self._dead_letter_events.append(
                    {
                        "event_id": kwargs.get("event_id"),
                        "event_type": kwargs.get("event_type"),
                        "event_data": kwargs.get("event_data"),
                        "error_message": kwargs.get("error_message"),
                        "retry_count": kwargs.get("retry_count", 0),
                        "created_at": datetime.now(UTC),
                    }
                )

        # Setup fetch response for queries
        async def mock_dlq_fetch(*args, **kwargs):
            return self._dead_letter_events

        for conn in self._connections:
            conn.execute.side_effect = mock_dlq_insert
            conn.fetch.return_value = self._dead_letter_events


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}
        self.get = AsyncMock(side_effect=self._get)
        self.set = AsyncMock(side_effect=self._set)
        self.delete = AsyncMock(side_effect=self._delete)
        self.exists = AsyncMock(side_effect=self._exists)
        self.expire = AsyncMock()
        self.ttl = AsyncMock(return_value=60)
        self.incr = AsyncMock(return_value=1)
        self.decr = AsyncMock(return_value=0)
        self.hget = AsyncMock()
        self.hset = AsyncMock()
        self.hdel = AsyncMock()
        self.hgetall = AsyncMock(return_value={})
        self.lpush = AsyncMock()
        self.rpop = AsyncMock()
        self.llen = AsyncMock(return_value=0)

    async def _get(self, key: str) -> str | None:
        return self._data.get(key)

    async def _set(self, key: str, value: str, ex: int = None) -> None:
        self._data[key] = value

    async def _delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def _exists(self, key: str) -> bool:
        return key in self._data


def create_mock_db_pool() -> MockDatabasePool:
    """Create a configured mock database pool."""
    pool = MockDatabasePool()

    # Setup default connection behavior
    async def acquire_connection():
        return pool.get_connection()

    pool.acquire = AsyncMock(side_effect=acquire_connection)
    return pool


def create_mock_batch_upsert():
    """Create a mock batch_upsert function."""

    async def mock_batch_upsert(
        conn,
        table_name: str,
        records: list[dict[str, Any]],
        on_conflict: str = "update",
        conflict_columns: list[str] = None,
    ) -> int:
        # Simulate successful batch upsert
        return len(records)

    return AsyncMock(side_effect=mock_batch_upsert)


def create_mock_transaction_context():
    """Create a mock transaction_context function."""

    class MockTransactionContext:
        def __init__(self):
            self.conn = MockConnection()

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    return MockTransactionContext


def create_mock_execute_with_retry():
    """Create a mock execute_with_retry function."""

    async def mock_execute_with_retry(
        func, *args, max_retries: int = 3, retry_delay: float = 1.0, **kwargs
    ):
        # Just execute the function once
        return await func(*args, **kwargs)

    return AsyncMock(side_effect=mock_execute_with_retry)


# Helper functions for setting up test data
def setup_test_dlq_data(pool: MockDatabasePool, count: int = 5) -> list[dict[str, Any]]:
    """Setup test data in the mock DLQ."""
    events = []
    for i in range(count):
        event = {
            "id": i + 1,
            "event_id": f"test_event_{i}",
            "event_type": "TEST_EVENT",
            "event_data": json.dumps({"test": True, "index": i}),
            "error_message": f"Test error {i}",
            "retry_count": i % 3,
            "created_at": datetime.now(UTC),
            "last_retry": datetime.now(UTC) if i % 2 else None,
        }
        events.append(event)

    pool.setup_dlq_responses(events)
    return events

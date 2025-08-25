"""
Unit tests for the audit logger functionality.

Tests cover core audit logging features including performance,
reliability, context management, and error handling.
"""

import asyncio
import threading
import time
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.infrastructure.audit.config import AuditConfig
from src.infrastructure.audit.events import OrderEvent
from src.infrastructure.audit.exceptions import AuditException
from src.infrastructure.audit.formatters import JSONFormatter
from src.infrastructure.audit.logger import AsyncAuditLogger, AuditContext, AuditLogger
from src.infrastructure.audit.storage import AuditStorage


class MockStorage(AuditStorage):
    """Mock storage for testing."""

    def __init__(self):
        super().__init__(
            AuditConfig(
                security_config=SecurityConfig(
                    encryption_enabled=False,
                    digital_signatures_enabled=False,
                    access_control_enabled=False,
                    integrity_checks_enabled=False,
                    tls_enabled=False,
                    tamper_detection_enabled=False,
                ),
                storage_config=StorageConfig(
                    primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
                ),
            )
        )
        self.stored_events = []
        self.should_fail = False

    def store(self, event_data):
        if self.should_fail:
            raise Exception("Storage failure")
        self.stored_events.append(event_data)
        return f"mock_ref_{len(self.stored_events)}"

    def store_batch(self, events):
        if self.should_fail:
            raise Exception("Batch storage failure")
        self.stored_events.extend(events)
        return [f"mock_ref_{i}" for i in range(len(events))]

    def retrieve(self, event_id):
        return None

    def query(self, **kwargs):
        return iter([])

    def delete_expired(self, cutoff_date):
        return 0

    def verify_integrity(self):
        return True

    def get_storage_stats(self):
        return {}

    def close(self):
        pass


@pytest.fixture
def mock_config():
    """Create mock audit configuration."""
    from src.infrastructure.audit.config import SecurityConfig, StorageBackend, StorageConfig

    # Create config with security disabled for testing
    security_config = SecurityConfig(
        encryption_enabled=False,
        digital_signatures_enabled=False,
        access_control_enabled=False,
        integrity_checks_enabled=False,
        tls_enabled=False,
        tamper_detection_enabled=False,
    )
    # Use memory storage for testing
    storage_config = StorageConfig(
        primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
    )
    config = AuditConfig(security_config=security_config, storage_config=storage_config)
    return config


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    return MockStorage()


@pytest.fixture
def mock_formatter():
    """Create mock formatter."""
    return JSONFormatter()


@pytest.fixture
def audit_logger(mock_config, mock_storage, mock_formatter):
    """Create audit logger instance for testing."""
    return AuditLogger(
        config=mock_config,
        storage=mock_storage,
        formatter=mock_formatter,
        enable_async=False,  # Disable async for simpler testing
    )


@pytest.fixture
def async_audit_logger(mock_config, mock_storage, mock_formatter):
    """Create async audit logger instance for testing."""
    return AsyncAuditLogger(config=mock_config, storage=mock_storage, formatter=mock_formatter)


@pytest.fixture
def sample_order_event():
    """Create sample order event for testing."""
    return OrderEvent(
        event_type="order_create",
        resource_id="order_123",
        action="create",
        order_id="order_123",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        price=Decimal("150.00"),
        order_type="limit",
    )


@pytest.fixture
def sample_audit_context():
    """Create sample audit context for testing."""
    return AuditContext(
        user_id="user_123",
        session_id="session_456",
        request_id="request_789",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )


class TestAuditLogger:
    """Test suite for AuditLogger class."""

    def test_init(self, mock_config, mock_storage, mock_formatter):
        """Test audit logger initialization."""
        logger = AuditLogger(
            config=mock_config,
            storage=mock_storage,
            formatter=mock_formatter,
            enable_async=True,
            max_workers=2,
        )

        assert logger.config == mock_config
        assert logger.storage == mock_storage
        assert logger.formatter == mock_formatter
        assert logger.enable_async == True
        assert logger.max_workers == 2
        assert logger._event_count == 0
        assert logger._error_count == 0
        assert logger._executor is not None

    def test_context_manager(self, audit_logger):
        """Test audit logger as context manager."""
        with audit_logger as logger:
            assert logger is not None
        # Should not raise exception

    def test_log_event_success(self, audit_logger, sample_order_event, mock_storage):
        """Test successful event logging."""
        event_id = audit_logger.log_event(sample_order_event)

        assert event_id is not None
        assert len(mock_storage.stored_events) == 1
        assert audit_logger._event_count == 1
        assert audit_logger._error_count == 0

        stored_event = mock_storage.stored_events[0]
        assert stored_event["event_type"] == "order_create"
        assert stored_event["resource_type"] == "order"
        assert stored_event["action"] == "create"
        assert "event_id" in stored_event
        assert "timestamp_utc" in stored_event
        assert "integrity_hash" in stored_event

    def test_log_event_with_context(
        self, audit_logger, sample_order_event, sample_audit_context, mock_storage
    ):
        """Test event logging with audit context."""
        event_id = audit_logger.log_event(sample_order_event, context=sample_audit_context)

        assert event_id is not None
        assert len(mock_storage.stored_events) == 1

        stored_event = mock_storage.stored_events[0]
        assert "context" in stored_event
        assert stored_event["context"]["user_id"] == "user_123"
        assert stored_event["context"]["session_id"] == "session_456"
        assert stored_event["context"]["ip_address"] == "192.168.1.1"

    def test_log_event_thread_context(
        self, audit_logger, sample_order_event, sample_audit_context, mock_storage
    ):
        """Test event logging with thread-local context."""
        with audit_logger.audit_context(sample_audit_context):
            event_id = audit_logger.log_event(sample_order_event)

        assert event_id is not None
        stored_event = mock_storage.stored_events[0]
        assert stored_event["context"]["user_id"] == "user_123"

    def test_log_event_immediate_processing(self, mock_config, mock_storage, mock_formatter):
        """Test immediate event processing."""
        logger = AuditLogger(
            config=mock_config,
            storage=mock_storage,
            formatter=mock_formatter,
            enable_async=True,  # Enable async but force immediate
        )

        event = OrderEvent(
            event_type="order_create",
            resource_id="order_123",
            action="create",
            order_id="order_123",
            symbol="AAPL",
        )

        event_id = logger.log_event(event, immediate=True)
        assert event_id is not None
        assert len(mock_storage.stored_events) == 1

    def test_log_event_storage_failure(self, audit_logger, sample_order_event, mock_storage):
        """Test event logging with storage failure."""
        mock_storage.should_fail = True

        with pytest.raises(AuditException) as exc_info:
            audit_logger.log_event(sample_order_event)

        assert "Failed to log audit event" in str(exc_info)
        assert audit_logger._error_count == 1

    def test_log_event_critical_failure_handling(self, audit_logger, mock_storage):
        """Test handling of critical event failures."""
        critical_event = OrderEvent(
            event_type="order_create",
            resource_id="order_123",
            action="create",
            order_id="order_123",
            symbol="AAPL",
            is_critical=True,
        )

        mock_storage.should_fail = True

        with patch.object(audit_logger, "_log_failure") as mock_log_failure:
            with pytest.raises(AuditException):
                audit_logger.log_event(critical_event)

            mock_log_failure.assert_called_once()

    def test_convenience_methods(self, audit_logger, mock_storage):
        """Test convenience logging methods."""
        # Test order event logging
        event_id = audit_logger.log_order_event(
            event_type="order_create",
            order_id="order_123",
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            user_id="user_123",
        )
        assert event_id is not None

        # Test position event logging
        event_id = audit_logger.log_position_event(
            event_type="position_open",
            position_id="pos_456",
            symbol="GOOGL",
            quantity=50.0,
            user_id="user_123",
        )
        assert event_id is not None

        # Test portfolio event logging
        event_id = audit_logger.log_portfolio_event(
            event_type="portfolio_update", portfolio_id="port_789", user_id="user_123"
        )
        assert event_id is not None

        # Test risk event logging
        event_id = audit_logger.log_risk_event(
            event_type="risk_breach",
            risk_type="position_limit",
            threshold_value=100000.0,
            current_value=150000.0,
            user_id="user_123",
        )
        assert event_id is not None

        assert len(mock_storage.stored_events) == 4

    def test_monitoring_hooks(self, audit_logger, sample_order_event):
        """Test monitoring hook functionality."""
        hook_calls = []

        def monitoring_hook(event_data):
            hook_calls.append(event_data)

        # Add hook
        audit_logger.add_monitoring_hook(monitoring_hook)

        # Log event
        audit_logger.log_event(sample_order_event)

        assert len(hook_calls) == 1
        assert hook_calls[0]["event_type"] == "order_create"

        # Remove hook
        audit_logger.remove_monitoring_hook(monitoring_hook)

        # Log another event
        audit_logger.log_event(sample_order_event)

        # Hook should not be called again
        assert len(hook_calls) == 1

    def test_monitoring_hook_failure(self, audit_logger, sample_order_event):
        """Test monitoring hook failure handling."""

        def failing_hook(event_data):
            raise Exception("Hook failed")

        audit_logger.add_monitoring_hook(failing_hook)

        # Should not raise exception despite hook failure
        event_id = audit_logger.log_event(sample_order_event)
        assert event_id is not None

    def test_get_metrics(self, audit_logger, sample_order_event, mock_storage):
        """Test metrics collection."""
        initial_metrics = audit_logger.get_metrics()
        assert initial_metrics["event_count"] == 0
        assert initial_metrics["error_count"] == 0
        assert initial_metrics["events_per_second"] == 0

        # Log some events
        for i in range(5):
            audit_logger.log_event(sample_order_event)

        # Simulate a failure
        mock_storage.should_fail = True
        try:
            audit_logger.log_event(sample_order_event)
        except AuditException:
            pass

        metrics = audit_logger.get_metrics()
        assert metrics["event_count"] == 5
        assert metrics["error_count"] == 1
        assert metrics["events_per_second"] > 0
        assert metrics["error_rate"] == 1 / 6

    def test_event_validation(self, mock_config, mock_storage, mock_formatter):
        """Test event validation."""
        mock_config.strict_validation = True
        logger = AuditLogger(mock_config, mock_storage, mock_formatter)

        # Create invalid event (missing required fields)
        invalid_event = OrderEvent(
            event_type="",  # Empty event type
            resource_id="order_123",
            action="create",
            order_id="order_123",
            symbol="AAPL",
        )

        with pytest.raises(AuditException):
            logger.log_event(invalid_event)

    def test_integrity_hash_calculation(self, audit_logger, sample_order_event, mock_storage):
        """Test integrity hash calculation."""
        audit_logger.log_event(sample_order_event)

        stored_event = mock_storage.stored_events[0]
        assert "integrity_hash" in stored_event
        assert len(stored_event["integrity_hash"]) == 64  # SHA-256 hash length

    def test_concurrent_logging(self, audit_logger, sample_order_event, mock_storage):
        """Test concurrent event logging."""

        def log_events():
            for i in range(10):
                audit_logger.log_event(sample_order_event)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_events)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(mock_storage.stored_events) == 50
        assert audit_logger._event_count == 50

    def test_performance_benchmarking(self, audit_logger, sample_order_event):
        """Test logging performance."""
        num_events = 100
        start_time = time.time()

        for i in range(num_events):
            audit_logger.log_event(sample_order_event)

        end_time = time.time()
        duration = end_time - start_time
        events_per_second = num_events / duration

        # Should be able to log at least 1000 events per second
        assert events_per_second > 1000

    def test_close_cleanup(self, mock_config, mock_storage, mock_formatter):
        """Test proper cleanup on close."""
        logger = AuditLogger(mock_config, mock_storage, mock_formatter, enable_async=True)

        # Close should not raise exception
        logger.close()

        # Executor should be shutdown
        assert logger._executor._shutdown


class TestAsyncAuditLogger:
    """Test suite for AsyncAuditLogger class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_config, mock_storage, mock_formatter):
        """Test async audit logger initialization."""
        logger = AsyncAuditLogger(mock_config, mock_storage, mock_formatter)

        assert logger.config == mock_config
        assert logger.storage == mock_storage
        assert logger.formatter == mock_formatter
        assert logger._event_count == 0
        assert logger._error_count == 0

    @pytest.mark.asyncio
    async def test_log_event_success(self, async_audit_logger, sample_order_event, mock_storage):
        """Test successful async event logging."""
        await async_audit_logger.start()

        event_id = await async_audit_logger.log_event(sample_order_event)

        # Allow time for async processing
        await asyncio.sleep(0.1)

        assert event_id is not None
        assert async_audit_logger._event_count == 1

        await async_audit_logger.stop()

    @pytest.mark.asyncio
    async def test_queue_overflow(self, mock_config, mock_storage, mock_formatter):
        """Test queue overflow handling."""
        # Create logger with small queue
        mock_config.performance_config.event_queue_size = 5
        logger = AsyncAuditLogger(mock_config, mock_storage, mock_formatter)
        logger._event_queue = asyncio.Queue(maxsize=2)  # Very small queue

        event = OrderEvent(
            event_type="order_create",
            resource_id="order_123",
            action="create",
            order_id="order_123",
            symbol="AAPL",
        )

        # Fill the queue
        await logger.log_event(event)
        await logger.log_event(event)

        # This should raise an exception due to queue overflow
        with pytest.raises(AuditException) as exc_info:
            await logger.log_event(event)

        assert "queue full" in str(exc_info).lower()

    @pytest.mark.asyncio
    async def test_batch_processing(self, async_audit_logger, mock_storage):
        """Test batch processing of events."""
        await async_audit_logger.start()

        events = []
        for i in range(5):
            event = OrderEvent(
                event_type="order_create",
                resource_id=f"order_{i}",
                action="create",
                order_id=f"order_{i}",
                symbol="AAPL",
            )
            events.append(event)
            await async_audit_logger.log_event(event)

        # Allow time for batch processing
        await asyncio.sleep(0.2)
        await async_audit_logger.stop()

        # Events should be processed in batches
        assert len(mock_storage.stored_events) >= 5

    @pytest.mark.asyncio
    async def test_metrics(self, async_audit_logger, sample_order_event):
        """Test async logger metrics."""
        initial_metrics = async_audit_logger.get_metrics()
        assert initial_metrics["event_count"] == 0
        assert initial_metrics["queue_size"] == 0

        await async_audit_logger.log_event(sample_order_event)

        metrics = async_audit_logger.get_metrics()
        assert metrics["event_count"] == 1

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, async_audit_logger):
        """Test start/stop lifecycle."""
        # Start processing
        await async_audit_logger.start()
        assert async_audit_logger._processing_task is not None

        # Stop processing
        await async_audit_logger.stop()
        assert async_audit_logger._shutdown_event.is_set()


class TestAuditContext:
    """Test suite for AuditContext class."""

    def test_init(self):
        """Test audit context initialization."""
        context = AuditContext(
            user_id="user_123",
            session_id="session_456",
            request_id="request_789",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            metadata={"key": "value"},
        )

        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.request_id == "request_789"
        assert context.ip_address == "192.168.1.1"
        assert context.user_agent == "TestAgent/1.0"
        assert context.metadata == {"key": "value"}
        assert len(context.session_id) > 0  # Should have default session ID

    def test_default_session_id(self):
        """Test default session ID generation."""
        context = AuditContext()
        assert len(context.session_id) > 0
        assert context.session_id.count("-") == 4  # UUID format

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = AuditContext(
            user_id="user_123", ip_address="192.168.1.1", metadata={"key": "value"}
        )

        context_dict = context.to_dict()

        assert context_dict["user_id"] == "user_123"
        assert context_dict["ip_address"] == "192.168.1.1"
        assert context_dict["metadata"] == {"key": "value"}
        assert "session_id" in context_dict

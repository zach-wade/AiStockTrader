"""Unit tests for event_types module."""

# Standard library imports
from datetime import UTC, datetime

# Third-party imports
import pytest

# Local imports
from main.interfaces.events.event_types import (
    ErrorEvent,
    Event,
    EventPriority,
    EventType,
    FeatureRequestEvent,
    OrderEvent,
    ScannerAlertEvent,
)
from main.utils.core import ensure_utc


class TestEventEnums:
    """Test event enumeration types."""

    def test_event_type_values(self):
        """Test EventType enum values."""
        assert EventType.SCANNER_ALERT.value == "scanner_alert"
        assert EventType.FEATURE_REQUEST.value == "feature_request"
        assert EventType.ORDER_PLACED.value == "order_placed"
        assert EventType.ERROR.value == "error"

    def test_event_priority_values(self):
        """Test EventPriority enum values."""
        assert EventPriority.LOW.value == 1
        assert EventPriority.NORMAL.value == 5
        assert EventPriority.HIGH.value == 7
        assert EventPriority.CRITICAL.value == 10

    def test_event_type_system_values(self):
        """Test system EventType enum values."""
        assert EventType.SYSTEM_STATUS.value == "system_status"
        assert EventType.DATA_INGESTED.value == "data_ingested"
        assert EventType.DATA_PROCESSED.value == "data_processed"
        assert EventType.DATA_VALIDATED.value == "data_validated"


class TestEvent:
    """Test base Event class."""

    def test_event_creation(self):
        """Test creating a basic event."""
        event = Event(
            event_type=EventType.SCANNER_ALERT, source="test_source", metadata={"key": "value"}
        )

        assert event.event_type == EventType.SCANNER_ALERT
        assert event.source == "test_source"
        assert event.metadata == {"key": "value"}
        assert event.event_id is not None
        assert event.correlation_id is None  # correlation_id is optional and defaults to None
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None

    def test_event_with_metadata(self):
        """Test event with metadata."""
        metadata = {"user": "test", "version": "1.0"}
        event = Event(event_type=EventType.SYSTEM_STATUS, source="test", metadata=metadata)

        assert event.metadata == metadata

    def test_event_with_correlation_id(self):
        """Test event with custom correlation ID."""
        event = Event(
            event_type=EventType.ERROR, source="test", correlation_id="custom_correlation_123"
        )

        assert event.correlation_id == "custom_correlation_123"
        assert event.event_id != event.correlation_id

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_type=EventType.FEATURE_REQUEST,
            source="test",
            metadata={"symbols": ["AAPL"], "batch": 1},
        )

        event_dict = event.to_dict()
        assert event_dict["event_type"] == "feature_request"
        assert event_dict["source"] == "test"
        assert event_dict["metadata"] == {"symbols": ["AAPL"], "batch": 1}
        assert "timestamp" in event_dict
        assert "event_id" in event_dict

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        event_dict = {
            "event_type": "scanner_alert",
            "source": "test",
            "metadata": {"alert": True},
            "event_id": "test_123",
            "timestamp": "2024-01-01T12:00:00+00:00",
        }

        event = Event.from_dict(event_dict)
        assert event.event_type == EventType.SCANNER_ALERT
        assert event.source == "test"
        assert event.metadata == {"alert": True}
        assert event.event_id == "test_123"
        assert isinstance(event.timestamp, datetime)

    def test_event_json_serialization(self):
        """Test JSON serialization of event."""
        event = Event(
            event_type=EventType.ORDER_PLACED,
            source="trading_engine",
            metadata={"symbol": "AAPL", "quantity": 100},
        )

        # Test to_json and from_json methods
        json_str = event.to_json()
        assert json_str is not None

        # Should be deserializable
        restored_event = Event.from_json(json_str)
        assert restored_event.event_type == event.event_type
        assert restored_event.source == event.source
        assert restored_event.metadata == event.metadata


class TestScannerAlertEvent:
    """Test ScannerAlertEvent class."""

    def test_scanner_alert_creation(self):
        """Test creating a scanner alert event."""
        alert = ScannerAlertEvent(
            symbol="AAPL",
            alert_type="volume_spike",
            score=0.85,
            scanner_name="volume_scanner",
            source="volume_scanner",
        )

        assert alert.symbol == "AAPL"
        assert alert.alert_type == "volume_spike"
        assert alert.score == 0.85
        assert alert.scanner_name == "volume_scanner"
        assert alert.event_type == EventType.SCANNER_ALERT

    def test_scanner_alert_with_metadata(self):
        """Test scanner alert with additional metadata."""
        metadata = {"volume_multiplier": 5.2, "average_volume": 1000000, "current_volume": 5200000}

        alert = ScannerAlertEvent(
            symbol="TSLA",
            alert_type="volatility_spike",
            score=0.92,
            scanner_name="volatility_scanner",
            source="scanner",
            metadata=metadata,
        )

        assert alert.metadata == metadata

    def test_scanner_alert_priority(self):
        """Test scanner alert with different priorities."""
        # High score should get high priority
        alert1 = ScannerAlertEvent(
            symbol="AAPL",
            alert_type="ml_signal",
            score=0.95,
            scanner_name="ml_scanner",
            source="scanner",
            priority=EventPriority.HIGH,
        )

        assert alert1.priority == EventPriority.HIGH

    def test_scanner_alert_timestamp(self):
        """Test scanner alert timestamp handling."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        alert = ScannerAlertEvent(
            symbol="GOOGL",
            alert_type="news_sentiment",
            score=0.75,
            scanner_name="news_scanner",
            source="scanner",
            timestamp=custom_time,
        )

        assert alert.timestamp == custom_time
        assert alert.timestamp.tzinfo is not None


class TestOrderEvent:
    """Test OrderEvent class."""

    def test_order_event_creation(self):
        """Test creating an order event."""
        order = OrderEvent(
            order_id="order_123",
            symbol="AAPL",
            quantity=100.0,
            side="buy",
            order_type="limit",
            price=150.0,
        )

        assert order.order_id == "order_123"
        assert order.symbol == "AAPL"
        assert order.quantity == 100.0
        assert order.price == 150.0
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.event_type == EventType.ORDER_PLACED

    def test_order_event_market_order(self):
        """Test market order event."""
        order = OrderEvent(
            order_id="order_456",
            symbol="MSFT",
            quantity=50.0,
            side="sell",
            order_type="market",
            price=None,  # Market orders may not have price
        )

        assert order.price is None
        assert order.order_type == "market"

    def test_order_event_with_metadata(self):
        """Test order event with metadata."""
        order = OrderEvent(
            order_id="order_789",
            symbol="GOOGL",
            quantity=25.0,
            side="buy",
            order_type="limit",
            price=2500.0,
            metadata={"strategy": "momentum", "signal_strength": 0.8},
        )

        assert order.metadata["strategy"] == "momentum"
        assert order.metadata["signal_strength"] == 0.8


class TestFeatureRequestEvent:
    """Test FeatureRequestEvent class."""

    def test_feature_request_creation(self):
        """Test creating a feature request event."""
        request = FeatureRequestEvent(
            symbols=["AAPL", "MSFT"],
            features=["rsi", "macd"],
            requester="scanner",
            priority=EventPriority.HIGH,
        )

        assert request.symbols == ["AAPL", "MSFT"]
        assert request.features == ["rsi", "macd"]
        assert request.requester == "scanner"
        assert request.priority == EventPriority.HIGH
        assert request.event_type == EventType.FEATURE_REQUEST

    def test_feature_request_with_metadata(self):
        """Test feature request with metadata."""
        request = FeatureRequestEvent(
            symbols=["TSLA"],
            features=["volume_ratio"],
            requester="event_driven_engine",
            metadata={"batch_id": "batch_123", "urgency": "high"},
        )

        assert request.metadata["batch_id"] == "batch_123"
        assert request.metadata["urgency"] == "high"


class TestErrorEvent:
    """Test ErrorEvent class."""

    def test_error_event_creation(self):
        """Test creating an error event."""
        error = ErrorEvent(
            error_type="ValueError",
            message="Invalid parameter",
            component="scanner",
            stack_trace="Traceback...",
            recoverable=True,
        )

        assert error.error_type == "ValueError"
        assert error.message == "Invalid parameter"
        assert error.component == "scanner"
        assert error.stack_trace == "Traceback..."
        assert error.recoverable is True
        assert error.event_type == EventType.ERROR

    def test_error_event_critical(self):
        """Test critical error event."""
        error = ErrorEvent(
            error_type="SystemCriticalError",
            message="Database connection lost",
            component="data_pipeline",
            recoverable=False,
            priority=EventPriority.CRITICAL,
        )

        assert error.recoverable is False
        assert error.priority == EventPriority.CRITICAL


class TestUtilityFunctions:
    """Test utility functions."""

    def test_ensure_utc_with_naive_datetime(self):
        """Test ensure_utc with naive datetime."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        utc_dt = ensure_utc(naive_dt)

        assert utc_dt.tzinfo == UTC
        assert utc_dt.hour == 12

    def test_ensure_utc_with_aware_datetime(self):
        """Test ensure_utc with timezone-aware datetime."""
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        utc_dt = ensure_utc(aware_dt)

        assert utc_dt == aware_dt
        assert utc_dt.tzinfo == UTC

    def test_ensure_utc_with_none(self):
        """Test ensure_utc with None returns current UTC time."""
        before = datetime.now(UTC)
        utc_dt = ensure_utc(None)
        after = datetime.now(UTC)

        assert before <= utc_dt <= after
        assert utc_dt.tzinfo == UTC


class TestEventValidation:
    """Test event validation scenarios."""

    def test_invalid_event_type(self):
        """Test handling of invalid event type in from_dict."""
        event_dict = {"event_type": "invalid_type", "source": "test", "data": {}}

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, KeyError)):
            Event.from_dict(event_dict)

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing event_type (required positional argument)
        with pytest.raises(TypeError):
            Event(source="test")

        # source is optional in the new Event API - this should work
        event = Event(event_type=EventType.SYSTEM_STATUS)
        assert event.event_type == EventType.SYSTEM_STATUS
        assert event.source is None

    def test_scanner_alert_score_validation(self):
        """Test scanner alert score is within valid range."""
        alert = ScannerAlertEvent(
            symbol="AAPL",
            alert_type="volume_spike",
            score=1.5,  # Score can be > 1.0 in the new system
            scanner_name="test_scanner",
            source="scanner",
        )

        # Score is not restricted in the new system
        assert alert.score == 1.5

"""Mock events for testing."""

# Standard library imports
from datetime import UTC, datetime

# Local imports
from main.events.types import AlertType, FillEvent, RiskEvent, ScanAlert
from main.interfaces.events import Event, EventPriority, EventType, OrderEvent


def create_scan_alert(
    symbol: str = "AAPL",
    alert_type: AlertType = AlertType.VOLUME_SPIKE,
    score: float = 0.85,
    **kwargs,
) -> ScanAlert:
    """Create a mock ScanAlert for testing."""
    return ScanAlert(
        event_type=EventType.SCANNER_ALERT,
        source="test_scanner",
        symbol=symbol,
        alert_type=alert_type,
        score=score,
        message=kwargs.get("message", f"Test alert for {symbol}"),
        data=kwargs.get(
            "data", {"volume_multiplier": 3.5, "price_change": 0.05, "volatility_level": "high"}
        ),
        metadata=kwargs.get("metadata", {"test": True}),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
        priority=kwargs.get("priority", EventPriority.NORMAL),
    )


def create_order_event(
    symbol: str = "AAPL", quantity: int = 100, price: float = 150.0, side: str = "buy", **kwargs
) -> OrderEvent:
    """Create a mock OrderEvent for testing."""
    return OrderEvent(
        symbol=symbol,
        quantity=quantity,
        price=price,
        side=side,
        order_type=kwargs.get("order_type", "limit"),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


def create_fill_event(
    symbol: str = "AAPL",
    quantity: int = 100,
    direction: str = "buy",
    fill_cost: float = 15000.0,
    **kwargs,
) -> FillEvent:
    """Create a mock FillEvent for testing."""
    return FillEvent(
        symbol=symbol,
        quantity=quantity,
        direction=direction,
        fill_cost=fill_cost,
        commission=kwargs.get("commission", 1.0),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


def create_risk_event(
    event_type: EventType = EventType.RISK_LIMIT_BREACH, severity: str = "warning", **kwargs
) -> RiskEvent:
    """Create a mock RiskEvent for testing."""
    return RiskEvent(
        event_type=event_type,
        severity=severity,
        source=kwargs.get("source", "risk_monitor"),
        data=kwargs.get("data", {"metric": "position_size", "current_value": 0.15, "limit": 0.10}),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


def create_feature_request_event(symbols: list = None, features: list = None, **kwargs) -> Event:
    """Create a mock feature request event for testing."""
    if symbols is None:
        symbols = ["AAPL", "GOOGL"]
    if features is None:
        features = ["price_features", "volume_features"]

    return Event(
        event_type=EventType.FEATURE_REQUEST,
        source=kwargs.get("source", "scanner_bridge"),
        data={"symbols": symbols, "features": features, "priority": kwargs.get("priority", 5)},
        metadata=kwargs.get("metadata", {"batch_id": "test_batch_001"}),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


def create_feature_computed_event(
    symbols_processed: list = None, computation_time: float = 1.5, **kwargs
) -> Event:
    """Create a mock feature computed event for testing."""
    if symbols_processed is None:
        symbols_processed = ["AAPL", "GOOGL"]

    return Event(
        event_type=EventType.FEATURE_COMPUTED,
        source=kwargs.get("source", "feature_worker"),
        data={
            "symbols_processed": symbols_processed,
            "requested_features": kwargs.get("features", ["price_features"]),
            "results_available": True,
            "computation_time_seconds": computation_time,
        },
        metadata=kwargs.get("metadata", {"worker_id": 1}),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


def create_error_event(
    error_type: str = "computation_failed", error_message: str = "Test error", **kwargs
) -> Event:
    """Create a mock error event for testing."""
    return Event(
        event_type=EventType.ERROR,
        source=kwargs.get("source", "test_component"),
        data={
            "error_type": error_type,
            "error_message": error_message,
            "component": kwargs.get("component", "test"),
            "retry_count": kwargs.get("retry_count", 0),
        },
        metadata=kwargs.get("metadata", {"test": True}),
        timestamp=kwargs.get("timestamp", datetime.now(UTC)),
    )


# Batch creation functions
def create_multiple_alerts(count: int = 5, **kwargs) -> list:
    """Create multiple scan alerts for testing."""
    alerts = []
    symbols = kwargs.get("symbols", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
    alert_types = [
        AlertType.HIGH_VOLUME,
        AlertType.PRICE_BREAKOUT,
        AlertType.VOLATILITY_SPIKE,
        AlertType.MOMENTUM_SHIFT,
        AlertType.NEWS_SENTIMENT,
    ]

    for i in range(count):
        symbol = symbols[i % len(symbols)]
        alert_type = alert_types[i % len(alert_types)]
        score = 0.7 + (i * 0.05)  # Varying scores

        alerts.append(
            create_scan_alert(symbol=symbol, alert_type=alert_type, score=min(score, 1.0), **kwargs)
        )

    return alerts


def create_event_sequence(**kwargs) -> dict:
    """Create a sequence of related events for integration testing."""
    alert = create_scan_alert(**kwargs)
    feature_request = create_feature_request_event(symbols=[alert.symbol], source="scanner_bridge")
    feature_computed = create_feature_computed_event(
        symbols_processed=[alert.symbol], source="feature_worker"
    )
    order = create_order_event(symbol=alert.symbol, source="trading_engine")
    fill = create_fill_event(symbol=alert.symbol, source="broker")

    return {
        "alert": alert,
        "feature_request": feature_request,
        "feature_computed": feature_computed,
        "order": order,
        "fill": fill,
    }

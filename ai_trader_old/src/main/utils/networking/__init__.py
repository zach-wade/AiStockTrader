"""Networking utilities package."""

# Import from api module
from .buffering import MessageBuffer
from .connection import WebSocketConnection
from .failover import ConnectionPool, FailoverManager, websocket_context
from .optimizer import (
    OptimizedWebSocketClient,
    WebSocketManager,
    create_optimized_websocket,
    get_websocket_manager,
)
from .types import (
    BufferConfig,
    ConnectionConfig,
    ConnectionState,
    ConnectionStats,
    LatencyMetrics,
    MessagePriority,
    WebSocketMessage,
)
from ..api.base_client import AuthMethod, BaseAPIClient, RateLimitConfig

__all__ = [
    # Types
    "ConnectionState",
    "MessagePriority",
    "LatencyMetrics",
    "BufferConfig",
    "ConnectionConfig",
    "WebSocketMessage",
    "ConnectionStats",
    # Buffering
    "MessageBuffer",
    # Connection
    "WebSocketConnection",
    # Optimizer
    "OptimizedWebSocketClient",
    "WebSocketManager",
    "get_websocket_manager",
    "create_optimized_websocket",
    # Failover
    "FailoverManager",
    "websocket_context",
    "ConnectionPool",
    # API Client
    "BaseAPIClient",
    "AuthMethod",
    "RateLimitConfig",
]

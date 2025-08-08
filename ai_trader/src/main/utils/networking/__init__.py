"""Networking utilities package."""

from .types import (
    ConnectionState,
    MessagePriority,
    LatencyMetrics,
    BufferConfig,
    ConnectionConfig,
    WebSocketMessage,
    ConnectionStats
)

from .buffering import MessageBuffer

from .connection import WebSocketConnection

from .optimizer import (
    OptimizedWebSocketClient,
    WebSocketManager,
    get_websocket_manager,
    create_optimized_websocket
)

from .failover import (
    FailoverManager,
    websocket_context,
    ConnectionPool
)

# Import from api module
from ..api.base_client import BaseAPIClient, AuthMethod, RateLimitConfig

__all__ = [
    # Types
    'ConnectionState',
    'MessagePriority',
    'LatencyMetrics',
    'BufferConfig',
    'ConnectionConfig',
    'WebSocketMessage',
    'ConnectionStats',
    
    # Buffering
    'MessageBuffer',
    
    # Connection
    'WebSocketConnection',
    
    # Optimizer
    'OptimizedWebSocketClient',
    'WebSocketManager',
    'get_websocket_manager',
    'create_optimized_websocket',
    
    # Failover
    'FailoverManager',
    'websocket_context',
    'ConnectionPool',
    
    # API Client
    'BaseAPIClient',
    'AuthMethod',
    'RateLimitConfig'
]
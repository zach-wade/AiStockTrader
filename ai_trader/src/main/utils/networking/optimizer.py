"""
WebSocket Optimizer

Main WebSocket client with optimization features.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Union

from .types import (
    ConnectionState, ConnectionConfig, BufferConfig, WebSocketMessage, 
    MessagePriority, LatencyMetrics
)
from .connection import WebSocketConnection
from .buffering import MessageBuffer
from .failover import FailoverManager

logger = logging.getLogger(__name__)


class OptimizedWebSocketClient:
    """High-performance WebSocket client with advanced features"""
    
    def __init__(
        self,
        connection_config: ConnectionConfig,
        buffer_config: Optional[BufferConfig] = None,
        name: str = "ws_client"
    ):
        self.config = connection_config
        self.buffer_config = buffer_config or BufferConfig()
        self.name = name
        
        # Components
        self.buffer = MessageBuffer(self.buffer_config)
        self.connection = WebSocketConnection(connection_config, self.buffer, name)
        self.failover = FailoverManager(self.connection, connection_config)
        
        # State proxy
        self.state = self.connection.state
        
        # Threading
        self._lock = threading.Lock()
        
        logger.info(f"OptimizedWebSocketClient '{name}' initialized")
    
    async def connect(self) -> bool:
        """Connect to WebSocket with failover support"""
        try:
            # First try direct connection
            if await self.connection.connect():
                self.state = self.connection.state
                return True
            
            # If direct connection fails, try failover
            return await self.failover.connect_with_failover()
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        await self.connection.disconnect()
        self.state = self.connection.state
    
    async def send_message(
        self,
        data: Union[str, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send message with circuit breaker protection"""
        return await self.connection.send_message(data, priority)
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message type"""
        self.connection.register_message_handler(message_type, handler)
    
    def add_error_handler(self, handler: Callable):
        """Add error handler"""
        self.connection.register_error_handler(handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket statistics"""
        with self._lock:
            return {
                'connection': self.connection.get_stats(),
                'latency': self.connection.get_latency_summary(),
                'buffer': self.buffer.get_stats(),
                'failover': self.failover.get_stats()
            }
    
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.connection.is_connected()
    
    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        return self.connection.is_healthy()
    
    async def close(self):
        """Close WebSocket connection and cleanup"""
        logger.info(f"Closing WebSocket client '{self.name}'")
        
        # Stop failover
        await self.failover.stop()
        
        # Close connection
        await self.connection.disconnect()
        
        logger.info(f"WebSocket client '{self.name}' closed")


class WebSocketManager:
    """Centralized WebSocket connection manager"""
    
    def __init__(self):
        self.clients: Dict[str, OptimizedWebSocketClient] = {}
        self._lock = threading.Lock()
    
    def create_client(
        self,
        name: str,
        connection_config: ConnectionConfig,
        buffer_config: Optional[BufferConfig] = None
    ) -> OptimizedWebSocketClient:
        """Create and register a WebSocket client"""
        with self._lock:
            if name in self.clients:
                logger.warning(f"Client '{name}' already exists, returning existing client")
                return self.clients[name]
            
            client = OptimizedWebSocketClient(connection_config, buffer_config, name)
            self.clients[name] = client
            
            logger.info(f"Created WebSocket client: {name}")
            return client
    
    def get_client(self, name: str) -> Optional[OptimizedWebSocketClient]:
        """Get WebSocket client by name"""
        return self.clients.get(name)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all registered clients"""
        results = {}
        
        tasks = []
        for name, client in self.clients.items():
            task = asyncio.create_task(client.connect())
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Failed to connect client '{name}': {e}")
                results[name] = False
        
        return results
    
    async def close_all(self):
        """Close all WebSocket clients"""
        tasks = []
        
        for client in self.clients.values():
            tasks.append(asyncio.create_task(client.close()))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        with self._lock:
            self.clients.clear()
        
        logger.info("All WebSocket clients closed")
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get statistics for all clients"""
        with self._lock:
            return {
                'total_clients': len(self.clients),
                'clients': {
                    name: client.get_stats()
                    for name, client in self.clients.items()
                }
            }
    
    def remove_client(self, name: str) -> bool:
        """Remove client from manager"""
        with self._lock:
            if name in self.clients:
                del self.clients[name]
                logger.info(f"Removed client: {name}")
                return True
            return False
    
    def list_clients(self) -> List[str]:
        """List all client names"""
        with self._lock:
            return list(self.clients.keys())


# Global manager
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


# Convenience functions
async def create_optimized_websocket(
    name: str,
    url: str,
    auth_data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    buffer_config: Optional[BufferConfig] = None
) -> OptimizedWebSocketClient:
    """
    Create an optimized WebSocket client with default configuration
    
    Args:
        name: Client name
        url: WebSocket URL
        auth_data: Authentication data
        headers: Connection headers
        buffer_config: Buffer configuration
        
    Returns:
        Configured WebSocket client
    """
    connection_config = ConnectionConfig(
        url=url,
        headers=headers or {},
        auth_data=auth_data
    )
    
    manager = get_websocket_manager()
    return manager.create_client(name, connection_config, buffer_config)
"""
WebSocket Connection Management

Connection handling, authentication, and lifecycle management.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union

import websockets

from .types import (
    ConnectionState, ConnectionConfig, ConnectionStats, 
    WebSocketMessage, MessagePriority, LatencyMetrics
)
from .buffering import MessageBuffer

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """WebSocket connection with advanced features"""
    
    def __init__(
        self,
        config: ConnectionConfig,
        buffer: MessageBuffer,
        name: str = "ws_connection"
    ):
        self.config = config
        self.buffer = buffer
        self.name = name
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_start_time: Optional[float] = None
        
        # Authentication
        self.authenticated = False
        self.auth_token: Optional[str] = None
        
        # Reconnection management
        self.reconnect_attempts = 0
        self.current_backoff = self.config.reconnect_backoff
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.error_handlers: List[Callable] = []
        
        # Performance monitoring
        self.latency_metrics = LatencyMetrics()
        self.stats = ConnectionStats()
        
        # Heartbeat
        self.last_ping: Optional[float] = None
        self.last_pong: Optional[float] = None
        self.heartbeat_failures = 0
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        
        # Threading
        self._lock = threading.Lock()
        
        logger.info(f"WebSocket connection '{name}' initialized")
    
    async def connect(self) -> bool:
        """
        Connect to WebSocket server
        
        Returns:
            True if connected successfully
        """
        try:
            self.state = ConnectionState.CONNECTING
            self.connection_start_time = time.time()
            
            logger.info(f"Connecting to {self.config.url}")
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    extra_headers=self.config.headers,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout
                ),
                timeout=self.config.connection_timeout
            )
            
            self.state = ConnectionState.CONNECTED
            self.stats.connected_at = datetime.now()
            
            # Authenticate if needed
            if self.config.auth_data:
                await self._authenticate()
            else:
                self.state = ConnectionState.AUTHENTICATED
                self.authenticated = True
            
            # Start tasks
            await self._start_tasks()
            
            # Reset reconnection state
            self.reconnect_attempts = 0
            self.current_backoff = self.config.reconnect_backoff
            
            connection_time = time.time() - self.connection_start_time
            logger.info(f"WebSocket connected in {connection_time:.3f}s")
            
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.stats.connection_errors += 1
            logger.error(f"Failed to connect: {e}")
            await self._handle_connection_error(e)
            return False
    
    async def _authenticate(self):
        """Authenticate WebSocket connection"""
        if not self.config.auth_data:
            return
        
        try:
            auth_message = json.dumps(self.config.auth_data)
            await self.websocket.send(auth_message)
            
            # Wait for auth response with timeout
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=5.0
            )
            
            # Parse auth response (implementation depends on provider)
            response_data = json.loads(response)
            if response_data.get('status') == 'authenticated':
                self.state = ConnectionState.AUTHENTICATED
                self.authenticated = True
                self.auth_token = response_data.get('token')
                logger.info("WebSocket authenticated successfully")
            else:
                raise Exception(f"Authentication failed: {response_data}")
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        logger.info("Disconnecting WebSocket")
        
        self.state = ConnectionState.DISCONNECTED
        self.authenticated = False
        self.stats.disconnected_at = datetime.now()
        
        # Cancel tasks
        await self._stop_tasks()
        
        # Close connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("WebSocket disconnected")
    
    async def _start_tasks(self):
        """Start background tasks"""
        # Start receive task
        self._receive_task = asyncio.create_task(self._receive_loop())
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start message processing task
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        logger.debug("Background tasks started")
    
    async def _stop_tasks(self):
        """Stop background tasks"""
        tasks = [self._receive_task, self._heartbeat_task, self._processing_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.debug("Background tasks stopped")
    
    async def _receive_loop(self):
        """Main message receiving loop"""
        try:
            while self.websocket and not self.websocket.closed:
                try:
                    # Measure network latency
                    receive_start = time.time()
                    
                    message = await self.websocket.recv()
                    
                    network_time = time.time() - receive_start
                    self.stats.total_messages_received += 1
                    self.stats.total_bytes_received += len(message.encode() if isinstance(message, str) else message)
                    
                    # Create WebSocket message with metadata
                    ws_message = WebSocketMessage(
                        data=message,
                        timestamp=datetime.now(),
                        priority=self._determine_message_priority(message),
                        source=self.name
                    )
                    
                    # Add to buffer
                    if not self.buffer.add_message(ws_message):
                        logger.warning("Message dropped due to buffer overflow")
                    
                    # Update latency metrics
                    self.latency_metrics.add_measurement(
                        rtt=network_time,
                        network_time=network_time
                    )
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    await self._handle_receive_error(e)
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in receive loop: {e}")
        finally:
            await self._cleanup_connection()
    
    def _determine_message_priority(self, message: Union[str, bytes]) -> MessagePriority:
        """Determine message priority based on content"""
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            if isinstance(message, str):
                data = json.loads(message)
                
                # Check message type/content for priority
                msg_type = data.get('type', data.get('T', '')).lower()
                
                if msg_type in ['trade', 't', 'execution', 'fill']:
                    return MessagePriority.CRITICAL
                elif msg_type in ['quote', 'q', 'price', 'ticker']:
                    return MessagePriority.HIGH
                elif msg_type in ['heartbeat', 'ping', 'status']:
                    return MessagePriority.LOW
                else:
                    return MessagePriority.NORMAL
                    
        except Exception:
            # Default to normal priority if we can't parse
            return MessagePriority.NORMAL
    
    async def _processing_loop(self):
        """Process messages from buffer"""
        try:
            while True:
                try:
                    # Get batch of messages
                    batch = await self.buffer.get_batch()
                    
                    if not batch:
                        await asyncio.sleep(0.001)  # Small delay if no messages
                        continue
                    
                    # Process batch
                    processing_start = time.time()
                    await self._process_message_batch(batch)
                    processing_time = time.time() - processing_start
                    
                    # Mark as processed
                    self.buffer.mark_processed(batch)
                    
                    # Update processing time metrics
                    for message in batch:
                        if message.processing_time > 0:
                            self.latency_metrics.add_measurement(
                                rtt=message.processing_time,
                                processing_time=processing_time / len(batch)
                            )
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing message batch: {e}")
                    
        except asyncio.CancelledError:
            logger.debug("Processing loop cancelled")
    
    async def _process_message_batch(self, messages: List[WebSocketMessage]):
        """Process a batch of messages"""
        for message in messages:
            try:
                await self._process_single_message(message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                for handler in self.error_handlers:
                    try:
                        handler(e, message)
                    except Exception as handler_error:
                        logger.error(f"Error in error handler: {handler_error}")
    
    async def _process_single_message(self, message: WebSocketMessage):
        """Process a single message"""
        try:
            # Parse message data
            if isinstance(message.data, str):
                data = json.loads(message.data)
            else:
                data = message.data
            
            # Get message type for routing
            msg_type = data.get('type', data.get('T', 'unknown'))
            
            # Route to appropriate handler
            handler = self.message_handlers.get(msg_type)
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data, message)
                else:
                    handler(data, message)
            else:
                # Default handler
                await self._default_message_handler(data, message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
    
    async def _default_message_handler(self, data: Dict[str, Any], message: WebSocketMessage):
        """Default message handler"""
        logger.debug(f"Received message: {data}")
    
    async def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        try:
            while True:
                await asyncio.sleep(self.config.ping_interval)
                
                if self.websocket and not self.websocket.closed:
                    try:
                        # Send ping and measure response time
                        ping_start = time.time()
                        self.last_ping = ping_start
                        
                        await asyncio.wait_for(
                            self.websocket.ping(),
                            timeout=self.config.ping_timeout
                        )
                        
                        self.last_pong = time.time()
                        ping_time = self.last_pong - ping_start
                        
                        # Update latency metrics
                        self.latency_metrics.add_measurement(ping_time)
                        
                        logger.debug(f"Ping: {ping_time*1000:.1f}ms")
                        
                    except asyncio.TimeoutError:
                        self.heartbeat_failures += 1
                        self.stats.heartbeat_failures += 1
                        logger.warning(f"Ping timeout ({self.heartbeat_failures} failures)")
                        
                        if self.heartbeat_failures >= 3:
                            logger.error("Multiple ping failures, reconnecting...")
                            await self._trigger_reconnection()
                            break
                            
                    except Exception as e:
                        logger.error(f"Heartbeat error: {e}")
                        self.heartbeat_failures += 1
                else:
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
    
    async def send_message(
        self,
        data: Union[str, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Send message with priority and latency tracking
        
        Args:
            data: Message data to send
            priority: Message priority (for future use)
            
        Returns:
            True if sent successfully
        """
        if not self.websocket or self.websocket.closed:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Convert to string if needed
            if isinstance(data, dict):
                message = json.dumps(data)
            else:
                message = str(data)
            
            # Send message
            await self.websocket.send(message)
            
            # Update statistics
            self.stats.total_messages_sent += 1
            self.stats.total_bytes_sent += len(message.encode('utf-8'))
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def register_error_handler(self, handler: Callable):
        """Register error handler"""
        self.error_handlers.append(handler)
        logger.debug("Registered error handler")
    
    async def _handle_connection_error(self, error: Exception):
        """Handle connection errors"""
        logger.error(f"Connection error: {error}")
        self.state = ConnectionState.ERROR
        
        # Notify error handlers
        for handler in self.error_handlers:
            try:
                handler(error, None)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    async def _handle_receive_error(self, error: Exception):
        """Handle receive errors"""
        logger.error(f"Receive error: {error}")
        
        # Notify error handlers
        for handler in self.error_handlers:
            try:
                handler(error, None)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    async def _trigger_reconnection(self):
        """Trigger reconnection attempt"""
        logger.info("Triggering reconnection...")
        self.state = ConnectionState.RECONNECTING
        self.stats.reconnect_count += 1
        
        # This would be handled by the failover system
        # For now, just disconnect
        await self.disconnect()
    
    async def _cleanup_connection(self):
        """Cleanup connection resources"""
        logger.debug("Cleaning up connection resources")
        
        # Stop tasks
        await self._stop_tasks()
        
        # Update state
        if self.state != ConnectionState.DISCONNECTED:
            self.state = ConnectionState.DISCONNECTED
            self.stats.disconnected_at = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return self.stats.to_dict()
    
    def get_latency_summary(self) -> Dict[str, Any]:
        """Get latency summary"""
        return self.latency_metrics.get_summary()
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return (self.websocket and 
                not self.websocket.closed and 
                self.state == ConnectionState.AUTHENTICATED)
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return (self.is_connected() and 
                self.heartbeat_failures < 3 and
                self.latency_metrics.avg_rtt * 1000 < self.latency_metrics.rtt_critical_ms)
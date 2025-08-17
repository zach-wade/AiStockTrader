"""
Failover and Recovery

Automatic failover and recovery functionality for WebSocket connections.
"""

# Standard library imports
import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Any

from .types import ConnectionConfig

logger = logging.getLogger(__name__)


class FailoverManager:
    """Manages failover and recovery for WebSocket connections"""

    def __init__(self, connection, config: ConnectionConfig):
        self.connection = connection
        self.config = config

        # Failover state
        self.failover_urls: list[str] = []
        self.current_url_index = 0
        self.failover_active = False

        # Recovery state
        self.recovery_task: asyncio.Task | None = None
        self.recovery_running = False

        # Statistics
        self.failover_count = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0

        logger.info("FailoverManager initialized")

    def add_failover_url(self, url: str):
        """Add a failover URL"""
        self.failover_urls.append(url)
        logger.info(f"Added failover URL: {url}")

    def set_failover_urls(self, urls: list[str]):
        """Set list of failover URLs"""
        self.failover_urls = urls
        self.current_url_index = 0
        logger.info(f"Set {len(urls)} failover URLs")

    async def connect_with_failover(self) -> bool:
        """Connect with automatic failover"""
        original_url = self.config.url

        try:
            # Try primary URL first
            if await self.connection.connect():
                return True

            # Try failover URLs
            for i, failover_url in enumerate(self.failover_urls):
                logger.info(f"Trying failover URL {i+1}/{len(self.failover_urls)}: {failover_url}")

                # Update config with failover URL
                self.config.url = failover_url
                self.current_url_index = i

                if await self.connection.connect():
                    self.failover_active = True
                    self.failover_count += 1
                    logger.info(f"Connected to failover URL: {failover_url}")
                    return True

            logger.error("All failover attempts failed")
            return False

        except Exception as e:
            logger.error(f"Failover connection failed: {e}")
            return False
        finally:
            # Restore original URL if no failover was successful
            if not self.failover_active:
                self.config.url = original_url

    async def start_recovery_monitor(self):
        """Start monitoring for recovery opportunities"""
        if self.recovery_running:
            return

        self.recovery_running = True
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        logger.info("Recovery monitor started")

    async def stop_recovery_monitor(self):
        """Stop recovery monitoring"""
        self.recovery_running = False

        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass

        logger.info("Recovery monitor stopped")

    async def _recovery_loop(self):
        """Background recovery monitoring"""
        while self.recovery_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Only attempt recovery if we're on failover
                if self.failover_active:
                    await self._attempt_recovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")

    async def _attempt_recovery(self):
        """Attempt to recover to primary connection"""
        if not self.failover_active:
            return

        logger.info("Attempting recovery to primary connection")
        self.recovery_attempts += 1

        # Store current failover URL
        failover_url = self.config.url

        # Try to connect to primary URL
        primary_url = self.failover_urls[0] if self.failover_urls else self.config.url

        try:
            # Test connection to primary
            test_config = ConnectionConfig(
                url=primary_url,
                headers=self.config.headers,
                auth_data=self.config.auth_data,
                connection_timeout=5.0,  # Quick test
            )

            # Create a test connection
            from .buffering import MessageBuffer
            from .connection import WebSocketConnection
            from .types import BufferConfig

            test_buffer = MessageBuffer(BufferConfig())
            test_connection = WebSocketConnection(test_config, test_buffer, "recovery_test")

            # Test primary connection
            if await test_connection.connect():
                await test_connection.disconnect()

                # Primary is available, switch back
                logger.info("Primary connection available, switching back")

                # Disconnect from failover
                await self.connection.disconnect()

                # Update config to primary
                self.config.url = primary_url
                self.current_url_index = 0

                # Reconnect to primary
                if await self.connection.connect():
                    self.failover_active = False
                    self.successful_recoveries += 1
                    logger.info("Successfully recovered to primary connection")
                else:
                    # Recovery failed, switch back to failover
                    self.config.url = failover_url
                    await self.connection.connect()
                    logger.warning("Recovery failed, switched back to failover")
            else:
                logger.debug("Primary connection still not available")

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            # Ensure we're still connected to failover
            if not self.connection.is_connected():
                self.config.url = failover_url
                await self.connection.connect()

    async def force_failover(self) -> bool:
        """Force failover to next available URL"""
        if not self.failover_urls:
            logger.error("No failover URLs configured")
            return False

        # Find next available URL
        start_index = self.current_url_index

        for i in range(len(self.failover_urls)):
            next_index = (start_index + i + 1) % len(self.failover_urls)
            failover_url = self.failover_urls[next_index]

            logger.info(f"Forcing failover to: {failover_url}")

            # Disconnect current connection
            await self.connection.disconnect()

            # Update config
            self.config.url = failover_url
            self.current_url_index = next_index

            # Try to connect
            if await self.connection.connect():
                self.failover_active = True
                self.failover_count += 1
                logger.info(f"Forced failover successful: {failover_url}")
                return True

        logger.error("All forced failover attempts failed")
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get failover statistics"""
        return {
            "failover_active": self.failover_active,
            "current_url": self.config.url,
            "current_url_index": self.current_url_index,
            "failover_count": self.failover_count,
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failover_urls_count": len(self.failover_urls),
            "recovery_running": self.recovery_running,
        }

    def reset_stats(self):
        """Reset failover statistics"""
        self.failover_count = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        logger.info("Failover statistics reset")

    async def stop(self):
        """Stop failover manager"""
        await self.stop_recovery_monitor()
        logger.info("FailoverManager stopped")


@asynccontextmanager
async def websocket_context(
    name: str,
    url: str,
    auth_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    failover_urls: list[str] | None = None,
):
    """Context manager for WebSocket connections with failover"""
    from .optimizer import create_optimized_websocket

    client = await create_optimized_websocket(name, url, auth_data, headers)

    # Set up failover if URLs provided
    if failover_urls:
        client.failover.set_failover_urls(failover_urls)

    try:
        await client.connect()

        # Start recovery monitoring if on failover
        if client.failover.failover_active:
            await client.failover.start_recovery_monitor()

        yield client

    finally:
        await client.close()


class ConnectionPool:
    """Pool of WebSocket connections for load balancing"""

    def __init__(self, name: str):
        self.name = name
        self.connections: list = []
        self.current_index = 0
        self.active_connections = 0

        logger.info(f"ConnectionPool '{name}' initialized")

    def add_connection(self, connection):
        """Add connection to pool"""
        self.connections.append(connection)
        if connection.is_connected():
            self.active_connections += 1

        logger.info(f"Added connection to pool '{self.name}' (total: {len(self.connections)})")

    def get_next_connection(self):
        """Get next connection using round-robin"""
        if not self.connections:
            return None

        # Find next healthy connection
        start_index = self.current_index

        for i in range(len(self.connections)):
            index = (start_index + i) % len(self.connections)
            connection = self.connections[index]

            if connection.is_healthy():
                self.current_index = (index + 1) % len(self.connections)
                return connection

        # No healthy connections, return any connection
        if self.connections:
            self.current_index = (self.current_index + 1) % len(self.connections)
            return self.connections[self.current_index - 1]

        return None

    def get_healthy_connections(self) -> list:
        """Get list of healthy connections"""
        return [conn for conn in self.connections if conn.is_healthy()]

    def get_pool_stats(self) -> dict[str, Any]:
        """Get pool statistics"""
        healthy_count = len(self.get_healthy_connections())

        return {
            "name": self.name,
            "total_connections": len(self.connections),
            "healthy_connections": healthy_count,
            "active_connections": self.active_connections,
            "health_ratio": healthy_count / len(self.connections) if self.connections else 0,
        }

    async def close_all(self):
        """Close all connections in pool"""
        tasks = []

        for connection in self.connections:
            tasks.append(asyncio.create_task(connection.close()))

        await asyncio.gather(*tasks, return_exceptions=True)

        self.connections.clear()
        self.active_connections = 0

        logger.info(f"All connections in pool '{self.name}' closed")

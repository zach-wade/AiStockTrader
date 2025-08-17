"""
Message Buffering

High-performance message buffering with adaptive batching and priority handling.
"""

# Standard library imports
import asyncio
from collections import deque
import heapq
import json
import logging
import threading
import time
from typing import Any

from .types import BufferConfig, MessagePriority, WebSocketMessage

logger = logging.getLogger(__name__)


class MessageBuffer:
    """High-performance message buffer with adaptive batching"""

    def __init__(self, config: BufferConfig):
        self.config = config

        # Message storage
        self.buffer: deque = deque(maxlen=config.max_buffer_size)
        self.priority_queue: list[WebSocketMessage] = []

        # Batching
        self.current_batch: list[WebSocketMessage] = []
        self.batch_start_time: float | None = None

        # Statistics
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_dropped = 0
        self.bytes_received = 0
        self.batch_count = 0

        # Threading
        self._lock = threading.Lock()

        # Auto-flush task
        self._flush_task: asyncio.Task | None = None
        self._running = False

    def add_message(self, message: WebSocketMessage) -> bool:
        """
        Add message to buffer with priority handling

        Args:
            message: Message to add

        Returns:
            True if added successfully, False if dropped
        """
        with self._lock:
            self.messages_received += 1

            # Calculate message size
            if isinstance(message.data, str):
                size = len(message.data.encode("utf-8"))
            elif isinstance(message.data, bytes):
                size = len(message.data)
            else:
                size = len(json.dumps(message.data).encode("utf-8"))

            self.bytes_received += size

            # Check memory limits (simplified check)
            if self.bytes_received > self.config.memory_limit_mb * 1024 * 1024:
                logger.warning("Buffer memory limit exceeded")
                self._drop_oldest_messages(0.1)  # Drop 10% of oldest messages

            # Add to appropriate queue based on priority
            if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
                if len(self.priority_queue) < self.config.priority_queue_size:
                    heapq.heappush(self.priority_queue, message)
                elif (
                    self.priority_queue
                    and message.priority.value < self.priority_queue[0].priority.value
                ):
                    heapq.heapreplace(self.priority_queue, message)
                    self.messages_dropped += 1
                else:
                    self.messages_dropped += 1
                    return False
            else:
                # Normal/low priority messages go to regular buffer
                if len(self.buffer) >= self.config.max_buffer_size:
                    self.messages_dropped += 1
                    return False

                self.buffer.append(message)

            return True

    def _drop_oldest_messages(self, fraction: float):
        """Drop a fraction of oldest messages to free memory"""
        with self._lock:
            messages_to_drop = int(len(self.buffer) * fraction)
            for _ in range(min(messages_to_drop, len(self.buffer))):
                self.buffer.popleft()
                self.messages_dropped += 1

    async def get_batch(self, max_size: int | None = None) -> list[WebSocketMessage]:
        """
        Get a batch of messages for processing

        Args:
            max_size: Maximum batch size (uses config default if None)

        Returns:
            List of messages to process
        """
        batch_size = max_size or self.config.batch_size
        batch = []

        with self._lock:
            # First, get all priority messages
            while self.priority_queue and len(batch) < batch_size:
                message = heapq.heappop(self.priority_queue)
                message.processing_start = time.time()
                batch.append(message)

            # Fill remaining batch with normal messages
            while self.buffer and len(batch) < batch_size:
                message = self.buffer.popleft()
                message.processing_start = time.time()
                batch.append(message)

        if batch:
            self.batch_count += 1
            logger.debug(f"Retrieved batch of {len(batch)} messages")

        return batch

    def mark_processed(self, messages: list[WebSocketMessage]):
        """Mark messages as processed"""
        current_time = time.time()
        with self._lock:
            for message in messages:
                message.processing_end = current_time
                self.messages_processed += 1

    async def start_auto_flush(self):
        """Start automatic buffer flushing"""
        self._running = True
        self._flush_task = asyncio.create_task(self._auto_flush_loop())

    async def stop_auto_flush(self):
        """Stop automatic buffer flushing"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

    async def _auto_flush_loop(self):
        """Automatic flush loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_ms / 1000.0)

                # Check if we should flush
                with self._lock:
                    should_flush = (
                        len(self.buffer) >= self.config.batch_size
                        or len(self.priority_queue) > 0
                        or (
                            self.batch_start_time
                            and time.time() - self.batch_start_time
                            > self.config.flush_interval_ms / 1000.0
                        )
                    )

                if should_flush:
                    # This would trigger batch processing in the parent
                    logger.debug("Auto-flush triggered")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-flush loop: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_dropped": self.messages_dropped,
                "bytes_received": self.bytes_received,
                "batch_count": self.batch_count,
                "buffer_size": len(self.buffer),
                "priority_queue_size": len(self.priority_queue),
                "drop_rate": self.messages_dropped / max(self.messages_received, 1) * 100,
                "processing_rate": self.messages_processed / max(self.messages_received, 1) * 100,
            }

    def clear(self):
        """Clear all buffered messages"""
        with self._lock:
            self.buffer.clear()
            self.priority_queue.clear()
            self.current_batch.clear()
            logger.info("Message buffer cleared")

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return len(self.buffer) == 0 and len(self.priority_queue) == 0

    def get_pending_count(self) -> int:
        """Get total number of pending messages"""
        with self._lock:
            return len(self.buffer) + len(self.priority_queue)

    def set_memory_limit(self, limit_mb: float):
        """Update memory limit"""
        with self._lock:
            self.config.memory_limit_mb = limit_mb
            logger.info(f"Buffer memory limit updated to {limit_mb}MB")

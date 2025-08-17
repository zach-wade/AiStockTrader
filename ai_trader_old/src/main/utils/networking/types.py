"""
Networking Types

Data classes and enums for WebSocket networking.
"""

# Standard library imports
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from typing import Any


class ConnectionState(Enum):
    """WebSocket connection states"""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    AUTHENTICATED = "AUTHENTICATED"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


class MessagePriority(Enum):
    """Message priority levels"""

    CRITICAL = 1  # Market data, trades
    HIGH = 2  # Quotes, order updates
    NORMAL = 3  # General messages
    LOW = 4  # Heartbeats, status updates


@dataclass
class LatencyMetrics:
    """Latency tracking metrics"""

    round_trip_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    network_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Current statistics
    avg_rtt: float = 0.0
    p95_rtt: float = 0.0
    p99_rtt: float = 0.0
    max_rtt: float = 0.0
    min_rtt: float = float("inf")

    # Alert thresholds
    rtt_warning_ms: float = 50.0
    rtt_critical_ms: float = 100.0

    def add_measurement(self, rtt: float, processing_time: float = 0.0, network_time: float = 0.0):
        """Add latency measurement"""
        self.round_trip_times.append(rtt)
        if processing_time > 0:
            self.processing_times.append(processing_time)
        if network_time > 0:
            self.network_times.append(network_time)

        # Update statistics
        self._update_stats()

    def _update_stats(self):
        """Update latency statistics"""
        if not self.round_trip_times:
            return

        rtts = list(self.round_trip_times)
        self.avg_rtt = statistics.mean(rtts)
        self.max_rtt = max(rtts)
        self.min_rtt = min(rtts)

        if len(rtts) >= 20:  # Need sufficient samples for percentiles
            sorted_rtts = sorted(rtts)
            self.p95_rtt = sorted_rtts[int(len(sorted_rtts) * 0.95)]
            self.p99_rtt = sorted_rtts[int(len(sorted_rtts) * 0.99)]

    def get_summary(self) -> dict[str, Any]:
        """Get latency summary"""
        return {
            "avg_rtt_ms": self.avg_rtt * 1000,
            "p95_rtt_ms": self.p95_rtt * 1000,
            "p99_rtt_ms": self.p99_rtt * 1000,
            "max_rtt_ms": self.max_rtt * 1000,
            "min_rtt_ms": self.min_rtt * 1000,
            "samples_count": len(self.round_trip_times),
            "avg_processing_ms": (
                statistics.mean(self.processing_times) * 1000 if self.processing_times else 0
            ),
            "high_latency_alerts": self.avg_rtt * 1000 > self.rtt_warning_ms,
        }


@dataclass
class BufferConfig:
    """Configuration for message buffering"""

    max_buffer_size: int = 10000  # Maximum messages in buffer
    batch_size: int = 100  # Messages per batch processing
    flush_interval_ms: float = 10.0  # Auto-flush interval
    priority_queue_size: int = 1000  # Priority queue size
    compression_threshold: int = 5000  # Compress when buffer exceeds this
    memory_limit_mb: float = 50.0  # Memory limit for buffers


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration"""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth_data: dict[str, Any] | None = None
    ping_interval: float = 30.0  # Heartbeat interval
    ping_timeout: float = 10.0  # Ping timeout
    max_reconnect_attempts: int = 10  # Max reconnection attempts
    reconnect_backoff: float = 1.0  # Initial backoff time
    max_backoff: float = 60.0  # Maximum backoff time
    connection_timeout: float = 10.0  # Connection timeout


@dataclass
class WebSocketMessage:
    """Enhanced WebSocket message with metadata"""

    data: str | bytes | dict[str, Any]
    timestamp: datetime
    priority: MessagePriority
    message_id: str | None = None
    source: str | None = None
    processing_start: float | None = None
    processing_end: float | None = None

    @property
    def processing_time(self) -> float:
        """Calculate processing time"""
        if self.processing_start and self.processing_end:
            return self.processing_end - self.processing_start
        return 0.0

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value


@dataclass
class ConnectionStats:
    """Connection statistics"""

    connected_at: datetime | None = None
    disconnected_at: datetime | None = None
    reconnect_count: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    heartbeat_failures: int = 0
    connection_errors: int = 0

    @property
    def uptime_seconds(self) -> float:
        """Calculate connection uptime"""
        if self.connected_at:
            end_time = self.disconnected_at or datetime.now()
            return (end_time - self.connected_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "disconnected_at": self.disconnected_at.isoformat() if self.disconnected_at else None,
            "uptime_seconds": self.uptime_seconds,
            "reconnect_count": self.reconnect_count,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "heartbeat_failures": self.heartbeat_failures,
            "connection_errors": self.connection_errors,
        }

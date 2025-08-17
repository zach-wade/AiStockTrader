"""
Enhanced Monitoring System

This module provides enhanced monitoring capabilities with database persistence,
thresholds, and advanced aggregation features while maintaining backward
compatibility with the existing monitoring interface.
"""

# Standard library imports
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import statistics
from typing import Any

# Local imports
from main.utils.core import get_logger
from main.utils.database import DatabasePool

from .types import Alert, AlertLevel, MetricType

# AlertManager import removed to avoid circular dependency
# If needed, pass AlertManager instance via constructor

logger = get_logger(__name__)


@dataclass
class EnhancedMetricDefinition:
    """Enhanced metric definition with thresholds and persistence settings."""

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    # Alert thresholds
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    threshold_operator: str = "gt"  # gt, lt, eq, ne

    # Persistence settings
    persist_to_db: bool = True
    retention_hours: int = 168  # 7 days default

    # Aggregation settings
    default_aggregation: str = "avg"  # avg, sum, min, max, count
    collection_interval: int = 60  # seconds


@dataclass
class AggregatedMetric:
    """Aggregated metric result."""

    name: str
    value: float
    aggregation_type: str
    period_start: datetime
    period_end: datetime
    sample_count: int
    tags: dict[str, str] = field(default_factory=dict)


class EnhancedMonitor:
    """
    Enhanced monitoring system with database persistence and advanced features.

    This class provides:
    - Database persistence for metrics
    - Threshold-based alerting
    - Metric aggregation and queries
    - Backward compatibility with existing monitoring
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        alert_manager: Any | None = None,  # Type: AlertManager
        history_size: int = 10000,
        enable_persistence: bool = True,
    ):
        """
        Initialize enhanced monitor.

        Args:
            db_pool: Optional database pool for persistence
            alert_manager: Optional alert manager (creates one if not provided)
            history_size: Size of in-memory metric history
            enable_persistence: Whether to persist metrics to database
        """
        self.db_pool = db_pool
        self.alert_manager = alert_manager  # No default creation to avoid circular import
        self.history_size = history_size
        self.enable_persistence = enable_persistence and db_pool is not None

        # Metric definitions
        self._metric_definitions: dict[str, EnhancedMetricDefinition] = {}

        # In-memory storage (always maintained for fast access)
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._gauges: dict[str, float] = {}

        # Alert state
        self._active_alerts: dict[str, Alert] = {}

        # Background tasks
        self._persistence_task: asyncio.Task | None = None
        self._aggregation_task: asyncio.Task | None = None
        self._is_running = False

        # Persistence queue
        self._persistence_queue: asyncio.Queue = asyncio.Queue()

        # Register common metrics
        self._register_common_metrics()

        logger.info(
            f"Enhanced monitor initialized "
            f"(persistence={'enabled' if self.enable_persistence else 'disabled'})"
        )

    def register_metric(self, definition: EnhancedMetricDefinition) -> None:
        """Register a metric definition."""
        self._metric_definitions[definition.name] = definition
        logger.debug(f"Registered metric: {definition.name}")

    def record_metric(
        self,
        name: str,
        value: float | int,
        metric_type: MetricType | None = None,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record a metric value.

        This method is compatible with the existing monitoring interface.
        """
        timestamp = timestamp or datetime.now(UTC)
        tags = tags or {}

        # Auto-register if not registered
        if name not in self._metric_definitions:
            self._auto_register_metric(name, metric_type)

        definition = self._metric_definitions[name]

        # Store in memory
        metric_data = {"timestamp": timestamp, "value": float(value), "tags": tags}

        if definition.metric_type == MetricType.GAUGE:
            self._gauges[name] = float(value)
        else:
            self._metrics[name].append(metric_data)

        # Check thresholds
        self._check_thresholds(name, value, definition)

        # Queue for persistence if enabled
        if self.enable_persistence and definition.persist_to_db:
            asyncio.create_task(self._queue_for_persistence(name, metric_data))

    async def get_metric_value(
        self,
        name: str,
        aggregation: str = "last",
        period_minutes: int = 5,
        tags: dict[str, str] | None = None,
    ) -> float | None:
        """Get aggregated metric value."""
        if aggregation == "last":
            # For gauges, return current value
            if name in self._gauges:
                return self._gauges[name]

            # For other metrics, return last value
            if self._metrics.get(name):
                return self._metrics[name][-1]["value"]

            return None

        # Get values within time period
        cutoff = datetime.now(UTC) - timedelta(minutes=period_minutes)
        values = []

        for metric in self._metrics.get(name, []):
            if metric["timestamp"] > cutoff:
                if tags:
                    # Check if tags match
                    if all(metric["tags"].get(k) == v for k, v in tags.items()):
                        values.append(metric["value"])
                else:
                    values.append(metric["value"])

        if not values:
            return None

        # Aggregate
        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            return None

    async def get_metric_series(
        self, name: str, period_minutes: int = 60, tags: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Get time series data for a metric."""
        cutoff = datetime.now(UTC) - timedelta(minutes=period_minutes)
        series = []

        # Get from memory first
        for metric in self._metrics.get(name, []):
            if metric["timestamp"] > cutoff:
                if tags is None or all(metric["tags"].get(k) == v for k, v in tags.items()):
                    series.append(
                        {
                            "timestamp": metric["timestamp"].isoformat(),
                            "value": metric["value"],
                            "tags": metric["tags"],
                        }
                    )

        # If persistence enabled and not enough data, query database
        if self.enable_persistence and len(series) < 10:
            db_series = await self._query_metric_series(name, period_minutes, tags)
            series.extend(db_series)

        return sorted(series, key=lambda x: x["timestamp"])

    async def get_aggregated_metrics(
        self,
        names: list[str],
        aggregation: str = "avg",
        period_minutes: int = 60,
        group_by_interval: int | None = None,  # minutes
    ) -> list[AggregatedMetric]:
        """Get aggregated metrics for multiple metric names."""
        results = []

        for name in names:
            if group_by_interval:
                # Get time-bucketed aggregations
                buckets = await self._get_time_buckets(
                    name, period_minutes, group_by_interval, aggregation
                )
                results.extend(buckets)
            else:
                # Single aggregation for the period
                value = await self.get_metric_value(name, aggregation, period_minutes)
                if value is not None:
                    now = datetime.now(UTC)
                    results.append(
                        AggregatedMetric(
                            name=name,
                            value=value,
                            aggregation_type=aggregation,
                            period_start=now - timedelta(minutes=period_minutes),
                            period_end=now,
                            sample_count=len(self._metrics.get(name, [])),
                        )
                    )

        return results

    def get_active_alerts(self) -> list[Alert]:
        """Get currently active alerts."""
        return list(self._active_alerts.values())

    def get_metric_definitions(self) -> dict[str, EnhancedMetricDefinition]:
        """Get all registered metric definitions."""
        return self._metric_definitions.copy()

    async def start(self):
        """Start background tasks."""
        if self._is_running:
            return

        self._is_running = True

        if self.enable_persistence:
            self._persistence_task = asyncio.create_task(self._persistence_loop())
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())

        logger.info("Enhanced monitor started")

    async def stop(self):
        """Stop background tasks."""
        self._is_running = False

        # Cancel tasks
        for task in [self._persistence_task, self._aggregation_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Flush persistence queue
        if self.enable_persistence:
            await self._flush_persistence_queue()

        logger.info("Enhanced monitor stopped")

    def _auto_register_metric(self, name: str, metric_type: MetricType | None = None):
        """Auto-register a metric based on name patterns."""
        # Infer metric type
        if metric_type is None:
            if any(pattern in name for pattern in ["count", "total", "requests"]):
                metric_type = MetricType.COUNTER
            elif any(pattern in name for pattern in ["duration", "latency", "time"]):
                metric_type = MetricType.TIMER
            elif any(pattern in name for pattern in ["rate", "per_second"]):
                metric_type = MetricType.RATE
            else:
                metric_type = MetricType.GAUGE

        # Infer thresholds based on common patterns
        warning_threshold = None
        critical_threshold = None
        operator = "gt"

        if "error" in name or "failure" in name:
            warning_threshold = 5
            critical_threshold = 10
        elif "cpu" in name:
            warning_threshold = 70
            critical_threshold = 90
        elif "memory" in name:
            warning_threshold = 80
            critical_threshold = 95
        elif "latency" in name or "duration" in name:
            warning_threshold = 1000  # 1 second
            critical_threshold = 5000  # 5 seconds

        definition = EnhancedMetricDefinition(
            name=name,
            metric_type=metric_type,
            description=f"Auto-registered metric: {name}",
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            threshold_operator=operator,
        )

        self.register_metric(definition)

    def _check_thresholds(self, name: str, value: float, definition: EnhancedMetricDefinition):
        """Check if metric value exceeds thresholds."""
        alert_level = None
        threshold = None

        # Check critical threshold first
        if definition.critical_threshold is not None:
            if self._compare_threshold(
                value, definition.critical_threshold, definition.threshold_operator
            ):
                alert_level = AlertLevel.CRITICAL
                threshold = definition.critical_threshold
        # Then warning threshold
        elif definition.warning_threshold is not None:
            if self._compare_threshold(
                value, definition.warning_threshold, definition.threshold_operator
            ):
                alert_level = AlertLevel.WARNING
                threshold = definition.warning_threshold

        if alert_level:
            # Create or update alert
            alert = Alert(
                message=f"Metric {name} = {value:.2f} exceeds {alert_level.value} threshold ({threshold})",
                level=alert_level,
                source=f"metric:{name}",
                metric_name=name,
                metric_value=value,
                threshold=threshold,
            )

            self._active_alerts[name] = alert
            self.alert_manager.add_alert(alert)
        elif name in self._active_alerts:
            del self._active_alerts[name]

    def _compare_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Compare value against threshold using operator."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "ne":
            return value != threshold
        else:
            return False

    def _register_common_metrics(self):
        """Register commonly used metrics."""
        common_metrics = [
            EnhancedMetricDefinition(
                name="system.cpu_usage",
                metric_type=MetricType.GAUGE,
                description="CPU usage percentage",
                unit="%",
                warning_threshold=70,
                critical_threshold=90,
            ),
            EnhancedMetricDefinition(
                name="system.memory_usage",
                metric_type=MetricType.GAUGE,
                description="Memory usage in MB",
                unit="MB",
                warning_threshold=80,
                critical_threshold=95,
            ),
            EnhancedMetricDefinition(
                name="api.request_count",
                metric_type=MetricType.COUNTER,
                description="API request count",
            ),
            EnhancedMetricDefinition(
                name="api.request_duration",
                metric_type=MetricType.TIMER,
                description="API request duration",
                unit="ms",
                warning_threshold=1000,
                critical_threshold=5000,
            ),
            EnhancedMetricDefinition(
                name="api.error_rate",
                metric_type=MetricType.RATE,
                description="API error rate",
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
            ),
        ]

        for metric in common_metrics:
            self.register_metric(metric)

    async def _queue_for_persistence(self, name: str, metric_data: dict[str, Any]):
        """Queue metric for persistence."""
        await self._persistence_queue.put((name, metric_data))

    async def _persistence_loop(self):
        """Background loop for persisting metrics to database."""
        batch = []
        last_flush = datetime.now(UTC)

        while self._is_running:
            try:
                # Collect metrics for batch insert
                timeout = 1.0  # 1 second timeout
                deadline = datetime.now(UTC) + timedelta(seconds=timeout)

                while datetime.now(UTC) < deadline and len(batch) < 1000:
                    try:
                        remaining = (deadline - datetime.now(UTC)).total_seconds()
                        if remaining <= 0:
                            break

                        name, data = await asyncio.wait_for(
                            self._persistence_queue.get(), timeout=remaining
                        )
                        batch.append((name, data))
                    except TimeoutError:
                        break

                # Persist batch if we have data or it's been too long
                if batch or (datetime.now(UTC) - last_flush).seconds > 10:
                    if batch:
                        await self._persist_batch(batch)
                        batch.clear()
                    last_flush = datetime.now(UTC)

            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
                await asyncio.sleep(1)

    async def _persist_batch(self, batch: list[tuple[str, dict[str, Any]]]):
        """Persist a batch of metrics to database."""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Create metrics table if not exists
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        tags JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
                    ON metrics(name, timestamp DESC);
                """
                )

                # Batch insert
                values = []
                for name, data in batch:
                    values.append(
                        (name, data["value"], data["timestamp"], json.dumps(data["tags"]))
                    )

                await conn.executemany(
                    """
                    INSERT INTO metrics (name, value, timestamp, tags)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    values,
                )

                logger.debug(f"Persisted {len(batch)} metrics to database")

        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")

    async def _aggregation_loop(self):
        """Background loop for creating metric aggregations."""
        while self._is_running:
            try:
                # Run aggregation every 5 minutes
                await asyncio.sleep(300)
                await self._create_aggregations()
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")

    async def _create_aggregations(self):
        """Create metric aggregations in database."""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Create aggregations table if not exists
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metric_aggregations (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        aggregation_type VARCHAR(50) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        period_start TIMESTAMPTZ NOT NULL,
                        period_end TIMESTAMPTZ NOT NULL,
                        sample_count INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_aggregations_name_period
                    ON metric_aggregations(name, period_end DESC);
                """
                )

                # Create hourly aggregations for each metric
                for name, definition in self._metric_definitions.items():
                    if definition.persist_to_db:
                        await self._create_metric_aggregation(conn, name, definition)

        except Exception as e:
            logger.error(f"Error creating aggregations: {e}")

    async def _create_metric_aggregation(
        self, conn, name: str, definition: EnhancedMetricDefinition
    ):
        """Create aggregation for a specific metric."""
        # Get last aggregation time
        row = await conn.fetchrow(
            """
            SELECT MAX(period_end) as last_end
            FROM metric_aggregations
            WHERE name = $1
            """,
            name,
        )

        last_end = (
            row["last_end"] if row and row["last_end"] else datetime.now(UTC) - timedelta(hours=24)
        )

        # Create hourly aggregations up to current hour
        current_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

        while last_end < current_hour:
            period_start = last_end
            period_end = period_start + timedelta(hours=1)

            # Get metrics for this period
            rows = await conn.fetch(
                """
                SELECT value
                FROM metrics
                WHERE name = $1
                AND timestamp >= $2
                AND timestamp < $3
                """,
                name,
                period_start,
                period_end,
            )

            if rows:
                values = [row["value"] for row in rows]

                # Calculate aggregations
                aggregations = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "sum": sum(values),
                    "count": len(values),
                }

                # Insert aggregations
                for agg_type, value in aggregations.items():
                    await conn.execute(
                        """
                        INSERT INTO metric_aggregations
                        (name, aggregation_type, value, period_start, period_end, sample_count)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        name,
                        agg_type,
                        value,
                        period_start,
                        period_end,
                        len(values),
                    )

            last_end = period_end

    async def _query_metric_series(
        self, name: str, period_minutes: int, tags: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Query metric series from database."""
        if not self.db_pool:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                cutoff = datetime.now(UTC) - timedelta(minutes=period_minutes)

                if tags:
                    rows = await conn.fetch(
                        """
                        SELECT timestamp, value, tags
                        FROM metrics
                        WHERE name = $1
                        AND timestamp > $2
                        AND tags @> $3::jsonb
                        ORDER BY timestamp
                        """,
                        name,
                        cutoff,
                        json.dumps(tags),
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT timestamp, value, tags
                        FROM metrics
                        WHERE name = $1
                        AND timestamp > $2
                        ORDER BY timestamp
                        """,
                        name,
                        cutoff,
                    )

                return [
                    {
                        "timestamp": row["timestamp"].isoformat(),
                        "value": row["value"],
                        "tags": json.loads(row["tags"]) if row["tags"] else {},
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error querying metric series: {e}")
            return []

    async def _get_time_buckets(
        self, name: str, period_minutes: int, bucket_minutes: int, aggregation: str
    ) -> list[AggregatedMetric]:
        """Get time-bucketed aggregations."""
        buckets = []
        now = datetime.now(UTC)
        period_start = now - timedelta(minutes=period_minutes)

        # Create buckets
        bucket_start = period_start
        while bucket_start < now:
            bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
            if bucket_end > now:
                bucket_end = now

            # Get values in this bucket
            values = []
            for metric in self._metrics.get(name, []):
                if bucket_start <= metric["timestamp"] < bucket_end:
                    values.append(metric["value"])

            if values:
                # Calculate aggregation
                if aggregation == "sum":
                    value = sum(values)
                elif aggregation == "avg":
                    value = statistics.mean(values)
                elif aggregation == "min":
                    value = min(values)
                elif aggregation == "max":
                    value = max(values)
                elif aggregation == "count":
                    value = len(values)
                else:
                    continue

                buckets.append(
                    AggregatedMetric(
                        name=name,
                        value=value,
                        aggregation_type=aggregation,
                        period_start=bucket_start,
                        period_end=bucket_end,
                        sample_count=len(values),
                    )
                )

            bucket_start = bucket_end

        return buckets

    async def _flush_persistence_queue(self):
        """Flush any remaining metrics in the persistence queue."""
        batch = []

        while not self._persistence_queue.empty():
            try:
                name, data = self._persistence_queue.get_nowait()
                batch.append((name, data))
            except asyncio.QueueEmpty:
                break

        if batch:
            await self._persist_batch(batch)

    def cleanup_old_metrics(self, retention_hours: int | None = None):
        """Clean up old metrics from memory and database."""
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours or 168)

        # Clean memory
        for name, metrics in self._metrics.items():
            # Remove old metrics
            while metrics and metrics[0]["timestamp"] < cutoff:
                metrics.popleft()

        # Schedule database cleanup
        if self.enable_persistence:
            asyncio.create_task(self._cleanup_database_metrics(cutoff))

    async def _cleanup_database_metrics(self, cutoff: datetime):
        """Clean up old metrics from database."""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Delete old metrics
                deleted = await conn.execute("DELETE FROM metrics WHERE timestamp < $1", cutoff)

                # Delete old aggregations
                deleted_agg = await conn.execute(
                    "DELETE FROM metric_aggregations WHERE period_end < $1", cutoff
                )

                logger.info(
                    f"Cleaned up old metrics: {deleted} metrics, " f"{deleted_agg} aggregations"
                )

        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")


# Global enhanced monitor instance
_enhanced_monitor: EnhancedMonitor | None = None


def get_enhanced_monitor(
    db_pool: DatabasePool | None = None, create_if_missing: bool = True
) -> EnhancedMonitor | None:
    """Get or create the global enhanced monitor instance."""
    global _enhanced_monitor

    if _enhanced_monitor is None and create_if_missing:
        _enhanced_monitor = EnhancedMonitor(db_pool=db_pool)

    return _enhanced_monitor


def set_enhanced_monitor(monitor: EnhancedMonitor) -> None:
    """Set the global enhanced monitor instance."""
    global _enhanced_monitor
    _enhanced_monitor = monitor

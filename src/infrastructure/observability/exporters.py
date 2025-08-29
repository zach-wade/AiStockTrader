"""
Observability Exporters for AI Trading System

Exporters for observability data to various systems:
- Prometheus metrics export
- OTLP (OpenTelemetry Protocol) export
- Custom format exporters
- File and console exporters
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, write_to_textfile

from .collector import ObservabilityEvent

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of export operation."""

    success: bool
    events_exported: int
    error: str | None = None
    export_duration_ms: float = 0.0
    metadata: dict[str, Any] | None = None


class ObservabilityExporter(ABC):
    """Abstract base class for observability exporters."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._export_count = 0
        self._last_export_time: float | None = None
        self._error_count = 0

    @abstractmethod
    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export observability events."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get exporter statistics."""
        return {
            "name": self.name,
            "export_count": self._export_count,
            "last_export_time": self._last_export_time,
            "error_count": self._error_count,
        }

    def _record_export(self, result: ExportResult) -> None:
        """Record export attempt."""
        self._export_count += 1
        self._last_export_time = time.time()
        if not result.success:
            self._error_count += 1


class ConsoleExporter(ObservabilityExporter):
    """Export events to console output."""

    def __init__(self, name: str = "console", format_json: bool = True) -> None:
        super().__init__(name)
        self.format_json = format_json

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events to console."""
        start_time = time.perf_counter()

        try:
            for event in events:
                if self.format_json:
                    print(json.dumps(event.to_dict(), indent=2))
                else:
                    print(
                        f"[{event.timestamp}] {event.event_type}.{event.operation}: {event.status}"
                    )

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=True, events_exported=len(events), export_duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result


class FileExporter(ObservabilityExporter):
    """Export events to file."""

    def __init__(
        self,
        name: str = "file",
        file_path: str = "observability_events.jsonl",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        rotate_files: bool = True,
    ):
        super().__init__(name)
        self.file_path = Path(file_path)
        self.max_file_size = max_file_size
        self.rotate_files = rotate_files

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events to file."""
        start_time = time.perf_counter()

        try:
            # Check if file rotation is needed
            if self.rotate_files and self._should_rotate_file():
                self._rotate_file()

            # Write events to file (JSONL format)
            with open(self.file_path, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + "\n")

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=True, events_exported=len(events), export_duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result

    def _should_rotate_file(self) -> bool:
        """Check if file should be rotated."""
        if not self.file_path.exists():
            return False

        return self.file_path.stat().st_size > self.max_file_size

    def _rotate_file(self) -> None:
        """Rotate log file."""
        if not self.file_path.exists():
            return

        timestamp = int(time.time())
        backup_path = self.file_path.with_suffix(f".{timestamp}.jsonl")
        self.file_path.rename(backup_path)

        logger.info(f"Rotated observability file to {backup_path}")


class PrometheusExporter(ObservabilityExporter):
    """Export metrics to Prometheus format."""

    def __init__(
        self,
        name: str = "prometheus",
        metrics_file: str | None = None,
        registry: CollectorRegistry | None = None,
    ):
        super().__init__(name)
        self.metrics_file = metrics_file
        self.registry = registry or CollectorRegistry()

        # Create Prometheus metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self.event_counter = Counter(
            "trading_events_total",
            "Total number of trading events",
            ["event_type", "operation", "status"],
            registry=self.registry,
        )

        self.event_duration = Histogram(
            "trading_event_duration_seconds",
            "Duration of trading events",
            ["event_type", "operation"],
            registry=self.registry,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.active_orders = Gauge(
            "trading_active_orders", "Number of active orders", registry=self.registry
        )

        self.portfolio_value = Gauge(
            "trading_portfolio_value_usd",
            "Portfolio value in USD",
            ["portfolio_id"],
            registry=self.registry,
        )

        self.error_rate = Gauge(
            "trading_error_rate",
            "Error rate by operation type",
            ["event_type", "operation"],
            registry=self.registry,
        )

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events as Prometheus metrics."""
        start_time = time.perf_counter()

        try:
            # Track event metrics
            event_counts: dict[tuple[str, str], int] = {}
            error_counts: dict[tuple[str, str], int] = {}

            for event in events:
                # Count events
                labels = [event.event_type, event.operation, event.status]
                self.event_counter.labels(*labels).inc()

                # Track for error rate calculation
                key = (event.event_type, event.operation)
                event_counts[key] = event_counts.get(key, 0) + 1
                if event.status == "error":
                    error_counts[key] = error_counts.get(key, 0) + 1

                # Duration metrics
                if event.duration_ms is not None:
                    duration_seconds = event.duration_ms / 1000
                    self.event_duration.labels(event.event_type, event.operation).observe(
                        duration_seconds
                    )

                # Business-specific metrics
                if event.event_type == "order" and event.status == "success":
                    if event.operation == "submitted":
                        # This would track active orders in a real implementation
                        pass

                if event.event_type == "portfolio" and "value" in event.metrics:
                    portfolio_id = event.portfolio_id or "default"
                    value = event.metrics["value"]
                    if isinstance(value, (int, float)):
                        self.portfolio_value.labels(portfolio_id).set(float(value))

            # Update error rates
            for key, total_count in event_counts.items():
                error_count = error_counts.get(key, 0)
                error_rate = error_count / total_count if total_count > 0 else 0
                self.error_rate.labels(*key).set(error_rate)

            # Write to file if configured
            if self.metrics_file:
                write_to_textfile(self.metrics_file, self.registry)

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=True, events_exported=len(events), export_duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result


class OTLPExporter(ObservabilityExporter):
    """Export events via OpenTelemetry Protocol."""

    def __init__(
        self,
        name: str = "otlp",
        endpoint: str = "http://localhost:4318/v1/logs",
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        super().__init__(name)
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events via OTLP."""
        start_time = time.perf_counter()

        try:
            # Convert events to OTLP format
            otlp_data = self._convert_to_otlp(events)

            # Send to OTLP endpoint
            response = self._send_otlp_data(otlp_data)

            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.is_success:
                result = ExportResult(
                    success=True,
                    events_exported=len(events),
                    export_duration_ms=duration_ms,
                    metadata={"status_code": response.status_code},
                )
            else:
                result = ExportResult(
                    success=False,
                    events_exported=0,
                    error=f"HTTP {response.status_code}: {response.text}",
                    export_duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result

    def _convert_to_otlp(self, events: list[ObservabilityEvent]) -> dict[str, Any]:
        """Convert events to OTLP log format."""
        log_records = []

        for event in events:
            # Convert event to OTLP log record format
            attributes_list: list[dict[str, Any]] = [
                {"key": "event_type", "value": {"stringValue": event.event_type}},
                {"key": "operation", "value": {"stringValue": event.operation}},
                {"key": "status", "value": {"stringValue": event.status}},
            ]

            log_record: dict[str, Any] = {
                "timeUnixNano": str(int(event.timestamp * 1_000_000_000)),
                "severityNumber": self._get_severity_number(event.status),
                "severityText": event.status.upper(),
                "body": {"stringValue": f"{event.event_type}.{event.operation}"},
                "attributes": attributes_list,
            }

            # Add optional fields
            if event.duration_ms is not None:
                log_record["attributes"].append(
                    {"key": "duration_ms", "value": {"doubleValue": event.duration_ms}}
                )

            if event.symbol:
                log_record["attributes"].append(
                    {"key": "symbol", "value": {"stringValue": event.symbol}}
                )

            if event.order_id:
                log_record["attributes"].append(
                    {"key": "order_id", "value": {"stringValue": event.order_id}}
                )

            if event.portfolio_id:
                log_record["attributes"].append(
                    {"key": "portfolio_id", "value": {"stringValue": event.portfolio_id}}
                )

            if event.correlation_id:
                log_record["attributes"].append(
                    {"key": "correlation_id", "value": {"stringValue": event.correlation_id}}
                )

            if event.trace_id:
                log_record["attributes"].append(
                    {"key": "trace_id", "value": {"stringValue": event.trace_id}}
                )

            if event.span_id:
                log_record["attributes"].append(
                    {"key": "span_id", "value": {"stringValue": event.span_id}}
                )

            # Add context as attributes
            for key, value in event.context.items():
                log_record["attributes"].append(
                    {"key": f"context.{key}", "value": {"stringValue": str(value)}}
                )

            log_records.append(log_record)

        # OTLP logs format
        return {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "ai-trading-system"}},
                            {"key": "service.version", "value": {"stringValue": "1.0.0"}},
                        ]
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "observability.collector", "version": "1.0.0"},
                            "logRecords": log_records,
                        }
                    ],
                }
            ]
        }

    def _get_severity_number(self, status: str) -> int:
        """Map status to OTLP severity number."""
        severity_map = {
            "success": 9,  # INFO
            "info": 9,  # INFO
            "warning": 13,  # WARN
            "error": 17,  # ERROR
            "critical": 21,  # FATAL
        }
        return severity_map.get(status, 9)

    def _send_otlp_data(self, data: dict[str, Any]) -> httpx.Response:
        """Send data to OTLP endpoint."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.endpoint,
                json=data,
                headers={"Content-Type": "application/json", **self.headers},
            )
            return response


class AsyncOTLPExporter(ObservabilityExporter):
    """Async version of OTLP exporter."""

    def __init__(
        self,
        name: str = "async_otlp",
        endpoint: str = "http://localhost:4318/v1/logs",
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        super().__init__(name)
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events via async OTLP (runs in event loop)."""
        # This is a sync method that schedules async work
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, schedule the async export
                task = asyncio.create_task(self._async_export(events))
                # For now, return a pending result - in production this would be handled differently
                return ExportResult(
                    success=True, events_exported=len(events), metadata={"async_export": True}
                )
            else:
                # Run the async export
                return asyncio.run(self._async_export(events))
        except Exception as e:
            return ExportResult(success=False, events_exported=0, error=str(e))

    async def _async_export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Async export implementation."""
        start_time = time.perf_counter()

        try:
            # Convert events to OTLP format (reuse from sync version)
            otlp_exporter = OTLPExporter("temp", self.endpoint, self.headers, self.timeout)
            otlp_data = otlp_exporter._convert_to_otlp(events)

            # Send async HTTP request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json=otlp_data,
                    headers={"Content-Type": "application/json", **self.headers},
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.is_success:
                result = ExportResult(
                    success=True,
                    events_exported=len(events),
                    export_duration_ms=duration_ms,
                    metadata={"status_code": response.status_code},
                )
            else:
                result = ExportResult(
                    success=False,
                    events_exported=0,
                    error=f"HTTP {response.status_code}: {response.text}",
                    export_duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result


class CustomFormatExporter(ObservabilityExporter):
    """Export events in custom format for specific integrations."""

    def __init__(
        self,
        name: str,
        formatter: Callable[[list[ObservabilityEvent]], Any],
        output_handler: Callable[[Any], None],
    ):
        super().__init__(name)
        self.formatter = formatter
        self.output_handler = output_handler

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events using custom format and handler."""
        start_time = time.perf_counter()

        try:
            # Format events
            formatted_data = self.formatter(events)

            # Handle output
            self.output_handler(formatted_data)

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=True, events_exported=len(events), export_duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = ExportResult(
                success=False, events_exported=0, error=str(e), export_duration_ms=duration_ms
            )

        self._record_export(result)
        return result


class MultiExporter(ObservabilityExporter):
    """Export events to multiple exporters."""

    def __init__(
        self, name: str = "multi", exporters: list[ObservabilityExporter] | None = None
    ) -> None:
        super().__init__(name)
        self.exporters = exporters or []

    def add_exporter(self, exporter: ObservabilityExporter) -> None:
        """Add an exporter."""
        self.exporters.append(exporter)

    def export(self, events: list[ObservabilityEvent]) -> ExportResult:
        """Export events to all configured exporters."""
        start_time = time.perf_counter()

        results = []
        total_exported = 0
        errors = []

        for exporter in self.exporters:
            try:
                result = exporter.export(events)
                results.append(result)

                if result.success:
                    total_exported += result.events_exported
                else:
                    errors.append(f"{exporter.name}: {result.error}")

            except Exception as e:
                errors.append(f"{exporter.name}: {e!s}")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Overall result
        success = len(errors) == 0
        error_msg = "; ".join(errors) if errors else None

        result = ExportResult(
            success=success,
            events_exported=total_exported,
            error=error_msg,
            export_duration_ms=duration_ms,
            metadata={
                "individual_results": [
                    r.get_stats() if hasattr(r, "get_stats") else str(r) for r in results
                ]
            },
        )

        self._record_export(result)
        return result


# Factory functions
def create_console_exporter(format_json: bool = True) -> "ConsoleExporter":
    """Create console exporter."""
    return ConsoleExporter(format_json=format_json)


def create_file_exporter(
    file_path: str = "observability_events.jsonl", **kwargs: Any
) -> "FileExporter":
    """Create file exporter."""
    return FileExporter(file_path=file_path, **kwargs)


def create_prometheus_exporter(metrics_file: str | None = None) -> "PrometheusExporter":
    """Create Prometheus exporter."""
    return PrometheusExporter(metrics_file=metrics_file)


def create_otlp_exporter(
    endpoint: str = "http://localhost:4318/v1/logs", **kwargs: Any
) -> "OTLPExporter":
    """Create OTLP exporter."""
    return OTLPExporter(endpoint=endpoint, **kwargs)


def create_multi_exporter(exporters: list[ObservabilityExporter]) -> MultiExporter:
    """Create multi-exporter."""
    return MultiExporter(exporters=exporters)


# Global metrics exporter instance
_metrics_exporter: ObservabilityExporter | None = None


def initialize_metrics_exporter(exporter: ObservabilityExporter) -> None:
    """Initialize global metrics exporter."""
    global _metrics_exporter
    _metrics_exporter = exporter


def get_metrics_exporter() -> ObservabilityExporter:
    """Get global metrics exporter."""
    if not _metrics_exporter:
        # Return default console exporter
        return create_console_exporter()
    return _metrics_exporter

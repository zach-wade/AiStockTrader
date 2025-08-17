#!/usr/bin/env python3
"""
Tests for Archive Dashboard Widget

Verifies widget rendering, metric updates, and alert generation.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
import json
import sys
from unittest.mock import AsyncMock, Mock

# Third-party imports
import pytest
import pytest_asyncio

sys.path.insert(0, "src")

# Local imports
from main.monitoring.dashboards.widgets.archive_widget import ArchiveWidget
from main.monitoring.metrics.archive_metrics_collector import StorageMetrics


class TestArchiveWidget:
    """Tests for the archive dashboard widget."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector."""
        collector = Mock()

        # Mock storage metrics using real StorageMetrics dataclass
        storage_metrics = StorageMetrics(
            total_size_bytes=10000000000,  # 10GB total disk
            used_size_bytes=5692375040,  # ~5.43GB used (matches 5432.1 MB)
            available_size_bytes=4307624960,  # ~4GB available
            file_count=1000,
            avg_file_size_bytes=5692375,  # ~5.43MB per file
            compression_ratio=2.5,
            growth_rate_bytes_per_hour=0,
            estimated_days_until_full=None,
            by_data_type={
                "parquet": {"files": 800, "bytes": 4194304000, "avg_size": 5242880},
                "json": {"files": 200, "bytes": 1502003200, "avg_size": 7510016},
            },
        )

        # Mock async methods
        async def get_storage_metrics():
            return storage_metrics

        collector.get_storage_metrics = AsyncMock(side_effect=get_storage_metrics)

        # Mock operation stats
        collector.get_operation_stats = Mock(
            return_value={
                "write": {"total": 10000, "success": 9950, "failed": 50},
                "read": {"total": 5000, "success": 4990, "failed": 10},
            }
        )

        # Mock performance metrics
        collector.get_performance_metrics = Mock(
            return_value={
                "write": {"avg_latency_ms": 45.2, "p95_latency_ms": 120.5},
                "read": {"avg_latency_ms": 12.3, "p95_latency_ms": 35.7},
            }
        )

        # Mock recent operations
        collector._get_recent_operations = Mock(
            return_value=[{"timestamp": datetime.now(UTC), "operation": "write"}] * 60
        )

        return collector

    @pytest_asyncio.fixture
    async def widget(self, mock_metrics_collector):
        """Create an archive widget with mock collector."""
        widget = ArchiveWidget(
            metrics_collector=mock_metrics_collector, config={"refresh_interval": 1}
        )
        return widget

    @pytest.mark.asyncio
    async def test_widget_initialization(self):
        """Test widget initialization without collector."""
        widget = ArchiveWidget()

        assert widget.metrics_collector is None
        assert widget.refresh_interval == 5
        assert widget.time_window == 3600
        assert widget._cached_metrics is None

    @pytest.mark.asyncio
    async def test_widget_render(self, widget):
        """Test widget rendering."""
        result = await widget.render()

        assert "title" in result
        assert result["title"] == "Archive System Monitoring"
        assert "panels" in result
        assert "layout" in result
        assert "refresh_interval" in result

        # Check all expected panels are present
        panels = result["panels"]
        assert "archive_overview" in panels
        assert "archive_operations" in panels
        assert "archive_storage" in panels
        assert "archive_performance" in panels
        assert "archive_health" in panels

    @pytest.mark.asyncio
    async def test_update_data(self, widget, mock_metrics_collector):
        """Test data update from metrics collector."""
        metrics = await widget.update_data()

        assert metrics is not None
        assert "storage" in metrics
        assert "operations" in metrics
        assert "performance" in metrics
        assert "timestamp" in metrics

        # Verify storage metrics (using calculated values from StorageMetrics properties)
        assert metrics["storage"]["total_files"] == 1000
        assert (
            abs(metrics["storage"]["total_size_mb"] - 5428.67) < 0.1
        )  # Allow small floating point differences
        assert metrics["storage"]["compression_ratio"] == 2.5

        # Verify operations metrics
        assert metrics["operations"]["write"]["total"] == 10000
        assert metrics["operations"]["read"]["total"] == 5000

    @pytest.mark.asyncio
    async def test_caching(self, widget):
        """Test metrics caching."""
        # First update
        metrics1 = await widget.update_data()

        # Second update within refresh interval - should return cached
        metrics2 = await widget.update_data()
        assert metrics1 == metrics2

        # Wait for refresh interval
        widget.refresh_interval = 0.1
        await asyncio.sleep(0.2)

        # Third update - should fetch new data
        metrics3 = await widget.update_data()
        # Timestamp should be different
        assert metrics3["timestamp"] != metrics1["timestamp"]

    @pytest.mark.asyncio
    async def test_overview_panel(self, widget):
        """Test overview panel creation."""
        await widget.update_data()
        panel = widget._create_overview_panel()

        assert panel["type"] == "stat"
        assert panel["title"] == "Archive Overview"
        assert "targets" in panel

        # Check targets
        targets = {t["name"]: t for t in panel["targets"]}
        assert "Total Files" in targets
        assert "Storage Used" in targets
        assert "Compression Ratio" in targets
        assert "Avg File Size" in targets

        assert targets["Total Files"]["value"] == 1000
        assert (
            abs(float(targets["Storage Used"]["value"]) - 5428.67) < 0.1
        )  # Allow small floating point differences

    @pytest.mark.asyncio
    async def test_health_panel(self, widget):
        """Test health panel creation."""
        await widget.update_data()
        panel = widget._create_health_panel()

        assert panel["type"] == "table"
        assert panel["title"] == "System Health"
        assert "rows" in panel

        # Check health indicators
        rows = panel["rows"]
        assert len(rows) == 3  # Storage, Error Rate, Compression

        # Find specific health items
        health_items = {row["name"]: row for row in rows}
        assert "Storage" in health_items
        assert "Error Rate" in health_items
        assert "Compression" in health_items

        # Compression should be OK (2.5x > 1.5)
        assert health_items["Compression"]["status"] == "OK"

    @pytest.mark.asyncio
    async def test_alerts_generation(self, widget):
        """Test alert generation."""
        await widget.update_data()
        alerts = widget.get_alerts()

        # With good metrics, should have no critical alerts
        assert len(alerts) == 0

        # Modify metrics to trigger alerts
        widget._cached_metrics["operations"]["write"]["failed"] = 1000
        widget._cached_metrics["storage"]["compression_ratio"] = 0.8

        alerts = widget.get_alerts()
        assert len(alerts) > 0

        # Check for specific alerts
        alert_names = [a["name"] for a in alerts]
        assert "High Archive Error Rate" in alert_names
        assert "Poor Compression Ratio" in alert_names

    @pytest.mark.asyncio
    async def test_export_grafana_config(self, widget):
        """Test Grafana configuration export."""
        await widget.update_data()
        config_json = await widget.export_config(format="grafana")

        config = json.loads(config_json)
        assert "dashboard" in config
        assert "panels" in config["dashboard"]
        assert config["dashboard"]["title"] == "Archive System Monitoring"
        assert config["dashboard"]["refresh"] == "1s"

    @pytest.mark.asyncio
    async def test_export_prometheus_queries(self, widget):
        """Test Prometheus query export."""
        queries = await widget.export_config(format="prometheus")

        assert "archive_storage_total_files" in queries
        assert "archive_storage_size_mb" in queries
        assert "archive_compression_ratio" in queries
        assert 'archive_operations_total{operation="write"}' in queries

    @pytest.mark.asyncio
    async def test_without_metrics_collector(self):
        """Test widget behavior without metrics collector."""
        widget = ArchiveWidget()

        # Should use default metrics
        metrics = await widget.update_data()
        assert metrics["storage"]["total_files"] == 0
        assert metrics["storage"]["total_size_mb"] == 0.0

        # Should still render
        result = await widget.render()
        assert "panels" in result

        # Should have no alerts
        alerts = widget.get_alerts()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_performance_panel(self, widget):
        """Test performance metrics panel."""
        await widget.update_data()
        panel = widget._create_performance_panel()

        assert panel["type"] == "gauge"
        assert panel["title"] == "Operation Latency"
        assert "targets" in panel

        targets = {t["name"]: t for t in panel["targets"]}
        assert "Write Latency" in targets
        assert "Read Latency" in targets

        # Check thresholds
        assert "thresholds" in targets["Write Latency"]
        assert targets["Write Latency"]["value"] == 45.2
        assert targets["Read Latency"]["value"] == 12.3


if __name__ == "__main__":
    # Standard library imports
    import subprocess

    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"], capture_output=False, check=False
    )
    sys.exit(result.returncode)

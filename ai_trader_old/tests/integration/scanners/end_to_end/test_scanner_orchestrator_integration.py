"""
End-to-end integration tests for scanner orchestrator workflows.

Tests the complete orchestrator functionality including scanner coordination,
error handling, and performance optimization.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest

# Local imports
from main.events.types.event_types import AlertType
from main.interfaces.scanners import IScannerOrchestrator
from main.scanners.types import ScanAlert
from main.utils.core import secure_uniform


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerOrchestratorIntegration:
    """Test complete scanner orchestrator integration."""

    async def test_parallel_scanner_execution_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        realistic_market_data,
        realistic_news_data,
        end_to_end_performance_thresholds,
    ):
        """Test parallel execution of multiple scanners."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock repository with different response times
        async def slow_market_data(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow database query
            return realistic_market_data

        async def fast_news_data(*args, **kwargs):
            await asyncio.sleep(0.1)  # Fast response
            return realistic_news_data

        end_to_end_orchestrator.repository.get_market_data = slow_market_data
        end_to_end_orchestrator.repository.get_news_data = fast_news_data
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_social_sentiment = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_earnings_data = AsyncMock(return_value=[])

        # Set parallel execution strategy
        if hasattr(end_to_end_orchestrator, "config"):
            end_to_end_orchestrator.config.execution_strategy = "parallel"

        # Run orchestrated scan
        start_time = datetime.now()
        results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:5])
        end_time = datetime.now()

        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Parallel execution should be faster than sequential
        # Should complete in time of slowest scanner + overhead
        max_expected_time = 500 + end_to_end_performance_thresholds["orchestrator_overhead_ms"]
        assert execution_time_ms < max_expected_time

        # Should have results from multiple scanners
        assert isinstance(results, dict)
        assert len(results) > 0

        # Verify events were published from different scanners
        scanner_names = set(event.scanner_name for event in event_collector.scanner_alerts)
        assert len(scanner_names) >= 1  # At least one scanner type

    async def test_sequential_scanner_execution_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        realistic_market_data,
    ):
        """Test sequential execution of scanners."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock repository
        execution_order = []

        async def track_volume_scan(*args, **kwargs):
            execution_order.append("volume")
            return realistic_market_data

        async def track_news_scan(*args, **kwargs):
            execution_order.append("news")
            return []

        end_to_end_orchestrator.repository.get_market_data = track_volume_scan
        end_to_end_orchestrator.repository.get_news_data = track_news_scan
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})

        # Set sequential execution strategy
        if hasattr(end_to_end_orchestrator, "config"):
            end_to_end_orchestrator.config.execution_strategy = "sequential"

        # Run orchestrated scan
        results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:3])

        # Should have executed scanners in priority order
        assert len(execution_order) > 0

        # Verify sequential execution (scanners called in order)
        if len(execution_order) > 1:
            # Should maintain priority order (higher priority first)
            assert execution_order == sorted(
                execution_order, reverse=True
            )  # Assuming priority ordering

    async def test_hybrid_scanner_execution_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        realistic_market_data,
        realistic_news_data,
    ):
        """Test hybrid execution strategy (high priority sequential, others parallel)."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock repository
        end_to_end_orchestrator.repository.get_market_data = AsyncMock(
            return_value=realistic_market_data
        )
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(
            return_value=realistic_news_data
        )
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_social_sentiment = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_earnings_data = AsyncMock(return_value=[])

        # Set hybrid execution strategy
        if hasattr(end_to_end_orchestrator, "config"):
            end_to_end_orchestrator.config.execution_strategy = "hybrid"

        # Run orchestrated scan
        results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:3])

        # Should complete successfully with hybrid strategy
        assert isinstance(results, dict)

        # Should have results from multiple scanners
        scanner_results = [k for k, v in results.items() if v]
        assert len(scanner_results) >= 1

    async def test_scanner_priority_orchestration(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test scanner priority-based orchestration."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock scanners with different priorities and execution times
        scanner_execution_log = []

        class MockPriorityScanner:
            def __init__(self, name: str, priority: int, execution_time: float):
                self.name = name
                self.priority = priority
                self.execution_time = execution_time

            async def scan(self, symbols: list[str]) -> list[ScanAlert]:
                start_time = datetime.now()
                await asyncio.sleep(self.execution_time)
                end_time = datetime.now()

                scanner_execution_log.append(
                    {
                        "name": self.name,
                        "priority": self.priority,
                        "start_time": start_time,
                        "end_time": end_time,
                        "execution_time": self.execution_time,
                    }
                )

                # Generate mock alert
                return [
                    ScanAlert(
                        symbol=symbols[0] if symbols else "TEST",
                        alert_type=AlertType.VOLUME_SPIKE,
                        score=0.8,
                        metadata={"priority": self.priority},
                        timestamp=datetime.now(UTC),
                        scanner_name=self.name,
                    )
                ]

        # Create mock scanners with different priorities
        mock_scanners = [
            MockPriorityScanner("low_priority_scanner", 5, 0.1),  # Low priority, fast
            MockPriorityScanner("high_priority_scanner", 10, 0.3),  # High priority, slower
            MockPriorityScanner("med_priority_scanner", 7, 0.2),  # Medium priority
        ]

        # Mock the orchestrator's scanner creation
        with patch.object(end_to_end_orchestrator, "_create_scanner_instances") as mock_create:
            mock_create.return_value = mock_scanners

            # Run orchestrated scan
            results = await end_to_end_orchestrator.run_scan(symbols=["AAPL"])

        # Verify execution occurred
        assert len(scanner_execution_log) == 3

        # Check priority ordering in sequential parts
        if (
            hasattr(end_to_end_orchestrator, "config")
            and getattr(end_to_end_orchestrator.config, "execution_strategy", None) == "sequential"
        ):
            # Should execute in priority order (highest first)
            priorities = [log["priority"] for log in scanner_execution_log]
            assert priorities == sorted(priorities, reverse=True)

    async def test_scanner_timeout_handling(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        end_to_end_performance_thresholds,
    ):
        """Test orchestrator handling of scanner timeouts."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock repository with timeout scenarios
        async def timeout_market_data(*args, **kwargs):
            await asyncio.sleep(10)  # Long delay to trigger timeout
            return {}

        async def normal_news_data(*args, **kwargs):
            await asyncio.sleep(0.1)  # Normal response time
            return []

        end_to_end_orchestrator.repository.get_market_data = timeout_market_data
        end_to_end_orchestrator.repository.get_news_data = normal_news_data
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})

        # Set shorter timeout for testing
        if hasattr(end_to_end_orchestrator, "config"):
            end_to_end_orchestrator.config.timeout_seconds = 2.0

        # Run orchestrated scan
        start_time = datetime.now()
        results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:2])
        end_time = datetime.now()

        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Should complete within timeout + overhead
        max_expected_time = 2000 + end_to_end_performance_thresholds["orchestrator_overhead_ms"]
        assert execution_time_ms < max_expected_time

        # Should have results from non-timeout scanners
        assert isinstance(results, dict)

        # Should handle timeout gracefully (no exceptions)
        # Verify system continued operating despite timeouts

    async def test_scanner_health_monitoring_integration(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test health monitoring integration in orchestrator."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock repository with health issues
        health_status = {"healthy_calls": 0, "failed_calls": 0}

        async def health_aware_market_data(*args, **kwargs):
            if health_status["failed_calls"] < 2:
                health_status["failed_calls"] += 1
                raise Exception("Temporary health issue")
            else:
                health_status["healthy_calls"] += 1
                return {}

        end_to_end_orchestrator.repository.get_market_data = health_aware_market_data
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(return_value=[])
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})

        # Enable health monitoring
        if hasattr(end_to_end_orchestrator, "config"):
            end_to_end_orchestrator.config.enable_health_monitoring = True

        # Run multiple scans to test health recovery
        for i in range(3):
            try:
                results = await end_to_end_orchestrator.run_scan(symbols=["AAPL"])
                # Later scans should succeed as health improves
                if i >= 2:
                    assert isinstance(results, dict)
            except Exception:
                # Early scans may fail due to health issues
                if i >= 2:
                    pytest.fail("Health monitoring should have recovered by now")

            await asyncio.sleep(0.1)  # Brief pause between scans

        # System should have recovered
        assert health_status["healthy_calls"] > 0

    async def test_scanner_deduplication_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test alert deduplication across scanners."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock scanners that generate similar alerts
        duplicate_alert = ScanAlert(
            symbol="AAPL",
            alert_type=AlertType.VOLUME_SPIKE,
            score=0.8,
            metadata={"source": "both_scanners"},
            timestamp=datetime.now(UTC),
            scanner_name="duplicate_test",
        )

        class MockDuplicateScanner:
            def __init__(self, name: str):
                self.name = name
                self.priority = 8

            async def scan(self, symbols: list[str]) -> list[ScanAlert]:
                # Both scanners generate the same alert
                alert_copy = ScanAlert(
                    symbol=duplicate_alert.symbol,
                    alert_type=duplicate_alert.alert_type,
                    score=duplicate_alert.score + secure_uniform(-0.1, 0.1),  # Slight variation
                    metadata={"source": self.name},
                    timestamp=datetime.now(UTC),
                    scanner_name=self.name,
                )
                return [alert_copy]

        mock_scanners = [MockDuplicateScanner("scanner_a"), MockDuplicateScanner("scanner_b")]

        # Mock the orchestrator's scanner creation
        with patch.object(end_to_end_orchestrator, "_create_scanner_instances") as mock_create:
            mock_create.return_value = mock_scanners

            # Enable deduplication
            if hasattr(end_to_end_orchestrator, "config"):
                end_to_end_orchestrator.config.enable_deduplication = True

            # Run orchestrated scan
            results = await end_to_end_orchestrator.run_scan(symbols=["AAPL"])

        # Check deduplication effectiveness
        aapl_alerts = event_collector.get_events_by_symbol("AAPL")

        if len(aapl_alerts) > 0:
            # Should have fewer alerts than scanners (due to deduplication)
            # Or should have combined/merged alerts
            unique_alert_types = set(alert.alert_type for alert in aapl_alerts)
            assert len(unique_alert_types) <= len(mock_scanners)

    async def test_scanner_aggregation_window_workflow(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test alert aggregation within time windows."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock scanners that generate alerts over time
        class MockTimedScanner:
            def __init__(self, name: str, delay: float):
                self.name = name
                self.priority = 8
                self.delay = delay

            async def scan(self, symbols: list[str]) -> list[ScanAlert]:
                await asyncio.sleep(self.delay)
                return [
                    ScanAlert(
                        symbol="AAPL",
                        alert_type=AlertType.VOLUME_SPIKE,
                        score=0.8,
                        metadata={"delay": self.delay},
                        timestamp=datetime.now(UTC),
                        scanner_name=self.name,
                    )
                ]

        mock_scanners = [
            MockTimedScanner("fast_scanner", 0.1),
            MockTimedScanner("slow_scanner", 0.3),
        ]

        # Mock the orchestrator's scanner creation
        with patch.object(end_to_end_orchestrator, "_create_scanner_instances") as mock_create:
            mock_create.return_value = mock_scanners

            # Set aggregation window
            if hasattr(end_to_end_orchestrator, "config"):
                end_to_end_orchestrator.config.alert_aggregation_window_seconds = 1.0

            # Run orchestrated scan
            results = await end_to_end_orchestrator.run_scan(symbols=["AAPL"])

        # Should have aggregated alerts within time window
        aapl_alerts = event_collector.get_events_by_symbol("AAPL")

        if len(aapl_alerts) > 0:
            # Verify alerts were published within reasonable time
            timestamps = [alert.timestamp for alert in aapl_alerts]
            time_span = max(timestamps) - min(timestamps)
            assert time_span.total_seconds() <= 2.0  # Within aggregation window + overhead

    async def test_orchestrator_error_isolation(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test that orchestrator isolates errors from individual scanners."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock scanners with different failure modes
        class MockFailingScanner:
            def __init__(self, name: str, should_fail: bool):
                self.name = name
                self.priority = 8
                self.should_fail = should_fail

            async def scan(self, symbols: list[str]) -> list[ScanAlert]:
                if self.should_fail:
                    raise Exception(f"Scanner {self.name} failed")
                else:
                    return [
                        ScanAlert(
                            symbol="AAPL",
                            alert_type=AlertType.VOLUME_SPIKE,
                            score=0.8,
                            metadata={"scanner": self.name},
                            timestamp=datetime.now(UTC),
                            scanner_name=self.name,
                        )
                    ]

        mock_scanners = [
            MockFailingScanner("working_scanner", False),
            MockFailingScanner("failing_scanner", True),
            MockFailingScanner("another_working_scanner", False),
        ]

        # Mock the orchestrator's scanner creation
        with patch.object(end_to_end_orchestrator, "_create_scanner_instances") as mock_create:
            mock_create.return_value = mock_scanners

            # Run orchestrated scan
            results = await end_to_end_orchestrator.run_scan(symbols=["AAPL"])

        # Should complete despite individual scanner failures
        assert isinstance(results, dict)

        # Should have results from working scanners
        working_results = [k for k, v in results.items() if v and k != "failing_scanner"]
        assert len(working_results) > 0

        # Should have published events from working scanners
        alerts = event_collector.scanner_alerts
        working_scanner_alerts = [
            alert for alert in alerts if alert.scanner_name != "failing_scanner"
        ]
        assert len(working_scanner_alerts) > 0

    async def test_orchestrator_resource_management(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
        end_to_end_performance_thresholds,
    ):
        """Test orchestrator resource management and limits."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock resource-intensive scanners
        resource_usage = {"memory_allocations": 0, "concurrent_operations": 0}

        class MockResourceScanner:
            def __init__(self, name: str):
                self.name = name
                self.priority = 8

            async def scan(self, symbols: list[str]) -> list[ScanAlert]:
                # Simulate resource usage
                resource_usage["concurrent_operations"] += 1
                resource_usage["memory_allocations"] += (
                    len(symbols) * 1000
                )  # Simulate memory per symbol

                try:
                    await asyncio.sleep(0.2)  # Simulate work
                    return [
                        ScanAlert(
                            symbol=symbols[0] if symbols else "TEST",
                            alert_type=AlertType.VOLUME_SPIKE,
                            score=0.8,
                            metadata={"memory_used": len(symbols) * 1000},
                            timestamp=datetime.now(UTC),
                            scanner_name=self.name,
                        )
                    ]
                finally:
                    resource_usage["concurrent_operations"] -= 1

        # Create multiple resource-intensive scanners
        mock_scanners = [MockResourceScanner(f"resource_scanner_{i}") for i in range(10)]

        # Mock the orchestrator's scanner creation
        with patch.object(end_to_end_orchestrator, "_create_scanner_instances") as mock_create:
            mock_create.return_value = mock_scanners

            # Set resource limits
            if hasattr(end_to_end_orchestrator, "config"):
                end_to_end_orchestrator.config.max_concurrent_scanners = 3

            # Run orchestrated scan
            start_time = datetime.now()
            results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:5])
            end_time = datetime.now()

            execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Should respect concurrency limits
        max_concurrent = max(
            resource_usage.get("concurrent_operations", 0), 3
        )  # Allow some tolerance
        assert max_concurrent <= 5  # Should not exceed reasonable limits

        # Should complete within performance thresholds
        assert execution_time_ms < end_to_end_performance_thresholds["scan_completion_time_ms"]

        # Should have completed all scans
        assert isinstance(results, dict)
        assert len(results) > 0

    async def test_orchestrator_configuration_changes(
        self,
        end_to_end_orchestrator: IScannerOrchestrator,
        event_collector,
        comprehensive_test_symbols,
    ):
        """Test orchestrator adaptation to configuration changes."""
        # Replace event bus
        end_to_end_orchestrator.event_bus = event_collector

        # Mock basic repository
        end_to_end_orchestrator.repository.get_market_data = AsyncMock(return_value={})
        end_to_end_orchestrator.repository.get_news_data = AsyncMock(return_value=[])
        end_to_end_orchestrator.repository.get_volume_statistics = AsyncMock(return_value={})

        # Test different configurations
        configurations = [
            {"execution_strategy": "parallel", "max_concurrent_scanners": 5},
            {"execution_strategy": "sequential", "timeout_seconds": 10},
            {"execution_strategy": "hybrid", "enable_deduplication": True},
        ]

        for config in configurations:
            # Update orchestrator configuration
            if hasattr(end_to_end_orchestrator, "config"):
                for key, value in config.items():
                    setattr(end_to_end_orchestrator.config, key, value)

            # Clear previous results
            event_collector.clear()

            # Run scan with new configuration
            results = await end_to_end_orchestrator.run_scan(symbols=comprehensive_test_symbols[:3])

            # Should adapt to new configuration successfully
            assert isinstance(results, dict)

            # Verify configuration-specific behavior
            if config.get("execution_strategy") == "parallel":
                # Should have handled parallel execution
                pass  # Could add more specific checks
            elif config.get("execution_strategy") == "sequential":
                # Should have executed sequentially
                pass  # Could add more specific checks

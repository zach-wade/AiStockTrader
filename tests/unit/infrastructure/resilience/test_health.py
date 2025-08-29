"""
Tests for health monitoring system.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.infrastructure.resilience.health import (
    APIHealthCheck,
    HealthCheck,
    HealthChecker,
    HealthCheckResult,
    HealthMetrics,
    HealthStatus,
    ServiceHealth,
    ServiceHealthCheck,
)


class TestHealthMetrics:
    """Test health metrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = HealthMetrics()

        assert metrics.check_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.avg_response_time == 0.0
        assert metrics.last_check_time is None
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is None
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = HealthMetrics()

        # No checks yet
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 0.0

        # Add some data
        metrics.check_count = 10
        metrics.success_count = 7
        metrics.failure_count = 3

        assert metrics.success_rate == 0.7
        assert metrics.failure_rate == 0.3


class TestHealthCheckResult:
    """Test health check result."""

    def test_result_creation(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            response_time=1.5,
            timestamp=time.time(),
            details={"version": "1.0"},
            error=None,
        )

        assert result.service_name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 1.5
        assert result.details["version"] == "1.0"
        assert result.error is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = time.time()
        result = HealthCheckResult(
            service_name="test",
            status=HealthStatus.DEGRADED,
            response_time=2.0,
            timestamp=timestamp,
            details={"key": "value"},
            error="Some error",
        )

        result_dict = result.to_dict()

        assert result_dict["service_name"] == "test"
        assert result_dict["status"] == "degraded"
        assert result_dict["response_time"] == 2.0
        assert result_dict["timestamp"] == timestamp
        assert result_dict["details"] == {"key": "value"}
        assert result_dict["error"] == "Some error"


class MockHealthCheck(HealthCheck):
    """Mock health check for testing."""

    def __init__(self, name: str, should_pass: bool = True, delay: float = 0.1):
        super().__init__(name)
        self.should_pass = should_pass
        self.delay = delay
        self.call_count = 0

    async def check_health(self) -> HealthCheckResult:
        """Mock health check implementation."""
        self.call_count += 1
        start_time = time.time()

        await asyncio.sleep(self.delay)

        if self.should_pass:
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.HEALTHY,
                response_time=self.delay,
                timestamp=start_time,
                details={"call_count": self.call_count},
            )
        else:
            raise Exception(f"Mock failure for {self.name}")


class TestHealthCheck:
    """Test abstract health check base class."""

    @pytest.mark.asyncio
    async def test_safe_check_health_success(self):
        """Test safe health check with successful result."""
        health_check = MockHealthCheck("test", should_pass=True, delay=0.01)

        result = await health_check.safe_check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.service_name == "test"
        assert result.error is None
        assert result.response_time > 0

    @pytest.mark.asyncio
    async def test_safe_check_health_failure(self):
        """Test safe health check with failure."""
        health_check = MockHealthCheck("test", should_pass=False)

        result = await health_check.safe_check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.service_name == "test"
        assert result.error is not None
        assert "Mock failure for test" in result.error

    @pytest.mark.asyncio
    async def test_safe_check_health_timeout(self):
        """Test safe health check with timeout."""
        health_check = MockHealthCheck("test", should_pass=True, delay=0.2)
        health_check.timeout = 0.1  # Shorter than delay

        result = await health_check.safe_check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.service_name == "test"
        assert result.error is not None
        assert "timed out" in result.error


class TestAPIHealthCheck:
    """Test API health check."""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_api_health_check_success(self, mock_client_class):
        """Test successful API health check."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"OK"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        health_check = APIHealthCheck(
            name="test_api",
            url="https://api.example.com/health",
            expected_status=200,
            command_timeout=5.0,
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.service_name == "test_api"
        assert result.error is None
        assert result.details["status_code"] == 200
        assert result.details["url"] == "https://api.example.com/health"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_api_health_check_wrong_status(self, mock_client_class):
        """Test API health check with wrong status code."""
        # Mock HTTP response with wrong status
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.content = b"Internal Server Error"

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        health_check = APIHealthCheck(
            name="test_api",
            url="https://api.example.com/health",
            expected_status=200,
            command_timeout=5.0,
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.service_name == "test_api"
        assert result.error is not None
        assert "Expected status 200, got 500" in result.error
        assert result.details["status_code"] == 500

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_api_health_check_exception(self, mock_client_class):
        """Test API health check with connection exception."""
        # Mock connection exception
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        health_check = APIHealthCheck(
            name="test_api", url="https://api.example.com/health", command_timeout=5.0
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.service_name == "test_api"
        assert result.error is not None
        assert "Connection failed" in result.error


class TestServiceHealthCheck:
    """Test service health check."""

    @pytest.mark.asyncio
    async def test_service_health_check_async_success(self):
        """Test service health check with async function."""

        @pytest.mark.asyncio
        async def test_service():
            await asyncio.sleep(0.01)
            return "service_result"

        health_check = ServiceHealthCheck(
            name="test_service", service_func=test_service, command_timeout=1.0
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.service_name == "test_service"
        assert result.error is None
        assert result.details["result_type"] == "str"

    @pytest.mark.asyncio
    async def test_service_health_check_sync_success(self):
        """Test service health check with sync function."""

        def test_service():
            return "sync_result"

        health_check = ServiceHealthCheck(
            name="test_service", service_func=test_service, command_timeout=1.0
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.service_name == "test_service"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_service_health_check_failure(self):
        """Test service health check with function failure."""

        def failing_service():
            raise RuntimeError("Service failure")

        health_check = ServiceHealthCheck(
            name="test_service", service_func=failing_service, command_timeout=1.0
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.service_name == "test_service"
        assert result.error is not None
        assert "Service failure" in result.error


class TestHealthChecker:
    """Test health checker system."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker for testing."""
        return HealthChecker(
            check_interval=0.1,  # Fast for testing
            history_size=10,
            degraded_threshold=0.7,
            unhealthy_threshold=3,
        )

    def test_health_checker_initialization(self, health_checker):
        """Test health checker initialization."""
        assert health_checker.check_interval == 0.1
        assert health_checker.history_size == 10
        assert health_checker.degraded_threshold == 0.7
        assert health_checker.unhealthy_threshold == 3
        assert len(health_checker.health_checks) == 0

    def test_register_health_check(self, health_checker):
        """Test registering health checks."""
        mock_check = MockHealthCheck("test1")

        health_checker.register_health_check(mock_check)

        assert "test1" in health_checker.health_checks
        assert health_checker.health_checks["test1"] is mock_check

    def test_unregister_health_check(self, health_checker):
        """Test unregistering health checks."""
        mock_check = MockHealthCheck("test1")
        health_checker.register_health_check(mock_check)

        health_checker.unregister_health_check("test1")

        assert "test1" not in health_checker.health_checks

        # Unregistering non-existent check should not raise
        health_checker.unregister_health_check("nonexistent")

    @pytest.mark.asyncio
    async def test_check_all_health(self, health_checker):
        """Test checking all health checks."""
        mock_check1 = MockHealthCheck("service1", should_pass=True, delay=0.01)
        mock_check2 = MockHealthCheck("service2", should_pass=False, delay=0.01)

        health_checker.register_health_check(mock_check1)
        health_checker.register_health_check(mock_check2)

        results = await health_checker.check_all_health()

        assert len(results) == 2
        assert "service1" in results
        assert "service2" in results

        assert results["service1"].status == HealthStatus.HEALTHY
        assert results["service2"].status == HealthStatus.UNHEALTHY

        # Check metrics were updated
        metrics1 = health_checker.metrics["service1"]
        metrics2 = health_checker.metrics["service2"]

        assert metrics1.check_count == 1
        assert metrics1.success_count == 1
        assert metrics1.failure_count == 0

        assert metrics2.check_count == 1
        assert metrics2.success_count == 0
        assert metrics2.failure_count == 1

    @pytest.mark.asyncio
    async def test_check_specific_health(self, health_checker):
        """Test checking specific service health."""
        mock_check = MockHealthCheck("test_service", should_pass=True, delay=0.01)
        health_checker.register_health_check(mock_check)

        result = await health_checker.check_health("test_service")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.service_name == "test_service"

        # Non-existent service
        result = await health_checker.check_health("nonexistent")
        assert result is None

    def test_get_service_status(self, health_checker):
        """Test getting service status based on metrics."""
        service_name = "test_service"

        # Unknown service
        status = health_checker.get_service_status(service_name)
        assert status == HealthStatus.UNKNOWN

        # Healthy service
        metrics = health_checker.metrics[service_name]
        metrics.check_count = 10
        metrics.success_count = 10
        metrics.failure_count = 0
        metrics.consecutive_failures = 0

        status = health_checker.get_service_status(service_name)
        assert status == HealthStatus.HEALTHY

        # Degraded service (low success rate)
        metrics.success_count = 6
        metrics.failure_count = 4

        status = health_checker.get_service_status(service_name)
        assert status == HealthStatus.DEGRADED

        # Unhealthy service (consecutive failures)
        metrics.consecutive_failures = 3

        status = health_checker.get_service_status(service_name)
        assert status == HealthStatus.UNHEALTHY

    def test_get_overall_status(self, health_checker):
        """Test getting overall system status."""
        # No health checks
        status = health_checker.get_overall_status()
        assert status == HealthStatus.UNKNOWN

        # Add healthy services
        health_checker.health_checks["service1"] = MockHealthCheck("service1")
        health_checker.health_checks["service2"] = MockHealthCheck("service2")

        # All healthy
        health_checker.metrics["service1"].check_count = 5
        health_checker.metrics["service1"].success_count = 5
        health_checker.metrics["service2"].check_count = 5
        health_checker.metrics["service2"].success_count = 5

        status = health_checker.get_overall_status()
        assert status == HealthStatus.HEALTHY

        # One degraded
        health_checker.metrics["service1"].success_count = 3
        health_checker.metrics["service1"].failure_count = 2

        status = health_checker.get_overall_status()
        assert status == HealthStatus.DEGRADED

        # One unhealthy
        health_checker.metrics["service2"].consecutive_failures = 3

        status = health_checker.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_get_health_summary(self, health_checker):
        """Test getting comprehensive health summary."""
        mock_check = MockHealthCheck("test_service", should_pass=True, delay=0.01)
        health_checker.register_health_check(mock_check)

        # Generate some history
        await health_checker.check_all_health()
        await health_checker.check_all_health()

        summary = health_checker.get_health_summary()

        assert "overall_status" in summary
        assert "check_interval" in summary
        assert "timestamp" in summary
        assert "services" in summary

        assert "test_service" in summary["services"]
        service_data = summary["services"]["test_service"]

        assert service_data["status"] == HealthStatus.HEALTHY
        assert "metrics" in service_data
        assert "recent_results" in service_data

        assert service_data["metrics"]["check_count"] == 2
        assert service_data["metrics"]["success_count"] == 2
        assert len(service_data["recent_results"]) == 2

    @pytest.mark.asyncio
    async def test_background_monitoring(self, health_checker):
        """Test background health monitoring."""
        mock_check = MockHealthCheck("test_service", should_pass=True, delay=0.01)
        health_checker.register_health_check(mock_check)

        # Start monitoring
        await health_checker.start_monitoring()

        # Wait for a few checks
        await asyncio.sleep(0.25)

        # Stop monitoring
        await health_checker.stop_monitoring()

        # Should have performed multiple checks
        metrics = health_checker.metrics["test_service"]
        assert metrics.check_count >= 2  # At least a couple of checks
        assert metrics.success_count >= 2
        assert mock_check.call_count >= 2

    def test_reset_metrics(self, health_checker):
        """Test resetting health metrics."""
        service_name = "test_service"

        # Add some metrics
        metrics = health_checker.metrics[service_name]
        metrics.check_count = 10
        metrics.success_count = 8
        metrics.failure_count = 2

        health_checker.history[service_name].append(
            HealthCheckResult(service_name, HealthStatus.HEALTHY, 1.0, time.time())
        )

        # Reset specific service
        health_checker.reset_metrics(service_name)

        assert health_checker.metrics[service_name].check_count == 0
        assert len(health_checker.history[service_name]) == 0

        # Add metrics again
        health_checker.metrics["service2"].check_count = 5

        # Reset all
        health_checker.reset_metrics()

        assert len(health_checker.metrics) == 0
        assert len(health_checker.history) == 0


class TestServiceHealth:
    """Test service health wrapper."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker for testing."""
        checker = HealthChecker(check_interval=0.1)
        mock_check = MockHealthCheck("test_service", should_pass=True, delay=0.01)
        checker.register_health_check(mock_check)
        return checker

    def test_service_health_initialization(self, health_checker):
        """Test service health initialization."""
        service_health = ServiceHealth("test_service", health_checker)

        assert service_health.name == "test_service"
        assert service_health.health_checker is health_checker
        assert service_health._last_check_time == 0.0
        assert service_health._last_status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_is_healthy(self, health_checker):
        """Test is_healthy method."""
        service_health = ServiceHealth("test_service", health_checker)

        # First call should trigger health check
        is_healthy = await service_health.is_healthy()
        assert is_healthy is True

        # Second call within cache duration should use cached result
        is_healthy = await service_health.is_healthy()
        assert is_healthy is True

        # Check that cache is working (only one actual check)
        mock_check = health_checker.health_checks["test_service"]
        assert mock_check.call_count == 1

    @pytest.mark.asyncio
    async def test_is_healthy_with_failure(self, health_checker):
        """Test is_healthy with failing service."""
        # Replace with failing check
        health_checker.unregister_health_check("test_service")
        failing_check = MockHealthCheck("test_service", should_pass=False)
        health_checker.register_health_check(failing_check)

        service_health = ServiceHealth("test_service", health_checker)

        is_healthy = await service_health.is_healthy()
        assert is_healthy is False

    def test_get_status(self, health_checker):
        """Test get_status method."""
        service_health = ServiceHealth("test_service", health_checker)

        # Initially unknown
        status = service_health.get_status()
        assert status == HealthStatus.UNKNOWN

        # Add some metrics to health checker
        metrics = health_checker.metrics["test_service"]
        metrics.check_count = 5
        metrics.success_count = 5

        status = service_health.get_status()
        assert status == HealthStatus.HEALTHY


# Additional fixtures
@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    connection = AsyncMock()
    connection.is_connected = True
    connection.acquire.return_value.__aenter__ = AsyncMock()
    connection.acquire.return_value.__aexit__ = AsyncMock()
    connection.get_enhanced_pool_stats = AsyncMock(
        return_value={"status": "connected", "max_size": 20, "min_size": 5}
    )
    return connection

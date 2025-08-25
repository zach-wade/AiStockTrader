"""
Health Monitoring and Service Discovery

Production-grade health checking system for monitoring service availability,
database connections, API endpoints, and overall system health.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Health check metrics."""

    check_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    last_check_time: float | None = None
    last_success_time: float | None = None
    last_failure_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.check_count == 0:
            return 0.0
        return self.success_count / self.check_count

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.check_count == 0:
            return 0.0
        return self.failure_count / self.check_count


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    service_name: str
    status: HealthStatus
    response_time: float
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "response_time": self.response_time,
            "timestamp": self.timestamp,
            "details": self.details,
            "error": self.error,
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout: float = 30.0) -> None:
        self.name = name
        self.timeout = timeout

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            HealthCheckResult with status and details
        """
        pass

    async def safe_check_health(self) -> HealthCheckResult:
        """
        Safely perform health check with timeout and error handling.

        Returns:
            HealthCheckResult (never raises exceptions)
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(self.check_health(), timeout=self.timeout)
            return result

        except TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                error=f"Health check timed out after {self.timeout}s",
            )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                error=str(e),
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""

    def __init__(self, name: str, connection_factory: Any, timeout: float = 10.0) -> None:
        super().__init__(name, timeout)
        self.connection_factory = connection_factory

    async def check_health(self) -> HealthCheckResult:
        """Check database connection health."""
        start_time = time.time()

        try:
            # Get database connection
            connection = await self.connection_factory.get_connection()

            # Test query
            async with connection.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1 as health_check")
                    result = await cursor.fetchone()

                    if result and result[0] == 1:
                        response_time = time.time() - start_time

                        # Get pool stats
                        pool_stats = await connection.get_pool_stats()

                        return HealthCheckResult(
                            service_name=self.name,
                            status=HealthStatus.HEALTHY,
                            response_time=response_time,
                            timestamp=start_time,
                            details=pool_stats,
                        )
                    else:
                        raise Exception("Invalid health check response")

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                error=str(e),
            )


class APIHealthCheck(HealthCheck):
    """Health check for external API endpoints."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout: float = 15.0,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(name, timeout)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    async def check_health(self) -> HealthCheckResult:
        """Check API endpoint health."""
        import httpx

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.url, headers=self.headers)

                response_time = time.time() - start_time

                if response.status_code == self.expected_status:
                    status = HealthStatus.HEALTHY
                    error = None
                else:
                    status = HealthStatus.UNHEALTHY
                    error = f"Expected status {self.expected_status}, got {response.status_code}"

                return HealthCheckResult(
                    service_name=self.name,
                    status=status,
                    response_time=response_time,
                    timestamp=start_time,
                    details={
                        "url": self.url,
                        "status_code": response.status_code,
                        "content_length": len(response.content) if response.content else 0,
                    },
                    error=error,
                )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details={"url": self.url},
                error=str(e),
            )


class ServiceHealthCheck(HealthCheck):
    """Health check for internal services."""

    def __init__(
        self,
        name: str,
        service_func: Callable[..., Any],
        timeout: float = 10.0,
        test_args: tuple[Any, ...] = (),
        test_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, timeout)
        self.service_func = service_func
        self.test_args = test_args
        self.test_kwargs = test_kwargs or {}

    async def check_health(self) -> HealthCheckResult:
        """Check service function health."""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(self.service_func):
                result = await self.service_func(*self.test_args, **self.test_kwargs)
            else:
                result = self.service_func(*self.test_args, **self.test_kwargs)

            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details={"result_type": type(result).__name__},
            )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                error=str(e),
            )


class HealthChecker:
    """
    Production-grade health monitoring system.

    Features:
    - Multiple health check types
    - Configurable intervals and thresholds
    - Historical metrics tracking
    - Automatic degradation detection
    - Background monitoring
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 100,
        degraded_threshold: float = 0.7,  # Success rate threshold for degraded status
        unhealthy_threshold: int = 3,  # Consecutive failures for unhealthy status
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold

        # Health checks registry
        self.health_checks: dict[str, HealthCheck] = {}

        # Metrics tracking
        self.metrics: dict[str, HealthMetrics] = defaultdict(HealthMetrics)
        self.history: dict[str, deque[Any]] = defaultdict(lambda: deque[Any](maxlen=history_size))

        # Background monitoring
        self._monitoring_task: asyncio.Task[Any] | None = None
        self._stop_monitoring = False

        logger.info(f"Initialized HealthChecker with {check_interval}s interval")

    def register_health_check(self, health_check: HealthCheck) -> None:
        """
        Register a health check.

        Args:
            health_check: HealthCheck instance
        """
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")

    def unregister_health_check(self, name: str) -> None:
        """
        Unregister a health check.

        Args:
            name: Health check name
        """
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Unregistered health check: {name}")

    async def check_all_health(self) -> dict[str, HealthCheckResult]:
        """
        Check health of all registered services.

        Returns:
            Dictionary mapping service names to health check results
        """
        if not self.health_checks:
            return {}

        # Run all health checks concurrently
        tasks = []
        for name, health_check in self.health_checks.items():
            task = asyncio.create_task(health_check.safe_check_health())
            tasks.append((name, task))

        results = {}
        for name, task in tasks:
            result = await task
            results[name] = result

            # Update metrics
            self._update_metrics(result)

        return results

    async def check_health(self, service_name: str) -> HealthCheckResult | None:
        """
        Check health of a specific service.

        Args:
            service_name: Name of service to check

        Returns:
            HealthCheckResult or None if service not found
        """
        health_check = self.health_checks.get(service_name)
        if not health_check:
            return None

        result = await health_check.safe_check_health()
        self._update_metrics(result)

        return result

    def _update_metrics(self, result: HealthCheckResult) -> None:
        """Update metrics for a health check result."""
        metrics = self.metrics[result.service_name]

        metrics.check_count += 1
        metrics.last_check_time = result.timestamp

        if result.status == HealthStatus.HEALTHY:
            metrics.success_count += 1
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
            metrics.last_success_time = result.timestamp
        else:
            metrics.failure_count += 1
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            metrics.last_failure_time = result.timestamp

        # Update average response time
        if metrics.check_count == 1:
            metrics.avg_response_time = result.response_time
        else:
            # Exponential moving average
            alpha = 0.1
            metrics.avg_response_time = (
                alpha * result.response_time + (1 - alpha) * metrics.avg_response_time
            )

        # Add to history
        self.history[result.service_name].append(result)

    def get_service_status(self, service_name: str) -> HealthStatus:
        """
        Get current status of a service based on metrics.

        Args:
            service_name: Service name

        Returns:
            Current HealthStatus
        """
        metrics = self.metrics.get(service_name)
        if not metrics or metrics.check_count == 0:
            return HealthStatus.UNKNOWN

        # Check for consecutive failures
        if metrics.consecutive_failures >= self.unhealthy_threshold:
            return HealthStatus.UNHEALTHY

        # Check success rate for degraded status
        if metrics.success_rate < self.degraded_threshold:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall HealthStatus
        """
        if not self.health_checks:
            return HealthStatus.UNKNOWN

        service_statuses = [self.get_service_status(name) for name in self.health_checks.keys()]

        # If any service is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in service_statuses:
            return HealthStatus.UNHEALTHY

        # If any service is degraded, system is degraded
        if HealthStatus.DEGRADED in service_statuses:
            return HealthStatus.DEGRADED

        # If all services are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in service_statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get comprehensive health summary.

        Returns:
            Health summary dictionary
        """
        now = time.time()

        services = {}
        for name in self.health_checks.keys():
            metrics = self.metrics[name]
            history = list(self.history[name])

            services[name] = {
                "status": self.get_service_status(name).value,
                "metrics": {
                    "check_count": metrics.check_count,
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "consecutive_failures": metrics.consecutive_failures,
                    "consecutive_successes": metrics.consecutive_successes,
                    "last_check_age": (
                        now - metrics.last_check_time if metrics.last_check_time else None
                    ),
                    "last_success_age": (
                        now - metrics.last_success_time if metrics.last_success_time else None
                    ),
                },
                "recent_results": [result.to_dict() for result in history[-10:]],  # Last 10 results
            }

        return {
            "overall_status": self.get_overall_status().value,
            "check_interval": self.check_interval,
            "timestamp": now,
            "services": services,
        }

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Health monitoring is already running")
            return

        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started background health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._stop_monitoring = True

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                await self.check_all_health()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying

    def reset_metrics(self, service_name: str | None = None) -> None:
        """
        Reset metrics for a service or all services.

        Args:
            service_name: Service name to reset, or None for all services
        """
        if service_name:
            if service_name in self.metrics:
                self.metrics[service_name] = HealthMetrics()
                self.history[service_name].clear()
                logger.info(f"Reset metrics for service: {service_name}")
        else:
            self.metrics.clear()
            self.history.clear()
            logger.info("Reset all health metrics")


class ServiceHealth:
    """Simple service health wrapper for integration with existing services."""

    def __init__(self, name: str, health_checker: HealthChecker) -> None:
        self.name = name
        self.health_checker = health_checker
        self._last_check_time = 0.0
        self._last_status = HealthStatus.UNKNOWN
        self._check_cache_duration = 5.0  # Cache status for 5 seconds

    async def is_healthy(self) -> bool:
        """
        Check if service is healthy (with caching).

        Returns:
            True if service is healthy
        """
        now = time.time()

        # Use cached status if recent
        if now - self._last_check_time < self._check_cache_duration:
            return self._last_status == HealthStatus.HEALTHY

        # Perform new health check
        result = await self.health_checker.check_health(self.name)
        if result:
            self._last_status = result.status
            self._last_check_time = now
            return result.status == HealthStatus.HEALTHY

        return False

    def get_status(self) -> HealthStatus:
        """Get cached service status."""
        return self.health_checker.get_service_status(self.name)

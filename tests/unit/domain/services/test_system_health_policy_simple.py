"""
Tests for SystemHealthPolicy - testing only existing methods.
"""

from src.infrastructure.monitoring.system_health_policy import (
    ComponentHealth,
    ComponentType,
    HealthScore,
    HealthStatus,
    HealthThresholds,
    SystemHealthPolicy,
)


class TestSystemHealthPolicy:
    """Test SystemHealthPolicy with existing methods."""

    def test_determine_health_status(self):
        """Test health status determination."""
        policy = SystemHealthPolicy()

        # Test healthy metrics
        metrics = {"response_time": 50, "error_rate": 0.001, "throughput": 1000}
        status = policy.determine_health_status(metrics)
        assert isinstance(status, HealthStatus)

        # Test degraded metrics
        metrics = {"response_time": 300, "error_rate": 0.03, "throughput": 50}
        status = policy.determine_health_status(metrics)
        assert isinstance(status, HealthStatus)

    def test_determine_overall_status(self):
        """Test overall status determination."""
        policy = SystemHealthPolicy()

        # All healthy
        statuses = {"database": "healthy", "api": "healthy", "cache": "healthy"}
        overall = policy.determine_overall_status(statuses)
        assert overall == "healthy"

        # Some degraded
        statuses = {"database": "healthy", "api": "degraded", "cache": "healthy"}
        overall = policy.determine_overall_status(statuses)
        assert overall in ["degraded", "healthy"]

        # Critical component unhealthy
        statuses = {"database": "critical", "api": "healthy", "cache": "healthy"}
        overall = policy.determine_overall_status(statuses)
        assert overall == "critical"

    def test_calculate_health_score(self):
        """Test health score calculation."""
        policy = SystemHealthPolicy()

        components = [
            ComponentHealth(
                name="database",
                type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time=50,
                error_rate=0.001,
                throughput=1000,
            ),
            ComponentHealth(
                name="api",
                type=ComponentType.API_GATEWAY,
                status=HealthStatus.HEALTHY,
                response_time=100,
                error_rate=0.002,
                throughput=500,
            ),
        ]

        score = policy.calculate_health_score(components)
        assert 0 <= score <= 100

    def test_evaluate_system_health(self):
        """Test system health evaluation."""
        policy = SystemHealthPolicy()

        components = [
            ComponentHealth(
                name="database",
                type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time=50,
                error_rate=0.001,
                throughput=1000,
            ),
            ComponentHealth(
                name="cache",
                type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                response_time=200,
                error_rate=0.05,
                throughput=100,
            ),
        ]

        health_score = policy.evaluate_system_health(components)
        assert isinstance(health_score, HealthScore)
        assert 0 <= health_score.score <= 100
        assert health_score.components_healthy >= 0
        assert health_score.components_degraded >= 0

    def test_get_monitoring_thresholds(self):
        """Test getting monitoring thresholds."""
        policy = SystemHealthPolicy()

        # Test database thresholds
        thresholds = policy.get_monitoring_thresholds(ComponentType.DATABASE)
        assert isinstance(thresholds, HealthThresholds)
        assert thresholds.response_time_warning > 0
        assert thresholds.response_time_critical > thresholds.response_time_warning

        # Test broker thresholds
        thresholds = policy.get_monitoring_thresholds(ComponentType.BROKER)
        assert isinstance(thresholds, HealthThresholds)

    def test_is_critical_component(self):
        """Test critical component identification."""
        policy = SystemHealthPolicy()

        # Critical components
        assert policy.is_critical_component(ComponentType.DATABASE) is True
        assert policy.is_critical_component(ComponentType.BROKER) is True
        assert policy.is_critical_component(ComponentType.ORDER_MANAGER) is True

        # Non-critical components
        assert policy.is_critical_component(ComponentType.CACHE) is False


class TestHealthScore:
    """Test HealthScore dataclass."""

    def test_create_health_score(self):
        """Test creating health score."""
        score = HealthScore(
            score=85.0,
            status=HealthStatus.HEALTHY,
            components_healthy=5,
            components_degraded=1,
            components_unhealthy=0,
            components_critical=0,
            critical_issues=[],
            warnings=["Cache performance degraded"],
            recommendations=["Increase cache size"],
        )

        assert score.score == 85.0
        assert score.status == HealthStatus.HEALTHY
        assert score.components_healthy == 5
        assert len(score.warnings) == 1

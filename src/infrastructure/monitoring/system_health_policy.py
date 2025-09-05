"""
System Health Policy Service - Domain Layer

This service contains all business logic for system health assessment,
including health status determination, scoring, and monitoring thresholds.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""

    DATABASE = "database"
    BROKER = "broker"
    MARKET_DATA = "market_data"
    ORDER_MANAGER = "order_manager"
    RISK_ENGINE = "risk_engine"
    PORTFOLIO_MANAGER = "portfolio_manager"
    API_GATEWAY = "api_gateway"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    SCHEDULER = "scheduler"


@dataclass
class ComponentHealth:
    """Health information for a system component."""

    name: str
    type: ComponentType
    status: HealthStatus
    response_time: float | None = None  # milliseconds
    error_rate: float | None = None  # percentage
    throughput: float | None = None  # requests per second
    last_check: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class HealthThresholds:
    """Health thresholds for monitoring."""

    response_time_warning: float  # milliseconds
    response_time_critical: float
    error_rate_warning: float  # percentage
    error_rate_critical: float
    throughput_min_warning: float  # requests per second
    throughput_min_critical: float
    availability_warning: float  # percentage
    availability_critical: float


@dataclass
class HealthScore:
    """Overall system health score."""

    score: float  # 0-100
    status: HealthStatus
    components_healthy: int
    components_degraded: int
    components_unhealthy: int
    components_critical: int
    critical_issues: list[str]
    warnings: list[str]
    recommendations: list[str]


class SystemHealthPolicy:
    """
    Domain service for system health policies.

    Contains all business logic for determining system health status,
    calculating health scores, and defining monitoring thresholds.
    """

    # Component-specific health thresholds
    COMPONENT_THRESHOLDS = {
        ComponentType.DATABASE: HealthThresholds(
            response_time_warning=100,
            response_time_critical=500,
            error_rate_warning=1.0,
            error_rate_critical=5.0,
            throughput_min_warning=100,
            throughput_min_critical=10,
            availability_warning=99.0,
            availability_critical=95.0,
        ),
        ComponentType.BROKER: HealthThresholds(
            response_time_warning=50,
            response_time_critical=200,
            error_rate_warning=0.1,
            error_rate_critical=1.0,
            throughput_min_warning=1000,
            throughput_min_critical=100,
            availability_warning=99.9,
            availability_critical=99.0,
        ),
        ComponentType.MARKET_DATA: HealthThresholds(
            response_time_warning=20,
            response_time_critical=100,
            error_rate_warning=0.5,
            error_rate_critical=2.0,
            throughput_min_warning=5000,
            throughput_min_critical=1000,
            availability_warning=99.5,
            availability_critical=98.0,
        ),
        ComponentType.ORDER_MANAGER: HealthThresholds(
            response_time_warning=100,
            response_time_critical=500,
            error_rate_warning=0.01,
            error_rate_critical=0.1,
            throughput_min_warning=100,
            throughput_min_critical=10,
            availability_warning=99.99,
            availability_critical=99.9,
        ),
        ComponentType.RISK_ENGINE: HealthThresholds(
            response_time_warning=200,
            response_time_critical=1000,
            error_rate_warning=0.1,
            error_rate_critical=1.0,
            throughput_min_warning=50,
            throughput_min_critical=5,
            availability_warning=99.9,
            availability_critical=99.0,
        ),
    }

    # Component criticality weights for health scoring
    COMPONENT_WEIGHTS = {
        ComponentType.ORDER_MANAGER: 1.0,  # Most critical
        ComponentType.RISK_ENGINE: 1.0,
        ComponentType.BROKER: 0.9,
        ComponentType.DATABASE: 0.9,
        ComponentType.MARKET_DATA: 0.8,
        ComponentType.PORTFOLIO_MANAGER: 0.7,
        ComponentType.API_GATEWAY: 0.6,
        ComponentType.MESSAGE_QUEUE: 0.5,
        ComponentType.CACHE: 0.4,
        ComponentType.SCHEDULER: 0.3,
    }

    # Dependencies between components
    COMPONENT_DEPENDENCIES = {
        ComponentType.ORDER_MANAGER: {
            ComponentType.BROKER,
            ComponentType.RISK_ENGINE,
            ComponentType.DATABASE,
        },
        ComponentType.RISK_ENGINE: {ComponentType.DATABASE, ComponentType.MARKET_DATA},
        ComponentType.PORTFOLIO_MANAGER: {ComponentType.DATABASE, ComponentType.MARKET_DATA},
        ComponentType.API_GATEWAY: {ComponentType.ORDER_MANAGER, ComponentType.PORTFOLIO_MANAGER},
        ComponentType.BROKER: {ComponentType.MARKET_DATA},
    }

    def determine_health_status(self, metrics: dict[str, Any]) -> HealthStatus:
        """
        Determine system health based on metrics.

        Args:
            metrics: Dictionary of health metrics

        Returns:
            HealthStatus enum
        """
        # Extract key metrics
        response_time = metrics.get("response_time", 0)
        error_rate = metrics.get("error_rate", 0)
        throughput = metrics.get("throughput", float("inf"))
        availability = metrics.get("availability", 100)

        # Get component type for threshold lookup
        component_type = self._determine_component_type(metrics.get("component_name", ""))
        thresholds = self.COMPONENT_THRESHOLDS.get(component_type, self._get_default_thresholds())

        # Critical status checks
        if (
            error_rate >= thresholds.error_rate_critical
            or response_time >= thresholds.response_time_critical
            or throughput <= thresholds.throughput_min_critical
            or availability <= thresholds.availability_critical
        ):
            return HealthStatus.CRITICAL

        # Unhealthy status checks
        if (
            error_rate >= thresholds.error_rate_warning * 2
            or response_time >= thresholds.response_time_warning * 2
            or throughput <= thresholds.throughput_min_warning * 0.5
            or availability <= thresholds.availability_warning * 0.98
        ):
            return HealthStatus.UNHEALTHY

        # Degraded status checks
        if (
            error_rate >= thresholds.error_rate_warning
            or response_time >= thresholds.response_time_warning
            or throughput <= thresholds.throughput_min_warning
            or availability <= thresholds.availability_warning
        ):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def determine_overall_status(self, component_statuses: dict[str, str]) -> str:
        """
        Determine overall system status from component statuses.

        Args:
            component_statuses: Dictionary of component names to status strings

        Returns:
            Overall status string
        """
        # Count statuses
        status_counts = {"critical": 0, "unhealthy": 0, "degraded": 0, "healthy": 0, "unknown": 0}

        for status in component_statuses.values():
            status_lower = status.lower()
            if status_lower in status_counts:
                status_counts[status_lower] += 1

        # Determine overall status based on component statuses
        if status_counts["critical"] > 0:
            return "critical"
        elif status_counts["unhealthy"] > 0 or status_counts["degraded"] > status_counts["healthy"]:
            return "unhealthy"
        elif status_counts["degraded"] > 0:
            return "degraded"
        elif status_counts["unknown"] > status_counts["healthy"]:
            return "unknown"
        else:
            return "healthy"

    def calculate_health_score(self, components: list[ComponentHealth]) -> float:
        """
        Calculate overall health score (0-100).

        Args:
            components: List of component health information

        Returns:
            Health score from 0 to 100
        """
        if not components:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0

        for component in components:
            # Get component weight
            weight = self.COMPONENT_WEIGHTS.get(component.type, 0.5)
            total_weight += weight

            # Calculate component score
            component_score = self._calculate_component_score(component)
            weighted_score += component_score * weight

        # Normalize to 0-100 scale
        if total_weight > 0:
            overall_score = (weighted_score / total_weight) * 100
        else:
            overall_score = 0.0

        # Apply cascade effects for critical dependencies
        overall_score = self._apply_dependency_cascade(components, overall_score)

        return max(0.0, min(100.0, overall_score))

    def evaluate_system_health(self, components: list[ComponentHealth]) -> HealthScore:
        """
        Comprehensive system health evaluation.

        Args:
            components: List of component health information

        Returns:
            Complete HealthScore with analysis
        """
        # Calculate overall score
        score = self.calculate_health_score(components)

        # Count component statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0,
        }

        critical_issues = []
        warnings = []

        for component in components:
            status_counts[component.status] += 1

            # Collect critical issues
            if component.status == HealthStatus.CRITICAL:
                critical_issues.append(f"{component.name} is in critical state")
            elif component.status == HealthStatus.UNHEALTHY:
                warnings.append(f"{component.name} is unhealthy")
            elif component.status == HealthStatus.DEGRADED:
                warnings.append(f"{component.name} is degraded")

        # Determine overall status
        if score >= 90:
            overall_status = HealthStatus.HEALTHY
        elif score >= 70:
            overall_status = HealthStatus.DEGRADED
        elif score >= 50:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.CRITICAL

        # Generate recommendations
        recommendations = self._generate_health_recommendations(components, score)

        return HealthScore(
            score=score,
            status=overall_status,
            components_healthy=status_counts[HealthStatus.HEALTHY],
            components_degraded=status_counts[HealthStatus.DEGRADED],
            components_unhealthy=status_counts[HealthStatus.UNHEALTHY],
            components_critical=status_counts[HealthStatus.CRITICAL],
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_monitoring_thresholds(self, component_type: ComponentType) -> HealthThresholds:
        """
        Get monitoring thresholds for a component type.

        Args:
            component_type: Type of component

        Returns:
            HealthThresholds for the component
        """
        return self.COMPONENT_THRESHOLDS.get(component_type, self._get_default_thresholds())

    def is_critical_component(self, component_type: ComponentType) -> bool:
        """
        Determine if a component is critical to system operation.

        Args:
            component_type: Type of component

        Returns:
            True if component is critical
        """
        critical_types = {
            ComponentType.ORDER_MANAGER,
            ComponentType.RISK_ENGINE,
            ComponentType.BROKER,
            ComponentType.DATABASE,
        }
        return component_type in critical_types

    def _calculate_component_score(self, component: ComponentHealth) -> float:
        """Calculate health score for a single component."""
        if component.status == HealthStatus.HEALTHY:
            return 1.0
        elif component.status == HealthStatus.DEGRADED:
            return 0.7
        elif component.status == HealthStatus.UNHEALTHY:
            return 0.3
        elif component.status == HealthStatus.CRITICAL:
            return 0.0
        else:  # UNKNOWN
            return 0.5

    def _apply_dependency_cascade(self, components: list[ComponentHealth], score: float) -> float:
        """Apply cascade effects from component dependencies."""
        component_map = {c.name: c for c in components}

        for component in components:
            if component.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                dependencies = self.COMPONENT_DEPENDENCIES.get(component.type, set())

                # Check if critical component affects others
                if dependencies and self.is_critical_component(component.type):
                    # Reduce score based on impact
                    impact_factor = 0.9 if component.status == HealthStatus.UNHEALTHY else 0.7
                    score *= impact_factor

        return score

    def _generate_health_recommendations(
        self, components: list[ComponentHealth], score: float
    ) -> list[str]:
        """Generate recommendations based on health analysis."""
        recommendations = []

        # Score-based recommendations
        if score < 50:
            recommendations.append(
                "URGENT: System health is critical. Immediate intervention required."
            )
        elif score < 70:
            recommendations.append("System health is poor. Review critical components immediately.")
        elif score < 90:
            recommendations.append(
                "System health is degraded. Monitor closely and plan maintenance."
            )

        # Component-specific recommendations
        for component in components:
            if component.status == HealthStatus.CRITICAL:
                recommendations.append(f"Restart or failover {component.name} immediately")
            elif component.status == HealthStatus.UNHEALTHY:
                if component.error_rate and component.error_rate > 5:
                    recommendations.append(f"Investigate high error rate in {component.name}")
                if component.response_time and component.response_time > 1000:
                    recommendations.append(f"Optimize performance of {component.name}")

        return recommendations

    def _determine_component_type(self, component_name: str) -> ComponentType:
        """Determine component type from name."""
        name_lower = component_name.lower()

        if "database" in name_lower or "db" in name_lower:
            return ComponentType.DATABASE
        elif "broker" in name_lower:
            return ComponentType.BROKER
        elif "market" in name_lower:
            return ComponentType.MARKET_DATA
        elif "order" in name_lower:
            return ComponentType.ORDER_MANAGER
        elif "risk" in name_lower:
            return ComponentType.RISK_ENGINE
        elif "portfolio" in name_lower:
            return ComponentType.PORTFOLIO_MANAGER
        elif "api" in name_lower or "gateway" in name_lower:
            return ComponentType.API_GATEWAY
        elif "cache" in name_lower or "redis" in name_lower:
            return ComponentType.CACHE
        elif "queue" in name_lower or "mq" in name_lower:
            return ComponentType.MESSAGE_QUEUE
        else:
            return ComponentType.API_GATEWAY  # Default

    def _get_default_thresholds(self) -> HealthThresholds:
        """Get default health thresholds."""
        return HealthThresholds(
            response_time_warning=200,
            response_time_critical=1000,
            error_rate_warning=1.0,
            error_rate_critical=5.0,
            throughput_min_warning=10,
            throughput_min_critical=1,
            availability_warning=99.0,
            availability_critical=95.0,
        )

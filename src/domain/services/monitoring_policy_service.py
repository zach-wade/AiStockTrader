"""
Monitoring Policy Service - Domain Layer

This service contains all business logic for monitoring policies,
including metric thresholds, performance evaluation, and alerting rules.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MetricType(Enum):
    """Types of metrics to monitor."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    QUEUE_DEPTH = "queue_depth"
    CACHE_HIT_RATE = "cache_hit_rate"
    BUSINESS_METRIC = "business_metric"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceLevel(Enum):
    """Performance assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class MetricThresholds:
    """Thresholds for a specific metric."""

    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    unit: str  # ms, %, RPS, MB, etc.
    direction: str  # 'above' or 'below' threshold triggers alert
    evaluation_period: int  # seconds
    datapoints_required: int  # minimum datapoints for evaluation


@dataclass
class PerformanceAssessment:
    """Performance assessment result."""

    level: PerformanceLevel
    score: float  # 0-100
    metrics: dict[str, float]
    bottlenecks: list[str]
    recommendations: list[str]
    timestamp: datetime


@dataclass
class Recommendation:
    """Performance improvement recommendation."""

    category: str  # scaling, optimization, configuration, etc.
    priority: int  # 1-5, 1 being highest
    action: str
    expected_improvement: str
    effort_estimate: str  # low, medium, high
    risk_level: str  # low, medium, high


@dataclass
class AlertRule:
    """Alert rule configuration."""

    metric_type: MetricType
    thresholds: MetricThresholds
    severity: AlertSeverity
    notification_channels: list[str]
    cooldown_period: int  # seconds between alerts
    auto_resolve: bool
    escalation_policy: str | None = None


class MonitoringPolicyService:
    """
    Domain service for monitoring policies.

    Contains all business logic for determining monitoring thresholds,
    evaluating performance, and generating recommendations.
    """

    # Metric thresholds by type and context
    METRIC_THRESHOLDS = {
        "trading": {
            MetricType.LATENCY: MetricThresholds(
                warning_threshold=100,
                error_threshold=500,
                critical_threshold=1000,
                unit="ms",
                direction="above",
                evaluation_period=60,
                datapoints_required=3,
            ),
            MetricType.ERROR_RATE: MetricThresholds(
                warning_threshold=0.1,
                error_threshold=1.0,
                critical_threshold=5.0,
                unit="%",
                direction="above",
                evaluation_period=300,
                datapoints_required=5,
            ),
            MetricType.THROUGHPUT: MetricThresholds(
                warning_threshold=100,
                error_threshold=50,
                critical_threshold=10,
                unit="RPS",
                direction="below",
                evaluation_period=120,
                datapoints_required=4,
            ),
        },
        "database": {
            MetricType.LATENCY: MetricThresholds(
                warning_threshold=50,
                error_threshold=200,
                critical_threshold=500,
                unit="ms",
                direction="above",
                evaluation_period=60,
                datapoints_required=3,
            ),
            MetricType.CPU_USAGE: MetricThresholds(
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=95,
                unit="%",
                direction="above",
                evaluation_period=300,
                datapoints_required=5,
            ),
            MetricType.QUEUE_DEPTH: MetricThresholds(
                warning_threshold=1000,
                error_threshold=5000,
                critical_threshold=10000,
                unit="items",
                direction="above",
                evaluation_period=60,
                datapoints_required=2,
            ),
        },
        "system": {
            MetricType.CPU_USAGE: MetricThresholds(
                warning_threshold=60,
                error_threshold=80,
                critical_threshold=90,
                unit="%",
                direction="above",
                evaluation_period=120,
                datapoints_required=4,
            ),
            MetricType.MEMORY_USAGE: MetricThresholds(
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=95,
                unit="%",
                direction="above",
                evaluation_period=120,
                datapoints_required=4,
            ),
            MetricType.DISK_USAGE: MetricThresholds(
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=90,
                unit="%",
                direction="above",
                evaluation_period=600,
                datapoints_required=2,
            ),
        },
    }

    # Performance benchmarks
    PERFORMANCE_BENCHMARKS = {
        "order_execution": {
            "excellent": {"latency": 10, "success_rate": 99.99},
            "good": {"latency": 50, "success_rate": 99.9},
            "acceptable": {"latency": 100, "success_rate": 99.0},
            "poor": {"latency": 500, "success_rate": 95.0},
        },
        "market_data": {
            "excellent": {"latency": 5, "throughput": 10000},
            "good": {"latency": 20, "throughput": 5000},
            "acceptable": {"latency": 50, "throughput": 1000},
            "poor": {"latency": 200, "throughput": 100},
        },
        "api_response": {
            "excellent": {"p50": 50, "p95": 200, "p99": 500},
            "good": {"p50": 100, "p95": 500, "p99": 1000},
            "acceptable": {"p50": 200, "p95": 1000, "p99": 2000},
            "poor": {"p50": 500, "p95": 2000, "p99": 5000},
        },
    }

    def determine_metric_thresholds(
        self, metric_type: str, context: str = "system"
    ) -> MetricThresholds:
        """
        Determine thresholds for metrics based on type and context.

        Args:
            metric_type: Type of metric
            context: Context for the metric (trading, database, system, etc.)

        Returns:
            MetricThresholds with configured values
        """
        # Try to get metric type enum
        try:
            metric_enum = MetricType(metric_type)
        except ValueError:
            metric_enum = MetricType.BUSINESS_METRIC

        # Get context-specific thresholds
        context_thresholds = self.METRIC_THRESHOLDS.get(context, {})

        if metric_enum in context_thresholds:
            return context_thresholds[metric_enum]

        # Fall back to system defaults
        system_thresholds = self.METRIC_THRESHOLDS.get("system", {})
        if metric_enum in system_thresholds:
            return system_thresholds[metric_enum]

        # Return generic thresholds
        return self._get_default_thresholds(metric_enum)

    def evaluate_performance(
        self, metrics: dict[str, float], service: str = "api_response"
    ) -> PerformanceAssessment:
        """
        Evaluate system performance based on metrics.

        Args:
            metrics: Dictionary of metric values
            service: Service being evaluated

        Returns:
            PerformanceAssessment with analysis
        """
        benchmarks = self.PERFORMANCE_BENCHMARKS.get(
            service, self.PERFORMANCE_BENCHMARKS["api_response"]
        )

        # Calculate performance score
        score = 100.0
        bottlenecks = []
        level = PerformanceLevel.EXCELLENT

        for metric_name, metric_value in metrics.items():
            # Check against benchmarks
            for perf_level in ["excellent", "good", "acceptable", "poor"]:
                benchmark = benchmarks.get(perf_level, {}) if isinstance(benchmarks, dict) else {}

                if isinstance(benchmark, dict) and metric_name in benchmark:
                    threshold = benchmark[metric_name]

                    # Determine if metric is above or below threshold
                    if metric_name in ["latency", "p50", "p95", "p99"]:
                        # Lower is better
                        if metric_value > threshold:
                            if perf_level == "poor":
                                score -= 40
                                level = PerformanceLevel.UNACCEPTABLE
                                bottlenecks.append(f"High {metric_name}: {metric_value}")
                            elif perf_level == "acceptable":
                                score -= 20
                                if level == PerformanceLevel.EXCELLENT:
                                    level = PerformanceLevel.POOR
                            elif perf_level == "good":
                                score -= 10
                                if level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
                                    level = PerformanceLevel.ACCEPTABLE
                            break
                    elif metric_value < threshold:
                        if perf_level == "poor":
                            score -= 40
                            level = PerformanceLevel.UNACCEPTABLE
                            bottlenecks.append(f"Low {metric_name}: {metric_value}")
                        elif perf_level == "acceptable":
                            score -= 20
                            if level == PerformanceLevel.EXCELLENT:
                                level = PerformanceLevel.POOR
                        elif perf_level == "good":
                            score -= 10
                            if level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
                                level = PerformanceLevel.ACCEPTABLE
                        break

        # Ensure score is within bounds
        score = max(0, min(100, score))

        # Generate recommendations
        recommendations = self.generate_recommendations(metrics, bottlenecks, service)

        return PerformanceAssessment(
            level=level,
            score=score,
            metrics=metrics,
            bottlenecks=bottlenecks,
            recommendations=[r.action for r in recommendations],
            timestamp=datetime.now(),
        )

    def generate_recommendations(
        self, metrics: dict[str, float], bottlenecks: list[str], service: str
    ) -> list[Recommendation]:
        """
        Generate performance recommendations based on analysis.

        Args:
            metrics: Current metric values
            bottlenecks: Identified bottlenecks
            service: Service being analyzed

        Returns:
            List of recommendations
        """
        recommendations = []

        # Latency recommendations
        if any("latency" in b.lower() for b in bottlenecks):
            latency_value = metrics.get("latency", 0)
            if latency_value > 1000:
                recommendations.append(
                    Recommendation(
                        category="optimization",
                        priority=1,
                        action="Implement caching for frequently accessed data",
                        expected_improvement="50-70% latency reduction",
                        effort_estimate="medium",
                        risk_level="low",
                    )
                )
                recommendations.append(
                    Recommendation(
                        category="scaling",
                        priority=2,
                        action="Add read replicas to distribute database load",
                        expected_improvement="40-60% latency reduction",
                        effort_estimate="high",
                        risk_level="medium",
                    )
                )
            elif latency_value > 500:
                recommendations.append(
                    Recommendation(
                        category="optimization",
                        priority=2,
                        action="Optimize database queries and add indexes",
                        expected_improvement="30-50% latency reduction",
                        effort_estimate="low",
                        risk_level="low",
                    )
                )

        # Throughput recommendations
        if any("throughput" in b.lower() for b in bottlenecks):
            throughput_value = metrics.get("throughput", float("inf"))
            if throughput_value < 100:
                recommendations.append(
                    Recommendation(
                        category="scaling",
                        priority=1,
                        action="Scale horizontally by adding more service instances",
                        expected_improvement="Linear throughput increase",
                        effort_estimate="medium",
                        risk_level="low",
                    )
                )
                recommendations.append(
                    Recommendation(
                        category="optimization",
                        priority=2,
                        action="Implement connection pooling and batch processing",
                        expected_improvement="2-3x throughput increase",
                        effort_estimate="medium",
                        risk_level="medium",
                    )
                )

        # Error rate recommendations
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 5:
            recommendations.append(
                Recommendation(
                    category="reliability",
                    priority=1,
                    action="Implement circuit breakers and retry logic",
                    expected_improvement="80% error reduction",
                    effort_estimate="medium",
                    risk_level="low",
                )
            )
        elif error_rate > 1:
            recommendations.append(
                Recommendation(
                    category="reliability",
                    priority=2,
                    action="Add input validation and error handling",
                    expected_improvement="50% error reduction",
                    effort_estimate="low",
                    risk_level="low",
                )
            )

        # Resource usage recommendations
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)

        if cpu_usage > 80:
            recommendations.append(
                Recommendation(
                    category="scaling",
                    priority=1,
                    action="Upgrade to higher CPU tier or add more instances",
                    expected_improvement="Immediate CPU relief",
                    effort_estimate="low",
                    risk_level="low",
                )
            )
            recommendations.append(
                Recommendation(
                    category="optimization",
                    priority=2,
                    action="Profile and optimize CPU-intensive operations",
                    expected_improvement="20-40% CPU reduction",
                    effort_estimate="high",
                    risk_level="medium",
                )
            )

        if memory_usage > 85:
            recommendations.append(
                Recommendation(
                    category="scaling",
                    priority=1,
                    action="Increase memory allocation or add memory-optimized instances",
                    expected_improvement="Immediate memory relief",
                    effort_estimate="low",
                    risk_level="low",
                )
            )
            recommendations.append(
                Recommendation(
                    category="optimization",
                    priority=2,
                    action="Identify and fix memory leaks, optimize data structures",
                    expected_improvement="30-50% memory reduction",
                    effort_estimate="high",
                    risk_level="medium",
                )
            )

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations[:5]  # Return top 5 recommendations

    def create_alert_rule(
        self, metric_type: MetricType, context: str, severity: AlertSeverity
    ) -> AlertRule:
        """
        Create an alert rule for a metric.

        Args:
            metric_type: Type of metric to monitor
            context: Context for the metric
            severity: Alert severity level

        Returns:
            AlertRule configuration
        """
        thresholds = self.determine_metric_thresholds(metric_type.value, context)

        # Determine notification channels based on severity
        if severity == AlertSeverity.CRITICAL:
            channels = ["pagerduty", "slack", "email", "sms"]
            cooldown = 300  # 5 minutes
        elif severity == AlertSeverity.ERROR:
            channels = ["slack", "email"]
            cooldown = 600  # 10 minutes
        elif severity == AlertSeverity.WARNING:
            channels = ["slack"]
            cooldown = 1800  # 30 minutes
        else:
            channels = ["logging"]
            cooldown = 3600  # 1 hour

        return AlertRule(
            metric_type=metric_type,
            thresholds=thresholds,
            severity=severity,
            notification_channels=channels,
            cooldown_period=cooldown,
            auto_resolve=severity != AlertSeverity.CRITICAL,
            escalation_policy="on-call" if severity == AlertSeverity.CRITICAL else None,
        )

    def should_alert(
        self, metric_value: float, metric_type: str, context: str
    ) -> tuple[bool, AlertSeverity]:
        """
        Determine if an alert should be triggered.

        Args:
            metric_value: Current metric value
            metric_type: Type of metric
            context: Context for evaluation

        Returns:
            Tuple of (should_alert, severity)
        """
        thresholds = self.determine_metric_thresholds(metric_type, context)

        # Check thresholds based on direction
        if thresholds.direction == "above":
            if metric_value >= thresholds.critical_threshold:
                return True, AlertSeverity.CRITICAL
            elif metric_value >= thresholds.error_threshold:
                return True, AlertSeverity.ERROR
            elif metric_value >= thresholds.warning_threshold:
                return True, AlertSeverity.WARNING
        elif metric_value <= thresholds.critical_threshold:
            return True, AlertSeverity.CRITICAL
        elif metric_value <= thresholds.error_threshold:
            return True, AlertSeverity.ERROR
        elif metric_value <= thresholds.warning_threshold:
            return True, AlertSeverity.WARNING

        return False, AlertSeverity.INFO

    def _get_default_thresholds(self, metric_type: MetricType) -> MetricThresholds:
        """Get default thresholds for a metric type."""
        defaults = {
            MetricType.LATENCY: MetricThresholds(
                warning_threshold=200,
                error_threshold=500,
                critical_threshold=1000,
                unit="ms",
                direction="above",
                evaluation_period=60,
                datapoints_required=3,
            ),
            MetricType.ERROR_RATE: MetricThresholds(
                warning_threshold=1,
                error_threshold=5,
                critical_threshold=10,
                unit="%",
                direction="above",
                evaluation_period=300,
                datapoints_required=5,
            ),
            MetricType.CPU_USAGE: MetricThresholds(
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=95,
                unit="%",
                direction="above",
                evaluation_period=120,
                datapoints_required=4,
            ),
            MetricType.MEMORY_USAGE: MetricThresholds(
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=95,
                unit="%",
                direction="above",
                evaluation_period=120,
                datapoints_required=4,
            ),
        }

        return defaults.get(
            metric_type,
            MetricThresholds(
                warning_threshold=50,
                error_threshold=75,
                critical_threshold=90,
                unit="units",
                direction="above",
                evaluation_period=60,
                datapoints_required=3,
            ),
        )

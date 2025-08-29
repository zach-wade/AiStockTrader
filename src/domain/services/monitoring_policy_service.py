"""
Monitoring Policy Service - Domain Layer

This service contains all business logic for monitoring policies,
including metric thresholds, performance evaluation, and alerting rules.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


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

        # Also check for resource usage metrics that aren't in benchmarks
        if "cpu_usage" in metrics and metrics["cpu_usage"] > 0.9:
            score -= 30
            bottlenecks.append(f"High cpu_usage: {metrics['cpu_usage']}")

        if "memory_usage" in metrics and metrics["memory_usage"] > 0.85:
            score -= 20
            bottlenecks.append(f"High memory_usage: {metrics['memory_usage']}")

        if "error_rate" in metrics and metrics["error_rate"] > 0.01:
            score -= 25
            bottlenecks.append(f"High error_rate: {metrics['error_rate']}")

        for metric_name, metric_value in metrics.items():
            # Skip metrics we already handled
            if metric_name in ["cpu_usage", "memory_usage", "error_rate"]:
                continue

            # Check against benchmarks
            for perf_level in ["poor", "acceptable", "good", "excellent"]:
                benchmark = benchmarks.get(perf_level, {}) if isinstance(benchmarks, dict) else {}

                if isinstance(benchmark, dict) and metric_name in benchmark:
                    threshold = benchmark[metric_name]

                    # Determine if metric is above or below threshold
                    if metric_name in ["latency", "p50", "p95", "p99"]:
                        # Lower is better - check if we exceed the threshold
                        if metric_value > threshold:
                            if perf_level == "poor":
                                score -= 40
                                bottlenecks.append(f"High {metric_name}: {metric_value}")
                            elif perf_level == "acceptable":
                                score -= 20
                            elif perf_level == "good":
                                score -= 10
                        else:
                            break  # Met this level, stop checking worse levels
                    elif metric_name in ["throughput", "success_rate"]:
                        # Higher is better - check if we're below the threshold
                        if metric_value < threshold:
                            if perf_level == "poor":
                                score -= 40
                                bottlenecks.append(f"Low {metric_name}: {metric_value}")
                            elif perf_level == "acceptable":
                                score -= 20
                            elif perf_level == "good":
                                score -= 10
                        else:
                            break  # Met this level, stop checking worse levels

        # Ensure score is within bounds and determine level
        score = max(0, min(100, score))

        if score >= 90:
            level = PerformanceLevel.EXCELLENT
        elif score >= 70:
            level = PerformanceLevel.GOOD
        elif score >= 50:
            level = PerformanceLevel.ACCEPTABLE
        elif score >= 30:
            level = PerformanceLevel.POOR
        else:
            level = PerformanceLevel.UNACCEPTABLE

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

    def _generate_recommendations_internal(
        self, metrics: dict[str, float], bottlenecks: list[str], service: str
    ) -> list[Recommendation]:
        """
        Internal method to generate performance recommendations.

        Args:
            metrics: Current metric values
            bottlenecks: Identified bottlenecks
            service: Service being analyzed

        Returns:
            List of recommendations
        """
        recommendations = []

        # Latency recommendations
        if any(
            "latency" in b.lower() or "p99" in b.lower() or "p95" in b.lower() or "p50" in b.lower()
            for b in bottlenecks
        ):
            latency_value = metrics.get(
                "latency", metrics.get("p99", metrics.get("p95", metrics.get("p50", 0)))
            )
            if latency_value > 500:
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

        # CPU usage recommendations
        if any("cpu" in b.lower() for b in bottlenecks):
            cpu_usage = metrics.get("cpu_usage", 0)
            if cpu_usage > 0.8:
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

        # Memory usage recommendations
        if any("memory" in b.lower() for b in bottlenecks):
            memory_usage = metrics.get("memory_usage", 0)
            if memory_usage > 0.85:
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

    def get_thresholds(self, metric_type: MetricType, context: str = "general") -> MetricThresholds:
        """
        Get thresholds for a specific metric type and context.

        Args:
            metric_type: Type of metric
            context: Context for the metric (api, database, general, etc.)

        Returns:
            MetricThresholds configuration
        """
        # API context thresholds
        if context == "api":
            if metric_type == MetricType.LATENCY:
                return MetricThresholds(
                    warning_threshold=100,
                    error_threshold=300,
                    critical_threshold=500,
                    unit="ms",
                    direction="above",
                    evaluation_period=60,
                    datapoints_required=3,
                )

        # Map context names for compatibility
        context_map = {"api": "trading", "general": "system"}
        mapped_context = context_map.get(context, context)

        return self.determine_metric_thresholds(metric_type.value, mapped_context)

    def evaluate_metric(
        self, metric_type: MetricType, value: float, context: str = "general"
    ) -> AlertSeverity | None:
        """
        Evaluate a metric value against thresholds.

        Args:
            metric_type: Type of metric
            value: Current metric value
            context: Context for evaluation (can be custom metric name)

        Returns:
            AlertSeverity if threshold exceeded, None otherwise
        """
        # Check if this is a custom metric
        thresholds: MetricThresholds
        if hasattr(self, "_custom_metrics") and context in self._custom_metrics:
            custom_metric = self._custom_metrics[context]
            if custom_metric["type"] == metric_type:
                thresholds = custom_metric["thresholds"]  # type: ignore[assignment]
            else:
                thresholds = self.get_thresholds(metric_type, context)
        else:
            thresholds = self.get_thresholds(metric_type, context)

        # Check thresholds based on direction
        if thresholds.direction == "above":
            if value >= thresholds.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= thresholds.error_threshold:
                return AlertSeverity.ERROR
            elif value >= thresholds.warning_threshold:
                return AlertSeverity.WARNING
        elif value <= thresholds.critical_threshold:
            return AlertSeverity.CRITICAL
        elif value <= thresholds.error_threshold:
            return AlertSeverity.ERROR
        elif value <= thresholds.warning_threshold:
            return AlertSeverity.WARNING

        return None

    def assess_performance(
        self, metrics: dict[str, float], service: str = "api_response"
    ) -> PerformanceAssessment:
        """
        Assess overall performance based on metrics.

        Args:
            metrics: Dictionary of metric values
            service: Service being assessed

        Returns:
            PerformanceAssessment with analysis
        """
        # Normalize metric names for compatibility
        normalized_metrics = {}
        for key, value in metrics.items():
            # Convert latency_pXX to pXX format
            if key.startswith("latency_p"):
                normalized_metrics[key.replace("latency_", "")] = value
            else:
                normalized_metrics[key] = value

        return self.evaluate_performance(normalized_metrics, service)

    def identify_bottlenecks(self, metrics: dict[str, float]) -> list[str]:
        """
        Identify performance bottlenecks from metrics.

        Args:
            metrics: Current metric values

        Returns:
            List of identified bottleneck types
        """
        bottlenecks = []

        # Check latency
        if "latency_p99" in metrics and metrics["latency_p99"] > 1000:
            bottlenecks.append("high_latency")

        # Check CPU
        if "cpu_usage" in metrics and metrics["cpu_usage"] > 0.9:
            bottlenecks.append("cpu_saturation")

        # Check memory
        if "memory_usage" in metrics and metrics["memory_usage"] > 0.85:
            bottlenecks.append("memory_pressure")

        # Check disk
        if "disk_usage" in metrics and metrics["disk_usage"] > 0.8:
            bottlenecks.append("disk_pressure")

        # Check error rate
        if "error_rate" in metrics and metrics["error_rate"] > 0.01:
            bottlenecks.append("high_error_rate")

        return bottlenecks

    def generate_recommendations(self, *args: Any) -> list[Recommendation]:
        """
        Generate recommendations - supports multiple signatures for compatibility.

        Can be called as:
        - generate_recommendations(bottlenecks, metrics) - for tests
        - generate_recommendations(metrics, bottlenecks, service) - for internal use

        Returns:
            List of recommendations
        """
        if len(args) == 2:
            # Test signature: (bottlenecks, metrics)
            bottlenecks, metrics = args
            service = "general"
        elif len(args) == 3:
            # Internal signature: (metrics, bottlenecks, service)
            metrics, bottlenecks, service = args
        else:
            raise TypeError(
                f"generate_recommendations() takes 2 or 3 arguments ({len(args)} given)"
            )

        # Call the internal method
        return self._generate_recommendations_internal(metrics, bottlenecks, service)

    def check_sla_compliance(
        self, metrics: dict[str, float], sla_requirements: dict[str, float]
    ) -> dict[str, Any]:
        """
        Check if metrics meet SLA requirements.

        Args:
            metrics: Current metric values
            sla_requirements: SLA threshold requirements

        Returns:
            Compliance status dictionary
        """
        compliance: dict[str, Any] = {"is_compliant": True, "violations": []}

        for metric, requirement in sla_requirements.items():
            actual = metrics.get(metric, 0)

            if metric in ["availability", "cache_hit_rate", "success_rate"]:
                # Higher is better
                compliant = actual >= requirement
            elif metric in ["latency_p99", "error_rate", "downtime"]:
                # Lower is better
                compliant = actual <= requirement
            else:
                # Default: lower is better
                compliant = actual <= requirement

            compliance[metric] = {
                "compliant": compliant,
                "actual": actual,
                "requirement": requirement,
                "margin": abs(actual - requirement),
            }

            if not compliant:
                compliance["is_compliant"] = False
                compliance["violations"].append(metric)

        return compliance

    def calculate_resource_utilization(self, metrics: dict[str, float]) -> dict[str, Any]:
        """
        Calculate resource utilization summary.

        Args:
            metrics: Resource metric values

        Returns:
            Utilization summary
        """
        utilization: dict[str, Any] = {
            "cpu": metrics.get("cpu_usage", 0),
            "memory": metrics.get("memory_usage", 0),
            "disk": metrics.get("disk_usage", 0),
            "network": metrics.get("network_usage", 0),
        }

        # Calculate overall utilization
        values = [v for v in utilization.values() if isinstance(v, (int, float)) and v > 0]
        overall = sum(values) / len(values) if values else 0
        utilization["overall_utilization"] = overall

        # Calculate efficiency metrics
        utilization["cpu_efficiency"] = (
            1.0 - utilization["cpu"] if utilization["cpu"] < 1.0 else 0.0
        )
        utilization["memory_efficiency"] = (
            1.0 - utilization["memory"] if utilization["memory"] < 1.0 else 0.0
        )

        # Calculate resource balance (lower variance = better balance)
        if values:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            utilization["resource_balance"] = 1.0 - min(
                variance, 1.0
            )  # Inverted so higher is better
        else:
            utilization["resource_balance"] = 1.0

        # Determine status
        if overall > 0.9:
            utilization["status"] = "critical"
        elif overall > 0.8:
            utilization["status"] = "high"
        elif overall > 0.6:
            utilization["status"] = "moderate"
        else:
            utilization["status"] = "normal"

        return utilization

    def analyze_trend(
        self,
        metric_history: list[tuple[datetime, float]] | None = None,
        window_hours: int = 24,
        metric_type: MetricType | None = None,
        historical_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze metric trends over time.

        Args:
            metric_history: List of (timestamp, value) tuples (legacy)
            window_hours: Hours to analyze
            metric_type: Type of metric being analyzed
            historical_data: List of dicts with 'timestamp' and 'value' keys

        Returns:
            Trend analysis results
        """
        # Handle new signature with historical_data
        if historical_data is not None:
            metric_history = [(d["timestamp"], d["value"]) for d in historical_data]

        if not metric_history:
            return {
                "trend": "unknown",
                "change_rate": 0,
                "direction": "stable",
                "rate_of_change": 0,
                "is_anomaly": False,
            }

        # Filter to window
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [(t, v) for t, v in metric_history if t >= cutoff]

        if len(recent) < 2:
            return {
                "trend": "insufficient_data",
                "change_rate": 0,
                "direction": "stable",
                "rate_of_change": 0,
                "is_anomaly": False,
            }

        # Calculate trend
        values = [v for _, v in recent]
        first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
        second_half = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)

        change_rate = (second_half - first_half) / first_half if first_half else 0

        if abs(change_rate) < 0.05:
            trend = "stable"
            direction = "stable"
        elif change_rate > 0.2:
            trend = "increasing_rapidly"
            direction = "increasing"
        elif change_rate > 0:
            trend = "increasing"
            direction = "increasing"
        elif change_rate < -0.2:
            trend = "decreasing_rapidly"
            direction = "decreasing"
        else:
            trend = "decreasing"
            direction = "decreasing"

        # Simple linear prediction for next hour
        if len(values) >= 2:
            # Calculate rate of change per hour
            hours_span = (recent[-1][0] - recent[0][0]).total_seconds() / 3600
            if hours_span > 0:
                hourly_rate = (values[-1] - values[0]) / hours_span
                prediction_next_hour = values[-1] + hourly_rate
            else:
                prediction_next_hour = values[-1]
        else:
            prediction_next_hour = values[-1] if values else 0

        # Check for anomalies using Interquartile Range (IQR) method
        # This is more robust to outliers than standard deviation
        mean = sum(values) / len(values)
        is_anomaly = False
        anomaly_score = 0.0

        if len(values) > 3:
            # Use IQR method for anomaly detection
            sorted_values = sorted(values)
            q1_idx = len(sorted_values) // 4
            q3_idx = 3 * len(sorted_values) // 4
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1

            # Define outlier bounds (1.5 * IQR is standard, 3 * IQR for extreme outliers)
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            # Also calculate using standard deviation for comparison
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = variance**0.5

            # Check each value for anomalies
            max_score = 0
            for v in values:
                # A value is anomalous if it's outside the IQR bounds
                # OR if it's more than 3 std deviations from mean
                is_outlier_iqr = v < lower_bound or v > upper_bound

                # Calculate deviation score
                if iqr > 0:
                    iqr_score = max(0, (abs(v - mean) - 1.5 * iqr) / (1.5 * iqr))
                else:
                    iqr_score = 0

                if std_dev > 0:
                    std_score = abs(v - mean) / std_dev / 3  # Normalize to 0-1 for 3 std devs
                else:
                    std_score = 0

                # Use maximum of both methods
                score = max(iqr_score, std_score)
                if score > max_score:
                    max_score = score

                # Consider it an anomaly if either method detects it
                if is_outlier_iqr or (std_dev > 0 and abs(v - mean) > 3 * std_dev):
                    is_anomaly = True

            # Anomaly score is the maximum deviation score
            anomaly_score = min(1.0, max_score)

        return {
            "trend": trend,
            "direction": direction,
            "change_rate": change_rate,
            "rate_of_change": change_rate,
            "min": min(values),
            "max": max(values),
            "avg": mean,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "prediction_next_hour": prediction_next_hour,
        }

    def detect_anomaly(self, value: float, historical_values: list[float]) -> bool:
        """
        Detect if a value is anomalous based on history.

        Args:
            value: Current value to check
            historical_values: Historical values for comparison

        Returns:
            True if anomalous, False otherwise
        """
        if len(historical_values) < 10:
            return False

        # Calculate statistics
        mean = sum(historical_values) / len(historical_values)
        variance = sum((x - mean) ** 2 for x in historical_values) / len(historical_values)
        std_dev = variance**0.5

        # Check if value is beyond 3 standard deviations
        if std_dev > 0:
            return bool(abs(value - mean) > (3 * std_dev))
        else:
            return False

    def evaluate_alert_rules(
        self, rules: list[AlertRule], metrics: dict[str, float]
    ) -> list[dict[str, Any]]:
        """
        Evaluate multiple alert rules against metrics.

        Args:
            rules: List of alert rules to evaluate
            metrics: Current metric values

        Returns:
            List of triggered alerts
        """
        triggered = []

        for rule in rules:
            metric_value = metrics.get(rule.metric_type.value, None)
            if metric_value is None:
                continue

            # Evaluate against the rule's thresholds
            thresholds = rule.thresholds
            severity = None

            if thresholds.direction == "above":
                if metric_value >= thresholds.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif metric_value >= thresholds.error_threshold:
                    severity = AlertSeverity.ERROR
                elif metric_value >= thresholds.warning_threshold:
                    severity = AlertSeverity.WARNING
            elif metric_value <= thresholds.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif metric_value <= thresholds.error_threshold:
                severity = AlertSeverity.ERROR
            elif metric_value <= thresholds.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                triggered.append(
                    {
                        "metric": rule.metric_type.value,
                        "value": metric_value,
                        "severity": severity,
                        "rule": rule,
                        "timestamp": datetime.now(),
                    }
                )

        return triggered

    def calculate_availability(self, uptime_seconds: int, total_seconds: int) -> float:
        """
        Calculate availability percentage.

        Args:
            uptime_seconds: Seconds of uptime
            total_seconds: Total seconds in period

        Returns:
            Availability percentage (0-1)
        """
        if total_seconds == 0:
            return 0.0
        return uptime_seconds / total_seconds

    def create_alert_rule(
        self,
        metric_type: MetricType,
        warning_threshold: float | None = None,
        error_threshold: float | None = None,
        critical_threshold: float | None = None,
        notification_channels: list[str] | None = None,
        cooldown_period: int = 300,
        auto_resolve: bool = True,
        escalation_policy: str | None = None,
    ) -> AlertRule:
        """
        Create an alert rule.

        Args:
            metric_type: Type of metric
            warning_threshold: Warning threshold
            error_threshold: Error threshold
            critical_threshold: Critical threshold
            notification_channels: Notification channels
            cooldown_period: Cooldown period in seconds
            auto_resolve: Whether to auto-resolve
            escalation_policy: Escalation policy

        Returns:
            AlertRule instance
        """
        # Get default thresholds for the metric type
        defaults = self._get_default_thresholds(metric_type)

        # Create thresholds with provided values or defaults
        thresholds = MetricThresholds(
            warning_threshold=warning_threshold or defaults.warning_threshold,
            error_threshold=error_threshold or defaults.error_threshold,
            critical_threshold=critical_threshold or defaults.critical_threshold,
            unit=defaults.unit,
            direction=defaults.direction,
            evaluation_period=defaults.evaluation_period,
            datapoints_required=defaults.datapoints_required,
        )

        # Determine severity based on thresholds
        if critical_threshold:
            severity = AlertSeverity.CRITICAL
        elif error_threshold:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        return AlertRule(
            metric_type=metric_type,
            thresholds=thresholds,
            severity=severity,
            notification_channels=notification_channels or [],
            cooldown_period=cooldown_period,
            auto_resolve=auto_resolve,
            escalation_policy=escalation_policy,
        )

    def get_performance_history(
        self, service: str | None = None, hours: int = 24
    ) -> list[PerformanceAssessment]:
        """
        Get performance history for a service.

        Args:
            service: Service name (optional)
            hours: Hours of history to retrieve

        Returns:
            List of performance assessments
        """
        # Use the internal method to get assessments
        return self._get_historical_assessments(service, hours)

    def _get_historical_assessments(
        self, service: str | None = None, hours: int = 24
    ) -> list[PerformanceAssessment]:
        """
        Internal method to get historical assessments.
        This would normally query a database.

        Args:
            service: Service name (optional)
            hours: Hours of history to retrieve

        Returns:
            List of performance assessments
        """
        # This would normally query a database
        # For now, return empty list
        return []

    def aggregate_metrics(
        self,
        metric_data: list[dict[str, Any]],
        aggregation_type: str = "avg",
        group_by: str | None = None,
    ) -> dict[str, float]:
        """
        Aggregate metrics data.

        Args:
            metric_data: List of metric data points
            aggregation_type: Type of aggregation (avg, sum, max, min, p95, p99, percentile)
            group_by: Optional field to group by

        Returns:
            Aggregated metrics
        """
        if not metric_data:
            return {}

        # Handle percentile aggregation specially
        if aggregation_type == "percentile":
            values = [item.get("value", 0) for item in metric_data]
            if not values:
                return {}

            sorted_values = sorted(values)
            result = {
                "p50": sorted_values[len(sorted_values) // 2],  # Median
                "p95": (
                    sorted_values[int(len(sorted_values) * 0.95)]
                    if len(sorted_values) > 1
                    else sorted_values[0]
                ),
                "p99": (
                    sorted_values[int(len(sorted_values) * 0.99)]
                    if len(sorted_values) > 1
                    else sorted_values[-1]
                ),
            }
            return result

        # Group data if requested
        if group_by:
            grouped: dict[str, list[float]] = {}
            for item in metric_data:
                key = item.get(group_by, "unknown")
                value = item.get("value", 0)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(value)

            # Aggregate each group
            result = {}
            for key, values in grouped.items():
                result[key] = self._aggregate_values(values, aggregation_type)
            return result
        else:
            # Aggregate all values
            values = [item.get("value", 0) for item in metric_data]
            return {"aggregated": self._aggregate_values(values, aggregation_type)}

    def _aggregate_values(self, values: list[float], aggregation_type: str) -> float:
        """Helper to aggregate a list of values."""
        if not values:
            return 0.0

        if aggregation_type == "avg":
            return sum(values) / len(values)
        elif aggregation_type == "sum":
            return sum(values)
        elif aggregation_type == "max":
            return max(values)
        elif aggregation_type == "min":
            return min(values)
        elif aggregation_type == "p95":
            sorted_values = sorted(values)
            index = int(len(sorted_values) * 0.95)
            return sorted_values[min(index, len(sorted_values) - 1)]
        elif aggregation_type == "p99":
            sorted_values = sorted(values)
            index = int(len(sorted_values) * 0.99)
            return sorted_values[min(index, len(sorted_values) - 1)]
        else:
            return sum(values) / len(values)  # Default to average

    def generate_capacity_plan(
        self,
        current_metrics: dict[str, float],
        months_ahead: int = 6,
    ) -> dict[str, Any]:
        """
        Generate a capacity plan based on current metrics and growth.

        Args:
            current_metrics: Current resource metrics including growth_rate
            months_ahead: Number of months to plan ahead

        Returns:
            Capacity planning recommendations
        """
        growth_rate = current_metrics.get("growth_rate", 0.1)

        # Calculate predicted usage
        predicted_cpu = current_metrics.get("cpu_usage", 0) * (1 + growth_rate * months_ahead)
        predicted_memory = current_metrics.get("memory_usage", 0) * (1 + growth_rate * months_ahead)

        # Generate scaling recommendation
        scaling_needed = predicted_cpu > 0.8 or predicted_memory > 0.8

        if scaling_needed:
            scaling_recommendation = "Scale up infrastructure to handle projected load"
        else:
            scaling_recommendation = "Current infrastructure sufficient for projected load"

        return {
            "predicted_cpu_usage": predicted_cpu,
            "predicted_memory_usage": predicted_memory,
            "scaling_recommendation": scaling_recommendation,
            "months_planned": months_ahead,
            "growth_rate": growth_rate,
        }

    def plan_capacity(
        self,
        current_usage: dict[str, float],
        growth_rate: float,
        planning_horizon_days: int = 90,
    ) -> dict[str, Any]:
        """
        Plan capacity based on current usage and growth.

        Args:
            current_usage: Current resource usage
            growth_rate: Expected growth rate (e.g., 0.1 for 10%)
            planning_horizon_days: Days to plan ahead

        Returns:
            Capacity planning recommendations
        """
        recommendations = []
        projected_usage = {}

        # Calculate projected usage
        for resource, usage in current_usage.items():
            projected = usage * (1 + growth_rate * planning_horizon_days / 30)
            projected_usage[resource] = projected

            # Generate recommendations if projected usage is high
            if projected > 0.8:
                recommendations.append(
                    {
                        "resource": resource,
                        "action": f"Scale {resource}",
                        "urgency": "high" if projected > 0.9 else "medium",
                        "projected_usage": projected,
                    }
                )

        return {
            "current_usage": current_usage,
            "projected_usage": projected_usage,
            "growth_rate": growth_rate,
            "planning_horizon_days": planning_horizon_days,
            "recommendations": recommendations,
            "scaling_needed": len(recommendations) > 0,
        }

    def correlate_alerts(
        self,
        alerts: list[dict[str, Any]],
        time_window_seconds: int = 300,
    ) -> dict[str, Any]:
        """
        Correlate related alerts to identify potential root causes.

        Args:
            alerts: List of alerts
            time_window_seconds: Time window for correlation

        Returns:
            Correlation analysis with root cause probability
        """
        if not alerts:
            return {
                "root_cause_probability": {},
                "related_alerts": [],
                "correlation_strength": 0.0,
            }

        # Group alerts by severity
        severity_groups: dict[AlertSeverity, list[dict[str, Any]]] = {}
        for alert in alerts:
            severity = alert.get("severity", AlertSeverity.WARNING)
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(alert)

        # Calculate root cause probability based on alert patterns
        root_cause_prob = {}
        total_alerts = len(alerts)

        # Higher severity alerts are more likely to be symptoms than causes
        if AlertSeverity.ERROR in severity_groups:
            # CPU/memory issues often cause other problems
            for alert in alerts:
                metric = alert.get("metric", "unknown")
                if metric in ["cpu_usage", "memory_usage"]:
                    root_cause_prob[metric] = 0.7
                elif metric == "latency":
                    root_cause_prob[metric] = 0.5
                else:
                    root_cause_prob[metric] = 0.3

        # Find related alerts (those occurring close in time)
        related = []
        for i, alert1 in enumerate(alerts):
            for alert2 in alerts[i + 1 :]:
                time1 = alert1.get("timestamp", datetime.now())
                time2 = alert2.get("timestamp", datetime.now())
                if isinstance(time1, datetime) and isinstance(time2, datetime):
                    if abs((time2 - time1).total_seconds()) <= time_window_seconds:
                        related.append(
                            {
                                "alert1": alert1.get("metric"),
                                "alert2": alert2.get("metric"),
                                "time_diff": abs((time2 - time1).total_seconds()),
                            }
                        )

        return {
            "root_cause_probability": root_cause_prob,
            "related_alerts": related,
            "correlation_strength": len(related) / max(1, total_alerts * (total_alerts - 1) / 2),
        }

    def register_custom_metric(
        self,
        name: str | None = None,
        metric_type: MetricType | None = None,
        thresholds: MetricThresholds | None = None,
        metric_name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Register a custom metric.

        Args:
            name: Name of the custom metric
            metric_type: Type of metric (MetricType enum)
            thresholds: Metric thresholds
            metric_name: Alternative name parameter (for compatibility)
            unit: Unit of measurement
            description: Optional description
        """
        # Use name or metric_name (for parameter compatibility)
        actual_name = name or metric_name or "custom_metric"

        # Store custom metric configuration (in memory for now)
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}

        if metric_type and thresholds:
            self._custom_metrics[actual_name] = {
                "type": metric_type,
                "thresholds": thresholds,
                "unit": unit or thresholds.unit if thresholds else "units",
                "description": description or f"Custom metric: {actual_name}",
            }

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
            MetricType.CACHE_HIT_RATE: MetricThresholds(
                warning_threshold=0.8,
                error_threshold=0.5,
                critical_threshold=0.3,
                unit="%",
                direction="below",
                evaluation_period=300,
                datapoints_required=5,
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

"""
Comprehensive tests for MonitoringPolicyService.

This test suite covers:
- Metric threshold validation
- Performance assessment logic
- Alert rule evaluation
- Recommendation generation
- SLA compliance checking
- Resource utilization analysis
- Trend analysis and anomaly detection
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.domain.services.monitoring_policy_service import (
    AlertRule,
    AlertSeverity,
    MetricThresholds,
    MetricType,
    MonitoringPolicyService,
    PerformanceAssessment,
    PerformanceLevel,
    Recommendation,
)


class TestMetricThresholds:
    """Test MetricThresholds dataclass."""

    def test_create_thresholds(self):
        """Test creating metric thresholds."""
        thresholds = MetricThresholds(
            warning_threshold=100,
            error_threshold=200,
            critical_threshold=500,
            unit="ms",
            direction="above",
            evaluation_period=60,
            datapoints_required=3,
        )

        assert thresholds.warning_threshold == 100
        assert thresholds.error_threshold == 200
        assert thresholds.critical_threshold == 500
        assert thresholds.unit == "ms"
        assert thresholds.direction == "above"
        assert thresholds.evaluation_period == 60
        assert thresholds.datapoints_required == 3

    def test_thresholds_with_below_direction(self):
        """Test thresholds with below direction."""
        thresholds = MetricThresholds(
            warning_threshold=0.8,
            error_threshold=0.5,
            critical_threshold=0.3,
            unit="%",
            direction="below",
            evaluation_period=300,
            datapoints_required=5,
        )

        assert thresholds.direction == "below"
        assert thresholds.warning_threshold > thresholds.critical_threshold


class TestMonitoringPolicyService:
    """Test MonitoringPolicyService functionality."""

    @pytest.fixture
    def service(self):
        """Create a monitoring policy service instance."""
        return MonitoringPolicyService()

    def test_get_default_thresholds(self, service):
        """Test getting default thresholds for metrics."""
        # Test latency thresholds
        latency_thresholds = service.get_thresholds(MetricType.LATENCY, context="api")
        assert latency_thresholds is not None
        assert latency_thresholds.unit == "ms"
        assert latency_thresholds.direction == "above"

        # Test error rate thresholds
        error_thresholds = service.get_thresholds(MetricType.ERROR_RATE, context="general")
        assert error_thresholds is not None
        assert error_thresholds.unit == "%"
        assert error_thresholds.direction == "above"

    def test_get_thresholds_with_custom_context(self, service):
        """Test getting thresholds with custom context."""
        # Database context
        db_thresholds = service.get_thresholds(MetricType.LATENCY, context="database")
        assert db_thresholds is not None
        assert (
            db_thresholds.warning_threshold
            != service.get_thresholds(MetricType.LATENCY, context="api").warning_threshold
        )

    def test_evaluate_metric(self, service):
        """Test evaluating a single metric against thresholds."""
        # Test normal value
        severity = service.evaluate_metric(metric_type=MetricType.LATENCY, value=50, context="api")
        assert severity is None  # Below warning threshold

        # Test warning level
        severity = service.evaluate_metric(metric_type=MetricType.LATENCY, value=150, context="api")
        assert severity == AlertSeverity.WARNING

        # Test error level
        severity = service.evaluate_metric(metric_type=MetricType.LATENCY, value=350, context="api")
        assert severity == AlertSeverity.ERROR

        # Test critical level
        severity = service.evaluate_metric(metric_type=MetricType.LATENCY, value=600, context="api")
        assert severity == AlertSeverity.CRITICAL

    def test_evaluate_metric_below_direction(self, service):
        """Test evaluating metrics with below direction thresholds."""
        # Test cache hit rate (below thresholds trigger alerts)
        severity = service.evaluate_metric(
            metric_type=MetricType.CACHE_HIT_RATE, value=0.95, context="general"
        )
        assert severity is None  # Above all thresholds (good)

        severity = service.evaluate_metric(
            metric_type=MetricType.CACHE_HIT_RATE, value=0.75, context="general"
        )
        assert severity == AlertSeverity.WARNING

        severity = service.evaluate_metric(
            metric_type=MetricType.CACHE_HIT_RATE, value=0.45, context="general"
        )
        assert severity == AlertSeverity.ERROR

    def test_assess_performance(self, service):
        """Test overall performance assessment."""
        metrics = {
            "latency_p50": 50,
            "latency_p95": 100,
            "latency_p99": 200,
            "error_rate": 0.001,
            "throughput": 1000,
            "cpu_usage": 0.4,
            "memory_usage": 0.5,
        }

        assessment = service.assess_performance(metrics)

        assert isinstance(assessment, PerformanceAssessment)
        assert assessment.level in PerformanceLevel
        assert 0 <= assessment.score <= 100
        assert isinstance(assessment.metrics, dict)
        assert isinstance(assessment.bottlenecks, list)
        assert isinstance(assessment.recommendations, list)
        assert isinstance(assessment.timestamp, datetime)

    def test_assess_performance_excellent(self, service):
        """Test performance assessment for excellent metrics."""
        metrics = {
            "latency_p50": 10,
            "latency_p95": 20,
            "latency_p99": 30,
            "error_rate": 0.0001,
            "throughput": 5000,
            "cpu_usage": 0.2,
            "memory_usage": 0.3,
        }

        assessment = service.assess_performance(metrics)

        assert assessment.level == PerformanceLevel.EXCELLENT
        assert assessment.score >= 90
        assert len(assessment.bottlenecks) == 0

    def test_assess_performance_poor(self, service):
        """Test performance assessment for poor metrics."""
        metrics = {
            "latency_p50": 500,
            "latency_p95": 1000,
            "latency_p99": 2000,
            "error_rate": 0.05,
            "throughput": 100,
            "cpu_usage": 0.95,
            "memory_usage": 0.9,
        }

        assessment = service.assess_performance(metrics)

        assert assessment.level in [PerformanceLevel.POOR, PerformanceLevel.UNACCEPTABLE]
        assert assessment.score < 50
        assert len(assessment.bottlenecks) > 0
        assert len(assessment.recommendations) > 0

    def test_identify_bottlenecks(self, service):
        """Test bottleneck identification."""
        metrics = {
            "latency_p99": 1500,  # High latency
            "cpu_usage": 0.95,  # High CPU
            "memory_usage": 0.5,  # Normal memory
            "disk_usage": 0.85,  # High disk
            "error_rate": 0.001,  # Normal errors
        }

        bottlenecks = service.identify_bottlenecks(metrics)

        assert "high_latency" in bottlenecks
        assert "cpu_saturation" in bottlenecks
        assert "disk_pressure" in bottlenecks
        assert "memory_pressure" not in bottlenecks

    def test_generate_recommendations(self, service):
        """Test recommendation generation."""
        bottlenecks = ["high_latency", "cpu_saturation", "memory_pressure"]
        metrics = {
            "latency_p99": 1000,
            "cpu_usage": 0.9,
            "memory_usage": 0.85,
        }

        recommendations = service.generate_recommendations(bottlenecks, metrics)

        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.priority >= 1 and rec.priority <= 5
            assert rec.effort_estimate in ["low", "medium", "high"]
            assert rec.risk_level in ["low", "medium", "high"]

    def test_check_sla_compliance(self, service):
        """Test SLA compliance checking."""
        metrics = {
            "availability": 0.999,  # 99.9%
            "latency_p99": 200,
            "error_rate": 0.001,
        }

        sla_requirements = {
            "availability": 0.99,  # 99%
            "latency_p99": 500,
            "error_rate": 0.01,
        }

        compliance = service.check_sla_compliance(metrics, sla_requirements)

        assert compliance["is_compliant"] is True
        assert compliance["availability"]["compliant"] is True
        assert compliance["latency_p99"]["compliant"] is True
        assert compliance["error_rate"]["compliant"] is True

    def test_check_sla_violation(self, service):
        """Test SLA violation detection."""
        metrics = {
            "availability": 0.95,  # Below SLA
            "latency_p99": 600,  # Above SLA
            "error_rate": 0.001,  # Within SLA
        }

        sla_requirements = {
            "availability": 0.99,
            "latency_p99": 500,
            "error_rate": 0.01,
        }

        compliance = service.check_sla_compliance(metrics, sla_requirements)

        assert compliance["is_compliant"] is False
        assert compliance["availability"]["compliant"] is False
        assert compliance["latency_p99"]["compliant"] is False
        assert compliance["error_rate"]["compliant"] is True
        assert len(compliance["violations"]) == 2

    def test_calculate_resource_utilization(self, service):
        """Test resource utilization calculation."""
        metrics = {
            "cpu_usage": 0.6,
            "memory_usage": 0.7,
            "disk_usage": 0.5,
            "network_io": 0.4,
        }

        utilization = service.calculate_resource_utilization(metrics)

        assert "overall_utilization" in utilization
        assert 0 <= utilization["overall_utilization"] <= 1
        assert utilization["cpu_efficiency"] is not None
        assert utilization["memory_efficiency"] is not None
        assert "resource_balance" in utilization

    def test_analyze_trend(self, service):
        """Test trend analysis."""
        historical_data = [
            {"timestamp": datetime.now() - timedelta(hours=5), "value": 100},
            {"timestamp": datetime.now() - timedelta(hours=4), "value": 110},
            {"timestamp": datetime.now() - timedelta(hours=3), "value": 120},
            {"timestamp": datetime.now() - timedelta(hours=2), "value": 135},
            {"timestamp": datetime.now() - timedelta(hours=1), "value": 150},
            {"timestamp": datetime.now(), "value": 170},
        ]

        trend = service.analyze_trend(
            metric_type=MetricType.LATENCY, historical_data=historical_data
        )

        assert trend["direction"] == "increasing"
        assert trend["rate_of_change"] > 0
        assert trend["is_anomaly"] is False
        assert "prediction_next_hour" in trend

    def test_detect_anomaly(self, service):
        """Test anomaly detection."""
        historical_data = [
            {"timestamp": datetime.now() - timedelta(hours=5), "value": 100},
            {"timestamp": datetime.now() - timedelta(hours=4), "value": 105},
            {"timestamp": datetime.now() - timedelta(hours=3), "value": 102},
            {"timestamp": datetime.now() - timedelta(hours=2), "value": 98},
            {"timestamp": datetime.now() - timedelta(hours=1), "value": 500},  # Anomaly
            {"timestamp": datetime.now(), "value": 103},
        ]

        trend = service.analyze_trend(
            metric_type=MetricType.LATENCY, historical_data=historical_data
        )

        assert trend["is_anomaly"] is True
        assert trend["anomaly_score"] > 0.5

    def test_create_alert_rule(self, service):
        """Test creating alert rules."""
        rule = service.create_alert_rule(
            metric_type=MetricType.ERROR_RATE,
            warning_threshold=0.01,
            error_threshold=0.05,
            critical_threshold=0.1,
            notification_channels=["email", "slack"],
            cooldown_period=300,
        )

        assert isinstance(rule, AlertRule)
        assert rule.metric_type == MetricType.ERROR_RATE
        assert rule.severity in AlertSeverity
        assert len(rule.notification_channels) == 2
        assert rule.cooldown_period == 300

    def test_evaluate_alert_rules(self, service):
        """Test evaluating multiple alert rules."""
        rules = [
            service.create_alert_rule(
                MetricType.LATENCY,
                warning_threshold=100,
                error_threshold=200,
                critical_threshold=500,
                notification_channels=["email"],
            ),
            service.create_alert_rule(
                MetricType.ERROR_RATE,
                warning_threshold=0.01,
                error_threshold=0.05,
                critical_threshold=0.1,
                notification_channels=["slack"],
            ),
        ]

        metrics = {
            "latency": 250,  # Error level
            "error_rate": 0.02,  # Warning level
        }

        triggered_alerts = service.evaluate_alert_rules(rules, metrics)

        assert len(triggered_alerts) == 2
        assert any(alert["severity"] == AlertSeverity.ERROR for alert in triggered_alerts)
        assert any(alert["severity"] == AlertSeverity.WARNING for alert in triggered_alerts)

    def test_calculate_availability(self, service):
        """Test availability calculation."""
        uptime_seconds = 86400 - 300  # 24 hours minus 5 minutes
        total_seconds = 86400

        availability = service.calculate_availability(uptime_seconds, total_seconds)

        assert availability > 0.99
        assert availability < 1.0

    def test_get_performance_history(self, service):
        """Test getting performance history."""
        with patch.object(service, "_get_historical_assessments") as mock_history:
            mock_history.return_value = [
                PerformanceAssessment(
                    level=PerformanceLevel.GOOD,
                    score=75,
                    metrics={},
                    bottlenecks=[],
                    recommendations=[],
                    timestamp=datetime.now() - timedelta(hours=1),
                ),
                PerformanceAssessment(
                    level=PerformanceLevel.EXCELLENT,
                    score=92,
                    metrics={},
                    bottlenecks=[],
                    recommendations=[],
                    timestamp=datetime.now(),
                ),
            ]

            history = service.get_performance_history(hours=24)

            assert len(history) == 2
            assert history[-1].score > history[0].score

    def test_metric_aggregation(self, service):
        """Test metric aggregation logic."""
        raw_metrics = [
            {"timestamp": datetime.now(), "value": 100},
            {"timestamp": datetime.now(), "value": 150},
            {"timestamp": datetime.now(), "value": 200},
            {"timestamp": datetime.now(), "value": 120},
            {"timestamp": datetime.now(), "value": 180},
        ]

        aggregated = service.aggregate_metrics(raw_metrics, aggregation_type="percentile")

        assert "p50" in aggregated
        assert "p95" in aggregated
        assert "p99" in aggregated
        assert aggregated["p50"] <= aggregated["p95"]
        assert aggregated["p95"] <= aggregated["p99"]

    def test_capacity_planning(self, service):
        """Test capacity planning recommendations."""
        current_metrics = {
            "cpu_usage": 0.7,
            "memory_usage": 0.6,
            "throughput": 1000,
            "growth_rate": 0.1,  # 10% monthly growth
        }

        capacity_plan = service.generate_capacity_plan(current_metrics, months_ahead=6)

        assert "predicted_cpu_usage" in capacity_plan
        assert "predicted_memory_usage" in capacity_plan
        assert "scaling_recommendation" in capacity_plan
        assert capacity_plan["predicted_cpu_usage"] > current_metrics["cpu_usage"]

    def test_alert_correlation(self, service):
        """Test correlating multiple alerts."""
        alerts = [
            {"metric": "cpu_usage", "severity": AlertSeverity.ERROR, "timestamp": datetime.now()},
            {
                "metric": "memory_usage",
                "severity": AlertSeverity.WARNING,
                "timestamp": datetime.now(),
            },
            {"metric": "latency", "severity": AlertSeverity.ERROR, "timestamp": datetime.now()},
        ]

        correlated = service.correlate_alerts(alerts)

        assert "root_cause_probability" in correlated
        assert "related_alerts" in correlated
        assert len(correlated["related_alerts"]) > 0

    def test_custom_metric_registration(self, service):
        """Test registering custom business metrics."""
        service.register_custom_metric(
            name="order_processing_time",
            metric_type=MetricType.BUSINESS_METRIC,
            thresholds=MetricThresholds(
                warning_threshold=5000,
                error_threshold=10000,
                critical_threshold=20000,
                unit="ms",
                direction="above",
                evaluation_period=60,
                datapoints_required=3,
            ),
        )

        # Should be able to evaluate custom metric
        severity = service.evaluate_metric(
            metric_type=MetricType.BUSINESS_METRIC, value=7500, context="order_processing_time"
        )

        assert severity == AlertSeverity.WARNING

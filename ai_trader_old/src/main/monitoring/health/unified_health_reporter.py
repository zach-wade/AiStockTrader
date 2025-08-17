"""
Unified Health Reporter

This module provides comprehensive health reporting using the UnifiedMetrics
and UnifiedAlertSystem for automated health monitoring and reporting.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import statistics
from typing import Any

# Local imports
from main.monitoring.alerts.unified_alert_integration import UnifiedAlertIntegration
from main.monitoring.metrics.unified_metrics import UnifiedMetrics
from main.utils.core import ErrorHandlingMixin, get_logger

logger = get_logger(__name__)


class UnifiedHealthReporter(ErrorHandlingMixin):
    """
    Unified health reporter for comprehensive system health monitoring.

    Features:
    - Automated health score calculation
    - Scheduled health reports
    - Performance trend analysis
    - Integration with unified metrics and alerts
    - Email report generation
    """

    def __init__(
        self,
        unified_metrics: UnifiedMetrics,
        alert_integration: UnifiedAlertIntegration,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize unified health reporter.

        Args:
            unified_metrics: UnifiedMetrics instance
            alert_integration: UnifiedAlertIntegration instance
            config: Configuration for reporting
        """
        super().__init__()
        self.unified_metrics = unified_metrics
        self.alert_integration = alert_integration
        self.config = config or {}

        # Report configuration
        self.report_dir = Path(self.config.get("report_directory", "reports/health"))
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_history: dict[str, list[dict[str, Any]]] = {}
        self.max_history_days = self.config.get("max_history_days", 30)

        # Reporting schedule
        self.reporting_schedule = self.config.get(
            "schedule",
            {
                "daily": {"enabled": True, "hour": 9},
                "weekly": {"enabled": True, "day": 0, "hour": 9},  # Monday
                "monthly": {"enabled": True, "day": 1, "hour": 9},
            },
        )

        # Health thresholds
        self.health_thresholds = self.config.get(
            "health_thresholds", {"excellent": 95, "good": 85, "fair": 70, "poor": 50}
        )

        # Background tasks
        self._scheduler_task: asyncio.Task | None = None
        self._tracking_task: asyncio.Task | None = None
        self._running = False

        # Last report times
        self._last_daily_report: datetime | None = None
        self._last_weekly_report: datetime | None = None
        self._last_monthly_report: datetime | None = None

        logger.info("UnifiedHealthReporter initialized")

    async def start(self):
        """Start automated health reporting."""
        with self._handle_error("starting health reporter"):
            if self._running:
                logger.warning("Health reporter already running")
                return

            self._running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self._tracking_task = asyncio.create_task(self._tracking_loop())

            logger.info("Health reporter started")

    async def stop(self):
        """Stop automated health reporting."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass

        logger.info("Health reporter stopped")

    async def generate_health_report(self) -> dict[str, Any]:
        """Generate comprehensive health report."""
        with self._handle_error("generating health report"):
            # Get system health score
            health_score_data = await self.unified_metrics.get_system_health_score()

            # Get metric statistics
            metric_stats = self.unified_metrics.get_statistics()

            # Get alert statistics
            alert_stats = self.alert_integration.get_metrics()

            # Get active alerts
            active_alerts = await self.unified_metrics.get_active_alerts()

            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()

            # Build report
            report = {
                "timestamp": datetime.now(UTC).isoformat(),
                "health_score": health_score_data,
                "health_status": self._determine_health_status(health_score_data["overall_score"]),
                "metrics": {
                    "total_metrics": metric_stats["registered_metrics"],
                    "processed": metric_stats["metrics_processed"],
                    "active_alerts": len(active_alerts),
                    "alert_history": alert_stats["alert_system_metrics"],
                },
                "performance": performance_metrics,
                "recommendations": self._generate_recommendations(health_score_data, active_alerts),
                "active_alerts": [
                    {
                        "metric": alert.metric_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "triggered_at": alert.triggered_at.isoformat(),
                    }
                    for alert in active_alerts
                ],
                "top_issues": self._identify_top_issues(health_score_data, active_alerts),
            }

            return report

    async def generate_daily_report(self) -> None:
        """Generate and save daily health report."""
        try:
            logger.info("Generating daily health report")
            report = await self.generate_health_report()

            # Add daily-specific information
            report["report_type"] = "daily"
            report["period"] = {
                "start": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
                "end": datetime.now(UTC).isoformat(),
            }

            # Save report
            filename = f"daily_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.report_dir / filename

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Daily health report saved to {filepath}")

            # Send alert if health is poor
            if report["health_status"] in ["poor", "critical"]:
                await self.alert_integration.alert_system_error(
                    component="health_reporter",
                    error_message=f"System health is {report['health_status']} with score {report['health_score']['overall_score']:.1f}",
                    error_type="health_degradation",
                )

            self._last_daily_report = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Error generating daily report: {e}")

    async def generate_weekly_report(self) -> None:
        """Generate and save weekly health report."""
        try:
            logger.info("Generating weekly health report")
            report = await self.generate_health_report()

            # Add weekly-specific information
            report["report_type"] = "weekly"
            report["period"] = {
                "start": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
                "end": datetime.now(UTC).isoformat(),
            }

            # Add trend analysis
            report["trends"] = await self._analyze_weekly_trends()

            # Save report
            filename = f"weekly_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.report_dir / filename

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Weekly health report saved to {filepath}")

            self._last_weekly_report = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")

    async def generate_monthly_report(self) -> None:
        """Generate and save monthly health report."""
        try:
            logger.info("Generating monthly health report")
            report = await self.generate_health_report()

            # Add monthly-specific information
            report["report_type"] = "monthly"
            report["period"] = {
                "start": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
                "end": datetime.now(UTC).isoformat(),
            }

            # Add comprehensive analysis
            report["monthly_analysis"] = await self._analyze_monthly_performance()

            # Save report
            filename = f"monthly_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.report_dir / filename

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Monthly health report saved to {filepath}")

            self._last_monthly_report = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")

    async def _scheduler_loop(self):
        """Background loop for scheduled reporting."""
        while self._running:
            try:
                now = datetime.now(UTC)

                # Check daily report
                if self._should_generate_daily_report(now):
                    await self.generate_daily_report()

                # Check weekly report
                if self._should_generate_weekly_report(now):
                    await self.generate_weekly_report()

                # Check monthly report
                if self._should_generate_monthly_report(now):
                    await self.generate_monthly_report()

                # Sleep for 1 hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(3600)

    async def _tracking_loop(self):
        """Background loop for performance tracking."""
        while self._running:
            try:
                await self._capture_performance_snapshot()
                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                await asyncio.sleep(300)

    async def _capture_performance_snapshot(self):
        """Capture performance snapshot for trend analysis."""
        health_score = await self.unified_metrics.get_system_health_score()

        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": health_score["overall_score"],
            "status": health_score["status"],
            "active_alerts": health_score["active_alerts"],
            "metrics": health_score["metric_scores"],
        }

        # Store in history
        history_key = datetime.now(UTC).strftime("%Y-%m-%d")
        if history_key not in self.performance_history:
            self.performance_history[history_key] = []

        self.performance_history[history_key].append(snapshot)

        # Clean old history
        self._clean_old_history()

    def _should_generate_daily_report(self, now: datetime) -> bool:
        """Check if daily report should be generated."""
        if not self.reporting_schedule["daily"]["enabled"]:
            return False

        if self._last_daily_report:
            if (now - self._last_daily_report).days < 1:
                return False

        return now.hour == self.reporting_schedule["daily"]["hour"]

    def _should_generate_weekly_report(self, now: datetime) -> bool:
        """Check if weekly report should be generated."""
        if not self.reporting_schedule["weekly"]["enabled"]:
            return False

        if self._last_weekly_report:
            if (now - self._last_weekly_report).days < 7:
                return False

        return (
            now.weekday() == self.reporting_schedule["weekly"]["day"]
            and now.hour == self.reporting_schedule["weekly"]["hour"]
        )

    def _should_generate_monthly_report(self, now: datetime) -> bool:
        """Check if monthly report should be generated."""
        if not self.reporting_schedule["monthly"]["enabled"]:
            return False

        if self._last_monthly_report:
            if (now - self._last_monthly_report).days < 28:
                return False

        return (
            now.day == self.reporting_schedule["monthly"]["day"]
            and now.hour == self.reporting_schedule["monthly"]["hour"]
        )

    def _determine_health_status(self, score: float) -> str:
        """Determine health status from score."""
        if score >= self.health_thresholds["excellent"]:
            return "excellent"
        elif score >= self.health_thresholds["good"]:
            return "good"
        elif score >= self.health_thresholds["fair"]:
            return "fair"
        elif score >= self.health_thresholds["poor"]:
            return "poor"
        else:
            return "critical"

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics summary."""
        # Get trading metrics
        trading_metrics = {}
        for metric_name in ["trading.orders.executed", "trading.pnl.unrealized"]:
            value = await self.unified_metrics.get_metric_value(metric_name, "sum", 60)
            if value is not None:
                trading_metrics[metric_name] = value

        # Get system metrics
        system_metrics = {}
        for metric_name in ["system.cpu.usage_percent", "system.memory.usage_percent"]:
            value = await self.unified_metrics.get_metric_value(metric_name, "avg", 60)
            if value is not None:
                system_metrics[metric_name] = value

        return {"trading": trading_metrics, "system": system_metrics}

    def _generate_recommendations(
        self, health_score_data: dict[str, Any], active_alerts: list[Any]
    ) -> list[str]:
        """Generate recommendations based on health data."""
        recommendations = []

        # Check overall health
        if health_score_data["overall_score"] < self.health_thresholds["good"]:
            recommendations.append(
                "System health is below optimal levels. Review active alerts and metrics."
            )

        # Check specific metrics
        for metric_name, metric_data in health_score_data["metric_scores"].items():
            if metric_data["score"] < 70:
                if "cpu" in metric_name:
                    recommendations.append(
                        "High CPU usage detected. Consider scaling resources or optimizing processes."
                    )
                elif "memory" in metric_name:
                    recommendations.append(
                        "High memory usage detected. Check for memory leaks or increase available memory."
                    )
                elif "disk" in metric_name:
                    recommendations.append(
                        "High disk usage detected. Clean up old data or increase storage capacity."
                    )

        # Check alerts
        critical_alerts = [a for a in active_alerts if a.severity.value == "critical"]
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately.")

        return recommendations

    def _identify_top_issues(
        self, health_score_data: dict[str, Any], active_alerts: list[Any]
    ) -> list[dict[str, Any]]:
        """Identify top issues affecting system health."""
        issues = []

        # Add low-scoring metrics
        for metric_name, metric_data in health_score_data["metric_scores"].items():
            if metric_data["score"] < 70:
                issues.append(
                    {
                        "type": "metric",
                        "name": metric_name,
                        "score": metric_data["score"],
                        "value": metric_data["value"],
                        "severity": "high" if metric_data["score"] < 50 else "medium",
                    }
                )

        # Add critical alerts
        for alert in active_alerts:
            if alert.severity.value in ["critical", "warning"]:
                issues.append(
                    {
                        "type": "alert",
                        "name": alert.metric_name,
                        "message": alert.message,
                        "severity": alert.severity.value,
                    }
                )

        # Sort by severity and return top 5
        issues.sort(
            key=lambda x: (
                0
                if x.get("severity") == "critical"
                else (
                    1 if x.get("severity") == "high" else 2 if x.get("severity") == "warning" else 3
                )
            )
        )

        return issues[:5]

    async def _analyze_weekly_trends(self) -> dict[str, Any]:
        """Analyze weekly performance trends."""
        trends = {}

        # Get 7-day metric series
        for metric_name in [
            "system.cpu.usage_percent",
            "system.memory.usage_percent",
            "trading.orders.executed",
            "trading.pnl.unrealized",
        ]:
            series = await self.unified_metrics.get_metric_series(
                metric_name, period_minutes=7 * 24 * 60
            )

            if series:
                values = [point["value"] for point in series]
                trends[metric_name] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": "increasing" if values[-1] > values[0] else "decreasing",
                }

        return trends

    async def _analyze_monthly_performance(self) -> dict[str, Any]:
        """Analyze monthly performance comprehensively."""
        analysis = {"health_trend": [], "alert_frequency": {}, "performance_summary": {}}

        # Analyze health trend from history
        for day_key in sorted(self.performance_history.keys())[-30:]:
            snapshots = self.performance_history[day_key]
            if snapshots:
                daily_avg = statistics.mean(s["health_score"] for s in snapshots)
                analysis["health_trend"].append({"date": day_key, "avg_score": daily_avg})

        # Get alert history
        alert_history = self.alert_integration.get_alert_history(limit=1000)
        for alert in alert_history:
            category = alert.category.value
            analysis["alert_frequency"][category] = analysis["alert_frequency"].get(category, 0) + 1

        return analysis

    def _clean_old_history(self):
        """Clean performance history older than max_history_days."""
        cutoff_date = (datetime.now(UTC) - timedelta(days=self.max_history_days)).strftime(
            "%Y-%m-%d"
        )

        old_keys = [key for key in self.performance_history.keys() if key < cutoff_date]
        for key in old_keys:
            del self.performance_history[key]

    def get_latest_health_status(self) -> dict[str, Any]:
        """Get latest health status summary."""
        latest_snapshot = None

        # Get most recent snapshot
        for day_key in sorted(self.performance_history.keys(), reverse=True):
            if self.performance_history[day_key]:
                latest_snapshot = self.performance_history[day_key][-1]
                break

        if latest_snapshot:
            return {
                "timestamp": latest_snapshot["timestamp"],
                "health_score": latest_snapshot["health_score"],
                "status": latest_snapshot["status"],
                "active_alerts": latest_snapshot["active_alerts"],
            }

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "health_score": 0,
            "status": "unknown",
            "active_alerts": 0,
        }

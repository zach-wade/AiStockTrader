"""
Examples of using the unified monitoring system.

This file demonstrates various usage patterns for the monitoring system.
"""

# Standard library imports
import asyncio

# Local imports
from main.utils.monitoring import (
    MetricType,
    get_system_summary,
    record_metric,
    start_monitoring,
    stop_monitoring,
    timer,
)
from main.utils.monitoring.dashboard_adapters import (
    DashboardHealthReporter,
    create_dashboard_adapter,
)
from main.utils.monitoring.migration import create_monitor


async def basic_monitoring_example():
    """Example of basic monitoring usage."""
    print("=== Basic Monitoring Example ===")

    # Start monitoring
    await start_monitoring()

    # Record some metrics
    record_metric("example.counter", 1, MetricType.COUNTER)
    record_metric("example.gauge", 42.5, MetricType.GAUGE)
    record_metric(
        "example.api_latency", 150, MetricType.TIMER, tags={"endpoint": "/users", "method": "GET"}
    )

    # Use timer decorator
    @timer("example.function")
    async def example_function():
        await asyncio.sleep(0.1)
        return "done"

    result = await example_function()

    # Get system summary
    summary = get_system_summary()
    print(f"System CPU: {summary['current']['cpu_percent']:.1f}%")
    print(f"System Memory: {summary['current']['memory_percent']:.1f}%")

    await stop_monitoring()


async def enhanced_monitoring_example(db_pool=None):
    """Example of enhanced monitoring with database."""
    print("\n=== Enhanced Monitoring Example ===")

    # Create enhanced monitor
    monitor = create_monitor(db_pool=db_pool)
    await monitor.start_monitoring()

    # Register metric with thresholds
    monitor.register_metric_with_thresholds(
        name="example.error_rate",
        metric_type=MetricType.GAUGE,
        warning_threshold=0.05,  # 5% warning
        critical_threshold=0.10,  # 10% critical
        description="Example error rate percentage",
    )

    # Simulate some metrics
    for i in range(10):
        # Normal error rate
        monitor.record_metric("example.error_rate", 0.02)  # 2%

        # API metrics
        monitor.record_metric("api.request_count", 1, MetricType.COUNTER)
        monitor.record_metric("api.request_duration", 100 + i * 10, MetricType.TIMER)

        await asyncio.sleep(0.1)

    # Simulate high error rate (should trigger alert)
    monitor.record_metric("example.error_rate", 0.15)  # 15% - critical!

    # Get metric value with aggregation
    avg_latency = await monitor.get_metric_value("api.request_duration", "avg", period_minutes=1)
    print(f"Average API latency: {avg_latency:.1f}ms")

    # Get time series
    series = await monitor.get_metric_series("api.request_count", period_minutes=1)
    print(f"API requests in last minute: {len(series)}")

    # Check health score
    health = monitor.get_system_health_score()
    print(f"System health: {health['status']} (score: {health['overall_score']})")
    print(f"Active alerts: {health['active_alerts']}")

    await monitor.stop_monitoring()


async def dashboard_integration_example(db_pool=None):
    """Example of dashboard integration."""
    print("\n=== Dashboard Integration Example ===")

    # Create dashboard adapter
    adapter = create_dashboard_adapter(db_pool)

    # Start monitoring
    monitor = adapter.monitor
    await monitor.start_monitoring()

    # Record some trading metrics
    adapter.record_metric("trading.orders.executed", 10)
    adapter.record_metric("trading.orders.failed", 1)
    adapter.record_metric("trading.pnl.unrealized", 1500.50)
    adapter.record_metric("portfolio.total_value", 100000)

    # Get dashboard-friendly data
    health_score = await adapter.get_system_health_score()
    print(f"Dashboard health score: {health_score['overall_score']}")

    # Create health reporter
    health_reporter = DashboardHealthReporter(adapter)
    report = await health_reporter.generate_health_report()

    print("\n--- Health Report ---")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Overall Status: {report['health_score']['status']}")
    print(f"CPU Usage: {report['system_metrics']['cpu']['current']:.1f}%")
    print(f"Memory Usage: {report['system_metrics']['memory']['current']:.1f}%")
    print(f"Active Alerts: {len(report['active_alerts'])}")

    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

    # Get performance metrics
    # Local imports
    from main.utils.monitoring.dashboard_adapters import DashboardPerformanceTracker

    perf_tracker = DashboardPerformanceTracker(adapter)

    perf_metrics = await perf_tracker.get_performance_metrics("1h")
    print("\nPerformance Metrics (last hour):")
    for key, value in perf_metrics["metrics"].items():
        print(f"  {key}: {value}")

    await monitor.stop_monitoring()


async def migration_example():
    """Example showing migration from old to new system."""
    print("\n=== Migration Example ===")

    # Old style (still works!)
    # Local imports
    from main.utils.monitoring import record_metric

    record_metric("old_style_metric", 42)

    # New style with enhanced features
    # Local imports
    from main.utils.monitoring.global_monitor import get_global_monitor

    monitor = get_global_monitor()

    # Check if we have enhanced features
    if hasattr(monitor, "use_enhanced"):
        print(f"Enhanced features: {'enabled' if monitor.use_enhanced else 'disabled'}")

    # The monitor automatically uses enhanced features if DB is available
    await monitor.start_monitoring()

    # Everything works the same
    monitor.record_metric("new_style_metric", 100, MetricType.GAUGE)

    await monitor.stop_monitoring()


async def main():
    """Run all examples."""
    # Basic monitoring (always available)
    await basic_monitoring_example()

    # Try to get database pool for enhanced examples
    db_pool = None
    try:
        # Local imports
        from main.utils.database import get_default_db_pool

        db_pool = get_default_db_pool()
        print("\nDatabase available - running enhanced examples")
    except Exception:
        print("\nNo database available - skipping enhanced examples")

    if db_pool:
        await enhanced_monitoring_example(db_pool)
        await dashboard_integration_example(db_pool)

    # Migration example
    await migration_example()


if __name__ == "__main__":
    asyncio.run(main())

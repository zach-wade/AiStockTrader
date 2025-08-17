# Monitoring Module Documentation

## Overview

The monitoring module provides comprehensive system monitoring, alerting, and visualization capabilities for the AI Trading System. It follows a clear separation of concerns between presentation layer (dashboards) and shared utilities.

## Architecture

### Module Structure

```
monitoring/
├── dashboards/              # Presentation layer - UI and visualization
│   ├── unified_trading_dashboard.py
│   ├── unified_system_dashboard.py
│   ├── economic_dashboard.py
│   ├── api/                # REST API endpoints
│   ├── websocket/          # Real-time data streaming
│   ├── events/             # Event-driven updates
│   └── services/           # Data aggregation services
├── metrics/                # Metrics collection and aggregation
│   ├── unified_metrics.py  # Main metrics coordination
│   └── collector.py        # Metrics collection implementation
├── alerts/                 # Alert system
│   ├── unified_alerts.py   # Main alert coordination
│   ├── alert_manager.py    # High-level alert orchestration
│   └── unified_alert_integration.py
├── health/                 # Health reporting
│   └── unified_health_reporter.py
├── performance/            # Performance tracking
│   ├── performance_tracker.py
│   ├── calculators/        # Performance calculations
│   ├── models/             # Data models
│   └── alerts/             # Performance-specific alerts
└── performance_dashboard.py # Specialized database performance tool

utils/
├── monitoring/             # Shared monitoring utilities
│   ├── metrics/            # Metrics utilities
│   │   ├── buffer.py       # Metric buffering
│   │   └── exporter.py     # Export to Prometheus, etc.
│   └── alerts/             # Alert channels
│       ├── email_channel.py
│       ├── slack_channel.py
│       └── sms_channel.py
└── logging/                # Specialized loggers
    ├── trade_logger.py
    ├── performance_logger.py
    └── error_logger.py
```

## Design Principles

### 1. Separation of Concerns

- **monitoring/**: Contains presentation layer (dashboards, UI components)
- **utils/**: Contains shared services and utilities
- Clear boundary between UI concerns and business logic

### 2. Unified Systems

- **UnifiedMetrics**: Centralized metrics collection and aggregation
- **UnifiedAlerts**: Centralized alert routing and delivery
- **UnifiedHealthReporter**: Comprehensive health reporting

### 3. Modularity

- Each dashboard is self-contained with its own services
- Alert channels are pluggable and configurable
- Performance tracking is modular with separate calculators

## Components

### Dashboards

#### UnifiedTradingDashboard (Port 8050)

- Real-time trading metrics and portfolio performance
- Risk analytics and position monitoring
- Integration with economic indicators
- WebSocket support for live updates

#### UnifiedSystemDashboard (Port 8052)

- System health monitoring
- Resource utilization tracking
- Service status and alerts
- Component dependency visualization

#### Performance Dashboard (Port 8888)

- Database performance metrics
- Query analysis and optimization
- Index management tools
- Cache performance monitoring
- **Note**: Specialized tool for database administrators

### Metrics System

#### UnifiedMetrics

- Centralized metric collection
- Database persistence (PostgreSQL)
- Automatic aggregation and rollups
- Integration with utils monitoring

#### MetricsCollector

- Low-level metric collection
- In-memory buffering
- Export capabilities

### Alert System

#### UnifiedAlertSystem

- Multi-channel alert delivery (Email, Slack, SMS)
- Alert routing based on severity and category
- Rate limiting and deduplication
- Circuit breaker protection

#### AlertManager

- High-level alert orchestration
- Configuration-based channel setup
- Convenience methods for common alerts
- Backward compatibility layer

### Performance Tracking

#### PerformanceTracker

- Trading performance metrics
- Risk-adjusted returns
- System performance monitoring
- Alert generation for threshold breaches

## Usage

### Starting Dashboards

```bash
# Trading Dashboard
python -m main.monitoring.dashboards.unified_trading_dashboard

# System Dashboard
python -m main.monitoring.dashboards.unified_system_dashboard

# Database Performance Dashboard
python -m main.monitoring.performance_dashboard --port 8888
```

### Configuration

Dashboards and monitoring components are configured through the main application config:

```yaml
monitoring:
  dashboards:
    trading:
      port: 8050
      debug: false
    system:
      port: 8052
      debug: false
  alerts:
    email:
      enabled: true
      smtp_host: smtp.gmail.com
      # ... email config
    slack:
      enabled: true
      webhooks:
        general: https://hooks.slack.com/...
    sms:
      enabled: false
      # ... Twilio config
```

### Integration

The monitoring module integrates with the main application through:

1. **Database Pool**: Shared database connections
2. **Event Bus**: Real-time event notifications
3. **Utils Monitoring**: Shared monitoring infrastructure

Example integration:

```python
from main.monitoring.metrics import UnifiedMetrics
from main.monitoring.alerts import UnifiedAlertSystem

# Initialize with database pool
metrics = UnifiedMetrics(db_pool)
alerts = UnifiedAlertSystem(config)

# Record metrics
await metrics.record_metric("trading.order.placed", 1)

# Send alerts
await alerts.send_alert(
    title="Risk Limit Exceeded",
    message="Portfolio VaR exceeds limit",
    level=AlertLevel.WARNING,
    category=AlertCategory.RISK
)
```

## Migration Notes

### Recent Changes (2025-07-29)

1. **Moved to utils/**:
   - Logging module → `utils/logging/`
   - Alert channels → `utils/monitoring/alerts/`
   - Metrics utilities → `utils/monitoring/metrics/`

2. **Deleted deprecated files**:
   - `metrics_collector.py` (replaced by UnifiedMetrics)
   - `alerting_system.py` (replaced by UnifiedAlerts)
   - `alert_integration.py` (replaced by UnifiedAlertIntegration)
   - `health_reporter.py` (replaced by UnifiedHealthReporter)
   - `health_monitor.db` (now uses PostgreSQL)

3. **Kept for compatibility**:
   - `metrics/collector.py` (used by utils)
   - `alerts/alert_manager.py` (useful orchestration layer)
   - `performance/alerts/` (specialized for performance alerts)

### Breaking Changes

- Import paths for logging have changed:

  ```python
  # OLD
  from main.monitoring.logging import TradeLogger

  # NEW
  from main.utils.logging import TradeLogger
  ```

- Alert system no longer requires alerting_system.py imports

## Best Practices

1. **Use Unified Systems**: Prefer UnifiedMetrics and UnifiedAlerts over direct implementations
2. **Dashboard Services**: Use service classes for data aggregation in dashboards
3. **Event-Driven Updates**: Use EventBus for real-time dashboard updates
4. **Configuration**: Keep all configuration in central config files
5. **Error Handling**: Use circuit breakers for external services

## Troubleshooting

### Dashboards Not Starting

- Check if ports are already in use
- Verify database pool is initialized
- Ensure all required services are running

### Alerts Not Sending

- Verify channel configuration in config
- Check channel-specific credentials
- Review alert routing rules

### Metrics Not Recording

- Ensure database connection is active
- Check metric naming conventions
- Verify UnifiedMetrics is initialized

## Future Enhancements

1. **Grafana Integration**: Export metrics to Grafana for advanced visualization
2. **Alert Templates**: Customizable alert templates for different scenarios
3. **Dashboard Plugins**: Plugin system for custom dashboard widgets
4. **Mobile Alerts**: Push notifications for mobile devices
5. **ML-based Anomaly Detection**: Automatic alert generation from anomalies

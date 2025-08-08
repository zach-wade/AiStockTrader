# System Dashboard V2 Documentation

## Overview

The System Dashboard V2 is a comprehensive monitoring interface for system health, data pipeline status, infrastructure metrics, and analytical model performance. It provides real-time insights into the technical operations of the AI trading system.

**URL**: http://localhost:8052  
**Technology Stack**: Python, Dash, Plotly, psutil, AsyncIO  
**Update Frequency**: 5 seconds (system metrics), 30 seconds (pipeline status)

## Architecture

### Dashboard Structure
```
SystemDashboardV2
├── Tab 1: System Health
├── Tab 2: Data Pipeline
├── Tab 3: Infrastructure
└── Tab 4: Analytics
```

### Data Collection
1. **System Metrics**: Direct OS-level monitoring via `psutil`
2. **Database Queries**: Pipeline and component status from database
3. **Process Monitoring**: Real-time process health checks
4. **Resource Tracking**: CPU, memory, disk, network utilization

## Tab Details

### Tab 1: System Health

#### Components

1. **Resource Usage Gauges**
   - **CPU Usage**: Real-time CPU utilization percentage
     - Data Source: `psutil.cpu_percent(interval=1)`
     - Color coding: Green (<60%), Yellow (60-80%), Red (>80%)
   
   - **Memory Usage**: RAM utilization
     - Data Source: `psutil.virtual_memory()`
     - Shows: Used/Total GB and percentage
   
   - **Disk Usage**: Storage utilization
     - Data Source: `psutil.disk_usage('/')`
     - Monitors: Root filesystem by default

2. **System Metrics Time Series**
   - 60-minute rolling window
   - Metrics tracked:
     - CPU percentage (per core and average)
     - Memory usage (MB)
     - Disk I/O (read/write MB/s)
     - Network I/O (sent/received MB/s)
   - Update frequency: Every 5 seconds
   - Data stored in-memory (not persisted)

3. **Component Health Status**
   - **Database Connection**: PostgreSQL health
     - Checks: Connection pool status, active connections
     - Data Source: Database pool metrics
   
   - **Market Data Feed**: Real-time data stream health
     - Monitors: Alpaca/Polygon connection status
     - Latency tracking: Milliseconds delay
   
   - **Trading Engine**: Core engine status
     - Process health check
     - Last heartbeat timestamp
   
   - **Feature Pipeline**: Data processing status
     - Queue depth monitoring
     - Processing rate (records/second)

4. **Active Processes Table**
   - All system-related processes
   - Columns: PID, Name, CPU%, Memory MB, Status, Uptime
   - Data Source: `psutil.process_iter()`
   - Filtered for: python, postgres, redis processes

#### Data Collection Methods
```python
# CPU Monitoring
cpu_percent = psutil.cpu_percent(interval=1)
cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

# Memory Monitoring
memory = psutil.virtual_memory()
memory_used_gb = memory.used / (1024**3)
memory_percent = memory.percent

# Disk Monitoring
disk = psutil.disk_usage('/')
disk_io = psutil.disk_io_counters()

# Network Monitoring
network = psutil.net_io_counters()
bytes_sent = network.bytes_sent
bytes_recv = network.bytes_recv
```

### Tab 2: Data Pipeline

#### Components

1. **Pipeline Flow Diagram**
   - Visual representation of data flow
   - Stages: Ingestion → Processing → Storage → Analysis
   - Color-coded status per stage
   - Click for detailed metrics per stage

2. **Data Source Status Grid**
   - **Alpaca Market Data**
     - Connection status
     - Messages/second rate
     - Last update timestamp
     - Error count (24h)
   
   - **Polygon.io**
     - API health status
     - Request rate (per minute)
     - Quota usage
     - Response time (ms)
   
   - **Yahoo Finance**
     - Availability status
     - Fetch success rate
     - Average response time
   
   - **News APIs**
     - Combined status for news sources
     - Articles processed/hour
     - Sentiment analysis queue depth

3. **Ingestion Metrics**
   - **Real-time Feeds**
     - WebSocket connections: Active count
     - Message rate: Per second
     - Lag: Milliseconds behind real-time
     - Error rate: Percentage failed
   
   - **Batch Processing**
     - Active jobs count
     - Records processed (24h)
     - Average processing time
     - Failed jobs queue

4. **Data Quality Indicators**
   - Missing data percentage by source
   - Data validation failures
   - Duplicate detection rate
   - Schema compliance score

#### Database Queries
```sql
-- Pipeline Status Query
SELECT 
    component_name,
    status,
    last_update,
    metrics_json,
    error_count
FROM pipeline_status
WHERE last_update > NOW() - INTERVAL '5 minutes';

-- Data Source Health
SELECT 
    source_name,
    is_active,
    last_successful_fetch,
    total_fetches_24h,
    error_rate_24h
FROM data_source_status;
```

### Tab 3: Infrastructure

#### Components

1. **Server Metrics Dashboard**
   - **CPU Details**
     - Model and core count
     - Temperature monitoring (if available)
     - Load average (1, 5, 15 min)
     - Context switches/sec
   
   - **Memory Breakdown**
     - Physical RAM usage
     - Swap usage and activity
     - Cache and buffer memory
     - Memory pressure indicators

2. **Storage Analytics**
   - **Disk Space by Mount**
     - All mounted filesystems
     - Space utilization percentages
     - I/O operations per second
     - Read/Write throughput MB/s
   
   - **Database Storage**
     - Table sizes ranking
     - Index sizes and efficiency
     - WAL size and archival status
     - Vacuum/analyze status

3. **Network Statistics**
   - **Interface Metrics**
     - All network interfaces
     - Bandwidth utilization
     - Packet statistics (sent/received/dropped)
     - Error and collision rates
   
   - **Connection Tracking**
     - Active TCP connections
     - Connection states distribution
     - Top bandwidth consumers
     - Port utilization

4. **Process Resource Usage**
   - Top 10 CPU consumers
   - Top 10 memory consumers
   - Process tree visualization
   - Zombie process detection

#### System Monitoring Code
```python
# Detailed CPU Info
cpu_info = {
    'physical_cores': psutil.cpu_count(logical=False),
    'logical_cores': psutil.cpu_count(logical=True),
    'frequency': psutil.cpu_freq(),
    'load_average': os.getloadavg()
}

# Storage Details
partitions = psutil.disk_partitions()
for partition in partitions:
    usage = psutil.disk_usage(partition.mountpoint)
    io_counters = psutil.disk_io_counters(perdisk=True)
```

### Tab 4: Analytics

#### Components

1. **Model Performance Metrics**
   - **Active Models Grid**
     - Model name and version
     - Accuracy/performance score
     - Predictions per hour
     - Last training date
     - Resource usage
   
   - **Model Comparison Chart**
     - Performance over time
     - A/B test results
     - Prediction accuracy trends

2. **Feature Engineering Status**
   - **Feature Calculation Pipeline**
     - Active feature sets
     - Calculation latency
     - Cache hit rates
     - Feature importance rankings
   
   - **Feature Store Metrics**
     - Total features stored
     - Storage size per feature set
     - Access patterns heatmap
     - Staleness indicators

3. **Backtesting Dashboard**
   - Active backtest jobs
   - Completed backtests (last 7 days)
   - Resource allocation
   - Performance metrics summary
   - Queue depth and wait times

4. **System Intelligence**
   - Anomaly detection alerts
   - Performance optimization suggestions
   - Resource usage predictions
   - Capacity planning insights

#### Analytics Queries
```sql
-- Model Performance
SELECT 
    model_name,
    version,
    accuracy_score,
    predictions_count_24h,
    avg_latency_ms,
    last_updated
FROM model_performance
ORDER BY accuracy_score DESC;

-- Feature Usage
SELECT 
    feature_name,
    calculation_time_ms,
    usage_count_24h,
    cache_hit_rate,
    last_calculated
FROM feature_metrics
WHERE is_active = true;
```

## Data Sources and Updates

### Real-time Metrics (5 seconds)
- System resources (CPU, memory, disk, network)
- Process health checks
- Active connection counts

### Near Real-time (30 seconds)
- Pipeline component status
- Data source health
- Queue depths

### Periodic Updates (5 minutes)
- Model performance metrics
- Feature engineering statistics
- Infrastructure analytics

### Daily Aggregations
- Historical performance trends
- Capacity planning data
- Cost analysis metrics

## Performance Monitoring

### Dashboard Performance
- **Render Time**: Target <100ms per update
- **Data Fetch**: Parallel async queries
- **Memory Usage**: ~50-100MB typical
- **CPU Impact**: <2% during updates

### Optimization Techniques
1. **Data Sampling**: Large datasets sampled for visualization
2. **Caching**: Recent metrics cached in memory
3. **Batch Updates**: Multiple metrics fetched together
4. **Selective Rendering**: Only visible tabs update

## Alert Integration

### System Alerts
1. **Critical Alerts**
   - CPU > 90% for 5 minutes
   - Memory > 95%
   - Disk space < 10%
   - Component health failures

2. **Warning Alerts**
   - CPU > 80% for 10 minutes
   - Memory > 85%
   - Disk space < 20%
   - High error rates

3. **Info Alerts**
   - Component restarts
   - Configuration changes
   - Maintenance events

### Alert Actions
- Dashboard notification banner
- Database alert record
- Optional webhook notifications
- Email alerts (if configured)

## Security and Access

### Data Protection
1. **Sensitive Data**: Database passwords masked
2. **Process Filtering**: Only system processes shown
3. **Resource Limits**: Queries timeout after 30s
4. **Access Control**: Dashboard access via localhost only

### Audit Trail
- All configuration changes logged
- Alert acknowledgments tracked
- Performance baseline changes recorded

## Future Enhancements

### Planned Features

1. **Advanced Monitoring**
   - Distributed tracing integration
   - APM (Application Performance Monitoring)
   - Custom metric definitions
   - Predictive failure analysis

2. **Infrastructure Automation**
   - Auto-scaling triggers
   - Automated remediation actions
   - Deployment pipeline integration
   - Backup/restore monitoring

3. **Enhanced Analytics**
   - ML model drift detection
   - Feature importance evolution
   - Cost-performance optimization
   - Workload prediction

4. **Visualization Improvements**
   - 3D resource utilization maps
   - Interactive dependency graphs
   - Real-time log streaming
   - Custom dashboard builder

5. **Integration Capabilities**
   - Prometheus/Grafana export
   - PagerDuty integration
   - Slack notifications
   - JIRA ticket creation

### Technical Roadmap

1. **Architecture Evolution**
   - Microservices monitoring
   - Container orchestration metrics
   - Service mesh integration
   - Multi-cloud support

2. **Performance Enhancements**
   - Time-series database backend
   - Streaming analytics engine
   - Edge computing metrics
   - GPU monitoring support

3. **Operational Intelligence**
   - AIOps capabilities
   - Root cause analysis
   - Capacity forecasting
   - Automated runbooks

## Troubleshooting Guide

### Common Issues

1. **High Resource Usage**
   ```bash
   # Check top processes
   ps aux | sort -rk 3,3 | head -20  # By CPU
   ps aux | sort -rk 4,4 | head -20  # By Memory
   
   # Database connections
   psql -c "SELECT count(*) FROM pg_stat_activity;"
   ```

2. **Component Health Failures**
   - Verify network connectivity
   - Check service logs
   - Restart failed components
   - Review error patterns

3. **Data Pipeline Issues**
   - Check data source credentials
   - Verify network access
   - Review rate limits
   - Monitor queue backlogs

### Debug Tools

1. **Enable Verbose Logging**
   ```python
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. **Performance Profiling**
   ```python
   import cProfile
   cProfile.run('dashboard.update_callback()')
   ```

3. **Resource Tracking**
   ```bash
   # Monitor dashboard process
   top -p $(pgrep -f "system_dashboard")
   ```

## Maintenance Procedures

### Daily Checks
- [ ] Review critical alerts
- [ ] Check component health status
- [ ] Monitor resource trends
- [ ] Verify data pipeline flow

### Weekly Tasks
- [ ] Analyze performance metrics
- [ ] Review error logs
- [ ] Update baseline thresholds
- [ ] Test alert mechanisms

### Monthly Activities
- [ ] Capacity planning review
- [ ] Performance optimization
- [ ] Security audit
- [ ] Documentation updates

## Configuration Reference

### Environment Variables
```bash
# System Monitoring
MONITOR_CPU_THRESHOLD=80
MONITOR_MEMORY_THRESHOLD=85
MONITOR_DISK_THRESHOLD=90

# Data Pipeline
PIPELINE_HEALTH_CHECK_INTERVAL=30
PIPELINE_ERROR_THRESHOLD=0.05

# Infrastructure
INFRA_METRIC_RETENTION_DAYS=30
INFRA_SAMPLING_RATE=5

# Analytics
MODEL_PERFORMANCE_THRESHOLD=0.7
FEATURE_STALENESS_HOURS=24
```

### Dashboard Settings
```python
# Update intervals (seconds)
SYSTEM_METRICS_INTERVAL = 5
PIPELINE_STATUS_INTERVAL = 30
INFRASTRUCTURE_INTERVAL = 60
ANALYTICS_INTERVAL = 300

# Display limits
MAX_PROCESSES_SHOWN = 50
MAX_TIMELINE_POINTS = 720  # 1 hour at 5s intervals
MAX_LOG_ENTRIES = 1000
```

---
*Last Updated: 2025-07-30*
*Version: 2.0*
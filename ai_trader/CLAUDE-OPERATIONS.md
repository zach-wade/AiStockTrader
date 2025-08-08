# CLAUDE-OPERATIONS.md - Operational Procedures & Troubleshooting

This document provides operational procedures, service management, and troubleshooting guides for the AI Trading System.

---

## 🚨 PRE-PRODUCTION CHECKLIST

### CRITICAL: Code Review Not Complete (Phase 5 In Progress)

**WARNING**: As of 2025-08-09, only 8.3% of the codebase has been reviewed.
- 722 of 787 files (91.7%) have NEVER been reviewed for correctness
- System passes initialization tests but actual functionality is unverified
- Multiple production blockers remain

#### 0. Complete Code Review (PHASE 5 - IN PROGRESS)
- [ ] Review data_pipeline module (170 files, 40K lines)
- [ ] Review feature_pipeline module (90 files, 44K lines)
- [ ] Review utils module (145 files, 36K lines)
- [ ] Review models module (101 files, 24K lines)
- [ ] Review trading_engine module (33 files, 13K lines)
- [ ] Refactor 146 files that are >500 lines
- [ ] Remove duplicate code and deprecated modules

#### 1. Replace All Test Implementations
- [ ] **TestPositionManager** → Real PositionManager
  - Location: `test_helpers/test_position_manager.py`
  - Risk: Position tracking will fail
  - Issue: ISSUE-059
- [ ] Search codebase for "TEST IMPLEMENTATION" warnings
- [ ] Verify no test helpers are imported in production code

#### 2. API Verification
- [ ] Test Polygon API with live market data
- [ ] Verify Alpaca paper trading for 1+ week
- [ ] Confirm rate limits are properly configured
- [ ] Test circuit breakers under load

#### 3. Integration Testing
- [ ] Run full end-to-end test with real components
- [ ] Test multi-symbol concurrent trading
- [ ] Verify risk management under stress
- [ ] Test graceful shutdown and recovery

#### 4. Database & Infrastructure
- [ ] Verify partition management is working
- [ ] Test backup and recovery procedures
- [ ] Confirm monitoring and alerting
- [ ] Review resource requirements

#### 5. Risk Management Validation
- [ ] Verify position limits enforcement
- [ ] Test stop-loss mechanisms
- [ ] Validate circuit breaker triggers
- [ ] Confirm drawdown controls

### Production Deployment Steps
1. Complete ALL checklist items above
2. Review with team/stakeholders
3. Start with minimal capital
4. Monitor closely for first week
5. Scale gradually based on performance

---

## 🚀 Service Management

### Starting Services

#### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d postgres redis

# Start with rebuild
docker-compose up -d --build app

# View logs
docker-compose logs -f app
docker-compose logs --tail=100 postgres
```

#### Manual Start (Development)
```bash
# 1. Start PostgreSQL
pg_ctl start -D /usr/local/var/postgres

# 2. Start Redis
redis-server /usr/local/etc/redis.conf

# 3. Activate Python environment
source venv/bin/activate

# 4. Start application
python ai_trader.py trade --mode paper
```

### Stopping Services
```bash
# Stop all Docker services
docker-compose down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose down -v

# Stop specific service
docker-compose stop app

# Emergency shutdown
python ai_trader.py shutdown --force
```

### Service Health Checks
```bash
# Check all service status
docker-compose ps

# Check specific service health
docker exec aitrader-db pg_isready
docker exec aitrader-cache redis-cli ping

# Application health check
curl http://localhost:8000/health

# Database connection test
psql -h localhost -U ai_trader -c "SELECT 1"

# Redis connection test
redis-cli -h localhost ping
```

---

## 📅 Job Scheduling Operations

### JobScheduler Location (Updated 2025-08-08)
**Important**: JobScheduler has been relocated from `/scripts/scheduler/` to proper module location:
- **Class Location**: `/src/main/orchestration/job_scheduler.py`
- **CLI Script**: `/scripts/scheduler/master_scheduler.py` (thin wrapper)
- **Config File**: `/scripts/scheduler/job_definitions.yaml`

### Running Scheduled Jobs
```bash
# Start job scheduler daemon
python scripts/scheduler/master_scheduler.py

# Run specific job once
python scripts/scheduler/master_scheduler.py --run-job data_backfill

# Check job status
python scripts/scheduler/master_scheduler.py --status

# Using from Python code
from main.orchestration import JobScheduler
scheduler = JobScheduler()
scheduler.run_job("data_backfill")
```

### Managing Job Definitions
- Edit `/scripts/scheduler/job_definitions.yaml`
- Jobs support dependencies, retries, and market hours awareness
- Resource limits enforced (CPU, memory)

---

## 🗄️ Database Operations

### Connecting to Database
```bash
# Via psql
psql -h localhost -p 5432 -U ai_trader -d ai_trader

# Via Docker
docker exec -it aitrader-db psql -U ai_trader

# Connection string format
postgresql://ai_trader:password@localhost:5432/ai_trader
```

### Common Database Queries
```sql
-- Check data freshness
SELECT symbol, MAX(timestamp) as latest
FROM market_data_1h
GROUP BY symbol
ORDER BY latest DESC
LIMIT 10;

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Active connections
SELECT pid, usename, application_name, state, query_start
FROM pg_stat_activity
WHERE state != 'idle';

-- Check partitions
SELECT 
    parent.relname AS parent_table,
    child.relname AS partition_name,
    pg_size_pretty(pg_relation_size(child.oid)) AS size
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
ORDER BY parent.relname, child.relname;
```

### Database Maintenance
```bash
# Run migrations
python scripts/run_migrations.py

# Create missing partitions
python scripts/create_partitions.py --days-ahead 30

# Vacuum and analyze
psql -c "VACUUM ANALYZE;"

# Backup database
pg_dump -h localhost -U ai_trader ai_trader > backup_$(date +%Y%m%d).sql

# Restore database
psql -h localhost -U ai_trader ai_trader < backup_20240101.sql

# Archive old data
python scripts/archive_old_data.py --days 90
```

### Troubleshooting Database Issues

#### Connection Pool Exhaustion
```bash
# Check connection count
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
psql -c "SELECT pg_terminate_backend(pid) 
         FROM pg_stat_activity 
         WHERE state = 'idle' 
         AND state_change < now() - interval '10 minutes';"

# Increase connection limit (postgresql.conf)
max_connections = 200
```

#### Slow Queries
```sql
-- Find slow queries
SELECT 
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 100
    AND correlation < 0.1
ORDER BY n_distinct DESC;
```

---

## 📝 Log Management

### Log File Locations
```
logs/
├── ai_trader.log           # Main application log
├── trading/
│   ├── orders.log         # Order execution logs
│   ├── positions.log      # Position changes
│   └── risk.log          # Risk events
├── backfill/
│   ├── market_data.log    # Market data ingestion
│   ├── news.log          # News ingestion
│   └── errors.log        # Backfill errors
├── scanner/
│   ├── layer0.log        # Universe scanner
│   ├── layer1.log        # Liquidity scanner
│   └── alerts.log        # Scanner alerts
└── errors/
    ├── critical.log       # Critical errors
    └── exceptions.log     # Stack traces
```

### Viewing Logs
```bash
# Tail main log
tail -f logs/ai_trader.log

# Search for errors
grep ERROR logs/ai_trader.log | tail -20

# View today's trading activity
grep "$(date +%Y-%m-%d)" logs/trading/orders.log

# Count errors by type
awk '/ERROR/ {print $5}' logs/ai_trader.log | sort | uniq -c | sort -rn

# View logs in Docker
docker-compose logs -f app | grep ERROR
```

### Log Rotation
```bash
# Manual rotation
mv logs/ai_trader.log logs/ai_trader.log.$(date +%Y%m%d)
touch logs/ai_trader.log

# Configure logrotate (/etc/logrotate.d/aitrader)
/path/to/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 aitrader aitrader
}
```

### Common Log Patterns
```
# Successful trade execution
INFO [2024-01-01 09:30:15] Order executed: BUY 100 AAPL @ 150.25

# API rate limit
WARNING [2024-01-01 09:30:15] Rate limit approaching: 95/100 requests

# Connection error
ERROR [2024-01-01 09:30:15] Database connection failed: timeout

# Critical system error
CRITICAL [2024-01-01 09:30:15] Trading engine crashed: insufficient funds
```

---

## 📊 Code Review & Audit

### Audit Documentation (August 2025)
The project is undergoing a comprehensive audit. Reference these documents:

- **[PROJECT_AUDIT.md](PROJECT_AUDIT.md)** - Audit methodology and current findings
- **[ISSUE_REGISTRY.md](ISSUE_REGISTRY.md)** - All known issues with P0-P3 priorities
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete code structure analysis
- **[review_progress.json](review_progress.json)** - Track review progress by module

### Checking Review Status
```bash
# View current review progress
cat review_progress.json | jq '.statistics'

# Check module-specific status
cat review_progress.json | jq '.modules.trading_engine'

# Count remaining files to review
cat review_progress.json | jq '.statistics.files_reviewed, .statistics.total_files'

# Run code inventory analysis
./code_inventory_v2.sh
```

### Updating Audit Documents
When fixing issues or reviewing code:
1. Update `review_progress.json` with files reviewed
2. Mark issues as resolved in `ISSUE_REGISTRY.md`
3. Update statistics in `PROJECT_AUDIT.md`
4. Document any new issues found

### Priority Issue Categories
- **P0 (Critical)**: System breaking, must fix immediately
- **P1 (High)**: Major functionality broken
- **P2 (Medium)**: Performance/quality issues
- **P3 (Low)**: Code quality/maintenance

---

## 🔍 Monitoring & Metrics

### Accessing Dashboards
```bash
# Grafana Dashboard
http://localhost:3000
Default: admin/admin

# Prometheus Metrics
http://localhost:9090

# Application Metrics
http://localhost:8000/metrics
```

### Key Metrics to Monitor

#### System Health
```bash
# CPU usage
docker stats aitrader-app

# Memory usage
ps aux | grep python | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Disk usage
df -h /path/to/data_lake

# Network latency to APIs
ping api.polygon.io
```

#### Trading Metrics
```python
# Check via CLI
python ai_trader.py status --metrics

# Metrics endpoint
curl http://localhost:8000/metrics | grep trading_

# Key metrics:
# - trading_orders_total
# - trading_positions_open
# - trading_pnl_total
# - trading_risk_score
```

### Alert Configuration
```yaml
# prometheus_alerts.yml
groups:
  - name: trading_alerts
    rules:
      - alert: HighDrawdown
        expr: trading_drawdown > 0.1
        for: 5m
        annotations:
          summary: "Drawdown exceeds 10%"
          
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        annotations:
          summary: "PostgreSQL is down"
```

---

## 🛠️ Troubleshooting Guide

**📝 Note**: Check [ISSUE_REGISTRY.md](ISSUE_REGISTRY.md) for complete list of 50+ known issues with priorities and status.

### Common Issues & Solutions

#### Issue: API Rate Limit Exceeded
```bash
# Symptoms
ERROR: HTTP 429 Too Many Requests

# Solution 1: Check current usage
python scripts/check_api_usage.py

# Solution 2: Reduce concurrent requests
# In config/yaml/defaults/rate_limits.yaml
polygon:
  requests_per_second: 5  # Reduce from 10

# Solution 3: Enable rate limit backoff
export ENABLE_RATE_LIMIT_BACKOFF=true
```

#### Issue: Memory Leak / High Memory Usage
```bash
# Diagnose
python -m memory_profiler ai_trader.py status

# Solution 1: Reduce batch sizes
# In config/yaml/defaults/data.yaml
batch_size: 500  # Reduce from 1000

# Solution 2: Increase garbage collection
export PYTHONOPTIMIZE=1
export PYTHONHASHSEED=0

# Solution 3: Restart periodically
crontab -e
0 */6 * * * docker-compose restart app
```

#### Issue: Database Performance Degradation
```bash
# Diagnose
psql -c "SELECT * FROM pg_stat_user_tables;"

# Solution 1: Run VACUUM
psql -c "VACUUM ANALYZE;"

# Solution 2: Rebuild indexes
psql -c "REINDEX DATABASE ai_trader;"

# Solution 3: Archive old data
python scripts/archive_old_data.py --days 30
```

#### Issue: Order Execution Failures
```bash
# Check broker connection
python scripts/test_broker_connection.py

# Check account status
python ai_trader.py status --account

# Review order logs
grep "Order failed" logs/trading/orders.log

# Common causes:
# - Insufficient buying power
# - Symbol not tradable
# - Market closed
# - Position limit exceeded
```

---

## 🔄 Data Pipeline Operations

### Backfill Operations
```bash
# Full backfill for symbol
python ai_trader.py backfill --symbols AAPL --days 90 --stage all

# Backfill specific data type
python ai_trader.py backfill --symbols AAPL --days 30 --stage market_data

# Backfill layer
python ai_trader.py backfill --layer layer1 --days 30

# Resume failed backfill
python ai_trader.py backfill --resume --job-id 12345

# Check backfill status
python scripts/check_backfill_status.py
```

### Data Validation
```bash
# Validate data completeness
python scripts/validate_data.py --symbols AAPL --days 7

# Check for gaps
python scripts/find_data_gaps.py --table market_data_1h

# Verify data quality
python ai_trader.py validate --component data_pipeline
```

### Archive Management
```bash
# Check archive size
du -sh data_lake/raw/

# Compress old files
find data_lake/raw -name "*.parquet" -mtime +30 -exec gzip {} \;

# Sync to S3 (if configured)
aws s3 sync data_lake/raw/ s3://aitrader-data-lake/raw/

# Restore from archive
python scripts/restore_from_archive.py --date 2024-01-01 --symbol AAPL
```

---

## 🔌 API Testing & Integration

### Testing Polygon API
```bash
# Test connection
curl -X GET "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-01" \
  -H "Authorization: Bearer $POLYGON_API_KEY"

# Check rate limits
curl -I "https://api.polygon.io/v2/reference/tickers" \
  -H "Authorization: Bearer $POLYGON_API_KEY" | grep -i rate

# Test data quality
python scripts/test_polygon_data.py --symbol AAPL
```

### Testing Alpaca API
```bash
# Test connection
curl -X GET "https://paper-api.alpaca.markets/v2/account" \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET"

# Check market status
curl -X GET "https://paper-api.alpaca.markets/v2/clock" \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY"

# Test order placement (paper)
python scripts/test_order_placement.py --symbol AAPL --qty 1
```

### WebSocket Testing
```bash
# Test market data stream
python scripts/test_websocket.py --stream market_data

# Test order updates stream
python scripts/test_websocket.py --stream trade_updates

# Monitor WebSocket health
netstat -an | grep 8001
```

---

## 🚨 Emergency Procedures

### Emergency Market Exit
```bash
# 1. Halt all trading immediately
python ai_trader.py shutdown --force

# 2. Close all positions
python scripts/emergency_liquidate.py --confirm

# 3. Cancel all orders
python scripts/cancel_all_orders.py

# 4. Notify
python scripts/send_alert.py --level critical --message "Emergency shutdown executed"
```

### Data Corruption Recovery
```bash
# 1. Stop services
docker-compose down

# 2. Backup corrupted data
cp -r data_lake data_lake.backup

# 3. Restore from last known good
python scripts/restore_from_backup.py --date yesterday

# 4. Validate
python ai_trader.py validate --all

# 5. Restart
docker-compose up -d
```

### System Recovery
```bash
# 1. Check system status
python ai_trader.py status --detailed

# 2. Clear caches
redis-cli FLUSHALL

# 3. Reset connections
python scripts/reset_connections.py

# 4. Reinitialize
python scripts/init_system.py --clean

# 5. Test components
python ai_trader.py validate --all
```

---

## 📊 Performance Tuning

### Database Optimization
```sql
-- Update statistics
ANALYZE;

-- Find missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE n_distinct > 100 AND correlation < 0.5;

-- Create suggested indexes
CREATE INDEX CONCURRENTLY idx_market_data_symbol_timestamp 
ON market_data_1h(symbol, timestamp DESC);
```

### Python Optimization
```bash
# Profile CPU usage
python -m cProfile -o profile.stats ai_trader.py features --symbols AAPL

# Analyze profile
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler ai_trader.py

# Use PyPy for better performance (experimental)
pypy3 ai_trader.py trade --mode paper
```

### Cache Optimization
```bash
# Check Redis memory usage
redis-cli INFO memory

# Set memory limit
redis-cli CONFIG SET maxmemory 4gb

# Configure eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Monitor cache hit rate
redis-cli INFO stats | grep keyspace_hits
```

---

## 🔐 Security Operations

### API Key Rotation
```bash
# 1. Generate new keys from provider dashboards

# 2. Update environment
export ALPACA_API_KEY_NEW="new_key"
export POLYGON_API_KEY_NEW="new_key"

# 3. Test new keys
python scripts/test_api_keys.py --new

# 4. Rotate keys
python scripts/rotate_api_keys.py

# 5. Verify
python ai_trader.py status --check-apis
```

### Audit Logging
```bash
# Check authentication logs
grep "AUTH" logs/security.log

# Monitor API usage
python scripts/audit_api_usage.py --days 7

# Review trading activity
python scripts/audit_trades.py --start-date 2024-01-01
```

---

*Last Updated: 2025-08-08 22:30 (Phase 3.0 - All systems operational)*  
*System Status: FULLY FUNCTIONAL (10/10 components passing)*
*Version: 1.2*
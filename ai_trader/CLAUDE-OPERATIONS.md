# CLAUDE-OPERATIONS.md - Operational Procedures & Troubleshooting

This document provides operational procedures, service management, and troubleshooting guides for the AI Trading System.

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

*Last Updated: 2025-08-08*
*Version: 1.0*
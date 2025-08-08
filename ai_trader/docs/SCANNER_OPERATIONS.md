# Scanner Operations Guide

## Overview

This guide provides operational procedures for managing the Layer 0-3 scanning system in production, including monitoring, troubleshooting, and maintenance workflows.

## Production Operations

### Daily Operations Checklist

#### Pre-Market (6:00 AM EST)
```bash
# 1. Universe Health Check
python ai_trader.py universe --health
# ✅ Healthy: >8000 Layer 0, >1500 Layer 1, >15 Layer 3

# 2. Refresh Universe (Layer 0)
python ai_trader.py universe --populate
# Expected: ~10,000 symbols discovered, <60 seconds runtime

# 3. Verify Layer 1 Completion
python ai_trader.py universe --stats | grep "Layer 1"
# Expected: 2000-3000 qualified symbols

# 4. Check Critical Backfill Coverage
python ai_trader.py validate --component data --layer 3
# Expected: >95% data coverage for Layer 3 symbols
```

#### Market Hours (9:30 AM - 4:00 PM EST)
```bash
# 1. Monitor Layer 2 Technical Scanning (every 30 minutes)
python ai_trader.py universe --layer 2 | wc -l
# Expected: 50-200 symbols

# 2. Check Real-time Data Flow
python ai_trader.py status --component data | grep "last_update"
# Expected: <5 minutes ago

# 3. Monitor Trading Universe Stability
LAYER3_COUNT=$(python ai_trader.py universe --layer 3 | wc -l)
echo "Layer 3 symbols: $LAYER3_COUNT"
# Expected: 20-50 symbols, <20% change from previous day
```

#### Post-Market (4:30 PM EST)
```bash
# 1. Verify Layer 3 Fundamental Analysis
python ai_trader.py universe --layer 3 --verbose
# Check for completion timestamp within last hour

# 2. Compare Daily Universe Changes
python ai_trader.py universe --stats > /tmp/today_stats.txt
diff /tmp/yesterday_stats.txt /tmp/today_stats.txt
# Review significant changes

# 3. Check Event Processing
python ai_trader.py status --component events | grep "scanner"
# Verify no backlog or failed events
```

### Weekly Operations

#### Sunday Evening Maintenance
```bash
# 1. Full Universe Refresh
python ai_trader.py universe --populate --force-refresh
# Discovers new IPOs, removes delisted symbols

# 2. Historical Data Validation
python ai_trader.py validate --component data --full-scan
# Checks data integrity across all layers

# 3. Performance Review
python ai_trader.py universe --performance-report --days 7
# Reviews qualification rates and model performance

# 4. Configuration Review
python ai_trader.py universe --config-audit
# Validates all layer configurations and thresholds
```

## Monitoring and Alerting

### Key Performance Indicators

#### Universe Health Metrics
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Layer 0 Symbols | >8,000 | <8,000 | <5,000 |
| Layer 1 Qualification Rate | 20-30% | <15% or >40% | <10% or >50% |
| Layer 2 Symbols | 50-200 | <30 or >300 | <15 or >500 |
| Layer 3 Symbols | 20-50 | <15 or >70 | <10 or >100 |
| Layer 3 Stability | <20% daily change | 20-30% | >30% |

#### Data Quality Metrics
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Layer 3 Data Coverage | >95% | 90-95% | <90% |
| Backfill Lag | <1 hour | 1-6 hours | >6 hours |
| Feature Sync Lag | <15 minutes | 15-60 minutes | >1 hour |
| Event Processing Lag | <5 minutes | 5-30 minutes | >30 minutes |

### Automated Monitoring Setup

#### Prometheus Metrics
```yaml
# prometheus_scanner_metrics.yml
scanner_metrics:
  - layer_symbol_count{layer="0,1,2,3"}
  - layer_qualification_rate{layer="1,2,3"}  
  - scanner_execution_duration{layer="0,1,2,3"}
  - scanner_failure_count{layer="0,1,2,3"}
  - data_coverage_percentage{layer="3"}
  - universe_change_rate{layer="3", period="daily"}
```

#### Grafana Dashboard Queries
```sql
-- Layer 3 Universe Stability
rate(layer_symbol_count{layer="3"}[1h]) * 100

-- Data Coverage by Layer
avg(data_coverage_percentage{layer="3"}) by (symbol)

-- Scanner Performance
histogram_quantile(0.95, scanner_execution_duration{layer="1"})
```

#### Alert Rules
```yaml
# scanner_alerts.yml
alerts:
  - alert: Layer3UniverseTooSmall
    expr: layer_symbol_count{layer="3"} < 15
    for: 15m
    labels:
      severity: critical
    annotations:
      summary: "Layer 3 universe has insufficient symbols"
      
  - alert: Layer1QualificationRateLow  
    expr: layer_qualification_rate{layer="1"} < 0.15
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Layer 1 qualification rate below normal"

  - alert: DataCoverageLow
    expr: data_coverage_percentage{layer="3"} < 90
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Layer 3 data coverage below threshold"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Layer 0: Universe Discovery Failures

**Symptoms**:
- Layer 0 symbol count drops below 8,000
- Universe population command fails
- New symbols not appearing

**Diagnosis**:
```bash
# Check API connectivity
python ai_trader.py universe --health --verbose

# Test data source connections
python ai_trader.py validate --component data-sources

# Check API rate limits
grep "rate_limit" /var/log/ai_trader/scanner.log
```

**Solutions**:
```bash
# 1. Verify API credentials
echo $ALPACA_API_KEY | wc -c  # Should be >20 characters
echo $POLYGON_API_KEY | wc -c  # Should be >20 characters

# 2. Check network connectivity
curl -s "https://paper-api.alpaca.markets/v2/assets" \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" | jq '.[] | length'

# 3. Clear cache and retry
rm -rf /tmp/ai_trader_cache/universe/*
python ai_trader.py universe --populate --force-refresh

# 4. Fallback to manual symbol list
python ai_trader.py universe --populate --source manual --file config/symbols/fallback_universe.txt
```

#### Layer 1: Low Qualification Rates

**Symptoms**:
- Layer 1 qualification rate <15%
- Fewer than 1,500 Layer 1 symbols
- Liquidity filter too restrictive

**Diagnosis**:
```bash
# Check market conditions
python ai_trader.py universe --market-conditions

# Review qualification criteria
cat config/scanners/layer1_liquidity.yml

# Analyze failed symbols
python ai_trader.py universe --layer 0 --qualification-details | grep "failed"
```

**Solutions**:
```bash
# 1. Temporarily relax criteria (bear market conditions)
# Edit config/scanners/layer1_liquidity.yml:
# min_volume: 500000      # Reduced from 1000000
# min_market_cap: 50000000 # Reduced from 100000000

# 2. Re-run Layer 1 qualification
python ai_trader.py universe --requalify --layer 1

# 3. Monitor impact
python ai_trader.py universe --stats --compare-previous
```

#### Layer 2: Technical Analysis Failures

**Symptoms**:
- Layer 2 produces <30 symbols
- Technical indicators returning NaN
- Real-time data lag

**Diagnosis**:
```bash
# Check technical indicator data
python ai_trader.py validate --component technical-indicators --sample-symbols AAPL,MSFT

# Review real-time data freshness
python ai_trader.py status --component realtime-data

# Check for calculation errors
grep "ERROR.*technical" /var/log/ai_trader/scanner.log | tail -20
```

**Solutions**:
```bash
# 1. Verify price data quality
python ai_trader.py validate --component price-data --layer 2

# 2. Recalculate technical indicators
python ai_trader.py universe --recalculate-indicators --layer 2

# 3. Check for data gaps
python ai_trader.py universe --data-gaps --layer 2 --fill-missing

# 4. Adjust technical criteria (if market conditions changed)
# Edit config/scanners/layer2_technical.yml as needed
```

#### Layer 3: Fundamental Data Missing

**Symptoms**:
- Layer 3 produces <15 symbols
- Fundamental data incomplete
- News sentiment analysis failing

**Diagnosis**:
```bash
# Check fundamental data sources
python ai_trader.py validate --component fundamental-data

# Test news sentiment pipeline
python ai_trader.py validate --component news-sentiment --test-symbol AAPL

# Review data source status
python ai_trader.py status --component external-apis
```

**Solutions**:
```bash
# 1. Verify financial data API access
python ai_trader.py validate --component financial-apis --test-connection

# 2. Refresh fundamental data cache
python ai_trader.py universe --refresh-fundamentals --layer 2

# 3. Check news sentiment configuration
cat config/sentiment/news_analysis.yml
python ai_trader.py universe --test-sentiment --symbols AAPL,MSFT,GOOGL

# 4. Fallback to alternative data sources
python ai_trader.py universe --requalify --layer 3 --use-fallback-data
```

### Performance Optimization

#### Database Performance

**Symptoms**:
- Scanner execution time >5 minutes
- Database timeout errors
- High memory usage

**Diagnosis**:
```bash
# Check database performance
python ai_trader.py status --component database --detailed

# Monitor query performance
grep "slow_query" /var/log/postgresql/postgresql.log

# Check memory usage
python ai_trader.py status --component memory --scanner-breakdown
```

**Solutions**:
```bash
# 1. Optimize database indexes
psql ai_trader_db -c "
  CREATE INDEX CONCURRENTLY idx_companies_active_layer 
  ON companies(active, layer_qualification);
  
  CREATE INDEX CONCURRENTLY idx_market_data_symbol_date 
  ON market_data(symbol, date DESC);
"

# 2. Increase batch sizes for bulk operations
# Edit config/database/performance.yml:
# batch_size: 2000  # Increased from 1000

# 3. Enable query caching
# Edit config/database/caching.yml:
# enable_query_cache: true
# cache_duration: 300  # 5 minutes

# 4. Archive old data
python ai_trader.py maintenance --archive-old-data --days 365
```

#### Memory Optimization

**Solutions**:
```bash
# 1. Implement chunked processing
# Edit config/scanners/performance.yml:
# chunk_size: 500      # Process 500 symbols at a time
# enable_chunking: true

# 2. Clear intermediate caches
python ai_trader.py maintenance --clear-scanner-cache

# 3. Optimize feature computation
# Only compute expensive features for Layer 3
python ai_trader.py universe --optimize-features --layer-aware
```

## Maintenance Procedures

### Regular Maintenance Schedule

#### Daily (Automated)
- Universe health check and alerting
- Layer 1-3 qualification execution
- Performance metrics collection
- Log rotation and cleanup

#### Weekly (Manual Review)
- Review qualification rate trends
- Analyze performance bottlenecks
- Update configuration if needed
- Validate data quality metrics

#### Monthly (Deep Maintenance)
- Full system performance review
- Configuration optimization
- Historical data archival
- Security audit of API keys

### Configuration Management

#### Configuration Validation
```bash
# Validate all scanner configurations
python ai_trader.py universe --validate-config --all-layers

# Test configuration changes before deployment
python ai_trader.py universe --test-config --dry-run --layer 1

# Backup current configuration
cp -r config/scanners config/scanners.backup.$(date +%Y%m%d)
```

#### Safe Configuration Updates
```bash
# 1. Test in development environment first
export AI_TRADER_ENV=development
python ai_trader.py universe --test-config-update --layer 1

# 2. Apply with gradual rollout
python ai_trader.py universe --update-config --layer 1 --gradual-rollout

# 3. Monitor impact
python ai_trader.py universe --monitor-config-impact --duration 1h

# 4. Rollback if needed
python ai_trader.py universe --rollback-config --layer 1
```

### Disaster Recovery

#### Backup Procedures
```bash
# 1. Backup universe database
pg_dump ai_trader_db -t companies -t layer_qualifications > universe_backup.sql

# 2. Backup configuration
tar -czf scanner_config_backup.tar.gz config/scanners/

# 3. Backup historical qualification data
python ai_trader.py universe --export-qualification-history --days 90 > qualification_history.json
```

#### Recovery Procedures
```bash
# 1. Restore from backup
psql ai_trader_db < universe_backup.sql

# 2. Verify data integrity
python ai_trader.py universe --verify-integrity --repair-if-needed

# 3. Re-run qualifications if needed
python ai_trader.py universe --requalify-all --parallel

# 4. Validate system health
python ai_trader.py universe --health --full-check
```

## Security Considerations

### API Key Management
- Rotate API keys monthly
- Use environment variables, never hardcode
- Monitor API usage for anomalies
- Implement rate limiting

### Data Protection
- Encrypt sensitive market data at rest
- Use TLS for all external API calls  
- Audit data access patterns
- Implement data retention policies

### Access Control
- Role-based access to scanner configuration
- Audit logs for all configuration changes
- Separate development/production environments
- Regular security reviews

---

**Related Documentation**:
- [SCANNER_WORKFLOW.md](SCANNER_WORKFLOW.md) - Complete workflow and layer specifications
- [SCANNER_INTEGRATION.md](SCANNER_INTEGRATION.md) - Integration with backfill and training
- [SYSTEM_ENTRY_POINTS.md](SYSTEM_ENTRY_POINTS.md) - CLI usage patterns
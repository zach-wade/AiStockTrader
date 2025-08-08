# Validation Profile Comparison

This document provides a comprehensive comparison of the different validation profiles available in the AI Trader validation system.

## Profile Overview

| Profile | Description | Use Case | Performance Impact |
|---------|-------------|----------|-------------------|
| **STRICT** | Production profile with all checks enabled | Live trading, production data | High - All validations run |
| **HISTORICAL** | Relaxed profile for historical data imports | Bulk imports, backfilling | Medium - Core checks only |
| **REALTIME** | Optimized for speed with minimal checks | Real-time feeds, low latency | Low - Essential checks only |
| **DEBUG** | All checks with verbose logging | Development, troubleshooting | Very High - Extra logging |

## Validation Checks by Profile

| Check Type | STRICT | HISTORICAL | REALTIME | DEBUG |
|------------|--------|------------|----------|-------|
| **Required Fields** | ✅ | ✅ | ✅ | ✅ |
| **Data Types** | ✅ | ✅ | ❌ | ✅ |
| **Business Rules** | ✅ | ✅ | ⚡ Basic only | ✅ |
| **Statistical Outliers** | ✅ | ❌ | ❌ | ✅ |
| **Cross-field Validation** | ✅ | ✅ | ❌ | ✅ |
| **Timestamp Consistency** | ✅ | ❌ | ❌ | ✅ |
| **Volume Spikes** | ✅ | ❌ | ❌ | ✅ |
| **Price Anomalies** | ✅ | ❌ | ⚡ Critical only | ✅ |

## Threshold Comparison

| Threshold | STRICT | HISTORICAL | REALTIME | DEBUG |
|-----------|--------|------------|----------|-------|
| **Outlier Std Devs** | 3σ | 5σ | 10σ | 3σ |
| **Volume Spike Multiplier** | 10x | 50x | 100x | 10x |
| **Price Change Max %** | 20% | 50% | 100% | 20% |
| **Max Null Percentage** | 0% | 5% | 10% | 0% |
| **Min History Required** | 252 days | 100 days | 20 days | 252 days |

## Business Rules Applied

### OHLC Validation Rules

| Rule | STRICT | HISTORICAL | REALTIME | DEBUG |
|------|--------|------------|----------|-------|
| High >= Low | ✅ | ✅ | ✅ | ✅ |
| High >= Open | ✅ | ✅ | ✅ | ✅ |
| High >= Close | ✅ | ✅ | ✅ | ✅ |
| Low <= Open | ✅ | ✅ | ✅ | ✅ |
| Low <= Close | ✅ | ✅ | ✅ | ✅ |
| Positive Volume | ✅ | ✅ | ✅ | ✅ |
| Reasonable Spread (<50%) | ✅ | ❌ | ✅ | ✅ |

### Quote Data Rules

| Rule | STRICT | HISTORICAL | REALTIME | DEBUG |
|------|--------|------------|----------|-------|
| Bid <= Ask | ✅ | ✅ | ✅ | ✅ |
| Positive Sizes | ✅ | ✅ | ✅ | ✅ |
| Spread Limits | ✅ | ❌ | ❌ | ✅ |

## Error Handling by Profile

| Stage | Severity | STRICT Action | HISTORICAL Action | REALTIME Action | DEBUG Action |
|-------|----------|---------------|-------------------|-----------------|--------------|
| **INGEST** | ERROR | DROP_ROW | DROP_ROW | FLAG_ROW | DROP_ROW + Log |
| **INGEST** | WARNING | FLAG_ROW | CONTINUE | CONTINUE | FLAG_ROW + Log |
| **POST_ETL** | ERROR | SKIP_SYMBOL | SKIP_SYMBOL | FLAG_AND_CONTINUE | SKIP_SYMBOL + Log |
| **POST_ETL** | WARNING | FLAG_AND_CONTINUE | CONTINUE | CONTINUE | FLAG_AND_CONTINUE + Log |
| **FEATURE_READY** | ERROR | USE_LAST_GOOD | USE_LAST_GOOD | USE_LAST_GOOD | USE_LAST_GOOD + Log |
| **FEATURE_READY** | WARNING | CONTINUE_WITH_WARNING | CONTINUE | CONTINUE | CONTINUE_WITH_WARNING + Log |

## Performance Characteristics

### STRICT Profile
- **Latency**: 50-100ms per batch
- **CPU Usage**: High
- **Memory**: Moderate (stores validation history)
- **Suitable for**: Production trading, regulatory compliance

### HISTORICAL Profile
- **Latency**: 20-40ms per batch
- **CPU Usage**: Moderate
- **Memory**: Low
- **Suitable for**: Bulk imports, backfilling, research

### REALTIME Profile
- **Latency**: 5-10ms per batch
- **CPU Usage**: Low
- **Memory**: Minimal
- **Suitable for**: High-frequency data, streaming feeds

### DEBUG Profile
- **Latency**: 100-200ms per batch
- **CPU Usage**: Very High
- **Memory**: High (detailed logging)
- **Suitable for**: Development, troubleshooting, testing

## Monitoring and Alerting

| Event | STRICT | HISTORICAL | REALTIME | DEBUG |
|-------|--------|------------|----------|-------|
| **Validation Failures** | Alert | Log | Log | Alert + Debug |
| **Data Quality Issues** | Alert | Log | Ignore | Alert + Debug |
| **Performance Degradation** | Alert | Log | Alert | Log |
| **System Errors** | Alert | Alert | Alert | Alert + Debug |

## Profile Selection Guidelines

### Choose STRICT when:
- Running in production
- Trading with real money
- Regulatory compliance is required
- Data quality is critical

### Choose HISTORICAL when:
- Importing historical data
- Backfilling missing periods
- Running research backtests
- Data source quality varies

### Choose REALTIME when:
- Processing streaming data
- Low latency is critical
- Basic validation suffices
- High data volume

### Choose DEBUG when:
- Developing new features
- Troubleshooting issues
- Testing validation rules
- Need detailed logs

## Configuration Examples

### Using Profiles in Code

```python
from ai_trader.data_pipeline.validation import ValidationPipeline

# Production setup
pipeline = ValidationPipeline()
result = await pipeline.validate_ingest(
    data, 
    source_type='alpaca',
    profile='STRICT'
)

# Historical import
result = await pipeline.validate_ingest(
    historical_data,
    source_type='yahoo',
    profile='HISTORICAL'
)

# Real-time streaming
result = await pipeline.validate_ingest(
    stream_data,
    source_type='polygon',
    profile='REALTIME'
)
```

### Environment-based Selection

```python
import os

# Automatically select profile based on environment
ENV = os.getenv('ENVIRONMENT', 'development')
PROFILE_MAP = {
    'production': 'STRICT',
    'staging': 'STRICT',
    'development': 'DEBUG',
    'backtest': 'HISTORICAL'
}

profile = PROFILE_MAP.get(ENV, 'STRICT')
```

## Quick Reference Card

| Need | Recommended Profile |
|------|-------------------|
| Production trading | STRICT |
| Development | DEBUG |
| Bulk historical import | HISTORICAL |
| Real-time feeds | REALTIME |
| Compliance audit | STRICT |
| Performance testing | REALTIME |
| Data quality analysis | DEBUG |
| Research backtest | HISTORICAL |
# Validation System Data Flow Boundaries

This document provides a clear visualization of data flow through the multi-stage validation system, including boundaries, checkpoints, and decision points.

## Data Flow Overview

```
┌─────────────────┐
│  External Data  │
│    Sources      │
│  • Alpaca       │
│  • Polygon      │
│  • Yahoo        │
│  • Reddit       │
└────────┬────────┘
         │
         ▼
╔═══════════════════════════════════════════════════════════════╗
║                    BOUNDARY 1: INGEST                         ║
╟───────────────────────────────────────────────────────────────╢
║ • Raw data enters the system                                  ║
║ • Source-specific format validation                           ║
║ • Schema validation                                           ║
║ • Required fields check                                       ║
╚═══════════════════════╤═══════════════════════════════════════╝
                        │
                        ▼
                ┌───────────────┐
                │ Ingest Stage  │──────► ❌ DROP_ROW (on ERROR)
                │  Validation   │──────► ⚠️ FLAG_ROW (on WARNING)
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │   Raw Data    │
                │    Store      │
                │ (Data Lake)   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │     ETL       │
                │  Processing   │
                │ • Transform   │
                │ • Aggregate   │
                │ • Standardize │
                └───────┬───────┘
                        │
                        ▼
╔═══════════════════════════════════════════════════════════════╗
║                   BOUNDARY 2: POST-ETL                        ║
╟───────────────────────────────────────────────────────────────╢
║ • Transformed data validation                                 ║
║ • Aggregation correctness                                     ║
║ • Data consistency checks                                     ║
║ • Business rules validation                                   ║
╚═══════════════════════╤═══════════════════════════════════════╝
                        │
                        ▼
                ┌───────────────┐
                │ Post-ETL      │──────► ❌ SKIP_SYMBOL (on ERROR)
                │ Validation    │──────► ⚠️ FLAG_AND_CONTINUE (on WARNING)
                └───────┬───────┘        📢 ALERT (on ERROR)
                        │
                        ▼
                ┌───────────────┐
                │  Processed    │
                │  Data Store   │
                │ (Hot Storage) │
                └───────┬───────┘
                        │
                        ▼
╔═══════════════════════════════════════════════════════════════╗
║                BOUNDARY 3: FEATURE-READY                      ║
╟───────────────────────────────────────────────────────────────╢
║ • Feature calculation prerequisites                           ║
║ • Sufficient history check                                    ║
║ • Statistical validity                                        ║
║ • Data completeness                                          ║
╚═══════════════════════╤═══════════════════════════════════════╝
                        │
                        ▼
                ┌───────────────┐
                │ Feature-Ready │──────► ❌ USE_LAST_GOOD (on ERROR)
                │  Validation   │──────► ⚠️ CONTINUE_WITH_WARNING
                └───────┬───────┘        📢 ALERT (on ERROR)
                        │
                        ▼
                ┌───────────────┐
                │   Feature     │
                │  Engineering  │
                │   Pipeline    │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │   Features    │
                │    Store      │
                └───────────────┘
```

## Detailed Boundary Specifications

### Boundary 1: INGEST

**Purpose**: Protect the system from malformed or invalid raw data

#### Input Contract

- **Format**: JSON, CSV, or API response objects
- **Sources**: External data providers (Alpaca, Polygon, Yahoo, Reddit)
- **Volume**: Variable (1-1M records per batch)

#### Validation Rules

```yaml
Required Checks:
  - Schema validation against source-specific formats
  - Field presence (timestamp, OHLCV for market data)
  - Data type validation (numeric prices, valid timestamps)
  - Basic range checks (prices > 0, volumes >= 0)

Actions on Failure:
  - ERROR → DROP_ROW (protect downstream)
  - WARNING → FLAG_ROW (allow but track)
```

#### Output Contract

- **Format**: Validated raw data
- **Guarantees**:
  - All required fields present
  - Basic data types correct
  - No null prices
- **Metadata**: Source, validation_timestamp, flags

### Boundary 2: POST-ETL

**Purpose**: Ensure data transformations maintain integrity

#### Input Contract

- **Format**: Pandas DataFrame or structured data
- **Source**: ETL pipeline output
- **Assumptions**: Already passed INGEST validation

#### Validation Rules

```yaml
Required Checks:
  - OHLC relationship integrity (High >= Low, etc.)
  - Time series consistency
  - Aggregation accuracy
  - No missing periods (configurable tolerance)
  - Statistical outlier detection

Actions on Failure:
  - ERROR → SKIP_SYMBOL + ALERT
  - WARNING → FLAG_AND_CONTINUE
```

#### Output Contract

- **Format**: Standardized DataFrame
- **Guarantees**:
  - Consistent time intervals
  - Valid OHLC relationships
  - No extreme outliers
  - Complete time series
- **Metadata**: aggregation_level, validation_profile, quality_score

### Boundary 3: FEATURE-READY

**Purpose**: Ensure data meets feature calculation requirements

#### Input Contract

- **Format**: Standardized DataFrame
- **Source**: Processed data store
- **Assumptions**: Passed POST-ETL validation

#### Validation Rules

```yaml
Required Checks:
  - Minimum history (e.g., 252 days for annual features)
  - Column completeness
  - No lookahead bias
  - Stationarity tests (for ML features)
  - Cross-sectional consistency

Actions on Failure:
  - ERROR → USE_LAST_GOOD_SNAPSHOT + ALERT
  - WARNING → CONTINUE_WITH_WARNING
```

#### Output Contract

- **Format**: Feature-ready DataFrame
- **Guarantees**:
  - Sufficient history for all calculations
  - All required columns present
  - Data statistically valid
  - No temporal leakage
- **Metadata**: feature_readiness_score, warnings

## Decision Trees at Each Boundary

### INGEST Decision Tree

```
Data Arrives
    │
    ├─ Schema Valid?
    │   ├─ NO → DROP_ROW
    │   └─ YES ↓
    │
    ├─ Required Fields Present?
    │   ├─ NO → DROP_ROW
    │   └─ YES ↓
    │
    ├─ Data Types Correct?
    │   ├─ NO → FLAG_ROW + Try Conversion
    │   │       ├─ Success → CONTINUE
    │   │       └─ Fail → DROP_ROW
    │   └─ YES ↓
    │
    └─ Business Rules Pass?
        ├─ NO → DROP_ROW (if ERROR)
        │       FLAG_ROW (if WARNING)
        └─ YES → PASS TO ETL
```

### POST-ETL Decision Tree

```
Transformed Data
    │
    ├─ OHLC Relationships Valid?
    │   ├─ NO → SKIP_SYMBOL + ALERT
    │   └─ YES ↓
    │
    ├─ Time Consistency OK?
    │   ├─ NO → Check Gap Size
    │   │       ├─ > Threshold → SKIP_SYMBOL
    │   │       └─ < Threshold → FLAG + INTERPOLATE
    │   └─ YES ↓
    │
    ├─ Outliers Detected?
    │   ├─ YES → Check Severity
    │   │        ├─ Extreme → SKIP_SYMBOL
    │   │        └─ Moderate → FLAG_AND_CONTINUE
    │   └─ NO ↓
    │
    └─ Quality Score
        ├─ < 0.8 → WARNING + CONTINUE
        └─ >= 0.8 → PASS TO STORAGE
```

### FEATURE-READY Decision Tree

```
Pre-Feature Data
    │
    ├─ Sufficient History?
    │   ├─ NO → USE_LAST_GOOD_SNAPSHOT
    │   │       ├─ Available & Fresh → USE IT
    │   │       └─ Not Available → SKIP_CALCULATION
    │   └─ YES ↓
    │
    ├─ All Columns Present?
    │   ├─ NO → Check if Optional
    │   │       ├─ Required → FAIL
    │   │       └─ Optional → CONTINUE_WITH_WARNING
    │   └─ YES ↓
    │
    ├─ Statistical Tests Pass?
    │   ├─ NO → Log Details
    │   │       ├─ Critical → USE_LAST_GOOD
    │   │       └─ Non-critical → WARNING
    │   └─ YES ↓
    │
    └─ Final Check
        └─ PASS TO FEATURE PIPELINE
```

## Data Quality Metrics at Boundaries

### Boundary Metrics

| Metric | INGEST | POST-ETL | FEATURE-READY |
|--------|--------|----------|---------------|
| **Schema Compliance** | ✓ | - | - |
| **Completeness** | ✓ | ✓ | ✓ |
| **Accuracy** | Basic | ✓ | ✓ |
| **Consistency** | - | ✓ | ✓ |
| **Timeliness** | ✓ | ✓ | - |
| **Statistical Validity** | - | ✓ | ✓ |

### Quality Score Calculation

```python
def calculate_quality_score(validation_results):
    """
    Calculate overall data quality score at each boundary
    """
    weights = {
        'completeness': 0.3,
        'accuracy': 0.3,
        'consistency': 0.2,
        'timeliness': 0.1,
        'validity': 0.1
    }

    scores = {
        'completeness': 1 - (missing_data_pct / 100),
        'accuracy': 1 - (error_count / total_count),
        'consistency': consistency_check_pass_rate,
        'timeliness': 1 - (data_age_hours / 24),
        'validity': statistical_test_pass_rate
    }

    return sum(weights[k] * scores[k] for k in weights)
```

## Integration Points

### With Monitoring Systems

```yaml
Prometheus Metrics:
  - validation_success_rate{stage="ingest", source="alpaca"}
  - validation_errors_total{stage="post_etl", error_type="ohlc_violation"}
  - data_quality_score{stage="feature_ready", symbol="AAPL"}
  - validation_duration_seconds{stage="ingest"}
```

### With Alerting Systems

```yaml
Alert Rules:
  - name: High Validation Failure Rate
    condition: validation_success_rate < 0.95
    stages: [post_etl, feature_ready]
    severity: warning

  - name: Critical Data Quality Issue
    condition: validation_success_rate < 0.80
    stages: all
    severity: critical

  - name: Data Staleness
    condition: hours_since_last_valid_data > 2
    stages: [ingest]
    severity: warning
```

### With Data Lineage

```yaml
Lineage Tracking:
  - validation_id: UUID for each validation run
  - parent_validation_id: Link to previous stage
  - transformations_applied: List of ETL operations
  - quality_impact: How validation affected quality score
```

## Best Practices

1. **Fail Fast**: Catch issues at the earliest boundary
2. **Clear Contracts**: Define explicit input/output expectations
3. **Graceful Degradation**: Use fallbacks when appropriate
4. **Audit Trail**: Log all validation decisions
5. **Performance Balance**: Profile-based validation for different use cases
6. **Monitoring**: Track metrics at each boundary
7. **Alerting**: Set up appropriate alerts for critical failures

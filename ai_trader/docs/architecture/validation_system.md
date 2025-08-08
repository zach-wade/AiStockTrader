# Multi-Stage Validation System

## Overview

The AI Trader validation system implements a comprehensive 3-stage validation pipeline that ensures data quality throughout the entire data processing workflow. This system catches data issues early and prevents corrupted or invalid data from propagating through the system.

## Architecture

### Three Validation Stages

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Stage 1:       │     │  Stage 2:       │     │  Stage 3:       │
│  INGEST         │ --> │  POST_ETL       │ --> │  PRE_FEATURE    │
│  Validation     │     │  Validation     │     │  Validation     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1. **Ingest Validation (Stage 1)**

**Purpose**: Validate raw data immediately upon ingestion from data sources

**Location**: Data source clients (Alpaca, Polygon, Yahoo, etc.)

**Checks**:
- Data format correctness
- Required fields presence
- Basic data type validation
- Timestamp validity
- Symbol format validation

**Example Integration**:
```python
# In alpaca_client.py
validated_df = await validate_on_ingest(
    df,
    context={
        'symbol': symbol,
        'source': 'alpaca',
        'data_type': 'market_data',
        'interval': interval
    }
)
```

### 2. **Post-ETL Validation (Stage 2)**

**Purpose**: Validate data after ETL transformations and standardization

**Location**: ETL processors and data standardizers

**Checks**:
- Data consistency after transformations
- Aggregation correctness
- No data loss during processing
- Business rule validation
- Cross-field validation

**Example Integration**:
```python
# In etl_processor.py
validation_result = await self.validation_pipeline.validate(
    combined_df,
    stage=ValidationStage.POST_ETL,
    context={
        'symbol': symbol,
        'source': 'etl_aggregation',
        'data_type': 'market_data'
    }
)
```

### 3. **Pre-Feature Validation (Stage 3)**

**Purpose**: Final validation before feature engineering and strategy consumption

**Location**: Feature engineering pipelines and strategy inputs

**Checks**:
- Data completeness for feature calculation
- Statistical validity (no extreme outliers)
- Sufficient history for calculations
- Data quality metrics
- Anomaly detection

**Example Integration**:
```python
# In main_orchestrator.py
validation_result = await self.validation_pipeline.validate(
    data,
    stage=ValidationStage.PRE_FEATURE,
    context={
        'symbol': symbol,
        'source': 'market_snapshot',
        'data_type': 'market_data'
    }
)
```

## Core Components

### ValidationPipeline Class

The central orchestrator for all validation operations:

```python
class ValidationPipeline:
    """Multi-stage validation pipeline for data quality assurance"""
    
    async def validate(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict],
        stage: ValidationStage,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Main validation entry point"""
```

### ValidationResult Class

Structured result containing validation outcome:

```python
@dataclass
class ValidationResult:
    is_valid: bool
    data: Any  # Cleaned/validated data
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    stage: ValidationStage
    timestamp: datetime
```

### Validator Classes

Stage-specific validators implementing validation logic:

- **IngestValidator**: Raw data validation
- **PostETLValidator**: Transformation validation
- **PreFeatureValidator**: Feature-ready validation

## Integration Points

### 1. Data Source Clients
- `alpaca_client.py`: Validates on data fetch
- `polygon_client.py`: Validates API responses
- `yahoo_client.py`: Validates historical data

### 2. ETL Pipeline
- `etl_processor.py`: Validates after aggregations
- `data_standardizer.py`: Validates after standardization

### 3. Orchestration Layer
- `main_orchestrator.py`: Validates before strategy execution
- Validates market snapshots before processing

### 4. Storage Layer
- `partition_manager.py`: Validates during consolidation
- Ensures data integrity during lifecycle transitions

## Validation Rules

### Market Data Validation

**Required Fields**:
- timestamp (datetime)
- open, high, low, close (float)
- volume (int)
- symbol (string)

**Business Rules**:
- high >= low
- high >= open, close
- low <= open, close
- volume >= 0
- timestamp is timezone-aware (UTC)

### News Data Validation

**Required Fields**:
- headline (string)
- timestamp (datetime)
- source (string)

**Optional Fields**:
- summary, body, author, url

### Validation Profiles

Different validation profiles for different use cases:

- **STRICT**: All checks enabled (production)
- **HISTORICAL**: Relaxed for historical data
- **REALTIME**: Optimized for speed
- **DEBUG**: Extra logging and metrics

## Error Handling

### Validation Failures

When validation fails:

1. **Stage 1 (Ingest)**: Data is rejected, empty DataFrame returned
2. **Stage 2 (Post-ETL)**: Processing skips to next symbol
3. **Stage 3 (Pre-Feature)**: Data excluded from strategy calculations

### Logging

All validation events are logged with appropriate levels:

```python
logger.warning(f"Validation failed for {symbol}: {validation_result.issues}")
logger.info(f"Validation passed for {symbol} with {len(warnings)} warnings")
```

## Performance Considerations

### Async Processing

All validation is async-enabled for non-blocking operation:

```python
validated_df = await validate_on_ingest(df, context)
```

### Caching

Validation results can be cached to avoid re-validation:

```python
cache_key = f"validation_{symbol}_{stage}_{data_hash}"
```

### Batch Processing

Validators support batch processing for efficiency:

```python
validated_batch = await validate_batch(dataframes, stage, context)
```

## Monitoring & Metrics

### Validation Metrics

Tracked metrics include:

- Validation success/failure rates by stage
- Common validation issues
- Processing time per stage
- Data quality scores

### Dashboard Integration

Validation metrics are exposed to monitoring dashboards:

```python
metrics = {
    'validation_success_rate': 0.98,
    'avg_validation_time_ms': 45,
    'issues_detected': 142,
    'data_quality_score': 0.95
}
```

## Usage Examples

### Basic Usage

```python
from ai_trader.data_pipeline.validation import ValidationPipeline, ValidationStage

# Initialize pipeline
pipeline = ValidationPipeline()

# Validate data
result = await pipeline.validate(
    df,
    stage=ValidationStage.POST_ETL,
    context={'symbol': 'AAPL', 'source': 'alpaca'}
)

if result.is_valid:
    process_data(result.data)
else:
    handle_validation_failure(result.issues)
```

### Custom Validation Rules

```python
# Add custom validator
pipeline.add_validator(
    stage=ValidationStage.PRE_FEATURE,
    validator=CustomBusinessRuleValidator()
)
```

### Convenience Functions

```python
# Stage-specific helpers
from ai_trader.data_pipeline.validation import (
    validate_on_ingest,
    validate_post_etl,
    validate_pre_feature
)

# Direct validation
validated_df = await validate_on_ingest(df, context)
```

## Best Practices

1. **Always validate at ingest**: Catch bad data early
2. **Use appropriate stages**: Each stage has specific purposes
3. **Handle failures gracefully**: Don't crash on bad data
4. **Log validation events**: For debugging and monitoring
5. **Monitor validation metrics**: Track system health
6. **Update validation rules**: As business requirements change

## Future Enhancements

1. **Machine Learning Validation**: Use ML to detect anomalies
2. **Real-time Validation Dashboard**: Live validation monitoring
3. **Validation Rule Builder**: UI for creating custom rules
4. **Historical Validation Reports**: Trend analysis of data quality
5. **Auto-correction**: Automatic fixing of common issues
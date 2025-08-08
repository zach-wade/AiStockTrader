# Data Pipeline Integration Guide

## Overview

The AI Trader data pipeline provides a comprehensive framework for collecting, processing, storing, and distributing financial market data. It supports multiple data sources, real-time streaming, batch processing, and intelligent storage tiering.

## Architecture

### Core Components

1. **DataPipelineOrchestrator** (`orchestrator.py`)
   - Top-level coordinator for entire data pipeline
   - Manages ingestion, processing, and historical stages
   - Supports batch, real-time, and hybrid modes

2. **DataPipelineManager** (`data_pipeline_manager.py`)
   - Orchestration layer component
   - Provides lifecycle management and health monitoring
   - Integrates with the main trading system

3. **IngestionOrchestrator** (`ingestion/orchestrator.py`)
   - Coordinates data collection from multiple sources
   - Manages rate limiting and error recovery
   - Supports parallel symbol processing

4. **ProcessingManager** (`processing/manager.py`)
   - Handles data transformation and standardization
   - Performs validation and quality checks
   - Calculates derived features

5. **StorageRouter** (`storage/storage_router.py`)
   - Routes data to appropriate storage tiers
   - Manages hot/cold storage decisions
   - Handles data lifecycle management

### Data Sources

The pipeline supports multiple data sources:

- **Market Data**: Alpaca, Polygon, Yahoo Finance
- **News & Sentiment**: Polygon News, Benzinga, Reddit
- **Corporate Actions**: Dividends, splits, earnings
- **Options Data**: Option chains and pricing
- **Fundamentals**: Financial statements, ratios

## Integration Points

### 1. Trading System Integration

The data pipeline feeds processed data to the trading system:

```python
# Data flows from pipeline to trading algorithms
Pipeline → Storage → Trading Algorithms → Execution Engine
```

### 2. Feature Engineering

Integration with ML pipeline for feature calculation:

```python
# Raw data → Features → ML Models
Pipeline → Feature Store → ML Pipeline → Predictions
```

### 3. Real-Time Streaming

WebSocket integration for live market data:

```python
# Real-time data flow
Market Data Sources → Stream Processor → Trading System
```

## Usage Examples

### Basic Pipeline Execution

```python
from main.orchestration.managers.data_pipeline_manager import DataPipelineManager

# Initialize pipeline
pipeline_manager = DataPipelineManager(orchestrator)
await pipeline_manager.initialize()

# Start pipeline
await pipeline_manager.start()

# Run full pipeline
result = await pipeline_manager.run_pipeline()

# Check status
status = await pipeline_manager.get_pipeline_status()
print(f"Pipeline status: {status['status']}")
```

### Targeted Data Ingestion

```python
# Ingest specific symbols and date range
symbols = ['AAPL', 'GOOGL', 'MSFT']
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

result = await pipeline_manager.run_ingestion(
    data_type='market_data',
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    timeframe='1day'
)
```

### Real-Time Streaming Setup

```python
from main.data_pipeline.orchestrator import DataPipelineOrchestrator, PipelineMode

# Configure for real-time
flow_config = DataFlowConfig(
    mode=PipelineMode.REAL_TIME,
    batch_size=100,
    interval_seconds=1.0
)

# Start streaming
await orchestrator.start_real_time_flow(['AAPL', 'GOOGL'])
```

### Selective Stage Execution

```python
# Run only specific stages
await pipeline_manager.run_pipeline(stages=['ingestion'])
await pipeline_manager.run_pipeline(stages=['processing', 'historical'])
```

## Configuration

### Pipeline Configuration (`data_pipeline_config.yaml`)

```yaml
data_pipeline:
  collection:
    max_parallel_symbols: 5
    batch_size: 50
    retry_attempts: 3
    
  rate_limits:
    polygon:
      requests_per_second: 50
    alpaca:
      requests_per_second: 150
      
  storage:
    routing:
      hot_data_days: 30
      cold_fallback_enabled: true
      
  streaming:
    buffer_size: 10000
    window_seconds: 60
```

### Storage Tiers

1. **Hot Storage (PostgreSQL)**
   - Recent data (< 30 days)
   - High-frequency queries
   - Real-time scanners

2. **Cold Storage (Data Lake)**
   - Historical data
   - Bulk analysis
   - Long-term archival

### Rate Limiting

Rate limits are configured per data source:

- Polygon: 50 requests/second (unlimited API calls)
- Alpaca: 150 requests/second (10k requests/minute)
- Yahoo: 5 requests/second (free tier)

## Data Flow

### 1. Ingestion Flow

```
Sources → Rate Limiter → Ingestion Queue → Validators → Raw Storage
```

### 2. Processing Flow

```
Raw Storage → Transformer → Standardizer → Feature Calculator → Processed Storage
```

### 3. Distribution Flow

```
Processed Storage → Cache → API/Stream → Trading Algorithms
```

## Monitoring and Health Checks

### Pipeline Metrics

```python
# Get current metrics
metrics = await pipeline_manager.get_pipeline_metrics()
print(f"Records ingested: {metrics.ingestion_records}")
print(f"Quality score: {metrics.quality_score:.2%}")
print(f"Throughput: {metrics.throughput_records_per_second:.2f} rec/s")
```

### Health Monitoring

```python
# Trigger health check
health = await pipeline_manager.trigger_health_check()
if not health['healthy']:
    print(f"Issues detected: {health['issues']}")
```

### Performance Monitoring

- Ingestion rate (records/second)
- Processing latency
- Storage utilization
- Error rates
- Data quality scores

## Error Handling

### Retry Strategies

1. **Exponential Backoff**: For transient API errors
2. **Circuit Breaker**: For persistent failures
3. **Dead Letter Queue**: For unprocessable records

### Recovery Mechanisms

```python
# Pipeline automatically recovers from:
- API rate limits
- Network interruptions
- Partial data failures
- Storage errors
```

## Best Practices

### 1. Resource Management

```python
# Use appropriate batch sizes
config['collection']['batch_size'] = 50  # Balance speed vs. resource usage

# Configure parallel processing
config['collection']['max_parallel_symbols'] = 5  # Based on system capacity
```

### 2. Data Quality

```python
# Enable validation
config['quality']['validation_mode'] = 'strict'

# Monitor quality scores
if metrics.quality_score < 0.8:
    logger.warning("Data quality below threshold")
```

### 3. Storage Optimization

```python
# Configure hot/cold routing
config['storage']['routing']['hot_data_days'] = 30

# Use appropriate compression
config['storage']['data_lake']['compression'] = 'snappy'
```

### 4. Error Recovery

```python
# Configure retries
config['collection']['retry_attempts'] = 3
config['collection']['exponential_backoff'] = True
```

## Testing

### Integration Tests

```bash
# Run data pipeline integration tests
pytest tests/integration/test_data_pipeline_full_integration.py -m integration
```

### Performance Tests

```bash
# Run stress tests
pytest tests/performance/test_integration_pipeline_stress.py
```

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Check rate limit configuration
   - Verify API plan limits
   - Monitor request rates

2. **Data Gaps**
   - Run historical backfill
   - Check source availability
   - Verify date ranges

3. **Storage Issues**
   - Monitor disk space
   - Check database connections
   - Verify permissions

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('main.data_pipeline').setLevel(logging.DEBUG)
```

## Advanced Features

### 1. Custom Data Sources

```python
# Implement custom data source
class CustomDataSource(BaseSource):
    async def fetch_data(self, symbol, start_date, end_date):
        # Custom implementation
        pass
```

### 2. Stream Processing

```python
# Add custom stream processor
async def custom_processor(symbol, data):
    # Process streaming data
    processed = transform_data(data)
    return processed
```

### 3. Feature Engineering

```python
# Add custom features
class CustomFeatureBuilder(FeatureBuilder):
    def calculate_features(self, data):
        # Calculate custom features
        return features
```

## Next Steps

1. Configure data sources in environment
2. Set up storage backends
3. Run initial data collection
4. Monitor pipeline health
5. Optimize based on usage patterns
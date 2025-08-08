# Feature Storage Architecture

## Overview

The AI Trader system uses a **dual-store architecture** for feature storage, optimized for both training and live trading requirements. This architecture balances the needs of:
- High-speed feature access during live trading
- Efficient bulk storage for model training
- Version control for feature engineering iterations
- Scalability for large historical datasets

## Architecture Components

### 1. Hot Storage - PostgreSQL (FeatureStoreRepository)

**Purpose**: Real-time feature access for live trading

**Location**: PostgreSQL database, `feature_store` table

**Key Characteristics**:
- Fast SQL queries for feature retrieval
- Stores recent features (last 30 days by default)
- JSON field storage for flexible feature schemas
- Optimized for low-latency lookups
- Supports complex filtering and aggregations

**Use Cases**:
- Live trading feature lookups
- Real-time strategy execution
- Recent feature analysis
- Quick backtests on recent data

**Storage Format**:
```sql
TABLE feature_store (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    feature_set VARCHAR,     -- e.g., 'technical', 'sentiment'
    version VARCHAR,         -- e.g., '1.0'
    features JSON,          -- Actual feature values
    PRIMARY KEY (symbol, timestamp, feature_set, version)
)
```

### 2. Cold Storage - HDF5 Files (FeatureStoreV2)

**Purpose**: Historical feature storage for training and backtesting

**Location**: Data Lake at `data_lake/features/`

**Key Characteristics**:
- Efficient storage of large numerical arrays
- Version-based file naming (v1, v2, v3...)
- Automatic cleanup of old versions
- Compression support (gzip)
- Partitioned by symbol/feature_type/year/month

**Use Cases**:
- Model training on historical data
- Long-term backtesting
- Feature engineering experiments
- Data archival

**Storage Format**:
```
features/
  market_signals/
    symbol=AAPL/
      feature_type=technical_indicators/
        version=v1/
          year=2024/
            month=01/
              data.h5
```

## Data Flow

### Training Pipeline Flow
```
1. Raw Data → Feature Calculation
2. Features → FeatureStoreV2 (HDF5 with versioning)
3. Training reads from FeatureStoreV2
4. Models trained on historical features
```

### Production Pipeline Flow
```
1. Raw Data → Feature Calculation
2. Features → Both Stores:
   - FeatureStoreRepository (PostgreSQL) for live access
   - FeatureStoreV2 (HDF5) for archival
3. Live trading reads from FeatureStoreRepository
4. Backtesting can read from either store
```

## Storage Selection Logic

The system automatically selects the appropriate storage based on use case:

### When to Use PostgreSQL (Hot Storage)
- Data from last 30 days
- Live trading feature requests
- Real-time strategy execution
- Low-latency requirements
- Need for SQL queries/filtering

### When to Use HDF5 (Cold Storage)
- Historical data (> 30 days old)
- Model training datasets
- Bulk feature loading
- Backtesting on long time periods
- Feature versioning requirements

## Version Management

### HDF5 Versioning
- Each feature file has a version (v1, v2, v3...)
- New calculations create new versions
- Automatic cleanup keeps only last N versions (configurable)
- Prevents disk space issues from accumulating versions

### Configuration
```yaml
features:
  version_management:
    max_versions: 3          # Keep last 3 versions
    auto_cleanup: true       # Auto-delete old versions
  storage:
    compression: gzip        # Compression type
    compression_level: 4     # Compression level (1-9)
```

## Usage Examples

### 1. Using the Compatibility Wrapper
```python
from ai_trader.feature_pipeline import FeatureStore

# Initialize (handles both stores automatically)
fs = FeatureStore()

# Save features (writes to both stores)
fs.save_features(
    symbol="AAPL",
    features_df=features_df,
    feature_type="technical_indicators"
)

# Load features (intelligently selects store)
features = fs.load_features(
    symbol="AAPL",
    feature_type="technical_indicators",
    start_date=start_date,
    end_date=end_date
)
```

### 2. Direct Store Access
```python
# For specific PostgreSQL access
from ai_trader.feature_pipeline import FeatureStoreRepository
feature_repo = FeatureStoreRepository(db_adapter)

# For specific HDF5 access
from ai_trader.feature_pipeline import FeatureStoreV2
feature_store = FeatureStoreV2(data_lake_path)
```

### 3. DataLoader Integration
```python
from ai_trader.feature_pipeline import DataLoader

loader = DataLoader()
data = await loader.load_features(
    symbols=["AAPL", "GOOGL"],
    start_date=start_date,
    end_date=end_date,
    # Automatically uses appropriate store
)
```

## CLI Tools

### Feature Version Management
```bash
# List all feature versions
python -m ai_trader.data_pipeline.storage.feature_version_cli /path/to/data_lake list

# Clean up old versions
python -m ai_trader.data_pipeline.storage.feature_version_cli /path/to/data_lake cleanup --max-versions 3 --execute

# Show storage statistics
python -m ai_trader.data_pipeline.storage.feature_version_cli /path/to/data_lake stats
```

## Best Practices

1. **Feature Calculation**: Always save to both stores during feature calculation
2. **Live Trading**: Always read from PostgreSQL for lowest latency
3. **Training**: Always read from HDF5 for efficient bulk loading
4. **Cleanup**: Run periodic cleanup to manage disk space
5. **Versioning**: Use semantic versioning for major feature changes

## Migration Path

For existing systems:
1. Continue using existing feature files
2. New features automatically use versioned storage
3. Old hash-based files (e.g., `AAPL_b1686acd.h5`) remain accessible
4. Gradual migration as features are recalculated

## Performance Considerations

### PostgreSQL Performance
- Indexed on (symbol, timestamp, feature_set, version)
- JSON fields allow flexible schemas but slower than columns
- Best for small, recent datasets
- Consider partitioning for very large deployments

### HDF5 Performance
- Excellent compression ratios for numerical data
- Fast bulk reads for training
- Slower individual record access
- Memory-efficient chunked reading

## Future Enhancements

1. **Automatic Tiering**: Move old PostgreSQL data to HDF5 automatically
2. **Feature Lineage**: Track feature calculation history
3. **Cloud Storage**: S3/GCS backend for HDF5 files
4. **Feature Serving**: Dedicated feature serving API
5. **Real-time Sync**: Stream features to both stores simultaneously
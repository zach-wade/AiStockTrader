# Hot/Cold Storage Architecture

## Overview

The AI Trader system implements a hot/cold storage architecture to optimize performance and cost:

- **Hot Storage (PostgreSQL)**: Recent 30 days of data for fast access
- **Cold Storage (Data Lake)**: Historical data in Parquet format for long-term storage

## Configuration

### Hot Storage Window

The system is configured to maintain only the last 30 days of data in PostgreSQL. This is configurable in `unified_config.yaml`:

```yaml
storage:
  lifecycle:
    hot_days: 30  # Days to keep in PostgreSQL
    hot_intervals:  # Which intervals to keep hot
      - "1day"  # Only daily bars in hot storage
    archive_on_backfill: true  # Run archival after backfill
    auto_rotation:
      enabled: true
      schedule: "0 2 * * *"  # 2 AM daily
```

### Data Intervals

To reduce storage by ~95%, only daily (1day) intervals are stored in hot storage. Other intervals (1min, 5min, etc.) are available in cold storage and can be queried on demand.

## Components

### 1. DataLifecycleManager

Manages the automated archival of aged data from hot to cold storage.

**Location**: `src/main/data_pipeline/storage/data_lifecycle_manager.py`

**Key Features**:
- Identifies data older than the hot window
- Archives data to cold storage with partitioning
- Cleans up hot storage after successful archival
- Runs database optimization (VACUUM)

### 2. Storage Rotation Job

Automated job to run the archival process.

**Location**: `src/main/jobs/storage_rotation_job.py`

**Usage**:
```bash
# Run storage rotation (normal mode)
python src/main/jobs/storage_rotation_job.py

# Dry run (simulate without changes)
python src/main/jobs/storage_rotation_job.py --dry-run

# Force run even if disabled
python src/main/jobs/storage_rotation_job.py --force

# Archive all repository types
python src/main/jobs/storage_rotation_job.py --full
```

### 3. Hot Storage Cleanup Script

One-time script to migrate existing data to the 30-day window.

**Location**: `src/main/scripts/cleanup_hot_storage.py`

**Usage**:
```bash
# Analyze current hot storage
python src/main/scripts/cleanup_hot_storage.py --analyze-only

# Dry run cleanup
python src/main/scripts/cleanup_hot_storage.py --dry-run

# Live cleanup (will prompt for confirmation)
python src/main/scripts/cleanup_hot_storage.py

# Cleanup with custom window
python src/main/scripts/cleanup_hot_storage.py --hot-days 60
```

### 4. Cold Storage Query Engine

Provides efficient query access to archived data.

**Location**: `src/main/data_pipeline/storage/cold_storage_query_engine.py`

**Features**:
- Parallel file reads
- Pushdown filtering to Parquet
- In-memory caching
- Aggregation support

## Data Flow

1. **Backfill Process**:
   - Downloads data from sources
   - Stores ALL data in cold storage (Data Lake)
   - Loads only recent 30 days of daily data to PostgreSQL
   - Automatically runs archival if configured

2. **Daily Rotation**:
   - Scheduled job runs at 2 AM
   - Identifies data older than 30 days
   - Archives to cold storage
   - Removes from hot storage
   - Optimizes database

3. **Query Routing**:
   - Recent queries (< 30 days) → PostgreSQL
   - Historical queries → Cold Storage Query Engine
   - Scanner layers use StorageRouter for automatic routing

## Cold Storage Structure

Data is stored in a partitioned structure for efficient querying:

```
data_lake/cold_storage/
├── market_data/
│   └── year=2024/month=01/day=15/
│       └── market_data_20240115.parquet
├── news/
│   └── year=2024/month=01/day=15/
│       └── news_20240115.parquet
├── features/
│   └── year=2024/month=01/day=15/
│       └── features_20240115.parquet
└── social_sentiment/
    └── year=2024/month=01/day=15/
        └── social_sentiment_20240115.parquet
```

## Migration Guide

### Initial Setup

1. **Update Configuration**:
   - Ensure `unified_config.yaml` has the storage lifecycle settings
   - Set `hot_days: 30` and `hot_intervals: ["1day"]`

2. **Run Cleanup Script**:
   ```bash
   # First analyze what will be cleaned
   python src/main/scripts/cleanup_hot_storage.py --analyze-only
   
   # Then run cleanup
   python src/main/scripts/cleanup_hot_storage.py
   ```

3. **Setup Cron Job**:
   ```bash
   # Add to crontab
   crontab -e
   
   # Add this line (adjust path)
   0 2 * * * cd /path/to/ai_trader && python src/main/jobs/storage_rotation_job.py >> logs/storage_rotation.log 2>&1
   ```

### Ongoing Operations

1. **Monitor Storage**:
   - Check PostgreSQL size regularly
   - Monitor archival job logs
   - Verify cold storage integrity

2. **Adjust Settings**:
   - Change `hot_days` if needed
   - Add/remove intervals from `hot_intervals`
   - Modify archival schedule

3. **Query Historical Data**:
   - Use ColdStorageQueryEngine for old data
   - Scanner layers automatically route queries
   - Consider caching frequently accessed historical data

## Benefits

1. **Performance**: 
   - Faster queries on recent data
   - Reduced PostgreSQL load
   - Better cache utilization

2. **Cost**: 
   - ~95% reduction in database storage
   - Cheaper cold storage for historical data
   - Lower backup costs

3. **Scalability**:
   - Can store years of historical data
   - Easy to adjust hot window
   - Parallel processing for cold queries

## Troubleshooting

### Common Issues

1. **Archival Job Fails**:
   - Check disk space on both hot and cold storage
   - Verify database permissions
   - Check archive path exists and is writable

2. **Queries Missing Data**:
   - Ensure data is within hot window for PostgreSQL queries
   - Check cold storage for historical data
   - Verify scanner layers use proper routing

3. **Performance Issues**:
   - Consider increasing `hot_days` if needed
   - Add more intervals to `hot_intervals`
   - Optimize cold storage query patterns

### Monitoring

Check archival status:
```python
from main.data_pipeline.storage.data_lifecycle_manager import DataLifecycleManager
status = await lifecycle_manager.get_archival_status()
```

View logs:
```bash
tail -f logs/storage_rotation.log
```

## Future Enhancements

1. **S3 Integration**: Move cold storage to S3 for better scalability
2. **Compression**: Add compression for cold storage files
3. **Tiered Storage**: Add warm tier for 30-90 day data
4. **Smart Caching**: Cache frequently accessed cold data in Redis
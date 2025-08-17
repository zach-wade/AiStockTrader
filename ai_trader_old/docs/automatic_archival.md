# Automatic Data Archival After Backfill

## Overview

The AI Trader system now automatically archives old data to cold storage after every backfill operation. This means you don't need to set up cron jobs unless you want additional scheduled archival runs.

## How It Works

### 1. During Backfill

When you run the backfill command:

```bash
python ai_trader.py backfill --days 30
```

The system automatically:

1. Downloads data from your configured sources
2. Stores ALL data in cold storage (Data Lake)
3. Loads only the last 30 days of daily (1day) data to PostgreSQL
4. **Runs automatic archival** to clean up any data older than 30 days

### 2. Configuration

The automatic archival is controlled by this setting in `unified_config.yaml`:

```yaml
storage:
  lifecycle:
    hot_days: 30  # Days to keep in PostgreSQL
    hot_intervals:  # Which intervals to keep hot
      - "1day"  # Only daily bars in hot storage
    archive_on_backfill: true  # THIS ENABLES AUTOMATIC ARCHIVAL
```

### 3. What Gets Archived

- **Market Data**: Any records older than 30 days
- **News Data**: Any records older than 30 days
- **Social Sentiment**: Any records older than 30 days
- **Features**: Any computed features older than 30 days

### 4. Where Archived Data Goes

Archived data is stored in partitioned Parquet files:

```
data_lake/cold_storage/
├── market_data/
│   └── year=2024/month=01/day=15/
│       └── market_data_20240115.parquet
└── news_data/
    └── year=2024/month=01/day=15/
        └── news_20240115.parquet
```

## Monitoring Archival

### During Backfill

Watch for these messages:

```
ℹ️ Running data archival cycle
   Moving old data to cold storage...
✅ Data archival completed
   Archived 50,000 records to cold storage
```

Or if no data needs archiving:

```
ℹ️ Data archival
   No data eligible for archiving
```

### Check Logs

Detailed archival logs are in your application logs. Look for:

- `DataLifecycleManager` entries
- Archive operations
- Cleanup confirmations

## Manual Archival (Optional)

If you want to run archival without a full backfill:

### One-Time Cleanup

```bash
# Analyze what would be archived
python src/main/scripts/cleanup_hot_storage.py --analyze-only

# Run cleanup (will prompt for confirmation)
python src/main/scripts/cleanup_hot_storage.py
```

### Scheduled Job

```bash
# Run the storage rotation job manually
python src/main/jobs/storage_rotation_job.py

# Dry run to see what would be archived
python src/main/jobs/storage_rotation_job.py --dry-run
```

### Cron Schedule (Optional)

Only needed if you want archival to run independently of backfill:

```bash
# Add to crontab for daily 2 AM runs
0 2 * * * cd /path/to/ai_trader && python src/main/jobs/storage_rotation_job.py
```

## Troubleshooting

### Issue: Archival Not Running

1. Check configuration: `archive_on_backfill: true`
2. Look for errors in logs after "backfill_symbols returned"
3. Verify DataLifecycleManager is initialized properly

### Issue: Data Not Being Archived

1. Check if data is actually older than 30 days
2. Verify table names match (market_data, news_data, etc.)
3. Check cold storage permissions

### Issue: Timeout Errors

1. Database might be under heavy load
2. Try running archival separately with storage rotation job
3. Consider increasing timeouts in configuration

## Benefits

1. **Automatic**: No manual intervention needed
2. **Efficient**: Runs only when new data is added
3. **Safe**: Archives to cold storage before deletion
4. **Configurable**: Easy to adjust retention period
5. **Integrated**: Part of normal backfill workflow

## FAQ

**Q: Do I need to set up cron jobs?**
A: No! Archival runs automatically after every backfill. Cron is only needed if you want additional scheduled runs.

**Q: What if I want to keep more than 30 days?**
A: Change `hot_days` in the configuration to your desired retention period.

**Q: Can I disable automatic archival?**
A: Yes, set `archive_on_backfill: false` in the configuration.

**Q: How do I query archived data?**
A: Use the ColdStorageQueryEngine or scanner layers will automatically route to cold storage for historical queries.

**Q: Is my data safe during archival?**
A: Yes! Data is copied to cold storage and verified before being removed from hot storage.

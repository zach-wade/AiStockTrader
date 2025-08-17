# Resetting Large Market Data Table

## When You Have 160M+ Records

If your market_data table has grown to hundreds of millions of records, the most efficient approach is to drop and recreate it rather than trying to delete old records.

## Step-by-Step Process

### 1. Analyze Current State

First, check what you're dealing with:

```bash
python src/main/scripts/reset_market_data_table.py --analyze-only
```

This will show:

- Current row count (might timeout if too large)
- Table size on disk
- Index sizes
- Date range of data

### 2. Ensure Cold Storage Has Your Data

Before dropping the table, verify your historical data is safe in cold storage:

```bash
# Check cold storage
ls -la data_lake/cold_storage/market_data/
```

### 3. Reset the Table

Run the reset script:

```bash
python src/main/scripts/reset_market_data_table.py
```

This will:

1. Backup the table structure to a .sql file
2. Prompt for confirmation (type 'yes')
3. DROP the market_data table
4. Recreate it with proper indexes

To skip confirmation (careful!):

```bash
python src/main/scripts/reset_market_data_table.py --force
```

### 4. Run Fresh Backfill

Now populate with only 30 days of daily data:

```bash
python ai_trader.py backfill --days 30
```

With the new configuration, this will:

- Download data for the last 30 days
- Store only daily (1day) intervals in PostgreSQL
- Keep all other intervals in cold storage
- Automatically archive any old data going forward

### 5. Verify Success

Check the new table size:

```bash
python src/main/scripts/reset_market_data_table.py --analyze-only
```

You should see:

- Much smaller row count (thousands instead of millions)
- Reasonable table size
- Data only for the last 30 days

## Expected Results

### Before Reset

- 160 million records
- Gigabytes of storage
- Slow queries and timeouts
- All intervals stored

### After Reset

- ~30,000 records (1000 symbols × 30 days)
- Megabytes of storage
- Fast queries
- Only daily intervals

## Why This Works

1. **DROP is instant**: Unlike DELETE which has to process each row
2. **Clean slate**: No fragmentation or bloat
3. **Proper configuration**: New data follows 30-day/daily-only rules
4. **Automatic maintenance**: Future backfills will maintain the 30-day window

## Troubleshooting

### Can't drop table (locks)

```sql
-- Check for locks
SELECT * FROM pg_locks WHERE relation = 'market_data'::regclass;

-- Kill blocking connections if needed
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'your_db';
```

### Want to keep some recent data

Instead of full reset, you could:

1. Create a temporary table with recent data
2. Drop the original
3. Rename temporary to market_data

But with 160M records, a clean reset is usually faster.

## Alternative: Manual Cleanup

If you really don't want to drop the table:

```sql
-- This will be VERY slow with 160M records
DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL '30 days';
VACUUM FULL market_data;  -- Reclaim space
```

But this could take hours and lock the table.

## Prevention

With the new configuration:

- Only 30 days kept in hot storage
- Only daily intervals stored
- Automatic archival after each backfill
- No more accumulation of millions of records!

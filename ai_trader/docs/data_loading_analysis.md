# Data Loading Analysis - Duplicate Records Issue

## Summary

The discrepancy between reported loaded records (31,227) and actual database records is due to **duplicate data in the archive**, not a bug in the loading process.

## Key Findings

### 1. Database Contains Correct Data
- **AA has 1,256 1day records** in the `market_data_1h` table (verified)
- **AA has 17,976 1hour records** in the same table
- Total: 19,232 unique records
- No duplicate (symbol, timestamp, interval) combinations exist

### 2. Archive Contains Duplicates
Analysis of archive files shows:
- **1day data**: 0% duplication (1,256 records, all unique)
- **1hour data**: 50% duplication (7,196 total, 3,598 unique)
- **Other intervals**: ~45-50% duplication rates

### 3. Why Duplicates Exist
The archive contains multiple overlapping files:
- Historical backfills create files with full date ranges
- Incremental updates create files with recent data
- Files overlap significantly in their date ranges

### 4. How the System Handles Duplicates

#### Within Buffer (Same Batch)
- The bulk loader tracks seen timestamps in `_seen_timestamps`
- Duplicates within the same 10,000-record buffer are skipped
- This prevents duplicates when processing a single large file

#### Across Buffers (Different Batches)
- When buffer flushes, `_seen_timestamps` is cleared
- Duplicates from different files/batches reach the database
- PostgreSQL's `ON CONFLICT (symbol, timestamp, interval)` clause handles these
- Existing records are updated, not duplicated

### 5. Counting Discrepancy Explained

The load script counts:
1. **All records processed** from archive files (including duplicates)
2. **Records added to buffer** (excluding within-buffer duplicates)
3. **Records in each flush** (may include cross-buffer duplicates)

The database stores:
- **Only unique records** due to the ON CONFLICT constraint

## Improvements Made

1. **Enhanced Logging**:
   - Added duplicate detection logging
   - Show both processed and loaded record counts
   - Track duplicates skipped within buffers

2. **Better Summary**:
   - Show "Market Data Processed" (total from files)
   - Show "Market Data Loaded" (unique records)
   - Show "Duplicates Skipped" count

3. **Diagnostic Scripts**:
   - `check_market_data.py` - Verify database contents
   - `verify_conflict_behavior.py` - Check constraint behavior
   - `analyze_archive_duplicates.py` - Analyze archive duplication

## Recommendations

1. **This is working as designed** - The system correctly handles duplicates
2. **Archive cleanup** (optional) - Could periodically consolidate overlapping files
3. **Use diagnostic scripts** - To verify data integrity when needed

## Usage

To verify your data:
```bash
# Check what's in the database
python scripts/check_market_data.py --symbol AA

# Analyze archive duplicates
python scripts/analyze_archive_duplicates.py --symbol AA --days 90

# Load data with improved logging
python scripts/load_datalake_to_db.py --symbols AA --days 90
```
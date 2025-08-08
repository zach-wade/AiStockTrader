#!/usr/bin/env python3
"""
Diagnostic script to check market_data_1h table contents.

This script helps diagnose issues with data loading by showing:
- Total records by interval
- Date ranges for each interval
- Sample records
- Potential duplicates
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def check_market_data(symbol: str = "AA"):
    """Check market data in database for diagnostics."""
    
    # Initialize config and database
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        print(f"\n=== Checking market_data_1h table for {symbol} ===\n")
        
        # 1. Check total records by interval
        query1 = """
            SELECT interval, COUNT(*) as count
            FROM market_data_1h
            WHERE symbol = $1
            GROUP BY interval
            ORDER BY interval
        """
        
        rows = await db_adapter.fetch_all(query1, {'symbol': symbol})
        print("Records by interval:")
        total = 0
        for row in rows:
            print(f"  {row['interval']}: {row['count']:,} records")
            total += row['count']
        print(f"  TOTAL: {total:,} records\n")
        
        # 2. Check date ranges for each interval
        query2 = """
            SELECT interval, 
                   MIN(timestamp) as min_date, 
                   MAX(timestamp) as max_date,
                   COUNT(DISTINCT DATE(timestamp)) as unique_days
            FROM market_data_1h
            WHERE symbol = $1
            GROUP BY interval
            ORDER BY interval
        """
        
        rows = await db_adapter.fetch_all(query2, {'symbol': symbol})
        print("Date ranges by interval:")
        for row in rows:
            print(f"  {row['interval']}:")
            print(f"    From: {row['min_date']}")
            print(f"    To:   {row['max_date']}")
            print(f"    Unique days: {row['unique_days']}")
        print()
        
        # 3. Check for potential timestamp overlaps between intervals
        query3 = """
            WITH timestamps_by_interval AS (
                SELECT timestamp, interval
                FROM market_data_1h
                WHERE symbol = $1
            )
            SELECT t1.timestamp, 
                   STRING_AGG(DISTINCT t1.interval, ', ' ORDER BY t1.interval) as intervals
            FROM timestamps_by_interval t1
            GROUP BY t1.timestamp
            HAVING COUNT(DISTINCT t1.interval) > 1
            ORDER BY t1.timestamp DESC
            LIMIT 10
        """
        
        rows = await db_adapter.fetch_all(query3, {'symbol': symbol})
        if rows:
            print("Timestamps with multiple intervals (potential conflicts):")
            for row in rows:
                print(f"  {row['timestamp']}: {row['intervals']}")
            print(f"  ... (showing first 10 of potentially more)\n")
        else:
            print("No timestamps with multiple intervals found (good!)\n")
        
        # 4. Sample recent 1day records
        query4 = """
            SELECT timestamp, open, high, low, close, volume, source
            FROM market_data_1h
            WHERE symbol = $1 AND interval = '1day'
            ORDER BY timestamp DESC
            LIMIT 10
        """
        
        rows = await db_adapter.fetch_all(query4, {'symbol': symbol})
        print(f"Recent 1day records (showing {len(rows)}):")
        for row in rows:
            print(f"  {row['timestamp'].strftime('%Y-%m-%d')}: "
                  f"O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f} "
                  f"V={row['volume']:,} Source={row['source']}")
        print()
        
        # 5. Check for gaps in 1day data
        query5 = """
            WITH daily_data AS (
                SELECT DATE(timestamp) as trading_date
                FROM market_data_1h
                WHERE symbol = $1 AND interval = '1day'
                ORDER BY timestamp
            ),
            date_series AS (
                SELECT generate_series(
                    (SELECT MIN(trading_date) FROM daily_data),
                    (SELECT MAX(trading_date) FROM daily_data),
                    '1 day'::interval
                )::date as expected_date
            )
            SELECT COUNT(*) as missing_days
            FROM date_series
            LEFT JOIN daily_data ON date_series.expected_date = daily_data.trading_date
            WHERE daily_data.trading_date IS NULL
              AND EXTRACT(DOW FROM date_series.expected_date) NOT IN (0, 6)  -- Exclude weekends
        """
        
        result = await db_adapter.fetch_one(query5, {'symbol': symbol})
        if result and result['missing_days'] > 0:
            print(f"WARNING: Found {result['missing_days']} missing weekdays in 1day data\n")
        
        # 6. Check partitions
        query6 = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables
            WHERE tablename LIKE 'market_data_1h_%'
              AND tablename ~ '^market_data_1h_y[0-9]{4}_w[0-9]{2}$'
            ORDER BY tablename DESC
            LIMIT 5
        """
        
        rows = await db_adapter.fetch_all(query6)
        if rows:
            print("Recent partitions:")
            for row in rows:
                print(f"  {row['tablename']}: {row['size']}")
        
    finally:
        await db_adapter.close()


async def check_specific_dates(symbol: str = "AA", start_date: str = None, end_date: str = None):
    """Check specific date range for debugging."""
    
    # Initialize config and database
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        print(f"\n=== Checking specific dates for {symbol} ===\n")
        
        # Build date filter
        date_filter = ""
        params = {'symbol': symbol}
        
        if start_date:
            date_filter += " AND timestamp >= $2"
            params['start_date'] = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_date:
            date_filter += f" AND timestamp <= ${len(params) + 1}"
            params['end_date'] = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        query = f"""
            SELECT DATE(timestamp) as date, interval, COUNT(*) as records
            FROM market_data_1h
            WHERE symbol = $1 {date_filter}
            GROUP BY DATE(timestamp), interval
            ORDER BY date DESC, interval
        """
        
        rows = await db_adapter.fetch_all(query, params)
        
        current_date = None
        for row in rows:
            if row['date'] != current_date:
                current_date = row['date']
                print(f"\n{current_date}:")
            print(f"  {row['interval']}: {row['records']} records")
        
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check market_data_1h table contents")
    parser.add_argument("--symbol", default="AA", help="Symbol to check")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--dates", action="store_true", help="Show detailed date breakdown")
    
    args = parser.parse_args()
    
    # Run main check
    asyncio.run(check_market_data(args.symbol))
    
    # Run date check if requested
    if args.dates or args.start or args.end:
        asyncio.run(check_specific_dates(args.symbol, args.start, args.end))
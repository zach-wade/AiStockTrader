#!/usr/bin/env python3
"""
Verify ON CONFLICT behavior for market_data_1h table.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def verify_conflict_behavior(symbol: str = "AA"):
    """Check for any issues with ON CONFLICT behavior."""
    
    # Initialize config and database
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        print(f"\n=== Verifying ON CONFLICT behavior for {symbol} ===\n")
        
        # 1. Check unique constraint
        query1 = """
            SELECT 
                tc.constraint_name,
                kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'UNIQUE' 
                AND tc.table_name = 'market_data_1h'
            ORDER BY kcu.ordinal_position;
        """
        
        rows = await db_adapter.fetch_all(query1)
        print("Unique constraints on market_data_1h:")
        current_constraint = None
        for row in rows:
            if row['constraint_name'] != current_constraint:
                if current_constraint:
                    print()
                current_constraint = row['constraint_name']
                print(f"  {current_constraint}:")
            print(f"    - {row['column_name']}")
        print()
        
        # 2. Check for duplicate (symbol, timestamp, interval) combinations
        query2 = """
            SELECT symbol, timestamp, interval, COUNT(*) as count
            FROM market_data_1h
            WHERE symbol = $1
            GROUP BY symbol, timestamp, interval
            HAVING COUNT(*) > 1
            LIMIT 10
        """
        
        rows = await db_adapter.fetch_all(query2, {'symbol': symbol})
        if rows:
            print("ERROR: Found duplicate (symbol, timestamp, interval) combinations:")
            for row in rows:
                print(f"  {row['symbol']} at {row['timestamp']} interval={row['interval']}: {row['count']} duplicates")
        else:
            print("✓ No duplicate (symbol, timestamp, interval) combinations found")
        print()
        
        # 3. Check if same timestamp exists with different intervals
        query3 = """
            SELECT 
                DATE(timestamp) as date,
                EXTRACT(HOUR FROM timestamp) as hour,
                COUNT(DISTINCT interval) as interval_count,
                STRING_AGG(DISTINCT interval, ', ' ORDER BY interval) as intervals
            FROM market_data_1h
            WHERE symbol = $1
            GROUP BY DATE(timestamp), EXTRACT(HOUR FROM timestamp)
            HAVING COUNT(DISTINCT interval) > 1
            ORDER BY date DESC, hour
            LIMIT 20
        """
        
        rows = await db_adapter.fetch_all(query3, {'symbol': symbol})
        if rows:
            print("Dates/hours with multiple intervals (expected behavior):")
            for row in rows:
                print(f"  {row['date']} {int(row['hour']):02d}:00 - {row['intervals']}")
            print("\n✓ This is correct - 1hour and 1day data can coexist at same timestamp")
        else:
            print("No timestamps with multiple intervals found")
        print()
        
        # 4. Sample some overlapping timestamps
        query4 = """
            WITH overlap_times AS (
                SELECT timestamp
                FROM market_data_1h
                WHERE symbol = $1
                GROUP BY timestamp
                HAVING COUNT(DISTINCT interval) > 1
                ORDER BY timestamp DESC
                LIMIT 5
            )
            SELECT 
                m.timestamp,
                m.interval,
                m.open,
                m.high,
                m.low,
                m.close,
                m.volume
            FROM market_data_1h m
            JOIN overlap_times o ON m.timestamp = o.timestamp
            WHERE m.symbol = $1
            ORDER BY m.timestamp DESC, m.interval
        """
        
        rows = await db_adapter.fetch_all(query4, {'symbol': symbol})
        if rows:
            print("Sample of overlapping timestamps with different intervals:")
            current_ts = None
            for row in rows:
                if row['timestamp'] != current_ts:
                    current_ts = row['timestamp']
                    print(f"\n  {current_ts}:")
                print(f"    {row['interval']:5} - O={row['open']:.2f} H={row['high']:.2f} "
                      f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:,}")
        
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify ON CONFLICT behavior")
    parser.add_argument("--symbol", default="AA", help="Symbol to check")
    
    args = parser.parse_args()
    
    asyncio.run(verify_conflict_behavior(args.symbol))
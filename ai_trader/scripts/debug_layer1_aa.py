#!/usr/bin/env python3
"""Debug why AA is not in Layer 1 despite meeting requirements."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def debug_layer1_aa():
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        # Simulate what Layer 1 scanner does
        layer1_params = config.get('universe.layer1_filters', {})
        min_dollar_volume = layer1_params.get('min_avg_dollar_volume', 5_000_000)
        min_price = layer1_params.get('min_price', 1.0)
        max_price = layer1_params.get('max_price', 2000.0)
        lookback_days = layer1_params.get('lookback_days', 20)
        target_size = layer1_params.get('target_universe_size', 2000)
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        min_data_points = max(5, int(lookback_days * 0.5))
        
        print("Simulating Layer 1 Scanner for AA...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Min data points: {min_data_points}\n")
        
        # Step 1: Get all active company symbols (what Layer 1 gets as input)
        input_query = """
            SELECT symbol FROM companies WHERE is_active = TRUE
        """
        input_rows = await db_adapter.fetch_all(input_query)
        input_symbols = [row['symbol'] for row in input_rows]
        
        print(f"Step 1: Input symbols from active companies: {len(input_symbols)}")
        print(f"  AA in input list: {'YES' if 'AA' in input_symbols else 'NO'}")
        
        # Step 2: Get liquidity data (same query as Layer 1)
        liquidity_query = """
            SELECT 
                symbol,
                AVG(close * volume) as avg_dollar_volume,
                AVG(close) as avg_price,
                AVG(volume) as avg_volume,
                COUNT(*) as data_points,
                MAX(close * volume) as max_dollar_volume,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close * volume) / 1000000 as liquidity_score
            FROM market_data
            WHERE symbol = ANY($1::text[])
                AND timestamp >= $2
                AND timestamp <= $3
                AND close > 0
                AND volume > 0
            GROUP BY symbol
            HAVING COUNT(*) >= $4
        """
        
        # Check specifically for AA
        aa_only = ['AA']
        aa_result = await db_adapter.fetch_all(
            liquidity_query,
            {'symbols': aa_only, 'start': start_date, 'end': end_date, 'min_points': min_data_points}
        )
        
        if aa_result:
            print("\nStep 2: AA liquidity data retrieved:")
            row = aa_result[0]
            print(f"  Avg dollar volume: ${row['avg_dollar_volume']:,.0f}")
            print(f"  Avg price: ${row['avg_price']:.2f}")
            print(f"  Liquidity score: {row['liquidity_score']:.2f}")
            print(f"  Data points: {row['data_points']}")
        else:
            print("\nStep 2: NO LIQUIDITY DATA RETURNED FOR AA!")
            print("This is the problem - the query is not returning AA data")
            
            # Debug why
            debug_query = """
                SELECT COUNT(*) as total_records,
                       MIN(timestamp) as oldest,
                       MAX(timestamp) as newest
                FROM market_data
                WHERE symbol = 'AA'
                    AND timestamp >= $1
                    AND timestamp <= $2
            """
            debug_result = await db_adapter.fetch_one(
                debug_query,
                {'start': start_date, 'end': end_date}
            )
            print(f"\nDebug info:")
            print(f"  AA records in date range: {debug_result['total_records']}")
            if debug_result['total_records'] > 0:
                print(f"  Date range: {debug_result['oldest']} to {debug_result['newest']}")
            
        # Step 3: Check if AA passes filters
        if aa_result:
            row = aa_result[0]
            volume_pass = row['avg_dollar_volume'] >= min_dollar_volume
            price_pass = min_price <= row['avg_price'] <= max_price
            
            print("\nStep 3: Filter check:")
            print(f"  Volume filter: {'PASS' if volume_pass else 'FAIL'}")
            print(f"  Price filter: {'PASS' if price_pass else 'FAIL'}")
            print(f"  Overall: {'PASS' if volume_pass and price_pass else 'FAIL'}")
            
        # Step 4: Check ranking
        print("\nStep 4: Checking liquidity ranking...")
        
        # Get all symbols that pass filters
        all_liquidity = await db_adapter.fetch_all(
            liquidity_query,
            {'symbols': input_symbols, 'start': start_date, 'end': end_date, 'min_points': min_data_points}
        )
        
        print(f"  Total symbols with liquidity data: {len(all_liquidity)}")
        
        # Apply filters
        filtered = []
        for row in all_liquidity:
            if (row['avg_dollar_volume'] >= min_dollar_volume and
                min_price <= row['avg_price'] <= max_price):
                filtered.append(row)
        
        print(f"  Symbols passing filters: {len(filtered)}")
        
        # Sort by liquidity score
        sorted_symbols = sorted(filtered, key=lambda x: x['liquidity_score'], reverse=True)
        
        # Find AA's position
        aa_position = None
        for i, row in enumerate(sorted_symbols):
            if row['symbol'] == 'AA':
                aa_position = i + 1
                break
        
        if aa_position:
            print(f"  AA's position: #{aa_position} out of {len(sorted_symbols)}")
            print(f"  Makes the cut for top {target_size}: {'YES' if aa_position <= target_size else 'NO'}")
        else:
            print("  AA not found in filtered list!")
            
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(debug_layer1_aa())
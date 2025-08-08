#!/usr/bin/env python3
"""Analyze why Layer 1 scanner only finds ~1450 symbols instead of 2000."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def analyze_layer1_issue():
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        # Get Layer 1 parameters
        layer1_params = config.get('universe.layer1_filters', {})
        min_dollar_volume = layer1_params.get('min_avg_dollar_volume', 5_000_000)
        min_price = layer1_params.get('min_price', 1.0)
        max_price = layer1_params.get('max_price', 2000.0)
        lookback_days = layer1_params.get('lookback_days', 20)
        
        print('Layer 1 Scanner Analysis:')
        print(f'Target universe size: 2000')
        print(f'Actual qualified: ~1450')
        print(f'\nWhy are we missing ~550 symbols?\n')
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        min_data_points = max(5, int(lookback_days * 0.5))
        
        print(f'Analysis period: {start_date.date()} to {end_date.date()}')
        print(f'Minimum data points required: {min_data_points}\n')
        
        # Count symbols that meet each criterion
        query1 = """
            SELECT COUNT(DISTINCT symbol) as total_symbols
            FROM market_data
            WHERE timestamp >= $1 AND timestamp <= $2
        """
        
        result1 = await db_adapter.fetch_one(query1, {'start': start_date, 'end': end_date})
        print(f'1. Total symbols with ANY data in last {lookback_days} days: {result1["total_symbols"]}')
        
        # Symbols with enough data points
        query2 = """
            SELECT COUNT(*) as symbols_with_enough_data
            FROM (
                SELECT symbol, COUNT(*) as data_points
                FROM market_data
                WHERE timestamp >= $1 AND timestamp <= $2
                GROUP BY symbol
                HAVING COUNT(*) >= $3
            ) t
        """
        
        result2 = await db_adapter.fetch_one(query2, {'start': start_date, 'end': end_date, 'min_points': min_data_points})
        print(f'2. Symbols with >= {min_data_points} data points: {result2["symbols_with_enough_data"]}')
        
        # Symbols meeting dollar volume requirement
        query3 = """
            SELECT COUNT(*) as symbols_meeting_volume
            FROM (
                SELECT symbol, AVG(close * volume) as avg_dollar_volume
                FROM market_data
                WHERE timestamp >= $1 AND timestamp <= $2
                    AND close > 0 AND volume > 0
                GROUP BY symbol
                HAVING COUNT(*) >= $3 AND AVG(close * volume) >= $4
            ) t
        """
        
        result3 = await db_adapter.fetch_one(
            query3, 
            {'start': start_date, 'end': end_date, 'min_points': min_data_points, 'min_vol': min_dollar_volume}
        )
        print(f'3. Symbols meeting ${min_dollar_volume:,} volume requirement: {result3["symbols_meeting_volume"]}')
        
        # Symbols meeting price requirements
        query3b = """
            SELECT COUNT(*) as symbols_in_price_range
            FROM (
                SELECT symbol, AVG(close) as avg_price
                FROM market_data
                WHERE timestamp >= $1 AND timestamp <= $2
                    AND close > 0
                GROUP BY symbol
                HAVING COUNT(*) >= $3 
                    AND AVG(close) >= $4
                    AND AVG(close) <= $5
            ) t
        """
        
        result3b = await db_adapter.fetch_one(
            query3b,
            {
                'start': start_date, 
                'end': end_date, 
                'min_points': min_data_points,
                'min_price': min_price,
                'max_price': max_price
            }
        )
        print(f'4. Symbols in price range ${min_price}-${max_price}: {result3b["symbols_in_price_range"]}')
        
        # Symbols meeting all requirements
        query4 = """
            SELECT COUNT(*) as symbols_meeting_all
            FROM (
                SELECT 
                    symbol,
                    AVG(close * volume) as avg_dollar_volume,
                    AVG(close) as avg_price
                FROM market_data
                WHERE timestamp >= $1 AND timestamp <= $2
                    AND close > 0 AND volume > 0
                GROUP BY symbol
                HAVING COUNT(*) >= $3 
                    AND AVG(close * volume) >= $4
                    AND AVG(close) >= $5
                    AND AVG(close) <= $6
            ) t
        """
        
        result4 = await db_adapter.fetch_one(
            query4,
            {
                'start': start_date, 
                'end': end_date, 
                'min_points': min_data_points,
                'min_vol': min_dollar_volume,
                'min_price': min_price,
                'max_price': max_price
            }
        )
        print(f'5. Symbols meeting ALL requirements: {result4["symbols_meeting_all"]}')
        
        print(f'\nConclusion:')
        print(f'- Only {result4["symbols_meeting_all"]} symbols meet all Layer 1 criteria')
        print(f'- The scanner cannot select 2000 symbols because only ~1450-1500 qualify')
        print(f'- AA is ranked #1914 by liquidity, but since only ~1450 qualify, it doesn\'t make the cut')
        
        # Show some examples of symbols that fail each filter
        print('\nExamples of symbols failing each filter:')
        
        # Symbols failing volume requirement
        query5 = """
            SELECT symbol, AVG(close * volume) as avg_dollar_volume
            FROM market_data
            WHERE timestamp >= $1 AND timestamp <= $2
                AND close > 0 AND volume > 0
            GROUP BY symbol
            HAVING COUNT(*) >= $3 
                AND AVG(close * volume) < $4
                AND AVG(close) >= $5
                AND AVG(close) <= $6
            ORDER BY AVG(close * volume) DESC
            LIMIT 5
        """
        
        rows = await db_adapter.fetch_all(
            query5,
            {
                'start': start_date, 
                'end': end_date, 
                'min_points': min_data_points,
                'min_vol': min_dollar_volume,
                'min_price': min_price,
                'max_price': max_price
            }
        )
        
        if rows:
            print(f'\nSymbols failing volume requirement (<${min_dollar_volume:,}):')
            for row in rows:
                print(f'  {row["symbol"]}: ${row["avg_dollar_volume"]:,.0f}')
        
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(analyze_layer1_issue())
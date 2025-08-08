#!/usr/bin/env python3
"""
Analyze archive files for duplicate data.

This script checks:
- How many archive files exist for each interval
- Whether files contain overlapping data
- Total unique vs duplicate records
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.archive import DataArchive


async def analyze_archive_duplicates(symbol: str = "AA", days: int = 1825):
    """Analyze archive for duplicate records."""
    
    # Initialize config and archive
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    archive_config = config.get('storage.archive', {})
    archive = DataArchive(archive_config)
    
    # Set date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    print(f"\n=== Analyzing archive for {symbol} ===")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")
    
    try:
        # Query archive
        raw_records = await archive.query_raw_records(
            source='polygon',
            data_type='market_data',
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Total raw records found: {len(raw_records)}")
        
        # Group by interval
        records_by_interval = defaultdict(list)
        for record in raw_records:
            interval = record.metadata.get('interval', 'unknown')
            records_by_interval[interval].append(record)
        
        print("\nRecords by interval:")
        for interval, records in sorted(records_by_interval.items()):
            print(f"  {interval}: {len(records)} files")
        
        # Analyze each interval
        for interval, interval_records in sorted(records_by_interval.items()):
            print(f"\n=== Analyzing {interval} data ===")
            
            # Track all timestamps across files
            all_timestamps = []
            file_info = []
            
            for record in interval_records:
                if record.data and 'data' in record.data:
                    market_data = record.data['data']
                    if isinstance(market_data, list) and market_data:
                        # Convert to DataFrame to analyze
                        df = pd.DataFrame(market_data)
                        
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            
                            # Collect file info
                            file_info.append({
                                'file': record.metadata.get('file_path', 'unknown'),
                                'records': len(df),
                                'min_date': df['timestamp'].min(),
                                'max_date': df['timestamp'].max(),
                                'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days
                            })
                            
                            # Collect all timestamps
                            all_timestamps.extend(df['timestamp'].tolist())
            
            # Analyze timestamps
            if all_timestamps:
                unique_timestamps = set(all_timestamps)
                total_timestamps = len(all_timestamps)
                unique_count = len(unique_timestamps)
                duplicate_count = total_timestamps - unique_count
                
                print(f"  Total records across all files: {total_timestamps:,}")
                print(f"  Unique timestamps: {unique_count:,}")
                print(f"  Duplicate timestamps: {duplicate_count:,}")
                print(f"  Duplication rate: {duplicate_count/total_timestamps*100:.1f}%")
                
                # Show file info
                print(f"\n  Files analyzed ({len(file_info)}):")
                for i, info in enumerate(file_info[:5]):  # Show first 5
                    print(f"    File {i+1}:")
                    print(f"      Records: {info['records']:,}")
                    print(f"      Date range: {info['min_date'].date()} to {info['max_date'].date()} ({info['date_range_days']} days)")
                
                if len(file_info) > 5:
                    print(f"    ... and {len(file_info) - 5} more files")
                
                # Check for overlapping date ranges
                print(f"\n  Checking for overlapping date ranges:")
                overlaps = 0
                for i in range(len(file_info)):
                    for j in range(i + 1, len(file_info)):
                        file1 = file_info[i]
                        file2 = file_info[j]
                        
                        # Check if date ranges overlap
                        if (file1['min_date'] <= file2['max_date'] and 
                            file1['max_date'] >= file2['min_date']):
                            overlaps += 1
                
                print(f"    Found {overlaps} overlapping file pairs")
                
                # Expected records in DB vs actual
                print(f"\n  Expected records in DB (unique timestamps): {unique_count:,}")
                if interval == '1day':
                    print(f"  Actual records in DB: 1,256")
                    if unique_count > 1256:
                        print(f"  Missing in DB: {unique_count - 1256:,} records")
                
    except Exception as e:
        print(f"Error analyzing archive: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze archive for duplicates")
    parser.add_argument("--symbol", default="AA", help="Symbol to analyze")
    parser.add_argument("--days", type=int, default=1825, help="Number of days to analyze")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_archive_duplicates(args.symbol, args.days))
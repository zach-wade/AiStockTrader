#!/usr/bin/env python3
"""
Validate TSLA data after backfill - comprehensive checks for database and data lake.
"""
import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.config import get_config_manager


async def validate_tsla_data():
    """Comprehensive validation of TSLA data after backfill."""
    
    # Get config and database
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "TSLA",
        "validations": {},
        "issues": [],
        "success": True
    }
    
    try:
        print("=" * 80)
        print("TSLA DATA VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {results['timestamp']}")
        print("\n")
        
        # 1. Market Data Coverage
        print("1. MARKET DATA COVERAGE")
        print("-" * 50)
        
        market_query = """
            SELECT 
                interval,
                COUNT(*) as records,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                COUNT(DISTINCT DATE(timestamp)) as trading_days
            FROM market_data
            WHERE symbol = 'TSLA'
            GROUP BY interval
            ORDER BY interval;
        """
        
        market_results = await db_adapter.fetch_all(market_query)
        results["validations"]["market_data"] = {}
        
        expected_intervals = ['1minute', '5minute', '15minute', '30minute', '1hour', '1day']
        found_intervals = []
        
        for row in market_results:
            interval = row['interval']
            found_intervals.append(interval)
            
            results["validations"]["market_data"][interval] = {
                "records": row['records'],
                "earliest": str(row['earliest']),
                "latest": str(row['latest']),
                "trading_days": row['trading_days']
            }
            
            # Calculate expected records (rough estimate)
            days_span = (row['latest'] - row['earliest']).days
            
            print(f"\n{interval}:")
            print(f"  Records: {row['records']:,}")
            print(f"  Date Range: {row['earliest'].date()} to {row['latest'].date()} ({days_span} days)")
            print(f"  Trading Days: {row['trading_days']}")
            
            # Validate date range (should be ~5 years)
            if days_span < 1800:  # Less than ~5 years
                issue = f"{interval} data spans only {days_span} days (expected ~1825)"
                results["issues"].append(issue)
                print(f"  ⚠️  {issue}")
        
        # Check for missing intervals
        missing_intervals = set(expected_intervals) - set(found_intervals)
        if missing_intervals:
            issue = f"Missing intervals: {missing_intervals}"
            results["issues"].append(issue)
            print(f"\n⚠️  {issue}")
            
        # 2. Data Quality Checks
        print("\n\n2. DATA QUALITY CHECKS")
        print("-" * 50)
        
        # Check for duplicate volumes (our previous issue)
        dup_query = """
            SELECT 
                interval,
                volume,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / 
                    (SELECT COUNT(*) FROM market_data 
                     WHERE symbol = 'TSLA' AND interval = md.interval), 2) as pct
            FROM market_data md
            WHERE symbol = 'TSLA'
            GROUP BY interval, volume
            HAVING COUNT(*) > 100
            ORDER BY interval, count DESC
            LIMIT 20;
        """
        
        dup_results = await db_adapter.fetch_all(dup_query)
        
        if dup_results:
            print("\n⚠️  Potential duplicate volume issues found:")
            for row in dup_results:
                if row['pct'] > 5.0:  # More than 5% with same volume is suspicious
                    issue = f"{row['interval']}: Volume {row['volume']:,.2f} appears {row['count']} times ({row['pct']}%)"
                    results["issues"].append(issue)
                    print(f"  {issue}")
        else:
            print("✓ No significant duplicate volumes found")
            
        # Check for data gaps
        gap_query = """
            WITH time_series AS (
                SELECT 
                    interval,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY interval ORDER BY timestamp) as prev_timestamp
                FROM market_data
                WHERE symbol = 'TSLA'
            )
            SELECT 
                interval,
                COUNT(*) as gap_count
            FROM time_series
            WHERE 
                CASE 
                    WHEN interval = '1minute' THEN timestamp - prev_timestamp > INTERVAL '5 minutes'
                    WHEN interval = '5minute' THEN timestamp - prev_timestamp > INTERVAL '15 minutes'
                    WHEN interval = '1hour' THEN timestamp - prev_timestamp > INTERVAL '3 hours'
                    WHEN interval = '1day' THEN timestamp - prev_timestamp > INTERVAL '5 days'
                    ELSE FALSE
                END
            GROUP BY interval;
        """
        
        gap_results = await db_adapter.fetch_all(gap_query)
        
        print("\nData Gaps:")
        if gap_results:
            for row in gap_results:
                print(f"  {row['interval']}: {row['gap_count']} gaps detected")
                if row['gap_count'] > 50:
                    issue = f"{row['interval']} has {row['gap_count']} time gaps"
                    results["issues"].append(issue)
        else:
            print("  ✓ No significant gaps found")
            
        # 3. News Data
        print("\n\n3. NEWS DATA")
        print("-" * 50)
        
        news_query = """
            SELECT 
                COUNT(*) as total_articles,
                COUNT(DISTINCT DATE(published_at)) as days_with_news,
                MIN(published_at) as earliest,
                MAX(published_at) as latest,
                AVG(sentiment_score) as avg_sentiment
            FROM news_data
            WHERE symbol = 'TSLA';
        """
        
        news_result = await db_adapter.fetch_one(news_query)
        
        if news_result and news_result['total_articles'] > 0:
            results["validations"]["news_data"] = {
                "total_articles": news_result['total_articles'],
                "days_with_news": news_result['days_with_news'],
                "earliest": str(news_result['earliest']),
                "latest": str(news_result['latest']),
                "avg_sentiment": float(news_result['avg_sentiment']) if news_result['avg_sentiment'] else None
            }
            
            print(f"Total Articles: {news_result['total_articles']:,}")
            print(f"Days with News: {news_result['days_with_news']}")
            print(f"Date Range: {news_result['earliest'].date()} to {news_result['latest'].date()}")
            print(f"Average Sentiment: {news_result['avg_sentiment']:.3f}" if news_result['avg_sentiment'] else "Average Sentiment: N/A")
        else:
            issue = "No news data found for TSLA"
            results["issues"].append(issue)
            print(f"⚠️  {issue}")
            
        # 4. Financials Data
        print("\n\n4. FINANCIALS DATA")
        print("-" * 50)
        
        fin_query = """
            SELECT 
                COUNT(*) as total_reports,
                COUNT(DISTINCT period_type) as period_types,
                MIN(period_start) as earliest,
                MAX(period_start) as latest
            FROM financials_data
            WHERE symbol = 'TSLA';
        """
        
        fin_result = await db_adapter.fetch_one(fin_query)
        
        if fin_result and fin_result['total_reports'] > 0:
            results["validations"]["financials_data"] = {
                "total_reports": fin_result['total_reports'],
                "period_types": fin_result['period_types'],
                "earliest": str(fin_result['earliest']),
                "latest": str(fin_result['latest'])
            }
            
            print(f"Total Reports: {fin_result['total_reports']}")
            print(f"Period Types: {fin_result['period_types']}")
            print(f"Date Range: {fin_result['earliest']} to {fin_result['latest']}")
            
            # Expect ~20 quarterly reports for 5 years
            if fin_result['total_reports'] < 15:
                issue = f"Only {fin_result['total_reports']} financial reports (expected ~20 for 5 years)"
                results["issues"].append(issue)
                print(f"⚠️  {issue}")
        else:
            issue = "No financials data found for TSLA"
            results["issues"].append(issue)
            print(f"⚠️  {issue}")
            
        # 5. Data Lake Validation
        print("\n\n5. DATA LAKE VALIDATION")
        print("-" * 50)
        
        data_lake_path = Path("data_lake/raw")
        
        # Check market data files
        market_files = list(data_lake_path.glob("market_data/symbol=TSLA/interval=*/date=*/*.parquet"))
        news_files = list(data_lake_path.glob("news/symbol=TSLA/date=*/*.parquet"))
        
        results["validations"]["data_lake"] = {
            "market_data_files": len(market_files),
            "news_files": len(news_files)
        }
        
        print(f"Market Data Files: {len(market_files)}")
        print(f"News Files: {len(news_files)}")
        
        # Sample validation of a parquet file
        if market_files:
            sample_file = market_files[0]
            try:
                df = pd.read_parquet(sample_file)
                print(f"\nSample file validation ({sample_file.name}):")
                print(f"  Records: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                
                # Check for our volume issue
                if 'volume' in df.columns:
                    vol_counts = df['volume'].value_counts()
                    top_vol = vol_counts.iloc[0]
                    if top_vol / len(df) > 0.05:  # More than 5%
                        issue = f"Data lake file has suspicious volumes: {vol_counts.index[0]} appears {top_vol} times"
                        results["issues"].append(issue)
                        print(f"  ⚠️  {issue}")
            except Exception as e:
                print(f"  Error reading sample file: {e}")
                
        # 6. Summary
        print("\n\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        if not results["issues"]:
            print("✅ ALL VALIDATIONS PASSED")
            results["success"] = True
        else:
            print(f"⚠️  FOUND {len(results['issues'])} ISSUES:")
            results["success"] = False
            for i, issue in enumerate(results["issues"], 1):
                print(f"{i}. {issue}")
                
        # Save results
        output_file = Path("data/validation/tsla_backfill_validation.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nDetailed results saved to: {output_file}")
        
        return results["success"]
        
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    success = asyncio.run(validate_tsla_data())
    sys.exit(0 if success else 1)
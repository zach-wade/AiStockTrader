#!/usr/bin/env python3
"""Check why AA is not Layer 1 qualified."""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def check_aa_liquidity():
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    try:
        # Get Layer 1 filter parameters
        layer1_params = config.get("universe.layer1_filters", {})
        min_dollar_volume = layer1_params.get("min_avg_dollar_volume", 5_000_000)
        min_price = layer1_params.get("min_price", 1.0)
        max_price = layer1_params.get("max_price", 2000.0)
        lookback_days = layer1_params.get("lookback_days", 20)
        target_size = layer1_params.get("target_universe_size", 2000)

        print("Layer 1 Filter Requirements:")
        print(f"  Min avg dollar volume: ${min_dollar_volume:,}")
        print(f"  Min price: ${min_price}")
        print(f"  Max price: ${max_price}")
        print(f"  Lookback days: {lookback_days}")
        print(f"  Target universe size: {target_size}")

        # Calculate AA's liquidity metrics using the same query as Layer 1 scanner
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=lookback_days)

        # First check if market_data view exists
        try:
            view_query = "SELECT 1 FROM market_data WHERE symbol = 'AA' LIMIT 1"
            await db_adapter.fetch_one(view_query)
            use_view = True
            print("\n✓ Using market_data view")
        except:
            use_view = False
            print("\n✗ market_data view not found, using market_data_1h table")

        if use_view:
            # Use the same query as Layer 1 scanner
            query = """
                SELECT
                    AVG(close * volume) as avg_dollar_volume,
                    AVG(close) as avg_price,
                    AVG(volume) as avg_volume,
                    COUNT(*) as data_points,
                    MIN(timestamp) as oldest_data,
                    MAX(timestamp) as newest_data,
                    AVG(close * volume) / 1000000 as liquidity_score
                FROM market_data
                WHERE symbol = 'AA'
                    AND timestamp >= $1
                    AND timestamp <= $2
                    AND close > 0
                    AND volume > 0
            """
        else:
            # Fallback to direct table query
            query = """
                SELECT
                    AVG(close * volume) as avg_dollar_volume,
                    AVG(close) as avg_price,
                    AVG(volume) as avg_volume,
                    COUNT(*) as data_points,
                    MIN(timestamp) as oldest_data,
                    MAX(timestamp) as newest_data,
                    AVG(close * volume) / 1000000 as liquidity_score
                FROM market_data_1h
                WHERE symbol = 'AA'
                    AND timestamp >= $1
                    AND timestamp <= $2
                    AND close > 0
                    AND volume > 0
                    AND interval = '1day'
            """

        result = await db_adapter.fetch_one(query, {"start_date": start_date, "end_date": end_date})

        print(f"\nAA Liquidity Metrics (last {lookback_days} days):")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")

        if result and result["avg_dollar_volume"]:
            avg_dollar_vol = float(result["avg_dollar_volume"])
            avg_price = float(result["avg_price"])
            liquidity_score = float(result["liquidity_score"]) if result["liquidity_score"] else 0

            print(f"  Avg dollar volume: ${avg_dollar_vol:,.0f}")
            print(f"  Avg price: ${avg_price:.2f}")
            print(f"  Liquidity score: {liquidity_score:.2f}")
            print(f'  Data points: {result["data_points"]}')
            print(
                f'  Actual date range: {result["oldest_data"].date()} to {result["newest_data"].date()}'
            )

            # Check qualification
            min_data_points = max(5, int(lookback_days * 0.5))

            print("\nQualification Check:")
            dollar_vol_pass = avg_dollar_vol >= min_dollar_volume
            print(
                f'  ✓ Dollar volume: ${avg_dollar_vol:,.0f} >= ${min_dollar_volume:,} : {"PASS" if dollar_vol_pass else "FAIL"}'
            )

            price_pass = min_price <= avg_price <= max_price
            print(
                f'  ✓ Price range: ${min_price} <= ${avg_price:.2f} <= ${max_price} : {"PASS" if price_pass else "FAIL"}'
            )

            data_points_pass = result["data_points"] >= min_data_points
            print(
                f'  ✓ Data points: {result["data_points"]} >= {min_data_points} : {"PASS" if data_points_pass else "FAIL"}'
            )

            # Overall qualification
            if dollar_vol_pass and price_pass and data_points_pass:
                print("\n✅ AA meets all Layer 1 requirements!")
                print("\nPossible reasons why it's not qualified:")
                print("  1. Not in top 2000 by liquidity score (target universe size)")
                print("  2. Layer 1 scanner hasn't been run recently")
                print("  3. Different data was available when scanner last ran")

                # Check ranking
                rank_query = """
                    SELECT COUNT(*) + 1 as rank
                    FROM (
                        SELECT symbol, AVG(close * volume) / 1000000 as liquidity_score
                        FROM market_data_1h
                        WHERE timestamp >= $1
                            AND timestamp <= $2
                            AND close > 0
                            AND volume > 0
                            AND interval = '1day'
                        GROUP BY symbol
                        HAVING COUNT(*) >= $3
                    ) t
                    WHERE liquidity_score > $4
                """

                rank_result = await db_adapter.fetch_one(
                    rank_query,
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "min_points": min_data_points,
                        "score": liquidity_score,
                    },
                )

                if rank_result:
                    print(
                        f'\n  AA liquidity rank: ~{rank_result["rank"]} (needs to be in top {target_size})'
                    )
            else:
                print("\n❌ AA does not meet Layer 1 requirements")
        else:
            print("  No liquidity data found for AA in the lookback period")

    finally:
        await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(check_aa_liquidity())

#!/usr/bin/env python3
"""Trace why Layer 1 scanner stops at 1456 symbols."""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from main.config import get_config_manager
from main.config.database_field_mappings import map_company_fields
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def trace_layer1_issue():
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    try:
        # Get Layer 1 parameters
        layer1_params = config.get("universe.layer1_filters", {})
        min_dollar_volume = layer1_params.get("min_avg_dollar_volume", 5_000_000)
        min_price = layer1_params.get("min_price", 1.0)
        max_price = layer1_params.get("max_price", 2000.0)
        lookback_days = layer1_params.get("lookback_days", 20)

        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=lookback_days)
        min_data_points = max(5, int(lookback_days * 0.5))

        print("Tracing Layer 1 Scanner Logic...")
        print(f"Parameters: Volume>=${min_dollar_volume:,}, Price=${min_price}-${max_price}")
        print(f"Date range: {start_date.date()} to {end_date.date()}\n")

        # Step 1: Get input symbols
        input_query = "SELECT symbol FROM companies WHERE is_active = TRUE"
        input_rows = await db_adapter.fetch_all(input_query)
        input_symbols = [row["symbol"] for row in input_rows]
        print(f"Step 1: {len(input_symbols)} active companies\n")

        # Step 2: Get liquidity data (same as scanner)
        query = """
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

        results = await db_adapter.fetch_all(
            query,
            {
                "symbols": input_symbols,
                "start": start_date,
                "end": end_date,
                "min_points": min_data_points,
            },
        )

        print(f"Step 2: {len(results)} symbols have liquidity data\n")

        # Step 3: Apply filters (simulating scanner logic)
        filtered = []
        skipped_mapping = 0
        skipped_volume = 0
        skipped_price = 0

        for row in results:
            data = {
                "symbol": row["symbol"],
                "avg_dollar_volume": (
                    float(row["avg_dollar_volume"]) if row["avg_dollar_volume"] else 0
                ),
                "avg_price": float(row["avg_price"]) if row["avg_price"] else 0,
                "avg_volume": float(row["avg_volume"]) if row["avg_volume"] else 0,
                "data_points": row["data_points"],
                "liquidity_score": float(row["liquidity_score"]) if row["liquidity_score"] else 0,
            }

            # Apply mapping (like scanner does)
            mapped_data = map_company_fields(data, to_db=False)

            # Check if essential fields exist after mapping
            if not all(k in mapped_data for k in ["avg_dollar_volume", "avg_price"]):
                skipped_mapping += 1
                print(f"  Skipped {data['symbol']} - missing fields after mapping")
                continue

            # Volume filter
            if mapped_data["avg_dollar_volume"] < min_dollar_volume:
                skipped_volume += 1
                continue

            # Price filter
            if not (min_price <= mapped_data["avg_price"] <= max_price):
                skipped_price += 1
                continue

            filtered.append(data)

        print("Step 3: Filter results:")
        print(f"  Skipped due to mapping issues: {skipped_mapping}")
        print(f"  Skipped due to volume < ${min_dollar_volume:,}: {skipped_volume}")
        print(f"  Skipped due to price outside ${min_price}-${max_price}: {skipped_price}")
        print(f"  Passed all filters: {len(filtered)}\n")

        # Step 4: Sort and limit
        sorted_symbols = sorted(filtered, key=lambda x: x.get("liquidity_score", 0), reverse=True)
        target_size = 2000
        final_symbols = sorted_symbols[:target_size]

        print("Step 4: Final selection:")
        print(f"  Target size: {target_size}")
        print(f"  Actual size: {len(final_symbols)}")

        # Check AA
        aa_in_filtered = any(s["symbol"] == "AA" for s in filtered)
        aa_in_final = any(s["symbol"] == "AA" for s in final_symbols)

        print("\nAA Status:")
        print(f"  In filtered list: {aa_in_filtered}")
        print(f"  In final list: {aa_in_final}")

        if aa_in_filtered:
            aa_pos = next((i for i, s in enumerate(sorted_symbols) if s["symbol"] == "AA"), None)
            if aa_pos is not None:
                print(f"  Position in sorted list: #{aa_pos + 1}")

    finally:
        await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(trace_layer1_issue())

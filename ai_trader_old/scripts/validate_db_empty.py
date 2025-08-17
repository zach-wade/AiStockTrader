#!/usr/bin/env python3
"""
Validate that the database is empty before starting TSLA backfill.
"""
# Standard library imports
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Standard library imports
from datetime import datetime
import json

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory


async def check_database_empty():
    """Check if all relevant tables are empty."""

    # Get config and database
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    results = {"timestamp": datetime.now().isoformat(), "is_empty": True, "tables": {}}

    # Tables to check
    tables = [
        "companies",
        "market_data_1m",
        "market_data_5m",
        "market_data_15m",
        "market_data_30m",
        "market_data_1h",
        "news_data",
        "financials_data",
        "corporate_actions",
        "features_technical",
        "features_sentiment",
        "scanner_alerts",
        "backtest_trades",
    ]

    try:
        print("=" * 70)
        print("DATABASE EMPTY VALIDATION")
        print("=" * 70)
        print(f"Timestamp: {results['timestamp']}")
        print("\nChecking tables...")
        print("-" * 50)

        for table in tables:
            try:
                # Get count from table
                query = f"SELECT COUNT(*) as count FROM {table}"
                result = await db_adapter.fetch_one(query)
                count = result["count"] if result else 0

                results["tables"][table] = count
                status = "✓ EMPTY" if count == 0 else f"✗ {count:,} records"
                print(f"{table:.<30} {status}")

                if count > 0:
                    results["is_empty"] = False

            except Exception as e:
                print(f"{table:.<30} ERROR: {e!s}")
                results["tables"][table] = f"ERROR: {e!s}"

        # Check partitions
        print("\nChecking partitions...")
        print("-" * 50)

        partition_query = """
            SELECT
                parent.relname AS parent_table,
                COUNT(*) as partition_count
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            WHERE parent.relname LIKE 'market_data_%'
            GROUP BY parent.relname
            ORDER BY parent.relname;
        """

        partition_results = await db_adapter.fetch_all(partition_query)
        for row in partition_results:
            print(f"{row['parent_table']} partitions: {row['partition_count']}")

        print("\n" + "=" * 70)
        print(f"DATABASE IS {'EMPTY' if results['is_empty'] else 'NOT EMPTY'}")
        print("=" * 70)

        # Save results
        output_file = Path("data/validation/db_empty_check.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        return results["is_empty"]

    finally:
        await db_adapter.close()


if __name__ == "__main__":
    is_empty = asyncio.run(check_database_empty())
    sys.exit(0 if is_empty else 1)

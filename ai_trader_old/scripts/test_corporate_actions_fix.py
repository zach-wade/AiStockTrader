#!/usr/bin/env python3
"""
Test script to verify corporate actions backfill fix.
"""
# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from main.config import get_config_manager
from main.data_pipeline.ingestion.polygon_corporate_actions_client import (
    PolygonCorporateActionsClient,
)
from main.data_pipeline.storage.archive import DataArchive

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_corporate_actions():
    """Test corporate actions fetch for TSLA with async archive."""

    print("=" * 80)
    print("TESTING CORPORATE ACTIONS ASYNC FIX")
    print("=" * 80)

    # Get config
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Create archive
    archive_config = config.data_pipeline.storage.archive
    archive = DataArchive(archive_config)

    # Create client - pass the polygon api credentials config
    # Standard library imports
    import os

    polygon_config = {"api_keys": {"polygon": {"key": os.getenv("POLYGON_API_KEY")}}}
    client = PolygonCorporateActionsClient(polygon_config)

    # Test with TSLA for 1 year (faster test)
    symbols = ["TSLA"]
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=365)  # 1 year for quick test

    print("\nTesting corporate actions fetch for TSLA")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("-" * 80)

    start_time = datetime.now(UTC)

    try:
        # Call fetch_and_archive directly
        result = await client.fetch_and_archive(
            data_type="corporate_actions",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            archive=archive,
        )

        duration = (datetime.now(UTC) - start_time).total_seconds()

        print(f"\n✅ SUCCESS! Corporate actions fetch completed in {duration:.2f} seconds")
        print(f"Result: {result}")

        # Check if files were created
        data_lake_path = Path("data_lake/raw/corporate_actions")
        if data_lake_path.exists():
            files = list(data_lake_path.rglob("*TSLA*.parquet"))
            print(f"\nData lake files created: {len(files)}")
            for file in files[:5]:  # Show first 5
                print(f"  - {file.relative_to('data_lake/raw')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        # Standard library imports
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Close the client session properly
        if hasattr(client, "_session") and client._session:
            await client._session.close()

    return True


async def main():
    """Run the test."""
    success = await test_corporate_actions()

    if success:
        print("\n✅ Corporate actions async fix is working correctly!")
    else:
        print("\n❌ Corporate actions async fix test failed!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

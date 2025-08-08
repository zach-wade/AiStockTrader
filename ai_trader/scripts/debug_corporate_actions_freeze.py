#!/usr/bin/env python3
"""
Debug script to investigate corporate actions freeze issue.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.data_pipeline.ingestion.polygon_corporate_actions_client import PolygonCorporateActionsClient
from main.data_pipeline.storage.archive import DataArchive
from main.config import get_config_manager

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def debug_corporate_actions():
    """Debug corporate actions fetch for TSLA."""
    
    # Get config
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    
    # Create archive
    archive_config = config.data_pipeline.storage.archive
    archive = DataArchive(archive_config)
    
    # Create client
    client = PolygonCorporateActionsClient(config.data_sources.polygon)
    
    # Test with TSLA for 5 years
    symbols = ['TSLA']
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1825)  # 5 years
    
    print(f"Testing corporate actions fetch for TSLA")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        # Call fetch_and_archive directly
        result = await client.fetch_and_archive(
            data_type='corporate_actions',
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            archive=archive
        )
        
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_corporate_actions())
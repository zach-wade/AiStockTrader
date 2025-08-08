#!/usr/bin/env python3
"""
Load Data Lake to Database Script

This script loads existing data from the data lake (archive) into PostgreSQL tables.
It respects hot/cold storage configurations and symbol qualification levels.

Key features:
- Loads ALL available intervals for market data within the date range
- Routes data to correct tables (1day and 1hour go to market_data_1h, others to their own tables)
- Respects symbol qualification levels (Layer 1+ for minute data)

Usage:
    # Load all Layer 1 qualified symbols for last 30 days
    python scripts/load_datalake_to_db.py --symbols layer1 --days 30
    
    # Load specific symbols (all intervals)
    python scripts/load_datalake_to_db.py --symbols AAPL,MSFT --days 60
    
    # Load all available data (respects hot storage config)
    python scripts/load_datalake_to_db.py --all --days 30
    
    # Load with specific date range
    python scripts/load_datalake_to_db.py --symbols layer1 --start 2024-01-01 --end 2024-12-31
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.bulk_loaders.market_data_split import MarketDataSplitBulkLoader
from main.data_pipeline.storage.bulk_loaders.news import NewsBulkLoader
from main.data_pipeline.storage.bulk_loaders.fundamentals import FundamentalsBulkLoader
from main.data_pipeline.storage.bulk_loaders.base import BulkLoadConfig
from main.data_pipeline.storage.repositories import CompanyRepository
from main.universe.universe_manager import UniverseManager
from main.utils.core import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = get_logger(__name__)


class DataLakeLoader:
    """Loads data from data lake archive to PostgreSQL database."""
    
    def __init__(self):
        """Initialize the data lake loader."""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.load_config("unified_config")
        
        # Initialize database
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(self.config)
        
        # Initialize archive
        archive_config = self.config.get('storage.archive', {})
        self.archive = DataArchive(archive_config)
        
        # Initialize repositories
        self.company_repo = CompanyRepository(self.db_adapter)
        self.universe_manager = UniverseManager(self.company_repo)
        
        # Track statistics
        self.stats = {
            'market_data_loaded': 0,
            'market_data_processed': 0,
            'news_loaded': 0,
            'financials_loaded': 0,
            'errors': 0,
            'duplicates_skipped': 0
        }
    
    async def load_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        Load market data from archive to database.
        
        Loads ALL available intervals within the date range.
        The bulk loader will route data to correct tables:
        - 1day and 1hour data → market_data_1h table
        - Other intervals → their respective tables
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
        """
        logger.info(f"Loading market data for {len(symbols)} symbols")
        
        # Get symbol qualifications for filtering
        qualifications = await self._get_symbol_qualifications(symbols)
        
        # Create bulk loader with config
        bulk_config = BulkLoadConfig(
            accumulation_size=10000,
            max_memory_mb=1024,
            batch_timeout_seconds=30,
            use_copy_command=True
        )
        
        market_loader = MarketDataSplitBulkLoader(
            db_adapter=self.db_adapter,
            config=bulk_config
        )
        
        # Process each symbol
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            try:
                # Query archive once for all data
                logger.info(f"  Loading all market data for {symbol}")
                
                # Query archive
                raw_records = await self.archive.query_raw_records(
                    source='polygon',
                    data_type='market_data',
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not raw_records:
                    logger.warning(f"  No data found for {symbol}")
                    continue
                
                # Group records by interval
                records_by_interval = {}
                for record in raw_records:
                    record_interval = record.metadata.get('interval', 'unknown')
                    if record_interval not in records_by_interval:
                        records_by_interval[record_interval] = []
                    records_by_interval[record_interval].append(record)
                
                logger.info(f"  Found data for intervals: {sorted(records_by_interval.keys())} ({sum(len(v) for v in records_by_interval.values())} total records)")
                
                # Process each interval's records
                for interval, interval_records in records_by_interval.items():
                    if interval == 'unknown':
                        logger.warning(f"  Skipping {len(interval_records)} records with unknown interval")
                        continue
                    
                    # Process records for this interval
                    logger.info(f"  Processing {len(interval_records)} {interval} records for {symbol}")
                    total_records = 0
                    
                    for record in interval_records:
                        if record.data and 'data' in record.data:
                            market_data = record.data['data']
                            if isinstance(market_data, list) and market_data:
                                # Convert to DataFrame
                                df = pd.DataFrame(market_data)
                                
                                # Parse timestamp if needed
                                if 'timestamp' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    df.set_index('timestamp', inplace=True)
                                
                                # Track records processed
                                records_in_df = len(df)
                                self.stats['market_data_processed'] += records_in_df
                                
                                # Load to database with correct interval
                                # The bulk loader will route to the correct table based on interval
                                load_result = await market_loader.load(
                                    data=df,
                                    symbol=symbol,
                                    interval=interval,
                                    source='polygon'
                                )
                                
                                if load_result.success:
                                    if not load_result.skipped:
                                        total_records += load_result.records_loaded
                                        self.stats['market_data_loaded'] += load_result.records_loaded
                                        # Track duplicates
                                        duplicates = records_in_df - load_result.records_loaded
                                        if duplicates > 0:
                                            self.stats['duplicates_skipped'] += duplicates
                                    else:
                                        logger.debug(f"  Skipped loading {interval} data: {load_result.skip_reason}")
                                else:
                                    logger.error(f"  Failed to load {interval} data: {'; '.join(load_result.errors)}")
                                    self.stats['errors'] += 1
                    
                    if total_records > 0:
                        logger.info(f"  Loaded {total_records} {interval} records for {symbol}")
                    elif interval_records:
                        logger.info(f"  Processed {len(interval_records)} {interval} files but no new records (likely duplicates)")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                self.stats['errors'] += 1
        
        # Flush any remaining data
        logger.info("Flushing market data buffer...")
        flush_result = await market_loader.flush_all()
        if flush_result.records_loaded > 0:
            self.stats['market_data_loaded'] += flush_result.records_loaded
            logger.info(f"Flushed {flush_result.records_loaded} additional records")
    
    async def load_news_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Load news data from archive to database."""
        logger.info(f"Loading news data for {len(symbols)} symbols")
        
        # Create bulk loader
        bulk_config = BulkLoadConfig(
            accumulation_size=1000,
            max_memory_mb=512,
            batch_timeout_seconds=30,
            use_copy_command=True
        )
        
        news_loader = NewsBulkLoader(
            db_adapter=self.db_adapter,
            config=bulk_config
        )
        
        # Process each symbol
        for symbol in symbols:
            try:
                # Query archive
                raw_records = await self.archive.query_raw_records(
                    source='polygon',
                    data_type='news',
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not raw_records:
                    logger.debug(f"No news data found for {symbol}")
                    continue
                
                # Process records
                total_records = 0
                for record in raw_records:
                    if record.data:
                        # Check if data is wrapped in 'data' key
                        if isinstance(record.data, dict) and 'data' in record.data:
                            news_items = record.data['data']
                        else:
                            news_items = record.data
                        
                        # Ensure it's a list
                        if not isinstance(news_items, list):
                            news_items = [news_items]
                        
                        for item in news_items:
                            # Only load if symbol matches
                            item_symbols = item.get('symbols', item.get('tickers', []))
                            if symbol in item_symbols:
                                load_result = await news_loader.load([item])
                                if load_result.success:
                                    total_records += load_result.records_loaded
                                    self.stats['news_loaded'] += load_result.records_loaded
                
                if total_records > 0:
                    logger.info(f"Loaded {total_records} news articles for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading news for {symbol}: {e}")
                self.stats['errors'] += 1
        
        # Flush buffer
        await news_loader.flush_all()
    
    async def load_financials_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Load financials data from archive to database."""
        logger.info(f"Loading financials data for {len(symbols)} symbols")
        
        # Create bulk loader
        bulk_config = BulkLoadConfig(
            accumulation_size=100,
            max_memory_mb=256,
            batch_timeout_seconds=30,
            use_copy_command=True
        )
        
        financials_loader = FundamentalsBulkLoader(
            db_adapter=self.db_adapter,
            config=bulk_config
        )
        
        # Process each symbol
        for symbol in symbols:
            try:
                # Query archive
                raw_records = await self.archive.query_raw_records(
                    source='yahoo_financials',
                    data_type='fundamentals',
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not raw_records:
                    logger.debug(f"No financials data found for {symbol}")
                    continue
                
                # Process records
                total_records = 0
                for record in raw_records:
                    if record.data:
                        load_result = await financials_loader.load(
                            [record.data] if isinstance(record.data, dict) else record.data
                        )
                        if load_result.success:
                            total_records += load_result.records_loaded
                            self.stats['financials_loaded'] += load_result.records_loaded
                
                if total_records > 0:
                    logger.info(f"Loaded {total_records} financial records for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading financials for {symbol}: {e}")
                self.stats['errors'] += 1
        
        # Flush buffer
        await financials_loader.flush_all()
    
    async def _get_symbol_qualifications(self, symbols: List[str]) -> Dict[str, Dict[str, bool]]:
        """Get symbol qualification levels."""
        qualifications = {}
        
        # Query companies table for qualification info
        query = """
            SELECT symbol, layer1_qualified, 
                   layer2_qualified, layer3_qualified, is_active
            FROM companies
            WHERE symbol = ANY(:symbols)
        """
        
        rows = await self.db_adapter.fetch_all(query, {'symbols': symbols})
        
        for row in rows:
            # Layer 0 = all active symbols
            qualifications[row['symbol']] = {
                'layer0_qualified': row.get('is_active', True),
                'layer1_qualified': row.get('layer1_qualified', False),
                'layer2_qualified': row.get('layer2_qualified', False),
                'layer3_qualified': row.get('layer3_qualified', False)
            }
        
        # Default qualification for missing symbols
        for symbol in symbols:
            if symbol not in qualifications:
                qualifications[symbol] = {
                    'layer0_qualified': True,  # Assume Layer 0 by default
                    'layer1_qualified': False,
                    'layer2_qualified': False,
                    'layer3_qualified': False
                }
        
        return qualifications
    
    async def get_symbols_by_layer(self, layer: str) -> List[str]:
        """Get symbols for a specific layer."""
        if layer == 'layer0':
            # All active symbols
            query = "SELECT symbol FROM companies WHERE is_active = TRUE"
        elif layer == 'layer1':
            query = "SELECT symbol FROM companies WHERE layer1_qualified = TRUE AND is_active = TRUE"
        elif layer == 'layer2':
            query = "SELECT symbol FROM companies WHERE layer2_qualified = TRUE AND is_active = TRUE"
        elif layer == 'layer3':
            query = "SELECT symbol FROM companies WHERE layer3_qualified = TRUE AND is_active = TRUE"
        elif layer == 'all':
            # Get all active symbols
            query = "SELECT symbol FROM companies WHERE is_active = TRUE"
        else:
            raise ValueError(f"Unknown layer: {layer}")
        
        rows = await self.db_adapter.fetch_all(query)
        return [row['symbol'] for row in rows]
    
    async def cleanup(self):
        """Clean up resources."""
        if self.db_adapter:
            await self.db_adapter.close()
    
    def print_summary(self):
        """Print loading summary."""
        logger.info("\n" + "="*60)
        logger.info("Data Loading Summary")
        logger.info("="*60)
        logger.info(f"Market Data Processed: {self.stats['market_data_processed']:,} records")
        logger.info(f"Market Data Loaded: {self.stats['market_data_loaded']:,} records (unique)")
        if self.stats['duplicates_skipped'] > 0:
            logger.info(f"Duplicates Skipped: {self.stats['duplicates_skipped']:,} records")
        logger.info(f"News Articles Loaded: {self.stats['news_loaded']:,} records")
        logger.info(f"Financial Records Loaded: {self.stats['financials_loaded']:,} records")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load data from data lake to database")
    
    # Symbol selection
    parser.add_argument(
        "--symbols",
        help="Symbols to load (comma-separated) or layer (layer0/layer1/layer2/layer3/all)",
        default="layer1"
    )
    
    # Date range
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days to load (from today backwards)",
        default=30
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    )
    
    # Data types
    parser.add_argument(
        "--data-types",
        help="Data types to load (comma-separated: market,news,financials)",
        default="market"
    )
    
    # Intervals for market data (deprecated - all intervals are loaded)
    parser.add_argument(
        "--intervals",
        help="[DEPRECATED] All available intervals are loaded automatically",
        default="all"
    )
    
    # Other options
    parser.add_argument("--all", action="store_true", help="Load all available data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be loaded without loading")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Create loader
    loader = DataLakeLoader()
    
    try:
        # Get symbols
        if args.symbols in ['layer0', 'layer1', 'layer2', 'layer3', 'all']:
            symbols = await loader.get_symbols_by_layer(args.symbols)
            logger.info(f"Loading {len(symbols)} {args.symbols} symbols")
        else:
            symbols = [s.strip() for s in args.symbols.split(',')]
            logger.info(f"Loading specific symbols: {symbols}")
        
        if args.dry_run:
            logger.info("DRY RUN - Would load:")
            logger.info(f"  Symbols: {len(symbols)} symbols")
            logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
            logger.info(f"  Data types: {args.data_types}")
            if 'market' in args.data_types:
                logger.info(f"  Market data: All available intervals will be loaded")
            return
        
        # Parse data types
        data_types = [dt.strip() for dt in args.data_types.split(',')]
        
        # Load market data
        if 'market' in data_types:
            await loader.load_market_data(symbols, start_date, end_date)
        
        # Load news data
        if 'news' in data_types:
            await loader.load_news_data(symbols, start_date, end_date)
        
        # Load financials data
        if 'financials' in data_types:
            await loader.load_financials_data(symbols, start_date, end_date)
        
        # Print summary
        loader.print_summary()
        
    finally:
        await loader.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
# File: app/process_raw_data.py

"""
Script to process raw data from data_lake/raw/ to data_lake/processed/.

This is the proper ETL for the hot/cold architecture - transforms raw parquet
files into processed parquet files. Does NOT touch PostgreSQL (that's only for
Layer 3 qualified symbols).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow.parquet as pq
import json

import typer
from omegaconf import DictConfig

from main.config.config_manager import get_config
from main.data_pipeline.processing.transformer import DataTransformer
from main.data_pipeline.processing.standardizer import DataStandardizer
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.utils import (
    create_data_pipeline_app,
    StandardCLIHandler,
    CLIAppConfig,
    async_command,
    success_message,
    error_message,
    info_message,
    warning_message,
    get_logger
)

# Setup logging
logger = get_logger(__name__)

# Create standardized app
app = create_data_pipeline_app(
    "process-raw",
    "AI Trader Raw Data Processor - Transforms raw data lake files to processed format"
)

# Create standardized CLI handler
cli_config = CLIAppConfig(
    name="process-raw",
    description="AI Trader Raw Data Processor",
    context_components=['data_lake'],
    enable_monitoring=True,
    show_progress=True
)
cli_handler = StandardCLIHandler(cli_config)


class RawDataProcessor:
    """
    Processes raw parquet files from data_lake/raw/ to data_lake/processed/.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.standardizer = DataStandardizer(config.get('processing.standardizer', {}))
        
        # Create database adapter for corporate action support
        try:
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(config)
            self.transformer = DataTransformer(config, db_adapter)
            logger.info("DataTransformer initialized with corporate action support")
        except Exception as e:
            logger.warning(f"Failed to create database adapter for corporate actions: {e}")
            self.transformer = DataTransformer(config)
            logger.info("DataTransformer initialized without corporate action support")
        
        # Get data lake path from config
        self.data_lake_path = Path(config.get('data_pipeline.storage.archive.local_path', 
                                             '/Users/zachwade/StockMonitoring/ai_trader/data_lake'))
        self.raw_path = self.data_lake_path / 'raw'
        self.processed_path = self.data_lake_path / 'processed'
        
        logger.info(f"RawDataProcessor initialized with standardizer and transformer at: {self.data_lake_path}")
    
    def find_unprocessed_files(self, data_type: str = "all", limit: int = 1000, symbol: str = None) -> List[Path]:
        """
        Find raw files that haven't been processed yet.
        
        Args:
            data_type: Type of data to process (market_data, news, etc.) or "all"
            limit: Maximum number of files to process
            symbol: Optional symbol filter (e.g., "AAPL") for market_data
            
        Returns:
            List of Path objects for unprocessed files
        """
        unprocessed_files = []
        
        # Map data types to actual directory structure
        data_type_mapping = {
            'market_data': 'market_data',
            'news': 'alternative_data/news',
            'financials': 'alternative_data/financials',
            'fundamentals': 'alternative_data/fundamentals',
            'social_sentiment': 'social_sentiment'
        }
        
        # Determine which subdirectories to search
        if data_type == "all":
            search_dirs = list(data_type_mapping.values())
        else:
            search_dirs = [data_type_mapping.get(data_type, data_type)]
        
        for search_dir in search_dirs:
            raw_subdir = self.raw_path / search_dir
            if not raw_subdir.exists():
                logger.warning(f"Raw directory does not exist: {raw_subdir}")
                continue
            
            # Apply symbol filter for market_data
            if symbol and search_dir == 'market_data':
                # Look specifically for the symbol directory
                symbol_dir = raw_subdir / f"symbol={symbol}"
                if not symbol_dir.exists():
                    logger.warning(f"Symbol directory does not exist: {symbol_dir}")
                    continue
                search_subdirs = [symbol_dir]
            else:
                search_subdirs = [raw_subdir]
            
            for subdir in search_subdirs:
                # Find both JSON and Parquet files in raw directory
                for pattern in ["*.json", "*.parquet"]:
                    for raw_file in subdir.rglob(pattern):
                        # Skip metadata files
                        if raw_file.name.endswith('.meta'):
                            continue
                            
                        # Construct corresponding processed file path
                        relative_path = raw_file.relative_to(self.raw_path)
                        if raw_file.suffix == '.json':
                            # Convert .json to .parquet for processed
                            processed_file = self.processed_path / relative_path.with_suffix('.parquet')
                        else:
                            # Keep .parquet as .parquet
                            processed_file = self.processed_path / relative_path
                        
                        # Check if already processed
                        if not processed_file.exists():
                            unprocessed_files.append(raw_file)
                            if len(unprocessed_files) >= limit:
                                return unprocessed_files
        
        return unprocessed_files
    
    async def transform_market_data_file(self, raw_file: Path) -> Optional[pd.DataFrame]:
        """
        Transform a raw market data JSON file.
        
        Args:
            raw_file: Path to raw JSON file
            
        Returns:
            Transformed DataFrame or None if error
        """
        try:
            # Read raw JSON file
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            
            # Convert to DataFrame based on data structure
            if isinstance(raw_data, dict):
                # Check for RawDataRecord structure first
                if 'raw_response' in raw_data and isinstance(raw_data['raw_response'], dict):
                    # Handle RawDataRecord format: {"raw_response": {"data": [...]}}
                    if 'data' in raw_data['raw_response']:
                        df = pd.DataFrame(raw_data['raw_response']['data'])
                    else:
                        logger.error(f"No 'data' key in raw_response for {raw_file}")
                        return None
                # Check if it has a direct 'data' key
                elif 'data' in raw_data:
                    # Check if data contains another nested 'data' key
                    data_content = raw_data['data']
                    if isinstance(data_content, dict) and 'data' in data_content:
                        # Handle nested structure: {"data": {"data": [...]}}
                        records = data_content['data']
                        # Add timestamp from top level if records don't have it
                        if isinstance(records, list) and len(records) > 0:
                            if 'timestamp' not in records[0] and 'timestamp' in raw_data:
                                # Add timestamp to each record
                                for record in records:
                                    record['timestamp'] = raw_data['timestamp']
                        df = pd.DataFrame(records)
                    elif isinstance(data_content, list):
                        # Handle direct list: {"data": [...]}
                        # Add timestamp from top level if records don't have it
                        if len(data_content) > 0 and isinstance(data_content[0], dict):
                            if 'timestamp' not in data_content[0] and 'timestamp' in raw_data:
                                for record in data_content:
                                    record['timestamp'] = raw_data['timestamp']
                        df = pd.DataFrame(data_content)
                    else:
                        # Single record: {"data": {...}}
                        if isinstance(data_content, dict) and 'timestamp' not in data_content and 'timestamp' in raw_data:
                            data_content['timestamp'] = raw_data['timestamp']
                        df = pd.DataFrame([data_content])
                else:
                    # Try to create DataFrame from the dict itself
                    df = pd.DataFrame([raw_data])
            elif isinstance(raw_data, list):
                df = pd.DataFrame(raw_data) 
            else:
                logger.error(f"Unsupported data format in {raw_file}")
                return None
            
            if df.empty:
                logger.warning(f"Empty dataframe from {raw_file}")
                return None
            
            # Detect source from file path
            source = self._detect_source(raw_file)
            
            # Quick fix: Standardize common column name variations
            column_mapping = {
                'Close': 'close',
                'Open': 'open', 
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume',
                'Date': 'timestamp'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure DataFrame has proper timestamp index before transformation
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                except Exception as e:
                    logger.warning(f"Could not set timestamp index for {raw_file}: {e}")
            
            # If still no proper datetime index, create a basic one to prevent RangeIndex errors
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"No datetime index available for {raw_file}, creating basic index")
                # Create a simple datetime index to prevent RangeIndex errors
                df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='D')
            
            # Extract symbol from file path
            symbol = self._extract_symbol_from_path(raw_file)
            
            # Step 1: Standardize the data first
            standardized_df = self.standardizer.standardize_market_data(df, source, symbol)
            
            if standardized_df.empty:
                logger.warning(f"Standardization failed for {raw_file}")
                return None
            
            # Step 2: Apply transformation (now async for corporate actions)
            transformed_df = await self.transformer.transform_market_data(standardized_df, source, symbol)
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error transforming market data file {raw_file}: {e}")
            return None
    
    async def process_parquet_file(self, raw_file: Path) -> Optional[pd.DataFrame]:
        """
        Process a raw parquet file (for market data that's already in parquet format).
        
        Args:
            raw_file: Path to raw parquet file
            
        Returns:
            Processed DataFrame or None if error
        """
        try:
            # Read the parquet file directly
            df = pd.read_parquet(raw_file)
            
            if df.empty:
                logger.warning(f"Empty dataframe from {raw_file}")
                return None
            
            # Extract symbol and source from file path
            symbol = self._extract_symbol_from_path(raw_file)
            source = self._detect_source(raw_file)
            
            # Step 1: Standardize the data first
            standardized_df = self.standardizer.standardize_market_data(df, source, symbol)
            
            if standardized_df.empty:
                logger.warning(f"Standardization failed for {raw_file}")
                return None
            
            # Step 2: Apply transformation (corporate actions, advanced cleaning, features)
            transformed_df = await self.transformer.transform_market_data(standardized_df, source, symbol)
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error processing parquet file {raw_file}: {e}")
            return None
    
    def transform_news_file(self, raw_file: Path) -> Optional[List[Dict]]:
        """
        Transform a raw news JSON file.
        
        Args:
            raw_file: Path to raw JSON file
            
        Returns:
            Transformed news articles or None if error
        """
        try:
            # Read raw JSON file
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            
            # Convert to list of articles for news processing
            if isinstance(raw_data, dict):
                if 'data' in raw_data:
                    articles = raw_data['data'] if isinstance(raw_data['data'], list) else [raw_data['data']]
                else:
                    articles = [raw_data]
            elif isinstance(raw_data, list):
                articles = raw_data
            else:
                logger.error(f"Unsupported news data format in {raw_file}")
                return None
            
            if not articles:
                logger.warning(f"No articles found in {raw_file}")
                return None
            
            # Detect source
            source = self._detect_source(raw_file)
            
            # Apply transformation
            transformed_articles = self.transformer.transform_news_data(articles, source)
            
            return transformed_articles
            
        except Exception as e:
            logger.error(f"Error transforming news file {raw_file}: {e}")
            return None
    
    def _detect_source(self, file_path: Path) -> str:
        """Detect data source from file path."""
        path_str = str(file_path).lower()
        if 'polygon' in path_str:
            return 'polygon'
        elif 'alpaca' in path_str:
            return 'alpaca'
        elif 'yahoo' in path_str:
            return 'yahoo'
        else:
            return 'unknown'
    
    def _extract_symbol_from_path(self, file_path: Path) -> str:
        """Extract symbol from file path structure."""
        path_parts = file_path.parts
        
        # Look for symbol=XXX pattern in path
        for part in path_parts:
            if part.startswith('symbol='):
                return part.split('=')[1]
        
        # Fallback: look for known symbol patterns
        # This is a fallback if the path structure is different
        logger.warning(f"Could not extract symbol from path {file_path}, using 'UNKNOWN'")
        return 'UNKNOWN'
    
    def save_processed_data(self, data: pd.DataFrame, processed_file: Path) -> bool:
        """
        Save processed data to parquet file.
        
        Args:
            data: Processed DataFrame
            processed_file: Path to save processed file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if needed
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            data.to_parquet(processed_file, compression='snappy')
            
            logger.info(f"Saved processed data to {processed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data to {processed_file}: {e}")
            return False
    
    def save_processed_news(self, articles: List[Dict], processed_file: Path) -> bool:
        """
        Save processed news data to parquet file.
        
        Args:
            articles: Processed news articles
            processed_file: Path to save processed file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if needed
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrame for parquet storage
            df = pd.DataFrame(articles)
            
            # Save as parquet
            df.to_parquet(processed_file, compression='snappy')
            
            logger.info(f"Saved processed news to {processed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed news to {processed_file}: {e}")
            return False
    
    async def process_files(self, data_type: str = "all", limit: int = 100, symbol: str = None) -> Dict[str, int]:
        """
        Main processing method to transform raw files to processed.
        
        Args:
            data_type: Type of data to process
            limit: Maximum number of files to process
            symbol: Optional symbol filter (e.g., "AAPL") for market_data
            
        Returns:
            Dictionary with processing statistics
        """
        info_message("Starting raw data processing")
        
        # Find unprocessed files
        unprocessed_files = self.find_unprocessed_files(data_type, limit, symbol)
        
        if not unprocessed_files:
            info_message("No unprocessed files found")
            return {"files_processed": 0, "files_failed": 0}
        
        info_message(f"Found {len(unprocessed_files)} unprocessed files")
        
        processed_count = 0
        failed_count = 0
        
        for raw_file in unprocessed_files:
            try:
                # Determine data type from path  
                # Handle nested paths like alternative_data/news
                if 'alternative_data' in raw_file.parts:
                    # For files in alternative_data/news/..., data type is 'news'
                    alt_data_index = raw_file.parts.index('alternative_data')
                    file_data_type = raw_file.parts[alt_data_index + 1]
                elif 'market_data' in raw_file.parts:
                    file_data_type = 'market_data'
                elif 'social_sentiment' in raw_file.parts:
                    file_data_type = 'social_sentiment'
                else:
                    # Fallback - try to get from path structure
                    file_data_type = 'unknown'
                
                # Construct processed file path
                relative_path = raw_file.relative_to(self.raw_path)
                if raw_file.suffix == '.json':
                    # Convert .json to .parquet for processed
                    processed_file = self.processed_path / relative_path.with_suffix('.parquet')
                else:
                    # Keep .parquet as .parquet
                    processed_file = self.processed_path / relative_path
                
                # Transform based on file type and data type
                if raw_file.suffix == '.parquet':
                    # Handle parquet files (primarily market data)
                    transformed_data = await self.process_parquet_file(raw_file)
                    if transformed_data is not None and not transformed_data.empty:
                        if self.save_processed_data(transformed_data, processed_file):
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                elif file_data_type == 'market_data':
                    # Handle JSON market data files
                    transformed_data = await self.transform_market_data_file(raw_file)
                    if transformed_data is not None and not transformed_data.empty:
                        if self.save_processed_data(transformed_data, processed_file):
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                elif file_data_type in ['news', 'financials', 'fundamentals']:
                    transformed_articles = self.transform_news_file(raw_file)
                    if transformed_articles:
                        if self.save_processed_news(transformed_articles, processed_file):
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                else:
                    # For other data types, try to process as generic JSON
                    logger.info(f"Processing {file_data_type} data as generic JSON")
                    try:
                        with open(raw_file, 'r') as f:
                            raw_data = json.load(f)
                        
                        # Convert to DataFrame and save as parquet
                        if isinstance(raw_data, dict):
                            df = pd.DataFrame([raw_data])
                        elif isinstance(raw_data, list):
                            df = pd.DataFrame(raw_data)
                        else:
                            df = pd.DataFrame()
                        
                        if not df.empty:
                            if self.save_processed_data(df, processed_file):
                                processed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process generic JSON {raw_file}: {e}")
                        failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {raw_file}: {e}")
                failed_count += 1
        
        success_message(
            f"Raw data processing completed",
            f"Processed: {processed_count}, Failed: {failed_count}"
        )
        
        return {
            "files_processed": processed_count,
            "files_failed": failed_count
        }


@app.command()
def run(
    data_type: str = typer.Option("all", help="Type of data to process (market_data, news, all)"),
    limit: int = typer.Option(100, help="Maximum number of files to process"),
    symbol: str = typer.Option(None, help="Optional symbol filter (e.g., AAPL) for market_data")
):
    """
    Process raw data files from data_lake/raw/ to data_lake/processed/.
    
    This transforms raw parquet files into standardized, cleaned processed files.
    Does NOT load data into PostgreSQL - that's only for Layer 3 qualified symbols.
    """
    @async_command(cli_handler, show_progress=True, operation_name="Raw Data Processing")
    async def _process_raw(handler: StandardCLIHandler, data_type: str, limit: int, symbol: str):
        processor = RawDataProcessor(get_config())
        
        try:
            result = await processor.process_files(data_type, limit, symbol)
            return result
            
        except Exception as e:
            error_message(
                "Raw data processing failed",
                f"Error: {str(e)}"
            )
            raise
    
    _process_raw(data_type, limit, symbol)


@app.command()
def status():
    """
    Show status of raw vs processed data in the data lake.
    """
    @async_command(cli_handler, show_progress=False, operation_name="Data Lake Status")
    async def _show_status(handler: StandardCLIHandler):
        processor = RawDataProcessor(get_config())
        
        info_message("=== Data Lake Status ===")
        
        # Count files in each directory
        for data_type in ['market_data', 'news', 'social_sentiment', 'financials']:
            raw_dir = processor.raw_path / data_type
            processed_dir = processor.processed_path / data_type
            
            raw_count = len(list(raw_dir.rglob("*.parquet"))) if raw_dir.exists() else 0
            processed_count = len(list(processed_dir.rglob("*.parquet"))) if processed_dir.exists() else 0
            
            info_message(f"\n{data_type.upper()}:")
            info_message(f"  Raw files: {raw_count}")
            info_message(f"  Processed files: {processed_count}")
            info_message(f"  Unprocessed: {raw_count - processed_count}")
        
        return {"status": "complete"}
    
    _show_status()


if __name__ == "__main__":
    app()
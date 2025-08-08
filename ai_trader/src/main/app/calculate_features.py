"""
Feature Calculation Engine

This module provides the command-line interface for calculating features.
It leverages existing components (UnifiedFeatureEngine, HistoricalManager, FeatureStore)
to avoid reinventing the wheel.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path
import os

from main.config.config_manager import get_config
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.feature_pipeline.feature_store_compat import FeatureStore
from main.utils.core import get_logger

logger = get_logger(__name__)


class FeatureCalculationEngine:
    """
    Engine for calculating and storing features via CLI.
    
    This class orchestrates:
    1. Fetching market data from the data lake
    2. Calculating features using UnifiedFeatureEngine
    3. Storing features in both PostgreSQL and HDF5
    """
    
    def __init__(self, config=None):
        """Initialize the feature calculation engine."""
        self.config = config or get_config()
        
        # Initialize components
        self.feature_engine = UnifiedFeatureEngine(self.config)
        self.feature_store = FeatureStore(config=self.config)
        
        logger.info("FeatureCalculationEngine initialized")
    
    async def run(self, feature_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run feature calculation based on provided configuration.
        
        Args:
            feature_config: Configuration dict with:
                - symbols: List of symbols to process
                - feature_sets: List of feature types to calculate
                - start_date: Start date string (YYYY-MM-DD)
                - end_date: End date string (YYYY-MM-DD)
                - output_file: Optional output file path
                
        Returns:
            Dict with calculation results and statistics
        """
        try:
            # Parse configuration
            symbols = feature_config.get('symbols', [])
            feature_sets = feature_config.get('feature_sets', ['technical', 'fundamental', 'sentiment'])
            start_date_str = feature_config.get('start_date')
            end_date_str = feature_config.get('end_date')
            output_file = feature_config.get('output_file')
            
            # Validate inputs
            if not symbols:
                raise ValueError("No symbols provided")
            
            # Parse dates with UTC timezone
            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                start_date = start_date.replace(tzinfo=timezone.utc)
            else:
                start_date = datetime.now(timezone.utc) - timedelta(days=365)
                
            if end_date_str:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date = end_date.replace(tzinfo=timezone.utc)
            else:
                end_date = datetime.now(timezone.utc)
            
            logger.info(f"Calculating features for {len(symbols)} symbols from {start_date} to {end_date}")
            logger.info(f"Feature sets: {feature_sets}")
            
            # No need to initialize historical manager anymore
            
            # Process each symbol
            results = {
                'symbols': [],
                'feature_sets': feature_sets,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'features_calculated': 0,
                'errors': []
            }
            
            for symbol in symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    
                    # Fetch market data
                    market_data = await self._fetch_market_data(
                        symbol, start_date, end_date
                    )
                    
                    if market_data.empty:
                        logger.warning(f"No market data found for {symbol}")
                        results['errors'].append(f"{symbol}: No market data")
                        continue
                    
                    # Calculate features
                    features_df = self._calculate_features(
                        market_data, symbol, feature_sets
                    )
                    
                    if features_df.empty:
                        logger.warning(f"No features calculated for {symbol}")
                        results['errors'].append(f"{symbol}: No features calculated")
                        continue
                    
                    # Store features
                    success = await self._store_features(
                        symbol, features_df, feature_sets
                    )
                    
                    if success:
                        results['symbols'].append(symbol)
                        results['features_calculated'] += len(features_df)
                        logger.info(f"Successfully calculated {len(features_df)} features for {symbol}")
                    else:
                        results['errors'].append(f"{symbol}: Failed to store features")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results['errors'].append(f"{symbol}: {str(e)}")
            
            # Save results to file if requested
            if output_file:
                await self._save_results_to_file(results, output_file)
            
            # Log summary
            logger.info(f"Feature calculation completed:")
            logger.info(f"  Symbols processed: {len(results['symbols'])}")
            logger.info(f"  Total features: {results['features_calculated']}")
            logger.info(f"  Errors: {len(results['errors'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            raise
    
    
    async def _fetch_market_data(self, symbol: str, 
                                start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Fetch market data from the data lake using direct file access."""
        try:
            # Ensure dates are timezone-aware (defensive programming)
            if start_date.tzinfo is None:
                logger.warning("start_date is timezone-naive, converting to UTC")
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                logger.warning("end_date is timezone-naive, converting to UTC")
                end_date = end_date.replace(tzinfo=timezone.utc)
            # Construct the path to market data
            data_lake_path = self.config.get('data_lake.path', 'data_lake')
            if not os.path.isabs(data_lake_path):
                # Make it absolute if relative
                data_lake_path = os.path.abspath(data_lake_path)
            
            market_data_path = Path(data_lake_path) / 'processed' / 'market_data' / f'symbol={symbol}' / 'interval=1day'
            
            if not market_data_path.exists():
                logger.warning(f"No data directory found for {symbol} at {market_data_path}")
                return pd.DataFrame()
            
            # Collect all parquet files
            all_data = []
            
            # Iterate through date directories
            for date_dir in sorted(market_data_path.iterdir()):
                if date_dir.is_dir() and date_dir.name.startswith('date='):
                    # Find all parquet files in this date directory
                    for parquet_file in date_dir.glob('*.parquet'):
                        try:
                            df = pd.read_parquet(parquet_file)
                            if not df.empty:
                                all_data.append(df)
                                logger.debug(f"Loaded {len(df)} rows from {parquet_file.name}")
                        except Exception as e:
                            logger.warning(f"Failed to read {parquet_file}: {e}")
            
            if not all_data:
                logger.warning(f"No parquet files found for {symbol}")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=False)
            logger.info(f"Combined {len(all_data)} files with {len(combined_data)} total rows for {symbol}")
            
            # Check if timestamp is in index or columns
            if combined_data.index.name == 'timestamp' or isinstance(combined_data.index, pd.DatetimeIndex):
                # Timestamp is in index
                logger.debug("Timestamp found in index")
                # Ensure index is datetime
                if not isinstance(combined_data.index, pd.DatetimeIndex):
                    combined_data.index = pd.to_datetime(combined_data.index)
                # Sort by index
                combined_data = combined_data.sort_index()
                # Filter by date range
                filtered_data = combined_data.loc[
                    (combined_data.index >= start_date) & 
                    (combined_data.index <= end_date)
                ].copy()
                logger.info(f"Filtered to {len(filtered_data)} rows within date range {start_date.date()} to {end_date.date()}")
            elif 'timestamp' in combined_data.columns:
                # Timestamp is a column
                logger.debug("Timestamp found in columns")
                combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
                # Set as index
                combined_data = combined_data.set_index('timestamp')
                # Sort by index
                combined_data = combined_data.sort_index()
                # Filter by date range
                filtered_data = combined_data.loc[
                    (combined_data.index >= start_date) & 
                    (combined_data.index <= end_date)
                ].copy()
                logger.info(f"Filtered to {len(filtered_data)} rows within date range {start_date.date()} to {end_date.date()}")
            else:
                logger.warning("No timestamp found in index or columns, returning all data")
                filtered_data = combined_data
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in filtered_data.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}. Available: {filtered_data.columns.tolist()}")
                return pd.DataFrame()
            
            # Return with DatetimeIndex preserved
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_features(self, market_data: pd.DataFrame, 
                          symbol: str,
                          feature_sets: List[str]) -> pd.DataFrame:
        """Calculate features using UnifiedFeatureEngine."""
        try:
            # Map feature set names to calculator names
            calculator_map = {
                'technical': 'technical',
                'fundamental': 'statistical',  # Use statistical as proxy
                'sentiment': 'sentiment',
                'microstructure': 'microstructure',
                'regime': 'regime',
                'cross_sectional': 'cross_sectional'
            }
            
            # Get calculators to use
            calculators = []
            for feature_set in feature_sets:
                if feature_set in calculator_map:
                    calculators.append(calculator_map[feature_set])
                elif feature_set == 'all':
                    calculators = list(calculator_map.values())
                    break
            
            # Calculate features
            features_df = self.feature_engine.calculate_features(
                data=market_data,
                symbol=symbol,
                calculators=calculators,
                use_cache=False
            )
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _store_features(self, symbol: str, 
                            features_df: pd.DataFrame,
                            feature_sets: List[str]) -> bool:
        """Store calculated features in both PostgreSQL and HDF5."""
        try:
            # Store each feature set separately
            for feature_set in feature_sets:
                if feature_set == 'all':
                    feature_type = 'all_features'
                else:
                    feature_type = f"{feature_set}_features"
                
                # Use feature store to save (handles both stores)
                success = self.feature_store.save_features(
                    symbol=symbol,
                    features_df=features_df,
                    feature_type=feature_type,
                    timestamp=datetime.now()
                )
                
                if not success:
                    logger.error(f"Failed to store {feature_type} for {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing features for {symbol}: {e}")
            return False
    
    async def _save_results_to_file(self, results: Dict[str, Any], 
                                  output_file: str):
        """Save calculation results to a file."""
        try:
            import json
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to file: {e}")
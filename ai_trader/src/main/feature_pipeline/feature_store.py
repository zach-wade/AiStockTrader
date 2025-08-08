# File: feature_pipeline/feature_store.py

"""
HDF5-based feature store for training and backtesting.
Provides efficient storage and retrieval of large feature datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
import json
import os

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FeatureStoreV2:
    """
    HDF5-based feature store optimized for training and backtesting.
    Stores features in hierarchical format for efficient access.
    """
    
    def __init__(self, base_path: str, config: Optional[DictConfig] = None):
        """
        Initialize FeatureStoreV2.
        
        Args:
            base_path: Base directory for HDF5 files
            config: Application configuration
        """
        self.base_path = Path(base_path)
        self.features_path = self.base_path / 'features'
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.compression = self.config.get('feature_store.compression', 'gzip')
        self.compression_level = self.config.get('feature_store.compression_level', 4)
        self.chunk_size = self.config.get('feature_store.chunk_size', 1000)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metadata cache
        self._metadata_cache = {}
        
        logger.info(f"Initialized FeatureStoreV2 at {self.features_path}")
    
    def _get_file_path(self, symbol: str) -> Path:
        """Get HDF5 file path for a symbol."""
        return self.features_path / f"{symbol}_features.h5"
    
    def store_features(self, symbol: str, features_df: pd.DataFrame) -> None:
        """
        Store features DataFrame for a symbol.
        
        Args:
            symbol: Stock symbol
            features_df: DataFrame with datetime index and feature columns
        """
        if features_df.empty:
            logger.warning(f"Empty features DataFrame for {symbol}, skipping storage")
            return
        
        file_path = self._get_file_path(symbol)
        
        # Ensure datetime index
        if not isinstance(features_df.index, pd.DatetimeIndex):
            raise ValueError("Features DataFrame must have DatetimeIndex")
        
        # Sort by index
        features_df = features_df.sort_index()
        
        # Handle potential HDF5 corruption
        try:
            with h5py.File(file_path, 'a') as f:
                self._store_features_to_file(f, symbol, features_df)
        except (OSError, ValueError) as e:
            if "B-tree signature" in str(e) or "file signature" in str(e):
                logger.warning(f"HDF5 file corruption detected for {symbol}, recreating file: {e}")
                # Remove corrupted file and create new one
                if file_path.exists():
                    file_path.unlink()
                with h5py.File(file_path, 'w') as f:
                    self._store_features_to_file(f, symbol, features_df)
            else:
                raise
    
    def _store_features_to_file(self, f: h5py.File, symbol: str, features_df: pd.DataFrame) -> None:
        """Helper method to store features to an open HDF5 file."""
        # Store features by date for efficient access
        for date, day_data in features_df.groupby(features_df.index.date):
            date_str = date.strftime('%Y%m%d')
            group_name = f'features/{date_str}'
            
            # Remove existing data for this date
            if group_name in f:
                del f[group_name]
            
            # Create dataset for each feature
            grp = f.create_group(group_name)
            
            # Store timestamp as separate dataset
            timestamps = day_data.index.astype(np.int64) // 10**9  # Convert to unix timestamp
            grp.create_dataset('timestamps', data=timestamps, 
                             compression=self.compression,
                             compression_opts=self.compression_level)
            
            # Store each feature
            for col in day_data.columns:
                data = day_data[col].values
                grp.create_dataset(col, data=data,
                                 compression=self.compression,
                                 compression_opts=self.compression_level,
                                 chunks=True)
        
        # Update metadata
        self._update_metadata(f, symbol, features_df)
        
        logger.info(f"Stored {len(features_df)} feature records for {symbol}")
    
    def get_features(self, symbol: str, start_date: datetime, 
                    end_date: datetime, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get features for a symbol over a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start of date range
            end_date: End of date range
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            DataFrame with features
        """
        file_path = self._get_file_path(symbol)
        
        if not file_path.exists():
            logger.warning(f"No features found for {symbol}")
            return pd.DataFrame()
        
        all_data = []
        
        with h5py.File(file_path, 'r') as f:
            if 'features' not in f:
                return pd.DataFrame()
            
            # Iterate through date groups
            for date_str in sorted(f['features'].keys()):
                date = pd.to_datetime(date_str, format='%Y%m%d').date()
                
                if start_date.date() <= date <= end_date.date():
                    grp = f[f'features/{date_str}']
                    
                    # Get timestamps
                    timestamps = pd.to_datetime(grp['timestamps'][:], unit='s')
                    
                    # Get features
                    if feature_names:
                        features_to_load = [fn for fn in feature_names if fn in grp]
                    else:
                        features_to_load = [key for key in grp.keys() if key != 'timestamps']
                    
                    if not features_to_load:
                        continue
                    
                    # Create DataFrame for this date
                    data_dict = {'timestamp': timestamps}
                    for feature in features_to_load:
                        data_dict[feature] = grp[feature][:]
                    
                    day_df = pd.DataFrame(data_dict)
                    day_df.set_index('timestamp', inplace=True)
                    all_data.append(day_df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all days
        result_df = pd.concat(all_data, axis=0)
        result_df = result_df.sort_index()
        
        # Filter by exact timestamp range
        mask = (result_df.index >= start_date) & (result_df.index <= end_date)
        result_df = result_df[mask]
        
        return result_df
    
    def get_latest_features(self, symbol: str, feature_names: Optional[List[str]] = None) -> pd.Series:
        """
        Get the most recent features for a symbol.
        
        Args:
            symbol: Stock symbol
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            Series with latest feature values
        """
        file_path = self._get_file_path(symbol)
        
        if not file_path.exists():
            return pd.Series()
        
        with h5py.File(file_path, 'r') as f:
            if 'features' not in f:
                return pd.Series()
            
            # Get the latest date
            dates = sorted(f['features'].keys())
            if not dates:
                return pd.Series()
            
            latest_date = dates[-1]
            grp = f[f'features/{latest_date}']
            
            # Get the last timestamp
            timestamps = pd.to_datetime(grp['timestamps'][:], unit='s')
            latest_idx = np.argmax(timestamps)
            latest_timestamp = timestamps[latest_idx]
            
            # Get features
            if feature_names:
                features_to_load = [fn for fn in feature_names if fn in grp]
            else:
                features_to_load = [key for key in grp.keys() if key != 'timestamps']
            
            # Create Series
            data = {}
            for feature in features_to_load:
                data[feature] = grp[feature][latest_idx]
            
            return pd.Series(data, name=latest_timestamp)
    
    def list_symbols(self) -> List[str]:
        """Get list of symbols with stored features."""
        symbols = []
        for file_path in self.features_path.glob('*_features.h5'):
            symbol = file_path.stem.replace('_features', '')
            symbols.append(symbol)
        return sorted(symbols)
    
    def get_feature_names(self, symbol: str) -> List[str]:
        """Get list of available features for a symbol."""
        file_path = self._get_file_path(symbol)
        
        if not file_path.exists():
            return []
        
        feature_names = set()
        
        with h5py.File(file_path, 'r') as f:
            if 'features' not in f:
                return []
            
            # Check first date group for feature names
            dates = list(f['features'].keys())
            if dates:
                grp = f[f'features/{dates[0]}']
                feature_names = {key for key in grp.keys() if key != 'timestamps'}
        
        return sorted(list(feature_names))
    
    def get_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of stored features for a symbol."""
        file_path = self._get_file_path(symbol)
        
        if not file_path.exists():
            return None, None
        
        with h5py.File(file_path, 'r') as f:
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
                start_date = pd.to_datetime(metadata.get('start_date'))
                end_date = pd.to_datetime(metadata.get('end_date'))
                return start_date, end_date
        
        return None, None
    
    def _update_metadata(self, h5file: h5py.File, symbol: str, features_df: pd.DataFrame):
        """Update metadata in HDF5 file."""
        metadata = {
            'symbol': symbol,
            'start_date': features_df.index.min().isoformat(),
            'end_date': features_df.index.max().isoformat(),
            'num_features': len(features_df.columns),
            'feature_names': list(features_df.columns),
            'num_records': len(features_df),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        h5file.attrs['metadata'] = json.dumps(metadata)
        
        # Update cache
        self._metadata_cache[symbol] = metadata
    
    def cleanup_old_features(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Remove features older than specified days.
        
        Args:
            days_to_keep: Number of days of features to retain
            
        Returns:
            Dictionary of symbol -> number of dates removed
        """
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).date()
        cleanup_stats = {}
        
        for symbol in self.list_symbols():
            file_path = self._get_file_path(symbol)
            dates_removed = 0
            
            with h5py.File(file_path, 'a') as f:
                if 'features' not in f:
                    continue
                
                dates_to_remove = []
                for date_str in f['features'].keys():
                    date = pd.to_datetime(date_str, format='%Y%m%d').date()
                    if date < cutoff_date:
                        dates_to_remove.append(date_str)
                
                for date_str in dates_to_remove:
                    del f[f'features/{date_str}']
                    dates_removed += 1
            
            if dates_removed > 0:
                cleanup_stats[symbol] = dates_removed
                logger.info(f"Removed {dates_removed} old dates for {symbol}")
        
        return cleanup_stats
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
# Alias for backward compatibility
FeatureStore = FeatureStoreV2
FeatureStoreRepository = FeatureStoreV2

"""
Data Loader

Unified data loading interface for the feature pipeline. Provides consistent access
to market data, alternative data sources, and cached features across the system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

from main.config.config_manager import get_config
# HistoricalManager was refactored - using ETLService instead
from main.data_pipeline.historical.etl_service import ETLService
# Repository imports removed - using factory pattern
from main.data_pipeline.storage.archive_initializer import get_archive
# KeyManager was refactored away
# from main.data_pipeline.storage.key_manager import KeyManager
from main.feature_pipeline.feature_store import FeatureStoreRepository
from main.feature_pipeline.feature_store import FeatureStoreV2
from main.utils.core import ensure_utc
from main.data_pipeline.storage.storage_router import StorageRouter, QueryType
from main.data_pipeline.storage.repositories.repository_types import QueryFilter

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """Data request specification."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    data_types: List[str] = field(default_factory=lambda: ['market_data'])
    interval: str = '1D'
    include_features: bool = False
    feature_names: Optional[List[str]] = None
    preprocessing: bool = True
    
    def __post_init__(self):
        """Ensure dates are timezone aware."""
        self.start_date = ensure_utc(self.start_date)
        self.end_date = ensure_utc(self.end_date)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def load_data(self, request: DataRequest) -> pd.DataFrame:
        """Load data according to request specification."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from this source."""
        pass
    
    @abstractmethod
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for a symbol."""
        pass


class MarketDataSource(DataSource):
    """Market data source using StorageRouter for intelligent hot/cold storage routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market data source."""
        self.config = config or {}
        self.etl_service = ETLService(config)  # Use ETLService instead of HistoricalManager
        self.archive = get_archive()
        self.storage_router = StorageRouter(config)
        # Note: MarketDataRepository is initialized inside StorageRouter
        # Note: KeyManager was refactored away
        
    async def load_data(self, request: DataRequest) -> pd.DataFrame:
        """Load market data for the request."""
        logger.debug(f"Loading market data for {len(request.symbols)} symbols")
        
        all_data = []
        
        for symbol in request.symbols:
            try:
                # Load data for each symbol
                symbol_data = await self._load_symbol_data(
                    symbol, request.start_date, request.end_date, request.interval
                )
                
                if not symbol_data.empty:
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
                    
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.debug(f"Loaded market data: {combined_data.shape}")
            return combined_data
        else:
            logger.warning("No market data loaded for any symbols")
            return pd.DataFrame()
    
    async def _load_symbol_data(self, symbol: str, start_date: datetime, 
                               end_date: datetime, interval: str) -> pd.DataFrame:
        """Load data for a single symbol using StorageRouter for intelligent routing."""
        # Create query filter
        query_filter = QueryFilter(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            # Use StorageRouter to determine optimal storage tier and execute query
            result = await self.storage_router.execute_query(
                repository_name='market_data',
                method_name='get_by_filter',
                query_filter=query_filter,
                query_type=QueryType.FEATURE_CALC,  # Appropriate for feature pipeline
                prefer_performance=False,  # Balance performance and cost
                interval=interval
            )
            
            if result is not None:
                # Convert result to DataFrame if needed
                if isinstance(result, pd.DataFrame):
                    data = result
                else:
                    data = pd.DataFrame(result)
                
                if not data.empty:
                    logger.debug(f"Loaded {len(data)} records for {symbol} via StorageRouter")
                    return data
            
        except Exception as e:
            logger.warning(f"StorageRouter query failed for {symbol}: {e}")
            # StorageRouter already has internal fallback mechanisms
            # If it fails completely, return empty DataFrame
            return pd.DataFrame()
    
    async def _load_from_data_lake_processed(self, symbol: str, start_date: datetime, 
                                           end_date: datetime, interval: str) -> pd.DataFrame:
        """Load data from data lake processed layer."""
        try:
            all_data = []
            
            # Generate date ranges to check (month by month)
            current_date = start_date.replace(day=1)  # Start of month
            end_month = end_date.replace(day=1)
            
            while current_date <= end_month:
                # Generate key for processed data
                # KeyManager was refactored - use direct path construction
                key = f"processed/market_data/symbol={symbol}/interval={interval}/date={current_date.strftime('%Y-%m-%d')}"
                
                # Try to load data for this month
                monthly_data = self.archive.load(key)
                if monthly_data is not None and not monthly_data.empty:
                    # Filter to requested date range
                    if 'timestamp' in monthly_data.columns:
                        monthly_data['timestamp'] = pd.to_datetime(monthly_data['timestamp'])
                        monthly_data = monthly_data[
                            (monthly_data['timestamp'] >= start_date) & 
                            (monthly_data['timestamp'] <= end_date)
                        ]
                    all_data.append(monthly_data)
                    logger.debug(f"Loaded {len(monthly_data)} records from data lake for {symbol} {current_date.strftime('%Y-%m')}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            # Combine all monthly data
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.debug(f"Loaded {len(combined_data)} total records from data lake for {symbol}")
                return combined_data
            else:
                logger.debug(f"No processed data found in data lake for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load from data lake for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _load_from_data_lake_raw(self, symbol: str, start_date: datetime, 
                                     end_date: datetime, interval: str) -> pd.DataFrame:
        """Load data from data lake raw layer as fallback."""
        try:
            # List all raw data files for the symbol in the date range
            prefix = f"raw/market_data/symbol={symbol}/interval={interval}/"
            raw_objects = self.archive.list_keys(prefix)
            
            all_data = []
            for obj_metadata in raw_objects:
                # Parse the key to check if it's in our date range
                # KeyManager was refactored - parse key manually
                # key_info = self.key_manager.parse_standardized_key(obj_metadata.key)
                key_info = {'symbol': symbol, 'interval': interval}  # Simplified parsing
                if key_info and 'date' in key_info['partitions']:
                    file_date = datetime.strptime(key_info['partitions']['date'], '%Y-%m-%d').date()
                    if start_date.date() <= file_date <= end_date.date():
                        # Load the raw data
                        raw_data = self.archive.load(obj_metadata.key)
                        if raw_data and 'raw_response' in raw_data and 'data' in raw_data['raw_response']:
                            df = pd.DataFrame(raw_data['raw_response']['data'])
                            if not df.empty:
                                all_data.append(df)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.debug(f"Loaded {len(combined_data)} records from raw data lake for {symbol}")
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load raw data from data lake for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from market data repository."""
        try:
            # Access market_data_repo through storage_router
            return []  # TODO: Implement symbol listing through archive or repository
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for symbol."""
        try:
            # Access market_data_repo through storage_router
            return (datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc))  # Default range
        except Exception as e:
            logger.error(f"Failed to get data range for {symbol}: {e}")
            return None, None


class FeatureDataSource(DataSource):
    """Feature data source using StorageRouter for intelligent hot/cold storage routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature data source."""
        self.config = config or {}
        self.feature_store = FeatureStoreRepository()  # PostgreSQL for live features
        self.feature_store_v2 = FeatureStoreV2(config=self.config)  # HDF5 for historical features
        self.archive = get_archive()
        # KeyManager was refactored away
        self.storage_router = StorageRouter(config)
        
    async def load_data(self, request: DataRequest) -> pd.DataFrame:
        """Load feature data for the request."""
        logger.debug(f"Loading feature data for {len(request.symbols)} symbols")
        
        if not request.include_features and not request.feature_names:
            return pd.DataFrame()
        
        all_features = []
        
        for symbol in request.symbols:
            try:
                # Load features for each symbol
                symbol_features = await self._load_symbol_features(
                    symbol, request.start_date, request.end_date, request.feature_names
                )
                
                if not symbol_features.empty:
                    symbol_features['symbol'] = symbol
                    all_features.append(symbol_features)
                    
            except Exception as e:
                logger.error(f"Failed to load features for {symbol}: {e}")
                continue
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            logger.debug(f"Loaded feature data: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No feature data loaded for any symbols")
            return pd.DataFrame()
    
    async def _load_symbol_features(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, feature_names: Optional[List[str]]) -> pd.DataFrame:
        """Load features for a single symbol using StorageRouter for intelligent routing."""
        # Create query filter
        query_filter = QueryFilter(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            # Use StorageRouter to determine optimal storage tier and execute query
            result = await self.storage_router.execute_query(
                repository_name='features',
                method_name='get_by_filter',
                query_filter=query_filter,
                query_type=QueryType.FEATURE_CALC,
                prefer_performance=True,  # Features often need performance
                feature_names=feature_names
            )
            
            if result is not None:
                # Convert result to DataFrame if needed
                if isinstance(result, pd.DataFrame):
                    features = result
                else:
                    features = pd.DataFrame(result)
                
                if not features.empty:
                    logger.debug(f"Loaded {len(features)} features for {symbol} via StorageRouter")
                    return features
            
        except Exception as e:
            logger.warning(f"StorageRouter query failed for features {symbol}: {e}, falling back to direct methods")
        
        # Fallback to direct HDF5 storage if StorageRouter fails
        logger.debug(f"Loading features for {symbol} from HDF5 cold storage (fallback)")
        return await self._load_from_hdf5_storage(symbol, start_date, end_date, feature_names)
    
    async def _load_from_hdf5_storage(self, symbol: str, start_date: datetime, 
                                     end_date: datetime, feature_names: Optional[List[str]]) -> pd.DataFrame:
        """Load features from HDF5 versioned storage."""
        try:
            all_features = []
            
            # Generate date ranges to check (month by month)
            current_date = start_date.replace(day=1)  # Start of month
            end_month = end_date.replace(day=1)
            
            # Define feature types to load
            if feature_names:
                # Map feature names to feature types
                feature_types = list(set([self._map_feature_to_type(f) for f in feature_names]))
            else:
                # Load common feature types
                feature_types = ['technical_indicators', 'momentum', 'volatility']
            
            while current_date <= end_month:
                for feature_type in feature_types:
                    # Load features using FeatureStoreV2
                    monthly_features = await self.feature_store_v2.load_features(
                        symbol=symbol,
                        feature_type=feature_type,
                        year=current_date.year,
                        month=current_date.month
                    )
                    
                    if monthly_features is not None and not monthly_features.empty:
                        # Filter to requested date range
                        if monthly_features.index.name == 'timestamp' or 'timestamp' in monthly_features.columns:
                            # Ensure timestamp is in index
                            if 'timestamp' in monthly_features.columns:
                                monthly_features.set_index('timestamp', inplace=True)
                            
                            # Filter by date range
                            monthly_features = monthly_features[
                                (monthly_features.index >= start_date) & 
                                (monthly_features.index <= end_date)
                            ]
                        
                        # Add metadata
                        monthly_features['feature_type'] = feature_type
                        monthly_features['symbol'] = symbol
                        all_features.append(monthly_features)
                        logger.debug(f"Loaded {len(monthly_features)} features from data lake for {symbol} {feature_type} {current_date.strftime('%Y-%m')}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            # Combine all feature data
            if all_features:
                combined_features = pd.concat(all_features, ignore_index=True)
                logger.debug(f"Loaded {len(combined_features)} total features from data lake for {symbol}")
                return combined_features
            else:
                logger.debug(f"No features found in data lake for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load features from data lake for {symbol}: {e}")
            return pd.DataFrame()
    
    def _map_feature_to_type(self, feature_name: str) -> str:
        """Map individual feature names to feature types."""
        # This is a simple mapping - can be made more sophisticated
        feature_type_mapping = {
            'sma': 'technical_indicators',
            'ema': 'technical_indicators',
            'rsi': 'momentum',
            'macd': 'momentum',
            'bb': 'volatility',
            'atr': 'volatility',
            'volume': 'volume',
            'vwap': 'volume',
            'close': 'price_action',
            'high': 'price_action',
            'low': 'price_action',
            'open': 'price_action'
        }
        
        # Check if feature name contains any of the mapped keywords
        feature_lower = feature_name.lower()
        for key, feature_type in feature_type_mapping.items():
            if key in feature_lower:
                return feature_type
        
        # Default to technical indicators
        return 'technical_indicators'
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from feature store."""
        try:
            return self.feature_store_v2.get_available_symbols()
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for features."""
        try:
            return self.feature_store_v2.get_data_range(symbol)
        except Exception as e:
            logger.error(f"Failed to get feature data range for {symbol}: {e}")
            return None, None


class AlternativeDataSource(DataSource):
    """Alternative data source using StorageRouter for intelligent hot/cold storage routing."""
    
    def __init__(self, source_type: str, config: Optional[Dict[str, Any]] = None):
        """Initialize alternative data source."""
        self.source_type = source_type
        self.config = config or {}
        self.storage_router = StorageRouter(config)
        
        # Initialize source-specific handlers
        if source_type == 'news':
            self._init_news_source()
        elif source_type == 'social':
            self._init_social_source()
        elif source_type == 'economic':
            self._init_economic_source()
    
    def _init_news_source(self):
        """Initialize news data source."""
        try:
            from main.data_pipeline.storage.repositories import get_repository_factory
            from main.interfaces.database import IAsyncDatabase
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.utils.database import DatabasePool
            
            # Initialize database adapter for news repository
            db_pool = DatabasePool()
            db_factory = DatabaseFactory()
            # Get full config, not just the alternative data config
            full_config = get_config() if not self.config else self.config
            db_adapter = db_factory.create_async_database(full_config)
            
            # Use factory to create news repository
            repo_factory = get_repository_factory()
            self.news_repo = repo_factory.create_news_repository(db_adapter)
            logger.info("News repository initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize news repository: {e}")
            self.news_repo = None
    
    def _init_social_source(self):
        """Initialize social media data source."""
        try:
            from main.data_pipeline.storage.repositories import get_repository_factory
            from main.interfaces.database import IAsyncDatabase
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.utils.database import DatabasePool
            
            # Initialize database adapter for social sentiment repository
            db_pool = DatabasePool()
            db_factory = DatabaseFactory()
            # Get full config, not just the alternative data config
            full_config = get_config() if not self.config else self.config
            db_adapter = db_factory.create_async_database(full_config)
            
            # Use factory to create social sentiment repository
            repo_factory = get_repository_factory()
            self.social_repo = repo_factory.create_social_sentiment_repository(db_adapter)
            logger.info("Social sentiment repository initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize social sentiment repository: {e}")
            self.social_repo = None
    
    def _init_economic_source(self):
        """Initialize economic data source."""
        try:
            # For now, use a placeholder that can be extended when economic data repository is available
            # Economic data sources like FRED, economic indicators, etc. can be integrated here
            from main.data_pipeline.storage.repositories.base_repository import BaseRepository
            from main.interfaces.database import IAsyncDatabase
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.utils.database import DatabasePool
            
            # Initialize database adapter for future economic repository
            db_pool = DatabasePool()
            db_factory = DatabaseFactory()
            # Get full config, not just the alternative data config
            full_config = get_config() if not self.config else self.config
            db_adapter = db_factory.create_async_database(full_config)
            
            # Create placeholder economic repository (can be replaced with actual implementation)
            class EconomicDataRepository(BaseRepository):
                """Placeholder economic data repository."""
                async def get_economic_data(self, indicators: List[str], start_date: datetime, end_date: datetime):
                    """Placeholder for economic data retrieval."""
                    logger.debug(f"Economic data requested for indicators: {indicators}")
                    return pd.DataFrame()  # Return empty for now
            
            self.economic_repo = EconomicDataRepository(db_adapter)
            logger.info("Economic data repository placeholder initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize economic data repository: {e}")
            self.economic_repo = None
    
    async def load_data(self, request: DataRequest) -> pd.DataFrame:
        """Load alternative data for the request."""
        logger.debug(f"Loading {self.source_type} data for {len(request.symbols)} symbols")
        
        try:
            if self.source_type == 'news' and self.news_repo:
                # Load news data for symbols
                news_data = await self._load_news_data(request)
                return news_data
            elif self.source_type == 'social' and self.social_repo:
                # Load social sentiment data for symbols
                social_data = await self._load_social_data(request)
                return social_data
            elif self.source_type == 'economic' and self.economic_repo:
                # Load economic data
                economic_data = await self._load_economic_data(request)
                return economic_data
            else:
                logger.warning(f"No repository available for {self.source_type} data source")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {self.source_type} data: {e}")
            return pd.DataFrame()
    
    async def _load_news_data(self, request: DataRequest) -> pd.DataFrame:
        """Load news data using StorageRouter for intelligent routing."""
        try:
            # Create query filter
            query_filter = QueryFilter(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Use StorageRouter for optimal storage tier selection
            result = await self.storage_router.execute_query(
                repository_name='news',
                method_name='get_by_filter',
                query_filter=query_filter,
                query_type=QueryType.ANALYSIS,  # News is typically used for analysis
                prefer_performance=False
            )
            
            if result is not None:
                # Convert result to DataFrame if needed
                if isinstance(result, pd.DataFrame):
                    news_df = result
                else:
                    news_df = pd.DataFrame(result)
                
                if not news_df.empty:
                    logger.info(f"Loaded {len(news_df)} news articles via StorageRouter")
                    return news_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"StorageRouter query failed for news: {e}, falling back to direct query")
            
            # Fallback to direct repository query
            try:
                filters = {
                    'symbols': request.symbols,
                    'start_date': request.start_date,
                    'end_date': request.end_date
                }
                news_results = await self.news_repo.query_by_filters(filters)
                
                if news_results:
                    news_df = pd.DataFrame([result.__dict__ for result in news_results])
                    logger.info(f"Loaded {len(news_df)} news articles (fallback)")
                    return news_df
                return pd.DataFrame()
            except Exception as fallback_error:
                logger.error(f"Fallback news data load also failed: {fallback_error}")
                return pd.DataFrame()
    
    async def _load_social_data(self, request: DataRequest) -> pd.DataFrame:
        """Load social sentiment data using StorageRouter for intelligent routing."""
        try:
            # Create query filter
            query_filter = QueryFilter(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Use StorageRouter for optimal storage tier selection
            result = await self.storage_router.execute_query(
                repository_name='social_sentiment',
                method_name='get_by_filter',
                query_filter=query_filter,
                query_type=QueryType.ANALYSIS,  # Social sentiment is typically used for analysis
                prefer_performance=False
            )
            
            if result is not None:
                # Convert result to DataFrame if needed
                if isinstance(result, pd.DataFrame):
                    social_df = result
                else:
                    social_df = pd.DataFrame(result)
                
                if not social_df.empty:
                    logger.info(f"Loaded {len(social_df)} social sentiment records via StorageRouter")
                    return social_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"StorageRouter query failed for social sentiment: {e}, falling back to direct query")
            
            # Fallback to direct repository query
            try:
                filters = {
                    'symbols': request.symbols,
                    'start_date': request.start_date,
                    'end_date': request.end_date
                }
                social_results = await self.social_repo.query_by_filters(filters)
                
                if social_results:
                    social_df = pd.DataFrame([result.__dict__ for result in social_results])
                    logger.info(f"Loaded {len(social_df)} social sentiment records (fallback)")
                    return social_df
                return pd.DataFrame()
            except Exception as fallback_error:
                logger.error(f"Fallback social data load also failed: {fallback_error}")
                return pd.DataFrame()
    
    async def _load_economic_data(self, request: DataRequest) -> pd.DataFrame:
        """Load economic data from repository."""
        try:
            # For now, use placeholder implementation
            # In the future, this would query economic indicators like GDP, CPI, unemployment, etc.
            economic_df = await self.economic_repo.get_economic_data(
                indicators=['GDP', 'CPI', 'UNEMPLOYMENT'],
                start_date=request.start_date,
                end_date=request.end_date
            )
            logger.info(f"Loaded economic data with {len(economic_df)} records")
            return economic_df
        except Exception as e:
            logger.error(f"Error loading economic data: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from alternative data source."""
        try:
            if self.source_type == 'news' and self.news_repo:
                # Get symbols from news repository
                symbols = asyncio.run(self._get_news_symbols())
                return symbols
            elif self.source_type == 'social' and self.social_repo:
                # Get symbols from social sentiment repository
                symbols = asyncio.run(self._get_social_symbols())
                return symbols
            elif self.source_type == 'economic' and self.economic_repo:
                # Economic data typically doesn't have symbol-specific data
                return []
            else:
                logger.warning(f"No repository available for {self.source_type} data source")
                return []
        except Exception as e:
            logger.error(f"Error getting available symbols for {self.source_type}: {e}")
            return []
    
    async def _get_news_symbols(self) -> List[str]:
        """Get available symbols from news repository."""
        try:
            # Query distinct symbols from news data
            symbols = await self.news_repo.get_distinct_symbols()
            return symbols if symbols else []
        except Exception as e:
            logger.error(f"Error getting news symbols: {e}")
            return []
    
    async def _get_social_symbols(self) -> List[str]:
        """Get available symbols from social sentiment repository."""
        try:
            # Query distinct symbols from social sentiment data
            symbols = await self.social_repo.get_distinct_symbols()
            return symbols if symbols else []
        except Exception as e:
            logger.error(f"Error getting social symbols: {e}")
            return []
    
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for alternative data."""
        try:
            if self.source_type == 'news' and self.news_repo:
                # Get date range from news repository
                date_range = asyncio.run(self._get_news_date_range(symbol))
                return date_range
            elif self.source_type == 'social' and self.social_repo:
                # Get date range from social sentiment repository
                date_range = asyncio.run(self._get_social_date_range(symbol))
                return date_range
            elif self.source_type == 'economic' and self.economic_repo:
                # Economic data typically has broader date ranges
                # Return a reasonable default range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * 5)  # 5 years
                return start_date, end_date
            else:
                logger.warning(f"No repository available for {self.source_type} data source")
                return None, None
        except Exception as e:
            logger.error(f"Error getting data range for {symbol} in {self.source_type}: {e}")
            return None, None
    
    async def _get_news_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get date range for news data."""
        try:
            # Query min/max dates for symbol from news repository
            date_range = await self.news_repo.get_date_range_for_symbol(symbol)
            return date_range if date_range else (None, None)
        except Exception as e:
            logger.error(f"Error getting news date range for {symbol}: {e}")
            return None, None
    
    async def _get_social_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get date range for social sentiment data."""
        try:
            # Query min/max dates for symbol from social sentiment repository
            date_range = await self.social_repo.get_date_range_for_symbol(symbol)
            return date_range if date_range else (None, None)
        except Exception as e:
            logger.error(f"Error getting social date range for {symbol}: {e}")
            return None, None


class DataLoader:
    """
    Unified data loader for the feature pipeline.
    
    Coordinates data loading from multiple sources and provides a consistent interface
    for accessing market data, features, and alternative data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data loader with configuration."""
        self.config = config or get_config().get('data_loader', {})
        
        # Initialize data sources
        self.sources = {
            'market_data': MarketDataSource(self.config.get('market_data', {})),
            'features': FeatureDataSource(self.config.get('features', {})),
            'news': AlternativeDataSource('news', self.config.get('news', {})),
            'social': AlternativeDataSource('social', self.config.get('social', {})),
            'economic': AlternativeDataSource('economic', self.config.get('economic', {}))
        }
        
        # Data preprocessing
        self.preprocessor = None
        if self.config.get('enable_preprocessing', True):
            from .data_preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        logger.info(f"DataLoader initialized with sources: {list(self.sources.keys())}")
    
    async def load_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """
        Load data according to request specification.
        
        Args:
            request: Data request specification
            
        Returns:
            Dictionary mapping data type to DataFrame
        """
        logger.info(f"Loading data for {len(request.symbols)} symbols, types: {request.data_types}")
        
        # Validate request
        if not request.symbols:
            raise ValueError("No symbols specified in data request")
        
        if request.start_date >= request.end_date:
            raise ValueError("Start date must be before end date")
        
        # Load data from requested sources
        results = {}
        tasks = []
        
        for data_type in request.data_types:
            if data_type in self.sources:
                task = asyncio.create_task(
                    self._load_from_source(data_type, request),
                    name=f"load_{data_type}"
                )
                tasks.append((data_type, task))
            else:
                logger.warning(f"Unknown data type requested: {data_type}")
        
        # Load features if requested
        if request.include_features or request.feature_names:
            task = asyncio.create_task(
                self._load_from_source('features', request),
                name="load_features"
            )
            tasks.append(('features', task))
        
        # Wait for all loading tasks to complete
        for data_type, task in tasks:
            try:
                data = await task
                if not data.empty:
                    results[data_type] = data
                    logger.debug(f"Loaded {data_type}: {data.shape}")
                else:
                    logger.warning(f"No data loaded for {data_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load {data_type}: {e}")
                results[data_type] = pd.DataFrame()
        
        # Apply preprocessing if enabled
        if self.preprocessor and request.preprocessing:
            results = await self._apply_preprocessing(results)
        
        logger.info(f"Data loading complete. Loaded {len(results)} data types")
        return results
    
    async def _load_from_source(self, data_type: str, request: DataRequest) -> pd.DataFrame:
        """Load data from a specific source."""
        source = self.sources[data_type]
        return await source.load_data(request)
    
    async def _apply_preprocessing(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply preprocessing to loaded data."""
        logger.debug("Applying preprocessing to loaded data")
        
        processed_data = {}
        
        for data_type, data in data_dict.items():
            if data.empty:
                processed_data[data_type] = data
                continue
                
            try:
                if data_type == 'market_data':
                    # Apply market data preprocessing
                    symbols = data['symbol'].unique() if 'symbol' in data.columns else ['unknown']
                    processed_dfs = []
                    
                    for symbol in symbols:
                        symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data
                        processed_symbol_data = self.preprocessor.preprocess_market_data(symbol_data, symbol)
                        processed_dfs.append(processed_symbol_data)
                    
                    processed_data[data_type] = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
                    
                elif data_type == 'features':
                    # Apply feature preprocessing
                    processed_data[data_type] = self.preprocessor.preprocess_feature_data(data, 'general')
                    
                elif data_type in ['news', 'social', 'economic']:
                    # Apply alternative data preprocessing
                    processed_data[data_type] = self.preprocessor.preprocess_alternative_data(data, data_type)
                    
                else:
                    # No specific preprocessing for this data type
                    processed_data[data_type] = data
                    
                logger.debug(f"Preprocessed {data_type}: {data.shape} -> {processed_data[data_type].shape}")
                
            except Exception as e:
                logger.error(f"Preprocessing failed for {data_type}: {e}")
                processed_data[data_type] = data  # Return original data if preprocessing fails
        
        return processed_data
    
    async def load_market_data(self, symbols: List[str], start_date: datetime, 
                              end_date: datetime, interval: str = '1D',
                              preprocessing: bool = True) -> pd.DataFrame:
        """
        Convenience method to load only market data.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            preprocessing: Whether to apply preprocessing
            
        Returns:
            Market data DataFrame
        """
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=['market_data'],
            interval=interval,
            preprocessing=preprocessing
        )
        
        results = await self.load_data(request)
        return results.get('market_data', pd.DataFrame())
    
    async def load_features(self, symbols: List[str], start_date: datetime,
                           end_date: datetime, feature_names: Optional[List[str]] = None,
                           preprocessing: bool = True) -> pd.DataFrame:
        """
        Convenience method to load only feature data.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for features
            end_date: End date for features
            feature_names: Specific features to load (None for all)
            preprocessing: Whether to apply preprocessing
            
        Returns:
            Feature data DataFrame
        """
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=[],
            include_features=True,
            feature_names=feature_names,
            preprocessing=preprocessing
        )
        
        results = await self.load_data(request)
        return results.get('features', pd.DataFrame())
    
    async def load_combined_data(self, symbols: List[str], start_date: datetime,
                                end_date: datetime, interval: str = '1D',
                                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and combine market data with features.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            feature_names: Features to include
            
        Returns:
            Combined DataFrame with market data and features
        """
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=['market_data'],
            interval=interval,
            include_features=True,
            feature_names=feature_names,
            preprocessing=True
        )
        
        results = await self.load_data(request)
        
        # Combine market data and features
        market_data = results.get('market_data', pd.DataFrame())
        features = results.get('features', pd.DataFrame())
        
        if market_data.empty:
            return features
        elif features.empty:
            return market_data
        else:
            # Merge on symbol and timestamp
            combined = pd.merge(
                market_data, features,
                on=['symbol', 'timestamp'],
                how='left'
            )
            return combined
    
    def get_available_symbols(self, data_type: str = 'market_data') -> List[str]:
        """Get available symbols from a data source."""
        if data_type in self.sources:
            return self.sources[data_type].get_available_symbols()
        else:
            logger.error(f"Unknown data type: {data_type}")
            return []
    
    def get_data_range(self, symbol: str, data_type: str = 'market_data') -> Tuple[datetime, datetime]:
        """Get available date range for a symbol."""
        if data_type in self.sources:
            return self.sources[data_type].get_data_range(symbol)
        else:
            logger.error(f"Unknown data type: {data_type}")
            return None, None
    
    async def close(self):
        """Close the data loader and cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        logger.info("DataLoader closed")


# Convenience functions for common loading patterns
async def load_market_data(symbols: List[str], start_date: datetime, end_date: datetime,
                          interval: str = '1D', config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load market data for symbols."""
    loader = DataLoader(config)
    try:
        return await loader.load_market_data(symbols, start_date, end_date, interval)
    finally:
        await loader.close()


async def load_features(symbols: List[str], start_date: datetime, end_date: datetime,
                       feature_names: Optional[List[str]] = None,
                       config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load feature data for symbols."""
    loader = DataLoader(config)
    try:
        return await loader.load_features(symbols, start_date, end_date, feature_names)
    finally:
        await loader.close()


async def load_combined_data(symbols: List[str], start_date: datetime, end_date: datetime,
                           interval: str = '1D', feature_names: Optional[List[str]] = None,
                           config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load combined market data and features."""
    loader = DataLoader(config)
    try:
        return await loader.load_combined_data(symbols, start_date, end_date, interval, feature_names)
    finally:
        await loader.close()
"""
Feature Repository

Repository for ML feature storage and retrieval.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import json
import time

from main.interfaces.repositories.feature import IFeatureRepository
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import (
    RepositoryConfig,
    QueryFilter,
    OperationResult
)

from .base_repository import BaseRepository
from .helpers import (
    QueryBuilder,
    BatchProcessor,
    CrudExecutor,
    RepositoryMetricsCollector
)

from main.utils.core import get_logger, ensure_utc
from .constants import DEFAULT_BATCH_SIZE, DEFAULT_MAX_PARALLEL_WORKERS

logger = get_logger(__name__)


class FeatureRepository(BaseRepository, IFeatureRepository):
    """
    Repository for ML feature storage with JSON serialization.
    
    Handles complex feature storage, retrieval, statistics, and
    correlation analysis for machine learning models.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ):
        """
        Initialize the FeatureRepository.
        
        Args:
            db_adapter: Database adapter
            config: Optional repository configuration
        """
        # Initialize with features table
        super().__init__(db_adapter, type('Feature', (), {'__tablename__': 'features'}), config)
        
        # Additional components
        self.query_builder = QueryBuilder('features')
        self.crud_executor = CrudExecutor(
            db_adapter,
            'features',
            transaction_strategy=config.transaction_strategy if config else None
        )
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size if config else DEFAULT_BATCH_SIZE,
            max_parallel=config.max_parallel_workers if config else DEFAULT_MAX_PARALLEL_WORKERS
        )
        self.metrics = RepositoryMetricsCollector(
            'FeatureRepository',
            enable_metrics=config.enable_metrics if config else True
        )
        
        logger.info("FeatureRepository initialized with JSON feature storage")
    
    # Required abstract methods from BaseRepository
    def get_required_fields(self) -> List[str]:
        """Get required fields for feature data."""
        return ['symbol', 'timestamp', 'features']
    
    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate feature record."""
        errors = []
        
        # Check required fields
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate features structure
        if 'features' in record and record['features'] is not None:
            if isinstance(record['features'], str):
                # Try to parse JSON string
                try:
                    features = json.loads(record['features'])
                    if not isinstance(features, dict):
                        errors.append("Features must be a dictionary")
                except json.JSONDecodeError:
                    errors.append("Features contains invalid JSON")
            elif not isinstance(record['features'], dict):
                errors.append("Features must be a dictionary or JSON string")
        
        return errors
    
    # IFeatureRepository interface implementation
    async def store_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """Store calculated features."""
        start_time = time.time()
        
        try:
            # Prepare record
            record = {
                'symbol': self._normalize_symbol(symbol),
                'timestamp': ensure_utc(timestamp),
                'features': json.dumps(features),
                'feature_count': len(features),
                'metadata': json.dumps(metadata) if metadata else None,
                'created_at': datetime.now(timezone.utc)
            }
            
            # Use upsert to handle duplicates
            query = """
                INSERT INTO features (symbol, timestamp, features, feature_count, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (symbol, timestamp) DO UPDATE
                SET features = EXCLUDED.features,
                    feature_count = EXCLUDED.feature_count,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at
                RETURNING id
            """
            
            params = [
                record['symbol'],
                record['timestamp'],
                record['features'],
                record['feature_count'],
                record['metadata'],
                record['created_at']
            ]
            
            result = await self.crud_executor.execute_upsert(query, params)
            
            # Invalidate cache
            await self._invalidate_cache(f"features_{symbol}*")
            
            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation(
                'store_features',
                duration,
                success=result.success,
                records=1 if result.success else 0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def get_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Retrieve stored features."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(
                f"features_{symbol}",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                features=','.join(feature_names) if feature_names else 'all'
            )
            
            cached_data = await self._get_from_cache(cache_key)
            if cached_data is not None:
                await self.metrics.record_cache_access(hit=True)
                return pd.DataFrame(cached_data)
            
            await self.metrics.record_cache_access(hit=False)
            
            # Query database
            query = """
                SELECT timestamp, features
                FROM features
                WHERE symbol = $1
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp ASC
            """
            
            params = [
                self._normalize_symbol(symbol),
                ensure_utc(start_date),
                ensure_utc(end_date)
            ]
            
            results = await self.db_adapter.fetch_all(query, *params)
            
            if not results:
                return pd.DataFrame()
            
            # Parse features and build DataFrame
            data = []
            for row in results:
                features_dict = json.loads(row['features'])
                
                # Filter features if specified
                if feature_names:
                    features_dict = {k: v for k, v in features_dict.items() if k in feature_names}
                
                features_dict['timestamp'] = row['timestamp']
                data.append(features_dict)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
                
                # Cache the result
                await self._set_in_cache(cache_key, df.reset_index().to_dict('records'))
            
            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation(
                'get_features',
                duration,
                success=True,
                records=len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            duration = time.time() - start_time
            await self.metrics.record_operation('get_features', duration, success=False)
            return pd.DataFrame()
    
    async def get_latest_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get latest features for a symbol."""
        try:
            query = """
                SELECT timestamp, features, metadata
                FROM features
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            result = await self.db_adapter.fetch_one(query, self._normalize_symbol(symbol))
            
            if result:
                features = json.loads(result['features'])
                
                # Filter features if specified
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                
                return {
                    'timestamp': result['timestamp'],
                    'features': features,
                    'metadata': json.loads(result['metadata']) if result['metadata'] else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest features: {e}")
            raise
    
    async def batch_store_features(self, features_data: pd.DataFrame) -> OperationResult:
        """Store features for multiple symbols/timestamps."""
        start_time = time.time()
        
        try:
            # Prepare records
            records = []
            for idx, row in features_data.iterrows():
                # Extract symbol and timestamp
                symbol = row.get('symbol')
                timestamp = idx if isinstance(idx, datetime) else row.get('timestamp')
                
                if not symbol or not timestamp:
                    continue
                
                # Get feature columns (exclude symbol and timestamp)
                feature_cols = [c for c in row.index if c not in ['symbol', 'timestamp']]
                features = {col: float(row[col]) for col in feature_cols if pd.notna(row[col])}
                
                records.append({
                    'symbol': self._normalize_symbol(symbol),
                    'timestamp': ensure_utc(timestamp),
                    'features': json.dumps(features),
                    'feature_count': len(features),
                    'created_at': datetime.now(timezone.utc)
                })
            
            # Process in batches
            async def store_batch(batch: List[Dict[str, Any]]) -> Any:
                # Build bulk upsert query
                for record in batch:
                    await self.store_features(
                        record['symbol'],
                        record['timestamp'],
                        json.loads(record['features'])
                    )
                return len(batch)
            
            result = await self.batch_processor.process_batch(
                records,
                store_batch
            )
            
            # Clear cache
            await self._invalidate_cache("features_*")
            
            duration = time.time() - start_time
            
            return OperationResult(
                success=result['success'],
                records_affected=result['statistics']['succeeded'],
                records_created=result['statistics']['succeeded'],
                duration_seconds=duration,
                metadata=result['statistics']
            )
            
        except Exception as e:
            logger.error(f"Error in batch feature storage: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def get_feature_statistics(
        self,
        symbol: str,
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Get statistics for features over a period."""
        try:
            # Get features
            df = await self.get_features(symbol, start_date, end_date, feature_names)
            
            if df.empty:
                return {}
            
            # Calculate statistics for each feature
            stats = {}
            for feature in feature_names:
                if feature in df.columns:
                    series = df[feature].dropna()
                    if len(series) > 0:
                        stats[feature] = {
                            'mean': float(series.mean()),
                            'std': float(series.std()),
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'median': float(series.median()),
                            'q25': float(series.quantile(0.25)),
                            'q75': float(series.quantile(0.75)),
                            'count': len(series),
                            'null_count': len(df) - len(series)
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {e}")
            return {}
    
    async def get_feature_correlation(
        self,
        symbol: str,
        features: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Calculate correlation matrix for features."""
        try:
            # Get features
            df = await self.get_features(symbol, start_date, end_date, features)
            
            if df.empty or len(df) < 2:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            correlation = df[features].corr()
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating feature correlation: {e}")
            return pd.DataFrame()
    
    async def cleanup_old_features(
        self,
        days_to_keep: int,
        feature_names: Optional[List[str]] = None
    ) -> OperationResult:
        """Clean up old feature data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Build delete query
            if feature_names:
                # Would need to filter by JSON content - complex query
                logger.warning("Feature name filtering not implemented for cleanup")
            
            query = "DELETE FROM features WHERE timestamp < $1"
            params = [cutoff_date]
            
            result = await self.crud_executor.execute_delete(query, params)
            
            # Clear cache
            await self._invalidate_cache()
            
            logger.info(f"Cleaned up {result.records_deleted} old feature records")
            
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old features: {e}")
            return OperationResult(success=False, error=str(e))
    
    async def get_feature_availability(
        self,
        symbols: List[str],
        feature_names: List[str],
        date: datetime
    ) -> Dict[str, Dict[str, bool]]:
        """Check feature availability for symbols on a date."""
        try:
            availability = {}
            
            # Check each symbol
            for symbol in symbols:
                # Get features for the date
                start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                df = await self.get_features(symbol, start_date, end_date)
                
                if df.empty:
                    availability[symbol] = {f: False for f in feature_names}
                else:
                    # Check which features are present
                    availability[symbol] = {}
                    for feature in feature_names:
                        availability[symbol][feature] = feature in df.columns and df[feature].notna().any()
            
            return availability
            
        except Exception as e:
            logger.error(f"Error checking feature availability: {e}")
            return {}
    
    async def get_missing_features(
        self,
        symbol: str,
        expected_features: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Find missing features in a date range."""
        try:
            # Get all features
            df = await self.get_features(symbol, start_date, end_date)
            
            if df.empty:
                return [{
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'missing_features': expected_features,
                    'reason': 'No data available'
                }]
            
            # Check for missing features
            missing_records = []
            
            for timestamp, row in df.iterrows():
                missing_features = []
                for feature in expected_features:
                    if feature not in row or pd.isna(row[feature]):
                        missing_features.append(feature)
                
                if missing_features:
                    missing_records.append({
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'missing_features': missing_features
                    })
            
            return missing_records
            
        except Exception as e:
            logger.error(f"Error finding missing features: {e}")
            return []
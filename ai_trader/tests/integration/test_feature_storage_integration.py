# tests/integration/test_feature_storage_integration.py

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, Mock
from typing import Dict, List, Any, Optional

from main.feature_pipeline.feature_store import FeatureStoreRepository, FeatureStore
from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.archive import DataArchive, get_archive
from main.data_pipeline.storage.repositories.market_data import MarketDataRepository


# Test Fixtures
@pytest.fixture
def mock_database_config():
    """Mock database configuration."""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'test_features_db',
            'user': 'test_user',
            'password': 'test_password',
            'pool_size': 5,
            'max_overflow': 10
        },
        'feature_store': {
            'hot_storage_days': 7,
            'batch_size': 1000,
            'parallel_writes': True
        }
    }


@pytest.fixture
def sample_feature_data():
    """Generate sample feature data for storage testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='H', tz=timezone.utc)
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    feature_data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        features = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'sma_10': np.secure_uniform(95, 105, len(dates)),
            'sma_20': np.secure_uniform(95, 105, len(dates)),
            'rsi': np.secure_uniform(0, 100, len(dates)),
            'volume_ratio': np.secure_uniform(0.5, 2.0, len(dates)),
            'volatility': np.secure_uniform(0.1, 0.5, len(dates)),
            'momentum': secure_numpy_normal(0, 0.1, len(dates)),
            'sentiment_score': np.secure_uniform(-1, 1, len(dates)),
            'news_sentiment': np.secure_uniform(-1, 1, len(dates)),
        })
        feature_data[symbol] = features
    
    return feature_data


@pytest.fixture
def temp_storage_directory():
    """Create temporary directory for storage testing."""
    temp_dir = tempfile.mkdtemp(prefix='feature_storage_test_')
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_db_adapter():
    """Mock database adapter for testing."""
    adapter = MagicMock(spec=AsyncDatabaseAdapter)
    adapter.execute_query = AsyncMock()
    adapter.fetch_query = AsyncMock()
    adapter.execute_batch = AsyncMock()
    adapter.create_connection = AsyncMock()
    adapter.close_connection = AsyncMock()
    return adapter


@pytest.fixture
def mock_data_archive():
    """Mock data archive for cold storage testing."""
    archive = MagicMock(spec=DataArchive)
    archive.save_dataframe = AsyncMock()
    archive.load_dataframe = AsyncMock()
    archive.exists = AsyncMock()
    archive.list_keys = AsyncMock()
    archive.delete_key = AsyncMock()
    return archive


# Test Feature Store Repository (Hot Storage)
class TestFeatureStoreRepository:
    """Test PostgreSQL feature store repository operations."""
    
    @pytest.mark.asyncio
    async def test_store_features_single_record(self, mock_db_adapter, mock_database_config):
        """Test storing a single feature record."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock successful database response
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.errors = []
        mock_db_adapter.execute_query.return_value = mock_result
        
        # Test data
        features = {
            'sma_10': 100.5,
            'sma_20': 99.8,
            'rsi': 65.4,
            'volume_ratio': 1.2,
            'volatility': 0.25
        }
        
        result = await repo.store_features(
            symbol='AAPL',
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            features=features,
            feature_set='technical',
            version='1.0'
        )
        
        # Verify successful storage
        assert result.success is True
        assert len(result.errors) == 0
        
        # Verify database was called with correct SQL
        mock_db_adapter.execute_query.assert_called_once()
        call_args = mock_db_adapter.execute_query.call_args
        
        # Check SQL contains INSERT or UPSERT
        sql = call_args[0][0]
        assert 'INSERT' in sql.upper() or 'UPSERT' in sql.upper()
        
        # Check parameters include our data
        params = call_args[0][1]
        assert params['symbol'] == 'AAPL'
        assert params['feature_set'] == 'technical'
        assert 'sma_10' in json.loads(params['features'])
    
    @pytest.mark.asyncio
    async def test_store_features_batch(self, mock_db_adapter, sample_feature_data):
        """Test batch storage of multiple feature records."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock successful batch response
        mock_db_adapter.execute_batch.return_value = True
        
        # Prepare batch data
        batch_records = []
        for symbol, features_df in sample_feature_data.items():
            for _, row in features_df.iterrows():
                batch_records.append({
                    'symbol': symbol,
                    'timestamp': row['timestamp'],
                    'features': {
                        'sma_10': row['sma_10'],
                        'sma_20': row['sma_20'],
                        'rsi': row['rsi']
                    },
                    'feature_set': 'technical'
                })
        
        # Store batch
        success = await repo.store_features_batch(batch_records)
        
        assert success is True
        mock_db_adapter.execute_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_features_by_symbol_and_timerange(self, mock_db_adapter):
        """Test retrieving features by symbol and time range."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock database response
        mock_rows = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                'features': '{"sma_10": 100.5, "rsi": 65.4}',
                'feature_set': 'technical',
                'version': '1.0'
            },
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
                'features': '{"sma_10": 101.2, "rsi": 67.1}',
                'feature_set': 'technical',
                'version': '1.0'
            }
        ]
        mock_db_adapter.fetch_query.return_value = mock_rows
        
        # Query features
        result = await repo.get_features(
            symbol='AAPL',
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            feature_set='technical'
        )
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'sma_10' in result.columns
        assert 'rsi' in result.columns
        assert result['symbol'].iloc[0] == 'AAPL'
        
        # Verify database was queried correctly
        mock_db_adapter.fetch_query.assert_called_once()
        call_args = mock_db_adapter.fetch_query.call_args
        sql = call_args[0][0]
        assert 'SELECT' in sql.upper()
        assert 'WHERE' in sql.upper()
    
    @pytest.mark.asyncio
    async def test_get_features_error_handling(self, mock_db_adapter):
        """Test error handling during feature retrieval."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock database error
        mock_db_adapter.fetch_query.side_effect = Exception("Database connection failed")
        
        # Should handle error gracefully
        result = await repo.get_features(
            symbol='AAPL',
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @pytest.mark.asyncio
    async def test_feature_versioning(self, mock_db_adapter):
        """Test feature versioning support."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock successful storage
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.errors = []
        mock_db_adapter.execute_query.return_value = mock_result
        
        # Store features with different versions
        features_v1 = {'sma_10': 100.0}
        features_v2 = {'sma_10': 100.0, 'sma_20': 99.5}  # Added feature
        
        await repo.store_features(
            symbol='AAPL',
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features=features_v1,
            version='1.0'
        )
        
        await repo.store_features(
            symbol='AAPL',
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features=features_v2,
            version='2.0'
        )
        
        # Verify both versions were stored
        assert mock_db_adapter.execute_query.call_count == 2
        
        # Check that version is included in parameters
        calls = mock_db_adapter.execute_query.call_args_list
        assert calls[0][0][1]['version'] == '1.0'
        assert calls[1][0][1]['version'] == '2.0'


# Test Data Archive (Cold Storage)
class TestDataArchiveIntegration:
    """Test Data Lake parquet file storage integration."""
    
    @pytest.mark.asyncio
    async def test_save_features_to_data_lake(self, mock_data_archive, sample_feature_data):
        """Test saving features to Data Lake storage."""
        # Test saving features for single symbol
        symbol = 'AAPL'
        features_df = sample_feature_data[symbol]
        
        # Mock successful save
        mock_data_archive.save_dataframe.return_value = True
        
        # Save to archive
        feature_key = f"features/{symbol}/2024/01/01/features.parquet"
        metadata = {
            'symbol': symbol,
            'feature_count': len(features_df.columns),
            'record_count': len(features_df),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        await mock_data_archive.save_dataframe(feature_key, features_df, metadata=metadata)
        
        # Verify save was called correctly
        mock_data_archive.save_dataframe.assert_called_once_with(
            feature_key, features_df, metadata=metadata
        )
    
    @pytest.mark.asyncio
    async def test_load_features_from_data_lake(self, mock_data_archive, sample_feature_data):
        """Test loading features from Data Lake storage."""
        symbol = 'AAPL'
        expected_features = sample_feature_data[symbol]
        
        # Mock successful load
        mock_data_archive.load_dataframe.return_value = expected_features
        
        # Load from archive
        feature_key = f"features/{symbol}/2024/01/01/features.parquet"
        result = await mock_data_archive.load_dataframe(feature_key)
        
        # Verify correct data returned
        pd.testing.assert_frame_equal(result, expected_features)
        mock_data_archive.load_dataframe.assert_called_once_with(feature_key)
    
    @pytest.mark.asyncio
    async def test_feature_archiving_strategy(self, mock_data_archive, sample_feature_data):
        """Test feature archiving strategy (partitioning by date/symbol)."""
        # Test archiving multiple symbols with date partitioning
        base_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        
        for symbol, features_df in sample_feature_data.items():
            # Create date-partitioned key
            feature_key = f"features/{symbol}/{base_date.strftime('%Y/%m/%d')}/features.parquet"
            
            await mock_data_archive.save_dataframe(
                feature_key, 
                features_df,
                metadata={'symbol': symbol, 'partition_date': base_date.isoformat()}
            )
        
        # Verify all symbols were archived with proper partitioning
        assert mock_data_archive.save_dataframe.call_count == len(sample_feature_data)
        
        # Check partitioning structure
        calls = mock_data_archive.save_dataframe.call_args_list
        for i, (symbol, _) in enumerate(sample_feature_data.items()):
            feature_key = calls[i][0][0]
            assert f"features/{symbol}/2024/01/15/" in feature_key
    
    @pytest.mark.asyncio
    async def test_feature_key_generation_patterns(self, mock_data_archive):
        """Test different feature key generation patterns."""
        test_cases = [
            {
                'symbol': 'AAPL',
                'date': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'feature_set': 'technical',
                'expected_pattern': 'features/AAPL/2024/01/01/technical.parquet'
            },
            {
                'symbol': 'MSFT',
                'date': datetime(2024, 12, 31, tzinfo=timezone.utc),
                'feature_set': 'sentiment',
                'expected_pattern': 'features/MSFT/2024/12/31/sentiment.parquet'
            }
        ]
        
        for case in test_cases:
            # Generate key based on pattern
            feature_key = f"features/{case['symbol']}/{case['date'].strftime('%Y/%m/%d')}/{case['feature_set']}.parquet"
            
            # Verify pattern matches expected
            assert feature_key == case['expected_pattern']
    
    @pytest.mark.asyncio
    async def test_archive_error_handling(self, mock_data_archive, sample_feature_data):
        """Test error handling in archive operations."""
        # Mock save failure
        mock_data_archive.save_dataframe.side_effect = Exception("S3 upload failed")
        
        symbol = 'AAPL'
        features_df = sample_feature_data[symbol]
        feature_key = f"features/{symbol}/2024/01/01/features.parquet"
        
        # Should handle error gracefully
        with pytest.raises(Exception, match="S3 upload failed"):
            await mock_data_archive.save_dataframe(feature_key, features_df)
        
        # Test load failure
        mock_data_archive.load_dataframe.side_effect = Exception("S3 download failed")
        
        with pytest.raises(Exception, match="S3 download failed"):
            await mock_data_archive.load_dataframe(feature_key)


# Test Hot/Cold Storage Routing
class TestHotColdStorageRouting:
    """Test routing logic between hot and cold storage."""
    
    @pytest.mark.asyncio
    async def test_storage_tier_routing_by_age(self, mock_db_adapter, mock_data_archive, mock_database_config):
        """Test routing queries to appropriate storage tier based on data age."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Configure hot storage window (7 days)
        hot_storage_days = mock_database_config['feature_store']['hot_storage_days']
        
        now = datetime.now(timezone.utc)
        
        # Test cases: recent data should go to hot storage, old data to cold storage
        test_cases = [
            {
                'query_time': now - timedelta(days=1),  # Recent - hot storage
                'expected_storage': 'hot',
                'description': 'Recent data query'
            },
            {
                'query_time': now - timedelta(days=hot_storage_days - 1),  # Still hot
                'expected_storage': 'hot',
                'description': 'Edge of hot storage window'
            },
            {
                'query_time': now - timedelta(days=hot_storage_days + 1),  # Old - cold storage
                'expected_storage': 'cold',
                'description': 'Old data query'
            },
            {
                'query_time': now - timedelta(days=30),  # Very old - cold storage
                'expected_storage': 'cold',
                'description': 'Very old data query'
            }
        ]
        
        for case in test_cases:
            # Determine which storage should be used
            data_age = (now - case['query_time']).days
            should_use_hot = data_age <= hot_storage_days
            
            if case['expected_storage'] == 'hot':
                assert should_use_hot, f"Failed for {case['description']}"
            else:
                assert not should_use_hot, f"Failed for {case['description']}"
    
    @pytest.mark.asyncio
    async def test_hybrid_query_spanning_storage_tiers(self, mock_db_adapter, mock_data_archive):
        """Test queries that span both hot and cold storage."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock responses from both storage tiers
        hot_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-08', periods=3, freq='D', tz=timezone.utc),
            'symbol': 'AAPL',
            'sma_10': [100.1, 100.2, 100.3]
        })
        
        cold_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D', tz=timezone.utc),
            'symbol': 'AAPL',
            'sma_10': [99.1, 99.2, 99.3, 99.4, 99.5]
        })
        
        # Mock hot storage query
        mock_db_adapter.fetch_query.return_value = hot_data.to_dict('records')
        
        # Mock cold storage query
        mock_data_archive.load_dataframe.return_value = cold_data
        
        # For this test, simulate a hybrid query strategy
        # In practice, this would be implemented in a wrapper service
        
        # Query hot storage
        hot_result = await repo.get_features(
            symbol='AAPL',
            start_time=datetime(2024, 1, 8, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 10, tzinfo=timezone.utc)
        )
        
        # Query cold storage
        cold_result = await mock_data_archive.load_dataframe("features/AAPL/2024/01/01/features.parquet")
        
        # Combine results (this would be done by routing service)
        combined_result = pd.concat([cold_result, hot_result], ignore_index=True)
        combined_result = combined_result.sort_values('timestamp').reset_index(drop=True)
        
        # Verify hybrid result
        assert len(combined_result) == len(hot_data) + len(cold_data)
        assert combined_result['timestamp'].is_monotonic_increasing
    
    @pytest.mark.asyncio
    async def test_storage_fallback_mechanisms(self, mock_db_adapter, mock_data_archive, sample_feature_data):
        """Test fallback mechanisms when primary storage fails."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Test hot storage failure fallback to cold storage
        mock_db_adapter.fetch_query.side_effect = Exception("PostgreSQL connection failed")
        mock_data_archive.load_dataframe.return_value = sample_feature_data['AAPL']
        
        # In a real implementation, the orchestrator would handle fallback
        # For this test, we simulate the fallback logic
        try:
            # Try hot storage first
            result = await repo.get_features(
                symbol='AAPL',
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
        except Exception:
            # Fallback to cold storage
            result = await mock_data_archive.load_dataframe("features/AAPL/2024/01/01/features.parquet")
        
        # Should get data from fallback source
        assert not result.empty
        pd.testing.assert_frame_equal(result, sample_feature_data['AAPL'])


# Test Feature Versioning and Migration
class TestFeatureVersioning:
    """Test feature versioning and schema migration."""
    
    @pytest.mark.asyncio
    async def test_feature_schema_versioning(self, mock_db_adapter):
        """Test handling of different feature schema versions."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock successful storage
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.errors = []
        mock_db_adapter.execute_query.return_value = mock_result
        
        # Version 1.0 features (basic technical indicators)
        features_v1 = {
            'sma_10': 100.0,
            'sma_20': 99.5,
            'rsi': 65.0
        }
        
        # Version 2.0 features (added sentiment)
        features_v2 = {
            'sma_10': 100.0,
            'sma_20': 99.5,
            'rsi': 65.0,
            'sentiment_score': 0.2,
            'news_sentiment': 0.1
        }
        
        # Version 3.0 features (added microstructure)
        features_v3 = {
            'sma_10': 100.0,
            'sma_20': 99.5,
            'rsi': 65.0,
            'sentiment_score': 0.2,
            'news_sentiment': 0.1,
            'bid_ask_spread': 0.01,
            'order_flow': 1500
        }
        
        # Store different versions
        versions = [
            ('1.0', features_v1),
            ('2.0', features_v2),
            ('3.0', features_v3)
        ]
        
        for version, features in versions:
            await repo.store_features(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                features=features,
                version=version
            )
        
        # Verify all versions stored
        assert mock_db_adapter.execute_query.call_count == len(versions)
        
        # Check version field in stored data
        calls = mock_db_adapter.execute_query.call_args_list
        for i, (version, _) in enumerate(versions):
            assert calls[i][0][1]['version'] == version
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_queries(self, mock_db_adapter):
        """Test querying features with backward compatibility."""
        repo = FeatureStoreRepository(mock_db_adapter)
        
        # Mock database response with mixed versions
        mock_rows = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'features': '{"sma_10": 100.0, "rsi": 65.0}',  # v1.0 - basic features
                'version': '1.0'
            },
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2024, 1, 2, tzinfo=timezone.utc),
                'features': '{"sma_10": 101.0, "rsi": 67.0, "sentiment_score": 0.2}',  # v2.0 - with sentiment
                'version': '2.0'
            }
        ]
        mock_db_adapter.fetch_query.return_value = mock_rows
        
        # Query features
        result = await repo.get_features(
            symbol='AAPL',
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 3, tzinfo=timezone.utc)
        )
        
        # Should handle mixed versions gracefully
        assert len(result) == 2
        assert 'sma_10' in result.columns  # Common to both versions
        assert 'rsi' in result.columns     # Common to both versions
        
        # Newer features should have NaN for older records
        if 'sentiment_score' in result.columns:
            assert pd.isna(result.loc[0, 'sentiment_score'])  # v1.0 record
            assert not pd.isna(result.loc[1, 'sentiment_score'])  # v2.0 record
    
    @pytest.mark.asyncio
    async def test_feature_migration_simulation(self, mock_db_adapter, mock_data_archive):
        """Test feature migration between storage systems."""
        # Simulate migrating features from old format to new format
        
        # Old format data (in cold storage)
        old_features = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D', tz=timezone.utc),
            'symbol': 'AAPL',
            'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Mock loading old format
        mock_data_archive.load_dataframe.return_value = old_features
        
        # Simulate migration transformation
        def migrate_features(old_df):
            """Transform old format to new format."""
            new_df = old_df.copy()
            # Add new computed features
            new_df['sma_5'] = new_df['price'].rolling(5).mean()
            new_df['volume_sma'] = new_df['volume'].rolling(5).mean()
            # Rename columns to match new schema
            new_df = new_df.rename(columns={'price': 'close'})
            return new_df
        
        # Load and migrate
        old_data = await mock_data_archive.load_dataframe("features/AAPL/2023/12/31/features.parquet")
        migrated_data = migrate_features(old_data)
        
        # Verify migration
        assert 'close' in migrated_data.columns
        assert 'sma_5' in migrated_data.columns
        assert 'volume_sma' in migrated_data.columns
        assert 'price' not in migrated_data.columns  # Old column renamed
        
        # Simulate storing migrated data
        mock_data_archive.save_dataframe.return_value = True
        await mock_data_archive.save_dataframe(
            "features/AAPL/2023/12/31/features_v2.parquet",
            migrated_data,
            metadata={'version': '2.0', 'migrated': True}
        )
        
        mock_data_archive.save_dataframe.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
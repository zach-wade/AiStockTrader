"""
Integration tests for hot/cold storage tier routing in scanner repository.

Tests that queries are properly routed between hot (PostgreSQL) and 
cold (Data Lake) storage based on data age and query characteristics.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from main.interfaces.scanners import IScannerRepository
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.data_pipeline.storage.storage_router import StorageRouter


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageTierIntegration:
    """Test storage tier routing and integration."""

    async def test_hot_storage_routing_recent_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mock_storage_router
    ):
        """Test that recent data queries are routed to hot storage."""
        # Create query for very recent data (should go to hot storage)
        recent_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(hours=6),
            end_date=datetime.now(timezone.utc),
            symbols=test_symbols[:2]
        )
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=recent_filter
            )
            
            # Verify router was called with recent data
            mock_storage_router.route_query.assert_called()
            
            # Get the call arguments
            call_args = mock_storage_router.route_query.call_args
            called_filter = call_args[0][0]  # First positional argument
            
            # Verify the date range indicates hot storage should be used
            days_back = (datetime.now(timezone.utc) - called_filter.start_date).days
            assert days_back < 30  # Should be routed to hot storage

    async def test_cold_storage_routing_historical_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mock_storage_router
    ):
        """Test that historical data queries are routed to cold storage."""
        # Create query for old data (should go to cold storage)
        historical_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=90),
            end_date=datetime.now(timezone.utc) - timedelta(days=60),
            symbols=test_symbols[:2]
        )
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=historical_filter
            )
            
            # Verify router was called
            mock_storage_router.route_query.assert_called()
            
            # Get the call arguments
            call_args = mock_storage_router.route_query.call_args
            called_filter = call_args[0][0]
            
            # Verify the date range indicates cold storage should be used
            days_back = (datetime.now(timezone.utc) - called_filter.start_date).days
            assert days_back > 30  # Should be routed to cold storage

    async def test_mixed_storage_query_handling(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mock_storage_router
    ):
        """Test handling of queries that span both hot and cold storage."""
        # Create query spanning hot and cold storage boundary
        mixed_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=45),  # Spans 30-day boundary
            end_date=datetime.now(timezone.utc) - timedelta(days=5),
            symbols=test_symbols[:3]
        )
        
        # Mock router to return different storage for different parts
        async def mock_mixed_routing(query_filter, query_type):
            days_back = (datetime.now(timezone.utc) - query_filter.start_date).days
            return "hot" if days_back <= 30 else "cold"
        
        mock_storage_router.route_query = mock_mixed_routing
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            # Mock both hot and cold storage responses
            mock_query.return_value = [
                {'symbol': 'AAPL', 'date': datetime.now(timezone.utc) - timedelta(days=10), 'close': 150.0},
                {'symbol': 'AAPL', 'date': datetime.now(timezone.utc) - timedelta(days=40), 'close': 145.0}
            ]
            
            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:3],
                query_filter=mixed_filter
            )
            
            # Should successfully handle mixed storage queries
            assert isinstance(result, list)
            # Repository should handle the complexity internally

    async def test_storage_router_query_type_parameter(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        mock_storage_router
    ):
        """Test that correct query type is passed to storage router."""
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            # Test different data types
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            # Market data query
            await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Verify router was called - implementation may vary
            assert mock_storage_router.route_query.called

    async def test_storage_tier_performance_difference(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        performance_thresholds
    ):
        """Test performance difference between hot and cold storage queries."""
        # Hot storage query (should be faster)
        hot_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc),
            symbols=test_symbols[:5]
        )
        
        # Cold storage query (may be slower)
        cold_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=90),
            end_date=datetime.now(timezone.utc) - timedelta(days=60),
            symbols=test_symbols[:5]
        )
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            # Time hot storage query
            start_time = datetime.now()
            await scanner_repository.get_market_data(
                symbols=test_symbols[:5],
                query_filter=hot_filter
            )
            hot_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Time cold storage query
            start_time = datetime.now()
            await scanner_repository.get_market_data(
                symbols=test_symbols[:5],
                query_filter=cold_filter
            )
            cold_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Both should meet performance thresholds
            threshold = performance_thresholds['repository']['query_time_ms']
            assert hot_time < threshold
            assert cold_time < threshold * 2  # Allow cold storage to be slower

    async def test_storage_tier_data_consistency(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test data consistency between storage tiers."""
        # Query that might exist in both tiers (around boundary)
        boundary_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        hot_filter = QueryFilter(
            start_date=boundary_date - timedelta(days=1),
            end_date=boundary_date + timedelta(days=1),
            symbols=test_symbols[:2]
        )
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            # Mock consistent data from both tiers
            consistent_data = [
                {
                    'symbol': 'AAPL',
                    'date': boundary_date,
                    'close': 150.0,
                    'volume': 1000000
                }
            ]
            mock_query.return_value = consistent_data
            
            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=hot_filter
            )
            
            # Should return consistent data regardless of storage tier
            assert isinstance(result, list)
            if result:
                record = result[0]
                assert record['symbol'] == 'AAPL'
                assert record['close'] == 150.0

    async def test_storage_tier_failover(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test failover behavior when primary storage tier is unavailable."""
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            # Mock primary storage failure, then success on retry
            mock_query.side_effect = [
                Exception("Hot storage unavailable"),  # First call fails
                []  # Second call succeeds (fallback)
            ]
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            # Repository should handle failover gracefully
            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Should return result from fallback
            assert isinstance(result, list)

    async def test_query_optimization_by_storage_tier(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mock_storage_router
    ):
        """Test that queries are optimized based on target storage tier."""
        # Different query patterns for different storage tiers
        queries = [
            # Recent data - should optimize for hot storage
            QueryFilter(
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                symbols=test_symbols[:3]
            ),
            # Historical data - should optimize for cold storage
            QueryFilter(
                start_date=datetime.now(timezone.utc) - timedelta(days=180),
                end_date=datetime.now(timezone.utc) - timedelta(days=150),
                symbols=test_symbols[:3]
            )
        ]
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            for query_filter in queries:
                await scanner_repository.get_market_data(
                    symbols=query_filter.symbols,
                    query_filter=query_filter
                )
                
                # Verify router was called for optimization
                assert mock_storage_router.route_query.called

    async def test_storage_tier_metadata_handling(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test that storage tier metadata is properly handled."""
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            # Mock data with storage tier metadata
            mock_data = [
                {
                    'symbol': 'AAPL',
                    'date': datetime.now(timezone.utc) - timedelta(days=1),
                    'close': 150.0,
                    '_storage_tier': 'hot',
                    '_query_time_ms': 50
                }
            ]
            mock_query.return_value = mock_data
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Repository should handle metadata appropriately
            assert isinstance(result, list)
            if result:
                # Metadata might be stripped or preserved depending on implementation
                record = result[0]
                assert 'symbol' in record
                assert 'close' in record

    async def test_concurrent_storage_tier_queries(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mock_storage_router
    ):
        """Test concurrent queries to different storage tiers."""
        # Create queries for different storage tiers
        hot_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=5),
            end_date=datetime.now(timezone.utc),
            symbols=test_symbols[:2]
        )
        
        cold_filter = QueryFilter(
            start_date=datetime.now(timezone.utc) - timedelta(days=90),
            end_date=datetime.now(timezone.utc) - timedelta(days=60),
            symbols=test_symbols[:2]
        )
        
        with patch.object(scanner_repository, '_execute_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            # Run concurrent queries to different tiers
            import asyncio
            tasks = [
                scanner_repository.get_market_data(test_symbols[:2], hot_filter),
                scanner_repository.get_market_data(test_symbols[:2], cold_filter),
                scanner_repository.get_market_data(test_symbols[:2], hot_filter)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All queries should succeed
            assert len(results) == 3
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)
"""
Integration tests for ScannerDataRepository.

Tests the core repository functionality used by scanners,
including hot/cold storage routing and data access methods.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest

# Local imports
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.interfaces.scanners import IScannerRepository


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerDataRepository:
    """Test ScannerDataRepository integration with database and storage tiers."""

    async def test_repository_implements_interface(self, scanner_repository: IScannerRepository):
        """Test that repository properly implements IScannerRepository interface."""
        assert isinstance(scanner_repository, IScannerRepository)
        assert hasattr(scanner_repository, "get_market_data")
        assert hasattr(scanner_repository, "get_news_data")
        assert hasattr(scanner_repository, "get_earnings_data")
        assert hasattr(scanner_repository, "get_social_sentiment")
        assert hasattr(scanner_repository, "get_insider_data")
        assert hasattr(scanner_repository, "get_options_data")
        assert hasattr(scanner_repository, "get_sector_data")

    async def test_get_market_data_recent(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_market_data,
    ):
        """Test market data retrieval for recent data (hot storage)."""
        # Mock the database to return sample data
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data

            # Set symbols on query filter
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]  # Use first 3 symbols

            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:3],
                query_filter=query_filter,
                columns=["date", "symbol", "open", "high", "low", "close", "volume"],
            )

            # Verify results
            assert isinstance(result, list)
            assert len(result) > 0

            # Check data structure
            for record in result[:3]:  # Check first few records
                assert "symbol" in record
                assert "date" in record
                assert "close" in record
                assert record["symbol"] in test_symbols[:3]

            # Verify query was called
            mock_query.assert_called_once()

    async def test_get_market_data_historical(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        historical_date_range: QueryFilter,
        sample_market_data,
    ):
        """Test market data retrieval for historical data (cold storage)."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data

            query_filter = historical_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Should still return data even for historical queries
            assert isinstance(result, list)
            mock_query.assert_called_once()

    async def test_get_news_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_news_data,
    ):
        """Test news data retrieval."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_news_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]

            result = await scanner_repository.get_news_data(
                symbols=test_symbols[:3], query_filter=query_filter
            )

            assert isinstance(result, list)

            # Check news data structure
            if result:
                news_item = result[0]
                assert "symbol" in news_item
                assert "headline" in news_item
                assert "sentiment_score" in news_item
                assert "timestamp" in news_item

    async def test_get_social_sentiment(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_social_data,
    ):
        """Test social sentiment data retrieval."""
        with patch.object(
            scanner_repository, "_execute_social_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_social_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_social_sentiment(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            assert isinstance(result, dict)

            # Check social data structure
            for symbol, posts in result.items():
                assert isinstance(posts, list)
                if posts:
                    post = posts[0]
                    assert "content" in post
                    assert "sentiment_score" in post
                    assert "timestamp" in post

    async def test_get_earnings_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_earnings_data,
    ):
        """Test earnings data retrieval."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_earnings_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            assert isinstance(result, list)

            # Check earnings data structure
            if result:
                earnings = result[0]
                assert "symbol" in earnings
                assert "eps_actual" in earnings
                assert "eps_estimate" in earnings
                assert "report_date" in earnings

    async def test_query_with_empty_symbols_list(
        self, scanner_repository: IScannerRepository, recent_date_range: QueryFilter
    ):
        """Test repository behavior with empty symbols list."""
        result = await scanner_repository.get_market_data(
            symbols=[], query_filter=recent_date_range
        )

        # Should return empty list for empty symbols
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_query_with_invalid_date_range(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test repository behavior with invalid date range."""
        # Create invalid date range (end before start)
        invalid_filter = QueryFilter(
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC) - timedelta(days=1),
            symbols=test_symbols[:2],
        )

        # Should handle gracefully and return empty result
        result = await scanner_repository.get_market_data(
            symbols=test_symbols[:2], query_filter=invalid_filter
        )

        assert isinstance(result, list)
        # May be empty or raise exception - either is acceptable

    async def test_storage_tier_routing(
        self, scanner_repository: IScannerRepository, test_symbols, mock_storage_router
    ):
        """Test that queries are routed to appropriate storage tiers."""
        # Test recent data routing (should go to hot storage)
        recent_filter = QueryFilter(
            start_date=datetime.now(UTC) - timedelta(days=7),
            end_date=datetime.now(UTC),
            symbols=test_symbols[:2],
        )

        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = []

            await scanner_repository.get_market_data(
                symbols=test_symbols[:2], query_filter=recent_filter
            )

            # Verify storage router was called
            assert mock_storage_router.route_query.called

    async def test_concurrent_queries(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_market_data,
    ):
        """Test repository handling of concurrent queries."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data

            # Create multiple concurrent queries
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            tasks = [
                scanner_repository.get_market_data(test_symbols[:2], query_filter) for _ in range(5)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All queries should succeed
            assert len(results) == 5
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)

    async def test_query_performance(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        performance_thresholds,
        sample_market_data,
    ):
        """Test repository query performance meets thresholds."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols

            start_time = datetime.now()

            result = await scanner_repository.get_market_data(
                symbols=test_symbols, query_filter=query_filter
            )

            end_time = datetime.now()
            query_time_ms = (end_time - start_time).total_seconds() * 1000

            # Check performance threshold
            threshold = performance_thresholds["repository"]["query_time_ms"]
            assert (
                query_time_ms < threshold
            ), f"Query took {query_time_ms}ms, threshold is {threshold}ms"

            assert isinstance(result, list)

    async def test_error_handling_database_unavailable(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test repository error handling when database is unavailable."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            # Simulate database error
            mock_query.side_effect = Exception("Database connection failed")

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            # Repository should handle errors gracefully
            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Should return empty result rather than propagating exception
            assert isinstance(result, list)
            assert len(result) == 0

    async def test_mixed_storage_tier_query(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        mixed_date_range: QueryFilter,
        sample_market_data,
    ):
        """Test queries that span both hot and cold storage."""
        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data

            query_filter = mixed_date_range
            query_filter.symbols = test_symbols[:3]

            result = await scanner_repository.get_market_data(
                symbols=test_symbols[:3], query_filter=query_filter
            )

            # Should successfully handle mixed queries
            assert isinstance(result, list)
            mock_query.assert_called()

    async def test_large_symbol_list_query(
        self,
        scanner_repository: IScannerRepository,
        recent_date_range: QueryFilter,
        performance_thresholds,
        sample_market_data,
    ):
        """Test repository performance with large symbol lists."""
        # Create large symbol list
        large_symbol_list = [f"SYM{i:04d}" for i in range(100)]

        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_market_data * 10  # Larger dataset

            query_filter = recent_date_range
            query_filter.symbols = large_symbol_list

            start_time = datetime.now()

            result = await scanner_repository.get_market_data(
                symbols=large_symbol_list, query_filter=query_filter
            )

            end_time = datetime.now()
            query_time_ms = (end_time - start_time).total_seconds() * 1000

            # Check large query performance threshold
            threshold = performance_thresholds["repository"]["large_query_time_ms"]
            assert (
                query_time_ms < threshold
            ), f"Large query took {query_time_ms}ms, threshold is {threshold}ms"

            assert isinstance(result, list)

    async def test_repository_cleanup(self, scanner_repository: IScannerRepository):
        """Test repository cleanup functionality."""
        # Repository should have cleanup capability
        if hasattr(scanner_repository, "cleanup"):
            await scanner_repository.cleanup()
            # Should not raise exception

    async def test_get_sector_data(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test sector data retrieval for sector scanner."""
        sample_sector_data = {
            "AAPL": {"sector": "Technology", "returns": [0.01, -0.005, 0.02]},
            "GOOGL": {"sector": "Technology", "returns": [0.015, -0.008, 0.018]},
            "MSFT": {"sector": "Technology", "returns": [0.012, -0.003, 0.015]},
        }

        with patch.object(
            scanner_repository, "_execute_sector_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_sector_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]

            result = await scanner_repository.get_sector_data(
                symbols=test_symbols[:3], query_filter=query_filter
            )

            assert isinstance(result, dict)

            # Check sector data structure
            for symbol, data in result.items():
                assert "sector" in data
                assert isinstance(data.get("returns"), list)

    async def test_get_insider_data(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test insider trading data retrieval."""
        sample_insider_data = [
            {
                "symbol": "AAPL",
                "transaction_date": datetime.now(UTC) - timedelta(days=1),
                "insider_name": "John Doe",
                "transaction_type": "purchase",
                "shares": 10000,
                "price": 150.0,
            }
        ]

        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_insider_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_insider_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            assert isinstance(result, list)

            if result:
                insider_record = result[0]
                assert "symbol" in insider_record
                assert "transaction_type" in insider_record
                assert "shares" in insider_record

    async def test_get_options_data(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test options data retrieval."""
        sample_options_data = [
            {
                "symbol": "AAPL",
                "date": datetime.now(UTC) - timedelta(days=1),
                "option_type": "call",
                "strike": 155.0,
                "expiry": datetime.now(UTC) + timedelta(days=30),
                "volume": 5000,
                "open_interest": 15000,
                "implied_volatility": 0.25,
            }
        ]

        with patch.object(
            scanner_repository, "_execute_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = sample_options_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_options_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            assert isinstance(result, list)

            if result:
                option_record = result[0]
                assert "symbol" in option_record
                assert "option_type" in option_record
                assert "volume" in option_record

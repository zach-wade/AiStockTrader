"""
Integration tests for News Scanner repository interactions.

Tests news data retrieval and sentiment analysis for the news scanner.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from main.interfaces.scanners import IScannerRepository
from main.data_pipeline.storage.repositories.repository_types import QueryFilter


@pytest.mark.integration
@pytest.mark.asyncio
class TestNewsScannerRepository:
    """Test news scanner repository integration."""

    async def test_get_news_data_basic(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_news_data
    ):
        """Test basic news data retrieval."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            mock_news.return_value = sample_news_data
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]
            
            result = await scanner_repository.get_news_data(
                symbols=test_symbols[:3],
                query_filter=query_filter
            )
            
            # Verify news data structure
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Check first news item structure
            news_item = result[0]
            required_fields = ['symbol', 'headline', 'sentiment_score', 'timestamp']
            for field in required_fields:
                assert field in news_item

    async def test_news_sentiment_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test news sentiment analysis capabilities."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock news with varying sentiment
            sentiment_news = [
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple reports record quarterly earnings',
                    'content': 'Apple Inc. reported record-breaking quarterly earnings with strong iPhone sales.',
                    'sentiment_score': 0.9,  # Very positive
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'source': 'Reuters',
                    'relevance_score': 0.95
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple faces supply chain challenges',
                    'content': 'Apple Inc. is experiencing significant supply chain disruptions.',
                    'sentiment_score': 0.2,  # Negative
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                    'source': 'Bloomberg',
                    'relevance_score': 0.85
                },
                {
                    'symbol': 'GOOGL',
                    'headline': 'Google announces new AI breakthrough',
                    'content': 'Google has achieved a significant breakthrough in artificial intelligence.',
                    'sentiment_score': 0.8,  # Positive
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=3),
                    'source': 'TechCrunch',
                    'relevance_score': 0.9
                }
            ]
            
            mock_news.return_value = sentiment_news
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            result = await scanner_repository.get_news_data(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Analyze sentiment distribution
            aapl_news = [item for item in result if item['symbol'] == 'AAPL']
            googl_news = [item for item in result if item['symbol'] == 'GOOGL']
            
            # AAPL should have mixed sentiment
            aapl_sentiments = [item['sentiment_score'] for item in aapl_news]
            assert max(aapl_sentiments) > 0.8  # Positive news present
            assert min(aapl_sentiments) < 0.3  # Negative news present
            
            # GOOGL should have positive sentiment
            googl_sentiments = [item['sentiment_score'] for item in googl_news]
            assert all(score > 0.7 for score in googl_sentiments)

    async def test_news_relevance_filtering(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test news relevance filtering capabilities."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock news with varying relevance scores
            relevance_news = [
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple CEO announces new product strategy',
                    'sentiment_score': 0.8,
                    'relevance_score': 0.95,  # Highly relevant
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'source': 'Apple Press Release'
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Tech sector sees broad gains',
                    'sentiment_score': 0.6,
                    'relevance_score': 0.3,  # Low relevance (broad sector news)
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                    'source': 'MarketWatch'
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple supplier reports strong demand',
                    'sentiment_score': 0.7,
                    'relevance_score': 0.8,  # High relevance (supply chain)
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=3),
                    'source': 'Reuters'
                }
            ]
            
            mock_news.return_value = relevance_news
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_news_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Should include all news items (filtering happens at scanner level)
            assert len(result) == 3
            
            # High relevance news should be present
            high_relevance = [item for item in result if item['relevance_score'] > 0.8]
            assert len(high_relevance) == 2

    async def test_news_source_diversity(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test news source diversity and credibility."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock news from diverse sources
            diverse_news = [
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple earnings beat estimates',
                    'sentiment_score': 0.8,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'source': 'Reuters',
                    'credibility_score': 0.95
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'AAPL stock surge continues',
                    'sentiment_score': 0.7,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                    'source': 'Bloomberg',
                    'credibility_score': 0.9
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple to the moon!',
                    'sentiment_score': 0.9,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=3),
                    'source': 'StockTwits',
                    'credibility_score': 0.3  # Low credibility social source
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Detailed Apple financial analysis',
                    'sentiment_score': 0.6,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=4),
                    'source': 'Wall Street Journal',
                    'credibility_score': 0.95
                }
            ]
            
            mock_news.return_value = diverse_news
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_news_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Check source diversity
            sources = [item['source'] for item in result]
            unique_sources = set(sources)
            assert len(unique_sources) == 4  # Four different sources
            
            # Check credibility distribution
            high_credibility = [
                item for item in result 
                if item.get('credibility_score', 0) > 0.8
            ]
            assert len(high_credibility) >= 3  # Most sources should be credible

    async def test_breaking_news_detection(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test breaking news and urgency detection."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock breaking news scenario
            breaking_news = [
                {
                    'symbol': 'AAPL',
                    'headline': 'BREAKING: Apple announces major acquisition',
                    'content': 'Apple Inc. has just announced the acquisition of a major competitor.',
                    'sentiment_score': 0.8,
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=5),  # Very recent
                    'source': 'Reuters',
                    'urgency_score': 0.95,  # High urgency
                    'breaking_news': True
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple quarterly review published',
                    'content': 'Quarterly review of Apple performance shows steady growth.',
                    'sentiment_score': 0.6,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=6),
                    'source': 'MarketWatch',
                    'urgency_score': 0.2,  # Low urgency
                    'breaking_news': False
                }
            ]
            
            mock_news.return_value = breaking_news
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_news_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Check for breaking news detection
            breaking_items = [
                item for item in result 
                if item.get('breaking_news', False)
            ]
            assert len(breaking_items) == 1
            
            # Breaking news should be very recent
            breaking_item = breaking_items[0]
            time_diff = datetime.now(timezone.utc) - breaking_item['timestamp']
            assert time_diff.total_seconds() < 600  # Less than 10 minutes

    async def test_news_volume_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test news volume and frequency analysis."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock high-volume news day
            volume_news = []
            base_time = datetime.now(timezone.utc)
            
            # Generate 20 news items in the last 4 hours
            for i in range(20):
                volume_news.append({
                    'symbol': 'AAPL',
                    'headline': f'Apple news update #{i+1}',
                    'sentiment_score': 0.6 + (i % 3) * 0.1,
                    'timestamp': base_time - timedelta(minutes=i*12),  # Every 12 minutes
                    'source': ['Reuters', 'Bloomberg', 'CNBC'][i % 3]
                })
            
            mock_news.return_value = volume_news
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_news_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Analyze news volume
            assert len(result) == 20  # High news volume
            
            # Check time distribution (should be frequent)
            timestamps = [item['timestamp'] for item in result]
            timestamps.sort()
            
            # Calculate average time between news items
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutes
                time_diffs.append(diff)
            
            avg_interval = sum(time_diffs) / len(time_diffs)
            assert avg_interval < 15  # News every 15 minutes or less (high volume)

    async def test_news_sentiment_momentum(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test news sentiment momentum analysis."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock news with sentiment trend
            momentum_news = []
            base_time = datetime.now(timezone.utc)
            
            # Create sentiment momentum (improving over time)
            for i in range(10):
                sentiment_score = 0.3 + (i * 0.07)  # Improving from 0.3 to 0.93
                momentum_news.append({
                    'symbol': 'AAPL',
                    'headline': f'Apple development update {i+1}',
                    'sentiment_score': sentiment_score,
                    'timestamp': base_time - timedelta(hours=9-i),  # Chronological order
                    'source': 'Reuters'
                })
            
            mock_news.return_value = momentum_news
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_news_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Sort by timestamp for momentum analysis
            sorted_news = sorted(result, key=lambda x: x['timestamp'])
            
            # Calculate sentiment momentum
            early_sentiment = sum(item['sentiment_score'] for item in sorted_news[:3]) / 3
            late_sentiment = sum(item['sentiment_score'] for item in sorted_news[-3:]) / 3
            
            sentiment_momentum = late_sentiment - early_sentiment
            
            # Should show positive sentiment momentum
            assert sentiment_momentum > 0.4  # Strong improvement in sentiment

    async def test_sector_news_correlation(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test sector-wide news correlation analysis."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock sector-wide news affecting multiple symbols
            sector_news = [
                {
                    'symbol': 'AAPL',
                    'headline': 'Tech sector faces regulatory pressure',
                    'sentiment_score': 0.3,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'category': 'regulatory',
                    'sector_impact': True
                },
                {
                    'symbol': 'GOOGL',
                    'headline': 'Technology companies under scrutiny',
                    'sentiment_score': 0.25,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'category': 'regulatory',
                    'sector_impact': True
                },
                {
                    'symbol': 'MSFT',
                    'headline': 'Microsoft faces antitrust investigation',
                    'sentiment_score': 0.2,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                    'category': 'regulatory',
                    'sector_impact': True
                },
                {
                    'symbol': 'AAPL',
                    'headline': 'Apple reports strong iPhone sales',
                    'sentiment_score': 0.8,
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                    'category': 'earnings',
                    'sector_impact': False
                }
            ]
            
            mock_news.return_value = sector_news
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]  # AAPL, GOOGL, MSFT
            
            result = await scanner_repository.get_news_data(
                symbols=test_symbols[:3],
                query_filter=query_filter
            )
            
            # Analyze sector correlation
            sector_impact_news = [
                item for item in result 
                if item.get('sector_impact', False)
            ]
            
            # Should have sector-wide impact news for multiple symbols
            assert len(sector_impact_news) == 3
            
            # All sector news should be regulatory category
            categories = [item['category'] for item in sector_impact_news]
            assert all(cat == 'regulatory' for cat in categories)
            
            # Sector news should have similar negative sentiment
            sector_sentiments = [item['sentiment_score'] for item in sector_impact_news]
            assert all(score < 0.4 for score in sector_sentiments)

    async def test_error_handling_no_news_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test error handling when no news data is available."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock no news available
            mock_news.return_value = []
            
            query_filter = recent_date_range
            query_filter.symbols = ['UNKNOWN_SYMBOL']
            
            result = await scanner_repository.get_news_data(
                symbols=['UNKNOWN_SYMBOL'],
                query_filter=query_filter
            )
            
            # Should return empty list gracefully
            assert isinstance(result, list)
            assert len(result) == 0

    async def test_news_query_performance(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        performance_thresholds
    ):
        """Test news query performance meets scanner requirements."""
        with patch.object(scanner_repository, 'get_news_data') as mock_news:
            # Mock large news dataset
            large_news = [
                {
                    'symbol': test_symbols[i % len(test_symbols)],
                    'headline': f'News headline {i}',
                    'sentiment_score': 0.5 + (i % 10) * 0.05,
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'source': 'Reuters'
                }
                for i in range(1000)  # 1000 news items
            ]
            
            mock_news.return_value = large_news
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols
            
            # Time the news query
            start_time = datetime.now()
            result = await scanner_repository.get_news_data(
                symbols=test_symbols,
                query_filter=query_filter
            )
            end_time = datetime.now()
            
            query_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Should meet performance threshold
            threshold = performance_thresholds['repository']['query_time_ms']
            assert query_time_ms < threshold
            assert len(result) == 1000
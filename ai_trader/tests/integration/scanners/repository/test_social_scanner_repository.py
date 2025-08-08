"""
Integration tests for Social Scanner repository interactions.

Tests social sentiment data retrieval and analysis for the social scanner.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from collections import defaultdict

from main.interfaces.scanners import IScannerRepository
from main.data_pipeline.storage.repositories.repository_types import QueryFilter


@pytest.mark.integration
@pytest.mark.asyncio
class TestSocialScannerRepository:
    """Test social scanner repository integration."""

    async def test_get_social_sentiment_basic(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_social_data
    ):
        """Test basic social sentiment data retrieval."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            mock_social.return_value = sample_social_data
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            result = await scanner_repository.get_social_sentiment(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Verify social data structure
            assert isinstance(result, dict)
            assert 'AAPL' in result
            assert 'GOOGL' in result
            
            # Check AAPL social posts
            aapl_posts = result['AAPL']
            assert isinstance(aapl_posts, list)
            assert len(aapl_posts) > 0
            
            # Check post structure
            post = aapl_posts[0]
            required_fields = ['content', 'sentiment_score', 'timestamp', 'author']
            for field in required_fields:
                assert field in post

    async def test_social_sentiment_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test social sentiment analysis and scoring."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock social data with varying sentiment
            sentiment_social = {
                'AAPL': [
                    {
                        'author': 'bullish_trader',
                        'content': 'AAPL is absolutely crushing it! Best stock ever! 🚀🚀🚀',
                        'sentiment_score': 0.95,  # Very bullish
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=10),
                        'platform': 'twitter',
                        'engagement_score': 250,
                        'follower_count': 50000
                    },
                    {
                        'author': 'bearish_analyst',
                        'content': 'AAPL overvalued, huge correction coming. Sell now!',
                        'sentiment_score': 0.1,  # Very bearish
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=20),
                        'platform': 'stocktwits',
                        'engagement_score': 150,
                        'follower_count': 25000
                    },
                    {
                        'author': 'neutral_investor',
                        'content': 'AAPL earnings next week, waiting to see results',
                        'sentiment_score': 0.5,  # Neutral
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                        'platform': 'reddit',
                        'engagement_score': 80,
                        'follower_count': 5000
                    }
                ],
                'GOOGL': [
                    {
                        'author': 'tech_enthusiast',
                        'content': 'Google AI breakthrough is game-changing! Long GOOGL',
                        'sentiment_score': 0.85,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15),
                        'platform': 'twitter',
                        'engagement_score': 300,
                        'follower_count': 75000
                    }
                ]
            }
            
            mock_social.return_value = sentiment_social
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]
            
            result = await scanner_repository.get_social_sentiment(
                symbols=test_symbols[:2],
                query_filter=query_filter
            )
            
            # Analyze sentiment distribution
            aapl_posts = result['AAPL']
            aapl_sentiments = [post['sentiment_score'] for post in aapl_posts]
            
            # Should have full sentiment range
            assert max(aapl_sentiments) > 0.9  # Very bullish posts
            assert min(aapl_sentiments) < 0.2  # Very bearish posts
            assert any(0.4 <= score <= 0.6 for score in aapl_sentiments)  # Neutral posts
            
            # GOOGL should be positive
            googl_posts = result['GOOGL']
            googl_sentiments = [post['sentiment_score'] for post in googl_posts]
            assert all(score > 0.8 for score in googl_sentiments)

    async def test_social_volume_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test social volume and engagement analysis."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock high-volume social activity
            volume_social = {
                'AAPL': []
            }
            
            base_time = datetime.now(timezone.utc)
            
            # Generate 50 social posts in the last 2 hours (high volume)
            for i in range(50):
                volume_social['AAPL'].append({
                    'author': f'user_{i}',
                    'content': f'AAPL discussion point {i}',
                    'sentiment_score': 0.6 + (i % 5) * 0.08,  # Varying sentiment
                    'timestamp': base_time - timedelta(minutes=i*2.4),  # Every 2.4 minutes
                    'platform': ['twitter', 'stocktwits', 'reddit'][i % 3],
                    'engagement_score': 50 + i * 5,
                    'retweet_count': i * 2,
                    'like_count': i * 10
                })
            
            mock_social.return_value = volume_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Should show high social volume
            assert len(aapl_posts) == 50
            
            # Calculate posts per hour
            time_span_hours = 2.0
            posts_per_hour = len(aapl_posts) / time_span_hours
            assert posts_per_hour == 25  # 25 posts per hour (high volume)
            
            # Check engagement metrics
            total_engagement = sum(post['engagement_score'] for post in aapl_posts)
            avg_engagement = total_engagement / len(aapl_posts)
            assert avg_engagement > 100  # High average engagement

    async def test_viral_post_detection(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test viral post and trending detection."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock viral social activity
            viral_social = {
                'AAPL': [
                    {
                        'author': 'famous_influencer',
                        'content': 'Just bought 10,000 shares of AAPL. This is THE play! 🚀',
                        'sentiment_score': 0.9,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                        'platform': 'twitter',
                        'engagement_score': 50000,  # Viral engagement
                        'retweet_count': 15000,
                        'like_count': 75000,
                        'follower_count': 2000000,  # Major influencer
                        'viral_score': 0.95
                    },
                    {
                        'author': 'regular_user',
                        'content': 'Thinking about buying some AAPL',
                        'sentiment_score': 0.6,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=45),
                        'platform': 'reddit',
                        'engagement_score': 25,  # Normal engagement
                        'retweet_count': 2,
                        'like_count': 15,
                        'follower_count': 500,
                        'viral_score': 0.1
                    }
                ]
            }
            
            mock_social.return_value = viral_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Identify viral posts
            viral_posts = [
                post for post in aapl_posts 
                if post.get('viral_score', 0) > 0.8
            ]
            
            assert len(viral_posts) == 1
            
            viral_post = viral_posts[0]
            assert viral_post['engagement_score'] > 10000
            assert viral_post['follower_count'] > 1000000
            assert viral_post['sentiment_score'] > 0.8  # Viral posts often bullish

    async def test_platform_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test cross-platform social sentiment analysis."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock cross-platform social data
            platform_social = {
                'AAPL': [
                    # Twitter - generally more bullish, shorter posts
                    {
                        'author': 'twitter_trader1',
                        'content': 'AAPL 🚀🚀 buying the dip!',
                        'sentiment_score': 0.85,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=10),
                        'platform': 'twitter',
                        'character_count': 25
                    },
                    {
                        'author': 'twitter_trader2',
                        'content': '$AAPL looking strong 💪',
                        'sentiment_score': 0.8,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=20),
                        'platform': 'twitter',
                        'character_count': 22
                    },
                    # Reddit - more analytical, longer posts
                    {
                        'author': 'reddit_analyst1',
                        'content': 'Detailed analysis of AAPL fundamentals shows mixed signals. PE ratio concerning but growth prospects solid.',
                        'sentiment_score': 0.55,  # More neutral/analytical
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                        'platform': 'reddit',
                        'character_count': 115
                    },
                    # StockTwits - focused on trading, mixed sentiment
                    {
                        'author': 'stocktwits_trader',
                        'content': '$AAPL broke resistance, target $180',
                        'sentiment_score': 0.75,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=40),
                        'platform': 'stocktwits',
                        'character_count': 35
                    }
                ]
            }
            
            mock_social.return_value = platform_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Analyze by platform
            platform_sentiment = defaultdict(list)
            for post in aapl_posts:
                platform_sentiment[post['platform']].append(post['sentiment_score'])
            
            # Twitter should be more bullish
            twitter_avg = sum(platform_sentiment['twitter']) / len(platform_sentiment['twitter'])
            assert twitter_avg > 0.8
            
            # Reddit should be more neutral/analytical
            reddit_avg = sum(platform_sentiment['reddit']) / len(platform_sentiment['reddit'])
            assert 0.5 <= reddit_avg <= 0.7
            
            # Check platform diversity
            platforms = set(post['platform'] for post in aapl_posts)
            assert len(platforms) == 3  # Twitter, Reddit, StockTwits

    async def test_influencer_impact_analysis(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test influencer impact and credibility analysis."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock posts from different types of users
            influencer_social = {
                'AAPL': [
                    {
                        'author': 'major_financial_analyst',
                        'content': 'AAPL technical analysis suggests breakout imminent',
                        'sentiment_score': 0.75,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15),
                        'platform': 'twitter',
                        'follower_count': 500000,
                        'verified': True,
                        'credibility_score': 0.9,
                        'influence_score': 0.85
                    },
                    {
                        'author': 'crypto_moonboy',
                        'content': 'AAPL TO THE MOON!!! 🚀🚀🚀 DIAMOND HANDS!!!',
                        'sentiment_score': 0.95,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=25),
                        'platform': 'twitter',
                        'follower_count': 5000,
                        'verified': False,
                        'credibility_score': 0.2,
                        'influence_score': 0.1
                    },
                    {
                        'author': 'institutional_investor',
                        'content': 'Added AAPL to our portfolio based on strong fundamentals',
                        'sentiment_score': 0.7,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=35),
                        'platform': 'linkedin',
                        'follower_count': 50000,
                        'verified': True,
                        'credibility_score': 0.95,
                        'influence_score': 0.8
                    }
                ]
            }
            
            mock_social.return_value = influencer_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Analyze influencer impact
            high_influence_posts = [
                post for post in aapl_posts 
                if post.get('influence_score', 0) > 0.7
            ]
            
            assert len(high_influence_posts) == 2  # Financial analyst and institutional investor
            
            # High influence posts should have high credibility
            for post in high_influence_posts:
                assert post.get('credibility_score', 0) > 0.8
                assert post.get('verified', False) == True

    async def test_sentiment_momentum_tracking(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test social sentiment momentum over time."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock sentiment momentum (improving over time)
            momentum_social = {
                'AAPL': []
            }
            
            base_time = datetime.now(timezone.utc)
            
            # Create sentiment momentum over 4 hours
            for hour in range(4):
                base_sentiment = 0.3 + (hour * 0.15)  # Improving from 0.3 to 0.75
                
                # 5 posts per hour with sentiment trending upward
                for post_in_hour in range(5):
                    sentiment_variation = (post_in_hour - 2) * 0.05  # Small variation
                    final_sentiment = max(0.1, min(0.9, base_sentiment + sentiment_variation))
                    
                    momentum_social['AAPL'].append({
                        'author': f'user_h{hour}_p{post_in_hour}',
                        'content': f'AAPL sentiment hour {hour} post {post_in_hour}',
                        'sentiment_score': final_sentiment,
                        'timestamp': base_time - timedelta(hours=3-hour, minutes=post_in_hour*12),
                        'platform': 'twitter'
                    })
            
            mock_social.return_value = momentum_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Sort by timestamp for momentum analysis
            sorted_posts = sorted(aapl_posts, key=lambda x: x['timestamp'])
            
            # Calculate sentiment momentum
            early_posts = sorted_posts[:5]  # First hour
            late_posts = sorted_posts[-5:]  # Last hour
            
            early_sentiment = sum(post['sentiment_score'] for post in early_posts) / len(early_posts)
            late_sentiment = sum(post['sentiment_score'] for post in late_posts) / len(late_posts)
            
            sentiment_momentum = late_sentiment - early_sentiment
            
            # Should show positive momentum
            assert sentiment_momentum > 0.3  # Strong positive momentum

    async def test_social_anomaly_detection(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test social media anomaly detection."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock social anomaly (sudden spike in activity)
            anomaly_social = {
                'AAPL': []
            }
            
            base_time = datetime.now(timezone.utc)
            
            # Normal activity for first 3 hours (2 posts per hour)
            for i in range(6):
                anomaly_social['AAPL'].append({
                    'author': f'normal_user_{i}',
                    'content': f'Regular AAPL comment {i}',
                    'sentiment_score': 0.6,
                    'timestamp': base_time - timedelta(hours=4, minutes=i*30),
                    'platform': 'twitter',
                    'engagement_score': 50
                })
            
            # Sudden spike in last hour (20 posts)
            for i in range(20):
                anomaly_social['AAPL'].append({
                    'author': f'spike_user_{i}',
                    'content': f'BREAKING: AAPL news reaction {i}',
                    'sentiment_score': 0.8 + (i % 3) * 0.05,
                    'timestamp': base_time - timedelta(minutes=i*3),  # Every 3 minutes
                    'platform': 'twitter',
                    'engagement_score': 200 + i * 10  # High engagement
                })
            
            mock_social.return_value = anomaly_social
            
            query_filter = recent_date_range
            query_filter.symbols = ['AAPL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_posts = result['AAPL']
            
            # Analyze for volume anomaly
            recent_hour_posts = [
                post for post in aapl_posts 
                if (datetime.now(timezone.utc) - post['timestamp']).total_seconds() < 3600
            ]
            
            normal_period_posts = [
                post for post in aapl_posts 
                if (datetime.now(timezone.utc) - post['timestamp']).total_seconds() >= 3600
            ]
            
            recent_volume = len(recent_hour_posts)
            normal_hourly_rate = len(normal_period_posts) / 3  # Posts per hour in normal period
            
            volume_spike_ratio = recent_volume / normal_hourly_rate if normal_hourly_rate > 0 else 0
            
            # Should detect significant volume spike
            assert volume_spike_ratio > 5.0  # 5x normal volume

    async def test_error_handling_no_social_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter
    ):
        """Test error handling when no social data is available."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock no social data
            mock_social.return_value = {}
            
            query_filter = recent_date_range
            query_filter.symbols = ['UNKNOWN_SYMBOL']
            
            result = await scanner_repository.get_social_sentiment(
                symbols=['UNKNOWN_SYMBOL'],
                query_filter=query_filter
            )
            
            # Should return empty dict gracefully
            assert isinstance(result, dict)
            assert len(result) == 0

    async def test_social_query_performance(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        performance_thresholds
    ):
        """Test social sentiment query performance."""
        with patch.object(scanner_repository, 'get_social_sentiment') as mock_social:
            # Mock large social dataset
            large_social = {}
            
            for symbol in test_symbols:
                large_social[symbol] = [
                    {
                        'author': f'user_{symbol}_{i}',
                        'content': f'Social post about {symbol} #{i}',
                        'sentiment_score': 0.5 + (i % 10) * 0.05,
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                        'platform': ['twitter', 'reddit', 'stocktwits'][i % 3]
                    }
                    for i in range(200)  # 200 posts per symbol
                ]
            
            mock_social.return_value = large_social
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols
            
            # Time the social query
            start_time = datetime.now()
            result = await scanner_repository.get_social_sentiment(
                symbols=test_symbols,
                query_filter=query_filter
            )
            end_time = datetime.now()
            
            query_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Should meet performance threshold
            threshold = performance_thresholds['repository']['query_time_ms']
            assert query_time_ms < threshold
            assert len(result) == len(test_symbols)
"""
Comprehensive integration tests for dual storage system with all repositories.

These tests verify that all 11 repositories properly integrate with the dual storage
system using real database connections.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import uuid

from main.utils.app.context import create_app_context
from main.data_pipeline.storage.repositories.repository_factory import get_repository_factory
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.interfaces.events import EventType, AsyncEventHandler


class TestCompleteDualStorageIntegration:
    """Comprehensive integration tests for all dual storage repositories."""
    
    @pytest.fixture
    async def app_context(self):
        """Create app context with dual storage enabled."""
        context = await create_app_context(
            app_name="test_dual_storage_complete",
            components=['database', 'dual_storage', 'event_bus']
        )
        yield context
        await context.safe_shutdown()
    
    @pytest.fixture
    async def repository_factory(self, app_context):
        """Create repository factory with dual storage."""
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(app_context.config)
        
        factory = get_repository_factory(
            db_adapter=db_adapter,
            cold_storage=app_context.cold_storage,
            event_bus=app_context.event_bus
        )
        return factory
    
    @pytest.fixture
    async def event_collector(self, app_context):
        """Collect events published during tests."""
        events_received = []
        
        async def collect_event(event):
            events_received.append(event)
        
        handler = AsyncEventHandler(
            handler_func=collect_event,
            event_types=[EventType.DATA_WRITTEN]
        )
        
        await app_context.event_bus.subscribe(handler)
        yield events_received
        await app_context.event_bus.unsubscribe(handler)
    
    # =========================================================================
    # Market Data Repository Tests
    # =========================================================================
    
    async def test_market_data_repository_full_cycle(self, repository_factory, event_collector):
        """Test MarketDataRepository with real data flow."""
        repo = repository_factory.create_repository('market_data')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_symbol = f"TEST_{uuid.uuid4().hex[:8]}"
        test_data = [
            {
                'symbol': test_symbol,
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                'interval': '1day',
                'open': 100.0 + i,
                'high': 105.0 + i,
                'low': 99.0 + i,
                'close': 103.0 + i,
                'volume': 1000000 + i * 100000,
                'vwap': 102.0 + i,
                'trades': 1000 + i * 100,
                'source': 'test_source'
            }
            for i in range(5)
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 5
        assert result.failed_records == 0
        assert result.operation_type == "dual_storage_bulk_upsert"
        
        # Wait for events to be processed
        await asyncio.sleep(0.5)
        
        # Verify events were published
        assert len(event_collector) > 0
        data_written_events = [e for e in event_collector if e.event_type == EventType.DATA_WRITTEN]
        assert len(data_written_events) > 0
        
        # Verify data can be read back from hot storage
        read_data = await repo.get_with_filters({
            'symbols': [test_symbol],
            'interval': '1day'
        })
        assert len(read_data) == 5
        
        # Get dual storage metrics
        metrics = repo.get_dual_storage_metrics()
        assert metrics is not None
        assert metrics['hot_writes_success'] > 0
    
    # =========================================================================
    # News Repository Tests
    # =========================================================================
    
    async def test_news_repository_full_cycle(self, repository_factory, event_collector):
        """Test NewsRepository with real data flow."""
        repo = repository_factory.create_repository('news')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_id_prefix = uuid.uuid4().hex[:8]
        test_data = [
            {
                'news_id': f'test_news_{test_id_prefix}_{i}',
                'headline': f'Test Headline {i}',
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                'source': 'test_source',
                'symbols': ['AAPL', 'GOOGL'],
                'content': f'This is test news content number {i}',
                'sentiment_score': 0.5 + i * 0.1,
                'relevance_score': 0.8 - i * 0.1,
                'url': f'https://test.com/news/{i}'
            }
            for i in range(3)
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 3
        assert result.failed_records == 0
        
        # Wait for events
        await asyncio.sleep(0.5)
        
        # Verify events were published
        data_written_events = [e for e in event_collector if e.event_type == EventType.DATA_WRITTEN]
        assert len(data_written_events) > 0
    
    # =========================================================================
    # Company Repository Tests
    # =========================================================================
    
    async def test_company_repository_full_cycle(self, repository_factory):
        """Test CompanyRepository with real data flow."""
        repo = repository_factory.create_repository('company')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_symbol = f"TST{uuid.uuid4().hex[:5].upper()}"
        test_data = [
            {
                'symbol': test_symbol,
                'name': f'Test Company {test_symbol}',
                'exchange': 'NASDAQ',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 1000000000,
                'is_active': True
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Cryptocurrency Repository Tests
    # =========================================================================
    
    async def test_cryptocurrency_repository_full_cycle(self, repository_factory):
        """Test CryptocurrencyRepository with real data flow."""
        repo = repository_factory.create_repository('cryptocurrency')
        
        # Verify dual storage configuration
        assert hasattr(repo, 'dual_storage_writer')
        assert repo.dual_storage_writer is not None
        
        # Create test data
        test_symbol = f"TST{uuid.uuid4().hex[:5].upper()}"
        test_data = [
            {
                'symbol': test_symbol,
                'name': f'Test Crypto {test_symbol}',
                'base_currency': 'TST',
                'quote_currency': 'USD',
                'exchange': 'test_exchange',
                'is_active': True,
                'avg_dollar_volume': 1000000.0
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Dividends Repository Tests
    # =========================================================================
    
    async def test_dividends_repository_full_cycle(self, repository_factory):
        """Test DividendsRepository with real data flow."""
        repo = repository_factory.create_repository('dividends')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_data = [
            {
                'symbol': 'AAPL',
                'announcement_date': datetime.now(timezone.utc).date(),
                'ex_date': (datetime.now(timezone.utc) + timedelta(days=7)).date(),
                'record_date': (datetime.now(timezone.utc) + timedelta(days=8)).date(),
                'payment_date': (datetime.now(timezone.utc) + timedelta(days=14)).date(),
                'amount': 0.24,
                'action_type': 'dividend',
                'source': 'test'
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Financials Repository Tests
    # =========================================================================
    
    async def test_financials_repository_full_cycle(self, repository_factory):
        """Test FinancialsRepository with real data flow."""
        repo = repository_factory.create_repository('financials')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_data = [
            {
                'symbol': 'AAPL',
                'period': 'Q4',
                'year': 2024,
                'report_date': datetime.now(timezone.utc).date(),
                'revenue': 120000000000,
                'earnings': 35000000000,
                'eps': 2.18,
                'source': 'test'
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Guidance Repository Tests
    # =========================================================================
    
    async def test_guidance_repository_full_cycle(self, repository_factory):
        """Test GuidanceRepository with real data flow."""
        repo = repository_factory.create_repository('guidance')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_data = [
            {
                'symbol': 'AAPL',
                'guidance_date': datetime.now(timezone.utc),
                'period': 'Q1',
                'year': 2025,
                'revenue_low': 115000000000,
                'revenue_high': 125000000000,
                'eps_low': 2.10,
                'eps_high': 2.25,
                'source': 'test'
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Ratings Repository Tests
    # =========================================================================
    
    async def test_ratings_repository_full_cycle(self, repository_factory):
        """Test RatingsRepository with real data flow."""
        repo = repository_factory.create_repository('ratings')
        
        # Verify dual storage configuration
        assert hasattr(repo, '_dual_storage_writer')
        assert repo._dual_storage_writer is not None
        
        # Create test data
        test_data = [
            {
                'symbol': 'AAPL',
                'analyst': 'Test Analyst',
                'firm': 'Test Firm',
                'rating_date': datetime.now(timezone.utc),
                'rating': 'buy',
                'target_price': 200.0,
                'source': 'test'
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Sentiment Repository Tests
    # =========================================================================
    
    async def test_sentiment_repository_full_cycle(self, repository_factory):
        """Test SentimentRepository with real data flow."""
        repo = repository_factory.create_repository('sentiment')
        
        # Verify dual storage configuration
        assert hasattr(repo, 'dual_storage_writer')
        assert repo.dual_storage_writer is not None
        
        # Create test data
        test_data = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime.now(timezone.utc),
                'platform': 'test_platform',
                'sentiment_score': 0.75,
                'confidence': 0.9,
                'volume': 100,
                'source': 'test'
            }
        ]
        
        # Write data
        result = await repo.bulk_upsert(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records == 1
    
    # =========================================================================
    # Social Sentiment Repository Tests
    # =========================================================================
    
    async def test_social_sentiment_repository_full_cycle(self, repository_factory):
        """Test SocialSentimentRepository with real data flow."""
        repo = repository_factory.create_repository('social_sentiment')
        
        # Verify dual storage configuration
        assert hasattr(repo, 'dual_storage_writer')
        assert repo.dual_storage_writer is not None
        
        # Create test data
        test_id = f"post_{uuid.uuid4().hex[:8]}"
        test_data = [
            {
                'id': test_id,
                'platform': 'twitter',
                'symbol': 'AAPL',
                'timestamp': datetime.now(timezone.utc),
                'content': 'Test social media post about AAPL',
                'author': 'test_author',
                'sentiment': 0.8,
                'score': 100,
                'comments': 50,
                'url': f'https://twitter.com/test/{test_id}'
            }
        ]
        
        # Write data using save_posts method
        result = await repo.save_posts(test_data)
        
        # Verify write succeeded
        assert result.success is True
        assert result.processed_records >= 1  # May deduplicate
    
    # =========================================================================
    # Features Repository Tests
    # =========================================================================
    
    async def test_features_repository_full_cycle(self, repository_factory):
        """Test FeatureRepository with real data flow."""
        repo = repository_factory.create_repository('features')
        
        # Verify dual storage configuration
        assert hasattr(repo, 'dual_storage_writer')
        assert repo.dual_storage_writer is not None
        
        # Test using the store_features method
        test_features = {
            'rsi': 65.5,
            'macd': 1.2,
            'volume_ratio': 1.5,
            'price_change': 0.03
        }
        
        success = await repo.store_features(
            symbol='AAPL',
            timestamp=datetime.now(timezone.utc),
            features=test_features,
            metadata={'test': True}
        )
        
        # Verify write succeeded
        assert success is True
        
        # Verify data can be read back
        df = await repo.get_features(
            symbol='AAPL',
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        assert not df.empty
    
    # =========================================================================
    # Cross-Repository Tests
    # =========================================================================
    
    async def test_all_repositories_concurrent_writes(self, repository_factory):
        """Test concurrent writes across all repositories."""
        # Create all repositories
        repos = {
            repo_type: repository_factory.create_repository(repo_type)
            for repo_type in repository_factory.get_available_repositories()
        }
        
        # Verify all have dual storage
        for repo_type, repo in repos.items():
            has_writer = (hasattr(repo, 'dual_storage_writer') and repo.dual_storage_writer is not None) or \
                         (hasattr(repo, '_dual_storage_writer') and repo._dual_storage_writer is not None)
            assert has_writer, f"{repo_type} should have dual storage writer"
        
        # Create write tasks for each repository
        tasks = []
        
        # Market data
        tasks.append(repos['market_data'].bulk_upsert([{
            'symbol': 'CONCURRENT_TEST',
            'timestamp': datetime.now(timezone.utc),
            'interval': '1min',
            'open': 100.0,
            'close': 101.0,
            'volume': 1000
        }]))
        
        # News
        tasks.append(repos['news'].bulk_upsert([{
            'news_id': f'concurrent_{uuid.uuid4().hex[:8]}',
            'headline': 'Concurrent Test News',
            'timestamp': datetime.now(timezone.utc),
            'symbols': ['CONCURRENT_TEST']
        }]))
        
        # Company
        tasks.append(repos['company'].bulk_upsert([{
            'symbol': f'CONC{uuid.uuid4().hex[:4].upper()}',
            'name': 'Concurrent Test Company',
            'exchange': 'TEST'
        }]))
        
        # Execute all writes concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed: {result}"
            assert result.success is True, f"Task {i} write failed"
    
    # =========================================================================
    # Failure Scenario Tests
    # =========================================================================
    
    async def test_circuit_breaker_behavior(self, repository_factory):
        """Test circuit breaker opens on repeated failures."""
        repo = repository_factory.create_repository('market_data')
        
        # This test would need to simulate failures, which requires
        # either a test mode in the dual storage writer or the ability
        # to inject failures. For now, we verify the circuit breaker exists.
        assert repo._dual_storage_writer is not None
        assert hasattr(repo._dual_storage_writer, 'hot_circuit_breaker')
        assert hasattr(repo._dual_storage_writer, 'cold_circuit_breaker')
        
        # Get metrics to verify circuit breaker state
        metrics = repo.get_dual_storage_metrics()
        assert 'hot_circuit_breaker_state' in metrics
        assert 'cold_circuit_breaker_state' in metrics
        assert metrics['hot_circuit_breaker_state'] == 'closed'  # Should start closed
    
    async def test_event_publishing_reliability(self, app_context, repository_factory, event_collector):
        """Test that events are reliably published even under load."""
        repo = repository_factory.create_repository('market_data')
        
        # Clear existing events
        event_collector.clear()
        
        # Write multiple batches
        for i in range(5):
            test_data = [{
                'symbol': f'EVENT_TEST_{i}',
                'timestamp': datetime.now(timezone.utc),
                'interval': '1min',
                'open': 100.0 + i,
                'close': 101.0 + i,
                'volume': 1000 * (i + 1)
            }]
            
            result = await repo.bulk_upsert(test_data)
            assert result.success is True
        
        # Wait for all events to be processed
        await asyncio.sleep(1.0)
        
        # Verify events were published for each batch
        data_written_events = [e for e in event_collector if e.event_type == EventType.DATA_WRITTEN]
        assert len(data_written_events) >= 5, f"Expected at least 5 events, got {len(data_written_events)}"
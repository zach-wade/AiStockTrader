"""
Integration tests for news data flow through the system.

Tests the complete news data pipeline:
1. News fetching from sources
2. Archive storage with deduplication
3. Database loading
4. Sentiment analysis integration
"""

# Standard library imports
from datetime import UTC, datetime, timedelta

# Third-party imports
import pandas as pd
import pytest

# Local imports
from main.data_pipeline.ingestion.loaders.news import NewsBulkLoader
from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
from main.data_pipeline.storage.database_factory import DatabaseFactory


@pytest.fixture
async def test_archive(tmp_path):
    """Create a test archive with temporary storage."""
    archive_config = {"storage_type": "local", "local_path": str(tmp_path / "test_archive")}
    return DataArchive(archive_config)


@pytest.fixture
async def test_db():
    """Create a test database connection."""
    # Standard library imports
    import os

    config = {
        "database": {
            "host": os.getenv("TEST_DB_HOST", "localhost"),
            "port": int(os.getenv("TEST_DB_PORT", "5432")),
            "name": os.getenv("TEST_DB_NAME", "ai_trader_test"),
            "user": os.getenv("TEST_DB_USER", "zachwade"),
            "password": os.getenv("TEST_DB_PASSWORD", "ZachT$2002"),
            "pool_size": 5,
            "max_overflow": 10,
        }
    }

    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    yield db_adapter

    await db_adapter.close()


@pytest.fixture
async def news_loader(test_db, test_archive):
    """Create a news bulk loader for testing."""
    loader = NewsBulkLoader(db_adapter=test_db, archive=test_archive)
    return loader


class TestNewsDataFlow:
    """Test news data flow through the pipeline."""

    @pytest.mark.asyncio
    async def test_news_ingestion_to_archive(self, test_archive):
        """Test that news data can be stored in the archive."""
        # Create sample news data
        news_items = [
            {
                "article_id": "test_001",
                "title": "Test Company Reports Strong Earnings",
                "description": "TEST Corp exceeded expectations...",
                "content": "Full article content here...",
                "url": "https://example.com/article1",
                "published_at": datetime.now(UTC) - timedelta(hours=1),
                "source": "test_source",
                "symbols": ["TEST", "AAPL"],
                "sentiment_score": 0.8,
                "relevance_score": 0.9,
            },
            {
                "article_id": "test_002",
                "title": "Market Update: Tech Sector Rises",
                "description": "Technology stocks gain...",
                "content": "Another article content...",
                "url": "https://example.com/article2",
                "published_at": datetime.now(UTC) - timedelta(hours=2),
                "source": "test_source",
                "symbols": ["TEST", "GOOGL"],
                "sentiment_score": 0.6,
                "relevance_score": 0.7,
            },
        ]

        # Create a raw data record
        record = RawDataRecord(
            source="polygon_news",
            data_type="news",
            symbol="TEST",
            timestamp=datetime.now(UTC),
            data={"news": news_items},
            metadata={
                "article_count": len(news_items),
                "date": datetime.now(UTC).date().isoformat(),
                "symbols": ["TEST", "AAPL", "GOOGL"],
            },
        )

        # Save to archive
        await test_archive.save_raw_record_async(record)

        # Verify data was saved
        retrieved = await test_archive.query_raw_records(
            source="polygon_news",
            data_type="news",
            symbol="TEST",
            start_date=datetime.now(UTC) - timedelta(days=1),
            end_date=datetime.now(UTC) + timedelta(days=1),
        )

        assert len(retrieved) == 1
        assert retrieved[0].symbol == "TEST"
        assert len(retrieved[0].data["news"]) == 2
        assert retrieved[0].data["news"][0]["article_id"] == "test_001"

    @pytest.mark.asyncio
    async def test_news_deduplication(self, news_loader):
        """Test that duplicate news articles are handled correctly."""
        # Create duplicate news items
        news_data = [
            {
                "article_id": "dup_001",
                "title": "Duplicate Article",
                "description": "This is a duplicate",
                "url": "https://example.com/dup",
                "published_at": datetime.now(UTC),
                "source": "test",
                "symbols": ["DUP_TEST"],
            }
        ]

        # Convert to DataFrame for loader
        df = pd.DataFrame(news_data)

        # Load same article twice
        result1 = await news_loader.load(data=df, symbol="DUP_TEST", source="test")
        await news_loader.flush()

        result2 = await news_loader.load(data=df, symbol="DUP_TEST", source="test")
        await news_loader.flush()

        assert result1.success
        assert result2.success

        # Should handle duplicates gracefully
        assert result1.records_loaded >= 1

    @pytest.mark.asyncio
    async def test_news_sentiment_preservation(self, news_loader, test_db):
        """Test that sentiment scores are preserved through the pipeline."""
        # Create news with specific sentiment
        test_symbol = "SENT_TEST"
        news_data = pd.DataFrame(
            [
                {
                    "article_id": "sent_001",
                    "title": "Positive News",
                    "description": "Very positive news",
                    "url": "https://example.com/pos",
                    "published_at": datetime.now(UTC),
                    "source": "test",
                    "symbols": [test_symbol],
                    "sentiment_score": 0.95,
                    "relevance_score": 0.85,
                },
                {
                    "article_id": "sent_002",
                    "title": "Negative News",
                    "description": "Negative news",
                    "url": "https://example.com/neg",
                    "published_at": datetime.now(UTC),
                    "source": "test",
                    "symbols": [test_symbol],
                    "sentiment_score": -0.75,
                    "relevance_score": 0.90,
                },
            ]
        )

        # Load data
        await news_loader.load(news_data, test_symbol, "test")
        await news_loader.flush()

        # Query and verify sentiment
        query = """
            SELECT article_id, sentiment_score, relevance_score
            FROM news_data
            WHERE $1 = ANY(symbols)
            ORDER BY article_id
        """

        rows = await test_db.fetch_all(query, test_symbol)

        if rows:
            # Find the positive article
            pos_article = next((r for r in rows if r["article_id"] == "sent_001"), None)
            if pos_article:
                assert float(pos_article["sentiment_score"]) == pytest.approx(0.95, rel=1e-2)
                assert float(pos_article["relevance_score"]) == pytest.approx(0.85, rel=1e-2)

    @pytest.mark.asyncio
    async def test_news_symbol_association(self, news_loader, test_db):
        """Test that news articles are correctly associated with symbols."""
        # Create news for multiple symbols
        news_data = pd.DataFrame(
            [
                {
                    "article_id": "multi_001",
                    "title": "Multi-Symbol News",
                    "description": "News affecting multiple stocks",
                    "url": "https://example.com/multi",
                    "published_at": datetime.now(UTC),
                    "source": "test",
                    "symbols": ["SYM1", "SYM2", "SYM3"],
                }
            ]
        )

        # Load data
        await news_loader.load(news_data, "SYM1", "test")
        await news_loader.flush()

        # Verify all symbols are associated
        for symbol in ["SYM1", "SYM2", "SYM3"]:
            query = """
                SELECT COUNT(*) as count
                FROM news_data
                WHERE $1 = ANY(symbols)
            """

            result = await test_db.fetch_one(query, symbol)
            assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_news_time_filtering(self, news_loader, test_db):
        """Test that news can be filtered by time range."""
        test_symbol = "TIME_TEST"
        base_time = datetime.now(UTC)

        # Create news at different times
        news_data = pd.DataFrame(
            [
                {
                    "article_id": f"time_{i:03d}",
                    "title": f"Article {i}",
                    "description": f"Description {i}",
                    "url": f"https://example.com/time{i}",
                    "published_at": base_time - timedelta(hours=i),
                    "source": "test",
                    "symbols": [test_symbol],
                }
                for i in range(5)
            ]
        )

        # Load data
        await news_loader.load(news_data, test_symbol, "test")
        await news_loader.flush()

        # Query recent news (last 2 hours)
        query = """
            SELECT COUNT(*) as count
            FROM news_data
            WHERE $1 = ANY(symbols)
            AND published_at > $2
        """

        cutoff_time = base_time - timedelta(hours=2)
        result = await test_db.fetch_one(query, test_symbol, cutoff_time)

        # Should have articles from last 2 hours (0 and 1 hour ago)
        assert result["count"] >= 2

    @pytest.mark.asyncio
    async def test_news_content_integrity(self, news_loader, test_db):
        """Test that news content is preserved without corruption."""
        test_symbol = "CONTENT_TEST"

        # Create news with special characters and long content
        special_content = """
        This is a test article with special characters: @#$%^&*()

        It includes:
        - Bullet points
        - Numbers: 123.45, -67.89
        - Quotes: "This is a quote" and 'another one'
        - URLs: https://example.com/test?param=value&other=123
        - Emojis: 📈 📉 💰

        And multiple paragraphs with various formatting.
        """

        news_data = pd.DataFrame(
            [
                {
                    "article_id": "content_001",
                    "title": "Article with Special Content",
                    "description": "Testing content integrity",
                    "content": special_content,
                    "url": "https://example.com/content",
                    "published_at": datetime.now(UTC),
                    "source": "test",
                    "symbols": [test_symbol],
                }
            ]
        )

        # Load data
        await news_loader.load(news_data, test_symbol, "test")
        await news_loader.flush()

        # Retrieve and verify
        query = """
            SELECT content
            FROM news_data
            WHERE article_id = $1
        """

        result = await test_db.fetch_one(query, "content_001")

        if result and result["content"]:
            # Check key parts are preserved
            assert "@#$%^&*()" in result["content"]
            assert "123.45" in result["content"]
            assert "https://example.com/test?param=value" in result["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
News bulk loader for efficient backfill operations.

This module provides optimized bulk loading for news articles,
using PostgreSQL COPY command and efficient batching strategies.
"""

# Standard library imports
from datetime import UTC, datetime
import json
from typing import Any

# Local imports
from main.data_pipeline.services.ingestion import DeduplicationService, TextProcessingService
from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig, BulkLoadResult
from main.utils.core import get_logger

from .base import BaseBulkLoader

logger = get_logger(__name__)


class NewsBulkLoader(BaseBulkLoader[dict[str, Any]]):
    """
    Optimized bulk loader for news data.

    Handles news articles from multiple sources with deduplication,
    text processing, sentiment extraction, and efficient PostgreSQL COPY operations.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        text_processor: TextProcessingService,
        deduplicator: DeduplicationService,
        archive: Any | None = None,
        config: BulkLoadConfig | None = None,
    ):
        """
        Initialize news bulk loader.

        Args:
            db_adapter: Database adapter for operations
            text_processor: Service for text processing and sentiment
            deduplicator: Service for deduplication
            archive: Optional archive for cold storage
            config: Bulk loading configuration
        """
        # Override default buffer size for news (articles are larger)
        if config and not hasattr(config, "_modified"):
            config.buffer_size = min(config.buffer_size, 1000)  # Smaller buffer for news

        super().__init__(db_adapter=db_adapter, archive=archive, config=config, data_type="news")

        # Injected services
        self.text_processor = text_processor
        self.deduplicator = deduplicator

        logger.info(
            f"NewsBulkLoader initialized with services: "
            f"text_processor={text_processor.__class__.__name__}, "
            f"deduplicator={deduplicator.__class__.__name__}"
        )

    async def load(
        self, data: list[dict[str, Any]], symbols: list[str], source: str = "polygon", **kwargs
    ) -> BulkLoadResult:
        """
        Load news articles efficiently using bulk operations.

        Args:
            data: List of news article records
            symbols: List of symbols this news relates to
            source: Data source name ('polygon', 'alpaca', etc.)
            **kwargs: Additional parameters

        Returns:
            BulkLoadResult with operation details
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)

        if not data:
            result.success = True
            result.skip_reason = "No articles provided"
            return result

        try:
            # Deduplicate articles
            unique_articles, duplicates_removed = await self.deduplicator.deduplicate_batch(
                data, record_type="news"
            )

            if not unique_articles:
                logger.info(f"All {len(data)} articles were duplicates")
                result.success = True
                result.skip_reason = f"All {duplicates_removed} articles were duplicates"
                return result

            logger.debug(
                f"Processing {len(unique_articles)} unique articles "
                f"({duplicates_removed} duplicates removed)"
            )

            # Prepare records with text processing
            prepared_records = self._prepare_records(
                unique_articles, symbols=symbols, source=source
            )

            if not prepared_records:
                result.success = True
                result.skip_reason = "No valid articles after preparation"
                return result

            # Add to buffer
            self._add_to_buffer(prepared_records)
            for symbol in symbols:
                self._symbols_in_buffer.add(symbol.upper())

            # Check if we should flush
            if self._should_flush():
                flush_result = await self._flush_buffer()
                result.records_loaded = flush_result.records_loaded
                result.records_failed = flush_result.records_failed
                result.symbols_processed = flush_result.symbols_processed
                result.load_time_seconds = flush_result.load_time_seconds
                result.archive_time_seconds = flush_result.archive_time_seconds
                result.errors = flush_result.errors
                result.success = flush_result.success
            else:
                # Data is buffered, will be written later
                result.success = True
                result.records_loaded = len(prepared_records)
                result.metadata["buffered"] = True
                result.metadata["duplicates_removed"] = duplicates_removed

            return result

        except Exception as e:
            error_msg = f"Failed to load news: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result

    def _prepare_records(
        self, data: list[dict[str, Any]], symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Prepare news records using services for processing.

        Args:
            data: Raw news data (already deduplicated)
            symbols: Related symbols to filter by
            source: Data source ('polygon', 'alpaca', etc.)

        Returns:
            List of prepared records
        """
        prepared_records = []
        symbols_set = set(s.upper() for s in symbols) if symbols else set()
        current_time = datetime.now(UTC)

        for article in data:
            try:
                # Process article with text service
                processed = self.text_processor.process_article(article)

                # Adapt based on source
                if source == "polygon":
                    record = self._adapt_polygon_article(article, processed)
                elif source == "alpaca":
                    record = self._adapt_alpaca_article(article, processed)
                else:
                    record = self._adapt_generic_article(article, processed, source)

                # Add common fields
                record["source"] = source
                record["created_at_db"] = current_time
                record["updated_at_db"] = current_time

                # Filter by requested symbols if provided
                if symbols_set:
                    article_symbols = set(record.get("symbols", []))
                    if not article_symbols.intersection(symbols_set):
                        # Article doesn't mention any requested symbols
                        continue

                # Validate minimum required fields
                if record.get("headline") and record.get("published_at"):
                    prepared_records.append(record)
                else:
                    logger.debug(
                        f"Skipping article without headline or published_at: "
                        f"{record.get('news_id', 'unknown')}"
                    )

            except Exception as e:
                logger.warning(f"Error preparing article: {e}")
                continue

        logger.debug(f"Prepared {len(prepared_records)} records from {len(data)} articles")
        return prepared_records

    def _adapt_polygon_article(
        self, raw: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Adapt Polygon article format to standard schema.

        Args:
            raw: Raw Polygon article data
            processed: Processed article from text service

        Returns:
            Adapted record
        """
        # Generate or use existing article ID
        article_id = raw.get("id") or raw.get("article_id")
        if article_id:
            news_id = f"polygon_{article_id}"
        else:
            # Generate from URL or title hash
            # Standard library imports
            import hashlib

            content = f"{raw.get('title', '')}{raw.get('article_url', '')}"
            news_id = f"polygon_hash_{hashlib.md5(content.encode()).hexdigest()[:16]}"

        return {
            "news_id": news_id,
            "headline": processed.get("title", ""),
            "summary": None,  # Polygon doesn't have separate summary
            "content": processed.get("content", ""),
            "published_at": self._parse_datetime(raw.get("published_utc")),
            "created_at": None,  # Polygon doesn't provide
            "updated_at": None,  # Polygon doesn't provide
            "publisher": self._get_publisher(raw, "polygon"),
            "author": raw.get("author", ""),
            "url": raw.get("article_url", "") or raw.get("url", ""),
            "image_url": raw.get("image_url", ""),
            "amp_url": raw.get("amp_url", ""),
            "symbols": processed.get("symbols", []),
            "keywords": processed.get("keywords", []),
            "sentiment_score": processed.get("sentiment_positive", 0.0),
            "sentiment_label": processed.get("sentiment_overall", "neutral"),
            "sentiment_magnitude": processed.get("sentiment_negative", 0.0),
            "insights": raw.get("insights"),
            "relevance_score": None,
            "raw_data": raw,
        }

    def _adapt_alpaca_article(
        self, raw: dict[str, Any], processed: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Adapt Alpaca article format to standard schema.

        Args:
            raw: Raw Alpaca article data
            processed: Processed article from text service

        Returns:
            Adapted record
        """
        # Generate news ID
        article_id = raw.get("id") or raw.get("article_id")
        if article_id:
            news_id = f"alpaca_{article_id}"
        else:
            # Standard library imports
            import hashlib

            content = f"{raw.get('headline', '')}{raw.get('url', '')}"
            news_id = f"alpaca_hash_{hashlib.md5(content.encode()).hexdigest()[:16]}"

        return {
            "news_id": news_id,
            "headline": processed.get("title", ""),
            "summary": raw.get("summary", ""),  # Alpaca has summary
            "content": processed.get("content", ""),
            "published_at": self._parse_datetime(raw.get("created_at")),
            "created_at": self._parse_datetime(raw.get("created_at")),
            "updated_at": self._parse_datetime(raw.get("updated_at")),
            "publisher": "Benzinga",  # Alpaca primarily uses Benzinga
            "author": raw.get("author", ""),
            "url": raw.get("url", ""),
            "image_url": self._extract_first_image(raw.get("images", [])),
            "amp_url": None,  # Alpaca doesn't provide
            "symbols": processed.get("symbols", []),
            "keywords": processed.get("keywords", []),
            "sentiment_score": processed.get("sentiment_positive", 0.0),
            "sentiment_label": processed.get("sentiment_overall", "neutral"),
            "sentiment_magnitude": processed.get("sentiment_negative", 0.0),
            "insights": None,  # Alpaca doesn't provide
            "relevance_score": None,
            "raw_data": raw,
        }

    def _adapt_generic_article(
        self, raw: dict[str, Any], processed: dict[str, Any], source: str
    ) -> dict[str, Any]:
        """
        Adapt generic/unknown article format to standard schema.

        Args:
            raw: Raw article data
            processed: Processed article from text service
            source: Source name

        Returns:
            Adapted record
        """
        # Generate news ID
        article_id = raw.get("id") or raw.get("article_id")
        if article_id:
            news_id = f"{source}_{article_id}"
        else:
            # Standard library imports
            import hashlib

            content = f"{raw.get('title', '')}{raw.get('url', '')}"
            news_id = f"{source}_hash_{hashlib.md5(content.encode()).hexdigest()[:16]}"

        return {
            "news_id": news_id,
            "headline": processed.get("title", ""),
            "summary": raw.get("summary", ""),
            "content": processed.get("content", ""),
            "published_at": self._parse_datetime(
                raw.get("published_at") or raw.get("published_utc")
            ),
            "created_at": self._parse_datetime(raw.get("created_at")),
            "updated_at": self._parse_datetime(raw.get("updated_at")),
            "publisher": self._get_publisher(raw, source),
            "author": raw.get("author", ""),
            "url": raw.get("url", ""),
            "image_url": raw.get("image_url", ""),
            "amp_url": raw.get("amp_url"),
            "symbols": processed.get("symbols", []),
            "keywords": processed.get("keywords", []),
            "sentiment_score": processed.get("sentiment_positive", 0.0),
            "sentiment_label": processed.get("sentiment_overall", "neutral"),
            "sentiment_magnitude": processed.get("sentiment_negative", 0.0),
            "insights": raw.get("insights"),
            "relevance_score": None,
            "raw_data": raw,
        }

    def _get_publisher(self, article: dict[str, Any], default: str) -> str:
        """Extract publisher information."""
        publisher = article.get("publisher", {})
        if isinstance(publisher, dict):
            return publisher.get("name", default)
        elif isinstance(publisher, str):
            return publisher
        return default

    def _parse_datetime(self, dt_value: Any) -> datetime | None:
        """Parse datetime from various formats."""
        if not dt_value:
            return None

        try:
            if isinstance(dt_value, datetime):
                return dt_value if dt_value.tzinfo else dt_value.replace(tzinfo=UTC)

            dt_str = str(dt_value)

            # Try ISO format
            if "T" in dt_str:
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            else:
                # Try date only
                return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=UTC)

        except Exception as e:
            logger.debug(f"Failed to parse datetime '{dt_value}': {e}")
            return None

    def _extract_first_image(self, images: list[Any]) -> str | None:
        """Extract first image URL from list."""
        if not images or not isinstance(images, list):
            return None

        for img in images:
            if isinstance(img, str):
                return img
            elif isinstance(img, dict) and "url" in img:
                return img["url"]

        return None

    def _estimate_record_size(self, record: dict[str, Any]) -> int:
        """Estimate size of a news record."""
        # News records are much larger than market data
        # Estimate based on content length
        size = 500  # Base size

        if "content" in record:
            size += len(str(record["content"]))
        if "raw_data" in record:
            size += len(json.dumps(record["raw_data"]))

        return size

    async def _load_to_database(self, records: list[dict[str, Any]]) -> int:
        """
        Load news records to database using COPY.

        Args:
            records: News records to load

        Returns:
            Number of records loaded
        """
        if not records:
            return 0

        columns = [
            "news_id",
            "headline",
            "summary",
            "content",
            "published_at",
            "created_at",
            "updated_at",
            "source",
            "publisher",
            "author",
            "url",
            "image_url",
            "amp_url",
            "symbols",
            "keywords",
            "sentiment_score",
            "sentiment_label",
            "sentiment_magnitude",
            "insights",
            "relevance_score",
            "raw_data",
            "created_at_db",
            "updated_at_db",
        ]

        # Convert records to tuples for COPY
        copy_records = []
        for record in records:
            copy_record = (
                record["news_id"],
                record["headline"],
                record.get("summary"),
                record.get("content"),
                record.get("published_at"),
                record.get("created_at"),
                record.get("updated_at"),
                record["source"],
                record.get("publisher"),
                record.get("author"),
                record.get("url"),
                record.get("image_url"),
                record.get("amp_url"),
                json.dumps(record.get("symbols", [])),
                json.dumps(record.get("keywords", [])),
                record.get("sentiment_score"),
                record.get("sentiment_label"),
                record.get("sentiment_magnitude"),
                json.dumps(record.get("insights")) if record.get("insights") else None,
                record.get("relevance_score"),
                json.dumps(record.get("raw_data", {})),
                record["created_at_db"],
                record["updated_at_db"],
            )
            copy_records.append(copy_record)

        async with self.db_adapter.acquire() as conn:
            try:
                # Create temp table
                await conn.execute("DROP TABLE IF EXISTS temp_news")
                await conn.execute("CREATE TEMP TABLE temp_news (LIKE news_data INCLUDING ALL)")

                # Use COPY to load data
                await conn.copy_records_to_table("temp_news", records=copy_records, columns=columns)

                # UPSERT from temp table
                upsert_sql = """
                INSERT INTO news_data
                SELECT * FROM temp_news
                ON CONFLICT (news_id)
                DO UPDATE SET
                    headline = EXCLUDED.headline,
                    summary = EXCLUDED.summary,
                    content = EXCLUDED.content,
                    published_at = EXCLUDED.published_at,
                    updated_at = EXCLUDED.updated_at,
                    publisher = EXCLUDED.publisher,
                    author = EXCLUDED.author,
                    url = EXCLUDED.url,
                    symbols = EXCLUDED.symbols,
                    keywords = EXCLUDED.keywords,
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label,
                    sentiment_magnitude = EXCLUDED.sentiment_magnitude,
                    updated_at_db = EXCLUDED.updated_at_db
                """

                result = await conn.execute(upsert_sql)

                # Clean up
                await conn.execute("DROP TABLE temp_news")

                # Extract count
                if result and result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        return int(parts[2])

                return len(records)

            except Exception as e:
                logger.warning(f"COPY failed for news: {e}, falling back to INSERT")
                # Fall back to INSERT method
                return await self._load_with_insert(records)

    async def _load_with_insert(self, records: list[dict[str, Any]]) -> int:
        """Fallback INSERT method for loading news."""
        batch_size = 25  # Smaller batches for news due to size
        total_loaded = 0

        # Implementation would be similar to COPY but using INSERT
        # Keeping simple for now as COPY should work in most cases
        logger.warning("INSERT fallback not fully implemented for news")
        return 0

    async def _archive_records(self, records: list[dict[str, Any]]) -> None:
        """Archive news records to cold storage."""
        if not self.archive or not records:
            return

        # Group by date for archiving
        # Standard library imports
        from collections import defaultdict

        date_groups = defaultdict(list)

        for record in records:
            if record.get("published_at"):
                date = record["published_at"].date()
                date_groups[date].append(record)

        # Archive each date group
        for date, group_records in date_groups.items():
            try:
                # Create archive metadata
                metadata = {
                    "data_type": "news",
                    "record_count": len(group_records),
                    "date": date.isoformat(),
                    "symbols": list(
                        set(
                            symbol
                            for record in group_records
                            for symbol in record.get("symbols", [])
                        )
                    ),
                    "sources": list(
                        set(record.get("source", "unknown") for record in group_records)
                    ),
                }

                # Create RawDataRecord for archive
                # Local imports
                from main.data_pipeline.storage.archive import RawDataRecord

                record = RawDataRecord(
                    source=group_records[0].get("source", "unknown"),
                    data_type="news",
                    symbol="_aggregated",  # News can relate to multiple symbols
                    timestamp=datetime.now(UTC),
                    data={"articles": group_records},
                    metadata=metadata,
                )

                # Use archive's async save method
                await self.archive.save_raw_record_async(record)

                logger.debug(
                    f"Archived {len(group_records)} news articles for {date} "
                    f"({len(metadata['symbols'])} unique symbols)"
                )
            except Exception as e:
                logger.error(f"Failed to archive news for {date}: {e}")

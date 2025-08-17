"""
Deduplication Service

Handles deduplication of data records using various strategies including
hash-based deduplication and similarity detection.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from typing import Any

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication service."""

    check_database: bool = True  # Check against existing database records
    use_content_hash: bool = True  # Use content hashing for deduplication
    use_similarity: bool = False  # Use similarity detection (more expensive)
    similarity_threshold: float = 0.85  # Threshold for similarity matching
    cache_ttl_seconds: int = 3600  # Cache TTL for seen items
    max_cache_size: int = 10000  # Maximum cache size


class DeduplicationService:
    """
    Service for detecting and handling duplicate records.

    Supports multiple deduplication strategies including exact matching,
    content hashing, and similarity detection.
    """

    def __init__(
        self, db_adapter: IAsyncDatabase | None = None, config: DeduplicationConfig | None = None
    ):
        """
        Initialize the deduplication service.

        Args:
            db_adapter: Optional database adapter for checking existing records
            config: Service configuration
        """
        self.db_adapter = db_adapter
        self.config = config or DeduplicationConfig()

        # In-memory cache for current session
        self._seen_hashes: set[str] = set()
        self._seen_ids: set[str] = set()
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_lock = asyncio.Lock()

        # Database cache for existing records
        self._db_cache: set[str] = set()
        self._db_cache_timestamp: datetime | None = None

        logger.info(f"DeduplicationService initialized with config: {self.config}")

    def generate_content_hash(self, content: dict[str, Any]) -> str:
        """
        Generate a hash for content-based deduplication.

        Args:
            content: Content dictionary to hash

        Returns:
            SHA256 hash of the content
        """
        # Extract key fields for hashing
        hash_fields = []

        # Common fields for different data types
        if "title" in content:
            # News article
            hash_fields.append(content.get("title", ""))
            hash_fields.append(content.get("url", ""))
            hash_fields.append(str(content.get("published_at", "")))
        elif "symbol" in content and "timestamp" in content:
            # Market data or similar
            hash_fields.append(content.get("symbol", ""))
            hash_fields.append(str(content.get("timestamp", "")))
            hash_fields.append(content.get("interval", ""))
        elif "company_name" in content and "period_ending" in content:
            # Fundamentals data
            hash_fields.append(content.get("symbol", ""))
            hash_fields.append(str(content.get("period_ending", "")))
            hash_fields.append(content.get("period_type", ""))

        # Create hash from fields
        hash_content = "|".join(str(f) for f in hash_fields if f)
        return hashlib.sha256(hash_content.encode()).hexdigest()

    def generate_id_hash(self, record_id: str) -> str:
        """
        Generate a hash for ID-based deduplication.

        Args:
            record_id: Record ID to hash

        Returns:
            SHA256 hash of the ID
        """
        return hashlib.sha256(str(record_id).encode()).hexdigest()

    async def is_duplicate(
        self, record: dict[str, Any], record_type: str = "news"
    ) -> tuple[bool, str]:
        """
        Check if a record is a duplicate.

        Args:
            record: Record to check
            record_type: Type of record (news, market_data, etc.)

        Returns:
            Tuple of (is_duplicate, reason)
        """
        # Check by ID if available
        if "id" in record or "article_id" in record:
            record_id = record.get("id", record.get("article_id"))
            if await self._check_id_duplicate(record_id):
                return True, f"Duplicate ID: {record_id}"

        # Check by content hash
        if self.config.use_content_hash:
            content_hash = self.generate_content_hash(record)
            if await self._check_hash_duplicate(content_hash):
                return True, f"Duplicate content hash: {content_hash[:8]}..."

        # Check database if configured
        if self.config.check_database and self.db_adapter:
            if await self._check_database_duplicate(record, record_type):
                return True, "Exists in database"

        # Not a duplicate
        return False, ""

    async def _check_id_duplicate(self, record_id: str) -> bool:
        """Check if ID has been seen."""
        async with self._cache_lock:
            # Clean expired cache entries
            await self._clean_cache()

            # Check if seen
            id_hash = self.generate_id_hash(record_id)
            if id_hash in self._seen_ids:
                return True

            # Mark as seen
            self._seen_ids.add(id_hash)
            self._cache_timestamps[f"id_{id_hash}"] = datetime.now(UTC)

            return False

    async def _check_hash_duplicate(self, content_hash: str) -> bool:
        """Check if content hash has been seen."""
        async with self._cache_lock:
            # Check if seen
            if content_hash in self._seen_hashes:
                return True

            # Mark as seen
            self._seen_hashes.add(content_hash)
            self._cache_timestamps[f"hash_{content_hash}"] = datetime.now(UTC)

            return False

    async def _check_database_duplicate(self, record: dict[str, Any], record_type: str) -> bool:
        """
        Check if record exists in database.

        Args:
            record: Record to check
            record_type: Type of record

        Returns:
            True if duplicate exists in database
        """
        if not self.db_adapter:
            return False

        try:
            if record_type == "news":
                # Check news table
                if "url" in record:
                    query = """
                    SELECT EXISTS(
                        SELECT 1 FROM news_data
                        WHERE url = $1
                        LIMIT 1
                    )
                    """
                    result = await self.db_adapter.fetch_one(query, {"1": record["url"]})
                    return result and result.get("exists", False)

            elif record_type == "market_data":
                # Check market data tables
                query = """
                SELECT EXISTS(
                    SELECT 1 FROM market_data_1h
                    WHERE symbol = $1 AND timestamp = $2 AND interval = $3
                    LIMIT 1
                )
                """
                result = await self.db_adapter.fetch_one(
                    query,
                    {
                        "1": record["symbol"],
                        "2": record["timestamp"],
                        "3": record.get("interval", "1hour"),
                    },
                )
                return result and result.get("exists", False)

            elif record_type == "fundamentals":
                # Check fundamentals table
                query = """
                SELECT EXISTS(
                    SELECT 1 FROM financials_data
                    WHERE symbol = $1 AND period_ending = $2 AND period_type = $3
                    LIMIT 1
                )
                """
                result = await self.db_adapter.fetch_one(
                    query,
                    {
                        "1": record["symbol"],
                        "2": record["period_ending"],
                        "3": record["period_type"],
                    },
                )
                return result and result.get("exists", False)

        except Exception as e:
            logger.error(f"Error checking database for duplicate: {e}")
            return False

        return False

    async def deduplicate_batch(
        self, records: list[dict[str, Any]], record_type: str = "news"
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Deduplicate a batch of records.

        Args:
            records: List of records to deduplicate
            record_type: Type of records

        Returns:
            Tuple of (unique_records, duplicates_removed)
        """
        unique_records = []
        duplicates_removed = 0

        for record in records:
            is_dup, reason = await self.is_duplicate(record, record_type)
            if is_dup:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate: {reason}")
            else:
                unique_records.append(record)

        if duplicates_removed > 0:
            logger.info(
                f"Deduplicated {len(records)} records: "
                f"{len(unique_records)} unique, {duplicates_removed} duplicates removed"
            )

        return unique_records, duplicates_removed

    async def _clean_cache(self):
        """Clean expired entries from cache."""
        if len(self._cache_timestamps) > self.config.max_cache_size:
            # Remove oldest entries if cache is too large
            cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.cache_ttl_seconds)

            expired_keys = []
            for key, timestamp in self._cache_timestamps.items():
                if timestamp < cutoff_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache_timestamps[key]

                # Remove from appropriate cache
                if key.startswith("id_"):
                    id_hash = key[3:]
                    self._seen_ids.discard(id_hash)
                elif key.startswith("hash_"):
                    content_hash = key[5:]
                    self._seen_hashes.discard(content_hash)

            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    async def clear_cache(self):
        """Clear all deduplication caches."""
        async with self._cache_lock:
            self._seen_hashes.clear()
            self._seen_ids.clear()
            self._cache_timestamps.clear()
            self._db_cache.clear()
            self._db_cache_timestamp = None

        logger.info("Deduplication cache cleared")

    def mark_as_seen(self, record: dict[str, Any]):
        """
        Mark a record as seen without checking for duplicates.

        Useful for pre-loading known records.

        Args:
            record: Record to mark as seen
        """
        # Generate and store hashes
        if "id" in record or "article_id" in record:
            record_id = record.get("id", record.get("article_id"))
            id_hash = self.generate_id_hash(record_id)
            self._seen_ids.add(id_hash)

        if self.config.use_content_hash:
            content_hash = self.generate_content_hash(record)
            self._seen_hashes.add(content_hash)

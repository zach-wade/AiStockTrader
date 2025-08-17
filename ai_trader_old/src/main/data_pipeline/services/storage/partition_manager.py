"""
Partition Manager for PostgreSQL partitioned tables.

This module provides functionality to manage table partitions,
ensuring partitions exist for the required date ranges.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class PartitionConfig:
    """Configuration for partition management."""

    partition_type: str = "weekly"  # 'weekly', 'monthly', 'daily'
    auto_create: bool = True
    check_before_flush: bool = True
    create_future_partitions: int = 4  # Number of future partitions to pre-create
    retention_check_enabled: bool = True


class PartitionManager:
    """
    Manages partitions for time-based partitioned tables.

    Handles creation of partitions for market data tables
    and ensures partitions exist before data insertion.
    """

    def __init__(self, db_adapter: IAsyncDatabase, config: PartitionConfig | None = None):
        """
        Initialize the partition manager.

        Args:
            db_adapter: Database adapter for executing queries
            config: Partition configuration
        """
        self.db_adapter = db_adapter
        self.config = config or PartitionConfig()

        # Cache of existing partitions to reduce queries
        self._partition_cache: dict[str, set[str]] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = 300  # 5 minutes

    async def ensure_partitions_exist(
        self, table_name: str, start_date: datetime, end_date: datetime
    ) -> int:
        """
        Ensure partitions exist for the given date range.

        Args:
            table_name: Name of the partitioned table
            start_date: Start date requiring partitions
            end_date: End date requiring partitions

        Returns:
            Number of partitions created
        """
        if not self.config.auto_create:
            logger.debug(f"Auto-creation disabled for {table_name}")
            return 0

        logger.debug(
            f"Ensuring partitions exist for {table_name} "
            f"from {start_date.date()} to {end_date.date()}"
        )

        # Get existing partitions (with caching)
        existing_partitions = await self._get_existing_partitions_cached(table_name)

        # Calculate required partitions
        required_partitions = self._calculate_required_partitions(
            table_name, start_date, end_date, existing_partitions
        )

        # Create missing partitions
        created_count = 0
        for partition_info in required_partitions:
            if await self._create_partition(
                table_name, partition_info["name"], partition_info["start"], partition_info["end"]
            ):
                created_count += 1
                # Add to cache
                if table_name in self._partition_cache:
                    self._partition_cache[table_name].add(partition_info["name"])

        if created_count > 0:
            logger.info(f"Created {created_count} new partitions for {table_name}")

        return created_count

    async def ensure_partitions_for_tables(
        self, table_ranges: dict[str, tuple[datetime, datetime]]
    ) -> dict[str, int]:
        """
        Ensure partitions exist for multiple tables.

        Args:
            table_ranges: Dictionary mapping table names to (start, end) date tuples

        Returns:
            Dictionary mapping table names to number of partitions created
        """
        results = {}

        for table_name, (start_date, end_date) in table_ranges.items():
            created = await self.ensure_partitions_exist(table_name, start_date, end_date)
            results[table_name] = created

        return results

    async def _get_existing_partitions_cached(self, table_name: str) -> set[str]:
        """
        Get existing partitions with caching.

        Args:
            table_name: Name of the table

        Returns:
            Set of existing partition names
        """
        now = datetime.now(UTC)

        # Check if cache is valid
        if (
            self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
            and table_name in self._partition_cache
        ):
            return self._partition_cache[table_name]

        # Query database for partitions
        partitions = await self._get_existing_partitions(table_name)

        # Update cache
        self._partition_cache[table_name] = partitions
        self._cache_timestamp = now

        return partitions

    async def _get_existing_partitions(self, table_name: str) -> set[str]:
        """
        Get list of existing partitions for a table.

        Args:
            table_name: Name of the table

        Returns:
            Set of partition names
        """
        try:
            query = """
            SELECT
                child.relname AS partition_name
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            WHERE parent.relname = $1
            """

            rows = await self.db_adapter.fetch_all(query, {"1": table_name})

            partitions = {row["partition_name"] for row in rows}

            if partitions:
                logger.debug(f"Found {len(partitions)} existing partitions for {table_name}")

            return partitions

        except Exception as e:
            logger.error(f"Error getting existing partitions for {table_name}: {e}")
            return set()

    def _calculate_required_partitions(
        self,
        table_name: str,
        start_date: datetime,
        end_date: datetime,
        existing_partitions: set[str],
    ) -> list[dict[str, any]]:
        """
        Calculate which partitions need to be created.

        Args:
            table_name: Name of the table
            start_date: Start of date range
            end_date: End of date range
            existing_partitions: Set of existing partition names

        Returns:
            List of partition info dictionaries to create
        """
        required = []

        if self.config.partition_type == "weekly":
            # Start from Monday of the week containing start_date
            current = start_date - timedelta(days=start_date.weekday())
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)

            # Include future partitions if configured
            final_date = end_date
            if self.config.create_future_partitions > 0:
                final_date = end_date + timedelta(weeks=self.config.create_future_partitions)

            while current <= final_date:
                partition_end = current + timedelta(weeks=1)

                # Format: tablename_yyyy_ww (e.g., market_data_1h_2025_01)
                year = current.year
                week = current.isocalendar()[1]
                partition_name = f"{table_name}_{year:04d}_w{week:02d}"

                if partition_name not in existing_partitions:
                    required.append(
                        {"name": partition_name, "start": current, "end": partition_end}
                    )

                current = partition_end

        elif self.config.partition_type == "monthly":
            # Start from first day of month
            current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            final_date = end_date
            if self.config.create_future_partitions > 0:
                # Add future months
                for _ in range(self.config.create_future_partitions):
                    if final_date.month == 12:
                        final_date = final_date.replace(year=final_date.year + 1, month=1)
                    else:
                        final_date = final_date.replace(month=final_date.month + 1)

            while current <= final_date:
                # Calculate next month
                if current.month == 12:
                    partition_end = current.replace(year=current.year + 1, month=1)
                else:
                    partition_end = current.replace(month=current.month + 1)

                # Format: tablename_yyyy_mm
                partition_name = f"{table_name}_{current.year:04d}_{current.month:02d}"

                if partition_name not in existing_partitions:
                    required.append(
                        {"name": partition_name, "start": current, "end": partition_end}
                    )

                current = partition_end

        elif self.config.partition_type == "daily":
            # Daily partitions
            current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

            final_date = end_date
            if self.config.create_future_partitions > 0:
                final_date = end_date + timedelta(days=self.config.create_future_partitions)

            while current <= final_date:
                partition_end = current + timedelta(days=1)

                # Format: tablename_yyyy_mm_dd
                partition_name = f"{table_name}_{current.strftime('%Y_%m_%d')}"

                if partition_name not in existing_partitions:
                    required.append(
                        {"name": partition_name, "start": current, "end": partition_end}
                    )

                current = partition_end

        return required

    async def _create_partition(
        self, table_name: str, partition_name: str, start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Create a single partition.

        Args:
            table_name: Parent table name
            partition_name: Name for the new partition
            start_date: Partition start date (inclusive)
            end_date: Partition end date (exclusive)

        Returns:
            True if partition was created successfully
        """
        try:
            # Format dates for PostgreSQL
            start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

            # Create partition
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF {table_name}
            FOR VALUES FROM ('{start_str}') TO ('{end_str}')
            """

            await self.db_adapter.execute_query(create_sql)

            logger.info(
                f"Created partition {partition_name} for {table_name} "
                f"from {start_date.date()} to {end_date.date()}"
            )

            return True

        except Exception as e:
            # Check if it's a "already exists" error
            error_msg = str(e).lower()
            if "already exists" in error_msg or "relation" in error_msg:
                logger.debug(f"Partition {partition_name} already exists")
                return False
            else:
                logger.error(f"Error creating partition {partition_name}: {e}")
                return False

    async def drop_old_partitions(self, table_name: str, retention_days: int) -> int:
        """
        Drop partitions older than retention period.

        Args:
            table_name: Name of the partitioned table
            retention_days: Number of days to retain

        Returns:
            Number of partitions dropped
        """
        if not self.config.retention_check_enabled:
            return 0

        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        dropped_count = 0

        try:
            # Get all partitions
            partitions = await self._get_existing_partitions(table_name)

            for partition_name in partitions:
                # Try to extract date from partition name
                if self._is_partition_old(partition_name, cutoff_date):
                    if await self._drop_partition(partition_name):
                        dropped_count += 1

            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} old partitions from {table_name}")

            return dropped_count

        except Exception as e:
            logger.error(f"Error dropping old partitions: {e}")
            return 0

    def _is_partition_old(self, partition_name: str, cutoff_date: datetime) -> bool:
        """
        Check if a partition is older than cutoff date based on its name.

        Args:
            partition_name: Name of the partition
            cutoff_date: Cutoff date for retention

        Returns:
            True if partition is old and should be dropped
        """
        # Try to extract date from partition name
        # Formats: table_yyyy_ww, table_yyyy_mm, table_yyyy_mm_dd

        # Weekly partition
        weekly_match = re.search(r"_(\d{4})_w(\d{2})$", partition_name)
        if weekly_match:
            year = int(weekly_match.group(1))
            week = int(weekly_match.group(2))
            # Calculate date from year and week
            partition_date = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")
            return partition_date < cutoff_date

        # Monthly partition
        monthly_match = re.search(r"_(\d{4})_(\d{2})$", partition_name)
        if monthly_match:
            year = int(monthly_match.group(1))
            month = int(monthly_match.group(2))
            partition_date = datetime(year, month, 1, tzinfo=UTC)
            return partition_date < cutoff_date

        # Daily partition
        daily_match = re.search(r"_(\d{4})_(\d{2})_(\d{2})$", partition_name)
        if daily_match:
            year = int(daily_match.group(1))
            month = int(daily_match.group(2))
            day = int(daily_match.group(3))
            partition_date = datetime(year, month, day, tzinfo=UTC)
            return partition_date < cutoff_date

        # Can't determine age
        return False

    async def _drop_partition(self, partition_name: str) -> bool:
        """
        Drop a single partition.

        Args:
            partition_name: Name of partition to drop

        Returns:
            True if dropped successfully
        """
        try:
            drop_sql = f"DROP TABLE IF EXISTS {partition_name}"
            await self.db_adapter.execute_query(drop_sql)
            logger.info(f"Dropped partition {partition_name}")

            # Remove from cache
            for table_partitions in self._partition_cache.values():
                table_partitions.discard(partition_name)

            return True

        except Exception as e:
            logger.error(f"Error dropping partition {partition_name}: {e}")
            return False

    def clear_cache(self):
        """Clear the partition cache."""
        self._partition_cache.clear()
        self._cache_timestamp = None
        logger.debug("Partition cache cleared")

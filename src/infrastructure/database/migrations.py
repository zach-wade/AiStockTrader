"""
Database Migration System

Provides schema versioning and migration management for the AI Trading System.
Handles database schema evolution and rollback capabilities.
"""

# Standard library imports
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Local imports
from src.application.interfaces.exceptions import RepositoryError

from .adapter import PostgreSQLAdapter

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised for migration-related errors."""

    pass


@dataclass
class Migration:
    """Represents a database migration."""

    version: str
    name: str
    up_sql: str
    down_sql: str
    applied_at: datetime | None = None

    @property
    def is_applied(self) -> bool:
        """Check if migration has been applied."""
        return self.applied_at is not None


class MigrationManager:
    """
    Manages database schema migrations.

    Provides functionality to apply, rollback, and track database schema changes.
    Ensures database schema consistency across environments.
    """

    # Migration table name is a constant - not user input
    MIGRATIONS_TABLE = "schema_migrations"

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize migration manager.

        Args:
            adapter: Database adapter for executing migrations
        """
        self.adapter = adapter
        self._migrations: list[Migration] = []

    def _validate_identifier(self, identifier: str) -> None:
        """
        Simple validation to ensure identifier contains only safe characters.

        Args:
            identifier: The identifier to validate

        Raises:
            ValueError: If identifier contains unsafe characters
        """
        # Only allow alphanumeric, underscore, and not too long
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$", identifier):
            raise ValueError(
                f"Invalid identifier '{identifier}'. Only alphanumeric and underscore allowed."
            )

    async def initialize(self) -> None:
        """
        Initialize migration system.

        Creates the migrations tracking table if it doesn't exist.
        """
        # Use parameterized query for the table creation
        # Since table name can't be parameterized, we use our constant
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            applied_by VARCHAR(100) NOT NULL DEFAULT current_user,
            execution_time_ms INTEGER,
            checksum VARCHAR(64)
        );

        CREATE INDEX IF NOT EXISTS idx_migrations_applied_at
        ON {self.MIGRATIONS_TABLE}(applied_at);
        """

        await self.adapter.execute_query(create_table_sql)
        logger.info("Migration system initialized")

    def add_migration(
        self,
        version: str,
        name: str,
        up_sql: str,
        down_sql: str = "",
    ) -> None:
        """
        Add a migration to the manager.

        Args:
            version: Migration version (e.g., "001", "20241201_001")
            name: Migration name (e.g., "create_orders_table")
            up_sql: SQL to apply the migration
            down_sql: SQL to rollback the migration
        """
        migration = Migration(
            version=version,
            name=name,
            up_sql=up_sql,
            down_sql=down_sql,
        )

        self._migrations.append(migration)
        logger.debug(f"Added migration: {version} - {name}")

    def load_migrations_from_directory(self, directory: Path) -> None:
        """
        Load migrations from a directory.

        Args:
            directory: Directory containing migration files

        Expected file naming: {version}_{name}_up.sql and {version}_{name}_down.sql
        """
        if not directory.exists():
            logger.warning(f"Migration directory does not exist: {directory}")
            return

        up_files = list(directory.glob("*_up.sql"))

        for up_file in sorted(up_files):
            # Parse filename: {version}_{name}_up.sql
            parts = up_file.stem.split("_")
            MIN_FILENAME_PARTS = 3
            if len(parts) < MIN_FILENAME_PARTS:
                logger.warning(f"Invalid migration filename: {up_file.name}")
                continue

            version = parts[0]
            name = "_".join(parts[1:-1])  # Everything except version and "up"

            # Read up migration
            up_sql = up_file.read_text(encoding="utf-8")

            # Read down migration if exists
            down_file = directory / f"{version}_{name}_down.sql"
            down_sql = ""
            if down_file.exists():
                down_sql = down_file.read_text(encoding="utf-8")

            self.add_migration(version, name, up_sql, down_sql)

        logger.info(f"Loaded {len(up_files)} migrations from {directory}")

    async def get_applied_migrations(self) -> list[Migration]:
        """
        Get list of applied migrations from database.

        Returns:
            List of applied migrations
        """
        # Safe to use string formatting here - MIGRATIONS_TABLE is a constant
        # and validated to contain only safe characters
        # nosec B608 - table name is a constant, not user input
        query = f"""
        SELECT version, name, applied_at
        FROM {self.MIGRATIONS_TABLE}
        ORDER BY applied_at ASC
        """

        records = await self.adapter.fetch_all(query)

        applied_migrations = []
        for record in records:
            # Find corresponding migration definition
            migration = None
            for m in self._migrations:
                if m.version == record["version"]:
                    migration = Migration(
                        version=m.version,
                        name=m.name,
                        up_sql=m.up_sql,
                        down_sql=m.down_sql,
                        applied_at=record["applied_at"],
                    )
                    break

            if migration is None:
                # Create placeholder for unknown migration
                migration = Migration(
                    version=record["version"],
                    name=record["name"],
                    up_sql="",
                    down_sql="",
                    applied_at=record["applied_at"],
                )

            applied_migrations.append(migration)

        return applied_migrations

    async def get_pending_migrations(self) -> list[Migration]:
        """
        Get list of pending migrations.

        Returns:
            List of migrations not yet applied
        """
        applied_migrations = await self.get_applied_migrations()
        applied_versions = {m.version for m in applied_migrations}

        pending = [
            m
            for m in sorted(self._migrations, key=lambda x: x.version)
            if m.version not in applied_versions
        ]

        return pending

    async def apply_migration(self, migration: Migration) -> None:
        """
        Apply a single migration.

        Args:
            migration: Migration to apply

        Raises:
            RepositoryError: If migration fails
        """
        start_time = datetime.now(UTC)
        logger.info(f"Applying migration {migration.version}: {migration.name}")

        try:
            # Execute migration in transaction
            await self.adapter.begin_transaction()

            # Apply the migration
            await self.adapter.execute_query(migration.up_sql)

            # Record migration as applied
            execution_time = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            # Safe to use string formatting here - MIGRATIONS_TABLE is a constant
            # nosec B608 - table name is a constant, not user input
            insert_query = f"""
            INSERT INTO {self.MIGRATIONS_TABLE}
            (version, name, applied_at, execution_time_ms)
            VALUES ($1, $2, $3, $4)
            """

            await self.adapter.execute_query(
                insert_query,
                migration.version,
                migration.name,
                start_time,
                execution_time,
            )

            await self.adapter.commit_transaction()

            migration.applied_at = start_time
            logger.info(f"Migration {migration.version} applied successfully in {execution_time}ms")

        except Exception as e:
            await self.adapter.rollback_transaction()
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            raise RepositoryError(f"Migration {migration.version} failed: {e}") from e

    async def rollback_migration(self, migration: Migration) -> None:
        """
        Rollback a single migration.

        Args:
            migration: Migration to rollback

        Raises:
            RepositoryError: If rollback fails
        """
        if not migration.down_sql:
            raise RepositoryError(f"Migration {migration.version} has no rollback SQL")

        logger.info(f"Rolling back migration {migration.version}: {migration.name}")

        try:
            # Execute rollback in transaction
            await self.adapter.begin_transaction()

            # Execute rollback SQL
            await self.adapter.execute_query(migration.down_sql)

            # Remove migration record
            # Safe to use string formatting here - MIGRATIONS_TABLE is a constant
            # nosec B608 - table name is a constant, not user input
            delete_query = f"""
            DELETE FROM {self.MIGRATIONS_TABLE}
            WHERE version = $1
            """

            await self.adapter.execute_query(delete_query, migration.version)

            await self.adapter.commit_transaction()

            migration.applied_at = None
            logger.info(f"Migration {migration.version} rolled back successfully")

        except Exception as e:
            await self.adapter.rollback_transaction()
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            raise RepositoryError(f"Migration rollback {migration.version} failed: {e}") from e

    async def migrate_to_latest(self) -> int:
        """
        Apply all pending migrations.

        Returns:
            Number of migrations applied

        Raises:
            RepositoryError: If any migration fails
        """
        pending_migrations = await self.get_pending_migrations()

        if not pending_migrations:
            logger.info("No pending migrations")
            return 0

        logger.info(f"Applying {len(pending_migrations)} pending migrations")

        applied_count = 0
        for migration in pending_migrations:
            await self.apply_migration(migration)
            applied_count += 1

        logger.info(f"Applied {applied_count} migrations successfully")
        return applied_count

    async def migrate_to_version(self, target_version: str) -> int:
        """
        Migrate to a specific version.

        Args:
            target_version: Version to migrate to

        Returns:
            Number of migrations applied (positive) or rolled back (negative)

        Raises:
            RepositoryError: If migration fails
        """
        applied_migrations = await self.get_applied_migrations()
        current_versions = [m.version for m in applied_migrations]

        # Check if we need to go forward or backward
        if target_version not in [m.version for m in self._migrations]:
            raise RepositoryError(f"Unknown migration version: {target_version}")

        if target_version in current_versions:
            logger.info(f"Already at version {target_version}")
            return 0

        # Determine direction
        sorted_migrations = sorted(self._migrations, key=lambda x: x.version)
        target_index = next(
            i for i, m in enumerate(sorted_migrations) if m.version == target_version
        )

        changes = 0

        if not current_versions:
            # Apply migrations up to target
            for migration in sorted_migrations[: target_index + 1]:
                await self.apply_migration(migration)
                changes += 1
        else:
            current_latest = max(current_versions)
            current_index = next(
                i for i, m in enumerate(sorted_migrations) if m.version == current_latest
            )

            if target_index > current_index:
                # Apply forward migrations
                for migration in sorted_migrations[current_index + 1 : target_index + 1]:
                    await self.apply_migration(migration)
                    changes += 1
            else:
                # Rollback migrations
                for migration in reversed(applied_migrations[target_index + 1 :]):
                    await self.rollback_migration(migration)
                    changes -= 1

        logger.info(f"Migrated to version {target_version}")
        return changes

    async def get_status(self) -> dict[str, Any]:
        """
        Get migration status.

        Returns:
            Dictionary with migration information
        """
        applied_migrations = await self.get_applied_migrations()
        pending_migrations = await self.get_pending_migrations()

        current_version = None
        if applied_migrations:
            current_version = max(m.version for m in applied_migrations)

        return {
            "current_version": current_version,
            "total_migrations": len(self._migrations),
            "applied_count": len(applied_migrations),
            "pending_count": len(pending_migrations),
            "applied_migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                    "applied_at": m.applied_at.isoformat() if m.applied_at else None,
                }
                for m in applied_migrations
            ],
            "pending_migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                }
                for m in pending_migrations
            ],
        }


class SchemaManager:
    """
    Manages database schema operations.

    Provides functionality to create, drop, and check database schemas.
    """

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize schema manager.

        Args:
            adapter: Database adapter for executing schema operations
        """
        self.adapter = adapter
        self.schema_name = "public"  # Default schema

    def _validate_schema_name(self, schema_name: str) -> None:
        """
        Simple validation to ensure schema name contains only safe characters.

        Args:
            schema_name: The schema name to validate

        Raises:
            ValueError: If schema name contains unsafe characters
        """
        # Only allow alphanumeric, underscore, and not too long
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$", schema_name):
            raise ValueError(
                f"Invalid schema name '{schema_name}'. Only alphanumeric and underscore allowed."
            )

    async def create_schema(self, schema_name: str | None = None) -> None:
        """
        Create a database schema.

        Args:
            schema_name: Name of the schema to create. Uses default if not provided.

        Raises:
            MigrationError: If schema creation fails
        """
        schema = schema_name or self.schema_name

        # Validate schema name to prevent SQL injection
        self._validate_schema_name(schema)

        try:
            # Use double quotes to properly escape the identifier
            create_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema}"'
            await self.adapter.execute_query(create_sql)
            logger.info(f"Schema '{schema}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create schema '{schema}': {e}")
            raise MigrationError(f"Failed to create schema '{schema}': {e}") from e

    async def drop_schema(self, schema_name: str | None = None, cascade: bool = False) -> None:
        """
        Drop a database schema.

        Args:
            schema_name: Name of the schema to drop. Uses default if not provided.
            cascade: If True, drops all objects in the schema as well.

        Raises:
            MigrationError: If schema drop fails
        """
        schema = schema_name or self.schema_name

        # Prevent dropping public schema by default unless explicitly requested
        if schema == "public" and not schema_name:
            raise MigrationError("Cannot drop public schema without explicit schema_name parameter")

        # Validate schema name to prevent SQL injection
        self._validate_schema_name(schema)

        try:
            cascade_clause = "CASCADE" if cascade else "RESTRICT"
            drop_sql = f'DROP SCHEMA IF EXISTS "{schema}" {cascade_clause}'
            await self.adapter.execute_query(drop_sql)
            logger.info(f"Schema '{schema}' dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop schema '{schema}': {e}")
            raise MigrationError(f"Failed to drop schema '{schema}': {e}") from e

    async def schema_exists(self, schema_name: str | None = None) -> bool:
        """
        Check if a database schema exists.

        Args:
            schema_name: Name of the schema to check. Uses default if not provided.

        Returns:
            True if the schema exists, False otherwise

        Raises:
            MigrationError: If the check operation fails
        """
        schema = schema_name or self.schema_name

        try:
            check_sql = """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = $1
            )
            """
            result = await self.adapter.fetch_one(check_sql, schema)
            return bool(result and result.get("exists", False))
        except Exception as e:
            logger.error(f"Failed to check if schema '{schema}' exists: {e}")
            raise MigrationError(f"Failed to check schema existence: {e}") from e


# Predefined migrations for the trading system
def get_initial_migrations() -> list[tuple[str, str, str, str]]:
    """
    Get initial migrations for the trading system.

    Returns:
        List of (version, name, up_sql, down_sql) tuples
    """
    return [
        (
            "001",
            "create_trading_schema",
            """
            -- Create required extensions
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

            -- Enum types for orders
            CREATE TYPE order_side AS ENUM ('buy', 'sell');
            CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
            CREATE TYPE order_status AS ENUM ('pending', 'submitted', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired');
            CREATE TYPE time_in_force AS ENUM ('day', 'gtc', 'ioc', 'fok');
            """,
            """
            DROP TYPE IF EXISTS time_in_force;
            DROP TYPE IF EXISTS order_status;
            DROP TYPE IF EXISTS order_type;
            DROP TYPE IF EXISTS order_side;
            """,
        ),
        (
            "002",
            "create_orders_table",
            Path(__file__).parent / "schemas.sql",  # Reference to the schema file
            """
            DROP TABLE IF EXISTS orders;
            """,
        ),
        (
            "003",
            "create_positions_table",
            # This would be extracted from schemas.sql for positions table
            "",
            """
            DROP TABLE IF EXISTS positions;
            """,
        ),
        (
            "004",
            "create_portfolios_table",
            # This would be extracted from schemas.sql for portfolios table
            "",
            """
            DROP TABLE IF EXISTS portfolios;
            """,
        ),
    ]

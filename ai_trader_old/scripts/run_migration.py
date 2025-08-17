#!/usr/bin/env python3
"""
Database migration runner for AI Trader system.

Usage:
    python scripts/run_migration.py --migration 001_add_is_sp500_column
    python scripts/run_migration.py --list
    python scripts/run_migration.py --status
"""

# Standard library imports
import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


class MigrationRunner:
    """Handles database migrations for the AI Trader system."""

    def __init__(self, db_adapter: IAsyncDatabase):
        self.db_adapter = db_adapter
        self.migrations_dir = Path(__file__).parent.parent / "sql" / "migrations"
        self.migrations_dir.mkdir(parents=True, exist_ok=True)

    async def initialize_migration_table(self):
        """Create migrations tracking table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            checksum VARCHAR(64)
        );

        CREATE INDEX IF NOT EXISTS idx_schema_migrations_name
        ON schema_migrations(migration_name);
        """

        async with self.db_adapter.acquire() as conn:
            await conn.execute(create_table_sql)

        logger.info("Migration tracking table initialized")

    async def get_executed_migrations(self) -> list[str]:
        """Get list of already executed migrations."""
        query = """
        SELECT migration_name
        FROM schema_migrations
        WHERE success = TRUE
        ORDER BY executed_at
        """

        async with self.db_adapter.acquire() as conn:
            rows = await conn.fetch(query)
            return [row["migration_name"] for row in rows]

    async def get_available_migrations(self) -> list[str]:
        """Get list of available migration files."""
        migrations = []
        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            migrations.append(file_path.stem)
        return migrations

    async def get_pending_migrations(self) -> list[str]:
        """Get list of migrations that haven't been executed."""
        executed = set(await self.get_executed_migrations())
        available = await self.get_available_migrations()
        return [m for m in available if m not in executed]

    async def execute_migration(self, migration_name: str) -> bool:
        """Execute a single migration."""
        migration_file = self.migrations_dir / f"{migration_name}.sql"

        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False

        # Read migration SQL
        migration_sql = migration_file.read_text()

        # Calculate checksum for verification
        # Standard library imports
        import hashlib

        checksum = hashlib.sha256(migration_sql.encode()).hexdigest()

        logger.info(f"Executing migration: {migration_name}")

        try:
            async with self.db_adapter.acquire() as conn:
                # Execute the migration in a transaction
                async with conn.transaction():
                    await conn.execute(migration_sql)

                    # Record successful execution
                    await conn.execute(
                        """
                        INSERT INTO schema_migrations (migration_name, success, checksum)
                        VALUES ($1, TRUE, $2)
                        ON CONFLICT (migration_name) DO UPDATE SET
                            executed_at = NOW(),
                            success = TRUE,
                            checksum = $2,
                            error_message = NULL
                        """,
                        migration_name,
                        checksum,
                    )

            logger.info(f"✅ Migration completed successfully: {migration_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Migration failed: {migration_name}")
            logger.error(f"Error: {e}")

            # Record failed execution
            try:
                async with self.db_adapter.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO schema_migrations (migration_name, success, error_message, checksum)
                        VALUES ($1, FALSE, $2, $3)
                        ON CONFLICT (migration_name) DO UPDATE SET
                            executed_at = NOW(),
                            success = FALSE,
                            error_message = $2,
                            checksum = $3
                        """,
                        migration_name,
                        str(e),
                        checksum,
                    )
            except Exception as record_error:
                logger.error(f"Failed to record migration failure: {record_error}")

            return False

    async def run_migrations(self, specific_migration: str = None) -> bool:
        """Run pending migrations or a specific migration."""
        await self.initialize_migration_table()

        if specific_migration:
            # Run specific migration
            if specific_migration in await self.get_executed_migrations():
                logger.warning(f"Migration already executed: {specific_migration}")
                return True

            return await self.execute_migration(specific_migration)
        else:
            # Run all pending migrations
            pending = await self.get_pending_migrations()

            if not pending:
                logger.info("No pending migrations to execute")
                return True

            logger.info(f"Found {len(pending)} pending migrations")

            success = True
            for migration in pending:
                if not await self.execute_migration(migration):
                    success = False
                    break

            return success

    async def migration_status(self) -> dict[str, Any]:
        """Get migration status information."""
        await self.initialize_migration_table()

        executed = await self.get_executed_migrations()
        available = await self.get_available_migrations()
        pending = await self.get_pending_migrations()

        return {
            "executed_count": len(executed),
            "available_count": len(available),
            "pending_count": len(pending),
            "executed_migrations": executed,
            "pending_migrations": pending,
        }


async def main():
    """Main entry point for migration runner."""
    parser = argparse.ArgumentParser(description="Database Migration Runner")
    parser.add_argument("--migration", help="Specific migration to run")
    parser.add_argument("--list", action="store_true", help="List available migrations")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--run-all", action="store_true", help="Run all pending migrations")

    args = parser.parse_args()

    # Initialize database connection
    config = get_config()
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    runner = MigrationRunner(db_adapter)

    try:
        if args.list:
            # List available migrations
            available = await runner.get_available_migrations()
            executed = await runner.get_executed_migrations()

            print("Available Migrations:")
            for migration in available:
                status = "✅ EXECUTED" if migration in executed else "⏳ PENDING"
                print(f"  {migration}: {status}")

        elif args.status:
            # Show migration status
            status = await runner.migration_status()
            print("Migration Status:")
            print(f"  Total migrations: {status['available_count']}")
            print(f"  Executed: {status['executed_count']}")
            print(f"  Pending: {status['pending_count']}")

            if status["pending_migrations"]:
                print("\nPending migrations:")
                for migration in status["pending_migrations"]:
                    print(f"  - {migration}")

        elif args.migration:
            # Run specific migration
            success = await runner.run_migrations(args.migration)
            sys.exit(0 if success else 1)

        elif args.run_all:
            # Run all pending migrations
            success = await runner.run_migrations()
            sys.exit(0 if success else 1)

        else:
            # Show help
            parser.print_help()

    finally:
        # Clean up database connection
        if hasattr(db_adapter, "close"):
            await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(main())

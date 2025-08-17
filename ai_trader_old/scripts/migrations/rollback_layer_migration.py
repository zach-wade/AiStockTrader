#!/usr/bin/env python3
"""
Rollback script for layer column migration.

This script removes the layer column and related changes,
reverting back to the layer[1-3]_qualified boolean columns.
"""

# Standard library imports
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.utils.core import get_logger

logger = get_logger(__name__)


async def backup_layer_data():
    """Backup layer data before rollback."""

    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db = db_factory.create_async_database(config)

    try:
        # Create backup table
        backup_query = """
            CREATE TABLE IF NOT EXISTS companies_layer_backup AS
            SELECT symbol, layer, layer_updated_at, layer_reason
            FROM companies
            WHERE layer IS NOT NULL
        """

        await db.execute_query(backup_query)

        # Count backed up records
        count_query = "SELECT COUNT(*) as count FROM companies_layer_backup"
        result = await db.fetch_one(count_query)

        logger.info(f"Backed up {result['count']} records to companies_layer_backup table")

        return True

    except Exception as e:
        logger.error(f"Failed to backup layer data: {e}")
        return False
    finally:
        await db.close()


async def run_rollback(force: bool = False):
    """Execute the rollback."""

    logger.warning("Starting rollback of layer column migration")

    if not force:
        # Backup data first
        logger.info("Creating backup of layer data...")
        if not await backup_layer_data():
            logger.error("Backup failed! Use --force to rollback without backup")
            return False

    # Get database connection
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")
    db_factory = DatabaseFactory()
    db = db_factory.create_async_database(config)

    try:
        # Read the rollback SQL file
        rollback_file = Path(__file__).parent / "rollback_layer_column.sql"
        with open(rollback_file) as f:
            rollback_sql = f.read()

        # Split into individual statements
        statements = [
            s.strip()
            for s in rollback_sql.split(";")
            if s.strip() and not s.strip().startswith("--")
        ]

        logger.info(f"Executing {len(statements)} rollback statements")

        for i, statement in enumerate(statements, 1):
            if statement.strip():
                try:
                    logger.debug(f"Executing rollback statement {i}/{len(statements)}")
                    await db.execute_query(statement + ";")
                except Exception as e:
                    # Some statements might fail if objects don't exist
                    if "does not exist" in str(e).lower():
                        logger.debug(f"Statement {i} skipped (object doesn't exist)")
                    else:
                        logger.error(f"Error executing rollback statement {i}: {e}")
                        raise

        # Verify rollback
        logger.info("Verifying rollback...")

        # Check that layer columns are gone
        column_check = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'companies'
            AND column_name IN ('layer', 'layer_updated_at', 'layer_reason')
        """

        columns = await db.fetch_all(column_check)

        if columns:
            logger.error(f"Rollback incomplete! Found {len(columns)} layer columns still present")
            return False
        else:
            logger.info("Layer columns successfully removed")

        # Check that old columns still exist
        old_column_check = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'companies'
            AND column_name IN ('layer1_qualified', 'layer2_qualified', 'layer3_qualified')
        """

        old_columns = await db.fetch_all(old_column_check)

        if len(old_columns) == 3:
            logger.info("Old layer[1-3]_qualified columns are intact")
        else:
            logger.warning(f"Only found {len(old_columns)} of 3 old layer columns")

        logger.info("Rollback completed successfully!")
        logger.info("Layer data backup preserved in companies_layer_backup table")

        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False
    finally:
        await db.close()


if __name__ == "__main__":
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="Rollback layer column migration")
    parser.add_argument("--force", action="store_true", help="Force rollback without backup")
    parser.add_argument(
        "--restore-backup", action="store_true", help="Restore from backup after rollback"
    )

    args = parser.parse_args()

    success = asyncio.run(run_rollback(force=args.force))
    sys.exit(0 if success else 1)

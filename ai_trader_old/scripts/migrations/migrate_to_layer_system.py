#!/usr/bin/env python3
"""
Migration script to transition from layer[1-3]_qualified boolean columns
to a single layer integer column in the companies table.

This implements the new layer-based architecture:
- Layer 0 (BASIC): ~10,000 symbols
- Layer 1 (LIQUID): ~2,000 symbols
- Layer 2 (CATALYST): ~500 symbols
- Layer 3 (ACTIVE): ~50 symbols
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


async def run_migration():
    """Execute the layer column migration."""

    logger.info("Starting migration to layer-based system")

    # Get database connection
    config_manager = get_config_manager()
    # Load base config with storage defaults
    config = config_manager.load_config("defaults/storage")
    db_factory = DatabaseFactory()
    db = db_factory.create_async_database(config)

    try:
        # Read the SQL migration file
        migration_file = Path(__file__).parent / "add_layer_column.sql"
        with open(migration_file) as f:
            migration_sql = f.read()

        # Split into individual statements (PostgreSQL doesn't like multiple statements in one execute)
        statements = [
            s.strip()
            for s in migration_sql.split(";")
            if s.strip() and not s.strip().startswith("--")
        ]

        logger.info(f"Executing {len(statements)} migration statements")

        for i, statement in enumerate(statements, 1):
            if statement.strip():
                try:
                    logger.debug(f"Executing statement {i}/{len(statements)}")
                    await db.execute_query(statement + ";")
                except Exception as e:
                    # Some statements might fail if already applied (IF NOT EXISTS)
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.debug(f"Statement {i} skipped (already applied): {str(e)[:100]}")
                    else:
                        logger.error(f"Error executing statement {i}: {e}")
                        raise

        # Verify migration
        logger.info("Verifying migration results...")

        # Check layer distribution
        layer_query = """
            SELECT
                layer,
                COUNT(*) as count,
                COUNT(*) FILTER (WHERE is_active = true) as active_count
            FROM companies
            GROUP BY layer
            ORDER BY layer
        """

        results = await db.fetch_all(layer_query)

        logger.info("Layer distribution after migration:")
        for row in results:
            layer_name = {0: "BASIC", 1: "LIQUID", 2: "CATALYST", 3: "ACTIVE"}.get(
                row["layer"], f"Layer {row['layer']}"
            )
            logger.info(f"  {layer_name}: {row['count']} total, {row['active_count']} active")

        # Check for any unmigrated records
        unmigrated_query = "SELECT COUNT(*) as count FROM companies WHERE layer IS NULL"
        unmigrated = await db.fetch_one(unmigrated_query)

        if unmigrated and unmigrated["count"] > 0:
            logger.warning(f"Found {unmigrated['count']} unmigrated records")
        else:
            logger.info("All records successfully migrated")

        # Sample some migrated records
        sample_query = """
            SELECT symbol, layer, layer_reason, layer_updated_at
            FROM companies
            WHERE layer > 0
            ORDER BY layer DESC, symbol
            LIMIT 10
        """

        samples = await db.fetch_all(sample_query)

        if samples:
            logger.info("Sample migrated records:")
            for row in samples:
                logger.info(f"  {row['symbol']}: Layer {row['layer']} - {row['layer_reason']}")

        logger.info("Migration completed successfully!")

        # Note about old columns
        logger.info(
            "\nNOTE: Old columns (layer1_qualified, layer2_qualified, layer3_qualified) are still present"
        )
        logger.info("They will be dropped after verifying the new system works correctly")
        logger.info(
            "Run 'python scripts/migrations/drop_old_layer_columns.py' when ready to remove them"
        )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await db.close()


async def verify_migration():
    """Verify the migration was successful."""

    config_manager = get_config_manager()
    # Load base config with storage defaults
    config = config_manager.load_config("defaults/storage")
    db_factory = DatabaseFactory()
    db = db_factory.create_async_database(config)

    try:
        # Check that layer column exists
        column_check = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'companies'
            AND column_name IN ('layer', 'layer_updated_at', 'layer_reason')
            ORDER BY column_name
        """

        columns = await db.fetch_all(column_check)

        logger.info("New columns added:")
        for col in columns:
            logger.info(f"  {col['column_name']}: {col['data_type']}")

        if len(columns) != 3:
            logger.error("Not all new columns were added!")
            return False

        return True

    finally:
        await db.close()


if __name__ == "__main__":
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="Migrate companies table to layer-based system")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify migration, don't run it"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force migration even if already applied"
    )

    args = parser.parse_args()

    if args.verify_only:
        success = asyncio.run(verify_migration())
        sys.exit(0 if success else 1)
    else:
        asyncio.run(run_migration())

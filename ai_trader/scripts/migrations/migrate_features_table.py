#!/usr/bin/env python3
"""
Migration script to create the features table.

This script creates the features table required for storing ML features.
It can be run multiple times safely as it uses IF NOT EXISTS clauses.
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from main.config import get_config_manager
from main.utils.core import get_logger

logger = get_logger(__name__)


async def run_migration():
    """Run the features table migration."""
    
    # Get database URL from environment or use default
    connection_string = os.getenv('DATABASE_URL')
    
    if not connection_string:
        # Build from individual environment variables
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'ai_trader')
        db_user = os.getenv('DB_USER', 'zachwade')
        db_password = os.getenv('DB_PASSWORD', '')
        
        if db_password:
            connection_string = (
                f"postgresql://{db_user}:{db_password}"
                f"@{db_host}:{db_port}/{db_name}"
            )
        else:
            connection_string = (
                f"postgresql://{db_user}"
                f"@{db_host}:{db_port}/{db_name}"
            )
    
    logger.info("Connecting to database...")
    
    # Connect to the database
    conn = await asyncpg.connect(connection_string)
    
    try:
        # Read the migration SQL
        migration_file = Path(__file__).parent / 'create_features_table.sql'
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        logger.info("Running features table migration...")
        
        # Execute the migration
        await conn.execute(migration_sql)
        
        # Verify the table was created
        check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'features'
            );
        """
        
        table_exists = await conn.fetchval(check_query)
        
        if table_exists:
            # Get table info
            info_query = """
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = 'features'
                ORDER BY ordinal_position;
            """
            columns = await conn.fetch(info_query)
            
            logger.info("✅ Features table created successfully!")
            logger.info("Table structure:")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f"DEFAULT {col['column_default']}" if col['column_default'] else ""
                logger.info(f"  - {col['column_name']}: {col['data_type']} {nullable} {default}")
            
            # Check indexes
            index_query = """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'features';
            """
            indexes = await conn.fetch(index_query)
            
            logger.info("\nIndexes created:")
            for idx in indexes:
                logger.info(f"  - {idx['indexname']}")
            
            return True
        else:
            logger.error("❌ Failed to create features table")
            return False
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await conn.close()
        logger.info("Database connection closed")


def main():
    """Main entry point."""
    try:
        logger.info("=" * 60)
        logger.info("Features Table Migration")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)
        
        result = asyncio.run(run_migration())
        
        if result:
            logger.info("\n🎉 Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Migration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nMigration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
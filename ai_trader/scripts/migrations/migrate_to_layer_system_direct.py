#!/usr/bin/env python3
"""
Direct migration script to transition from layer[1-3]_qualified boolean columns
to a single layer integer column in the companies table.

This version connects directly to the database without going through the config system
to avoid circular import issues.
"""

import asyncio
import asyncpg
import os
from pathlib import Path
from datetime import datetime

# Simple logging
def log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}")


async def run_migration():
    """Execute the layer column migration."""
    
    log("Starting migration to layer-based system")
    
    # Get database connection details from environment or defaults
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")  # Try default postgres db
    db_user = os.getenv("DB_USER", "zachwade")
    db_pass = os.getenv("DB_PASS", "ZachT$2002")
    
    # Create connection string
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    log(f"Connecting to database at {db_host}:{db_port}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(dsn)
        
        # Read the SQL migration file
        migration_file = Path(__file__).parent / "add_layer_column.sql"
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Split into individual statements
        statements = []
        current = []
        in_function = False
        
        for line in migration_sql.split('\n'):
            # Track if we're inside a function definition
            if 'CREATE OR REPLACE FUNCTION' in line or 'CREATE FUNCTION' in line:
                in_function = True
            elif '$$' in line and 'LANGUAGE' in line:
                in_function = False
                
            current.append(line)
            
            # End of statement
            if line.strip().endswith(';') and not in_function:
                stmt = '\n'.join(current).strip()
                if stmt and not stmt.startswith('--'):
                    statements.append(stmt)
                current = []
        
        log(f"Executing {len(statements)} migration statements")
        
        for i, statement in enumerate(statements, 1):
            if statement.strip():
                try:
                    log(f"Executing statement {i}/{len(statements)}", "DEBUG")
                    await conn.execute(statement)
                except Exception as e:
                    # Some statements might fail if already applied
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        log(f"Statement {i} skipped (already applied)", "DEBUG")
                    elif "does not exist" in str(e).lower() and "DROP" in statement:
                        log(f"Statement {i} skipped (object doesn't exist)", "DEBUG")
                    else:
                        log(f"Error executing statement {i}: {e}", "ERROR")
                        raise
        
        # Verify migration
        log("Verifying migration results...")
        
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
        
        results = await conn.fetch(layer_query)
        
        log("Layer distribution after migration:")
        for row in results:
            layer_name = {
                0: "BASIC",
                1: "LIQUID", 
                2: "CATALYST",
                3: "ACTIVE"
            }.get(row['layer'], f"Layer {row['layer']}")
            log(f"  {layer_name}: {row['count']} total, {row['active_count']} active")
        
        # Check for any unmigrated records
        unmigrated_query = "SELECT COUNT(*) as count FROM companies WHERE layer IS NULL"
        unmigrated = await conn.fetchrow(unmigrated_query)
        
        if unmigrated and unmigrated['count'] > 0:
            log(f"Found {unmigrated['count']} unmigrated records", "WARNING")
        else:
            log("All records successfully migrated")
        
        # Sample some migrated records
        sample_query = """
            SELECT symbol, layer, layer_reason, layer_updated_at
            FROM companies
            WHERE layer > 0
            ORDER BY layer DESC, symbol
            LIMIT 10
        """
        
        samples = await conn.fetch(sample_query)
        
        if samples:
            log("Sample migrated records:")
            for row in samples:
                log(f"  {row['symbol']}: Layer {row['layer']} - {row['layer_reason']}")
        
        log("Migration completed successfully!")
        
        # Note about old columns
        log("\nNOTE: Old columns (layer1_qualified, layer2_qualified, layer3_qualified) are still present")
        log("They will be dropped after verifying the new system works correctly")
        
        await conn.close()
        
    except asyncpg.InvalidCatalogNameError:
        log(f"Database '{db_name}' not found. Trying to list available databases...", "ERROR")
        
        # Connect to default postgres database to list databases
        try:
            dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/postgres"
            conn = await asyncpg.connect(dsn)
            
            databases = await conn.fetch("SELECT datname FROM pg_database WHERE datistemplate = false")
            log("Available databases:")
            for db in databases:
                log(f"  - {db['datname']}")
            
            await conn.close()
            log("\nPlease set DB_NAME environment variable to the correct database name and retry")
            
        except Exception as e:
            log(f"Failed to list databases: {e}", "ERROR")
            
    except Exception as e:
        log(f"Migration failed: {e}", "ERROR")
        raise


async def verify_migration():
    """Verify the migration was successful."""
    
    # Get database connection details
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER", "zachwade")
    db_pass = os.getenv("DB_PASS", "ZachT$2002")
    
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    try:
        conn = await asyncpg.connect(dsn)
        
        # Check that layer column exists
        column_check = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'companies' 
            AND column_name IN ('layer', 'layer_updated_at', 'layer_reason')
            ORDER BY column_name
        """
        
        columns = await conn.fetch(column_check)
        
        log("New columns added:")
        for col in columns:
            log(f"  {col['column_name']}: {col['data_type']}")
        
        if len(columns) != 3:
            log("Not all new columns were added!", "ERROR")
            await conn.close()
            return False
        
        await conn.close()
        return True
        
    except Exception as e:
        log(f"Verification failed: {e}", "ERROR")
        return False


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Migrate companies table to layer-based system")
    parser.add_argument("--verify-only", action="store_true", help="Only verify migration, don't run it")
    parser.add_argument("--db-name", help="Database name (can also use DB_NAME env var)")
    
    args = parser.parse_args()
    
    if args.db_name:
        os.environ["DB_NAME"] = args.db_name
    
    if args.verify_only:
        success = asyncio.run(verify_migration())
        sys.exit(0 if success else 1)
    else:
        asyncio.run(run_migration())
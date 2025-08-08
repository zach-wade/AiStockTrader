#!/usr/bin/env python
"""
Initialize Database Script
Sets up the PostgreSQL database schema and initial data.
"""

import sys
from pathlib import Path
import asyncio
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai_trader.utils.database import get_db_engine, Base
from ai_trader.data_pipeline.storage.models import (
    MarketData, Aggregates, News, Features, ScannerAlerts
)
from ai_trader.utils.core import setup_logger

logger = setup_logger(__name__)
load_dotenv()


def create_tables():
    """Create all database tables."""
    engine = get_db_engine()
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Create indexes
        with engine.connect() as conn:
            # Market data indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
                ON market_data(timestamp DESC);
            """))
            
            # Aggregates indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_aggregates_symbol_timeframe 
                ON aggregates(symbol, timeframe, timestamp DESC);
            """))
            
            # Features indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
                ON features(symbol, timestamp DESC);
            """))
            
            # Scanner alerts indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scanner_alerts_timestamp 
                ON scanner_alerts(timestamp DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scanner_alerts_symbol 
                ON scanner_alerts(symbol, timestamp DESC);
            """))
            
            conn.commit()
            
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def create_partitions():
    """Create time-based partitions for large tables."""
    engine = get_db_engine()
    
    try:
        with engine.connect() as conn:
            # Create partitioned market_data table
            conn.execute(text("""
                -- Create parent table if not exists
                CREATE TABLE IF NOT EXISTS market_data_partitioned (
                    LIKE market_data INCLUDING ALL
                ) PARTITION BY RANGE (timestamp);
                
                -- Create monthly partitions for the next 12 months
                DO $$
                DECLARE
                    start_date date := date_trunc('month', CURRENT_DATE);
                    end_date date;
                    partition_name text;
                BEGIN
                    FOR i IN 0..11 LOOP
                        end_date := start_date + interval '1 month';
                        partition_name := 'market_data_' || to_char(start_date, 'YYYY_MM');
                        
                        EXECUTE format('
                            CREATE TABLE IF NOT EXISTS %I PARTITION OF market_data_partitioned
                            FOR VALUES FROM (%L) TO (%L)',
                            partition_name, start_date, end_date
                        );
                        
                        start_date := end_date;
                    END LOOP;
                END $$;
            """))
            
            conn.commit()
            logger.info("Database partitions created successfully")
            
    except Exception as e:
        logger.warning(f"Failed to create partitions (may not be supported): {str(e)}")


def create_views():
    """Create useful database views."""
    engine = get_db_engine()
    
    try:
        with engine.connect() as conn:
            # Latest prices view
            conn.execute(text("""
                CREATE OR REPLACE VIEW latest_prices AS
                SELECT DISTINCT ON (symbol) 
                    symbol, timestamp, open, high, low, close, volume
                FROM market_data
                ORDER BY symbol, timestamp DESC;
            """))
            
            # Daily summary view
            conn.execute(text("""
                CREATE OR REPLACE VIEW daily_summary AS
                SELECT 
                    symbol,
                    date(timestamp) as trading_date,
                    MIN(timestamp) as first_tick,
                    MAX(timestamp) as last_tick,
                    FIRST_VALUE(open) OVER (PARTITION BY symbol, date(timestamp) ORDER BY timestamp) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    LAST_VALUE(close) OVER (PARTITION BY symbol, date(timestamp) ORDER BY timestamp) as close,
                    SUM(volume) as volume,
                    COUNT(*) as tick_count
                FROM market_data
                GROUP BY symbol, date(timestamp);
            """))
            
            # Active alerts view
            conn.execute(text("""
                CREATE OR REPLACE VIEW active_alerts AS
                SELECT *
                FROM scanner_alerts
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC;
            """))
            
            conn.commit()
            logger.info("Database views created successfully")
            
    except Exception as e:
        logger.error(f"Failed to create views: {str(e)}")
        raise


def insert_initial_data():
    """Insert any initial/reference data."""
    engine = get_db_engine()
    
    try:
        with engine.connect() as conn:
            # Add any reference data here
            # For example, a list of tracked symbols, exchanges, etc.
            
            logger.info("Initial data inserted successfully")
            
    except Exception as e:
        logger.warning(f"Failed to insert initial data: {str(e)}")


def main():
    """Initialize the database."""
    print("🗄️  Initializing AI Trader Database\n" + "="*50)
    
    try:
        # Create tables
        print("Creating tables...")
        create_tables()
        print("✅ Tables created")
        
        # Create partitions
        print("\nCreating partitions...")
        create_partitions()
        print("✅ Partitions created")
        
        # Create views
        print("\nCreating views...")
        create_views()
        print("✅ Views created")
        
        # Insert initial data
        print("\nInserting initial data...")
        insert_initial_data()
        print("✅ Initial data inserted")
        
        print("\n" + "="*50)
        print("✅ Database initialization complete!")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
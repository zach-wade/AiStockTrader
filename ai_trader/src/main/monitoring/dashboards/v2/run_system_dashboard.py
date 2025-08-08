#!/usr/bin/env python3
"""
Standalone script to run the System Dashboard V2.

This script is designed to be run as a separate process to avoid
multiprocessing/pickling issues.
"""

import sys
import os
import asyncio
import logging
import argparse
import json
from pathlib import Path

# Setup Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent  # Navigate up to ai_trader
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run System Dashboard V2')
    parser.add_argument('--config', required=True, help='Database config JSON')
    parser.add_argument('--port', type=int, default=8052, help='Dashboard port')
    args = parser.parse_args()
    
    # Parse database config
    try:
        db_config = json.loads(args.config)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse database config: {e}")
        sys.exit(1)
    
    # Load environment variables
    try:
        from main.config.env_loader import ensure_environment_loaded
        ensure_environment_loaded()
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        sys.exit(1)
    
    # Import dashboard components
    try:
        from main.monitoring.dashboards.v2.system_dashboard_v2 import SystemDashboardV2
        from main.utils.database import DatabasePool
    except Exception as e:
        logger.error(f"Failed to import dashboard modules: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info(f"Starting System Dashboard V2 on port {args.port}")
    
    try:
        # Create database pool
        logger.info("Creating database pool...")
        db_pool = DatabasePool()
        
        # Build database URL
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        logger.info(f"Initializing database pool with URL: postgresql://{db_config['user']}:****@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        
        # Initialize with database URL (synchronous method)
        db_pool.initialize(database_url=db_url)
        logger.info("Database pool initialized successfully")
        
        # Create and run dashboard
        logger.info("Creating dashboard instance...")
        # Note: orchestrator=None for now, can be passed via args if needed
        dashboard = SystemDashboardV2(db_pool, orchestrator=None, port=args.port)
        logger.info("Starting dashboard server...")
        dashboard.run(debug=False)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            # Note: DatabasePool doesn't have a close method, it manages its own lifecycle
            logger.info("Dashboard shutdown complete")
        except Exception:
            pass  # Pool manages its own lifecycle


if __name__ == "__main__":
    main()
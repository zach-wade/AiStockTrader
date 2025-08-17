#!/usr/bin/env python3
"""
AI Trader - Paper Trading Startup Helper

This script provides a simplified way to start the AI Trader system
in paper trading mode with all recommended settings.

Usage:
    python start_paper_trading.py [symbols] [--enable-ml]

Examples:
    python start_paper_trading.py  # Uses default symbols
    python start_paper_trading.py AAPL,MSFT,GOOGL  # Custom symbols
    python start_paper_trading.py AAPL,MSFT --enable-ml  # With ML enabled
"""

# Standard library imports
import os
from pathlib import Path
import subprocess
import sys
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")

    # Check .env file
    if not os.path.exists(".env"):
        print("❌ Error: .env file not found")
        print("\nPlease create a .env file with the following variables:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_SECRET_KEY=your_secret")
        print("  POLYGON_API_KEY=your_key")
        print("  DB_HOST=localhost")
        print("  DB_PORT=5432")
        print("  DB_NAME=ai_trader")
        print("  DB_USER=your_user")
        print("  DB_PASSWORD=your_password")
        return False

    # Load and check environment variables
    # Local imports
    from main.config.env_loader import ensure_environment_loaded

    ensure_environment_loaded()

    required_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False

    print("✅ Environment configured correctly")
    return True


def check_database():
    """Check database connection"""
    print("\n🔍 Checking database connection...")

    try:
        # Local imports
        from main.utils.database import DatabasePool

        pool = DatabasePool()
        pool.initialize()

        # Test connection
        with pool.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            if result:
                print("✅ Database connection successful")
                return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nPlease ensure PostgreSQL is running and credentials are correct")
        return False

    return False


def check_market_status():
    """Check if market is open"""
    print("\n🔍 Checking market status...")

    # Standard library imports
    from datetime import datetime

    # Local imports
    from main.utils.core import is_market_open

    market_open = is_market_open()
    now = datetime.now()

    if market_open:
        print("✅ Market is OPEN")
    else:
        print("⚠️  Market is CLOSED")
        print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("   Note: Paper trading will work but live data may be limited")

    return True


def start_paper_trading(symbols=None, enable_ml=False):
    """Start the paper trading system"""
    # Default symbols
    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("\n🚀 Starting AI Trader Paper Trading System")
    print("=" * 50)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"ML Trading: {'Enabled' if enable_ml else 'Disabled'}")
    print("Dashboard: http://localhost:8080")
    print("WebSocket: ws://localhost:8081")
    print("=" * 50)

    # Build command
    cmd = [
        sys.executable,
        "ai_trader.py",
        "trade",
        "--mode",
        "paper",
        "--symbols",
        ",".join(symbols),
        "--enable-monitoring",
        "--enable-streaming",
    ]

    if enable_ml:
        cmd.append("--enable-ml")

    print(f"\nCommand: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the system\n")

    # Start the system
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n\n✅ System stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


def main():
    """Main entry point"""
    print("🤖 AI Trader - Paper Trading Startup")
    print("=" * 50)

    # Parse arguments
    symbols = None
    enable_ml = False

    args = sys.argv[1:]
    if args:
        # Check for --enable-ml flag
        if "--enable-ml" in args:
            enable_ml = True
            args.remove("--enable-ml")

        # Remaining argument should be symbols
        if args:
            symbols = args[0].split(",")

    # Run checks
    if not check_environment():
        return 1

    if not check_database():
        return 1

    check_market_status()

    # Optional: Check if we need to populate universe
    print("\n🔍 Checking universe data...")
    try:
        # Standard library imports
        import asyncio

        # Local imports
        from main.config.config_manager import get_config
        from main.universe.universe_manager import UniverseManager

        config = get_config()

        async def check_universe():
            manager = UniverseManager(config)
            try:
                health = await manager.health_check()
                if not health["healthy"] or health["companies_count"] < 100:
                    print("⚠️  Universe data is limited")
                    print("   Consider running: python ai_trader.py universe --populate")
                else:
                    print(f"✅ Universe has {health['companies_count']} companies")
            finally:
                await manager.close()

        asyncio.run(check_universe())
    except Exception as e:
        print(f"⚠️  Could not check universe: {e}")

    # Start the system
    print("\n" + "=" * 50)
    time.sleep(1)  # Brief pause before starting

    return start_paper_trading(symbols, enable_ml)


if __name__ == "__main__":
    sys.exit(main())

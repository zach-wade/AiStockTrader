"""
CLI Commands Implementation

This module implements the actual command logic for the AI Trader CLI.
Each command function is called from the main ai_trader.py CLI entry point.
"""

import logging
import os
from decimal import Decimal
from typing import Any

from src.application.config import BrokerConfig, DatabaseConfig
from src.infrastructure.brokers.alpaca_broker import AlpacaBroker
from src.infrastructure.brokers.paper_broker import PaperBroker

logger = logging.getLogger(__name__)


async def check_system_status() -> dict[str, Any]:
    """
    Check the status of all system components.

    Returns:
        Dictionary containing status information for all components
    """
    status = {
        "database": {"status": "Unknown", "connection": "Not checked", "tables": 0},
        "broker": {"status": "Unknown", "type": "Not configured", "connected": False},
        "market_data": {"status": "Unknown", "provider": "Not configured"},
        "trading": {"positions": 0, "open_orders": 0, "account_value": 0.0},
        "config": {"environment": os.getenv("ENVIRONMENT", "development")},
        "overall_status": "Unknown",
    }

    try:
        # Check database connection
        db_status = await check_database_status()
        status["database"] = db_status

        # Check broker connection
        broker_status = await check_broker_status()
        status["broker"] = broker_status

        # Check market data provider
        market_status = await check_market_data_status()
        status["market_data"] = market_status

        # Get trading status if broker is connected
        if broker_status["connected"]:
            trading_status = await get_trading_status(broker_status.get("broker_instance"))
            status["trading"] = trading_status

        # Determine overall status
        if status["database"]["status"] == "Connected" and status["broker"]["status"] == "Ready":
            status["overall_status"] = "Operational"
        elif status["database"]["status"] == "Connected":
            status["overall_status"] = "Partially Operational"
        else:
            status["overall_status"] = "Offline"

    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        status["overall_status"] = "Error"

    return status


async def check_database_status() -> dict[str, Any]:
    """Check database connection and status."""
    try:
        # Create database config from environment
        db_config = DatabaseConfig.from_env()

        # Try to connect
        from psycopg_pool import AsyncConnectionPool

        connection_string = (
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )

        async with AsyncConnectionPool(
            connection_string, min_size=1, max_size=2, open=True
        ) as pool:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Check connection
                    await cur.execute("SELECT 1")
                    await cur.fetchone()

                    # Count tables
                    await cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                    """
                    )
                    table_count = (await cur.fetchone())[0]

                    return {
                        "status": "Connected",
                        "connection": f"{db_config.host}:{db_config.port}/{db_config.database}",
                        "tables": table_count,
                    }

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return {"status": "Disconnected", "connection": f"Error: {str(e)[:50]}", "tables": 0}


async def check_broker_status() -> dict[str, Any]:
    """Check broker connection and status."""
    try:
        broker_config = BrokerConfig.from_env()
        broker_type = broker_config.type

        if broker_type == "alpaca":
            # Check Alpaca broker
            broker = AlpacaBroker(paper=True)
            try:
                broker.connect()
                account_info = broker.get_account_info()

                return {
                    "status": "Ready",
                    "type": "Alpaca (Paper)" if broker.paper else "Alpaca (Live)",
                    "connected": True,
                    "broker_instance": broker,
                    "account_value": float(account_info.equity),
                }
            except Exception as e:
                logger.error(f"Alpaca broker error: {e}")
                return {
                    "status": "Error",
                    "type": "Alpaca",
                    "connected": False,
                    "error": str(e)[:50],
                }
        else:
            # Default to paper broker
            broker = PaperBroker(initial_capital=Decimal("100000"))
            broker.connect()

            return {
                "status": "Ready",
                "type": "Paper (Simulation)",
                "connected": True,
                "broker_instance": broker,
            }

    except Exception as e:
        logger.error(f"Broker check failed: {e}")
        return {"status": "Error", "type": "Unknown", "connected": False}


async def check_market_data_status() -> dict[str, Any]:
    """Check market data provider status."""
    try:
        # Check if Alpaca market data is available
        alpaca_key = os.getenv("ALPACA_API_KEY")
        polygon_key = os.getenv("POLYGON_API_KEY")

        if polygon_key:
            provider = "Polygon.io"
        elif alpaca_key:
            provider = "Alpaca Market Data"
        else:
            provider = "None configured"

        # Try to fetch a test quote
        if alpaca_key:
            from paper_trading_alpaca import AlpacaMarketData

            try:
                market_data = AlpacaMarketData()
                price = market_data.get_latest_price("SPY")
                return {
                    "status": "Connected",
                    "provider": provider,
                    "test_quote": f"SPY: ${float(price):.2f}",
                }
            except Exception as e:
                logger.error(f"Market data test failed: {e}")
                return {"status": "Error", "provider": provider, "error": str(e)[:50]}

        return {"status": "Not configured", "provider": provider}

    except Exception as e:
        logger.error(f"Market data check failed: {e}")
        return {"status": "Error", "provider": "Unknown"}


async def get_trading_status(broker) -> dict[str, Any]:
    """Get current trading status from broker."""
    try:
        if broker and hasattr(broker, "get_account_info"):
            account_info = broker.get_account_info()

            # Get positions count
            positions = []
            if hasattr(broker, "get_positions"):
                positions = broker.get_positions() or []

            # Get open orders count
            orders = []
            if hasattr(broker, "get_open_orders"):
                orders = broker.get_open_orders() or []

            return {
                "positions": len(positions),
                "open_orders": len(orders),
                "account_value": float(account_info.equity),
            }

    except Exception as e:
        logger.error(f"Failed to get trading status: {e}")

    return {"positions": 0, "open_orders": 0, "account_value": 0.0}

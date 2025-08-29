"""
Resilience Integration Demo

Demonstrates how to integrate the resilience infrastructure with
existing trading system components for production use.
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from src.application.interfaces.broker import AccountInfo, IBroker, MarketHours
from src.application.interfaces.market_data import IMarketDataProvider
from src.domain.entities.order import Order, OrderStatus
from src.domain.entities.position import Position
from src.infrastructure.database.connection import DatabaseConfig
from src.infrastructure.resilience.config import ConfigManager
from src.infrastructure.resilience.database import EnhancedDatabaseConfig
from src.infrastructure.resilience.integration import ResilienceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockBroker(IBroker):
    """Mock broker for demonstration."""

    def __init__(self, name: str, failure_rate: float = 0.0) -> None:
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0

    def submit_order(self, order: Order) -> Order:
        """Mock submit order."""
        self.call_count += 1
        import random

        if random.random() < self.failure_rate:
            raise ConnectionError(f"Mock {self.name} broker connection error")
        # Return order with broker ID set
        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Mock cancel order."""
        return True

    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """Mock get order status."""
        return OrderStatus.SUBMITTED

    def get_positions(self) -> list[Position]:
        """Mock get positions."""
        return []

    def get_account_info(self) -> AccountInfo:
        """Mock get account info."""
        from decimal import Decimal

        return AccountInfo(
            account_id=f"{self.name}_account",
            account_type="paper",
            equity=Decimal("10000.0"),
            cash=Decimal("10000.0"),
            buying_power=Decimal("10000.0"),
            positions_value=Decimal("0.0"),
            unrealized_pnl=Decimal("0.0"),
            realized_pnl=Decimal("0.0"),
        )

    def is_market_open(self) -> bool:
        """Mock market open check."""
        return True

    def get_market_hours(self) -> MarketHours:
        """Mock get market hours."""
        return MarketHours(is_open=True)

    def update_order(self, order: Order) -> Order:
        """Mock update order."""
        return order

    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Mock get recent orders."""
        return []

    def connect(self) -> None:
        """Mock connect."""
        pass

    def disconnect(self) -> None:
        """Mock disconnect."""
        pass

    def is_connected(self) -> bool:
        """Mock is connected."""
        return True


class MockMarketData(IMarketDataProvider):
    """Mock market data provider for demonstration."""

    def __init__(self, name: str, failure_rate: float = 0.0) -> None:
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0

    async def get_current_price(self, symbol: str) -> Any:
        """Mock get current price."""
        from decimal import Decimal

        from src.domain.value_objects.price import Price

        self.call_count += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Mock {self.name} market data connection error")

        price_value = Decimal(str(round(100 + random.uniform(-10, 10), 2)))
        return Price(price_value)

    async def get_current_quote(self, symbol: str) -> Any:
        """Mock get current quote."""
        from datetime import datetime
        from decimal import Decimal

        from src.application.interfaces.market_data import Quote
        from src.domain.value_objects.price import Price
        from src.domain.value_objects.symbol import Symbol

        self.call_count += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Mock {self.name} market data connection error")

        base_price = Decimal(str(round(100 + random.uniform(-10, 10), 2)))
        bid_price = Price(base_price - Decimal("0.01"))
        ask_price = Price(base_price + Decimal("0.01"))

        return Quote(
            symbol=Symbol(symbol),
            timestamp=datetime.now(),
            bid_price=bid_price,
            bid_size=100,
            ask_price=ask_price,
            ask_size=100,
        )

    async def get_historical_bars(
        self, symbol: str, start: Any, end: Any, timeframe: str = "1min"
    ) -> list[Any]:
        """Mock get historical bars."""
        self.call_count += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Mock {self.name} market data connection error")
        return []

    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[Any]:
        """Mock stream prices."""

        # This would be an async generator in a real implementation
        # Return empty async iterator for demo
        async def empty_stream() -> AsyncIterator[Any]:
            # Empty generator - never yields anything
            return
            yield None  # pragma: no cover

        return empty_stream()

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Any]:
        """Mock stream quotes."""

        # This would be an async generator in a real implementation
        # Return empty async iterator for demo
        async def empty_stream() -> AsyncIterator[Any]:
            # Empty generator - never yields anything
            return
            yield None  # pragma: no cover

        return empty_stream()

    async def is_market_open(self) -> bool:
        """Mock market open check."""
        return True

    async def get_symbol_info(self, symbol: str) -> dict[str, str]:
        """Mock get symbol info."""
        return {"symbol": symbol, "name": f"Mock {symbol}", "type": "stock"}


async def demonstrate_resilience_features() -> None:
    """Demonstrate the complete resilience infrastructure."""

    logger.info("=== Resilience Infrastructure Demo ===")

    # 1. Load Configuration
    logger.info("1. Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load_config()
    logger.info(f"Configuration loaded for environment: {config.environment.value}")

    # 2. Create Resilience Factory
    logger.info("2. Initializing resilience factory...")
    factory = ResilienceFactory(config)
    await factory.initialize_resilience()
    logger.info("Resilience infrastructure initialized")

    try:
        # 3. Create Mock Services
        logger.info("3. Creating mock services...")
        primary_broker = MockBroker("AlpacaPrimary", failure_rate=0.3)  # 30% failure rate
        backup_broker = MockBroker("AlpacaBackup", failure_rate=0.1)  # 10% failure rate

        primary_data = MockMarketData("Polygon", failure_rate=0.2)  # 20% failure rate
        backup_data = MockMarketData("AlphaVantage", failure_rate=0.1)  # 10% failure rate

        # 4. Create Resilient Wrappers
        logger.info("4. Creating resilient service wrappers...")
        resilient_primary_broker = factory.create_resilient_broker(primary_broker, "primary_broker")
        resilient_backup_broker = factory.create_resilient_broker(backup_broker, "backup_broker")

        resilient_primary_data = factory.create_resilient_market_data(primary_data, "primary_data")
        resilient_backup_data = factory.create_resilient_market_data(backup_data, "backup_data")

        logger.info("Resilient wrappers created and registered")

        # 5. Demonstrate Operations with Resilience
        logger.info("5. Testing resilient operations...")

        # Test broker operations
        for i in range(10):
            try:
                account_info = await resilient_primary_broker.get_account_info()
                logger.info(f"Broker call {i+1} succeeded: {account_info['account_id']}")
            except Exception as e:
                logger.error(f"Broker call {i+1} failed: {e}")

            # Brief pause
            await asyncio.sleep(0.1)

        # Test market data operations
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for symbol in symbols:
            try:
                price_data = await resilient_primary_data.get_current_price(symbol)
                logger.info(
                    f"Price for {symbol}: ${price_data['price']} from {price_data['provider']}"
                )
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")

        # 6. Monitor System Health
        logger.info("6. Checking system health...")
        await asyncio.sleep(1.0)  # Let health checks run

        orchestrator = factory.get_orchestrator()
        health_summary = await orchestrator.get_system_health()

        logger.info(f"Overall system status: {health_summary['overall_status']}")
        logger.info(f"Components monitored: {len(health_summary['components'])}")

        # Display service health
        for service_name, service_data in health_summary["health_checks"].items():
            status = service_data.get("status", "unknown")
            logger.info(f"  {service_name}: {status}")

        # Display circuit breaker status
        cb_metrics = health_summary.get("circuit_breakers", {})
        for cb_name, cb_data in cb_metrics.items():
            state = cb_data.get("state", "unknown")
            success_rate = cb_data.get("success_rate", 0.0)
            logger.info(f"  Circuit breaker {cb_name}: {state} (success rate: {success_rate:.2%})")

        # 7. Demonstrate Failure Recovery
        logger.info("7. Testing failure recovery...")

        # Simulate service recovery
        primary_broker.failure_rate = 0.0  # Fix the service
        primary_data.failure_rate = 0.0

        logger.info("Services recovered, testing operations...")

        # Wait for circuit breakers to potentially recover
        await asyncio.sleep(2.0)

        # Test operations again
        try:
            account_info = await resilient_primary_broker.get_account_info()
            logger.info(
                f"Broker recovery test: {account_info['account_id']} (calls: {account_info['calls']})"
            )
        except Exception as e:
            logger.error(f"Broker recovery test failed: {e}")

        try:
            price_data = await resilient_primary_data.get_current_price("AAPL")
            logger.info(f"Market data recovery test: AAPL at ${price_data['price']}")
        except Exception as e:
            logger.error(f"Market data recovery test failed: {e}")

        # 8. Final Health Check
        logger.info("8. Final system health check...")
        final_health = await orchestrator.get_system_health()
        logger.info(f"Final system status: {final_health['overall_status']}")

        # Display error metrics
        error_metrics = final_health.get("error_handling", {})
        total_errors = error_metrics.get("total_errors", 0)
        logger.info(f"Total errors handled: {total_errors}")

        if total_errors > 0:
            by_category = error_metrics.get("by_category", {})
            for category, count in by_category.items():
                if count > 0:
                    logger.info(f"  {category}: {count} errors")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

    finally:
        # 9. Cleanup
        logger.info("9. Shutting down resilience infrastructure...")
        await factory.shutdown_resilience()
        logger.info("Demo completed")


async def demonstrate_database_resilience() -> None:
    """Demonstrate database resilience features."""

    logger.info("=== Database Resilience Demo ===")

    try:
        # Note: This requires a real PostgreSQL database to be running
        # For demo purposes, we'll show the configuration

        base_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="ai_trader_test",
            user="test_user",
            password="test_password",
        )

        enhanced_config = EnhancedDatabaseConfig.from_base_config(
            base_config,
            circuit_breaker_enabled=True,
            retry_enabled=True,
            health_check_enabled=True,
            pool_pre_ping=True,
            enable_query_logging=True,
            slow_query_threshold=0.5,
        )

        logger.info("Enhanced database configuration created:")
        logger.info(f"  Host: {enhanced_config.host}:{enhanced_config.port}")
        logger.info(f"  Database: {enhanced_config.database}")
        logger.info(f"  Pool size: {enhanced_config.min_pool_size}-{enhanced_config.max_pool_size}")
        logger.info(f"  Circuit breaker: {enhanced_config.circuit_breaker_enabled}")
        logger.info(f"  Retry enabled: {enhanced_config.retry_enabled}")
        logger.info(f"  Health checks: {enhanced_config.health_check_enabled}")

        # Example of how to create resilient database connection
        logger.info("Would create resilient database connection with:")
        logger.info("  - Connection pool with health monitoring")
        logger.info("  - Circuit breaker for database failures")
        logger.info("  - Automatic retry on transient errors")
        logger.info("  - Query performance monitoring")
        logger.info("  - Pre-ping connection validation")

    except Exception as e:
        logger.error(f"Database demo setup failed: {e}")


def demonstrate_configuration_management() -> None:
    """Demonstrate configuration management features."""

    logger.info("=== Configuration Management Demo ===")

    config_manager = ConfigManager()

    # Load configuration
    config = config_manager.load_config()

    logger.info("Configuration loaded:")
    logger.info(f"  Environment: {config.environment.value}")
    logger.info(f"  Debug mode: {config.debug}")
    logger.info(f"  Log level: {config.log_level}")

    # Resilience settings
    resilience = config.resilience
    logger.info("Resilience configuration:")
    logger.info(f"  Circuit breaker enabled: {resilience.circuit_breaker_enabled}")
    logger.info(f"  Failure threshold: {resilience.circuit_breaker_failure_threshold}")
    logger.info(f"  Retry enabled: {resilience.retry_enabled}")
    logger.info(f"  Max retry attempts: {resilience.retry_max_attempts}")
    logger.info(f"  Health checks enabled: {resilience.health_check_enabled}")

    # Feature flags
    features = config.features
    logger.info("Feature flags:")
    logger.info(f"  Paper trading: {features.paper_trading_enabled}")
    logger.info(f"  Live trading: {features.live_trading_enabled}")
    logger.info(f"  Real-time data: {features.real_time_data_enabled}")
    logger.info(f"  Risk limits: {features.risk_limits_enforced}")
    logger.info(f"  Metrics collection: {features.metrics_collection_enabled}")

    # Demonstrate runtime feature toggle
    logger.info("Demonstrating runtime feature toggle...")
    original_state = features.machine_learning_enabled
    logger.info(f"ML feature before: {original_state}")

    config_manager.update_feature_flag("machine_learning_enabled", True)
    logger.info(f"ML feature after: {features.machine_learning_enabled}")

    # Reset
    config_manager.update_feature_flag("machine_learning_enabled", original_state)
    logger.info(f"ML feature reset: {features.machine_learning_enabled}")


async def main() -> None:
    """Main demo function."""
    print("🚀 AI Trading System Resilience Demo\n")

    try:
        # 1. Configuration Management Demo
        demonstrate_configuration_management()
        print("\n" + "=" * 50 + "\n")

        # 2. Database Resilience Demo
        await demonstrate_database_resilience()
        print("\n" + "=" * 50 + "\n")

        # 3. Full Resilience Infrastructure Demo
        await demonstrate_resilience_features()

        print("\n✅ All demos completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("\n❌ Demo failed - see logs for details")
        raise


if __name__ == "__main__":
    asyncio.run(main())

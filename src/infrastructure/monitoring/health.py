"""
Health Check Endpoints for AI Trading System

Deep health checks for all services with:
- Market hours awareness
- Dependency health (database, brokers, market data)
- Health check aggregation
- Trading system specific checks
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast

import pytz

from src.domain.services.market_hours_service import MarketHoursService
from src.domain.services.market_hours_service import MarketStatus as DomainMarketStatus
from src.infrastructure.time.timezone_service import LocalizedDatetimeAdapter, PythonTimeService

from ..resilience.health import HealthCheck, HealthChecker, HealthCheckResult, HealthStatus
from .telemetry import trace_trading_operation

logger = logging.getLogger(__name__)


# Re-export MarketStatus from domain for backward compatibility
class MarketStatus(Enum):
    """Market status enumeration - delegates to domain."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_MARKET = "after_market"
    HOLIDAY = "holiday"
    UNKNOWN = "unknown"

    @classmethod
    def from_domain(cls, domain_status: DomainMarketStatus) -> "MarketStatus":
        """Convert domain MarketStatus to infrastructure MarketStatus."""
        mapping = {
            DomainMarketStatus.OPEN: cls.OPEN,
            DomainMarketStatus.CLOSED: cls.CLOSED,
            DomainMarketStatus.PRE_MARKET: cls.PRE_MARKET,
            DomainMarketStatus.AFTER_MARKET: cls.AFTER_MARKET,
            DomainMarketStatus.HOLIDAY: cls.HOLIDAY,
        }
        return mapping.get(domain_status, cls.UNKNOWN)


@dataclass
class MarketHours:
    """Market hours configuration."""

    timezone: str = "America/New_York"
    regular_open_hour: int = 9
    regular_open_minute: int = 30
    regular_close_hour: int = 16
    regular_close_minute: int = 0

    pre_market_open_hour: int = 4
    pre_market_open_minute: int = 0
    after_market_close_hour: int = 20
    after_market_close_minute: int = 0

    # Market holidays (will be checked against external API in production)
    holidays: set[str] = field(
        default_factory=lambda: {
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # Martin Luther King Jr. Day
            "2024-02-19",  # Presidents Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving Day
            "2024-12-25",  # Christmas Day
        }
    )


class MarketHoursChecker:
    """Check current market status and hours."""

    def __init__(self, market_hours: MarketHours | None = None) -> None:
        self.market_hours = market_hours or MarketHours()
        self.timezone = pytz.timezone(self.market_hours.timezone)

        # Initialize domain service for market hours logic
        time_service = PythonTimeService()
        self._market_hours_service = MarketHoursService(
            time_service=time_service,
            timezone=self.market_hours.timezone,
            holidays=(
                set(self.market_hours.holidays) if hasattr(self.market_hours, "holidays") else None
            ),
            pre_market_open=datetime.min.time().replace(
                hour=self.market_hours.pre_market_open_hour,
                minute=self.market_hours.pre_market_open_minute,
            ),
            regular_open=datetime.min.time().replace(
                hour=self.market_hours.regular_open_hour,
                minute=self.market_hours.regular_open_minute,
            ),
            regular_close=datetime.min.time().replace(
                hour=self.market_hours.regular_close_hour,
                minute=self.market_hours.regular_close_minute,
            ),
            after_market_close=datetime.min.time().replace(
                hour=self.market_hours.after_market_close_hour,
                minute=self.market_hours.after_market_close_minute,
            ),
        )

    def get_current_market_status(self) -> MarketStatus:
        """
        Get current market status - delegates to domain service.

        This method now delegates all business logic to the domain service.
        The infrastructure layer only handles the conversion and delegation.
        """
        now = datetime.now(self.timezone)
        domain_status = self._market_hours_service.get_current_market_status(now)
        return MarketStatus.from_domain(domain_status)

    def _is_holiday(self, date: datetime) -> bool:
        """
        Check if date is a market holiday - delegates to domain service.

        This method is kept for backward compatibility but delegates
        all business logic to the domain service.
        """
        # Convert datetime to LocalizedDatetime for domain service
        if date.tzinfo is None:
            # Add timezone info if it's naive
            date = self.timezone.localize(date)
        from src.domain.interfaces.time_service import LocalizedDatetime

        localized_date = cast(LocalizedDatetime, LocalizedDatetimeAdapter(date))
        return self._market_hours_service.is_holiday(localized_date)

    def get_market_info(self) -> dict[str, Any]:
        """Get comprehensive market information."""
        now = datetime.now(self.timezone)
        status = self.get_current_market_status()

        next_open = self._get_next_market_open(now)
        next_close = self._get_next_market_close(now)

        return {
            "status": status.value,
            "current_time": now.isoformat(),
            "timezone": self.market_hours.timezone,
            "is_trading_day": status not in (MarketStatus.HOLIDAY, MarketStatus.CLOSED),
            "next_open": next_open.isoformat() if next_open else None,
            "next_close": next_close.isoformat() if next_close else None,
        }

    def _get_next_market_open(self, from_time: datetime) -> datetime | None:
        """Get next market open time."""
        current = from_time

        # Look up to 7 days ahead
        for _ in range(7):
            # Skip weekends and holidays
            if current.weekday() < 5 and not self._is_holiday(current):
                market_open = current.replace(
                    hour=self.market_hours.regular_open_hour,
                    minute=self.market_hours.regular_open_minute,
                    second=0,
                    microsecond=0,
                )

                if market_open > from_time:
                    return market_open

            # Move to next day
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)
            current += timedelta(days=1)

        return None

    def _get_next_market_close(self, from_time: datetime) -> datetime | None:
        """Get next market close time."""
        current = from_time

        # If market is currently open, return today's close
        if self.get_current_market_status() == MarketStatus.OPEN:
            return current.replace(
                hour=self.market_hours.regular_close_hour,
                minute=self.market_hours.regular_close_minute,
                second=0,
                microsecond=0,
            )

        # Otherwise find next trading day's close
        next_open = self._get_next_market_open(from_time)
        if next_open:
            return next_open.replace(
                hour=self.market_hours.regular_close_hour,
                minute=self.market_hours.regular_close_minute,
            )

        return None


class MarketHoursHealthCheck(HealthCheck):
    """Health check that considers market hours."""

    def __init__(
        self,
        name: str,
        underlying_check: HealthCheck,
        market_hours_checker: MarketHoursChecker,
        market_hours_required: bool = False,
        timeout: float = 30.0,
    ):
        super().__init__(name, timeout)
        self.underlying_check = underlying_check
        self.market_hours_checker = market_hours_checker
        self.market_hours_required = market_hours_required

    async def check_health(self) -> HealthCheckResult:
        """Check health with market hours awareness."""
        start_time = time.time()

        # Get market status
        market_status = self.market_hours_checker.get_current_market_status()
        market_info = self.market_hours_checker.get_market_info()

        # Run underlying health check
        underlying_result = await self.underlying_check.safe_check_health()

        # Determine final status
        final_status = underlying_result.status

        # If market hours are required and market is closed, degrade the status
        if (
            self.market_hours_required
            and market_status in (MarketStatus.CLOSED, MarketStatus.HOLIDAY)
            and underlying_result.status == HealthStatus.HEALTHY
        ):
            final_status = HealthStatus.DEGRADED

        response_time = time.time() - start_time

        # Combine details
        combined_details = {
            **underlying_result.details,
            "market_status": market_status.value,
            "market_info": market_info,
            "market_hours_required": self.market_hours_required,
        }

        return HealthCheckResult(
            service_name=self.name,
            status=final_status,
            response_time=response_time,
            timestamp=start_time,
            details=combined_details,
            error=underlying_result.error,
        )


class TradingDatabaseHealthCheck(HealthCheck):
    """Trading-specific database health check."""

    def __init__(
        self,
        name: str,
        connection_factory: Any,
        timeout: float = 10.0,
        check_trading_tables: bool = True,
    ):
        super().__init__(name, timeout)
        self.connection_factory = connection_factory
        self.check_trading_tables = check_trading_tables

    @trace_trading_operation(operation_name="database_health_check")
    async def check_health(self) -> HealthCheckResult:
        """Check trading database health."""
        start_time = time.time()
        details = {}

        try:
            # Get connection
            connection = await self.connection_factory.get_connection()

            async with connection.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Basic connectivity test
                    await cursor.execute("SELECT 1 as health_check")
                    result = await cursor.fetchone()

                    if not result or result[0] != 1:
                        raise Exception("Database connectivity test failed")

                    details["connectivity"] = "OK"

                    # Check trading-specific tables
                    if self.check_trading_tables:
                        trading_tables = [
                            "portfolios",
                            "positions",
                            "orders",
                            "market_data",
                            "risk_metrics",
                        ]

                        table_status = {}
                        for table in trading_tables:
                            try:
                                await cursor.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                                table_status[table] = "OK"
                            except Exception as e:
                                table_status[table] = f"ERROR: {e!s}"

                        details["trading_tables"] = table_status  # type: ignore[assignment]

                    # Get connection pool stats
                    try:
                        pool_stats = await connection.get_pool_stats()
                        details["pool_stats"] = pool_stats
                    except Exception:
                        pass

                    # Check database performance
                    perf_start = time.perf_counter()
                    await cursor.execute("SELECT COUNT(*) FROM portfolios")
                    perf_duration = time.perf_counter() - perf_start

                    details["query_performance_ms"] = str(perf_duration * 1000)

            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details=details,
            )

        except Exception as e:
            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details=details,
                error=str(e),
            )


class BrokerHealthCheck(HealthCheck):
    """Health check for trading brokers."""

    def __init__(
        self,
        name: str,
        broker_client: Any,
        timeout: float = 15.0,
        check_account_status: bool = True,
    ):
        super().__init__(name, timeout)
        self.broker_client = broker_client
        self.check_account_status = check_account_status

    @trace_trading_operation(operation_name="broker_health_check")
    async def check_health(self) -> HealthCheckResult:
        """Check broker connectivity and status."""
        start_time = time.time()
        details = {}

        try:
            # Test basic connectivity
            if hasattr(self.broker_client, "get_connection_status"):
                conn_status = await self.broker_client.get_connection_status()
                details["connection_status"] = conn_status

            # Check account status if requested
            if self.check_account_status:
                try:
                    if hasattr(self.broker_client, "get_account"):
                        account_info = await self.broker_client.get_account()
                        details["account_status"] = getattr(account_info, "status", "unknown")
                        details["buying_power"] = getattr(account_info, "buying_power", 0)
                        details["account_blocked"] = getattr(account_info, "account_blocked", False)
                except Exception as e:
                    details["account_error"] = str(e)

            # Test market data access
            try:
                if hasattr(self.broker_client, "get_latest_quote"):
                    # Test with a common symbol
                    quote = await self.broker_client.get_latest_quote("SPY")
                    if quote:
                        details["market_data_access"] = "OK"
                    else:
                        details["market_data_access"] = "NO_DATA"
            except Exception as e:
                details["market_data_error"] = str(e)

            # Check API rate limits
            try:
                if hasattr(self.broker_client, "get_rate_limit_status"):
                    rate_limit_status = await self.broker_client.get_rate_limit_status()
                    details["rate_limits"] = rate_limit_status
            except Exception as e:
                details["rate_limit_error"] = str(e)

            response_time = time.time() - start_time

            # Determine health status
            status = HealthStatus.HEALTHY

            # Check for critical issues
            if (
                details.get("account_blocked", False)
                or details.get("connection_status") == "disconnected"
            ):
                status = HealthStatus.UNHEALTHY
            elif "account_error" in details or "market_data_error" in details:
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                service_name=self.name,
                status=status,
                response_time=response_time,
                timestamp=start_time,
                details=details,
            )

        except Exception as e:
            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details=details,
                error=str(e),
            )


class MarketDataHealthCheck(HealthCheck):
    """Health check for market data providers."""

    def __init__(
        self,
        name: str,
        market_data_client: Any,
        timeout: float = 15.0,
        test_symbols: list[str] | None = None,
    ):
        super().__init__(name, timeout)
        self.market_data_client = market_data_client
        self.test_symbols = test_symbols or ["SPY", "QQQ", "IWM"]

    @trace_trading_operation(operation_name="market_data_health_check")
    async def check_health(self) -> HealthCheckResult:
        """Check market data provider health."""
        start_time = time.time()
        details = {}

        try:
            # Test connectivity
            if hasattr(self.market_data_client, "get_connection_status"):
                conn_status = await self.market_data_client.get_connection_status()
                details["connection_status"] = conn_status

            # Test data retrieval for each symbol
            symbol_results = {}
            for symbol in self.test_symbols:
                try:
                    quote_start = time.perf_counter()
                    quote = await self.market_data_client.get_latest_quote(symbol)
                    quote_duration = time.perf_counter() - quote_start

                    if quote:
                        symbol_results[symbol] = {
                            "status": "OK",
                            "latency_ms": quote_duration * 1000,
                            "bid": getattr(quote, "bid", None),
                            "ask": getattr(quote, "ask", None),
                            "timestamp": getattr(quote, "timestamp", None),
                        }
                    else:
                        symbol_results[symbol] = {"status": "NO_DATA"}

                except Exception as e:
                    symbol_results[symbol] = {"status": "ERROR", "error": str(e)}

            details["symbol_tests"] = symbol_results

            # Check API rate limits
            try:
                if hasattr(self.market_data_client, "get_rate_limit_status"):
                    rate_limits = await self.market_data_client.get_rate_limit_status()
                    details["rate_limits"] = rate_limits
            except Exception:
                pass

            # Check subscription status
            try:
                if hasattr(self.market_data_client, "get_subscription_status"):
                    subscription = await self.market_data_client.get_subscription_status()
                    details["subscription"] = subscription
            except Exception:
                pass

            response_time = time.time() - start_time

            # Determine health status
            successful_symbols = sum(
                1 for result in symbol_results.values() if result.get("status") == "OK"
            )

            if successful_symbols == 0:
                status = HealthStatus.UNHEALTHY
            elif successful_symbols < len(self.test_symbols) / 2:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            # Check latency
            latencies: list[float] = []
            for result in symbol_results.values():
                if result.get("status") == "OK":
                    latency = result.get("latency_ms", 0)
                    if latency is not None and isinstance(latency, (int, float, str)):
                        try:
                            latencies.append(float(latency))
                        except (TypeError, ValueError):
                            latencies.append(0.0)
            avg_latency = sum(latencies) / max(successful_symbols, 1)

            details["avg_latency_ms"] = avg_latency

            # Degrade if latency is too high
            if avg_latency > 1000 and status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                service_name=self.name,
                status=status,
                response_time=response_time,
                timestamp=start_time,
                details=details,
            )

        except Exception as e:
            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details=details,
                error=str(e),
            )


class TradingSystemHealthCheck(HealthCheck):
    """Comprehensive trading system health check."""

    def __init__(
        self,
        name: str = "trading_system",
        timeout: float = 30.0,
        risk_service: Any = None,
        position_service: Any = None,
        portfolio_service: Any = None,
    ):
        super().__init__(name, timeout)
        self.risk_service = risk_service
        self.position_service = position_service
        self.portfolio_service = portfolio_service

    @trace_trading_operation(operation_name="trading_system_health_check")
    async def check_health(self) -> HealthCheckResult:
        """Check overall trading system health."""
        start_time = time.time()
        details = {}

        try:
            # Check risk service
            if self.risk_service:
                try:
                    # Test risk calculation
                    if hasattr(self.risk_service, "health_check"):
                        risk_health = await self.risk_service.health_check()
                        details["risk_service"] = risk_health
                    else:
                        details["risk_service"] = "NO_HEALTH_CHECK"
                except Exception as e:
                    details["risk_service"] = f"ERROR: {e!s}"

            # Check position service
            if self.position_service:
                try:
                    if hasattr(self.position_service, "health_check"):
                        position_health = await self.position_service.health_check()
                        details["position_service"] = position_health
                    else:
                        details["position_service"] = "NO_HEALTH_CHECK"
                except Exception as e:
                    details["position_service"] = f"ERROR: {e!s}"

            # Check portfolio service
            if self.portfolio_service:
                try:
                    if hasattr(self.portfolio_service, "health_check"):
                        portfolio_health = await self.portfolio_service.health_check()
                        details["portfolio_service"] = portfolio_health
                    else:
                        details["portfolio_service"] = "NO_HEALTH_CHECK"
                except Exception as e:
                    details["portfolio_service"] = f"ERROR: {e!s}"

            # Add system resource checks
            import psutil

            details["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
            }

            response_time = time.time() - start_time

            # Determine health status based on services
            error_count = sum(
                1
                for value in details.values()
                if isinstance(value, str) and value.startswith("ERROR:")
            )

            if error_count == 0:
                status = HealthStatus.HEALTHY
            elif error_count <= 1:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            return HealthCheckResult(
                service_name=self.name,
                status=status,
                response_time=response_time,
                timestamp=start_time,
                details=details,
            )

        except Exception as e:
            response_time = time.time() - start_time

            return HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=start_time,
                details=details,
                error=str(e),
            )


class TradingHealthChecker(HealthChecker):
    """Extended health checker with trading-specific features."""

    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 100,
        market_hours_checker: MarketHoursChecker | None = None,
        **kwargs: Any,
    ):
        super().__init__(check_interval, history_size, **kwargs)
        self.market_hours_checker = market_hours_checker or MarketHoursChecker()

    def register_trading_health_check(
        self, health_check: HealthCheck, market_hours_required: bool = False
    ) -> None:
        """
        Register a trading health check with optional market hours dependency.

        Args:
            health_check: Health check instance
            market_hours_required: Whether this check requires market hours
        """
        if market_hours_required:
            # Wrap with market hours awareness
            wrapped_check = MarketHoursHealthCheck(
                name=health_check.name,
                underlying_check=health_check,
                market_hours_checker=self.market_hours_checker,
                market_hours_required=True,
                timeout=health_check.timeout,
            )
            self.register_health_check(wrapped_check)
        else:
            self.register_health_check(health_check)

    def get_trading_health_summary(self) -> dict[str, Any]:
        """Get trading-specific health summary."""
        base_summary = self.get_health_summary()

        # Add market information
        market_info = self.market_hours_checker.get_market_info()

        # Add trading-specific status
        trading_summary = {
            **base_summary,
            "market_info": market_info,
            "trading_status": self._get_trading_status(base_summary["overall_status"], market_info),
        }

        return trading_summary

    def _get_trading_status(self, overall_status: str, market_info: dict[str, Any]) -> str:
        """Determine trading-specific status."""
        market_status = market_info.get("status", "unknown")

        if overall_status == "unhealthy":
            return "not_ready_for_trading"
        elif market_status in ("closed", "holiday"):
            return "market_closed"
        elif market_status == "open" and overall_status == "healthy":
            return "ready_for_trading"
        elif market_status in ("pre_market", "after_market"):
            return "limited_trading_available"
        else:
            return "degraded_trading"

    async def wait_for_market_open(self, timeout: float | None = None) -> bool:
        """
        Wait for market to open.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if market opened, False if timeout
        """
        start_time = time.time()

        while True:
            market_status = self.market_hours_checker.get_current_market_status()

            if market_status == MarketStatus.OPEN:
                return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            # Wait 30 seconds before checking again
            await asyncio.sleep(30)

    async def wait_for_healthy_system(self, timeout: float | None = None) -> bool:
        """
        Wait for system to be healthy.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if system is healthy, False if timeout
        """
        start_time = time.time()

        while True:
            overall_status = self.get_overall_status()

            if overall_status == HealthStatus.HEALTHY:
                return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            # Wait for check interval before checking again
            await asyncio.sleep(self.check_interval)


# Convenience functions
def create_trading_health_checker(
    database_connection_factory: Any = None,
    broker_client: Any = None,
    market_data_client: Any = None,
    risk_service: Any = None,
    position_service: Any = None,
    portfolio_service: Any = None,
    **kwargs: Any,
) -> TradingHealthChecker:
    """
    Create a fully configured trading health checker.

    Args:
        database_connection_factory: Database connection factory
        broker_client: Broker client instance
        market_data_client: Market data client instance
        risk_service: Risk management service
        position_service: Position management service
        portfolio_service: Portfolio management service
        **kwargs: Additional TradingHealthChecker arguments

    Returns:
        Configured TradingHealthChecker
    """
    health_checker = TradingHealthChecker(**kwargs)

    # Register database health check
    if database_connection_factory:
        db_check = TradingDatabaseHealthCheck("trading_database", database_connection_factory)
        health_checker.register_trading_health_check(db_check, market_hours_required=False)

    # Register broker health check
    if broker_client:
        broker_check = BrokerHealthCheck("trading_broker", broker_client)
        health_checker.register_trading_health_check(broker_check, market_hours_required=True)

    # Register market data health check
    if market_data_client:
        market_data_check = MarketDataHealthCheck("market_data_provider", market_data_client)
        health_checker.register_trading_health_check(market_data_check, market_hours_required=True)

    # Register trading system health check
    system_check = TradingSystemHealthCheck(
        "trading_system_services",
        risk_service=risk_service,
        position_service=position_service,
        portfolio_service=portfolio_service,
    )
    health_checker.register_trading_health_check(system_check, market_hours_required=False)

    return health_checker

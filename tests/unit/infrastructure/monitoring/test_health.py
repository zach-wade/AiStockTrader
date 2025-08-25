"""
Comprehensive unit tests for health monitoring module.

Tests health check mechanisms, market hours awareness, dependency health,
and trading system specific checks.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytz

from src.infrastructure.monitoring.health import (
    BrokerHealthCheck,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    MarketDataHealthCheck,
    MarketHours,
    MarketHoursChecker,
    MarketHoursHealthCheck,
    MarketStatus,
    TradingDatabaseHealthCheck,
    TradingHealthChecker,
    TradingSystemHealthCheck,
    create_trading_health_checker,
)


class TestMarketStatus:
    """Test MarketStatus enumeration."""

    def test_market_status_values(self):
        """Test MarketStatus enum values."""
        assert MarketStatus.OPEN == "open"
        assert MarketStatus.CLOSED == "closed"
        assert MarketStatus.PRE_MARKET == "pre_market"
        assert MarketStatus.AFTER_MARKET == "after_market"
        assert MarketStatus.HOLIDAY == "holiday"
        assert MarketStatus.UNKNOWN == "unknown"

    def test_from_domain_conversion(self):
        """Test conversion from domain MarketStatus."""
        from src.domain.services.market_hours_service import MarketStatus as DomainStatus

        assert MarketStatus.from_domain(DomainStatus.OPEN) == MarketStatus.OPEN
        assert MarketStatus.from_domain(DomainStatus.CLOSED) == MarketStatus.CLOSED
        assert MarketStatus.from_domain(DomainStatus.PRE_MARKET) == MarketStatus.PRE_MARKET
        assert MarketStatus.from_domain(DomainStatus.AFTER_MARKET) == MarketStatus.AFTER_MARKET
        assert MarketStatus.from_domain(DomainStatus.HOLIDAY) == MarketStatus.HOLIDAY


class TestMarketHours:
    """Test MarketHours configuration."""

    def test_default_market_hours(self):
        """Test default market hours configuration."""
        hours = MarketHours()

        assert hours.timezone == "America/New_York"
        assert hours.regular_open_hour == 9
        assert hours.regular_open_minute == 30
        assert hours.regular_close_hour == 16
        assert hours.regular_close_minute == 0
        assert hours.pre_market_open_hour == 4
        assert hours.after_market_close_hour == 20

    def test_custom_market_hours(self):
        """Test custom market hours configuration."""
        hours = MarketHours(
            timezone="Europe/London",
            regular_open_hour=8,
            regular_open_minute=0,
            regular_close_hour=16,
            regular_close_minute=30,
        )

        assert hours.timezone == "Europe/London"
        assert hours.regular_open_hour == 8
        assert hours.regular_open_minute == 0
        assert hours.regular_close_hour == 16
        assert hours.regular_close_minute == 30

    def test_holidays_set(self):
        """Test holidays are properly initialized."""
        hours = MarketHours()
        assert len(hours.holidays) > 0
        assert "2024-01-01" in hours.holidays  # New Year's Day
        assert "2024-12-25" in hours.holidays  # Christmas


class TestMarketHoursChecker:
    """Test MarketHoursChecker functionality."""

    def test_initialization(self):
        """Test MarketHoursChecker initialization."""
        checker = MarketHoursChecker()
        assert checker.market_hours is not None
        assert checker.timezone is not None
        assert checker._market_hours_service is not None

    def test_custom_initialization(self):
        """Test MarketHoursChecker with custom hours."""
        custom_hours = MarketHours(timezone="Europe/London", holidays={"2024-01-01", "2024-12-25"})
        checker = MarketHoursChecker(custom_hours)
        assert checker.market_hours == custom_hours
        assert checker.timezone.zone == "Europe/London"

    @patch("src.infrastructure.monitoring.health.datetime")
    def test_get_current_market_status(self, mock_datetime):
        """Test getting current market status."""
        checker = MarketHoursChecker()
        ny_tz = pytz.timezone("America/New_York")

        # Mock a market open time (Tuesday 10 AM ET)
        mock_now = datetime(2024, 1, 2, 10, 0, 0, tzinfo=ny_tz)
        mock_datetime.now.return_value = mock_now

        with patch.object(
            checker._market_hours_service, "get_current_market_status"
        ) as mock_status:
            from src.domain.services.market_hours_service import MarketStatus as DomainStatus

            mock_status.return_value = DomainStatus.OPEN

            status = checker.get_current_market_status()
            assert status == MarketStatus.OPEN
            mock_status.assert_called_once_with(mock_now)

    def test_is_holiday(self):
        """Test holiday checking."""
        checker = MarketHoursChecker()

        with patch.object(checker._market_hours_service, "is_holiday") as mock_holiday:
            mock_holiday.return_value = True

            test_date = datetime(2024, 1, 1)
            result = checker._is_holiday(test_date)

            assert result is True
            mock_holiday.assert_called_once_with(test_date)

    @patch("src.infrastructure.monitoring.health.datetime")
    def test_get_market_info(self, mock_datetime):
        """Test getting comprehensive market information."""
        checker = MarketHoursChecker()
        ny_tz = pytz.timezone("America/New_York")

        mock_now = datetime(2024, 1, 2, 10, 0, 0, tzinfo=ny_tz)
        mock_datetime.now.return_value = mock_now

        with patch.object(checker, "get_current_market_status") as mock_status:
            mock_status.return_value = MarketStatus.OPEN

            with patch.object(checker, "_get_next_market_open") as mock_open:
                mock_open.return_value = datetime(2024, 1, 3, 9, 30, 0, tzinfo=ny_tz)

                with patch.object(checker, "_get_next_market_close") as mock_close:
                    mock_close.return_value = datetime(2024, 1, 2, 16, 0, 0, tzinfo=ny_tz)

                    info = checker.get_market_info()

                    assert info["status"] == "open"
                    assert info["timezone"] == "America/New_York"
                    assert info["is_trading_day"] is True
                    assert info["next_open"] is not None
                    assert info["next_close"] is not None

    def test_get_next_market_open(self):
        """Test getting next market open time."""
        checker = MarketHoursChecker()
        ny_tz = pytz.timezone("America/New_York")

        # Test from a Friday evening
        friday_evening = datetime(2024, 1, 5, 18, 0, 0, tzinfo=ny_tz)

        with patch.object(checker, "_is_holiday") as mock_holiday:
            mock_holiday.return_value = False

            next_open = checker._get_next_market_open(friday_evening)

            # Should be Monday morning
            assert next_open is not None
            assert next_open.weekday() == 0  # Monday
            assert next_open.hour == 9
            assert next_open.minute == 30

    def test_get_next_market_close(self):
        """Test getting next market close time."""
        checker = MarketHoursChecker()

        with patch.object(checker, "get_current_market_status") as mock_status:
            mock_status.return_value = MarketStatus.OPEN

            ny_tz = pytz.timezone("America/New_York")
            current_time = datetime(2024, 1, 2, 10, 0, 0, tzinfo=ny_tz)

            next_close = checker._get_next_market_close(current_time)

            assert next_close is not None
            assert next_close.hour == 16
            assert next_close.minute == 0


class TestMarketHoursHealthCheck:
    """Test MarketHoursHealthCheck functionality."""

    @pytest.mark.asyncio
    async def test_health_check_market_open(self):
        """Test health check when market is open."""
        underlying_check = Mock(spec=HealthCheck)
        underlying_check.name = "test_check"
        underlying_check.timeout = 10.0
        underlying_check.safe_check_health = AsyncMock(
            return_value=HealthCheckResult(
                service_name="test_check",
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                timestamp=1234567890,
                details={"test": "data"},
            )
        )

        market_checker = Mock(spec=MarketHoursChecker)
        market_checker.get_current_market_status.return_value = MarketStatus.OPEN
        market_checker.get_market_info.return_value = {"status": "open", "is_trading_day": True}

        health_check = MarketHoursHealthCheck(
            name="market_aware_check",
            underlying_check=underlying_check,
            market_hours_checker=market_checker,
            market_hours_required=True,
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["market_status"] == "open"
        assert result.details["market_hours_required"] is True

    @pytest.mark.asyncio
    async def test_health_check_market_closed_degraded(self):
        """Test health check degrades when market is closed but required."""
        underlying_check = Mock(spec=HealthCheck)
        underlying_check.safe_check_health = AsyncMock(
            return_value=HealthCheckResult(
                service_name="test_check",
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                timestamp=1234567890,
                details={},
            )
        )

        market_checker = Mock(spec=MarketHoursChecker)
        market_checker.get_current_market_status.return_value = MarketStatus.CLOSED
        market_checker.get_market_info.return_value = {"status": "closed", "is_trading_day": False}

        health_check = MarketHoursHealthCheck(
            name="market_aware_check",
            underlying_check=underlying_check,
            market_hours_checker=market_checker,
            market_hours_required=True,
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["market_status"] == "closed"

    @pytest.mark.asyncio
    async def test_health_check_market_not_required(self):
        """Test health check when market hours not required."""
        underlying_check = Mock(spec=HealthCheck)
        underlying_check.safe_check_health = AsyncMock(
            return_value=HealthCheckResult(
                service_name="test_check",
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                timestamp=1234567890,
                details={},
            )
        )

        market_checker = Mock(spec=MarketHoursChecker)
        market_checker.get_current_market_status.return_value = MarketStatus.CLOSED
        market_checker.get_market_info.return_value = {"status": "closed", "is_trading_day": False}

        health_check = MarketHoursHealthCheck(
            name="market_aware_check",
            underlying_check=underlying_check,
            market_hours_checker=market_checker,
            market_hours_required=False,
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["market_hours_required"] is False


class TestTradingDatabaseHealthCheck:
    """Test TradingDatabaseHealthCheck functionality."""

    @pytest.mark.asyncio
    async def test_database_healthy(self):
        """Test healthy database connection."""
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))

        mock_conn = AsyncMock()
        mock_conn.cursor = AsyncMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock()

        mock_connection = AsyncMock()
        mock_connection.acquire = AsyncMock(return_value=mock_conn)
        mock_connection.get_pool_stats = AsyncMock(
            return_value={"size": 10, "in_use": 2, "available": 8}
        )

        mock_factory = Mock()
        mock_factory.get_connection = AsyncMock(return_value=mock_connection)

        health_check = TradingDatabaseHealthCheck(
            name="database", connection_factory=mock_factory, check_trading_tables=True
        )

        with patch("src.infrastructure.monitoring.health.time.time", return_value=1234567890):
            result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["connectivity"] == "OK"
        assert "pool_stats" in result.details

    @pytest.mark.asyncio
    async def test_database_unhealthy(self):
        """Test unhealthy database connection."""
        mock_factory = Mock()
        mock_factory.get_connection = AsyncMock(side_effect=Exception("Connection failed"))

        health_check = TradingDatabaseHealthCheck(name="database", connection_factory=mock_factory)

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.error == "Connection failed"

    @pytest.mark.asyncio
    async def test_database_table_checks(self):
        """Test checking specific trading tables."""
        mock_cursor = AsyncMock()

        # Mock successful connectivity check
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))

        # Mock table checks - some succeed, some fail
        async def execute_side_effect(query):
            if "portfolios" in query or "positions" in query:
                return None  # Success
            elif "orders" in query:
                raise Exception("Table not found")
            return None

        mock_cursor.execute = AsyncMock(side_effect=execute_side_effect)

        mock_conn = AsyncMock()
        mock_conn.cursor = AsyncMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock()

        mock_connection = AsyncMock()
        mock_connection.acquire = AsyncMock(return_value=mock_conn)
        mock_connection.get_pool_stats = AsyncMock(return_value={})

        mock_factory = Mock()
        mock_factory.get_connection = AsyncMock(return_value=mock_connection)

        health_check = TradingDatabaseHealthCheck(
            name="database", connection_factory=mock_factory, check_trading_tables=True
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert "trading_tables" in result.details


class TestBrokerHealthCheck:
    """Test BrokerHealthCheck functionality."""

    @pytest.mark.asyncio
    async def test_broker_healthy(self):
        """Test healthy broker connection."""
        mock_account = Mock()
        mock_account.status = "active"
        mock_account.buying_power = 100000.0
        mock_account.account_blocked = False

        mock_broker = Mock()
        mock_broker.get_connection_status = AsyncMock(return_value="connected")
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_latest_quote = AsyncMock(return_value={"bid": 100, "ask": 101})
        mock_broker.get_rate_limit_status = AsyncMock(
            return_value={"remaining": 100, "reset_time": 1234567890}
        )

        health_check = BrokerHealthCheck(
            name="broker", broker_client=mock_broker, check_account_status=True
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["connection_status"] == "connected"
        assert result.details["account_status"] == "active"
        assert result.details["buying_power"] == 100000.0
        assert result.details["market_data_access"] == "OK"

    @pytest.mark.asyncio
    async def test_broker_account_blocked(self):
        """Test broker with blocked account."""
        mock_account = Mock()
        mock_account.status = "blocked"
        mock_account.account_blocked = True

        mock_broker = Mock()
        mock_broker.get_connection_status = AsyncMock(return_value="connected")
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        health_check = BrokerHealthCheck(
            name="broker", broker_client=mock_broker, check_account_status=True
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.details["account_blocked"] is True

    @pytest.mark.asyncio
    async def test_broker_disconnected(self):
        """Test disconnected broker."""
        mock_broker = Mock()
        mock_broker.get_connection_status = AsyncMock(return_value="disconnected")

        health_check = BrokerHealthCheck(
            name="broker", broker_client=mock_broker, check_account_status=False
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.details["connection_status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_broker_degraded(self):
        """Test degraded broker status."""
        mock_broker = Mock()
        mock_broker.get_connection_status = AsyncMock(return_value="connected")
        mock_broker.get_account = AsyncMock(side_effect=Exception("Account API error"))
        mock_broker.get_latest_quote = AsyncMock(side_effect=Exception("Market data error"))

        health_check = BrokerHealthCheck(
            name="broker", broker_client=mock_broker, check_account_status=True
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert "account_error" in result.details
        assert "market_data_error" in result.details


class TestMarketDataHealthCheck:
    """Test MarketDataHealthCheck functionality."""

    @pytest.mark.asyncio
    async def test_market_data_healthy(self):
        """Test healthy market data provider."""
        mock_quote = Mock()
        mock_quote.bid = 100.0
        mock_quote.ask = 100.5
        mock_quote.timestamp = 1234567890

        mock_client = Mock()
        mock_client.get_connection_status = AsyncMock(return_value="connected")
        mock_client.get_latest_quote = AsyncMock(return_value=mock_quote)
        mock_client.get_rate_limit_status = AsyncMock(
            return_value={"remaining": 1000, "limit": 2000}
        )
        mock_client.get_subscription_status = AsyncMock(return_value="active")

        health_check = MarketDataHealthCheck(
            name="market_data", market_data_client=mock_client, test_symbols=["SPY", "QQQ"]
        )

        with patch(
            "src.infrastructure.monitoring.health.time.perf_counter",
            side_effect=[0, 0.001, 0, 0.002],
        ):
            result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["connection_status"] == "connected"
        assert "symbol_tests" in result.details
        assert result.details["symbol_tests"]["SPY"]["status"] == "OK"
        assert result.details["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_market_data_no_quotes(self):
        """Test market data provider with no quotes."""
        mock_client = Mock()
        mock_client.get_connection_status = AsyncMock(return_value="connected")
        mock_client.get_latest_quote = AsyncMock(return_value=None)

        health_check = MarketDataHealthCheck(
            name="market_data", market_data_client=mock_client, test_symbols=["SPY"]
        )

        result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.details["symbol_tests"]["SPY"]["status"] == "NO_DATA"

    @pytest.mark.asyncio
    async def test_market_data_high_latency(self):
        """Test market data provider with high latency."""
        mock_quote = Mock()
        mock_quote.bid = 100.0
        mock_quote.ask = 100.5

        mock_client = Mock()
        mock_client.get_connection_status = AsyncMock(return_value="connected")
        mock_client.get_latest_quote = AsyncMock(return_value=mock_quote)

        health_check = MarketDataHealthCheck(
            name="market_data", market_data_client=mock_client, test_symbols=["SPY"]
        )

        # Mock high latency (2 seconds)
        with patch("src.infrastructure.monitoring.health.time.perf_counter", side_effect=[0, 2.0]):
            result = await health_check.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["avg_latency_ms"] > 1000

    @pytest.mark.asyncio
    async def test_market_data_partial_failure(self):
        """Test market data provider with partial symbol failures."""
        mock_quote = Mock()
        mock_quote.bid = 100.0

        async def get_quote_side_effect(symbol):
            if symbol == "SPY":
                return mock_quote
            else:
                raise Exception(f"Failed to get {symbol}")

        mock_client = Mock()
        mock_client.get_connection_status = AsyncMock(return_value="connected")
        mock_client.get_latest_quote = AsyncMock(side_effect=get_quote_side_effect)

        health_check = MarketDataHealthCheck(
            name="market_data", market_data_client=mock_client, test_symbols=["SPY", "QQQ", "IWM"]
        )

        with patch(
            "src.infrastructure.monitoring.health.time.perf_counter", side_effect=[0, 0.001]
        ):
            result = await health_check.check_health()

        # Should be degraded because less than half succeeded
        assert result.status == HealthStatus.DEGRADED
        assert result.details["symbol_tests"]["SPY"]["status"] == "OK"
        assert result.details["symbol_tests"]["QQQ"]["status"] == "ERROR"


class TestTradingSystemHealthCheck:
    """Test TradingSystemHealthCheck functionality."""

    @pytest.mark.asyncio
    async def test_system_all_healthy(self):
        """Test system with all services healthy."""
        mock_risk = Mock()
        mock_risk.health_check = AsyncMock(return_value={"status": "healthy"})

        mock_position = Mock()
        mock_position.health_check = AsyncMock(return_value={"status": "healthy"})

        mock_portfolio = Mock()
        mock_portfolio.health_check = AsyncMock(return_value={"status": "healthy"})

        health_check = TradingSystemHealthCheck(
            risk_service=mock_risk, position_service=mock_position, portfolio_service=mock_portfolio
        )

        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.getloadavg", return_value=(1.0, 1.5, 2.0)),
        ):
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["risk_service"] == {"status": "healthy"}
        assert result.details["system_resources"]["cpu_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_system_one_service_error(self):
        """Test system with one service error."""
        mock_risk = Mock()
        mock_risk.health_check = AsyncMock(side_effect=Exception("Risk service error"))

        mock_position = Mock()
        mock_position.health_check = AsyncMock(return_value={"status": "healthy"})

        health_check = TradingSystemHealthCheck(
            risk_service=mock_risk, position_service=mock_position
        )

        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            result = await health_check.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert "ERROR:" in result.details["risk_service"]

    @pytest.mark.asyncio
    async def test_system_multiple_errors(self):
        """Test system with multiple service errors."""
        mock_risk = Mock()
        mock_risk.health_check = AsyncMock(side_effect=Exception("Risk error"))

        mock_position = Mock()
        mock_position.health_check = AsyncMock(side_effect=Exception("Position error"))

        mock_portfolio = Mock()
        mock_portfolio.health_check = AsyncMock(side_effect=Exception("Portfolio error"))

        health_check = TradingSystemHealthCheck(
            risk_service=mock_risk, position_service=mock_position, portfolio_service=mock_portfolio
        )

        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            result = await health_check.check_health()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_system_no_health_check_methods(self):
        """Test system with services lacking health_check methods."""
        mock_risk = Mock(spec=[])  # No health_check method
        mock_position = Mock(spec=[])

        health_check = TradingSystemHealthCheck(
            risk_service=mock_risk, position_service=mock_position
        )

        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            result = await health_check.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["risk_service"] == "NO_HEALTH_CHECK"
        assert result.details["position_service"] == "NO_HEALTH_CHECK"


class TestTradingHealthChecker:
    """Test TradingHealthChecker functionality."""

    def test_initialization(self):
        """Test TradingHealthChecker initialization."""
        checker = TradingHealthChecker(check_interval=60.0, history_size=50)

        assert checker.check_interval == 60.0
        assert checker.history_size == 50
        assert checker.market_hours_checker is not None

    def test_register_trading_health_check(self):
        """Test registering trading health checks."""
        checker = TradingHealthChecker()

        mock_check = Mock(spec=HealthCheck)
        mock_check.name = "test_check"
        mock_check.timeout = 10.0

        # Register without market hours requirement
        checker.register_trading_health_check(mock_check, market_hours_required=False)
        assert len(checker.health_checks) == 1

        # Register with market hours requirement
        mock_check2 = Mock(spec=HealthCheck)
        mock_check2.name = "market_check"
        mock_check2.timeout = 10.0

        checker.register_trading_health_check(mock_check2, market_hours_required=True)
        assert len(checker.health_checks) == 2

        # The second check should be wrapped
        wrapped = checker.health_checks[1]
        assert isinstance(wrapped, MarketHoursHealthCheck)

    def test_get_trading_health_summary(self):
        """Test getting trading health summary."""
        checker = TradingHealthChecker()

        with patch.object(checker, "get_health_summary") as mock_summary:
            mock_summary.return_value = {"overall_status": "healthy", "services": {}}

            with patch.object(checker.market_hours_checker, "get_market_info") as mock_info:
                mock_info.return_value = {"status": "open", "is_trading_day": True}

                summary = checker.get_trading_health_summary()

                assert "market_info" in summary
                assert summary["trading_status"] == "ready_for_trading"

    def test_get_trading_status(self):
        """Test determining trading status."""
        checker = TradingHealthChecker()

        # Test unhealthy system
        status = checker._get_trading_status("unhealthy", {"status": "open"})
        assert status == "not_ready_for_trading"

        # Test market closed
        status = checker._get_trading_status("healthy", {"status": "closed"})
        assert status == "market_closed"

        # Test healthy and open
        status = checker._get_trading_status("healthy", {"status": "open"})
        assert status == "ready_for_trading"

        # Test pre-market
        status = checker._get_trading_status("healthy", {"status": "pre_market"})
        assert status == "limited_trading_available"

        # Test degraded
        status = checker._get_trading_status("degraded", {"status": "open"})
        assert status == "degraded_trading"

    @pytest.mark.asyncio
    async def test_wait_for_market_open(self):
        """Test waiting for market to open."""
        checker = TradingHealthChecker()

        call_count = 0

        def get_status_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return MarketStatus.CLOSED
            return MarketStatus.OPEN

        with patch.object(
            checker.market_hours_checker,
            "get_current_market_status",
            side_effect=get_status_side_effect,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await checker.wait_for_market_open(command_timeout=5.0)
                assert result is True

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_wait_for_market_open_timeout(self):
        """Test waiting for market open with timeout."""
        checker = TradingHealthChecker()

        with patch.object(
            checker.market_hours_checker,
            "get_current_market_status",
            return_value=MarketStatus.CLOSED,
        ):
            with patch("time.time", side_effect=[0, 1, 2, 3, 4, 5, 6]):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await checker.wait_for_market_open(command_timeout=5.0)
                    assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_healthy_system(self):
        """Test waiting for healthy system."""
        checker = TradingHealthChecker()

        call_count = 0

        def get_status_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return HealthStatus.DEGRADED
            return HealthStatus.HEALTHY

        with patch.object(checker, "get_overall_status", side_effect=get_status_side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await checker.wait_for_healthy_system(command_timeout=10.0)
                assert result is True


class TestCreateTradingHealthChecker:
    """Test create_trading_health_checker function."""

    def test_create_with_all_components(self):
        """Test creating health checker with all components."""
        mock_db_factory = Mock()
        mock_broker = Mock()
        mock_market_data = Mock()
        mock_risk = Mock()
        mock_position = Mock()
        mock_portfolio = Mock()

        checker = create_trading_health_checker(
            database_connection_factory=mock_db_factory,
            broker_client=mock_broker,
            market_data_client=mock_market_data,
            risk_service=mock_risk,
            position_service=mock_position,
            portfolio_service=mock_portfolio,
        )

        assert isinstance(checker, TradingHealthChecker)
        # Should have 4 health checks registered
        assert len(checker.health_checks) == 4

    def test_create_with_partial_components(self):
        """Test creating health checker with partial components."""
        mock_db_factory = Mock()
        mock_broker = Mock()

        checker = create_trading_health_checker(
            database_connection_factory=mock_db_factory, broker_client=mock_broker
        )

        assert isinstance(checker, TradingHealthChecker)
        # Should have 3 health checks (db, broker, system)
        assert len(checker.health_checks) == 3

    def test_create_with_no_components(self):
        """Test creating health checker with no components."""
        checker = create_trading_health_checker()

        assert isinstance(checker, TradingHealthChecker)
        # Should only have system health check
        assert len(checker.health_checks) == 1

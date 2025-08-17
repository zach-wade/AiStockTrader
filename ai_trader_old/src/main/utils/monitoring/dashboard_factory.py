"""
Dashboard factory for creating dashboard instances.

This module provides factory methods for creating different types of dashboards
and dashboard managers, following the factory pattern used throughout the codebase.
"""

# Standard library imports
from typing import Any

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.metrics import IMetricsRecorder
from main.interfaces.monitoring import DashboardConfig, IDashboard, IDashboardManager
from main.utils.core import get_logger

logger = get_logger(__name__)


class DashboardFactory:
    """
    Factory for creating dashboard instances.

    This factory provides methods to create different types of dashboards
    and dashboard managers, ensuring proper dependency injection and
    configuration.
    """

    @staticmethod
    def create_trading_dashboard(
        db_pool: IAsyncDatabase,
        metrics_recorder: IMetricsRecorder,
        config: dict[str, Any] | DashboardConfig | None = None,
    ) -> IDashboard:
        """
        Create a trading dashboard instance.

        Args:
            db_pool: Database connection pool
            metrics_recorder: Metrics recording interface
            config: Dashboard configuration

        Returns:
            Trading dashboard instance implementing IDashboard

        Raises:
            ImportError: If trading dashboard module not available
            Exception: If creation fails
        """
        try:
            # Local imports
            from main.monitoring.dashboards.trading.adapter import TradingDashboardAdapter

            # Convert dict config to DashboardConfig if needed
            if isinstance(config, dict):
                dashboard_config = DashboardConfig(
                    name="trading",
                    host=config.get("host", "0.0.0.0"),
                    port=config.get("port", 8080),
                    debug=config.get("debug", False),
                    startup_mode=config.get("startup_mode", "process"),
                    extra_config=config,
                )
            else:
                dashboard_config = config or DashboardConfig(name="trading", port=8080)

            dashboard = TradingDashboardAdapter(
                db_pool=db_pool, metrics_recorder=metrics_recorder, config=dashboard_config
            )

            logger.info(f"Created trading dashboard on port {dashboard_config.port}")
            return dashboard

        except ImportError as e:
            logger.error(f"Failed to import trading dashboard: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create trading dashboard: {e}")
            raise

    @staticmethod
    def create_system_dashboard(
        db_pool: IAsyncDatabase,
        metrics_recorder: IMetricsRecorder,
        orchestrator: Any | None = None,
        config: dict[str, Any] | DashboardConfig | None = None,
    ) -> IDashboard:
        """
        Create a system monitoring dashboard instance.

        Args:
            db_pool: Database connection pool
            metrics_recorder: Metrics recording interface
            orchestrator: Optional ML orchestrator for system metrics
            config: Dashboard configuration

        Returns:
            System dashboard instance implementing IDashboard

        Raises:
            ImportError: If system dashboard module not available
            Exception: If creation fails
        """
        try:
            # Local imports
            from main.monitoring.dashboards.system.adapter import SystemDashboardAdapter

            # Convert dict config to DashboardConfig if needed
            if isinstance(config, dict):
                dashboard_config = DashboardConfig(
                    name="system",
                    host=config.get("host", "0.0.0.0"),
                    port=config.get("port", 8052),
                    debug=config.get("debug", False),
                    startup_mode=config.get("startup_mode", "process"),
                    extra_config=config,
                )
            else:
                dashboard_config = config or DashboardConfig(name="system", port=8052)

            dashboard = SystemDashboardAdapter(
                db_pool=db_pool,
                metrics_recorder=metrics_recorder,
                orchestrator=orchestrator,
                config=dashboard_config,
            )

            logger.info(f"Created system dashboard on port {dashboard_config.port}")
            return dashboard

        except ImportError as e:
            logger.error(f"Failed to import system dashboard: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create system dashboard: {e}")
            raise

    @staticmethod
    def create_economic_dashboard(
        db_pool: IAsyncDatabase, config: dict[str, Any] | DashboardConfig | None = None
    ) -> IDashboard:
        """
        Create an economic indicators dashboard instance.

        Args:
            db_pool: Database connection pool
            config: Dashboard configuration

        Returns:
            Economic dashboard instance implementing IDashboard

        Raises:
            ImportError: If economic dashboard module not available
            Exception: If creation fails
        """
        try:
            # Local imports
            from main.monitoring.dashboards.economic.adapter import EconomicDashboardAdapter

            # Convert dict config to DashboardConfig if needed
            if isinstance(config, dict):
                dashboard_config = DashboardConfig(
                    name="economic",
                    host=config.get("host", "0.0.0.0"),
                    port=config.get("port", 8054),
                    debug=config.get("debug", False),
                    startup_mode=config.get("startup_mode", "process"),
                    extra_config=config,
                )
            else:
                dashboard_config = config or DashboardConfig(name="economic", port=8054)

            dashboard = EconomicDashboardAdapter(db_pool=db_pool, config=dashboard_config)

            logger.info(f"Created economic dashboard on port {dashboard_config.port}")
            return dashboard

        except ImportError as e:
            logger.error(f"Failed to import economic dashboard: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create economic dashboard: {e}")
            raise

    @staticmethod
    def create_dashboard_manager(config: dict[str, Any] | None = None) -> IDashboardManager:
        """
        Create a dashboard manager instance.

        Args:
            config: Manager configuration including:
                - health_check_interval: Seconds between health checks
                - auto_restart_failed: Whether to auto-restart failed dashboards
                - max_restart_attempts: Maximum restart attempts

        Returns:
            Dashboard manager instance implementing IDashboardManager

        Raises:
            ImportError: If dashboard manager module not available
            Exception: If creation fails
        """
        try:
            # Local imports
            from main.monitoring.core.dashboard_manager import ProcessBasedDashboardManager

            manager_config = config or {
                "health_check_interval": 30,
                "auto_restart_failed": True,
                "max_restart_attempts": 3,
            }

            manager = ProcessBasedDashboardManager(manager_config)

            logger.info("Created process-based dashboard manager")
            return manager

        except ImportError as e:
            logger.error(f"Failed to import dashboard manager: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create dashboard manager: {e}")
            raise

    @staticmethod
    def create_test_dashboard(name: str = "test", port: int = 9999) -> IDashboard:
        """
        Create a mock dashboard for testing.

        Args:
            name: Dashboard name
            port: Dashboard port

        Returns:
            Mock dashboard instance for testing
        """
        try:
            # Local imports
            from main.monitoring.testing.mock_dashboard import MockDashboard

            config = DashboardConfig(
                name=name, port=port, startup_mode="async"  # Mock dashboards run async
            )

            return MockDashboard(config)

        except ImportError:
            # If no mock available, create a simple stub
            logger.warning("Mock dashboard not available, creating stub")

            class StubDashboard:
                def __init__(self, config):
                    self.config = config
                    self.status = {"state": "initialized"}

                async def initialize(self):
                    self.status["state"] = "ready"

                async def start(self):
                    self.status["state"] = "running"

                async def stop(self, timeout=30.0):
                    self.status["state"] = "stopped"

                def get_status(self):
                    return self.status.copy()

                def get_url(self):
                    return f"http://localhost:{self.config.port}"

                async def health_check(self):
                    return self.status["state"] == "running"

            return StubDashboard(config)

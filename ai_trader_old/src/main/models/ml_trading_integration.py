"""
ML Trading Integration Module

This module provides the integration layer between ML models and the live trading system.
It initializes and coordinates all ML-related components with the trading engine.
"""

# Standard library imports
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Local imports
# Configuration
from main.config.config_manager import get_config
from main.models.inference.model_registry import ModelRegistry
from main.models.inference.prediction_engine import PredictionEngine

# ML Components
from main.models.ml_trading_service import MLTradingService
from main.models.monitoring.model_monitor import ModelMonitor
from main.monitoring.alerts.alert_manager import AlertManager

# Trading Components
from main.trading_engine.core.execution_engine import ExecutionEngine
from main.trading_engine.signals.unified_signal import UnifiedSignalHandler

logger = logging.getLogger(__name__)


class MLTradingIntegration:
    """
    Integration layer for ML-based trading.

    This class:
    1. Initializes all ML components
    2. Connects ML service to trading engine
    3. Sets up monitoring and alerts
    4. Manages the ML trading lifecycle
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        signal_handler: UnifiedSignalHandler,
        alert_manager: AlertManager,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ML trading integration.

        Args:
            execution_engine: Main execution engine instance
            signal_handler: Unified signal handler instance
            alert_manager: Alert manager instance
            config: System configuration
        """
        self.config = config or get_config()
        self.ml_config = self.config.get("ml_trading", {})

        # Trading components
        self.execution_engine = execution_engine
        self.signal_handler = signal_handler
        self.alert_manager = alert_manager

        # ML components (to be initialized)
        self.model_registry: Optional[ModelRegistry] = None
        self.prediction_engine: Optional[PredictionEngine] = None
        self.model_monitor: Optional[ModelMonitor] = None
        self.ml_trading_service: Optional[MLTradingService] = None

        self.is_initialized = False

        logger.info("MLTradingIntegration created")

    async def initialize(self) -> bool:
        """
        Initialize all ML components and connect to trading system.

        Returns:
            True if initialization successful
        """
        if self.is_initialized:
            logger.warning("ML Trading Integration already initialized")
            return True

        try:
            logger.info("Initializing ML Trading Integration...")

            # Check if ML trading is enabled
            if not self.ml_config.get("enabled", False):
                logger.info("ML trading is disabled in configuration")
                return True

            # Initialize model registry
            models_dir = Path(self.config.get("paths", {}).get("models", "models/trained"))
            self.model_registry = ModelRegistry(
                models_dir=models_dir,
                prediction_engine=None,  # Will set after creating prediction engine
                config=self.config,
            )

            # Initialize prediction engine
            self.prediction_engine = PredictionEngine(config=self.config)

            # Update model registry with prediction engine
            self.model_registry.prediction_engine = self.prediction_engine

            # Initialize model monitor
            self.model_monitor = ModelMonitor(
                config=self.config,
                model_registry=self.model_registry,
                prediction_engine=self.prediction_engine,
                alert_manager=self.alert_manager,
            )

            # Initialize ML trading service
            self.ml_trading_service = MLTradingService(
                config=self.config,
                model_registry=self.model_registry,
                prediction_engine=self.prediction_engine,
                signal_handler=self.signal_handler,
                model_monitor=self.model_monitor,
            )

            # Initialize the ML trading service
            success = await self.ml_trading_service.initialize()
            if not success:
                logger.error("Failed to initialize ML Trading Service")
                return False

            # Start model monitoring
            await self.model_monitor.start()

            # Connect signal handler to execution engine
            self._connect_signal_flow()

            self.is_initialized = True
            logger.info("ML Trading Integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ML Trading Integration: {e}")
            return False

    def _connect_signal_flow(self):
        """Connect ML signal flow to execution engine."""

        # Set execution callback on signal handler
        async def ml_signal_execution_callback(instruction: Dict[str, Any]):
            """Callback to execute ML-generated signals."""
            try:
                signal = instruction.get("signal")
                if not signal:
                    return

                # Create order from signal
                # Local imports
                from main.models.common import Order, OrderSide, OrderType, TimeInForce

                order = Order(
                    order_id=f"ml_{instruction['source_name']}_{datetime.now().timestamp()}",
                    symbol=signal.symbol,
                    side=OrderSide.BUY if signal.direction == "buy" else OrderSide.SELL,
                    quantity=int(signal.size * 100),  # Convert size to shares
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                    strategy=instruction["source_name"],
                    metadata=instruction.get("metadata", {}),
                )

                # Submit order through execution engine
                order_id = await self.execution_engine.submit_cross_system_order(order)

                if order_id:
                    logger.info(f"ML order submitted: {order_id} for {signal.symbol}")
                else:
                    logger.warning(f"Failed to submit ML order for {signal.symbol}")

            except Exception as e:
                logger.error(f"Error executing ML signal: {e}")

        # Register callback if not already set
        if not self.signal_handler.execution_callback:
            self.signal_handler.set_execution_callback(ml_signal_execution_callback)
            logger.info("ML signal execution callback registered")

    async def start_ml_trading(self) -> bool:
        """
        Start ML trading operations.

        Returns:
            True if started successfully
        """
        if not self.is_initialized:
            logger.error("ML Trading Integration not initialized")
            return False

        try:
            if self.ml_trading_service:
                await self.ml_trading_service.start_trading()
                logger.info("ML trading started")
                return True
            else:
                logger.warning("No ML trading service available")
                return False

        except Exception as e:
            logger.error(f"Failed to start ML trading: {e}")
            return False

    async def stop_ml_trading(self):
        """Stop ML trading operations."""
        try:
            if self.ml_trading_service:
                await self.ml_trading_service.stop_trading()
                logger.info("ML trading stopped")

        except Exception as e:
            logger.error(f"Error stopping ML trading: {e}")

    async def shutdown(self):
        """Shutdown all ML components."""
        logger.info("Shutting down ML Trading Integration...")

        try:
            # Stop ML trading first
            await self.stop_ml_trading()

            # Shutdown components in reverse order
            if self.ml_trading_service:
                await self.ml_trading_service.shutdown()

            if self.model_monitor:
                await self.model_monitor.stop()

            self.is_initialized = False
            logger.info("ML Trading Integration shutdown complete")

        except Exception as e:
            logger.error(f"Error during ML integration shutdown: {e}")

    def get_ml_status(self) -> Dict[str, Any]:
        """Get comprehensive ML trading status."""
        status = {
            "initialized": self.is_initialized,
            "ml_enabled": self.ml_config.get("enabled", False),
        }

        if self.is_initialized:
            # Add component statuses
            if self.ml_trading_service:
                status["ml_service"] = self.ml_trading_service.get_service_status()

            if self.model_monitor:
                status["model_monitoring"] = self.model_monitor.get_monitoring_summary()

            if self.model_registry:
                status["registered_models"] = len(self.model_registry.list_models())
                status["active_models"] = len(self.model_registry.active_deployments)

        return status


# Factory function for creating ML integration
async def create_ml_trading_integration(
    execution_engine: ExecutionEngine,
    signal_handler: UnifiedSignalHandler,
    alert_manager: AlertManager,
    config: Optional[Dict[str, Any]] = None,
) -> MLTradingIntegration:
    """
    Create and initialize ML trading integration.

    Args:
        execution_engine: Execution engine instance
        signal_handler: Signal handler instance
        alert_manager: Alert manager instance
        config: System configuration

    Returns:
        Initialized MLTradingIntegration instance
    """
    integration = MLTradingIntegration(
        execution_engine=execution_engine,
        signal_handler=signal_handler,
        alert_manager=alert_manager,
        config=config,
    )

    success = await integration.initialize()
    if not success:
        raise RuntimeError("Failed to initialize ML Trading Integration")

    return integration

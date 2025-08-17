"""
ML Trading Orchestrator

Main orchestrator for ML-powered trading that coordinates between:
- ML models and prediction engine
- Trading system components (ExecutionEngine, TradingSystem)
- Signal generation and routing
- Risk management
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.models.inference.prediction_engine import PredictionEngine

# Import ML components
from main.models.ml_trading_service import MLTradingService
from main.models.strategies.ml_regression_strategy import MLRegressionStrategy

# Import monitoring and configuration
from main.monitoring.alerts.alert_manager import AlertManager
from main.trading_engine.brokers.broker_factory import BrokerFactory

# Import trading system components
from main.trading_engine.core.execution_engine import ExecutionEngine, ExecutionMode
from main.trading_engine.core.trading_system import TradingMode, TradingSystem
from main.trading_engine.signals.unified_signal import UnifiedSignalHandler
from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class MLOrchestratorStatus:
    """Status of ML orchestrator components."""

    is_running: bool
    trading_enabled: bool
    ml_enabled: bool
    broker_connected: bool
    active_models: list[str]
    active_strategies: list[str]
    error_count: int
    last_prediction_time: datetime | None
    system_health: str  # 'healthy', 'degraded', 'error'


class MLOrchestrator:
    """
    Main orchestrator for ML-powered trading system.

    Coordinates:
    - ML model predictions and signal generation
    - Trading system execution
    - Risk management and monitoring
    - System health and status
    """

    def __init__(self, config: Any, db_pool=None, data_source_manager=None, event_bus=None):
        """
        Initialize ML orchestrator.

        Args:
            config: System configuration
            db_pool: Optional database connection pool
            data_source_manager: Optional data source manager
            event_bus: Optional event bus for component communication
        """
        self.config = config
        self.is_running = False
        self.trading_enabled = False

        # External dependencies
        self.db_pool = db_pool
        self.data_source_manager = data_source_manager
        self.event_bus = event_bus

        # Core components (to be initialized)
        self.broker = None
        self.trading_system: TradingSystem | None = None
        self.execution_engine: ExecutionEngine | None = None
        self.signal_handler: UnifiedSignalHandler | None = None
        self.alert_manager: AlertManager | None = None

        # ML components
        self.ml_service: MLTradingService | None = None
        self.prediction_engine: PredictionEngine | None = None
        self.ml_strategies: dict[str, MLRegressionStrategy] = {}

        # Tracking
        self.active_tasks: list[asyncio.Task] = []
        self.error_count = 0
        self.last_prediction_time: datetime | None = None

        logger.info("ML Orchestrator initialized with external dependencies")

    async def initialize(self):
        """Initialize all components."""
        try:
            logger.info("Initializing ML Orchestrator components...")

            # 1. Create broker
            broker_type = self.config.get("broker.type", "paper")
            self.broker = BrokerFactory.create_broker(broker_type, self.config)
            await self.broker.connect()
            logger.info(f"✅ Broker connected: {broker_type}")

            # 2. Initialize trading system
            mode = TradingMode.PAPER if broker_type == "paper" else TradingMode.LIVE
            self.trading_system = TradingSystem(
                broker_interface=self.broker, config=self.config, mode=mode
            )
            await self.trading_system.initialize()
            logger.info("✅ Trading system initialized")

            # 3. Initialize execution engine
            self.execution_engine = ExecutionEngine(
                config=self.config, trading_mode=mode, execution_mode=ExecutionMode.SEMI_AUTO
            )
            logger.info("✅ Execution engine initialized")

            # 4. Initialize signal handler
            self.signal_handler = UnifiedSignalHandler(self.config)
            logger.info("✅ Signal handler initialized")

            # 5. Initialize alert manager
            self.alert_manager = AlertManager(self.config)
            logger.info("✅ Alert manager initialized")

            # 6. Initialize ML components
            await self._initialize_ml_components()

            self.is_running = True
            logger.info("✅ ML Orchestrator initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize ML Orchestrator: {e}")
            raise

    async def _initialize_ml_components(self):
        """Initialize ML-specific components."""
        try:
            # Check if ML trading is enabled
            ml_enabled = self.config.get("ml_trading.enabled", False)
            if not ml_enabled:
                logger.info("ML trading is disabled in configuration")
                return

            # Initialize prediction engine
            self.prediction_engine = PredictionEngine(self.config)

            # Initialize ML service
            self.ml_service = MLTradingService(
                execution_engine=self.execution_engine,
                signal_handler=self.signal_handler,
                alert_manager=self.alert_manager,
                config=self.config,
            )

            # Load configured models
            ml_models = self.config.get("ml_trading.models", [])
            for model_config in ml_models:
                if model_config.get("enabled", False):
                    model_id = model_config["model_id"]
                    symbol = model_config["symbol"]

                    # Create strategy for this model
                    strategy = MLRegressionStrategy(
                        model_id=model_id, symbol=symbol, config=self.config
                    )
                    self.ml_strategies[model_id] = strategy

                    logger.info(f"✅ Loaded ML model: {model_id} for {symbol}")

            logger.info(f"✅ ML components initialized with {len(self.ml_strategies)} models")

        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            # ML initialization failure is non-fatal
            self.ml_service = None

    async def run(self):
        """Run the orchestrator main loop."""
        if not self.is_running:
            await self.initialize()

        try:
            logger.info("Starting ML Orchestrator main loop...")

            # Start background tasks
            await self._start_background_tasks()

            # Enable trading if configured
            if self.config.get("trading.auto_enable", False):
                await self.enable_trading()

            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)

                # Check system health periodically
                if int(datetime.now().timestamp()) % 60 == 0:
                    await self._check_system_health()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in orchestrator main loop: {e}")
            self.error_count += 1
        finally:
            await self.shutdown()

    async def _start_background_tasks(self):
        """Start background processing tasks."""
        # Start ML prediction loop if enabled
        if self.ml_service:
            ml_task = asyncio.create_task(self._ml_prediction_loop())
            self.active_tasks.append(ml_task)
            logger.info("Started ML prediction loop")

        # Start signal processing
        signal_task = asyncio.create_task(self._signal_processing_loop())
        self.active_tasks.append(signal_task)

        # Start monitoring
        monitor_task = asyncio.create_task(self._monitoring_loop())
        self.active_tasks.append(monitor_task)

    async def _ml_prediction_loop(self):
        """Main loop for ML predictions."""
        while self.is_running and self.ml_service:
            try:
                if not self.trading_enabled:
                    await asyncio.sleep(5)
                    continue

                # Get active models
                for model_id, strategy in self.ml_strategies.items():
                    if not strategy.config.get("enabled", False):
                        continue

                    symbol = strategy.symbol

                    # Generate prediction
                    prediction = await self.prediction_engine.predict(model_id, symbol)
                    if prediction:
                        # Convert to signal
                        signal = await strategy.generate_signal(prediction)
                        if signal:
                            # Route signal
                            await self.signal_handler.process_signal(signal)
                            self.last_prediction_time = datetime.now(UTC)

                # Wait before next prediction cycle
                interval = self.config.get("ml_trading.prediction_interval_seconds", 60)
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in ML prediction loop: {e}")
                self.error_count += 1
                await asyncio.sleep(30)  # Wait before retry

    async def _signal_processing_loop(self):
        """Process signals from all sources."""
        while self.is_running:
            try:
                # Signal handler processes its own queue
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in signal processing: {e}")
                await asyncio.sleep(5)

    async def _monitoring_loop(self):
        """Monitor system health and performance."""
        while self.is_running:
            try:
                # Update metrics every 30 seconds
                await asyncio.sleep(30)

                # Collect metrics
                if self.trading_system:
                    system_status = await self.trading_system.get_system_status()
                    logger.debug(f"System status: {system_status}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_system_health(self):
        """Check overall system health."""
        try:
            health_issues = []

            # Check broker connection
            if self.broker and not await self.broker.is_connected():
                health_issues.append("Broker disconnected")

            # Check error rate
            if self.error_count > 10:
                health_issues.append(f"High error count: {self.error_count}")

            # Check ML predictions
            if self.ml_service and self.last_prediction_time:
                time_since_prediction = (datetime.now(UTC) - self.last_prediction_time).seconds
                if time_since_prediction > 300:  # 5 minutes
                    health_issues.append("No recent ML predictions")

            # Log health status
            if health_issues:
                logger.warning(f"System health issues: {health_issues}")
            else:
                logger.debug("System health check passed")

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    async def enable_trading(self):
        """Enable live trading."""
        if self.trading_system:
            await self.trading_system.enable_trading()
            self.trading_enabled = True
            logger.info("✅ Trading enabled")

    async def disable_trading(self):
        """Disable live trading."""
        if self.trading_system:
            await self.trading_system.disable_trading()
            self.trading_enabled = False
            logger.info("⚠️ Trading disabled")

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get trading system status
            trading_status = {}
            if self.trading_system:
                trading_status = await self.trading_system.get_system_status()

            # Get ML status
            ml_status = {}
            if self.ml_service:
                ml_status = self.ml_service.get_ml_status()

            # Build orchestrator status
            status = MLOrchestratorStatus(
                is_running=self.is_running,
                trading_enabled=self.trading_enabled,
                ml_enabled=self.ml_service is not None,
                broker_connected=self.broker is not None and await self.broker.is_connected(),
                active_models=list(self.ml_strategies.keys()),
                active_strategies=[s.__class__.__name__ for s in self.ml_strategies.values()],
                error_count=self.error_count,
                last_prediction_time=self.last_prediction_time,
                system_health="healthy" if self.error_count < 5 else "degraded",
            )

            return {
                "orchestrator": {
                    "is_running": status.is_running,
                    "trading_enabled": status.trading_enabled,
                    "ml_enabled": status.ml_enabled,
                    "broker_connected": status.broker_connected,
                    "active_models": status.active_models,
                    "error_count": status.error_count,
                    "last_prediction_time": (
                        status.last_prediction_time.isoformat()
                        if status.last_prediction_time
                        else None
                    ),
                    "system_health": status.system_health,
                },
                "trading_system": trading_status,
                "ml_system": ml_status,
                "health": status.system_health,
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"health": "ERROR", "error": str(e)}

    def set_broker(self, broker):
        """
        Set a custom broker instance.

        Args:
            broker: Broker instance to use
        """
        self.broker = broker
        logger.info(f"Broker set to: {type(broker).__name__}")

        # Update trading system if already initialized
        if self.trading_system:
            self.trading_system.broker_interface = broker

        # Update execution engine if already initialized
        # Note: ExecutionEngine doesn't have a direct broker setter
        # It manages brokers through trading_systems
        if self.execution_engine and self.execution_engine.trading_systems:
            # Update the broker in the trading systems
            for system_name, trading_system in self.execution_engine.trading_systems.items():
                trading_system.broker_interface = broker

    async def shutdown(self):
        """Shutdown orchestrator and all components."""
        logger.info("Shutting down ML Orchestrator...")
        self.is_running = False

        # Cancel background tasks
        for task in self.active_tasks:
            task.cancel()

        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

        # Shutdown components
        if self.trading_system:
            await self.trading_system.shutdown()

        if self.broker:
            await self.broker.disconnect()

        logger.info("✅ ML Orchestrator shutdown complete")

"""
ML Trading Service - Orchestrates ML model integration for live trading.

This service manages the lifecycle of ML-based trading strategies, coordinating
between market data, feature generation, predictions, signal generation, and execution.
"""

# Standard library imports
import asyncio
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Local imports
# Market data and configuration
from main.config.config_manager import get_config
from main.models.inference.feature_pipeline import RealTimeFeaturePipeline
from main.models.inference.model_registry import ModelRegistry

# Core ML components
from main.models.inference.prediction_engine import PredictionEngine
from main.models.monitoring.model_monitor import ModelMonitor
from main.models.strategies.base_strategy import Signal

# Strategy components
from main.models.strategies.ml_regression_strategy import MLRegressionStrategy

# Trading components
from main.trading_engine.signals.unified_signal import (
    SignalPriority,
    SignalSource,
    UnifiedSignal,
    UnifiedSignalHandler,
)
from main.utils.cache import CacheType, get_global_cache

logger = logging.getLogger(__name__)


class MLServiceStatus(Enum):
    """ML Trading Service status."""

    STOPPED = "stopped"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"


class MLTradingService:
    """
    Orchestrates ML model integration for live trading.

    This service:
    1. Loads trained models from ModelRegistry
    2. Subscribes to market data events
    3. Generates real-time features
    4. Makes predictions using loaded models
    5. Converts predictions to trading signals
    6. Routes signals to the trading engine
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_registry: Optional[ModelRegistry] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        signal_handler: Optional[UnifiedSignalHandler] = None,
        model_monitor: Optional[ModelMonitor] = None,
    ):
        """
        Initialize ML Trading Service.

        Args:
            config: System configuration
            model_registry: Model registry instance
            prediction_engine: Prediction engine instance
            signal_handler: Unified signal handler instance
            model_monitor: Model monitor instance
        """
        self.config = config or get_config()
        self.ml_config = self.config.get("ml_trading", {})

        # Core components
        self.model_registry = model_registry
        self.prediction_engine = prediction_engine
        self.signal_handler = signal_handler
        self.model_monitor = model_monitor

        # ML components
        self.feature_pipeline = RealTimeFeaturePipeline(
            lookback_periods=self.ml_config.get("feature_pipeline", {}).get("lookback_periods"),
            buffer_max_size=self.ml_config.get("feature_pipeline", {}).get("buffer_size", 500),
            cache_ttl_seconds=self.ml_config.get("feature_pipeline", {}).get(
                "cache_ttl_seconds", 5
            ),
        )

        # Strategy instances by model_id
        self.strategies: Dict[str, MLRegressionStrategy] = {}

        # Active models configuration
        self.model_configs = self.ml_config.get("models", [])
        self.active_models: Set[str] = set()

        # Service configuration
        self.prediction_interval = self.ml_config.get("service", {}).get(
            "prediction_interval_seconds", 60
        )
        self.max_concurrent_predictions = self.ml_config.get("service", {}).get(
            "max_concurrent_predictions", 10
        )

        # State management
        self.status = MLServiceStatus.STOPPED
        self.last_prediction_time: Dict[str, datetime] = {}

        # Background tasks
        self._prediction_task: Optional[asyncio.Task] = None
        self._market_data_task: Optional[asyncio.Task] = None

        # Cache for market data
        self.cache = get_global_cache()

        logger.info("MLTradingService initialized")

    async def initialize(self) -> bool:
        """
        Initialize the ML trading service and load models.

        Returns:
            True if initialization successful
        """
        try:
            self.status = MLServiceStatus.INITIALIZING
            logger.info("Initializing ML Trading Service...")

            # Validate components
            if not all(
                [
                    self.model_registry,
                    self.prediction_engine,
                    self.signal_handler,
                    self.model_monitor,
                ]
            ):
                logger.error("Missing required components for ML Trading Service")
                self.status = MLServiceStatus.ERROR
                return False

            # Load configured models
            await self._load_configured_models()

            if not self.active_models:
                logger.warning("No active models loaded for ML trading")
                self.status = MLServiceStatus.READY
                return True

            # Start background tasks
            self._prediction_task = asyncio.create_task(self._prediction_loop())
            self._market_data_task = asyncio.create_task(self._market_data_subscription())

            self.status = MLServiceStatus.READY
            logger.info(
                f"ML Trading Service initialized with {len(self.active_models)} active models"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ML Trading Service: {e}")
            self.status = MLServiceStatus.ERROR
            return False

    async def _load_configured_models(self):
        """Load models configured for live trading."""
        for model_config in self.model_configs:
            try:
                model_id = model_config["model_id"]
                symbol = model_config["symbol"]

                # Get latest production model version
                model_version = self.model_registry.get_latest_version(
                    model_id, status="production"
                )

                if not model_version:
                    logger.warning(f"No production version found for model {model_id}")
                    continue

                # Create strategy instance
                model_path = Path(model_version.model_file_path).parent
                strategy = MLRegressionStrategy(
                    model_path=str(model_path),
                    config=self.config,
                    feature_engine=None,  # Will use internal feature pipeline
                )

                # Store strategy and mark model as active
                self.strategies[model_id] = strategy
                self.active_models.add(model_id)

                logger.info(f"Loaded model {model_id} version {model_version.version} for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load model {model_config.get('model_id', 'unknown')}: {e}")

    async def start_trading(self):
        """Start active trading with ML models."""
        if self.status != MLServiceStatus.READY:
            logger.error("Service must be in READY status to start trading")
            return

        self.status = MLServiceStatus.ACTIVE
        logger.info("ML Trading Service started")

    async def stop_trading(self):
        """Stop ML trading operations."""
        self.status = MLServiceStatus.READY
        logger.info("ML Trading Service stopped")

    async def _prediction_loop(self):
        """Background loop for generating predictions."""
        while self.status in [MLServiceStatus.READY, MLServiceStatus.ACTIVE]:
            try:
                if self.status == MLServiceStatus.ACTIVE:
                    # Process each active model
                    tasks = []
                    for model_config in self.model_configs:
                        if model_config["model_id"] in self.active_models:
                            tasks.append(self._process_model_prediction(model_config))

                    # Run predictions concurrently with limit
                    if tasks:
                        # Process in batches to respect concurrency limit
                        for i in range(0, len(tasks), self.max_concurrent_predictions):
                            batch = tasks[i : i + self.max_concurrent_predictions]
                            await asyncio.gather(*batch, return_exceptions=True)

                # Wait for next prediction interval
                await asyncio.sleep(self.prediction_interval)

            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def _process_model_prediction(self, model_config: Dict[str, Any]):
        """Process prediction for a single model."""
        try:
            model_id = model_config["model_id"]
            symbol = model_config["symbol"]

            # Check if it's time for next prediction
            last_time = self.last_prediction_time.get(model_id)
            if last_time:
                elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
                if elapsed < self.prediction_interval:
                    return

            # Get latest market data
            market_data = await self._get_latest_market_data(symbol)
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return

            # Update feature pipeline with latest data
            features = await self.feature_pipeline.update_and_calculate_features(
                symbol=symbol,
                timestamp=market_data["timestamp"],
                latest_ohlcv=market_data,
                interval="1min",
            )

            if not features:
                logger.warning(f"No features calculated for {symbol}")
                return

            # Get strategy instance
            strategy = self.strategies.get(model_id)
            if not strategy:
                logger.error(f"No strategy found for model {model_id}")
                return

            # Convert features to DataFrame for strategy
            # Third-party imports
            import pandas as pd

            features_df = pd.DataFrame([features])
            features_df.index = pd.DatetimeIndex([market_data["timestamp"]])

            # Get current position if available
            current_position = await self._get_current_position(symbol)

            # Generate signals
            signals = await strategy.generate_signals(
                symbol=symbol, features=features_df, current_position=current_position
            )

            if signals:
                # Convert to unified signals and submit
                await self._submit_signals(signals, model_id, model_config)

                # Record prediction for monitoring
                if self.model_monitor and signals[0].metadata:
                    self.model_monitor.record_prediction(
                        model_id=model_id,
                        features=features,
                        prediction=signals[0].metadata.get("predicted_return", 0.0),
                    )

            # Update last prediction time
            self.last_prediction_time[model_id] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(
                f"Error processing prediction for {model_config.get('model_id', 'unknown')}: {e}"
            )

    async def _get_latest_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol."""
        try:
            # Try cache first
            cache_key = f"market_data:{symbol}:latest"
            cached_data = await self.cache.get(CacheType.MARKET, cache_key)

            if cached_data:
                return cached_data

            # If not in cache, would need to fetch from market data provider
            # For now, return None - this would be implemented based on your data source
            return None

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _get_current_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for symbol."""
        try:
            # Try cache for position data
            cache_key = f"position:{symbol}"
            position_data = await self.cache.get(CacheType.POSITION, cache_key)

            if position_data:
                return {
                    "symbol": symbol,
                    "quantity": position_data.get("quantity", 0),
                    "direction": "long" if position_data.get("quantity", 0) > 0 else "short",
                }

            return None

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None

    async def _submit_signals(
        self, signals: List[Signal], model_id: str, model_config: Dict[str, Any]
    ):
        """Submit signals to unified signal handler."""
        try:
            # Convert to unified signals
            unified_signals = []

            for signal in signals:
                # Skip non-actionable signals
                if signal.direction == "hold":
                    continue

                unified_signal = UnifiedSignal(
                    signal=signal,
                    source=SignalSource.STRATEGY,
                    source_name=f"ml_{model_id}",
                    priority=self._get_signal_priority(signal, model_config),
                    timestamp=datetime.now(timezone.utc),
                    correlation_group="ml_predictions",
                    metadata={
                        "model_id": model_id,
                        "model_config": model_config,
                        **signal.metadata,
                    },
                )
                unified_signals.append(unified_signal)

            if unified_signals:
                # Submit to signal handler
                await self.signal_handler.add_signals(
                    signals=[s.signal for s in unified_signals],
                    source=SignalSource.STRATEGY,
                    source_name=f"ml_{model_id}",
                    metadata={"ml_service": True},
                )

                logger.info(f"Submitted {len(unified_signals)} signals from {model_id}")

        except Exception as e:
            logger.error(f"Error submitting signals: {e}")

    def _get_signal_priority(self, signal: Signal, model_config: Dict[str, Any]) -> SignalPriority:
        """Determine signal priority based on confidence and configuration."""
        # High confidence signals get higher priority
        if signal.confidence > 0.8:
            return SignalPriority.HIGH
        elif signal.confidence > 0.6:
            return SignalPriority.NORMAL
        else:
            return SignalPriority.LOW

    async def _market_data_subscription(self):
        """Subscribe to market data updates."""
        # This would connect to your market data provider
        # For now, it's a placeholder that would be implemented
        # based on your specific data source (Polygon, Alpaca, etc.)
        while self.status in [MLServiceStatus.READY, MLServiceStatus.ACTIVE]:
            try:
                # Placeholder for market data subscription
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in market data subscription: {e}")
                await asyncio.sleep(5)

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            "status": self.status.value,
            "active_models": list(self.active_models),
            "total_models": len(self.model_configs),
            "strategies_loaded": len(self.strategies),
            "last_predictions": {
                model_id: time.isoformat() if time else None
                for model_id, time in self.last_prediction_time.items()
            },
            "feature_cache_size": self.feature_pipeline.get_feature_cache_size(),
        }

    async def shutdown(self):
        """Shutdown ML trading service."""
        logger.info("Shutting down ML Trading Service...")

        self.status = MLServiceStatus.STOPPED

        # Cancel background tasks
        if self._prediction_task:
            self._prediction_task.cancel()
        if self._market_data_task:
            self._market_data_task.cancel()

        # Wait for tasks to complete
        if self._prediction_task:
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass

        if self._market_data_task:
            try:
                await self._market_data_task
            except asyncio.CancelledError:
                pass

        logger.info("ML Trading Service shutdown complete")

"""
Automated model retraining scheduler.
Triggers retraining based on schedule, performance degradation, or drift detection.
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.processing.orchestrator import ProcessingOrchestrator
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
from main.models.inference.model_registry import ModelRegistry
from main.models.monitoring.model_monitor import ModelMonitor
from main.models.training.training_orchestrator import ModelTrainingOrchestrator

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Manages automated model retraining.

    Features:
    - Scheduled retraining (daily, weekly, monthly)
    - Performance-based retraining triggers
    - Drift-based retraining
    - Parallel model training
    - Automatic validation and deployment
    """

    def __init__(self, config: Any = None):
        """Initialize retraining scheduler."""
        if config is None:
            config = get_config()
        self.config = config

        # Training components (will be initialized when needed)
        self.training_orchestrator = None
        self.data_orchestrator = None
        self.feature_orchestrator = None
        self.model_registry = None
        self.model_monitor = None

        # Scheduling configuration
        self.schedules = {
            "daily": config.get("ml.retraining.daily_models", []),
            "weekly": config.get("ml.retraining.weekly_models", []),
            "monthly": config.get("ml.retraining.monthly_models", []),
        }

        # Performance thresholds
        self.performance_threshold = config.get("ml.retraining.performance_threshold", 0.2)
        self.drift_threshold = config.get("ml.retraining.drift_threshold", 0.1)

        # State tracking
        self.last_trained: Dict[str, datetime] = {}
        self.training_in_progress: Dict[str, bool] = {}

        # Scheduler task
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("RetrainingScheduler initialized")

    async def initialize_components(
        self, model_registry: ModelRegistry, model_monitor: ModelMonitor
    ):
        """Initialize training components."""
        self.model_registry = model_registry
        self.model_monitor = model_monitor

        # Initialize training pipeline components
        self.data_orchestrator = ProcessingOrchestrator(self.config)

        self.feature_orchestrator = FeatureOrchestrator(
            data_collector=self.data_orchestrator.collector, config=self.config
        )

        self.training_orchestrator = ModelTrainingOrchestrator(self.config)
        self.training_orchestrator.initialize_components(
            feature_store=self.feature_orchestrator.feature_store,
            cache_manager=self.data_orchestrator.cache_manager,
        )

        logger.info("Training components initialized")

    async def start(self):
        """Start retraining scheduler."""
        if self._running:
            logger.warning("Retraining scheduler already running")
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Retraining scheduler started")

    async def stop(self):
        """Stop retraining scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Retraining scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduling loop."""
        while self._running:
            try:
                # Check scheduled retraining
                await self._check_scheduled_retraining()

                # Check performance-triggered retraining
                await self._check_performance_triggers()

                # Check drift-triggered retraining
                await self._check_drift_triggers()

                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _check_scheduled_retraining(self):
        """Check for scheduled retraining."""
        now = datetime.now()

        # Daily models
        for model_id in self.schedules["daily"]:
            last_trained = self.last_trained.get(model_id, datetime.min)
            if now - last_trained > timedelta(days=1):
                await self._trigger_retraining(model_id, "scheduled_daily")

        # Weekly models
        for model_id in self.schedules["weekly"]:
            last_trained = self.last_trained.get(model_id, datetime.min)
            if now - last_trained > timedelta(days=7):
                await self._trigger_retraining(model_id, "scheduled_weekly")

        # Monthly models
        for model_id in self.schedules["monthly"]:
            last_trained = self.last_trained.get(model_id, datetime.min)
            if now - last_trained > timedelta(days=30):
                await self._trigger_retraining(model_id, "scheduled_monthly")

    async def _check_performance_triggers(self):
        """Check for performance-based retraining triggers."""
        if not self.model_monitor:
            return

        monitoring_summary = self.model_monitor.get_monitoring_summary()

        for model_id, metrics in monitoring_summary["performance_metrics"].items():
            if "mae" not in metrics:
                continue

            # Get baseline performance
            model_version = self.model_registry.production_models.get(model_id)
            if not model_version or "mae" not in model_version.metrics:
                continue

            baseline_mae = model_version.metrics["mae"]
            current_mae = metrics["mae"]

            # Check degradation
            degradation = (current_mae - baseline_mae) / baseline_mae
            if degradation > self.performance_threshold:
                await self._trigger_retraining(model_id, "performance_degradation")

    async def _check_drift_triggers(self):
        """Check for drift-based retraining triggers."""
        if not self.model_monitor:
            return

        monitoring_summary = self.model_monitor.get_monitoring_summary()

        for model_id, drift_score in monitoring_summary["drift_alerts"].items():
            if drift_score > self.drift_threshold:
                await self._trigger_retraining(model_id, "drift_detected")

    async def _trigger_retraining(self, model_id: str, reason: str):
        """Trigger model retraining."""
        # Check if already training
        if self.training_in_progress.get(model_id, False):
            logger.info(f"Model {model_id} already training, skipping")
            return

        logger.info(f"Triggering retraining for {model_id}, reason: {reason}")
        self.training_in_progress[model_id] = True

        try:
            # Get model configuration
            model_config = self._get_model_config(model_id)

            # Prepare training data
            symbols = model_config.get("symbols", [])
            lookback_days = model_config.get("lookback_days", 365)

            # Generate features
            await self.feature_orchestrator.calculate_and_store_features(symbols, lookback_days)

            # Train model
            results = await self.training_orchestrator.train_model(
                symbols=symbols, model_config=model_config
            )

            # Validate results
            if results and "model" in results:
                # Register new model version
                new_version = await self.model_registry.register_model(
                    model=results["model"],
                    model_id=model_id,
                    model_type=model_config.get("type", "ensemble"),
                    training_data={
                        "symbols": symbols,
                        "lookback_days": lookback_days,
                        "reason": reason,
                    },
                    hyperparameters=results.get("hyperparameters", {}),
                    metrics=results.get("metrics", {}),
                    features=results.get("features", []),
                    metadata={"retrain_reason": reason},
                )

                # Deploy as candidate for A/B testing
                await self.model_registry.deploy_model(
                    model_id, new_version.version, deployment_pct=0.1
                )

                logger.info(f"Successfully retrained {model_id}, deployed as candidate")

            # Update last trained timestamp
            self.last_trained[model_id] = datetime.now()

        except Exception as e:
            logger.error(f"Failed to retrain {model_id}: {e}")

        finally:
            self.training_in_progress[model_id] = False

    def _get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get model configuration."""
        # Default configuration
        default_config = {
            "symbols": self.config.get("universe.symbols", []),
            "lookback_days": 365,
            "type": "ensemble",
            "models": ["xgboost", "lightgbm", "random_forest"],
        }

        # Override with model-specific config
        model_configs = self.config.get("ml.models", {})
        if model_id in model_configs:
            default_config.update(model_configs[model_id])

        return default_config

    async def force_retrain(self, model_id: str):
        """Force immediate retraining of a model."""
        await self._trigger_retraining(model_id, "manual_trigger")

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "scheduled_models": self.schedules,
            "last_trained": {
                model_id: timestamp.isoformat() for model_id, timestamp in self.last_trained.items()
            },
            "training_in_progress": list(
                model_id
                for model_id, in_progress in self.training_in_progress.items()
                if in_progress
            ),
        }

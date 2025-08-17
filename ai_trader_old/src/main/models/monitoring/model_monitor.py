# File: src/ai_trader/models/monitoring/model_monitor.py

"""
Model monitoring service for real-time performance tracking and drift detection.
Monitors model predictions, tracks performance metrics, and triggers retraining.
"""
# Standard library imports
import asyncio
from collections import defaultdict, deque  # Not used directly anymore
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

# Local imports
# Corrected absolute imports
from main.config.config_manager import get_config
from main.models.inference.model_registry import ModelRegistry  # Central ModelRegistry
from main.models.inference.prediction_engine import PredictionEngine  # Core PredictionEngine
from main.models.monitoring.monitor_helpers.ab_test_analyzer import ABTestAnalyzer
from main.models.monitoring.monitor_helpers.drift_detector import DriftDetector
from main.models.monitoring.monitor_helpers.ml_ops_action_manager import MLOpsActionManager
from main.models.monitoring.monitor_helpers.monitor_reporter import MonitorReporter
from main.models.monitoring.monitor_helpers.performance_calculator import PerformanceCalculator

# Import the new monitor helper classes
from main.models.monitoring.monitor_helpers.prediction_data_collector import PredictionDataCollector
from main.monitoring.alerts.alert_manager import AlertManager
from main.utils.core import ErrorHandlingMixin  # Mixin for custom error handling

logger = logging.getLogger(__name__)


class ModelMonitor(ErrorHandlingMixin):
    """
    Monitors deployed model performance in real-time and detects data/prediction drift.

    This class acts as an orchestrator, delegating specific tasks like data collection,
    metric calculation, drift detection, A/B test analysis, and MLOps action
    management to specialized helper components.
    """

    def __init__(
        self,
        config: Any,
        model_registry: ModelRegistry,
        prediction_engine: PredictionEngine,
        alert_manager: AlertManager,
    ):
        """
        Initializes the ModelMonitor and its composing helper components.

        Args:
            config: Application configuration dictionary.
            model_registry: An initialized ModelRegistry instance.
            prediction_engine: An initialized PredictionEngine instance.
            alert_manager: An initialized AlertManager instance.
        """
        ErrorHandlingMixin.__init__(self)  # Initialize the mixin

        self.config = config
        self.model_registry = model_registry
        self.prediction_engine = (
            prediction_engine  # Not used directly in loop, but for recording predictions
        )
        self.alert_manager = alert_manager

        # Monitoring configuration values
        self.monitoring_interval = (
            self.config.get("ml", {}).get("monitoring", {}).get("interval_seconds", 300)
        )
        self.drift_threshold = (
            self.config.get("ml", {}).get("monitoring", {}).get("drift_threshold", 0.05)
        )
        self.performance_window_hours = (
            self.config.get("ml", {}).get("monitoring", {}).get("performance_window_hours", 24)
        )
        self.min_predictions_for_analysis = (
            self.config.get("ml", {}).get("monitoring", {}).get("min_predictions", 100)
        )

        # Initialize helper components
        self._data_collector = PredictionDataCollector(
            history_maxlen=self.config.get("ml", {})
            .get("monitoring", {})
            .get("prediction_history_maxlen", 10000),
            feature_history_maxlen=self.config.get("ml", {})
            .get("monitoring", {})
            .get("feature_history_maxlen", 1000),
        )
        self._performance_calculator = PerformanceCalculator()
        self._drift_detector = DriftDetector(drift_threshold=self.drift_threshold)

        self._mlops_action_manager = MLOpsActionManager(
            alert_manager=self.alert_manager, model_registry=self.model_registry, config=self.config
        )
        self._ab_test_analyzer = ABTestAnalyzer(
            model_registry=self.model_registry,
            performance_calculator=self._performance_calculator,  # PerformanceCalculator used for comparison
            alert_manager=self.alert_manager,
            config=self.config,
        )
        self._monitor_reporter = MonitorReporter(
            model_registry=self.model_registry,
            prediction_data_collector=self._data_collector,
            performance_calculator=self._performance_calculator,
            drift_detector=self._drift_detector,
        )

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("ModelMonitor initialized with helper components.")

    async def start(self):
        """Starts the model monitoring background loop."""
        if self._running:
            logger.warning("Model monitoring already running.")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Model monitoring started.")

    async def stop(self):
        """Stops the model monitoring background loop."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task  # Wait for task to finish cancellation
            except asyncio.CancelledError:
                logger.debug("Model monitoring task explicitly cancelled during shutdown.")
            except Exception as e:
                logger.error(f"Error during ModelMonitor stop: {e}", exc_info=True)
            self._monitoring_task = None
        logger.info("Model monitoring stopped.")

    async def _monitoring_loop(self):
        """Main asynchronous loop for periodic model monitoring."""
        logger.info(
            f"Model monitoring loop started with interval: {self.monitoring_interval} seconds."
        )
        while self._running:
            try:
                # Get all actively deployed models from the registry
                # Iterate over a copy of active_deployments to avoid issues if registry changes during loop
                active_models = list(self.model_registry.active_deployments.items())

                for model_id, model_version in active_models:
                    await self._monitor_single_model_workflow(model_id, model_version)

                await self._ab_test_analyzer.check_ab_test_results()  # Check A/B tests
                self._data_collector.clean_old_data(cutoff_days=7)  # Clean old prediction history

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                logger.info("Model monitoring loop explicitly cancelled.")
                break  # Exit loop on cancellation
            except Exception as e:
                self.handle_error(
                    f"Critical error in model monitoring loop: {e}", level="critical", exc_info=True
                )
                await asyncio.sleep(60)  # Wait before retrying loop

    async def _monitor_single_model_workflow(
        self, model_id: str, model_version: Any
    ):  # Use Any for ModelVersion type
        """
        Orchestrates the monitoring workflow for a single model version.
        This includes performance, prediction drift, and feature drift checks.
        """
        try:
            # Update ModelVersion metadata with last monitored timestamp
            model_version.metadata["last_monitored_utc"] = datetime.now(timezone.utc).isoformat()

            # 1. Calculate Performance Metrics
            recent_predictions_for_perf = self._data_collector.get_recent_predictions(
                model_id=model_id,
                window_hours=self.performance_window_hours,
                min_predictions=self.min_predictions_for_analysis,
            )

            if not recent_predictions_for_perf or not all(
                p.get("actual") is not None for p in recent_predictions_for_perf
            ):
                logger.debug(
                    f"Insufficient recent predictions with actuals for performance monitoring for {model_id}."
                )
                return  # Skip if not enough data with actuals

            current_performance = self._performance_calculator.calculate_metrics(
                recent_predictions_for_perf
            )

            # Update ModelVersion metrics (EMA is handled by PerformanceTracker helper)
            self.model_registry.update_performance(
                model_id, model_version.version, current_performance
            )

            # 2. Check for Performance Degradation
            baseline_mae = model_version.metrics.get(
                "mae", float("inf")
            )  # Use ModelVersion's EMA-updated MAE as baseline
            current_mae = current_performance.get("mae", float("inf"))

            if self._performance_calculator.check_performance_degradation(
                current_mae,
                baseline_mae,
                self.config.get("ml.monitoring.degradation_threshold_mae_pct", 0.20),
            ):
                await self._mlops_action_manager.handle_performance_degradation(
                    model_id, current_performance
                )

            # 3. Check for Prediction Drift
            prediction_values_history = [
                p["prediction"]
                for p in self._data_collector.prediction_history[model_id]
                if p.get("prediction") is not None
            ]
            drift_score_ks, _p_value = self._drift_detector.calculate_prediction_drift(
                prediction_values_history, self.min_predictions_for_analysis
            )

            if drift_score_ks > self.drift_threshold:
                # Store latest drift score in model metadata for reporting
                model_version.metadata["last_prediction_drift_score"] = drift_score_ks
                self.model_registry._save_registry_state()  # Persist metadata update
                await self._mlops_action_manager.handle_prediction_drift_detected(
                    model_id, drift_score_ks
                )

            # 4. Check for Feature Drift
            model_features_list = model_version.features  # Features used by this model version
            drifted_features = self._drift_detector.check_feature_drift(
                model_id,
                self._data_collector.feature_history,
                model_features_list,
                self.min_predictions_for_analysis,
            )

            if drifted_features:
                # Store latest feature drift details in model metadata
                model_version.metadata["last_feature_drift_details"] = drifted_features
                self.model_registry._save_registry_state()  # Persist metadata update
                await self._mlops_action_manager.handle_feature_drift_detected(
                    model_id, drifted_features
                )

        except Exception as e:
            self.handle_error(
                f"Error during monitoring workflow for model {model_id}: {e}",
                level="error",
                exc_info=True,
            )
            # Log this as an operational error affecting monitoring itself.

    def record_prediction(
        self,
        model_id: str,
        features: Dict[str, float],
        prediction: float,
        actual: Optional[float] = None,
    ):
        """
        Records a model prediction for monitoring. This is the external entry point
        for prediction results to be fed into the monitor.
        Delegates data collection to PredictionDataCollector.

        Args:
            model_id: The ID of the model that made the prediction.
            features: Dictionary of feature values used.
            prediction: The predicted value.
            actual: Optional. The actual outcome (for performance evaluation).
        """
        self._data_collector.record_prediction(model_id, features, prediction, actual)
        logger.debug(f"Prediction recorded for {model_id}.")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Retrieves a comprehensive summary of the current model monitoring status.
        Delegates reporting logic to MonitorReporter.

        Returns:
            A dictionary containing the monitoring summary.
        """
        return self._monitor_reporter.get_monitoring_summary(
            is_running=self._running,
            drift_threshold=self.drift_threshold,
            monitoring_config=self.config.get("ml", {}).get("monitoring", {}),
        )

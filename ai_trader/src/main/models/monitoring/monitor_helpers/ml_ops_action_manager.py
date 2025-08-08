# File: src/ai_trader/models/monitoring/monitor_helpers/ml_ops_action_manager.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

# Corrected absolute imports
from main.monitoring.alerts.alert_manager import AlertManager # For sending alerts

# Use TYPE_CHECKING for circular dependency if needed for ModelRegistry/ModelVersion
if TYPE_CHECKING:
    from main.models.inference.model_registry import ModelRegistry
    from main.models.inference.model_registry_types import ModelVersion

logger = logging.getLogger(__name__)

class MLOpsActionManager:
    """
    Manages automated responses to detected model monitoring issues
    (e.g., performance degradation, drift). Triggers alerts, retraining,
    or model rollbacks.
    """

    def __init__(self, alert_manager: AlertManager, model_registry: "ModelRegistry", config: Dict[str, Any], training_orchestrator: Optional[Any] = None):
        """
        Initializes the MLOpsActionManager.

        Args:
            alert_manager: An initialized AlertManager instance.
            model_registry: An initialized ModelRegistry instance (for deployment actions).
            config: Application configuration for MLOps thresholds and actions.
            training_orchestrator: Optional training orchestrator instance for direct retraining calls.
        """
        self.alert_manager = alert_manager
        self.model_registry = model_registry
        self.config = config
        self.training_orchestrator = training_orchestrator
        
        self.rollback_threshold = self.config.get('ml', {}).get('monitoring', {}).get('rollback_threshold_mae', 0.5) # MAE value
        self.retrain_threshold_drift = self.config.get('ml', {}).get('monitoring', {}).get('retrain_threshold_drift', 0.1) # KS statistic
        
        logger.debug("MLOpsActionManager initialized.")

    async def handle_performance_degradation(self, model_id: str, current_performance_metrics: Dict[str, float]):
        """
        Handles detected model performance degradation.
        Sends an alert and triggers a rollback if degradation is severe.

        Args:
            model_id: The ID of the degraded model.
            current_performance_metrics: Dictionary of current performance metrics.
        """
        logger.warning(f"Performance degradation detected for model {model_id}. Current MAE: {current_performance_metrics.get('mae', 'N/A'):.4f}")
        
        alert_message = (
            f"Model {model_id} performance has degraded. "
            f"Current MAE: {current_performance_metrics.get('mae', 'N/A'):.4f}."
        )
        await self.alert_manager.send_alert(
            level='warning',
            title=f'Model Performance Degradation: {model_id}',
            message=alert_message
        )
        
        # Trigger rollback if MAE exceeds a critical threshold
        if current_performance_metrics.get('mae', float('inf')) > self.rollback_threshold:
            logger.critical(f"MAE ({current_performance_metrics.get('mae'):.4f}) for model {model_id} exceeds rollback threshold ({self.rollback_threshold:.4f}). Initiating rollback.")
            await self._rollback_model(model_id)
        else:
            logger.info(f"Performance degradation for {model_id} noted, but not severe enough for automatic rollback.")

    async def handle_prediction_drift_detected(self, model_id: str, drift_score: float):
        """
        Handles detected prediction distribution drift.
        Sends an alert and triggers retraining if drift is severe.

        Args:
            model_id: The ID of the model where drift was detected.
            drift_score: The calculated drift score (e.g., KS statistic).
        """
        logger.warning(f"Prediction drift detected for model {model_id}: {drift_score:.4f}")
        
        alert_message = (
            f"Model {model_id} showing prediction distribution drift. "
            f"Drift score (KS): {drift_score:.4f}."
        )
        await self.alert_manager.send_alert(
            level='warning',
            title=f'Model Prediction Drift Detected: {model_id}',
            message=alert_message
        )
        
        # Trigger retraining if drift score exceeds a threshold
        if drift_score > self.retrain_threshold_drift:
            logger.info(f"Drift score ({drift_score:.4f}) for model {model_id} exceeds retraining threshold ({self.retrain_threshold_drift:.4f}). Triggering retraining.")
            await self._trigger_retraining(model_id)
        else:
            logger.info(f"Prediction drift for {model_id} noted, but not severe enough for automatic retraining.")

    async def handle_feature_drift_detected(self, model_id: str, drifted_features: Dict[str, float]):
        """
        Handles detected drift in individual feature distributions.
        Sends an informational alert.

        Args:
            model_id: The ID of the model affected by feature drift.
            drifted_features: A dictionary of {feature_name: drift_score} for drifted features.
        """
        logger.warning(f"Feature drift detected for model {model_id}. Features: {list(drifted_features.keys())}")
        
        alert_message = (
            f"Features showing drift for model {model_id}: "
            f"{', '.join([f'{f} (KS:{s:.3f})' for f, s in drifted_features.items()])}."
        )
        await self.alert_manager.send_alert(
            level='info',
            title=f'Model Feature Drift Detected: {model_id}',
            message=alert_message
        )

    async def _rollback_model(self, model_id: str):
        """
        Initiates a rollback to the previous production model version for a given model ID.
        """
        logger.critical(f"Attempting automatic rollback for model {model_id} due to severe degradation.")
        
        # Find the currently deployed model version (if any)
        current_prod_model = self.model_registry.get_latest_version(model_id, status='production')
        
        # Find a suitable previous version to roll back to (e.g., previous 'production' or the one before current)
        # This logic should typically be in ModelRegistry's rollback method or a higher-level MLOps orchestrator
        # For this context, we will call ModelRegistry's rollback_model which expects a specific version.
        
        # Simple rollback: Find the most recent 'archived' model that was once production or candidate
        versions = self.model_registry.versions.get(model_id, [])
        previous_stable_version: Optional[ModelVersion] = None
        
        # Iterate through versions (latest first by created_at) to find a suitable candidate
        for v in sorted(versions, key=lambda mv: mv.created_at, reverse=True):
            if v.version == (current_prod_model.version if current_prod_model else None):
                continue # Skip current one
            if v.status in ['production', 'candidate', 'archived'] and v.metrics.get('mae', float('inf')) < self.rollback_threshold:
                # Find an older version that was stable and not current production
                previous_stable_version = v
                break
        
        if previous_stable_version:
            success = await self.model_registry.rollback_model(model_id, previous_stable_version.version)
            if success:
                await self.alert_manager.send_alert(
                    level='critical',
                    title=f'Model Rollback SUCCESS: {model_id}',
                    message=f'Model {model_id} rolled back from v{(current_prod_model.version if current_prod_model else "unknown")} to v{previous_stable_version.version} due to degradation.'
                )
                logger.info(f"Successfully rolled back model {model_id} to version {previous_stable_version.version}.")
            else:
                await self.alert_manager.send_alert(
                    level='error',
                    title=f'Model Rollback FAILED: {model_id}',
                    message=f'Attempted rollback of model {model_id} to v{previous_stable_version.version} failed.'
                )
                logger.error(f"Failed to rollback model {model_id} to version {previous_stable_version.version}.")
        else:
            logger.error(f"No suitable previous version found for rollback of model {model_id}.")
            await self.alert_manager.send_alert(
                level='error',
                title=f'Model Rollback IMPOSSIBLE: {model_id}',
                message=f'No suitable previous production/candidate version found for {model_id}. Manual intervention required.'
            )

    async def _trigger_retraining(self, model_id: str):
        """
        Triggers model retraining for a given model ID.
        This would typically involve interacting with a training orchestration service.
        """
        logger.info(f"Triggering retraining for model {model_id} due to drift.")
        
        # Integrate with training pipeline orchestrator
        try:
            # Option 1: Use event system to trigger retraining
            from main.events.core import EventBusFactory
            from main.interfaces.events import Event, EventType
            
            event_bus = EventBusFactory.create()
            retraining_event = Event(
                event_type=EventType.FEATURE_REQUEST,  # Reuse existing event type for now
                data={
                    'action': 'retrain_model',
                    'model_id': model_id,
                    'reason': 'drift',
                    'drift_score': 'detected',
                    'timestamp': asyncio.get_event_loop().time()
                }
            )
            await event_bus.publish(retraining_event)
            logger.info(f"Retraining event published for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish retraining event for model {model_id}: {e}")
            
            # Option 2: Direct integration with training orchestrator (fallback)
            try:
                # Check if we have a training orchestrator available
                if hasattr(self, 'training_orchestrator') and self.training_orchestrator:
                    await self.training_orchestrator.schedule_retraining(model_id, reason='drift')
                    logger.info(f"Direct retraining scheduled for model {model_id}")
                else:
                    logger.warning(f"No training orchestrator available for model {model_id} - manual intervention required")
                    
            except Exception as e2:
                logger.error(f"Failed to schedule direct retraining for model {model_id}: {e2}")
                # Log critical error but continue with alert notification
        
        await self.alert_manager.send_alert(
            level='info',
            title=f'Model Retraining Triggered: {model_id}',
            message=f'Model {model_id} scheduled for retraining due to detected drift.'
        )
        logger.info(f"Alert sent for retraining trigger for model {model_id}.")
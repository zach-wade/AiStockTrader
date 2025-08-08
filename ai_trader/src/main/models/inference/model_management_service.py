# File: src/ai_trader/models/inference/model_management_service.py

"""
Service for managing the lifecycle of machine learning models in the registry.

Handles deployment, promotion, rollback, and archiving operations,
coordinating with the ModelRegistry and its low-level helpers.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

# Corrected absolute imports
from main.models.inference.model_registry_types import ModelVersion
from main.models.inference.model_registry_helpers.deployment_manager import DeploymentManager
from main.models.inference.model_registry_helpers.model_archiver import ModelArchiver
from main.models.inference.model_registry_helpers.performance_tracker import PerformanceTracker

# Type hinting the ModelRegistry to avoid circular imports at runtime
if TYPE_CHECKING:
    from main.models.inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelManagementService:
    """
    Manages the operational lifecycle of machine learning models within the registry.
    This includes deployment, promotion, rollback, archiving, and performance updates.
    It orchestrates interactions between the ModelRegistry's state and its helpers.
    """

    def __init__(self, model_registry: "ModelRegistry", config: Dict[str, Any]):
        """
        Initializes the ModelManagementService.

        Args:
            model_registry: The ModelRegistry instance which this service operates on.
            config: Application configuration.
        """
        self.model_registry = model_registry # The lean ModelRegistry instance
        self.config = config

        # Helpers from ModelRegistry are passed or accessed via registry
        self._deployment_manager = model_registry._deployment_manager
        self._model_archiver = model_registry._model_archiver
        self._performance_tracker = model_registry._performance_tracker
        
        logger.info("ModelManagementService initialized.")

    async def deploy_model(self, model_id: str, version: str, deployment_pct: float = 0.0) -> bool:
        """
        Deploys a specific model version with a specified traffic percentage.
        Updates the model's status and coordinates with the prediction engine via DeploymentManager.

        Args:
            model_id: The identifier of the model.
            version: The version string to deploy.
            deployment_pct: The percentage of traffic to route to this version (0-100).

        Returns:
            True if deployment status was successfully updated, False otherwise.
        """
        version_obj = self.model_registry.get_model_version(model_id, version)
        if not version_obj:
            logger.error(f"Model {model_id} version {version} not found in registry. Cannot deploy.")
            return False

        # Delegate core deployment logic
        success = await self._deployment_manager.deploy_model_version(version_obj, deployment_pct)
        
        if success:
            # Update ModelRegistry's internal active deployments tracker and save state
            if version_obj.deployment_pct > 0:
                self.model_registry.active_deployments[model_id] = version_obj
            elif model_id in self.model_registry.active_deployments and self.model_registry.active_deployments[model_id].version == version:
                del self.model_registry.active_deployments[model_id]
            self.model_registry._save_registry_state() # Persist changes through ModelRegistry

        return success

    async def promote_model(self, model_id: str, version: str) -> bool:
        """
        Promotes a candidate model to full production (100% traffic).

        Args:
            model_id: The identifier of the model.
            version: The version string to promote.

        Returns:
            True if promotion was successful, False otherwise.
        """
        logger.info(f"Attempting to promote model {model_id} version {version} to production.")
        return await self.deploy_model(model_id, version, 100.0)

    async def rollback_model(self, model_id: str, version: str) -> bool:
        """
        Rolls back to a previous model version, making it the new production model.
        This implicitly sets the current production model (if different) to archived.

        Args:
            model_id: The identifier of the model.
            version: The version string to roll back to.

        Returns:
            True if rollback was successful, False otherwise.
        """
        logger.warning(f"Rolling back model {model_id} to version {version}.")
        return await self.deploy_model(model_id, version, 100.0)

    async def archive_old_models(self, days: int = 30) -> int:
        """
        Archives (marks as 'archived' and sets deployment_pct to 0) model versions
        that are older than a specified number of days and are not currently
        in 'production' or 'candidate' status.

        Args:
            days: The age in days beyond which models should be archived.

        Returns:
            The number of model versions that were archived.
        """
        # Pass a flat list of all versions for processing by the archiver
        all_versions_flat: List[ModelVersion] = [v for sublist in self.model_registry.versions.values() for v in sublist]
        archived_count = self._model_archiver.archive_old_model_versions(all_versions_flat, days)
        
        if archived_count > 0:
            self.model_registry._save_registry_state() # Save registry after archiving
        
        return archived_count

    def update_model_performance(self, model_id: str, version: str, performance_data: Dict[str, float]):
        """
        Updates the performance metrics for a specific model version.
        Delegates metric consolidation to PerformanceTracker and persists changes.

        Args:
            model_id: Model identifier.
            version: Model version.
            performance_data: New performance metrics to update.
        """
        version_obj = self.model_registry.get_model_version(model_id, version)
        if not version_obj:
            logger.warning(f"Model {model_id} version {version} not found for performance update. Skipping.")
            return

        self._performance_tracker.update_model_performance(version_obj, performance_data)
        self.model_registry._save_registry_state() # Save registry after updating metrics
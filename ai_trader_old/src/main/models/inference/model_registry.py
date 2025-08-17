# File: src/ai_trader/models/inference/model_registry.py

"""
Centralized registry for managing trained machine learning models.

This module defines the core ModelRegistry class, which acts as a robust
data store for model versions. It primarily handles the registration of
new models, loading/saving its own metadata, and providing basic accessors
to model version information. Higher-level MLOps operations (deployment,
analysis) are delegated to specialized services.
"""

# Standard library imports
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Local imports
# Corrected absolute imports
from main.config.config_manager import get_config
from main.models.inference.model_registry_helpers.deployment_manager import DeploymentManager
from main.models.inference.model_registry_helpers.model_archiver import ModelArchiver
from main.models.inference.model_registry_helpers.model_comparison_analyzer import (
    ModelComparisonAnalyzer,
)
from main.models.inference.model_registry_helpers.model_exporter import ModelExporter
from main.models.inference.model_registry_helpers.model_file_manager import ModelFileManager
from main.models.inference.model_registry_helpers.performance_tracker import PerformanceTracker

# Import the new ModelRegistry helper classes (low-level specific tasks)
from main.models.inference.model_registry_helpers.registry_storage_manager import (
    RegistryStorageManager,
)
from main.models.inference.model_registry_helpers.traffic_router import TrafficRouter

# Import the ModelVersion dataclass (now in its own types file)
from main.models.inference.model_registry_types import ModelVersion
from main.models.inference.prediction_engine import PredictionEngine  # Dependency for its helpers

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized registry for managing trained machine learning models.

    This class serves as the authoritative data store for model versions and
    their metadata. It composes low-level helpers for persistence and
    delegates higher-level MLOps operations to specialized service classes
    (ModelManagementService, ModelAnalyticsService).
    """

    def __init__(
        self,
        models_dir: Path,
        prediction_engine: PredictionEngine,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the ModelRegistry and its composing low-level helper components.

        Args:
            models_dir: The base directory where actual model files (.pkl) are stored.
            prediction_engine: An instance of PredictionEngine for hot-loading models.
            config: Optional. Configuration dictionary for the registry.
        """
        self.config = config or get_config()
        self.models_dir = models_dir  # Base directory for model .pkl files
        self.prediction_engine = (
            prediction_engine  # Shared dependency for deployment manager helper
        )

        # Initialize low-level helper components.
        # These helpers manipulate specific aspects (storage, files, deployment mechanism, etc.)
        self._registry_storage_manager = RegistryStorageManager(registry_dir=self.models_dir.parent)
        self._model_file_manager = ModelFileManager(models_base_dir=self.models_dir)
        self._deployment_manager = DeploymentManager(
            prediction_engine=self.prediction_engine, model_file_manager=self._model_file_manager
        )
        self._comparison_analyzer = ModelComparisonAnalyzer()
        self._performance_tracker = PerformanceTracker(
            ema_alpha=self.config.get("model_registry", {}).get("performance_ema_alpha", 0.1)
        )
        self._traffic_router = TrafficRouter()
        self._model_archiver = ModelArchiver()
        self._model_exporter = ModelExporter(
            models_base_dir=self.models_dir
        )  # Explicitly initialize exporter

        # Registry state: Core data of the ModelRegistry
        self.versions: Dict[str, List[ModelVersion]] = defaultdict(
            list
        )  # model_id -> List[ModelVersion]
        self.active_deployments: Dict[str, ModelVersion] = (
            {}
        )  # model_id -> ModelVersion (actively serving traffic)

        self._load_registry_initial()  # Load initial state on startup

        logger.info(
            f"ModelRegistry initialized. Loaded {sum(len(v) for v in self.versions.values())} model versions from disk."
        )

    def _load_registry_initial(self):
        """
        Loads the registry state using RegistryStorageManager.
        Also triggers initial loading of active models into PredictionEngine.
        This runs at startup to restore the previous state.
        """
        self.versions, self.active_deployments = self._registry_storage_manager.load_registry()

        # After loading, ensure PredictionEngine is aware of all models with active traffic
        for model_id, version_obj in self.active_deployments.items():
            if version_obj.deployment_pct > 0:
                try:
                    # Delegate actual model loading to deployment manager's helper (which uses ModelFileManager)
                    # We call deploy_model_version to ensure it's loaded into PredictionEngine
                    # with correct traffic, without changing its status in the registry.
                    # This is for *hot-loading on startup*.
                    # The `deploy_model_version` method expects a traffic_pct.
                    # We pass the currently defined deployment_pct for this active model.
                    # This is a bit indirect, but leverages the existing prediction_engine interaction logic.
                    # Alternatively, could directly call prediction_engine.load_model here.

                    # Simpler approach: call prediction_engine.load_model directly.
                    # The deployment_manager's method involves status updates which we don't want on load.
                    model_obj, features, _ = self._model_file_manager.load_model(
                        model_id, version_obj.version
                    )
                    self.prediction_engine.load_model(
                        model_id=model_id,
                        model_obj=model_obj,
                        version=version_obj.version,
                        traffic_pct=version_obj.deployment_pct,
                        metadata=version_obj.to_dict(),  # Pass full ModelVersion metadata
                    )
                    logger.debug(
                        f"Hot-loaded active model {model_id} v{version_obj.version} ({version_obj.deployment_pct:.1f}%) into PredictionEngine on startup."
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to hot-load active model {model_id} v{version_obj.version} on startup: {e}. Marking as failed.",
                        exc_info=True,
                    )
                    # If model fails to load, mark it as failed and remove traffic from registry state
                    version_obj.status = "failed"
                    version_obj.deployment_pct = 0.0
                    self.active_deployments.pop(model_id, None)  # Remove from active list
                    self._save_registry_state()  # Persist this failure state

    def _save_registry_state(self):
        """Saves the current registry state to disk using RegistryStorageManager."""
        self._registry_storage_manager.save_registry(self.versions, self.active_deployments)

    def register_model(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        training_data_range: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        features: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Registers a new model version in the registry.
        Saves the model binary to disk and updates registry metadata.

        Args:
            model: Trained model object.
            model_id: Unique model identifier.
            model_type: Type of model (e.g., 'xgboost', 'ensemble').
            training_data_range: String describing the training data date range.
            hyperparameters: Model hyperparameters.
            metrics: Model performance metrics.
            features: List of feature names.
            metadata: Additional metadata.

        Returns:
            The newly created ModelVersion object.

        Raises:
            IOError: If the model file cannot be saved.
        """
        # Generate version number (simple increment for now)
        existing_versions = self.versions.get(model_id, [])
        version_num = len(existing_versions) + 1
        version_str = f"v{version_num}"

        # Check for existing version to prevent overwrites or duplicates during re-registration attempt
        if self.get_model_version(model_id, version_str):
            logger.warning(
                f"Model {model_id} version {version_str} already registered. Consider updating or generating a new version."
            )
            # Depending on policy, might return existing or raise error. For now, log and return existing.
            return self.get_model_version(model_id, version_str)

        # Create ModelVersion object (initially without path)
        model_version = ModelVersion(
            model_id=model_id,
            version=version_str,
            model_type=model_type,
            created_at=datetime.now(timezone.utc),
            trained_on=training_data_range,
            features=features,
            hyperparameters=hyperparameters,
            metrics=metrics,
            status="candidate",  # New models start as candidate
            deployment_pct=0.0,  # No traffic initially
            metadata=metadata or {},
        )

        # Save model object to disk using ModelFileManager. This updates model_version.model_file_path internally.
        try:
            model_version.model_file_path = self._model_file_manager.save_model(
                model, model_version
            )
        except IOError as e:
            logger.error(
                f"Failed to save model file for {model_id} v{version_str}. Model will not be registered.",
                exc_info=True,
            )
            model_version.status = "failed"  # Mark registration as failed due to file save failure
            raise e  # Re-raise to signal registration failure

        # Add to in-memory registry and persist
        self.versions[model_id].append(model_version)
        self.versions[model_id].sort(key=lambda v: v.created_at)  # Keep sorted by creation time
        self._save_registry_state()

        logger.info(f"Registered model: {model_id} {version_str}. Metrics: {metrics}")
        return model_version

    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Retrieves a specific model version by ID and version string."""
        for v in self.versions.get(model_id, []):
            if v.version == version:
                return v
        return None

    def get_latest_version(
        self, model_id: str, status: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Retrieves the latest version of a model, optionally filtered by status.
        'Latest' is determined by creation timestamp.
        """
        versions = sorted(self.versions.get(model_id, []), key=lambda v: v.created_at, reverse=True)
        if status:
            versions = [v for v in versions if v.status == status]
        return versions[0] if versions else None

    def list_models(
        self, model_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lists registered models, optionally filtered by model ID or status.

        Args:
            model_id: Optional. Filter by specific model identifier.
            status: Optional. Filter by deployment status ('candidate', 'production', 'archived', 'failed').

        Returns:
            A list of dictionaries, each representing a model version with key metadata.
        """
        models_list = []
        # Filter versions based on model_id first
        target_versions = (
            self.versions.get(model_id, [])
            if model_id
            else [v for sublist in self.versions.values() for v in sublist]
        )

        for version_obj in target_versions:
            if status is None or version_obj.status == status:
                models_list.append(version_obj.to_dict())  # Return as dict for easy consumption

        # Sort by creation date for consistent ordering (latest first)
        models_list.sort(key=lambda x: x["created_at"], reverse=True)

        logger.debug(f"Listed {len(models_list)} models (filter: id={model_id}, status={status}).")
        return models_list

    # --- Higher-level MLOps operations are now in ModelManagementService and ModelAnalyticsService ---
    # The ModelRegistry itself no longer directly contains deploy_model, compare_models etc.

# File: src/ai_trader/models/inference/model_analytics_service.py

"""
Service for analyzing and reporting on model registry data.

Provides methods for model comparison, lineage tracking, and retrieving
deployment configurations, using data from the ModelRegistry.
"""

# Standard library imports
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Local imports
from main.models.inference.model_registry_helpers.model_comparison_analyzer import (
    ModelComparisonAnalyzer,
)
from main.models.inference.model_registry_helpers.traffic_router import (  # For routing info in deployment config
    TrafficRouter,
)

# Corrected absolute imports
from main.models.inference.model_registry_types import ModelVersion

# Type hinting the ModelRegistry to avoid circular imports at runtime
if TYPE_CHECKING:
    # Local imports
    from main.models.inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelAnalyticsService:
    """
    Provides analytical and reporting capabilities for models within the registry.
    This includes comparing model versions, tracking lineage, and querying
    current deployment configurations.
    """

    def __init__(self, model_registry: "ModelRegistry"):
        """
        Initializes the ModelAnalyticsService.

        Args:
            model_registry: The ModelRegistry instance which this service operates on.
        """
        self.model_registry = model_registry  # The lean ModelRegistry instance

        # Helpers from ModelRegistry are accessed via registry
        self._comparison_analyzer = model_registry._comparison_analyzer
        self._traffic_router = model_registry._traffic_router

        logger.info("ModelAnalyticsService initialized.")

    def get_deployment_config(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieves the current deployment configuration for a specific model.
        Shows which versions are deployed and their traffic percentages.

        Args:
            model_id: The identifier of the model.

        Returns:
            A dictionary containing deployment details for the model.
        """
        config = {"model_id": model_id, "deployed_versions": [], "total_deployment_pct": 0.0}

        # Access versions directly from the registry
        for version in self.model_registry.versions.get(model_id, []):
            if version.deployment_pct > 0:  # Only include actively deployed versions
                config["deployed_versions"].append(
                    {
                        "version": version.version,
                        "status": version.status,
                        "deployment_pct": version.deployment_pct,
                        "created_at": version.created_at.isoformat(),
                        "metrics": version.metrics,  # Include metrics for quick overview
                    }
                )
                config["total_deployment_pct"] += version.deployment_pct

        return config

    def compare_model_versions(self, model_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compares two model versions, highlighting differences in metrics, features, and hyperparameters.
        Delegates comparison logic to ModelComparisonAnalyzer.

        Args:
            model_id: The identifier of the model.
            version1: The first version string to compare.
            version2: The second version string to compare.

        Returns:
            A dictionary detailing the differences between the two versions.
        """
        v1_obj = self.model_registry.get_model_version(model_id, version1)
        v2_obj = self.model_registry.get_model_version(model_id, version2)

        if not v1_obj or not v2_obj:
            logger.error(
                f"One or both versions not found for model {model_id}: v1='{version1}', v2='{version2}'. Cannot compare."
            )
            return {"error": "Model version(s) not found"}

        return self._comparison_analyzer.compare_model_versions(v1_obj, v2_obj)

    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the chronological lineage of all versions for a given model.
        Delegates lineage generation to ModelComparisonAnalyzer.

        Args:
            model_id: The identifier of the model.

        Returns:
            A list of dictionaries, each representing a version in chronological order.
        """
        versions_list = self.model_registry.versions.get(model_id, [])
        if not versions_list:
            logger.warning(f"No versions found for model {model_id}. Cannot generate lineage.")
            return []

        return self._comparison_analyzer.get_model_lineage(versions_list)

    def get_model_for_prediction(self, model_id: str) -> Optional[ModelVersion]:
        """
        Selects and returns the ModelVersion object to use for prediction,
        handling A/B testing and gradual rollout logic.
        Delegates routing decision to TrafficRouter.

        Args:
            model_id: Model identifier.

        Returns:
            The selected ModelVersion object, or None if no suitable model is found.
        """
        # Pass necessary data to the TrafficRouter from the registry's state
        return self._traffic_router.get_model_version_for_prediction(
            model_id=model_id,
            active_deployments=self.model_registry.active_deployments,
            all_versions_for_model=self.model_registry.versions.get(model_id, []),
        )

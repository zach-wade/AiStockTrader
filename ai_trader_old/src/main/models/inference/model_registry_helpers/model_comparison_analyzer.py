# File: src/ai_trader/models/inference/model_registry_helpers/model_comparison_analyzer.py

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party imports
import numpy as np
import pandas as pd  # For pd.isna, pd.notna

# Local imports
# Corrected absolute import for ModelVersion
from main.models.inference.model_registry_types import ModelVersion

logger = logging.getLogger(__name__)


class ModelComparisonAnalyzer:
    """
    Provides methods for comparing different versions of machine learning models.
    Analyzes differences in metrics, features, and hyperparameters.
    """

    def __init__(self):
        logger.debug("ModelComparisonAnalyzer initialized.")

    def compare_model_versions(
        self, version1_obj: ModelVersion, version2_obj: ModelVersion
    ) -> Dict[str, Any]:
        """
        Compares two model versions (ModelVersion objects), highlighting differences
        in metrics, features, and hyperparameters.

        Args:
            version1_obj: The first ModelVersion object.
            version2_obj: The second ModelVersion object.

        Returns:
            A dictionary detailing the differences between the two versions.
        """
        if version1_obj.model_id != version2_obj.model_id:
            logger.warning(
                f"Comparing different model_ids: {version1_obj.model_id} vs {version2_obj.model_id}. Comparison might be less meaningful."
            )

        comparison = {
            "model_id": version1_obj.model_id,
            "version1": version1_obj.version,
            "version2": version2_obj.version,
            "versions_info": {
                version1_obj.version: version1_obj.to_dict(),
                version2_obj.version: version2_obj.to_dict(),
            },
            "metrics_diff": {},
            "features_diff": {
                "added_in_v2": sorted(
                    list(set(version2_obj.features) - set(version1_obj.features))
                ),
                "removed_in_v2": sorted(
                    list(set(version1_obj.features) - set(version2_obj.features))
                ),
            },
            "hyperparameters_diff": {},
        }

        # --- Compare Metrics ---
        all_metrics_keys = set(version1_obj.metrics.keys()) | set(version2_obj.metrics.keys())
        for metric in all_metrics_keys:
            v1_val = version1_obj.metrics.get(metric)  # Can be None
            v2_val = version2_obj.metrics.get(metric)  # Can be None

            # Handle NaN/None consistently for comparison
            v1_val_num = (
                float(v1_val) if (isinstance(v1_val, (int, float)) and pd.notna(v1_val)) else np.nan
            )
            v2_val_num = (
                float(v2_val) if (isinstance(v2_val, (int, float)) and pd.notna(v2_val)) else np.nan
            )

            if pd.isna(v1_val_num) and pd.isna(v2_val_num):
                continue  # Both missing/NaN, no difference

            # Check for exact equality first for numerical stability
            if np.isclose(
                v1_val_num, v2_val_num, equal_nan=True
            ):  # equal_nan handles np.nan == np.nan
                continue

            change_pct = np.nan
            if pd.notna(v1_val_num) and v1_val_num != 0:
                change_pct = ((v2_val_num - v1_val_num) / v1_val_num) * 100
            elif pd.notna(v2_val_num) and (
                pd.isna(v1_val_num) or v1_val_num == 0
            ):  # From missing/zero to non-zero
                change_pct = float("inf") if v2_val_num > 0 else float("-inf")

            comparison["metrics_diff"][metric] = {
                "v1_value": float(v1_val_num) if pd.notna(v1_val_num) else None,
                "v2_value": float(v2_val_num) if pd.notna(v2_val_num) else None,
                "change_pct": float(change_pct) if pd.notna(change_pct) else None,
            }

        # --- Compare Hyperparameters ---
        all_hparam_keys = set(version1_obj.hyperparameters.keys()) | set(
            version2_obj.hyperparameters.keys()
        )
        for param in all_hparam_keys:
            v1_val = version1_obj.hyperparameters.get(param)
            v2_val = version2_obj.hyperparameters.get(param)

            if v1_val != v2_val:
                comparison["hyperparameters_diff"][param] = {"v1_value": v1_val, "v2_value": v2_val}

        logger.debug(
            f"Compared model '{version1_obj.model_id}' versions '{version1_obj.version}' and '{version2_obj.version}'."
        )
        return comparison

    def get_model_lineage(self, versions_list: List[ModelVersion]) -> List[Dict[str, Any]]:
        """
        Retrieves the chronological lineage of a list of model versions.

        Args:
            versions_list: A list of ModelVersion objects for a specific model_id.

        Returns:
            A list of dictionaries, each representing a version in chronological order,
            with key information for lineage tracking.
        """
        lineage = []
        # Sort all versions by creation date to establish lineage
        for version in sorted(versions_list, key=lambda v: v.created_at):
            lineage.append(
                {
                    "version": version.version,
                    "created_at": version.created_at.isoformat(),
                    "status": version.status,
                    "deployment_pct": version.deployment_pct,
                    "trained_on_data_range": version.trained_on,
                    "metrics": version.metrics,
                    "features_count": len(version.features),
                    "hyperparameters_count": len(version.hyperparameters),
                    # 'deployment_history': [] # This would require a separate log/history tracker for deployments
                }
            )

        logger.debug(f"Generated lineage for {len(lineage)} versions.")
        return lineage

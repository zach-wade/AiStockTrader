# File: src/ai_trader/models/monitoring/monitor_helpers/ab_test_analyzer.py

# Standard library imports
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np

# Use TYPE_CHECKING for circular dependency if needed for ModelRegistry/ModelVersion
if TYPE_CHECKING:
    # Local imports
    from main.models.inference.model_registry import ModelRegistry
    from main.models.inference.model_registry_types import ModelVersion

# Import PerformanceCalculator to use its metric comparison logic
# Local imports
from main.models.monitoring.monitor_helpers.performance_calculator import PerformanceCalculator
from main.monitoring.alerts.alert_manager import AlertManager  # For sending alerts

logger = logging.getLogger(__name__)


class ABTestAnalyzer:
    """
    Analyzes A/B test results between deployed model versions and
    determines if a candidate model should be promoted to production.
    """

    def __init__(
        self,
        model_registry: "ModelRegistry",
        performance_calculator: PerformanceCalculator,
        alert_manager: AlertManager,
        config: Dict[str, Any],
    ):
        """
        Initializes the ABTestAnalyzer.

        Args:
            model_registry: An initialized ModelRegistry instance.
            performance_calculator: An initialized PerformanceCalculator instance.
            alert_manager: An initialized AlertManager instance.
            config: Application configuration for A/B test thresholds.
        """
        self.model_registry = model_registry
        self.performance_calculator = performance_calculator
        self.alert_manager = alert_manager
        self.config = config

        self.min_predictions_for_ab_test = (
            self.config.get("ml", {}).get("monitoring", {}).get("min_predictions_for_ab_test", 500)
        )
        self.ab_test_improvement_threshold = (
            self.config.get("ml", {})
            .get("monitoring", {})
            .get("ab_test_improvement_threshold_pct", 10.0)
        )  # 10% MAE improvement

        logger.debug("ABTestAnalyzer initialized.")

    async def check_ab_test_results(self):
        """
        Iterates through models with active A/B tests (candidate deployments)
        and compares their performance against the production model to promote winners.
        """
        logger.info("Checking A/B test results for active candidate models.")

        # Get all models that have active candidates (deployment_pct > 0 and status is 'candidate')
        models_with_candidates: Dict[str, List["ModelVersion"]] = defaultdict(list)
        for model_id, versions_list in self.model_registry.versions.items():
            for version in versions_list:
                if version.status == "candidate" and version.deployment_pct > 0:
                    models_with_candidates[model_id].append(version)

        for model_id, candidates in models_with_candidates.items():
            production_model = self.model_registry.get_latest_version(model_id, status="production")

            if not production_model:
                logger.warning(
                    f"No production model found for {model_id} while candidates are active. Skipping A/B test comparison."
                )
                continue

            for candidate in candidates:
                logger.debug(
                    f"Comparing candidate {model_id} v{candidate.version} against production v{production_model.version}."
                )

                # Check performance and decide on promotion
                if await self._compare_models_for_promotion(candidate, production_model):
                    # Promote candidate if it wins the A/B test
                    await self.model_registry.promote_model(model_id, candidate.version)
                    logger.info(
                        f"Promoted candidate {candidate.version} to production for model {model_id} (won A/B test)."
                    )
                    await self.alert_manager.send_alert(
                        level="info",
                        title=f"A/B Test Win & Model Promotion: {model_id}",
                        message=f"Candidate v{candidate.version} outperformed Production v{production_model.version} and was promoted to 100% traffic.",
                    )
                else:
                    logger.debug(
                        f"Candidate {model_id} v{candidate.version} did not outperform production v{production_model.version} sufficiently."
                    )

    async def _compare_models_for_promotion(
        self, candidate: "ModelVersion", production: "ModelVersion"
    ) -> bool:
        """
        Compares candidate and production models based on recent performance metrics
        to determine if the candidate should be promoted.
        """
        # Get latest performance metrics from ModelRegistry's PerformanceTracker (via the monitor)
        # This implies PerformanceTracker's `update_model_performance` should push to ModelRegistry's
        # metrics directly, which it does. So we can retrieve from ModelVersion objects.

        # Fetch the latest overall performance metrics calculated by ModelMonitor for each model
        # Assuming model_registry.get_model_version.metrics is kept updated by PerformanceTracker.
        candidate_metrics = candidate.metrics  # Access directly from ModelVersion object
        production_metrics = production.metrics  # Access directly from ModelVersion object

        # Need sufficient data points for a statistically significant comparison
        if (
            candidate_metrics.get("count", 0) < self.min_predictions_for_ab_test
            or production_metrics.get("count", 0) < self.min_predictions_for_ab_test
        ):
            logger.info(
                f"Insufficient prediction data for A/B test comparison: Candidate has {candidate_metrics.get('count',0)}/{self.min_predictions_for_ab_test}, Production has {production_metrics.get('count',0)}/{self.min_predictions_for_ab_test}."
            )
            return False

        # Compare MAE (Mean Absolute Error - lower is better)
        candidate_mae = candidate_metrics.get("mae", float("inf"))
        production_mae = production_metrics.get("mae", float("inf"))

        if not np.isfinite(candidate_mae) or not np.isfinite(production_mae) or production_mae <= 0:
            logger.warning(
                f"Invalid MAE values for A/B comparison. Candidate: {candidate_mae}, Production: {production_mae}. Skipping comparison."
            )
            return False

        # Candidate should be significantly better (e.g., >X% improvement in MAE)
        # Improvement: (Old MAE - New MAE) / Old MAE
        improvement_ratio = (production_mae - candidate_mae) / production_mae
        improvement_pct = improvement_ratio * 100

        is_winner = improvement_pct > self.ab_test_improvement_threshold

        logger.debug(
            f"A/B Test: Model {candidate.model_id} (v{candidate.version} MAE: {candidate_mae:.4f}) vs Production (v{production.version} MAE: {production_mae:.4f}). Improvement: {improvement_pct:.2f}%. Threshold: {self.ab_test_improvement_threshold:.2f}%. Candidate wins: {is_winner}."
        )

        return is_winner

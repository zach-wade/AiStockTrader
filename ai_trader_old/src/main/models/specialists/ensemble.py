"""
Contains the CatalystSpecialistEnsemble class.
This class orchestrates all specialist models to produce a final prediction.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import pandas as pd

# Import other components from the package using relative imports
# Avoid circular imports by importing specific classes directly
from .base import CatalystPrediction
from .earnings import EarningsSpecialist
from .news import NewsSpecialist
from .options import OptionsSpecialist
from .social import SocialSpecialist
from .technical import TechnicalSpecialist


# Define EnsemblePrediction locally to avoid circular import
@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""

    final_probability: float
    final_confidence: float
    individual_predictions: List[CatalystPrediction]
    ensemble_metadata: Dict[str, Any] = None


# Local imports
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


# A mapping to easily instantiate specialists from the config file
# This allows us to add new specialists without changing this file's code.
SPECIALIST_CLASS_MAP = {
    "earnings": EarningsSpecialist,
    "social": SocialSpecialist,
    "technical": TechnicalSpecialist,
    "news": NewsSpecialist,
    "options": OptionsSpecialist,
}


class CatalystSpecialistEnsemble:
    """
    V3 Catalyst Specialist Ensemble - The "Killer" System
    Orchestrates multiple specialist models to predict catalyst outcomes.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config if config is not None else get_config()
        self.ensemble_config = self.config["ensemble"]

        self.specialists: Dict[str, BaseCatalystSpecialist] = {
            name: globals()[spec_class](self.config)
            for name, spec_class in self.config["specialists_registry"].items()
        }

        # ## REFACTOR: Ensemble parameters loaded from config
        self.min_specialists_required = self.ensemble_config.get("min_specialists_required", 1)
        self.confidence_threshold = self.ensemble_config.get("confidence_threshold", 0.3)
        self.probability_threshold = self.ensemble_config.get("probability_threshold", 0.6)

        self._load_all_specialists()

    async def predict_catalyst_outcome(
        self, catalyst_features: Dict[str, Any]
    ) -> Optional[EnsemblePrediction]:
        """## REFACTOR: Use asyncio.gather for concurrent specialist predictions."""
        logger.debug("Generating ensemble catalyst prediction")

        try:
            # Step 1: Run all specialist predictions in parallel
            prediction_tasks = [s.predict(catalyst_features) for s in self.specialists.values()]
            results = await asyncio.gather(*prediction_tasks)

            # Step 2: Filter for valid predictions that meet the confidence threshold
            specialist_predictions = {
                pred.specialist_type: pred
                for pred in results
                if pred and pred.confidence >= self.confidence_threshold
            }

            if len(specialist_predictions) < self.min_specialists_required:
                logger.debug(f"Insufficient active specialists: {len(specialist_predictions)}")
                return None

            # Step 3: Calculate ensemble outcome
            ensemble_weights = self._calculate_ensemble_weights(specialist_predictions)
            ensemble_probability = self._calculate_weighted_probability(
                specialist_predictions, ensemble_weights
            )
            ensemble_confidence = self._calculate_ensemble_confidence(specialist_predictions)
            dominant_catalyst = self._find_dominant_catalyst(specialist_predictions)
            recommendation = self._generate_recommendation(
                ensemble_probability, ensemble_confidence
            )

            return EnsemblePrediction(
                ensemble_probability=ensemble_probability,
                ensemble_confidence=ensemble_confidence,
                participating_specialists=list(specialist_predictions.keys()),
                specialist_predictions=specialist_predictions,
                catalyst_types_detected=self._extract_catalyst_types(catalyst_features),
                dominant_catalyst=dominant_catalyst,
                ensemble_weights=ensemble_weights,
                final_recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}", exc_info=True)
            raise

    def _calculate_ensemble_weights(self, preds: Dict[str, CatalystPrediction]) -> Dict[str, float]:
        """Calculate dynamic weights based on catalyst strength and confidence."""
        raw_weights = {name: p.catalyst_strength * p.confidence for name, p in preds.items()}
        total_weight = sum(raw_weights.values())
        return (
            {name: w / total_weight for name, w in raw_weights.items()} if total_weight > 0 else {}
        )

    def _calculate_weighted_probability(
        self, preds: Dict[str, CatalystPrediction], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted average of probabilities."""
        prob = sum(preds[name].probability * w for name, w in weights.items())
        return prob if weights else 0.5

    def _calculate_ensemble_confidence(self, preds: Dict[str, CatalystPrediction]) -> float:
        """Calculate a more nuanced ensemble confidence."""
        if not preds:
            return 0.0

        # Base confidence is the max confidence of any participating specialist
        base_confidence = max(p.confidence for p in preds.values())

        # Bonus for multiple specialists agreeing
        agreement_bonus = self.ensemble_config["confidence_bonus_per_specialist"] * (len(preds) - 1)

        final_confidence = min(base_confidence + agreement_bonus, 1.0)
        return max(final_confidence, 0.0)

    def _find_dominant_catalyst(self, preds: Dict[str, CatalystPrediction]) -> str:
        """Find the dominant catalyst based on strength and confidence."""
        if not preds:
            return "none"
        return max(preds.keys(), key=lambda k: preds[k].catalyst_strength * preds[k].confidence)

    def _generate_recommendation(self, probability: float, confidence: float) -> str:
        """Generate final recommendation based on configurable thresholds."""
        buy_prob = self.ensemble_config["recommendation_thresholds"]["buy"]["probability"]
        buy_conf = self.ensemble_config["recommendation_thresholds"]["buy"]["confidence"]
        pass_prob = self.ensemble_config["recommendation_thresholds"]["pass"]["probability"]

        if probability >= buy_prob and confidence >= buy_conf:
            return "BUY"
        if probability < pass_prob:
            return "PASS"
        return "HOLD"

    def _extract_catalyst_types(self, features: Dict[str, Any]) -> List[str]:
        """Extract list of all catalyst types present in the features."""
        return [
            name for name in self.specialists.keys() if features.get(f"has_{name}_catalyst", False)
        ]

    def train_all_specialists(self, training_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train all registered specialists."""
        logger.info(f"Starting training for all specialists on {len(training_data)} samples")
        results = {name: spec.train(training_data) for name, spec in self.specialists.items()}
        self._save_training_summary(results)
        return results

    def _load_all_specialists(self):
        """Load all trained specialist models."""
        for name, spec in self.specialists.items():
            spec.load_model()

    def _save_training_summary(self, training_results: Dict[str, Dict[str, Any]]):
        """Save training summary to disk."""
        summary_file = Path(self.config["models"]["save_directory"]) / "training_summary.json"
        summary_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "specialist_results": training_results,
            "ensemble_config": self.ensemble_config,
        }
        try:
            with summary_file.open("w") as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"Training summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving training summary: {e}")

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current status of the ensemble and its specialists."""
        spec_status = {
            name: {"is_trained": s.is_trained, "model_version": s.model_version}
            for name, s in self.specialists.items()
        }
        return {
            "ensemble_ready": all(s["is_trained"] for s in spec_status.values()),
            "specialists": spec_status,
        }

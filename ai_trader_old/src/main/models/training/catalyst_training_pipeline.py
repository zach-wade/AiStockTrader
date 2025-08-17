"""
V3 Catalyst Training Pipeline - Final Production Version

This pipeline orchestrates the entire training, validation, and reporting process
for the CatalystSpecialistEnsemble models, adhering to all established
architectural principles of the project.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.historical.catalyst_generator import HistoricalCatalystGenerator

# Project imports
from main.models.specialists.catalyst_specialists import CatalystSpecialistEnsemble

from .cross_validation import TimeSeriesCV

# REFACTOR: Import the new, centralized results objects
from .pipeline_results import PipelineResults, ResultsPrinter

logger = logging.getLogger(__name__)


@dataclass
class CatalystTrainingConfig:
    """Configuration for the catalyst specialist training pipeline."""

    min_training_samples: int = 1000
    cv_folds: int = 5
    feature_columns: List[str] = field(default_factory=list)


class CatalystTrainingPipeline:
    """
    Orchestrates the end-to-end training and evaluation process for V3
    catalyst specialist models.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.global_config = config if config is not None else get_config()
        self.training_config = CatalystTrainingConfig(
            **self.global_config.get("ml", {}).get("catalyst_training", {})
        )
        self.catalyst_generator = HistoricalCatalystGenerator(self.global_config)
        self.ensemble = CatalystSpecialistEnsemble(self.global_config)
        self.cv_tool = TimeSeriesCV(config=self.global_config)
        self.output_dir = Path("models/training/catalyst_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_training(
        self, start_date: str, end_date: str, force_regenerate_data: bool = False
    ) -> PipelineResults:
        """Main training method: Executes the full V3 training pipeline."""
        run_id = f"catalyst_training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"====== Starting V3 Training Run: {run_id} ======")

        # REFACTOR: Use the new, centralized results object
        results = PipelineResults(
            run_id=run_id,
            start_time=datetime.now(timezone.utc),
            config_snapshot=self.training_config.__dict__,
        )

        try:
            # Step 1: Prepare Dataset
            dataset = await self._prepare_catalyst_dataset(
                start_date, end_date, force_regenerate_data
            )
            if dataset is None or len(dataset) < self.training_config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(dataset) if dataset is not None else 0} samples."
                )

            results.metrics["dataset_info"] = self._analyze_dataset(dataset)

            # Step 2: Prepare Features & Targets
            X, y = self._prepare_features_and_targets(dataset)

            # Step 3: Purged Cross-Validation
            cv_results = self._run_centralized_cv(X, y)
            results.metrics["cross_validation"] = cv_results

            # Step 4: Train Final Specialist Models
            individual_results = self.ensemble.train_all_specialists(dataset)
            results.model_summaries = individual_results

            # Step 5: Feature Importance Analysis
            feature_importance = self._analyze_feature_importance()
            results.metrics["feature_importance"] = feature_importance

            # Step 6: Final Performance Evaluation
            performance_summary = await self._evaluate_final_ensemble(dataset)
            results.metrics["final_evaluation"] = performance_summary

            results.update_status("completed")

        except Exception as e:
            logger.critical(f"❌ V3 Training Pipeline Failed: {e}", exc_info=True)
            results.add_error(str(e))
            raise
        finally:
            results.end_time = datetime.now(timezone.utc)
            # Save results regardless of outcome
            await self._save_results(results)
            ResultsPrinter.print_summary(results)

        return results

    async def _prepare_catalyst_dataset(
        self, start: str, end: str, force: bool
    ) -> Optional[pd.DataFrame]:
        dataset_file = Path(f"data/historical_catalysts/catalysts_{start}_to_{end}.parquet")
        if dataset_file.exists() and not force:
            logger.info(f"Loading existing catalyst dataset: {dataset_file}")
            return pd.read_parquet(dataset_file)

        logger.info("Generating new historical catalyst dataset...")
        stats = await self.catalyst_generator.generate_training_dataset(start, end)
        if "error" in stats:
            raise ValueError(stats["error"])
        return pd.read_parquet(dataset_file)

    def _analyze_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        successful_outcomes = ["successful_breakout", "modest_gain"]
        success_count = dataset["outcome_label"].isin(successful_outcomes).sum()
        return {
            "total_samples": len(dataset),
            "date_range": (dataset["date"].min().isoformat(), dataset["date"].max().isoformat()),
            "unique_symbols": dataset["symbol"].nunique(),
            "outcome_distribution": dataset["outcome_label"].value_counts().to_dict(),
            "success_rate": success_count / len(dataset) if len(dataset) > 0 else 0,
        }

    def _prepare_features_and_targets(
        self, dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        available_features = [
            col for col in self.training_config.feature_columns if col in dataset.columns
        ]
        X = dataset[available_features].copy().fillna(0)
        y = dataset["outcome_label"].isin(["successful_breakout", "modest_gain"]).astype(int)
        X.index = pd.to_datetime(dataset["date"])
        y.index = X.index
        return X, y

    def _run_centralized_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Uses the centralized TimeSeriesCV utility to perform robust validation."""
        validation_model = RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight="balanced"
        )
        results = self.cv_tool.validate_model(model=validation_model, X=X, y=y)
        return results

    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyzes and aggregates feature importance from all trained specialists."""
        all_importances = defaultdict(list)
        for specialist in self.ensemble.specialists.values():
            if specialist.is_trained and hasattr(specialist.model, "feature_importances_"):
                for feature, importance in zip(
                    specialist.feature_columns, specialist.model.feature_importances_
                ):
                    all_importances[feature].append(importance)

        mean_importances = {feature: np.mean(scores) for feature, scores in all_importances.items()}
        return dict(sorted(mean_importances.items(), key=lambda item: item[1], reverse=True))

    async def _evaluate_final_ensemble(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates the final trained ensemble on a sample of the data."""
        test_sample = dataset.sample(min(1000, len(dataset)), random_state=42)
        y_true, y_pred, y_prob = [], [], []

        # Convert to list of dictionaries for batch processing
        test_records = test_sample.to_dict("records")

        # Process predictions asynchronously in batch
        prediction_tasks = [
            self.ensemble.predict_catalyst_outcome(record) for record in test_records
        ]

        predictions = await asyncio.gather(*prediction_tasks)

        # Vectorized processing of results
        successful_outcomes = {"successful_breakout", "modest_gain"}

        for i, prediction in enumerate(predictions):
            if prediction:
                y_true.append(test_records[i]["outcome_label"] in successful_outcomes)
                y_pred.append(prediction.final_recommendation == "BUY")
                y_prob.append(prediction.ensemble_probability)

        if not y_true:
            return {"error": "No predictions made on test sample."}

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

    async def _save_results(self, results: PipelineResults):
        """Saves the comprehensive training results using the ResultsPrinter."""
        results_file_path = self.output_dir / f"{results.run_id}_results.json"
        ResultsPrinter.save_to_json(results, str(results_file_path))
        results.add_artifact("results_json", str(results_file_path))

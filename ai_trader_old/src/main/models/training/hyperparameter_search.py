"""
Bayesian hyperparameter optimization
Created: 2025-06-16
"""

"""
Automated hyperparameter tuning for ML models using Bayesian optimization.

This module uses Optuna to efficiently search the hyperparameter space
and find optimal model configurations without manual tuning.
"""

# Standard library imports
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import lightgbm as lgb
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Local imports
from main.config.config_manager import get_config
from main.feature_pipeline.feature_store_compat import FeatureStore
from main.models.training.cross_validation import TimeSeriesCV

logger = logging.getLogger(__name__)


class HyperparameterSearch:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the hyperparameter search utility.

        Args:
            config: The global application configuration object.
        """
        self.config = config if config is not None else get_config()

        # Clean initialization using the global config
        hyperopt_config = self.config.get("hyperopt", {})
        self.n_trials = hyperopt_config.get("n_trials", 100)
        self.timeout_hours = hyperopt_config.get("timeout_hours", 6)

        # Instantiate dependencies from config
        feature_store_path = self.config.get("paths", {}).get("feature_store", "data/features")
        self.feature_store = FeatureStore(feature_store_path, self.config)
        self.cv_tool = TimeSeriesCV(config=self.config)

        results_path = self.config.get("paths", {}).get("hyperopt_results", "results/hyperopt")
        self.results_dir = Path(results_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Search space definitions remain the same
        self.search_spaces = {
            "xgboost": self._get_xgboost_space(),
            "lightgbm": self._get_lightgbm_space(),
            "random_forest": self._get_rf_space(),
        }
        logger.info("HyperparameterSearch toolkit initialized.")

    def _get_xgboost_space(self) -> Dict[str, Any]:
        """Define XGBoost hyperparameter search space."""
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "gamma": ("float", 0, 5),
            "reg_alpha": ("float", 0, 10),
            "reg_lambda": ("float", 0, 10),
            "min_child_weight": ("int", 1, 10),
        }

    def _get_lightgbm_space(self) -> Dict[str, Any]:
        """Define LightGBM hyperparameter search space."""
        return {
            "n_estimators": ("int", 100, 1000),
            "num_leaves": ("int", 20, 300),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "feature_fraction": ("float", 0.6, 1.0),
            "bagging_fraction": ("float", 0.6, 1.0),
            "bagging_freq": ("int", 1, 10),
            "min_child_samples": ("int", 5, 100),
            "lambda_l1": ("float", 0, 10),
            "lambda_l2": ("float", 0, 10),
            "min_gain_to_split": ("float", 0, 1),
        }

    def _get_rf_space(self) -> Dict[str, Any]:
        """Define RandomForest hyperparameter search space."""
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 5, 50),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 20),
            "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": ("categorical", [True, False]),
            "class_weight": ("categorical", [None, "balanced", "balanced_subsample"]),
        }

    def _sample_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Sample hyperparameters for the given model type using Optuna trial."""
        if model_type not in self.search_spaces:
            raise ValueError(f"No search space defined for model type: {model_type}")

        space = self.search_spaces[model_type]
        params = {}

        for param_name, space_def in space.items():
            if space_def[0] == "int":
                params[param_name] = trial.suggest_int(param_name, space_def[1], space_def[2])
            elif space_def[0] == "float":
                if len(space_def) > 3 and space_def[3] == "log":
                    params[param_name] = trial.suggest_float(
                        param_name, space_def[1], space_def[2], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(param_name, space_def[1], space_def[2])
            elif space_def[0] == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, space_def[1])

        return params

    def _create_objective(
        self, model_type: str, X: pd.DataFrame, y: pd.Series, metric: str
    ) -> Callable:
        """Creates the objective function that Optuna will optimize."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters using the new, cleaner pattern
            params = self._sample_params(trial, model_type)

            cv_scores = []
            # REFACTOR: Using our centralized, financially-sound CV tool
            for fold, (train_idx, val_idx) in enumerate(self.cv_tool.purged_kfold_split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = self._create_model(model_type, params)

                fit_params = {}
                if model_type in ["xgboost", "lightgbm"]:
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["callbacks"] = [
                        optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
                    ]

                model.fit(X_train, y_train, **fit_params)

                predictions = model.predict(X_val)
                score = 0.0

                # REFACTORED: Logic to handle multiple metric types
                if metric == "sharpe":
                    returns = self._calculate_strategy_returns(predictions, X_val)
                    score = self._calculate_sharpe_ratio(returns)
                else:  # Default to f1-score for classification
                    score = f1_score(y_val, predictions, average="weighted", zero_division=0)

                cv_scores.append(score)
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        return objective

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        if model_type == "xgboost":
            return xgb.XGBClassifier(
                **params, n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric="logloss"
            )
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _calculate_strategy_returns(
        self, predictions: np.ndarray, features: pd.DataFrame
    ) -> pd.Series:
        """Calculate strategy returns from predictions."""
        if "returns" not in features.columns:
            return pd.Series(0, index=features.index)

        positions = pd.Series(predictions, index=features.index)
        strategy_returns = positions * features["returns"]
        return strategy_returns

    def _calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio from returns series."""
        if returns.std() == 0:
            return 0.0
        return np.sqrt(periods_per_year) * returns.mean() / returns.std()

    def run_study(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        study_name: Optional[str] = None,
        metric: str = "f1",
    ) -> optuna.Study:
        """
        Runs a full hyperparameter optimization study for a given model.
        """
        if model_type not in self.search_spaces:
            raise ValueError(f"No search space defined for model type: {model_type}")

        if study_name is None:
            study_name = f"{model_type}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            f"Starting Optuna study '{study_name}' for model '{model_type}' optimizing for '{metric}'"
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        )

        objective = self._create_objective(model_type, X_train, y_train, metric)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_hours * 3600,
            show_progress_bar=True,
        )

        logger.info(f"Study '{study.study_name}' completed.")
        logger.info(f"Best {metric}: {study.best_value:.4f}")
        logger.info(f"Best parameters found: {study.best_params}")

        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study, model_type: str, metric: str):
        """Save optimization results."""
        # Save study summary
        summary = {
            "model_type": model_type,
            "metric": metric,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "datetime": datetime.now().isoformat(),
        }

        summary_file = self.results_dir / f"{study.study_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save all trials
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "duration": trial.duration.total_seconds() if trial.duration else None,
            }
            trials_data.append(trial_data)

        trials_df = pd.DataFrame(trials_data)
        trials_file = self.results_dir / f"{study.study_name}_trials.csv"
        trials_df.to_csv(trials_file, index=False)

        # Save importance plot
        try:
            # Third-party imports
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(study)
            importance_file = self.results_dir / f"{study.study_name}_importance.png"
            fig.write_image(str(importance_file))
        except Exception as e:
            logger.warning(f"Could not save importance plot: {e}")

    def grid_search_timeframes(
        self,
        model_type: str,
        timeframes: List[str] = ["1h", "1d", "1w"],
        lookback_periods: List[int] = [20, 50, 100],
    ) -> Dict[str, Any]:
        """
        Search for optimal timeframe and lookback combinations.
        This workflow is preserved from your original file.
        """
        results = {}

        for timeframe in timeframes:
            for lookback in lookback_periods:
                logger.info(f"Testing {timeframe} with {lookback} period lookback")

                # Load data for this configuration
                # This assumes your FeatureStore has a method to get data like this
                features = self.feature_store.get_features(
                    symbols=["SPY"],  # Use a market proxy for this search
                    timeframe=timeframe,
                    lookback_periods=lookback,
                )

                if features is None or len(features) < lookback * 2:
                    logger.warning(f"Insufficient data for {timeframe}/{lookback}, skipping.")
                    continue

                # Prepare data (assuming 'target' column exists for training)
                X = features.drop(
                    ["target", "returns", "symbol", "timestamp"], axis=1, errors="ignore"
                )
                y = features["target"]

                # REFACTOR: Call the new, standardized run_study method
                study_name = f"{model_type}_{timeframe}_{lookback}"
                study = self.run_study(
                    model_type=model_type, X_train=X, y_train=y, study_name=study_name
                )

                results[f"{timeframe}_{lookback}"] = {
                    "timeframe": timeframe,
                    "lookback": lookback,
                    "best_params": study.best_params,
                    "best_score": study.best_value,
                    "n_trials": len(study.trials),
                }

        # Save comparison results
        comparison_df = pd.DataFrame.from_dict(results, orient="index")
        comparison_file = self.results_dir / f"{model_type}_timeframe_comparison.csv"
        comparison_df.to_csv(comparison_file)

        if not comparison_df.empty:
            best_config = comparison_df.loc[comparison_df["best_score"].idxmax()]
            logger.info(f"Best configuration: {best_config.name}")
            logger.info(f"Score: {best_config['best_score']:.4f}")

        return results

    def ensemble_optimization(
        self, model_types: List[str], X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize ensemble weights for multiple models.
        This workflow is preserved from your original file.
        """
        models = {}
        predictions = {}

        for model_type in model_types:
            logger.info(f"Optimizing {model_type} for ensemble...")

            # REFACTOR: Call the new, standardized run_study method
            study = self.run_study(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                metric="accuracy",  # Use a simple metric for base model selection
            )
            best_params = study.best_params

            model = self._create_model(model_type, best_params)
            model.fit(X_train, y_train)
            models[model_type] = model

            predictions[model_type] = model.predict_proba(X_train)[:, 1]

        # Optimize ensemble weights
        def ensemble_objective(trial):
            weights = [trial.suggest_float(name, 0, 1) for name in model_types]
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0

            normalized_weights = [w / total_weight for w in weights]

            ensemble_pred = np.sum(
                [w * p for w, p in zip(normalized_weights, predictions.values())], axis=0
            )
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            return f1_score(y_train, ensemble_binary, average="weighted", zero_division=0)

        ensemble_study = optuna.create_study(study_name="ensemble_weights", direction="maximize")
        ensemble_study.optimize(ensemble_objective, n_trials=100)

        best_weights_raw = [ensemble_study.best_params[name] for name in model_types]
        total_weight = sum(best_weights_raw)
        optimal_weights = (
            [w / total_weight for w in best_weights_raw]
            if total_weight > 0
            else [1.0 / len(model_types)] * len(model_types)
        )

        ensemble_config = {
            "models": {name: m.get_params() for name, m in models.items()},
            "weights": dict(zip(model_types, optimal_weights)),
            "best_score": ensemble_study.best_value,
        }

        logger.info(f"Optimal ensemble weights: {ensemble_config['weights']}")
        logger.info(f"Ensemble score: {ensemble_config['best_score']:.4f}")

        return ensemble_config

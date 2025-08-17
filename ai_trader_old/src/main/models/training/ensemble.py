"""
Advanced Ensemble Methods Toolkit for Offline Model Training.

This module provides a suite of scikit-learn compatible classes for creating
sophisticated model ensembles, including dynamic weighting (voting), stacking,
and Bayesian averaging. This is a core component of our MLOps and research
framework.
"""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

# Import our official, centralized cross-validation utility.
from .cross_validation import TimeSeriesCV

logger = logging.getLogger(__name__)


class DynamicEnsemble(BaseEstimator, ClassifierMixin):
    """
    Dynamic weighted voting ensemble that optimizes weights using cross-validation.
    """

    def __init__(self, base_models: Dict[str, Any], cv_tool: TimeSeriesCV, **kwargs):
        self.base_models = base_models
        self.cv_tool = cv_tool  # Injected CV tool for consistency.
        self.min_weight = kwargs.get("min_weight", 0.05)

        # Use scikit-learn standard for fitted attributes (trailing underscore)
        self.weights_ = {name: 1.0 / len(base_models) for name in base_models}
        self.fitted_models_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all base models and optimize initial weights."""
        logger.info(f"Fitting DynamicEnsemble with {len(self.base_models)} models.")

        # 1. Optimize initial weights using our robust TimeSeriesCV
        self._optimize_initial_weights(X, y)

        # 2. Fit all base models on the full training data
        for name, model in self.base_models.items():
            logger.info(f"Training base model on full data: {name}")
            self.fitted_models_[name] = clone(model).fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted average probability predictions."""
        # Initialize with zeros, matching the number of samples and 2 classes
        ensemble_proba = np.zeros((X.shape[0], 2), dtype=float)

        for name, model in self.fitted_models_.items():
            proba = (
                model.predict_proba(X)
                if hasattr(model, "predict_proba")
                else self._to_proba(model.predict(X))
            )
            ensemble_proba += self.weights_[name] * proba

        return ensemble_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the weighted probabilities."""
        predictions = self.predict_proba(X)
        return (predictions[:, 1] > 0.5).astype(int)

    def _optimize_initial_weights(self, X: pd.DataFrame, y: pd.Series):
        """Finds optimal starting weights using out-of-sample CV predictions."""
        logger.info("Optimizing initial ensemble weights via cross-validation...")

        oos_predictions = {}
        y_true_oos = np.array([])

        is_first_iter = True
        for name, model in self.base_models.items():
            y_pred_oos_model = np.array([])

            # The target `y` is the same for each fold, so we only need to build it once.
            if is_first_iter:
                y_true_oos_model = np.array([])

            for train_idx, val_idx in self.cv_tool.purged_kfold_split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_model = clone(model).fit(X_train, y_train)
                proba = (
                    fold_model.predict_proba(X_val)
                    if hasattr(fold_model, "predict_proba")
                    else self._to_proba(fold_model.predict(X_val))
                )
                y_pred_oos_model = np.concatenate([y_pred_oos_model, proba[:, 1]])

                if is_first_iter:
                    y_true_oos_model = np.concatenate([y_true_oos_model, y_val])

            oos_predictions[name] = y_pred_oos_model
            is_first_iter = False

        y_true_oos = y_true_oos_model

        def objective(weights):
            # Objective is to minimize log loss on out-of-fold predictions
            ensemble_pred = np.sum(
                [w * p for w, p in zip(weights, oos_predictions.values())], axis=0
            )
            return log_loss(y_true_oos, ensemble_pred)

        model_names = list(self.base_models.keys())
        initial_weights = np.ones(len(model_names)) / len(model_names)
        bounds = [(self.min_weight, 1.0) for _ in model_names]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            self.weights_ = dict(zip(model_names, result.x))
            logger.info(
                f"Optimized initial weights: { {k: f'{v:.2%}' for k, v in self.weights_.items()} }"
            )
        else:
            logger.warning("Weight optimization failed. Using equal weights.")

    def _to_proba(self, preds: np.ndarray) -> np.ndarray:
        """Converts binary [0,1] predictions to a shape (n_samples, 2) probability array."""
        proba = np.zeros((len(preds), 2))
        proba[:, 1] = preds
        proba[:, 0] = 1 - preds
        return proba


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble that trains a meta-learner on the predictions of base models.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        cv_tool: TimeSeriesCV,
        meta_learner: Optional[Any] = None,
        **kwargs,
    ):
        self.base_models = base_models
        self.cv_tool = cv_tool
        self.meta_learner = meta_learner or LogisticRegression()
        self.use_probabilities = kwargs.get("use_probabilities", True)
        self.fitted_models_ = {}
        self.fitted_meta_learner_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the stacking ensemble."""
        logger.info("Training StackingEnsemble...")

        meta_features_list, meta_y_list = [], []

        for train_idx, val_idx in self.cv_tool.purged_kfold_split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_meta_features = self._get_meta_features_for_fit(X_val, X_train, y_train)
            meta_features_list.append(fold_meta_features)
            meta_y_list.append(y_val)

        meta_X = np.concatenate(meta_features_list)
        meta_y = np.concatenate(meta_y_list)

        logger.info(f"Training meta-learner on dataset of shape {meta_X.shape}")
        self.fitted_meta_learner_ = clone(self.meta_learner).fit(meta_X, meta_y)

        logger.info("Training base models on full dataset...")
        for name, model in self.base_models.items():
            self.fitted_models_[name] = clone(model).fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._get_meta_features_for_predict(X)
        return self.fitted_meta_learner_.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._get_meta_features_for_predict(X)
        return self.fitted_meta_learner_.predict(meta_features)

    def _get_meta_features_for_fit(self, X, X_train, y_train):
        predictions = []
        for model in self.base_models.values():
            model_to_predict = clone(model).fit(X_train, y_train)
            if self.use_probabilities and hasattr(model_to_predict, "predict_proba"):
                pred = model_to_predict.predict_proba(X)[:, 1]
            else:
                pred = model_to_predict.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)

    def _get_meta_features_for_predict(self, X):
        predictions = []
        for model in self.fitted_models_.values():
            if self.use_probabilities and hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)


class BayesianEnsemble(BaseEstimator, ClassifierMixin):
    """
    Bayesian model averaging ensemble.
    """

    def __init__(self, base_models: Dict[str, Any], **kwargs):
        self.base_models = base_models
        self.update_rate = kwargs.get("update_rate", 0.1)
        prior_weights = kwargs.get("prior_weights")
        n_models = len(base_models)
        self.weights_ = (
            prior_weights if prior_weights else {name: 1.0 / n_models for name in base_models}
        )
        self.fitted_models_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for name, model in self.base_models.items():
            self.fitted_models_[name] = clone(model).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = {name: model.predict_proba(X) for name, model in self.fitted_models_.items()}
        ensemble_proba = np.sum(
            [self.weights_[name] * proba for name, proba in predictions.items()], axis=0
        )
        return ensemble_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class EnsembleBuilder:
    """
    Helper class to build different types of ensembles based on configuration.
    This class is preserved from your original code.
    """

    def __init__(self, config: Dict, cv_tool: TimeSeriesCV):
        self.config = config.get("ensemble", {})
        self.cv_tool = cv_tool
        self.base_models = self._create_base_models()

    def _create_base_models(self) -> Dict[str, Any]:
        """Creates a pool of base models from configuration."""
        models = {
            "xgboost": xgb.XGBClassifier(random_state=42, **self.config.get("xgboost_params", {})),
            "lightgbm": lgb.LGBMClassifier(
                random_state=42, verbose=-1, **self.config.get("lightgbm_params", {})
            ),
            "random_forest": RandomForestClassifier(
                random_state=42, **self.config.get("rf_params", {})
            ),
        }
        return models

    def build(self) -> Union[DynamicEnsemble, StackingEnsemble, BayesianEnsemble]:
        """Builds and returns the configured ensemble model."""
        ensemble_type = self.config.get("type", "dynamic")
        logger.info(f"Building ensemble of type: '{ensemble_type}'")

        if ensemble_type == "dynamic":
            return DynamicEnsemble(self.base_models, self.cv_tool, **self.config)
        elif ensemble_type == "stacking":
            meta_learner = LogisticRegression()  # Could also be configured
            return StackingEnsemble(self.base_models, self.cv_tool, meta_learner, **self.config)
        elif ensemble_type == "bayesian":
            return BayesianEnsemble(self.base_models, **self.config)
        else:
            raise ValueError(f"Unknown ensemble type in config: {ensemble_type}")

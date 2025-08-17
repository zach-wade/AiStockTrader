"""
Walk-forward cross-validation
Created: 2025-06-16
"""

"""
Time series cross-validation for trading models.

Implements proper walk-forward analysis and prevents look-ahead bias
in model validation.
"""

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

# Local imports
from main.backtesting.analysis.performance_metrics import PerformanceAnalyzer
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Walk-forward cross-validation for time series data."""

    def __init__(self, config: Any = None):
        """Initialize cross-validation."""
        if config is None:
            config = get_config()
        self.config = config

        # CV parameters
        self.n_splits = config.get("cv.n_splits", 5)
        self.test_size = config.get("cv.test_size_days", 60)
        self.train_size = config.get("cv.train_size_days", 252)
        self.gap_size = config.get("cv.gap_days", 1)  # Prevent look-ahead

        # Purging parameters
        self.purge_days = config.get("cv.purge_days", 2)
        self.embargo_days = config.get("cv.embargo_days", 2)

        # Performance analyzer for trading metrics
        self.performance_analyzer = PerformanceAnalyzer()

    def split_by_date(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Create train/test splits based on dates.

        Args:
            X: Features dataframe with DatetimeIndex
            y: Target series with DatetimeIndex
            start_date: Start date for CV
            end_date: End date for CV

        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")

        # Filter date range
        if start_date:
            mask = X.index >= start_date
            X = X[mask]
            y = y[mask]

        if end_date:
            mask = X.index <= end_date
            X = X[mask]
            y = y[mask]

        # Calculate split dates
        splits = []
        total_days = (X.index[-1] - X.index[0]).days

        for i in range(self.n_splits):
            # Calculate train start and end
            train_start = X.index[0] + timedelta(days=i * self.test_size)
            train_end = train_start + timedelta(days=self.train_size)

            # Calculate test start and end with gap
            test_start = train_end + timedelta(days=self.gap_size + self.purge_days)
            test_end = test_start + timedelta(days=self.test_size)

            # Check if we have enough data
            if test_end > X.index[-1]:
                break

            # Create train/test sets
            train_mask = (X.index >= train_start) & (X.index < train_end)
            test_mask = (X.index >= test_start) & (X.index < test_end)

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Apply embargo (remove samples too close to test set)
            if self.embargo_days > 0:
                embargo_start = test_start - timedelta(days=self.embargo_days)
                embargo_mask = X_train.index < embargo_start
                X_train = X_train[embargo_mask]
                y_train = y_train[embargo_mask]

            if len(X_train) > 0 and len(X_test) > 0:
                splits.append((X_train, y_train, X_test, y_test))

                logger.info(
                    f"Split {i+1}: Train {train_start.date()} to {train_end.date()}, "
                    f"Test {test_start.date()} to {test_end.date()}"
                )
                logger.info(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        return splits

    def expanding_window_split(
        self, X: pd.DataFrame, y: pd.Series, min_train_size: int = None
    ) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Expanding window cross-validation.

        Training set grows over time while test set moves forward.
        """
        if min_train_size is None:
            min_train_size = self.train_size

        splits = []
        total_days = (X.index[-1] - X.index[0]).days

        # Start with minimum training size
        train_start = X.index[0]
        train_end = train_start + timedelta(days=min_train_size)

        split_num = 0
        while True:
            # Test period
            test_start = train_end + timedelta(days=self.gap_size + self.purge_days)
            test_end = test_start + timedelta(days=self.test_size)

            if test_end > X.index[-1]:
                break

            # Create splits
            train_mask = (X.index >= train_start) & (X.index < train_end)
            test_mask = (X.index >= test_start) & (X.index < test_end)

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            if len(X_train) > 0 and len(X_test) > 0:
                splits.append((X_train, y_train, X_test, y_test))

                logger.info(
                    f"Expanding split {split_num+1}: "
                    f"Train size: {len(X_train)}, Test size: {len(X_test)}"
                )

            # Expand training window
            train_end = test_end
            split_num += 1

        return splits

    def purged_kfold_split(
        self, X: pd.DataFrame, y: pd.Series, n_folds: int = None
    ) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        K-fold cross-validation with purging for time series.

        Ensures no data leakage between folds.
        """
        if n_folds is None:
            n_folds = self.n_splits

        # Sort by time
        sorted_indices = X.index.argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]

        # Calculate fold size
        n_samples = len(X)
        fold_size = n_samples // n_folds

        splits = []

        for i in range(n_folds):
            # Test fold indices
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < n_folds - 1 else n_samples

            # Training indices (all except test fold and purged samples)
            train_indices = []

            # Add samples before test fold (with purging)
            if test_start_idx > self.purge_days:
                train_indices.extend(range(0, test_start_idx - self.purge_days))

            # Add samples after test fold (with purging)
            if test_end_idx + self.purge_days < n_samples:
                train_indices.extend(range(test_end_idx + self.purge_days, n_samples))

            if len(train_indices) == 0:
                continue

            # Create splits
            X_train = X_sorted.iloc[train_indices]
            y_train = y_sorted.iloc[train_indices]
            X_test = X_sorted.iloc[test_start_idx:test_end_idx]
            y_test = y_sorted.iloc[test_start_idx:test_end_idx]

            splits.append((X_train, y_train, X_test, y_test))

            logger.info(
                f"Purged fold {i+1}: Train size: {len(X_train)}, " f"Test size: {len(X_test)}"
            )

        return splits

    def combinatorial_cv(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = None, n_test_groups: int = 2
    ) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Combinatorial purged cross-validation.

        More robust version that tests multiple combinations of data.
        """
        if n_splits is None:
            n_splits = self.n_splits

        # Create base groups
        n_samples = len(X)
        group_size = n_samples // (n_splits * n_test_groups)

        # Generate all possible combinations
        # Standard library imports
        from itertools import combinations

        groups = list(range(n_splits * n_test_groups))
        test_combinations = list(combinations(groups, n_test_groups))

        splits = []

        for test_groups in test_combinations[:n_splits]:  # Limit number of splits
            train_indices = []
            test_indices = []

            for group in groups:
                start_idx = group * group_size
                end_idx = (group + 1) * group_size if group < len(groups) - 1 else n_samples

                if group in test_groups:
                    test_indices.extend(range(start_idx, end_idx))
                else:
                    # Add to training with purging
                    for test_group in test_groups:
                        test_start = test_group * group_size
                        test_end = (test_group + 1) * group_size

                        # Check if current group is too close to test group
                        if abs(group - test_group) * group_size > self.purge_days:
                            train_indices.extend(range(start_idx, end_idx))
                            break

            if len(train_indices) > 0 and len(test_indices) > 0:
                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]

                splits.append((X_train, y_train, X_test, y_test))

        return splits

    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_method: str = "walk_forward",
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate model using specified cross-validation method.

        Args:
            model: Model to validate
            X: Features
            y: Target
            cv_method: CV method to use
            metrics: List of metrics to calculate

        Returns:
            Dictionary with validation results
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "auc", "sharpe"]

        # Get splits based on method
        if cv_method == "walk_forward":
            splits = self.split_by_date(X, y)
        elif cv_method == "expanding":
            splits = self.expanding_window_split(X, y)
        elif cv_method == "purged_kfold":
            splits = self.purged_kfold_split(X, y)
        elif cv_method == "combinatorial":
            splits = self.combinatorial_cv(X, y)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")

        # Validate on each split
        results = {metric: [] for metric in metrics}
        fold_predictions = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
            logger.info(f"Validating fold {i+1}/{len(splits)}")

            # Clone model to avoid data leakage
            model_clone = self._clone_model(model)

            # Train on fold
            model_clone.fit(X_train, y_train)

            # Predict
            y_pred = model_clone.predict(X_test)

            # For probabilistic models
            if hasattr(model_clone, "predict_proba"):
                y_proba = model_clone.predict_proba(X_test)
            else:
                y_proba = None

            # Calculate metrics
            fold_results = self._calculate_metrics(y_test, y_pred, y_proba, X_test, metrics)

            # Store results
            for metric, value in fold_results.items():
                results[metric].append(value)

            # Store predictions for later analysis
            fold_predictions.append(
                {
                    "fold": i,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "dates": X_test.index,
                }
            )

        # Calculate summary statistics
        summary = {}
        for metric, values in results.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)

        # Add additional analysis
        summary["n_folds"] = len(splits)
        summary["total_train_samples"] = sum(len(split[0]) for split in splits)
        summary["total_test_samples"] = sum(len(split[2]) for split in splits)
        summary["fold_results"] = results
        summary["predictions"] = fold_predictions

        return summary

    def _clone_model(self, model):
        """Clone a model to avoid data leakage."""
        # Third-party imports
        from sklearn.base import clone

        try:
            return clone(model)
        except (TypeError, AttributeError) as e:
            # For custom models that don't support sklearn clone
            logger.warning(f"Model clone failed, using deepcopy: {e}")
            # Standard library imports
            import copy

            return copy.deepcopy(model)

    def _calculate_metrics(
        self, y_true, y_pred, y_proba, X_test, metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate requested metrics."""
        results = {}

        # Classification metrics
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in metrics:
            results["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

        if "recall" in metrics:
            results["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        if "f1" in metrics:
            results["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if "auc" in metrics and y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    results["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    results["auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except (ValueError, IndexError) as e:
                logger.warning(f"AUC calculation failed: {e}")
                results["auc"] = 0.0

        # Trading metrics
        if "sharpe" in metrics:
            # Calculate returns from predictions
            if "returns" in X_test.columns:
                returns = X_test["returns"]
            else:
                returns = pd.Series(0, index=X_test.index)

            strategy_returns = y_pred * returns
            results["sharpe"] = self._calculate_sharpe(strategy_returns)

        if "max_drawdown" in metrics:
            if "returns" in X_test.columns:
                returns = X_test["returns"]
                strategy_returns = y_pred * returns
                results["max_drawdown"] = self._calculate_max_drawdown(strategy_returns)

        return results

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualize based on data frequency
        if hasattr(returns.index, "freq"):
            if returns.index.freq == "D":
                periods_per_year = 252
            elif returns.index.freq == "H":
                periods_per_year = 252 * 6.5  # Trading hours
            else:
                periods_per_year = 252
        else:
            periods_per_year = 252

        sharpe = np.sqrt(periods_per_year) * mean_return / std_return
        return sharpe

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def plot_cv_results(self, cv_results: Dict[str, Any], save_path: str = None):
        """Plot cross-validation results."""
        # Third-party imports
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Cross-Validation Results", fontsize=16)

        # Plot 1: Metrics across folds
        ax = axes[0, 0]
        metrics_df = pd.DataFrame(cv_results["fold_results"])
        metrics_df.plot(kind="box", ax=ax)
        ax.set_title("Metrics Distribution Across Folds")
        ax.set_ylabel("Score")

        # Plot 2: Learning curves
        ax = axes[0, 1]
        if "accuracy" in cv_results["fold_results"]:
            folds = range(1, len(cv_results["fold_results"]["accuracy"]) + 1)
            ax.plot(folds, cv_results["fold_results"]["accuracy"], "o-", label="Accuracy")
            if "f1" in cv_results["fold_results"]:
                ax.plot(folds, cv_results["fold_results"]["f1"], "s-", label="F1 Score")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Score")
            ax.set_title("Performance Across Folds")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Prediction distribution
        ax = axes[1, 0]
        all_true = []
        all_pred = []
        for fold_data in cv_results["predictions"]:
            all_true.extend(fold_data["y_true"])
            all_pred.extend(fold_data["y_pred"])

        # Third-party imports
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(all_true, all_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_title("Aggregated Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        summary_data = []
        for metric in ["accuracy", "precision", "recall", "f1"]:
            if f"{metric}_mean" in cv_results:
                summary_data.append(
                    {
                        "Metric": metric.capitalize(),
                        "Mean": cv_results[f"{metric}_mean"],
                        "Std": cv_results[f"{metric}_std"],
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            x = np.arange(len(summary_df))
            ax.bar(x, summary_df["Mean"], yerr=summary_df["Std"], capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df["Metric"])
            ax.set_ylabel("Score")
            ax.set_title("Summary Statistics (Mean ± Std)")
            ax.set_ylim(0, 1.1)

            # Add value labels
            for i, (mean, std) in enumerate(zip(summary_df["Mean"], summary_df["Std"])):
                ax.text(i, mean + std + 0.02, f"{mean:.3f}", ha="center")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

"""
Feature Data Validator - Interface Implementation

Enhanced data validator for feature engineering pipeline.
Implements IFeatureValidator interface for feature-specific validation.
"""

# Standard library imports
from datetime import datetime
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
# Config imports
from main.config.validation_models import DataPipelineConfig

# Core imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.validation.core.validation_types import ValidationResult

# Interface imports
from main.interfaces.validation import IValidationContext, IValidationResult, ValidationStage
from main.utils.core import get_logger

logger = get_logger(__name__)


class FeatureValidator:
    """
    Feature data validator implementation.

    Implements IFeatureValidator interface for comprehensive
    feature validation in the ML pipeline.
    """

    def __init__(self, config: dict[str, Any] | DataPipelineConfig.ValidationConfig | None = None):
        """
        Initialize the feature validator.

        Args:
            config: Configuration dictionary or ValidationConfig object
        """
        # Handle different config types
        if isinstance(config, DataPipelineConfig.ValidationConfig):
            self.validation_config = config
            self.config = {}  # Keep for backward compatibility
            # Extract settings from ValidationConfig
            self.max_nan_ratio = config.features.max_nan_ratio
            self.max_inf_count = config.features.max_inf_count
            self.min_feature_coverage = config.features.min_feature_coverage
            self.correlation_thresholds = {"max_correlation": config.features.max_correlation}
            self.distribution_checks = config.features.distribution_checks
        else:
            # Legacy dict-based configuration
            self.config = config or {}
            self.validation_config = None
            self.max_nan_ratio = self.config.get("max_nan_ratio", 0.3)
            self.max_inf_count = self.config.get("max_inf_count", 0)
            self.min_feature_coverage = self.config.get("min_feature_coverage", 0.8)
            self.correlation_thresholds = self.config.get("correlation_thresholds", {})
            self.distribution_checks = self.config.get("distribution_checks", {})

        logger.info("Initialized FeatureValidator with interface-based architecture")

    # IValidator interface methods
    async def validate(self, data: Any, context: IValidationContext) -> IValidationResult:
        """Validate data with given context."""
        if isinstance(data, pd.DataFrame):
            return await self.validate_feature_dataframe(data, context)
        else:
            start_time = datetime.now()
            errors = [f"Feature validator expects DataFrame, got {type(data)}"]

            return ValidationResult(
                stage=context.stage,
                passed=False,
                errors=errors,
                warnings=[],
                metadata={},
                timestamp=start_time,
                duration_ms=0.0,
            )

    async def get_validation_rules(self, context: IValidationContext) -> list[str]:
        """Get applicable validation rules for context."""
        return [
            "feature_completeness",
            "feature_distributions",
            "feature_correlations",
            "feature_drift_detection",
            "nan_ratio_check",
            "inf_values_check",
            "constant_features_check",
        ]

    async def is_applicable(self, context: IValidationContext) -> bool:
        """Check if validator applies to given context."""
        return context.data_type in [DataType.FEATURES, DataType.MARKET_DATA] and context.layer in [
            DataLayer.FEATURE,
            DataLayer.PROCESSED,
        ]

    # IFeatureValidator interface methods
    async def validate_feature_dataframe(
        self, features: pd.DataFrame, context: IValidationContext
    ) -> IValidationResult:
        """Validate a feature DataFrame."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Basic DataFrame checks
            if features.empty:
                errors.append("Feature DataFrame is empty")
                return self._create_result(
                    context.stage, False, errors, warnings, metrics, start_time
                )

            metrics["total_features"] = len(features.columns)
            metrics["total_rows"] = len(features)

            # Check for NaN values
            nan_metrics = self._analyze_nan_values(features)
            metrics.update(nan_metrics)

            if nan_metrics["nan_ratio"] > self.max_nan_ratio:
                errors.append(
                    f"NaN ratio {nan_metrics['nan_ratio']:.2%} exceeds threshold {self.max_nan_ratio:.2%}"
                )

            # Check for infinite values
            inf_metrics = self._analyze_inf_values(features)
            metrics.update(inf_metrics)

            if inf_metrics["inf_count"] > self.max_inf_count:
                errors.append(
                    f"Infinite values count {inf_metrics['inf_count']} exceeds threshold {self.max_inf_count}"
                )

            # Check for constant features
            constant_metrics = self._analyze_constant_features(features)
            metrics.update(constant_metrics)

            if constant_metrics["constant_columns"]:
                warnings.append(
                    f"Found {len(constant_metrics['constant_column_names'])} constant features"
                )

            # Feature coverage analysis
            coverage_score = self._calculate_feature_coverage(features)
            metrics["feature_coverage"] = coverage_score

            if coverage_score < self.min_feature_coverage:
                errors.append(
                    f"Feature coverage {coverage_score:.2%} below threshold {self.min_feature_coverage:.2%}"
                )

            # Data quality score
            quality_score = self._calculate_quality_score(metrics)
            metrics["quality_score"] = quality_score

        except Exception as e:
            logger.error(f"Feature DataFrame validation error: {e}", exc_info=True)
            errors.append(f"Feature validation error: {e!s}")

        passed = len(errors) == 0
        return self._create_result(context.stage, passed, errors, warnings, metrics, start_time)

    async def validate_feature_completeness(
        self, features: pd.DataFrame, required_features: list[str], context: IValidationContext
    ) -> tuple[bool, list[str]]:
        """Validate feature completeness."""
        errors = []

        missing_features = [f for f in required_features if f not in features.columns]
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")

        return len(errors) == 0, errors

    async def validate_feature_distributions(
        self,
        features: pd.DataFrame,
        distribution_constraints: dict[str, dict[str, Any]],
        context: IValidationContext,
    ) -> tuple[bool, list[str]]:
        """Validate feature distributions are within expected bounds."""
        errors = []

        for feature_name, constraints in distribution_constraints.items():
            if feature_name not in features.columns:
                continue

            feature_data = features[feature_name].dropna()
            if feature_data.empty:
                errors.append(f"Feature '{feature_name}' has no valid data for distribution check")
                continue

            # Check mean bounds
            if "mean_bounds" in constraints:
                mean_val = feature_data.mean()
                min_mean, max_mean = constraints["mean_bounds"]
                if not (min_mean <= mean_val <= max_mean):
                    errors.append(
                        f"Feature '{feature_name}' mean {mean_val:.4f} outside bounds [{min_mean}, {max_mean}]"
                    )

            # Check standard deviation bounds
            if "std_bounds" in constraints:
                std_val = feature_data.std()
                min_std, max_std = constraints["std_bounds"]
                if not (min_std <= std_val <= max_std):
                    errors.append(
                        f"Feature '{feature_name}' std {std_val:.4f} outside bounds [{min_std}, {max_std}]"
                    )

            # Check value range
            if "value_range" in constraints:
                min_val, max_val = constraints["value_range"]
                feature_min = feature_data.min()
                feature_max = feature_data.max()
                if feature_min < min_val or feature_max > max_val:
                    errors.append(
                        f"Feature '{feature_name}' values [{feature_min:.4f}, {feature_max:.4f}] outside allowed range [{min_val}, {max_val}]"
                    )

        return len(errors) == 0, errors

    async def validate_feature_correlations(
        self,
        features: pd.DataFrame,
        correlation_constraints: dict[str, float],
        context: IValidationContext,
    ) -> tuple[bool, list[str]]:
        """Validate feature correlations."""
        errors = []

        try:
            # Calculate correlation matrix for numeric features
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return True, []

            corr_matrix = numeric_features.corr()

            # Check for high correlations
            max_correlation = correlation_constraints.get("max_correlation", 0.95)

            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > max_correlation:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        high_corr_pairs.append((feature1, feature2, corr_val))

            if high_corr_pairs:
                for feature1, feature2, corr_val in high_corr_pairs:
                    errors.append(
                        f"High correlation {corr_val:.3f} between '{feature1}' and '{feature2}' "
                        f"(threshold: {max_correlation})"
                    )

        except Exception as e:
            logger.warning(f"Correlation validation error: {e}")
            errors.append(f"Correlation validation failed: {e!s}")

        return len(errors) == 0, errors

    async def detect_feature_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
        context: IValidationContext,
    ) -> dict[str, Any]:
        """Detect feature drift compared to reference data."""
        drift_results = {
            "has_drift": False,
            "drift_features": [],
            "drift_scores": {},
            "summary": {},
        }

        try:
            common_features = list(set(current_features.columns) & set(reference_features.columns))

            if not common_features:
                drift_results["summary"]["error"] = "No common features for drift detection"
                return drift_results

            if self.validation_config:
                drift_threshold = self.validation_config.features.drift_threshold
            else:
                drift_threshold = self.config.get("drift_threshold", 0.1)

            for feature in common_features:
                current_data = current_features[feature].dropna()
                reference_data = reference_features[feature].dropna()

                if current_data.empty or reference_data.empty:
                    continue

                # Simple statistical drift detection (KS test would be better)
                current_mean = current_data.mean()
                reference_mean = reference_data.mean()
                current_std = current_data.std()
                reference_std = reference_data.std()

                # Calculate normalized difference
                if reference_std > 0:
                    mean_drift = abs(current_mean - reference_mean) / reference_std
                    std_drift = abs(current_std - reference_std) / reference_std

                    combined_drift = (mean_drift + std_drift) / 2
                    drift_results["drift_scores"][feature] = combined_drift

                    if combined_drift > drift_threshold:
                        drift_results["has_drift"] = True
                        drift_results["drift_features"].append(feature)

            drift_results["summary"] = {
                "total_features_checked": len(common_features),
                "features_with_drift": len(drift_results["drift_features"]),
                "max_drift_score": (
                    max(drift_results["drift_scores"].values())
                    if drift_results["drift_scores"]
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Feature drift detection error: {e}", exc_info=True)
            drift_results["summary"]["error"] = str(e)

        return drift_results

    async def validate_feature_engineering(
        self,
        source_data: pd.DataFrame,
        engineered_features: pd.DataFrame,
        context: IValidationContext,
    ) -> IValidationResult:
        """Validate feature engineering results."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Check that engineered features have same row count as source
            if len(source_data) != len(engineered_features):
                errors.append(
                    f"Row count mismatch: source {len(source_data)}, features {len(engineered_features)}"
                )

            # Check for feature name collisions with source data
            if hasattr(source_data, "columns"):
                overlapping_cols = set(source_data.columns) & set(engineered_features.columns)
                if overlapping_cols:
                    warnings.append(
                        f"Feature names overlap with source data: {list(overlapping_cols)}"
                    )

            # Validate engineered features
            feature_result = await self.validate_feature_dataframe(engineered_features, context)

            # Combine results
            errors.extend(feature_result.errors)
            warnings.extend(feature_result.warnings)
            metrics.update(feature_result.metrics)
            metrics["source_rows"] = len(source_data)
            metrics["feature_engineering_validated"] = True

        except Exception as e:
            logger.error(f"Feature engineering validation error: {e}", exc_info=True)
            errors.append(f"Feature engineering validation error: {e!s}")

        passed = len(errors) == 0
        return self._create_result(context.stage, passed, errors, warnings, metrics, start_time)

    # Helper methods
    def _analyze_nan_values(self, features: pd.DataFrame) -> dict[str, Any]:
        """Analyze NaN values in features."""
        total_values = features.size
        nan_count = features.isnull().sum().sum()
        nan_ratio = nan_count / total_values if total_values > 0 else 0

        # Find columns with all NaN values
        all_nan_columns = features.columns[features.isnull().all()].tolist()

        # Find columns with high NaN ratio
        if self.validation_config:
            high_nan_threshold = self.validation_config.features.high_nan_threshold
        else:
            high_nan_threshold = self.config.get("high_nan_threshold", 0.5)
        column_nan_ratios = features.isnull().mean()
        high_nan_columns = column_nan_ratios[column_nan_ratios > high_nan_threshold].index.tolist()

        return {
            "nan_count": nan_count,
            "nan_ratio": nan_ratio,
            "all_nan_columns": len(all_nan_columns),
            "all_nan_column_names": all_nan_columns,
            "high_nan_columns": len(high_nan_columns),
            "high_nan_column_names": high_nan_columns,
        }

    def _analyze_inf_values(self, features: pd.DataFrame) -> dict[str, Any]:
        """Analyze infinite values in features."""
        numeric_features = features.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_features).sum().sum()

        inf_columns = []
        for col in numeric_features.columns:
            if np.isinf(numeric_features[col]).any():
                inf_columns.append(col)

        return {
            "inf_count": inf_count,
            "inf_columns": len(inf_columns),
            "inf_column_names": inf_columns,
        }

    def _analyze_constant_features(self, features: pd.DataFrame) -> dict[str, Any]:
        """Analyze constant features."""
        constant_columns = []

        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_columns.append(col)

        return {
            "constant_columns": len(constant_columns),
            "constant_column_names": constant_columns,
        }

    def _calculate_feature_coverage(self, features: pd.DataFrame) -> float:
        """Calculate feature coverage score."""
        if features.empty:
            return 0.0

        # Calculate coverage as ratio of non-null values
        total_values = features.size
        non_null_values = features.count().sum()

        return non_null_values / total_values if total_values > 0 else 0.0

    def _calculate_quality_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall feature quality score."""
        score = 100.0

        # Penalize high NaN ratio
        if "nan_ratio" in metrics:
            score -= metrics["nan_ratio"] * 50

        # Penalize infinite values
        if "inf_count" in metrics and metrics["inf_count"] > 0:
            score -= min(metrics["inf_count"] * 10, 30)

        # Penalize constant features
        if "constant_columns" in metrics and metrics["constant_columns"] > 0:
            total_features = metrics.get("total_features", 1)
            constant_ratio = metrics["constant_columns"] / total_features
            score -= constant_ratio * 20

        return max(score, 0.0)

    def _create_result(
        self,
        stage: ValidationStage,
        passed: bool,
        errors: list[str],
        warnings: list[str],
        metrics: dict[str, Any],
        start_time: datetime,
    ) -> IValidationResult:
        """Create validation result."""
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metadata=metrics,  # Map metrics to metadata field
            timestamp=start_time,
            duration_ms=duration_ms,
        )

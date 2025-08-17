"""
Data Quality Calculator - Interface Implementation

Calculates comprehensive data quality metrics for DataFrames.
Implements IDataQualityCalculator interface for quality assessment.
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
# Core imports
from main.data_pipeline.core.enums import DataType

# Interface imports
from main.interfaces.validation import IValidationContext
from main.utils.core import get_logger

logger = get_logger(__name__)


class DataQualityCalculator:
    """
    Data quality calculator implementation.

    Implements IDataQualityCalculator interface for comprehensive
    quality assessment of DataFrames.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the data quality calculator.

        Args:
            config: Configuration dictionary with quality settings
        """
        self.config = config

        # Quality thresholds
        self.quality_threshold = config.get("quality_threshold", 70.0)
        self.max_nan_ratio = config.get("max_nan_ratio", 0.3)
        self.max_price_deviation = config.get("max_price_deviation", 0.5)

        # Profile settings
        self.require_all_fields = config.get("require_all_fields", False)
        self.allow_zero_volume = config.get("allow_zero_volume", True)
        self.allow_weekend_trading = config.get("allow_weekend_trading", False)
        self.allow_future_timestamps = config.get("allow_future_timestamps", False)
        self.min_data_points = config.get("min_data_points", 10)

        logger.info("Initialized DataQualityCalculator with interface-based architecture")

    # IDataQualityCalculator interface methods
    async def calculate_quality_score(self, data: Any, context: IValidationContext) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        if not isinstance(data, pd.DataFrame):
            return 0.0

        if data.empty:
            return 0.0

        quality_metrics = await self.get_quality_metrics(data, context)
        return quality_metrics.get("quality_score", 0.0) / 100.0  # Convert to 0.0-1.0 scale

    async def calculate_completeness(self, data: Any, context: IValidationContext) -> float:
        """Calculate data completeness score."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return 0.0

        total_cells = data.size
        if total_cells == 0:
            return 0.0

        non_null_cells = data.count().sum()
        return non_null_cells / total_cells

    async def calculate_accuracy(self, data: Any, context: IValidationContext) -> float:
        """Calculate data accuracy score."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return 0.0

        accuracy_score = 100.0
        row_count = len(data)

        # Check OHLCV relationships for market data
        if context.data_type == DataType.MARKET_DATA:
            ohlc_cols = ["open", "high", "low", "close"]
            if all(col in data.columns for col in ohlc_cols):
                # Check OHLC relationship violations
                violations = self._count_ohlc_violations(data)
                violation_ratio = sum(violations.values()) / (row_count * len(violations))
                accuracy_score -= violation_ratio * 50  # Up to 50% penalty

        # Check for negative values where they shouldn't be
        if "volume" in data.columns:
            negative_volume_ratio = (data["volume"] < 0).sum() / row_count
            accuracy_score -= negative_volume_ratio * 30

        return max(accuracy_score, 0.0) / 100.0

    async def calculate_consistency(self, data: Any, context: IValidationContext) -> float:
        """Calculate data consistency score."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return 0.0

        consistency_score = 100.0

        # Check for data type consistency
        for col in data.columns:
            if data[col].dtype == "object":
                # Check if mixed types in object column
                try:
                    pd.to_numeric(data[col], errors="raise")
                except (ValueError, TypeError):
                    # Mixed types detected
                    consistency_score -= 5

        # Check for duplicate timestamps if DateTime index
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_ratio = data.index.duplicated().sum() / len(data)
            consistency_score -= duplicate_ratio * 20

        # Check for extreme outliers that might indicate inconsistency
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_ratio = (z_scores > 3).sum() / len(data)
                consistency_score -= outlier_ratio * 10

        return max(consistency_score, 0.0) / 100.0

    async def get_quality_metrics(self, data: Any, context: IValidationContext) -> dict[str, float]:
        """Get detailed quality metrics."""
        if not isinstance(data, pd.DataFrame):
            return {"error": "Data must be a pandas DataFrame"}

        if data.empty:
            return self._empty_df_metrics()

        issues = []
        quality_score = 100.0
        row_count = len(data)

        # Basic structure validation
        required_cols = self._get_required_columns(context.data_type)
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing essential columns: {missing_cols}")
            quality_score -= 30

        # OHLCV relationship validation for market data
        ohlc_valid = True
        if context.data_type == DataType.MARKET_DATA:
            ohlc_issues, ohlc_penalty = await self._validate_ohlcv_relationships(data)
            issues.extend(ohlc_issues)
            quality_score -= ohlc_penalty
            ohlc_valid = len(ohlc_issues) == 0

        # Missing data analysis
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0

        if missing_pct > self.max_nan_ratio * 100:
            issues.append(f"High missing data rate: {missing_pct:.1f}%")
            quality_score -= missing_pct

        # Index validation for time series data
        if isinstance(data.index, pd.DatetimeIndex):
            index_issues, index_penalty = await self._validate_datetime_index(data)
            issues.extend(index_issues)
            quality_score -= index_penalty
        else:
            issues.append("DataFrame index is not a DatetimeIndex")
            quality_score -= 15

        # Feature-specific metrics
        feature_metrics = await self._calculate_feature_metrics(data)

        # Time series validation
        if isinstance(data.index, pd.DatetimeIndex) and len(data) >= self.min_data_points:
            ts_issues, ts_penalty = await self._validate_time_series(data)
            issues.extend(ts_issues)
            quality_score -= ts_penalty

        # Final quality score adjustment
        quality_score = max(0.0, min(100.0, quality_score))

        # Determine validity
        is_valid = quality_score >= self.quality_threshold
        if missing_cols and self.require_all_fields:
            is_valid = False

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_score, missing_cols, missing_pct, issues
        )

        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "completeness": await self.calculate_completeness(data, context),
            "accuracy": await self.calculate_accuracy(data, context),
            "consistency": await self.calculate_consistency(data, context),
            "issues": list(set(issues)),
            "row_count": row_count,
            "missing_data_pct": missing_pct,
            "ohlc_valid": ohlc_valid,
            "feature_metrics": feature_metrics,
            "recommendations": recommendations,
        }

    # Helper methods
    def _get_required_columns(self, data_type: DataType) -> list[str]:
        """Get required columns for data type."""
        if data_type == DataType.MARKET_DATA:
            return ["open", "high", "low", "close", "volume"]
        elif data_type == DataType.NEWS:
            return ["title", "content", "timestamp"]
        elif data_type == DataType.FINANCIALS:
            return ["symbol", "period", "value"]
        else:
            return []

    async def _validate_ohlcv_relationships(self, data: pd.DataFrame) -> tuple[list[str], float]:
        """Validate OHLCV relationships."""
        issues = []
        penalty = 0.0
        row_count = len(data)

        ohlc_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in ohlc_cols):
            return issues, penalty

        # Check for non-positive prices
        for col in ohlc_cols:
            if data[col].dtype.kind in "fi":  # numeric type
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"Found {negative_count} non-positive values in {col}")
                    penalty += min(20, (negative_count / row_count) * 100)

        # Check OHLC relationship violations
        violations = self._count_ohlc_violations(data)
        for issue_type, count in violations.items():
            if count > 0:
                issues.append(f"Found {count} rows where {issue_type}")
                penalty += min(10, (count / row_count) * 100)

        # Check for extreme price deviations
        median_prices = data[ohlc_cols].median(axis=1)
        for col in ohlc_cols:
            if not median_prices.empty:
                deviations = abs(data[col] - median_prices) / median_prices.replace(0, np.nan)
                extreme_count = (deviations > self.max_price_deviation).sum()
                if extreme_count > 0:
                    issues.append(
                        f"Found {extreme_count} values in {col} with extreme deviation "
                        f"(> {self.max_price_deviation*100:.0f}%) from median"
                    )
                    penalty += min(15, (extreme_count / row_count) * 50)

        # Volume validation
        if "volume" in data.columns and data["volume"].dtype.kind in "fi":
            negative_volume = (data["volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Found {negative_volume} negative volume values")
                penalty += min(10, (negative_volume / row_count) * 100)

            zero_volume = (data["volume"] == 0).sum()
            if zero_volume > 0 and not self.allow_zero_volume:
                issues.append(f"Found {zero_volume} zero volume values (not allowed)")
                penalty += min(5, (zero_volume / row_count) * 100)

        return issues, penalty

    def _count_ohlc_violations(self, data: pd.DataFrame) -> dict[str, int]:
        """Count OHLC relationship violations."""
        return {
            "high < low": (data["high"] < data["low"]).sum(),
            "high < open": (data["high"] < data["open"]).sum(),
            "high < close": (data["high"] < data["close"]).sum(),
            "low > open": (data["low"] > data["open"]).sum(),
            "low > close": (data["low"] > data["close"]).sum(),
        }

    async def _validate_datetime_index(self, data: pd.DataFrame) -> tuple[list[str], float]:
        """Validate datetime index."""
        issues = []
        penalty = 0.0

        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            issues.append(f"Found {duplicate_count} duplicate timestamps")
            penalty += min(15, (duplicate_count / len(data)) * 100)

        return issues, penalty

    async def _validate_time_series(self, data: pd.DataFrame) -> tuple[list[str], float]:
        """Validate time series properties."""
        issues = []
        penalty = 0.0

        # Check for large time gaps
        time_diffs = data.index.to_series().diff().dt.total_seconds()
        median_diff = time_diffs.median()

        if pd.notna(median_diff) and median_diff > 0:
            large_gaps = (time_diffs > median_diff * 5).sum()
            if large_gaps > 0:
                issues.append(f"Detected {large_gaps} large time gaps")
                penalty += min(10, (large_gaps / len(data)) * 100)

        # Check for weekend trading
        if not self.allow_weekend_trading:
            weekend_count = data.index.dayofweek.isin([5, 6]).sum()
            if weekend_count > 0:
                issues.append(f"Weekend trading detected: {weekend_count} records")
                penalty += min(5, (weekend_count / len(data)) * 100)

        # Check for future timestamps
        if not self.allow_future_timestamps:
            now = pd.Timestamp.now(tz="UTC")
            future_count = (data.index > now).sum()
            if future_count > 0:
                issues.append(f"Future timestamps detected: {future_count} records")
                penalty += min(15, (future_count / len(data)) * 100)

        return issues, penalty

    async def _calculate_feature_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate detailed feature metrics."""
        if data.empty:
            return {"is_empty": True, "nan_ratio": 1.0, "feature_coverage": 0.0}

        metrics = {"is_empty": False}

        # NaN analysis
        total_values = data.size
        nan_count = data.isnull().sum().sum()
        metrics["nan_ratio"] = nan_count / total_values if total_values > 0 else 0.0

        # All NaN columns
        all_nan_cols = data.columns[data.isnull().all()].tolist()
        metrics["all_nan_columns"] = len(all_nan_cols)
        metrics["all_nan_column_names"] = all_nan_cols

        # High NaN columns
        column_nan_ratios = data.isnull().mean()
        high_nan_cols = column_nan_ratios[column_nan_ratios > self.max_nan_ratio].index.tolist()
        metrics["high_nan_columns"] = len(high_nan_cols)
        metrics["high_nan_column_names"] = high_nan_cols

        # Infinite values
        numeric_features = data.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            inf_count = np.isinf(numeric_features).sum().sum()
            metrics["inf_count"] = int(inf_count)
        else:
            metrics["inf_count"] = 0

        # Constant columns
        constant_cols = []
        if not numeric_features.empty:
            for col in numeric_features.columns:
                if numeric_features[col].nunique(dropna=True) <= 1:
                    constant_cols.append(col)
        metrics["constant_columns"] = len(constant_cols)
        metrics["constant_column_names"] = constant_cols

        # Feature coverage
        metrics["feature_coverage"] = (
            (1 - data.isnull().all(axis=1).sum() / len(data)) if len(data) > 0 else 0.0
        )

        return metrics

    def _generate_recommendations(
        self, quality_score: float, missing_cols: list[str], missing_pct: float, issues: list[str]
    ) -> list[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if quality_score < self.quality_threshold:
            recommendations.append(
                f"Data quality ({quality_score:.1f}%) is below threshold ({self.quality_threshold}%). "
                "Consider alternative data sources or more aggressive cleaning."
            )

        if missing_cols:
            recommendations.append(
                f"Ensure data source provides all required columns. Missing: {', '.join(missing_cols)}."
            )

        if missing_pct > self.max_nan_ratio * 100:
            recommendations.append(
                f"High missing data rate ({missing_pct:.1f}%). "
                "Check data source reliability or ingestion pipeline."
            )

        # Issue-specific recommendations
        if any("ohlc" in issue.lower() or "high" in issue.lower() for issue in issues):
            recommendations.append(
                "OHLC relationship violations detected. Data may be corrupted or "
                "reflect extreme market conditions."
            )

        if any("extreme deviation" in issue.lower() for issue in issues):
            recommendations.append(
                "Extreme price deviations detected. Consider implementing outlier filtering."
            )

        if any("negative volume" in issue.lower() for issue in issues):
            recommendations.append(
                "Negative volume detected. Investigate data source for invalid values."
            )

        if any("duplicate timestamps" in issue.lower() for issue in issues):
            recommendations.append(
                "Duplicate timestamps detected. Ensure data pipeline handles deduplication."
            )

        return list(set(recommendations))

    def _empty_df_metrics(self) -> dict[str, Any]:
        """Return metrics for empty DataFrame."""
        return {
            "is_valid": False,
            "quality_score": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "issues": ["DataFrame is empty"],
            "row_count": 0,
            "missing_data_pct": 100.0,
            "ohlc_valid": False,
            "feature_metrics": {"is_empty": True},
            "recommendations": ["Ensure data source provides data or check filtering criteria."],
        }

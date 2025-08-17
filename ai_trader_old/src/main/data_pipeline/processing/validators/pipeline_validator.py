"""
Pipeline Validator

Wrapper around ValidationUtils for comprehensive data validation.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.data import ValidationLevel, ValidationUtils, get_global_validator
from main.utils.monitoring import MetricType, record_metric, timer


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    data_type: DataType | None = None
    layer: DataLayer | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class PipelineValidator(ErrorHandlingMixin):
    """
    Pipeline validator using ValidationUtils for all operations.

    Provides layer-aware and data-type-specific validation.
    """

    def __init__(self):
        """Initialize validator with utils."""
        self.logger = get_logger(__name__)
        self.global_validator = get_global_validator()
        self._validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "warnings_generated": 0,
        }

    async def validate_data(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Main validation entry point with layer-aware rules.

        Args:
            data: DataFrame to validate
            data_type: Type of data
            layer: Data layer for validation rules
            context: Optional context

        Returns:
            ValidationResult with detailed information
        """
        with timer(
            "validate.process",
            tags={
                "data_type": data_type.value if hasattr(data_type, "value") else str(data_type),
                "layer": layer.name,
            },
        ):
            result = ValidationResult(is_valid=True, data_type=data_type, layer=layer)

            # Basic validation
            if data.empty:
                result.is_valid = False
                result.errors.append("Empty DataFrame")
                return result

            # Record input metrics
            result.metrics["row_count"] = len(data)
            result.metrics["column_count"] = len(data.columns)
            # gauge("validate.input_rows", len(data), tags={"layer": layer.name})

            # Data type specific validation
            if data_type == DataType.MARKET_DATA:
                await self._validate_market_data(data, result, layer)
            elif data_type == DataType.NEWS:
                await self._validate_news_data(data, result, layer)
            elif data_type == DataType.FINANCIALS:
                await self._validate_fundamentals_data(data, result, layer)
            elif data_type == DataType.CORPORATE_ACTIONS:
                await self._validate_corporate_actions(data, result, layer)

            # General data quality checks
            await self._validate_data_quality(data, result, layer)

            # Layer-specific validation
            if layer >= DataLayer.CATALYST:
                await self._apply_strict_validation(data, result)

            # Update stats
            self._validation_stats["total_validations"] += 1
            if result.is_valid:
                self._validation_stats["passed"] += 1
            else:
                self._validation_stats["failed"] += 1
            self._validation_stats["warnings_generated"] += len(result.warnings)

            # Record metrics
            record_metric(
                "validate.errors",
                len(result.errors),
                MetricType.COUNTER,
                tags={"data_type": str(data_type), "layer": layer.name},
            )
            record_metric(
                "validate.warnings",
                len(result.warnings),
                MetricType.COUNTER,
                tags={"data_type": str(data_type), "layer": layer.name},
            )

            return result

    async def _validate_market_data(
        self, data: pd.DataFrame, result: ValidationResult, layer: DataLayer
    ) -> None:
        """Validate market data using ValidationUtils."""
        # Use ValidationUtils for OHLCV validation
        is_valid, errors = ValidationUtils.validate_ohlcv_data(data)

        if not is_valid:
            result.is_valid = False
            result.errors.extend(errors)

        # Additional checks for higher layers
        if layer >= DataLayer.LIQUID:
            # Check for price continuity
            if "close" in data.columns and len(data) > 1:
                price_changes = data["close"].pct_change().abs()
                extreme_changes = price_changes[price_changes > 0.5]  # >50% change
                if len(extreme_changes) > 0:
                    result.warnings.append(
                        f"Found {len(extreme_changes)} extreme price changes (>50%)"
                    )

        # Check volume validity
        if "volume" in data.columns:
            zero_volume = (data["volume"] == 0).sum()
            if zero_volume > len(data) * 0.1:  # >10% zero volume
                result.warnings.append(
                    f"High proportion of zero volume: {zero_volume/len(data):.1%}"
                )

    async def _validate_news_data(
        self, data: pd.DataFrame, result: ValidationResult, layer: DataLayer
    ) -> None:
        """Validate news data."""
        required_cols = ["title", "published_at"]
        missing = [col for col in required_cols if col not in data.columns]

        if missing:
            result.is_valid = False
            result.errors.append(f"Missing required news columns: {missing}")

        # Check text fields
        if "title" in data.columns:
            empty_titles = data["title"].isna().sum() + (data["title"] == "").sum()
            if empty_titles > 0:
                result.errors.append(f"Found {empty_titles} empty titles")
                result.is_valid = False

        # Check timestamps
        if "published_at" in data.columns:
            try:
                pd.to_datetime(data["published_at"])
            except (ValueError, TypeError, pd.errors.ParserError) as e:
                logger.debug(f"Invalid timestamp format in published_at: {e}")
                result.errors.append("Invalid timestamp format in published_at")
                result.is_valid = False

    async def _validate_fundamentals_data(
        self, data: pd.DataFrame, result: ValidationResult, layer: DataLayer
    ) -> None:
        """Validate fundamentals data."""
        # Check for at least one financial metric
        financial_cols = ["revenue", "net_income", "eps_basic", "total_assets"]
        available = [col for col in financial_cols if col in data.columns]

        if not available:
            result.errors.append("No financial metrics found")
            result.is_valid = False

        # Check for negative values where they shouldn't be
        positive_only = ["revenue", "total_assets", "total_liabilities"]
        for col in positive_only:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    result.warnings.append(f"Found {negative_count} negative values in {col}")

    async def _validate_corporate_actions(
        self, data: pd.DataFrame, result: ValidationResult, layer: DataLayer
    ) -> None:
        """Validate corporate actions data."""
        # Check for action type
        if "action_type" in data.columns:
            valid_types = ["dividend", "split", "merger", "spinoff"]
            invalid = ~data["action_type"].isin(valid_types)
            if invalid.any():
                result.warnings.append(f"Found {invalid.sum()} invalid action types")

        # Validate dividends
        if "cash_amount" in data.columns:
            negative_dividends = (data["cash_amount"] < 0).sum()
            if negative_dividends > 0:
                result.errors.append(f"Found {negative_dividends} negative dividend amounts")
                result.is_valid = False

        # Validate splits
        if "split_from" in data.columns and "split_to" in data.columns:
            invalid_splits = ((data["split_from"] <= 0) | (data["split_to"] <= 0)).sum()
            if invalid_splits > 0:
                result.errors.append(f"Found {invalid_splits} invalid split ratios")
                result.is_valid = False

    async def _validate_data_quality(
        self, data: pd.DataFrame, result: ValidationResult, layer: DataLayer
    ) -> None:
        """General data quality validation using ValidationUtils."""
        # Determine max missing percentage based on layer
        max_missing_pct = {
            DataLayer.BASIC: 0.2,  # 20% allowed
            DataLayer.LIQUID: 0.1,  # 10% allowed
            DataLayer.CATALYST: 0.05,  # 5% allowed
            DataLayer.ACTIVE: 0.01,  # 1% allowed
        }.get(layer, 0.1)

        # Use ValidationUtils for quality check
        is_valid, quality_issues = ValidationUtils.check_data_quality(data, max_missing_pct)

        if not is_valid:
            for col, issue in quality_issues.items():
                result.warnings.append(f"Column '{col}': {issue}")

        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            duplicate_pct = duplicates / len(data) * 100
            result.warnings.append(f"Found {duplicates} duplicate rows ({duplicate_pct:.1f}%)")
            result.metrics["duplicate_rows"] = duplicates

    async def _apply_strict_validation(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Apply strict validation for CATALYST and ACTIVE layers."""
        # Check for data consistency
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for inf values
            inf_count = np.isinf(data[col]).sum()
            if inf_count > 0:
                result.errors.append(f"Found {inf_count} infinite values in {col}")
                result.is_valid = False

            # Check for unrealistic values
            if col in ["price", "open", "high", "low", "close"]:
                if (data[col] <= 0).any():
                    result.errors.append(f"Found non-positive prices in {col}")
                    result.is_valid = False

                # Check for unrealistic prices
                if (data[col] > 1000000).any():  # $1M per share
                    result.warnings.append(f"Found suspiciously high prices in {col}")

    def _get_validation_level(self, layer: DataLayer) -> ValidationLevel:
        """Map layer to validation level."""
        if layer >= DataLayer.ACTIVE:
            return ValidationLevel.STRICT
        elif layer >= DataLayer.CATALYST or layer >= DataLayer.LIQUID:
            return ValidationLevel.NORMAL
        else:
            return ValidationLevel.MINIMAL

    async def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        stats = self._validation_stats.copy()
        if stats["total_validations"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_validations"]
            stats["avg_warnings"] = stats["warnings_generated"] / stats["total_validations"]
        return stats

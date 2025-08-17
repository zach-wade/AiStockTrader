"""
Data Cleaner

Wrapper around ProcessingUtils for data cleaning operations.
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.processing import IDataCleaner
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.data import ProcessingUtils
from main.utils.monitoring import MetricType, record_metric, timer


class DataCleaner(IDataCleaner, ErrorHandlingMixin):
    """
    Data cleaner using ProcessingUtils for all operations.

    Handles missing values, outliers, duplicates with layer-aware strategies.
    """

    def __init__(self):
        """Initialize cleaner."""
        self.logger = get_logger(__name__)
        self._cleaning_stats = {
            "total_cleaned": 0,
            "duplicates_removed": 0,
            "outliers_handled": 0,
            "missing_handled": 0,
        }

    async def clean(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer,
        cleaning_profile: str | None = None,
    ) -> pd.DataFrame:
        """
        Clean data according to layer-specific rules.

        Args:
            data: Input DataFrame
            data_type: Type of data
            layer: Data layer for rules
            cleaning_profile: Optional cleaning profile

        Returns:
            Cleaned DataFrame
        """
        with timer(
            "clean.process",
            tags={
                "layer": layer.name,
                "data_type": data_type.value if hasattr(data_type, "value") else str(data_type),
                "profile": cleaning_profile or "default",
            },
        ):
            if data.empty:
                return data

            original_len = len(data)
            df = data.copy()

            # Apply cleaning steps based on layer
            df = await self.remove_duplicates(df, layer)
            df = await self.handle_missing_values(df, layer)

            # Detect and handle outliers
            outlier_info = await self.detect_outliers(df, layer)
            if outlier_info.get("outliers_found", False):
                df = await self.clean_outliers(df, outlier_info, layer)

            # Record cleaning metrics
            cleaned_len = len(df)
            reduction_pct = (1 - cleaned_len / original_len) * 100 if original_len > 0 else 0

            # gauge("clean.reduction_pct", reduction_pct, tags={"layer": layer.name})
            record_metric(
                "clean.rows_removed",
                original_len - cleaned_len,
                MetricType.COUNTER,
                tags={"layer": layer.name},
            )

            self._cleaning_stats["total_cleaned"] += 1

            return df

    async def remove_duplicates(
        self, data: pd.DataFrame, layer: DataLayer, dedup_strategy: str | None = None
    ) -> pd.DataFrame:
        """Remove duplicate records."""
        df = data.copy()
        original_len = len(df)

        # Determine strategy based on layer
        if dedup_strategy is None:
            if layer >= DataLayer.CATALYST:
                dedup_strategy = "strict"  # Remove all duplicates
            else:
                dedup_strategy = "keep_last"  # Keep most recent

        # Apply deduplication
        if dedup_strategy == "strict":
            # Remove all duplicates across all columns
            df = df.drop_duplicates()
        elif dedup_strategy == "keep_last":
            # Keep last occurrence
            df = df.drop_duplicates(keep="last")
        elif dedup_strategy == "keep_first":
            # Keep first occurrence
            df = df.drop_duplicates(keep="first")
        elif dedup_strategy == "by_key":
            # Deduplicate by key columns (timestamp + symbol if available)
            key_cols = []
            if "timestamp" in df.columns:
                key_cols.append("timestamp")
            if "symbol" in df.columns:
                key_cols.append("symbol")
            if key_cols:
                df = df.drop_duplicates(subset=key_cols, keep="last")

        removed = original_len - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate rows")
            self._cleaning_stats["duplicates_removed"] += removed

        return df

    async def handle_missing_values(
        self, data: pd.DataFrame, layer: DataLayer, strategy: str | None = None
    ) -> pd.DataFrame:
        """Handle missing values."""
        df = data.copy()

        # Determine strategy based on layer
        if strategy is None:
            if layer >= DataLayer.ACTIVE:
                strategy = "interpolate"  # Most sophisticated
            elif layer >= DataLayer.CATALYST:
                strategy = "forward_fill"  # Fill forward
            elif layer >= DataLayer.LIQUID:
                strategy = "backward_fill"  # Fill backward
            else:
                strategy = "drop"  # Simple drop

        # Apply strategy using ProcessingUtils
        before_missing = df.isnull().sum().sum()
        df = ProcessingUtils.handle_missing_values(df, method=strategy)
        after_missing = df.isnull().sum().sum()

        handled = before_missing - after_missing
        if handled > 0:
            self.logger.info(f"Handled {handled} missing values using {strategy}")
            self._cleaning_stats["missing_handled"] += handled

        return df

    async def detect_outliers(
        self, data: pd.DataFrame, layer: DataLayer, method: str | None = None
    ) -> dict[str, Any]:
        """Detect outliers in data."""
        if method is None:
            method = "iqr" if layer <= DataLayer.LIQUID else "zscore"

        outlier_info = {
            "method": method,
            "outliers_found": False,
            "outlier_columns": {},
            "total_outliers": 0,
        }

        # Only check numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ["volume", "trades"]:  # Skip volume-related columns
                continue

            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                threshold = 3 if layer <= DataLayer.LIQUID else 2
                outliers = z_scores > threshold
            else:
                outliers = pd.Series([False] * len(data))

            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_info["outliers_found"] = True
                outlier_info["outlier_columns"][col] = outlier_count
                outlier_info["total_outliers"] += outlier_count

        return outlier_info

    async def clean_outliers(
        self,
        data: pd.DataFrame,
        outlier_info: dict[str, Any],
        layer: DataLayer,
        action: str = "flag",
    ) -> pd.DataFrame:
        """Clean outliers from data."""
        df = data.copy()

        # Determine action based on layer
        if action == "flag":
            action = "clip" if layer >= DataLayer.CATALYST else "flag"

        method = outlier_info["method"]

        for col, count in outlier_info["outlier_columns"].items():
            if action == "remove":
                # Remove outliers using ProcessingUtils
                df = ProcessingUtils.remove_outliers(df, method=method)
            elif action == "clip":
                # Clip outliers to bounds
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                elif method == "zscore":
                    mean = df[col].mean()
                    std = df[col].std()
                    threshold = 3 if layer <= DataLayer.LIQUID else 2
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    df[col] = df[col].clip(lower_bound, upper_bound)
            elif action == "flag":
                # Just flag outliers with a new column
                df[f"{col}_outlier"] = False
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[f"{col}_outlier"] = (df[col] < lower_bound) | (df[col] > upper_bound)

        self._cleaning_stats["outliers_handled"] += outlier_info["total_outliers"]
        self.logger.info(f"Handled {outlier_info['total_outliers']} outliers using {action}")

        return df

    async def get_cleaning_stats(self) -> dict[str, Any]:
        """Get data cleaning statistics."""
        return self._cleaning_stats.copy()

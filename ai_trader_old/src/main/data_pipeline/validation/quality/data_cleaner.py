"""
Data Cleaner - Interface Implementation

Handles standardization and cleaning operations for DataFrames and records.
Implements IDataCleaner interface for data cleaning operations.
"""

# Standard library imports
from datetime import date
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


class QualityDataCleaner:
    """
    Quality data cleaner implementation.

    Implements IDataCleaner interface for comprehensive
    data cleaning and standardization operations.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the data cleaner.

        Args:
            config: Configuration dictionary with cleaning settings
        """
        self.config = config

        # Cleaning settings
        self.aggressive_cleaning = config.get("aggressive_cleaning", False)
        self.max_nan_ratio = config.get("max_nan_ratio", 0.5)
        self.fill_limit = config.get("fill_limit", 5)
        self.handle_constants = config.get("handle_constants", True)

        # Field mapping and schema information
        self.field_mappings = config.get("field_mappings", {})
        self.allowed_fields = config.get("allowed_fields", {})
        self.column_lengths = config.get("column_lengths", {})
        self.default_values = config.get("default_values", {})

        logger.info("Initialized DataCleaner with interface-based architecture")

    # IDataCleaner interface methods
    async def clean_data(
        self, data: Any, context: IValidationContext, in_place: bool = False
    ) -> Any:
        """Clean data according to validation rules."""
        if isinstance(data, pd.DataFrame):
            return await self._clean_dataframe(data, context, in_place)
        elif isinstance(data, dict):
            return await self._clean_record(data, context)
        elif isinstance(data, list):
            return await self._clean_record_list(data, context)
        else:
            logger.warning(f"Unsupported data type for cleaning: {type(data)}")
            return data

    async def remove_duplicates(self, data: Any, context: IValidationContext) -> Any:
        """Remove duplicate records."""
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.DatetimeIndex):
                # Remove duplicate indices (keep last for time series)
                return data[~data.index.duplicated(keep="last")]
            else:
                # Remove duplicate rows
                return data.drop_duplicates(keep="last")
        elif isinstance(data, list):
            # Remove duplicate records from list
            seen = set()
            result = []
            for item in data:
                if isinstance(item, dict):
                    # Create hashable key from dict
                    key = tuple(sorted(item.items()))
                    if key not in seen:
                        seen.add(key)
                        result.append(item)
                elif item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            return data

    async def handle_missing_values(
        self, data: Any, context: IValidationContext, strategy: str = "drop"
    ) -> Any:
        """Handle missing values with specified strategy."""
        if not isinstance(data, pd.DataFrame):
            return data

        if strategy == "drop":
            return data.dropna()
        elif strategy == "fill_forward":
            return data.ffill(limit=self.fill_limit)
        elif strategy == "fill_backward":
            return data.bfill(limit=self.fill_limit)
        elif strategy == "fill_both":
            return data.ffill(limit=self.fill_limit).bfill(limit=self.fill_limit)
        elif strategy == "fill_zero":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(0)
            return data
        elif strategy == "interpolate":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(
                method="linear", limit=self.fill_limit
            )
            return data
        else:
            logger.warning(f"Unknown missing value strategy: {strategy}")
            return data

    async def normalize_data(self, data: Any, context: IValidationContext) -> Any:
        """Normalize data formats and values."""
        if isinstance(data, pd.DataFrame):
            return await self._normalize_dataframe(data, context)
        elif isinstance(data, dict):
            return await self._normalize_record(data, context)
        else:
            return data

    async def get_cleaning_summary(
        self, original_data: Any, cleaned_data: Any, context: IValidationContext
    ) -> dict[str, Any]:
        """Get summary of cleaning operations performed."""
        summary = {"cleaning_performed": True, "operations": [], "statistics": {}}

        if isinstance(original_data, pd.DataFrame) and isinstance(cleaned_data, pd.DataFrame):
            original_shape = original_data.shape
            cleaned_shape = cleaned_data.shape

            summary["statistics"] = {
                "original_rows": original_shape[0],
                "original_columns": original_shape[1],
                "cleaned_rows": cleaned_shape[0],
                "cleaned_columns": cleaned_shape[1],
                "rows_removed": original_shape[0] - cleaned_shape[0],
                "columns_removed": original_shape[1] - cleaned_shape[1],
            }

            if original_shape != cleaned_shape:
                summary["operations"].append("Shape modification")

            # Check for missing value changes
            original_nulls = original_data.isnull().sum().sum()
            cleaned_nulls = cleaned_data.isnull().sum().sum()

            if original_nulls != cleaned_nulls:
                summary["operations"].append("Missing value handling")
                summary["statistics"]["original_nulls"] = original_nulls
                summary["statistics"]["cleaned_nulls"] = cleaned_nulls

        return summary

    # Public methods for specific cleaning operations
    async def standardize_dataframe(
        self,
        data: pd.DataFrame,
        symbol: str | None = None,
        context: IValidationContext | None = None,
    ) -> pd.DataFrame:
        """Standardize a DataFrame for consistent processing."""
        if data.empty:
            logger.warning(
                f"Empty DataFrame provided for standardization{' for ' + symbol if symbol else ''}"
            )
            return data

        df = data.copy()

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Ensure proper datetime index
        df = await self._ensure_datetime_index(df, symbol)

        # Sort by index if datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            # Remove duplicate indices (keep last)
            df = df[~df.index.duplicated(keep="last")]

        # Replace infinite values with NaN
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

        logger.debug(f"DataFrame standardized for {symbol if symbol else 'unknown'}")
        return df

    async def clean_features(
        self,
        features: pd.DataFrame,
        aggressive: bool | None = None,
        context: IValidationContext | None = None,
    ) -> pd.DataFrame:
        """Clean feature DataFrame with configurable aggressiveness."""
        if features.empty:
            logger.warning("Empty features DataFrame provided for cleaning")
            return features

        effective_aggressive = aggressive if aggressive is not None else self.aggressive_cleaning
        original_shape = features.shape
        df = features.copy()

        # Step 1: Remove entirely NaN columns
        df = df.dropna(axis=1, how="all")

        # Step 2: Replace infinite values with NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if effective_aggressive:
            logger.debug("Applying aggressive cleaning for features")

            # Step 3: Remove high NaN columns
            if len(df) > 0:
                nan_ratios = df.isnull().mean()
                high_nan_cols = nan_ratios[nan_ratios > self.max_nan_ratio].index.tolist()
                if high_nan_cols:
                    df = df.drop(columns=high_nan_cols)
                    logger.debug(f"Dropped high NaN columns: {high_nan_cols}")

            # Step 4: Remove constant columns
            if self.handle_constants:
                constant_cols = await self._find_constant_columns(df)
                if constant_cols:
                    df = df.drop(columns=constant_cols)
                    logger.debug(f"Dropped constant columns: {constant_cols}")

        # Step 5: Handle remaining missing values
        df = await self.handle_missing_values(df, context, strategy="fill_both")

        # Step 6: Fill any remaining NaN with 0 for numeric columns
        remaining_numeric = df.select_dtypes(include=[np.number]).columns
        if not remaining_numeric.empty:
            df[remaining_numeric] = df[remaining_numeric].fillna(0)

        final_shape = df.shape
        if original_shape != final_shape:
            logger.info(f"Feature cleaning: shape changed from {original_shape} to {final_shape}")

        return df

    # Private helper methods
    async def _clean_dataframe(
        self, data: pd.DataFrame, context: IValidationContext, in_place: bool = False
    ) -> pd.DataFrame:
        """Clean DataFrame based on context."""
        df = data if in_place else data.copy()

        if context.data_type == DataType.FEATURES:
            return await self.clean_features(df, context=context)
        else:
            return await self.standardize_dataframe(df, context.symbol, context)

    async def _clean_record(
        self, record: dict[str, Any], context: IValidationContext
    ) -> dict[str, Any]:
        """Clean individual record."""
        cleaned = record.copy()

        # Apply field mappings
        if context.data_type in self.field_mappings:
            mappings = self.field_mappings[context.data_type]
            for standard_field, source_field in mappings.items():
                if source_field in cleaned and standard_field not in cleaned:
                    cleaned[standard_field] = cleaned.pop(source_field)

        # Handle common field variations
        cleaned = await self._handle_field_variations(cleaned, context)

        # Apply default values
        if context.data_type in self.default_values:
            defaults = self.default_values[context.data_type]
            for field, default_val in defaults.items():
                if field not in cleaned or cleaned[field] is None:
                    cleaned[field] = default_val

        # Normalize field values
        cleaned = await self._normalize_record(cleaned, context)

        # Filter to allowed fields
        if context.data_type in self.allowed_fields:
            allowed = self.allowed_fields[context.data_type]
            cleaned = {k: v for k, v in cleaned.items() if k in allowed}

        return cleaned

    async def _clean_record_list(
        self, records: list[dict[str, Any]], context: IValidationContext
    ) -> list[dict[str, Any]]:
        """Clean list of records."""
        cleaned_records = []
        for record in records:
            cleaned = await self._clean_record(record, context)
            cleaned_records.append(cleaned)
        return cleaned_records

    async def _ensure_datetime_index(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> pd.DataFrame:
        """Ensure DataFrame has proper datetime index."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data

        # Look for datetime columns
        date_columns = ["timestamp", "date", "datetime", "time"]
        date_col_found = None

        for col in date_columns:
            if col in data.columns:
                date_col_found = col
                break

        if date_col_found:
            try:
                data = data.set_index(date_col_found)
                data.index = pd.to_datetime(data.index, utc=True)
            except Exception as e:
                logger.warning(
                    f"Failed to set datetime index using '{date_col_found}' for {symbol}: {e}"
                )
        else:
            # Try to convert existing index
            try:
                data.index = pd.to_datetime(data.index, utc=True)
            except Exception as e:
                logger.warning(f"Failed to convert index to datetime for {symbol}: {e}")

        return data

    async def _find_constant_columns(self, data: pd.DataFrame) -> list[str]:
        """Find columns with constant values."""
        constant_cols = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if data[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)

        return constant_cols

    async def _handle_field_variations(
        self, record: dict[str, Any], context: IValidationContext
    ) -> dict[str, Any]:
        """Handle common field name variations."""
        cleaned = record.copy()

        # Common variations for market data
        if context.data_type == DataType.MARKET_DATA:
            # Handle transaction/trades variations
            if "transactions" in cleaned and "trades" not in cleaned:
                cleaned["trades"] = cleaned.pop("transactions")
            elif "n" in cleaned and "trades" not in cleaned:
                cleaned["trades"] = cleaned.pop("n")

        return cleaned

    async def _normalize_dataframe(
        self, data: pd.DataFrame, context: IValidationContext
    ) -> pd.DataFrame:
        """Normalize DataFrame values."""
        df = data.copy()

        # Normalize datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_cols:
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
            else:
                df[col] = df[col].dt.tz_convert("UTC")

        # Normalize numeric columns (remove extreme outliers if configured)
        if self.config.get("normalize_outliers", False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].std() > 0:
                    # Use IQR method to cap extreme outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    async def _normalize_record(
        self, record: dict[str, Any], context: IValidationContext
    ) -> dict[str, Any]:
        """Normalize individual record values."""
        cleaned = record.copy()

        # Normalize string fields based on length limits
        for key, value in cleaned.items():
            if isinstance(value, str) and key in self.column_lengths:
                max_len = self.column_lengths[key]
                if max_len:
                    cleaned[key] = value.strip()[:max_len]
                    if not cleaned[key]:
                        cleaned[key] = None

        # Normalize datetime fields
        datetime_fields = ["timestamp", "published_date", "date", "datetime"]
        for field in datetime_fields:
            if field in cleaned and isinstance(cleaned[field], (str, date)):
                try:
                    cleaned[field] = pd.to_datetime(cleaned[field], utc=True)
                except Exception as e:
                    logger.warning(f"Failed to normalize datetime field '{field}': {e}")

        return cleaned

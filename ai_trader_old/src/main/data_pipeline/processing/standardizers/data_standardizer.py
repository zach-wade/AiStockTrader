"""
Data Standardizer

Wrapper around utils for data standardization operations.
"""

# Standard library imports
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.processing import IDataStandardizer
from main.utils.core import ErrorHandlingMixin, ensure_utc, get_logger
from main.utils.data import get_global_processor
from main.utils.monitoring import MetricType, record_metric, timer


class DataStandardizer(IDataStandardizer, ErrorHandlingMixin):
    """
    Data standardizer using utils for all operations.

    Handles column standardization, timestamp normalization,
    and symbol formatting with layer-aware rules.
    """

    # Standard column mappings for different sources
    COLUMN_MAPPINGS = {
        "polygon": {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
            "vw": "vwap",
            "n": "trades",
        },
        "alpaca": {
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "volume": "volume",
            "timestamp": "timestamp",
        },
        "yahoo": {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Date": "timestamp",
        },
    }

    def __init__(self):
        """Initialize standardizer with utils."""
        self.logger = get_logger(__name__)
        self.processor = get_global_processor()

    async def standardize(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        source: str,
        layer: DataLayer,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Standardize data according to layer-specific rules.

        Args:
            data: Input DataFrame
            data_type: Type of data
            source: Data source name
            layer: Data layer for rules
            context: Optional context

        Returns:
            Standardized DataFrame
        """
        with timer(
            "standardize.process",
            tags={
                "source": source,
                "layer": layer.name,
                "data_type": data_type.value if hasattr(data_type, "value") else str(data_type),
            },
        ):
            if data.empty:
                return data

            df = data.copy()

            # Apply standardization steps
            df = await self.standardize_columns(df, source, layer)
            df = await self.standardize_timestamps(df, layer)
            df = await self.standardize_symbols(df, layer)

            # Layer-specific standardization
            if layer >= DataLayer.CATALYST:
                df = await self._apply_strict_standardization(df, data_type)

            record_metric(
                "standardize.rows",
                len(df),
                MetricType.COUNTER,
                tags={"source": source, "layer": layer.name},
            )

            return df

    async def standardize_columns(
        self, data: pd.DataFrame, source: str, layer: DataLayer
    ) -> pd.DataFrame:
        """Standardize column names and types."""
        df = data.copy()

        # Get column mapping for source
        mapping = self.COLUMN_MAPPINGS.get(source.lower(), {})

        if mapping:
            # Rename columns based on mapping
            df = df.rename(columns=mapping)
            self.logger.debug(f"Applied column mapping for {source}")

        # Standardize column names (lowercase, replace spaces)
        df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

        # Ensure numeric columns are properly typed
        numeric_columns = ["open", "high", "low", "close", "volume", "vwap"]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to numeric: {e}")

        return df

    async def standardize_timestamps(self, data: pd.DataFrame, layer: DataLayer) -> pd.DataFrame:
        """Standardize timestamp formats."""
        df = data.copy()

        # Find timestamp column
        timestamp_cols = ["timestamp", "date", "datetime", "time"]
        timestamp_col = None

        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            # Convert to datetime
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                # Ensure UTC
                df[timestamp_col] = df[timestamp_col].apply(ensure_utc)

                # Rename to standard 'timestamp' if different
                if timestamp_col != "timestamp":
                    df = df.rename(columns={timestamp_col: "timestamp"})

                self.logger.debug(f"Standardized timestamp column from {timestamp_col}")
            except Exception as e:
                self.logger.error(f"Failed to standardize timestamps: {e}")

        # For higher layers, ensure timestamp index
        if layer >= DataLayer.LIQUID and "timestamp" in df.columns:
            df = df.set_index("timestamp", drop=False)
            df = df.sort_index()

        return df

    async def standardize_symbols(self, data: pd.DataFrame, layer: DataLayer) -> pd.DataFrame:
        """Standardize symbol formats."""
        df = data.copy()

        # Find symbol column
        symbol_cols = ["symbol", "ticker", "stock", "sym"]
        symbol_col = None

        for col in symbol_cols:
            if col in df.columns:
                symbol_col = col
                break

        if symbol_col:
            # Standardize to uppercase
            df[symbol_col] = df[symbol_col].str.upper().str.strip()

            # Remove invalid characters for higher layers
            if layer >= DataLayer.CATALYST:
                df[symbol_col] = df[symbol_col].str.replace(r"[^A-Z0-9\-\.]", "", regex=True)

            # Rename to standard 'symbol' if different
            if symbol_col != "symbol":
                df = df.rename(columns={symbol_col: "symbol"})

            self.logger.debug(f"Standardized symbol column from {symbol_col}")

        return df

    async def get_standardization_rules(self, layer: DataLayer) -> dict[str, Any]:
        """Get standardization rules for a layer."""
        rules = {
            "column_naming": "lowercase_underscore",
            "timestamp_format": "utc",
            "symbol_format": "uppercase",
            "numeric_precision": 4 if layer <= DataLayer.LIQUID else 6,
            "null_handling": "forward_fill" if layer >= DataLayer.LIQUID else "drop",
            "duplicate_handling": "keep_last",
            "validation_level": "strict" if layer >= DataLayer.CATALYST else "normal",
        }

        return rules

    async def _apply_strict_standardization(
        self, df: pd.DataFrame, data_type: DataType
    ) -> pd.DataFrame:
        """Apply strict standardization for higher layers."""
        # Remove any rows with critical missing data
        if data_type == DataType.MARKET_DATA:
            critical_cols = ["open", "high", "low", "close", "volume"]
            existing_critical = [col for col in critical_cols if col in df.columns]
            if existing_critical:
                before_len = len(df)
                df = df.dropna(subset=existing_critical)
                if len(df) < before_len:
                    self.logger.info(
                        f"Dropped {before_len - len(df)} rows with missing critical data"
                    )

        # Round numeric values for consistency
        numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
        for col in numeric_cols:
            if col in ["open", "high", "low", "close", "vwap"]:
                df[col] = df[col].round(4)
            elif col in ["volume", "trades"]:
                df[col] = df[col].round(0).astype("int64", errors="ignore")

        return df

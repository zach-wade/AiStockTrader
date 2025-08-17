"""
Market Data Validator - Interface Implementation

Specialized validator for market data validation.
Implements IMarketDataValidator interface for OHLCV data validation.
"""

# Standard library imports
from datetime import datetime, time
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
# Config imports
from main.config.validation_models import DataPipelineConfig

# Core imports
from main.data_pipeline.core.enums import DataType
from main.data_pipeline.validation.core.validation_types import ValidationResult

# Interface imports
from main.interfaces.validation import IValidationContext, IValidationResult, ValidationStage
from main.utils.core import get_logger

logger = get_logger(__name__)


class MarketDataValidator:
    """
    Market data validator implementation.

    Implements IMarketDataValidator interface for comprehensive
    OHLCV market data validation.
    """

    def __init__(self, config: dict[str, Any] | DataPipelineConfig.ValidationConfig | None = None):
        """
        Initialize the market data validator.

        Args:
            config: ValidationConfig from main.config or dict for backward compatibility
        """
        # Handle both ValidationConfig and dict for backward compatibility
        if config is None:
            config = DataPipelineConfig.ValidationConfig()

        if isinstance(config, DataPipelineConfig.ValidationConfig):
            # Use typed config from main.config
            self.validation_config = config
            self.required_ohlcv_fields = config.market_data.required_ohlcv_fields
            self.max_price_deviation = config.market_data.max_price_deviation
            self.allow_zero_volume = config.market_data.allow_zero_volume
            self.allow_weekend_trading = config.market_data.allow_weekend_trading
            self.allow_future_timestamps = config.market_data.allow_future_timestamps
            self.max_nan_ratio = config.quality_thresholds.max_nan_ratio
            self.min_quality_score = config.quality_thresholds.min_quality_score
        else:
            # Backward compatibility with dict config
            self.validation_config = None
            self.required_ohlcv_fields = config.get(
                "required_ohlcv_fields", ["open", "high", "low", "close", "volume"]
            )
            self.max_price_deviation = config.get("max_price_deviation", 0.5)
            self.allow_zero_volume = config.get("allow_zero_volume", True)
            self.allow_weekend_trading = config.get("allow_weekend_trading", False)
            self.allow_future_timestamps = config.get("allow_future_timestamps", False)
            self.max_nan_ratio = config.get("max_nan_ratio", 0.1)
            self.min_quality_score = config.get("min_quality_score", 80.0)

        self.price_precision = 4  # Standard precision for price validation
        logger.info("Initialized MarketDataValidator with main.config integration")

    # IValidator interface methods
    async def validate(self, data: Any, context: IValidationContext) -> IValidationResult:
        """Validate data with given context."""
        return await self.validate_ohlcv_data(data, context.symbol, context)

    async def get_validation_rules(self, context: IValidationContext) -> list[str]:
        """Get applicable validation rules for context."""
        return [
            "ohlcv_field_presence",
            "price_consistency",
            "volume_validation",
            "trading_hours_check",
            "data_continuity",
            "price_range_validation",
        ]

    async def is_applicable(self, context: IValidationContext) -> bool:
        """Check if validator applies to given context."""
        return context.data_type in [DataType.MARKET_DATA, DataType.FINANCIALS]

    # IMarketDataValidator interface methods
    async def validate_ohlcv_data(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        symbol: str | None = None,
        context: IValidationContext | None = None,
    ) -> IValidationResult:
        """Validate OHLCV market data."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                if not data:
                    errors.append("Empty market data list")
                    return self._create_result(
                        ValidationStage.INGEST, False, errors, warnings, metrics, start_time
                    )
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                errors.append(f"Unsupported data type for OHLCV validation: {type(data)}")
                return self._create_result(
                    ValidationStage.INGEST, False, errors, warnings, metrics, start_time
                )

            if df.empty:
                errors.append("Empty market data DataFrame")
                return self._create_result(
                    ValidationStage.INGEST, False, errors, warnings, metrics, start_time
                )

            metrics["total_records"] = len(df)
            metrics["symbol"] = symbol

            # Validate required fields
            missing_fields = [
                field for field in self.required_ohlcv_fields if field not in df.columns
            ]
            if missing_fields:
                errors.append(f"Missing required OHLCV fields: {missing_fields}")

            # Validate each record for price consistency
            price_passed, price_errors = await self.validate_price_consistency(df, context)
            errors.extend(price_errors)

            # Validate volume data
            volume_passed, volume_errors = await self.validate_volume_data(df, context)
            errors.extend(volume_errors)
            warnings.extend([e for e in volume_errors if "zero volume" in e.lower()])

            # Validate trading hours if configured
            if self.trading_hours:
                hours_passed, hours_errors = await self.validate_trading_hours(df, context)
                warnings.extend(hours_errors)  # Trading hours violations are warnings

            # Data quality metrics
            quality_metrics = self._calculate_quality_metrics(df)
            metrics.update(quality_metrics)

            # Overall quality assessment
            quality_score = self._calculate_quality_score(df, len(errors), len(warnings))
            metrics["quality_score"] = quality_score

        except Exception as e:
            logger.error(f"OHLCV validation error: {e}", exc_info=True)
            errors.append(f"OHLCV validation error: {e!s}")

        passed = len(errors) == 0
        stage = context.stage if context else ValidationStage.INGEST
        return self._create_result(stage, passed, errors, warnings, metrics, start_time)

    async def validate_price_consistency(
        self, data: pd.DataFrame, context: IValidationContext = None
    ) -> tuple[bool, list[str]]:
        """Validate price consistency (OHLC relationships)."""
        errors = []

        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing OHLC columns for price consistency check: {missing_cols}")
            return False, errors

        # Check OHLC relationships for each row
        for idx, row in data.iterrows():
            row_errors = self._validate_ohlc_relationships(row.to_dict(), idx)
            errors.extend(row_errors)

        return len(errors) == 0, errors

    async def validate_volume_data(
        self, data: pd.DataFrame, context: IValidationContext = None
    ) -> tuple[bool, list[str]]:
        """Validate volume data."""
        errors = []

        if "volume" not in data.columns:
            errors.append("Volume column missing from market data")
            return False, errors

        # Check for negative volumes
        negative_mask = data["volume"] < 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            errors.append(f"Found {negative_count} records with negative volume")

        # Check for null volumes
        null_mask = data["volume"].isnull()
        null_count = null_mask.sum()
        if null_count > 0:
            errors.append(f"Found {null_count} records with null volume")

        # Check for zero volumes (warning if not allowed)
        if not self.allow_zero_volume:
            zero_mask = data["volume"] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                errors.append(f"Found {zero_count} records with zero volume (not allowed)")

        # Volume data type validation
        if data["volume"].dtype not in ["int64", "float64"]:
            errors.append(f"Volume column has invalid data type: {data['volume'].dtype}")

        return len(errors) == 0, errors

    async def validate_trading_hours(
        self, data: pd.DataFrame, context: IValidationContext = None
    ) -> tuple[bool, list[str]]:
        """Validate data falls within trading hours."""
        warnings = []

        if not self.trading_hours:
            return True, []

        # Check if we have timestamp information
        timestamp_col = None
        if "timestamp" in data.columns:
            timestamp_col = "timestamp"
        elif isinstance(data.index, pd.DatetimeIndex):
            timestamp_col = data.index
        else:
            warnings.append("No timestamp information found for trading hours validation")
            return True, warnings

        try:
            # Get timestamps
            if timestamp_col == data.index:
                timestamps = data.index
            else:
                timestamps = pd.to_datetime(data[timestamp_col])

            # Extract trading hours config
            market_open = time.fromisoformat(self.trading_hours.get("market_open", "09:30:00"))
            market_close = time.fromisoformat(self.trading_hours.get("market_close", "16:00:00"))

            # Check each timestamp
            outside_hours_count = 0
            for ts in timestamps:
                ts_time = ts.time()
                if not (market_open <= ts_time <= market_close):
                    outside_hours_count += 1

            if outside_hours_count > 0:
                warnings.append(
                    f"Found {outside_hours_count} records outside trading hours "
                    f"({market_open} - {market_close})"
                )

        except Exception as e:
            warnings.append(f"Trading hours validation error: {e!s}")

        return True, warnings

    async def validate_corporate_actions(
        self,
        market_data: pd.DataFrame,
        corporate_actions: list[dict[str, Any]],
        context: IValidationContext = None,
    ) -> tuple[bool, list[str]]:
        """Validate market data against corporate actions."""
        warnings = []

        if not corporate_actions:
            return True, []

        try:
            # Check for price adjustments around corporate action dates
            for action in corporate_actions:
                action_date = pd.to_datetime(action.get("date"))
                action_type = action.get("type", "")

                if action_type.lower() in ["split", "dividend"]:
                    # Look for data around the corporate action date
                    date_range = pd.date_range(
                        start=action_date - pd.Timedelta(days=2),
                        end=action_date + pd.Timedelta(days=2),
                    )

                    if isinstance(market_data.index, pd.DatetimeIndex):
                        relevant_data = market_data[market_data.index.isin(date_range)]
                    elif "timestamp" in market_data.columns:
                        ts_col = pd.to_datetime(market_data["timestamp"])
                        relevant_data = market_data[ts_col.isin(date_range)]
                    else:
                        continue

                    if len(relevant_data) > 1:
                        # Check for unusual price movements
                        price_changes = relevant_data["close"].pct_change().abs()
                        if price_changes.max() > 0.5:  # 50% change threshold
                            warnings.append(
                                f"Large price movement around {action_type} on {action_date.date()}"
                            )

        except Exception as e:
            warnings.append(f"Corporate actions validation error: {e!s}")

        return True, warnings

    # Helper methods
    def _validate_ohlc_relationships(
        self, record: dict[str, Any], row_index: int | None = None
    ) -> list[str]:
        """Validate OHLC price relationships for a single record."""
        errors = []
        row_prefix = f"Row {row_index}: " if row_index is not None else ""

        try:
            ohlc_fields = ["open", "high", "low", "close"]
            if not all(field in record for field in ohlc_fields):
                return []  # Skip if not all OHLC fields present

            # Convert to float and handle NaN
            o = float(record["open"]) if pd.notna(record["open"]) else np.nan
            h = float(record["high"]) if pd.notna(record["high"]) else np.nan
            l = float(record["low"]) if pd.notna(record["low"]) else np.nan
            c = float(record["close"]) if pd.notna(record["close"]) else np.nan

            if any(pd.isna([o, h, l, c])):
                errors.append(f"{row_prefix}OHLC validation skipped due to NaN values")
                return errors

            # Validate all values are positive
            for price, name in [(o, "open"), (h, "high"), (l, "low"), (c, "close")]:
                if price <= 0:
                    errors.append(f"{row_prefix}Non-positive {name} price: {price}")

            # Basic OHLC relationship validation
            if h < l:
                errors.append(f"{row_prefix}High ({h}) < Low ({l})")
            if h < o:
                errors.append(f"{row_prefix}High ({h}) < Open ({o})")
            if h < c:
                errors.append(f"{row_prefix}High ({h}) < Close ({c})")
            if l > o:
                errors.append(f"{row_prefix}Low ({l}) > Open ({o})")
            if l > c:
                errors.append(f"{row_prefix}Low ({l}) > Close ({c})")

            # Check for extreme price deviations
            prices = [o, h, l, c]
            median_price = np.median(prices)

            if median_price > 0:
                for price, name in zip(prices, ohlc_fields):
                    deviation = abs(price - median_price) / median_price
                    if deviation > self.max_price_deviation:
                        errors.append(
                            f"{row_prefix}{name.capitalize()} price deviates {deviation:.1%} from median "
                            f"(max allowed: {self.max_price_deviation:.1%})"
                        )

            # Check price precision
            for price, name in zip(prices, ohlc_fields):
                if not self._check_price_precision(price):
                    errors.append(
                        f"{row_prefix}{name.capitalize()} price has excessive precision: {price}"
                    )

        except Exception as e:
            errors.append(f"{row_prefix}Error validating OHLC relationships: {e}")
            logger.warning(f"OHLC validation error for row {row_index}: {e}", exc_info=True)

        return errors

    def _check_price_precision(self, price: float) -> bool:
        """Check if price has reasonable precision."""
        if pd.isna(price) or price == 0:
            return True

        # Count decimal places
        decimal_str = str(price).split(".")
        if len(decimal_str) == 1:
            return True  # No decimal places

        decimal_places = len(decimal_str[1].rstrip("0"))
        return decimal_places <= self.price_precision

    def _calculate_quality_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate data quality metrics for market data."""
        metrics = {}

        # Data completeness
        total_values = data.size
        null_values = data.isnull().sum().sum()
        completeness = 1 - (null_values / total_values) if total_values > 0 else 0
        metrics["completeness"] = completeness

        # OHLCV field coverage
        ohlcv_present = sum(1 for field in self.required_ohlcv_fields if field in data.columns)
        field_coverage = ohlcv_present / len(self.required_ohlcv_fields)
        metrics["field_coverage"] = field_coverage

        # Volume statistics
        if "volume" in data.columns:
            volume_data = data["volume"].dropna()
            if not volume_data.empty:
                metrics["avg_volume"] = volume_data.mean()
                metrics["zero_volume_count"] = (volume_data == 0).sum()
                metrics["negative_volume_count"] = (volume_data < 0).sum()

        # Price statistics
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            price_data = data[["open", "high", "low", "close"]].dropna()
            if not price_data.empty:
                metrics["avg_price_range"] = (price_data["high"] - price_data["low"]).mean()
                metrics["price_volatility"] = price_data["close"].pct_change().std()

        return metrics

    def _calculate_quality_score(
        self, data: pd.DataFrame, error_count: int, warning_count: int
    ) -> float:
        """Calculate overall quality score."""
        base_score = 100.0

        # Penalize errors more than warnings
        error_penalty = min(error_count * 10, 70)
        warning_penalty = min(warning_count * 2, 20)

        # Penalize missing data
        if not data.empty:
            null_ratio = data.isnull().sum().sum() / data.size
            null_penalty = null_ratio * 10
        else:
            null_penalty = 100  # Empty data gets 0 score

        final_score = base_score - error_penalty - warning_penalty - null_penalty
        return max(final_score, 0.0)

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

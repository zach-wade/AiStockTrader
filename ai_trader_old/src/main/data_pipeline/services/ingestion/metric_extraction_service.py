"""
Metric Extraction Service

Service for extracting and validating financial metrics from various data formats.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class MetricExtractionConfig:
    """Configuration for metric extraction."""

    max_eps_value: float = 999999.0  # Database constraint: NUMERIC(10,4)
    warn_eps_threshold: float = 10000.0  # Warn for suspiciously high EPS
    default_value: Any | None = None
    strict_validation: bool = False


class MetricExtractionService:
    """
    Service for extracting financial metrics from various data formats.

    Handles metric name mapping, type conversion, validation, and
    database constraint compliance.
    """

    def __init__(self, config: MetricExtractionConfig | None = None):
        """
        Initialize the metric extraction service.

        Args:
            config: Service configuration
        """
        self.config = config or MetricExtractionConfig()

        # Define metric name mappings for each metric type
        self._revenue_keys = [
            "Total Revenue",
            "Operating Revenue",
            "Revenue",
            "Revenues",
            "total_revenue",
            "operating_revenue",
            "revenue",
            "revenues",
        ]

        self._net_income_keys = [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
            "net_income",
            "net_income_common_stockholders",
        ]

        self._total_assets_keys = [
            "Total Assets",
            "Total Non Current Assets",
            "Current Assets",
            "total_assets",
            "current_assets",
        ]

        self._total_liabilities_keys = [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities",
            "Total Liabilities And Stockholders Equity",
            "total_liabilities",
            "total_liabilities_net_minority_interest",
        ]

        self._operating_cash_flow_keys = [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Cash Flow From Operating Activities",
            "operating_cash_flow",
            "cash_flow_from_operating_activities",
        ]

        self._gross_profit_keys = ["Gross Profit", "gross_profit"]

        self._operating_income_keys = [
            "Operating Income",
            "EBIT",
            "Operating Income Loss",
            "operating_income",
            "operating_income_loss",
            "ebit",
        ]

        self._eps_basic_keys = [
            "Basic EPS",
            "Basic Earnings Per Share",
            "EPS Basic",
            "basic_eps",
            "eps_basic",
            "basic_earnings_per_share",
        ]

        self._eps_diluted_keys = [
            "Diluted EPS",
            "Diluted Earnings Per Share",
            "EPS Diluted",
            "diluted_eps",
            "eps_diluted",
            "diluted_earnings_per_share",
        ]

        self._current_assets_keys = ["Current Assets", "current_assets"]

        self._current_liabilities_keys = ["Current Liabilities", "current_liabilities"]

        self._stockholders_equity_keys = [
            "Stockholders Equity",
            "Total Stockholders Equity",
            "Total Equity",
            "stockholders_equity",
            "total_stockholders_equity",
            "total_equity",
        ]

    def extract_revenue(self, data: dict[str, Any]) -> int | None:
        """Extract revenue metric."""
        return self._extract_int_metric(data, self._revenue_keys)

    def extract_net_income(self, data: dict[str, Any]) -> int | None:
        """Extract net income metric."""
        return self._extract_int_metric(data, self._net_income_keys)

    def extract_total_assets(self, data: dict[str, Any]) -> int | None:
        """Extract total assets metric."""
        return self._extract_int_metric(data, self._total_assets_keys)

    def extract_total_liabilities(self, data: dict[str, Any]) -> int | None:
        """Extract total liabilities metric."""
        return self._extract_int_metric(data, self._total_liabilities_keys)

    def extract_operating_cash_flow(self, data: dict[str, Any]) -> int | None:
        """Extract operating cash flow metric."""
        return self._extract_int_metric(data, self._operating_cash_flow_keys)

    def extract_gross_profit(self, data: dict[str, Any]) -> int | None:
        """Extract gross profit metric."""
        return self._extract_int_metric(data, self._gross_profit_keys)

    def extract_operating_income(self, data: dict[str, Any]) -> int | None:
        """Extract operating income metric."""
        return self._extract_int_metric(data, self._operating_income_keys)

    def extract_current_assets(self, data: dict[str, Any]) -> int | None:
        """Extract current assets metric."""
        return self._extract_int_metric(data, self._current_assets_keys)

    def extract_current_liabilities(self, data: dict[str, Any]) -> int | None:
        """Extract current liabilities metric."""
        return self._extract_int_metric(data, self._current_liabilities_keys)

    def extract_stockholders_equity(self, data: dict[str, Any]) -> int | None:
        """Extract stockholders equity metric."""
        return self._extract_int_metric(data, self._stockholders_equity_keys)

    def extract_eps_basic(self, data: dict[str, Any]) -> float | None:
        """Extract basic EPS metric with validation."""
        return self._extract_eps_metric(data, self._eps_basic_keys)

    def extract_eps_diluted(self, data: dict[str, Any]) -> float | None:
        """Extract diluted EPS metric with validation."""
        return self._extract_eps_metric(data, self._eps_diluted_keys)

    def _extract_int_metric(self, data: dict[str, Any], possible_keys: list[str]) -> int | None:
        """
        Extract an integer metric value using possible key names.

        Args:
            data: Data dictionary to extract from
            possible_keys: List of possible key names to try

        Returns:
            Extracted integer value or None
        """
        if not data:
            return self.config.default_value

        for key in possible_keys:
            if key in data and data[key] is not None:
                try:
                    value = data[key]

                    # Handle various input types
                    if isinstance(value, str):
                        # Remove decimal part if it's .0
                        if value.endswith(".0"):
                            return int(float(value))
                        else:
                            return int(float(value))
                    elif hasattr(value, "item"):  # numpy scalar
                        return int(value.item())
                    else:
                        return int(float(value))

                except (ValueError, TypeError) as e:
                    if self.config.strict_validation:
                        logger.warning(f"Failed to extract int metric for key '{key}': {e}")
                    continue

        return self.config.default_value

    def _extract_float_metric(self, data: dict[str, Any], possible_keys: list[str]) -> float | None:
        """
        Extract a float metric value using possible key names.

        Args:
            data: Data dictionary to extract from
            possible_keys: List of possible key names to try

        Returns:
            Extracted float value or None
        """
        if not data:
            return self.config.default_value

        for key in possible_keys:
            if key in data and data[key] is not None:
                try:
                    value = data[key]

                    # Handle various input types
                    if isinstance(value, str):
                        return float(value)
                    elif hasattr(value, "item"):  # numpy scalar
                        return float(value.item())
                    else:
                        return float(value)

                except (ValueError, TypeError) as e:
                    if self.config.strict_validation:
                        logger.warning(f"Failed to extract float metric for key '{key}': {e}")
                    continue

        return self.config.default_value

    def _extract_eps_metric(self, data: dict[str, Any], possible_keys: list[str]) -> float | None:
        """
        Extract and validate EPS metric to prevent database overflow.

        EPS values must fit in NUMERIC(10,4) which allows max 999999.9999.
        Realistically, EPS should be between -1000 and 1000.

        Args:
            data: Data dictionary to extract from
            possible_keys: List of possible key names to try

        Returns:
            Validated EPS value or None
        """
        value = self._extract_float_metric(data, possible_keys)

        if value is not None:
            # Check for unrealistic EPS values that would cause overflow
            if abs(value) >= self.config.max_eps_value:
                logger.warning(
                    f"Detected unrealistic EPS value {value} for keys {possible_keys}. "
                    f"Setting to None to prevent database overflow."
                )
                return None

            # Warn for suspiciously high but valid values
            elif abs(value) > self.config.warn_eps_threshold:
                logger.warning(
                    f"Detected suspiciously high EPS value {value} for keys {possible_keys}. "
                    f"This may indicate bad data from the API."
                )

        return value

    def validate_metrics(self, record: dict[str, Any]) -> bool:
        """
        Validate that a financial record has minimum required metrics.

        Args:
            record: Financial record to validate

        Returns:
            True if record has minimum required data
        """
        # At minimum, we need symbol, year, and period
        if not record.get("symbol") or not record.get("year") or not record.get("period"):
            return False

        # Should have at least one financial metric
        financial_metrics = [
            "revenue",
            "net_income",
            "total_assets",
            "total_liabilities",
            "operating_cash_flow",
            "gross_profit",
            "operating_income",
            "eps_basic",
            "eps_diluted",
            "current_assets",
            "current_liabilities",
            "stockholders_equity",
        ]

        has_metric = any(record.get(metric) is not None for metric in financial_metrics)

        if not has_metric and self.config.strict_validation:
            logger.warning(
                f"Record for {record.get('symbol')} {record.get('year')} {record.get('period')} has no financial metrics"
            )

        return has_metric or not self.config.strict_validation

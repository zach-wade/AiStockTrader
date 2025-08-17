"""
Yahoo Format Handler

Handles financial data in Yahoo Finance format.
"""

# Standard library imports
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base import BaseFormatHandler

logger = get_logger(__name__)


class YahooFormatHandler(BaseFormatHandler):
    """
    Handler for Yahoo Finance financial data format.

    Yahoo data typically comes in two formats:
    1. DataFrame with dates as columns and metrics as rows
    2. List of dicts with 'index' (metric name) and date columns
    """

    def can_handle(self, data: Any) -> bool:
        """
        Check if data is in Yahoo Finance format.

        Yahoo data characteristics:
        - DataFrame with dates as columns
        - List of dicts with 'index' field (metric name) and date columns
        - Wrapped in a single-item list with 'data' field

        Args:
            data: Raw data to check

        Returns:
            True if data appears to be in Yahoo format
        """
        # Check for wrapped format
        if isinstance(data, list) and len(data) == 1:
            if isinstance(data[0], dict) and "data" in data[0]:
                actual_data = data[0]["data"]
                if isinstance(actual_data, dict) and "data" in actual_data:
                    actual_data = actual_data["data"]
                # Check if it's a DataFrame
                if hasattr(actual_data, "to_dict"):
                    return True

        # Check for list of metric dicts format
        if isinstance(data, list) and len(data) > 0:
            # Check if items have 'index' field (metric name)
            sample = data[0] if data else {}
            if isinstance(sample, dict) and "index" in sample:
                # Check for date-like keys
                for key in sample.keys():
                    if key != "index":
                        try:
                            pd.to_datetime(key)
                            return True
                        except (ValueError, TypeError, pd.errors.ParserError):
                            pass

        # Check for direct DataFrame
        if hasattr(data, "columns") and hasattr(data, "index"):
            return True

        return False

    def process(
        self, data: list | pd.DataFrame | dict, symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Process Yahoo Finance data into standardized records.

        Args:
            data: Yahoo financial data (DataFrame or list format)
            symbols: List of symbols this data relates to
            source: Data source name (typically 'yahoo')

        Returns:
            List of standardized financial records
        """
        prepared_records = []

        logger.debug(f"Processing Yahoo financial data for symbols: {symbols}")

        # Unwrap data if necessary
        actual_data = self._unwrap_data(data)

        # Process based on data type
        if hasattr(actual_data, "columns"):
            # DataFrame format
            prepared_records = self._process_dataframe(actual_data, symbols, source)
        elif isinstance(actual_data, list):
            # List of metric dicts format
            prepared_records = self._process_metric_list(actual_data, symbols, source)
        else:
            logger.warning(f"Unexpected Yahoo data format: {type(actual_data)}")

        logger.info(f"Yahoo handler prepared {len(prepared_records)} records")

        return prepared_records

    def _unwrap_data(self, data: Any) -> Any:
        """
        Unwrap nested data structure from Yahoo Finance.

        Args:
            data: Potentially wrapped data

        Returns:
            Unwrapped data
        """
        # Check for wrapped format
        if isinstance(data, list) and len(data) == 1:
            if isinstance(data[0], dict) and "data" in data[0]:
                actual_data = data[0]["data"]
                if isinstance(actual_data, dict) and "data" in actual_data:
                    return actual_data["data"]
                return actual_data

        return data

    def _process_dataframe(
        self, df: pd.DataFrame, symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Process DataFrame format (dates as columns, metrics as rows).

        Args:
            df: DataFrame with financial data
            symbols: List of symbols
            source: Data source

        Returns:
            List of standardized records
        """
        prepared_records = []
        symbol = symbols[0] if symbols else "UNKNOWN"

        # Process each column (date)
        for col in df.columns:
            try:
                # Parse date
                if not isinstance(col, pd.Timestamp):
                    date = pd.to_datetime(col)
                else:
                    date = col

                year = date.year

                # Yahoo Finance typically provides annual data
                period = "FY"

                # Check for duplicates
                if self._should_skip_duplicate(symbol, year, period):
                    continue

                # Extract metrics for this date
                col_data = df[col].to_dict()

                # Create standardized record
                record = self._create_record(
                    symbol=symbol,
                    year=year,
                    period=period,
                    metrics=col_data,
                    source=source,
                    filing_date=date.date() if isinstance(date, pd.Timestamp) else None,
                )

                # Validate and add record
                if self.metric_extractor.validate_metrics(record):
                    prepared_records.append(record)
                    logger.debug(f"Added Yahoo DataFrame record for {symbol} {year} {period}")

            except Exception as e:
                logger.error(f"Error processing DataFrame column {col}: {e}")
                continue

        return prepared_records

    def _process_metric_list(
        self, data: list[dict[str, Any]], symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Process list of metric dictionaries format.

        Each dict has 'index' (metric name) and date columns.

        Args:
            data: List of metric dictionaries
            symbols: List of symbols
            source: Data source

        Returns:
            List of standardized records
        """
        prepared_records = []
        symbol = symbols[0] if symbols else "UNKNOWN"

        # First, collect all dates and build metrics by date
        all_dates = set()
        metrics_by_name = {}

        for item in data:
            if not isinstance(item, dict) or "index" not in item:
                continue

            metric_name = item["index"]
            metrics_by_name[metric_name] = item

            # Collect date keys
            for key in item.keys():
                if key != "index":
                    try:
                        pd.to_datetime(key)
                        all_dates.add(key)
                    except (ValueError, TypeError, pd.errors.ParserError):
                        pass

        logger.debug(f"Found {len(all_dates)} dates and {len(metrics_by_name)} metrics")

        # Create record for each date
        for date_str in sorted(all_dates):
            try:
                date = pd.to_datetime(date_str)
                year = date.year

                # Yahoo Finance typically provides annual data
                period = "FY"

                # Check for duplicates
                if self._should_skip_duplicate(symbol, year, period):
                    continue

                # Collect all metrics for this date
                date_metrics = {}
                for metric_name, metric_data in metrics_by_name.items():
                    if date_str in metric_data:
                        date_metrics[metric_name] = metric_data[date_str]

                # Create standardized record
                record = self._create_record(
                    symbol=symbol,
                    year=year,
                    period=period,
                    metrics=date_metrics,
                    source=source,
                    filing_date=date.date() if isinstance(date, pd.Timestamp) else None,
                )

                # Validate and add record
                if self.metric_extractor.validate_metrics(record):
                    prepared_records.append(record)
                    logger.debug(f"Added Yahoo list record for {symbol} {year} {period}")

            except Exception as e:
                logger.error(f"Error processing date {date_str}: {e}")
                continue

        return prepared_records

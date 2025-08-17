"""
Pre-processed Format Handler

Handles financial data that has already been processed and stored in archive.
"""

# Standard library imports
from datetime import UTC
from typing import Any

# Local imports
from main.utils.core import get_logger

from .base import BaseFormatHandler

logger = get_logger(__name__)


class PreProcessedFormatHandler(BaseFormatHandler):
    """
    Handler for pre-processed financial data from archives.

    Pre-processed data characteristics:
    - Already in standardized format with 'year' and 'period' fields
    - May have been processed from any original source
    - Minimal transformation needed
    """

    def can_handle(self, data: Any) -> bool:
        """
        Check if data is in pre-processed format.

        Pre-processed data characteristics:
        - List of dictionaries
        - Contains 'year' and 'period' fields
        - Already in standardized format

        Args:
            data: Raw data to check

        Returns:
            True if data appears to be pre-processed
        """
        if not isinstance(data, list) or len(data) == 0:
            return False

        # Check first few items for pre-processed fields
        sample_size = min(10, len(data))
        for item in data[:sample_size]:
            if isinstance(item, dict):
                # Pre-processed data has year and period directly
                if "year" in item and "period" in item:
                    return True

        return False

    def process(
        self, data: list[dict[str, Any]], symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Process pre-processed financial data.

        Since data is already in standardized format, minimal processing is needed.
        Main tasks are validation and duplicate checking.

        Args:
            data: List of pre-processed financial records
            symbols: List of symbols this data relates to
            source: Data source name

        Returns:
            List of standardized financial records
        """
        prepared_records = []
        records_processed = 0
        duplicates_skipped = 0
        invalid_skipped = 0

        logger.debug(
            f"Processing {len(data)} pre-processed financial records for symbols: {symbols}"
        )

        for item in data:
            try:
                if not isinstance(item, dict):
                    continue

                records_processed += 1

                # Validate required fields
                if "year" not in item or "period" not in item:
                    invalid_skipped += 1
                    logger.debug("Skipping item without year or period")
                    continue

                # Get or set symbol
                symbol = item.get("symbol")
                if not symbol:
                    symbol = symbols[0] if symbols else "UNKNOWN"
                    item["symbol"] = symbol

                # Extract key fields
                year = self._validate_year(item["year"])
                if not year:
                    invalid_skipped += 1
                    continue

                period = self._validate_period(item["period"])
                statement_type = item.get("statement_type", "income_statement")

                # Check for duplicates
                if self._should_skip_duplicate(symbol, year, period, statement_type):
                    duplicates_skipped += 1
                    continue

                # Update timestamps if not present
                if "created_at" not in item:
                    # Standard library imports
                    from datetime import datetime

                    item["created_at"] = datetime.now(UTC)
                if "updated_at" not in item:
                    # Standard library imports
                    from datetime import datetime

                    item["updated_at"] = datetime.now(UTC)

                # Update source if different
                if "source" not in item or item["source"] != source:
                    item["source"] = source

                # Clean raw data for JSON serialization
                if "raw_data" in item:
                    item["raw_data"] = self._clean_raw_data(item["raw_data"])

                # Validate metrics
                if self.metric_extractor.validate_metrics(item):
                    prepared_records.append(item)
                    logger.debug(f"Added pre-processed record for {symbol} {year} {period}")
                else:
                    invalid_skipped += 1
                    logger.debug(
                        f"Skipping pre-processed record with insufficient metrics: {symbol} {year} {period}"
                    )

            except Exception as e:
                logger.error(f"Error processing pre-processed record: {e}", exc_info=True)
                invalid_skipped += 1
                continue

        logger.info(
            f"Pre-processed handler processed {records_processed} records, "
            f"prepared {len(prepared_records)}, skipped {duplicates_skipped} duplicates, "
            f"{invalid_skipped} invalid"
        )

        return prepared_records

    def _validate_year(self, year_value: Any) -> int | None:
        """
        Validate and convert year to integer.

        Args:
            year_value: Year value in various formats

        Returns:
            Valid year as integer or None
        """
        try:
            year = int(year_value)
            # Validate reasonable year range
            if 1900 <= year <= 2100:
                return year
            else:
                logger.warning(f"Invalid year value: {year}")
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse year '{year_value}': {e}")
            return None

    def _validate_period(self, period_value: Any) -> str:
        """
        Validate and standardize period.

        Args:
            period_value: Period value

        Returns:
            Standardized period (FY, Q1, Q2, Q3, Q4)
        """
        if not period_value:
            return "FY"

        period = str(period_value).upper()

        # Map common variations
        period_map = {
            "FY": "FY",
            "Y": "FY",
            "ANNUAL": "FY",
            "Q1": "Q1",
            "1": "Q1",
            "Q2": "Q2",
            "2": "Q2",
            "Q3": "Q3",
            "3": "Q3",
            "Q4": "Q4",
            "4": "Q4",
        }

        return period_map.get(period, period if period in ["FY", "Q1", "Q2", "Q3", "Q4"] else "FY")

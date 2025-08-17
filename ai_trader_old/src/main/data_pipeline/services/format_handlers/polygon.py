"""
Polygon Format Handler

Handles financial data in Polygon.io format.
"""

# Standard library imports
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base import BaseFormatHandler

logger = get_logger(__name__)


class PolygonFormatHandler(BaseFormatHandler):
    """
    Handler for Polygon.io financial data format.

    Polygon data typically contains:
    - fiscal_period: Q1, Q2, Q3, Q4, or FY
    - fiscal_year: Year as integer
    - timeframe: 'annual' or 'quarterly'
    - filing_date: Date of filing
    - Various financial metrics with specific naming conventions
    """

    def can_handle(self, data: Any) -> bool:
        """
        Check if data is in Polygon format.

        Polygon data characteristics:
        - List of dictionaries
        - Contains 'fiscal_period' field
        - Contains 'fiscal_year' field

        Args:
            data: Raw data to check

        Returns:
            True if data appears to be in Polygon format
        """
        if not isinstance(data, list) or len(data) == 0:
            return False

        # Check first few items for Polygon-specific fields
        sample_size = min(3, len(data))
        for item in data[:sample_size]:
            if not isinstance(item, dict):
                return False
            if "fiscal_period" not in item and "fiscal_year" not in item:
                return False

        return True

    def process(
        self, data: list[dict[str, Any]], symbols: list[str], source: str
    ) -> list[dict[str, Any]]:
        """
        Process Polygon financial data into standardized records.

        Args:
            data: List of Polygon financial records
            symbols: List of symbols this data relates to
            source: Data source name (typically 'polygon')

        Returns:
            List of standardized financial records
        """
        prepared_records = []
        records_processed = 0
        duplicates_skipped = 0

        logger.debug(f"Processing {len(data)} Polygon financial records for symbols: {symbols}")

        for item in data:
            try:
                if not isinstance(item, dict):
                    continue

                records_processed += 1

                # Extract period information
                fiscal_year = self._extract_fiscal_year(item)
                if not fiscal_year:
                    logger.debug("Skipping item without valid fiscal_year")
                    continue

                fiscal_period = item.get("fiscal_period", "FY")
                timeframe = item.get("timeframe", "annual")

                # Determine standardized period
                period = self._determine_period(fiscal_period, timeframe)

                # Get symbol (use provided or from data)
                symbol = item.get("symbol", symbols[0] if symbols else "UNKNOWN")

                # Check for duplicates
                if self._should_skip_duplicate(symbol, fiscal_year, period):
                    duplicates_skipped += 1
                    continue

                # Extract filing date
                filing_date = self._extract_filing_date(item)

                # Create standardized record
                record = self._create_record(
                    symbol=symbol,
                    year=fiscal_year,
                    period=period,
                    metrics=item,
                    source=source,
                    filing_date=filing_date,
                )

                # Validate record has minimum data
                if self.metric_extractor.validate_metrics(record):
                    prepared_records.append(record)
                    logger.debug(f"Added Polygon record for {symbol} {fiscal_year} {period}")
                else:
                    logger.debug(
                        f"Skipping Polygon record with insufficient metrics: {symbol} {fiscal_year} {period}"
                    )

            except Exception as e:
                logger.error(f"Error processing Polygon financial record: {e}", exc_info=True)
                continue

        logger.info(
            f"Polygon handler processed {records_processed} records, "
            f"prepared {len(prepared_records)}, skipped {duplicates_skipped} duplicates"
        )

        return prepared_records

    def _extract_fiscal_year(self, item: dict[str, Any]) -> int | None:
        """
        Extract and validate fiscal year from Polygon data.

        Args:
            item: Polygon data item

        Returns:
            Fiscal year as integer or None
        """
        fiscal_year = item.get("fiscal_year")
        if fiscal_year is None:
            return None

        try:
            year = int(fiscal_year)
            # Validate reasonable year range
            if 1900 <= year <= 2100:
                return year
            else:
                logger.warning(f"Invalid fiscal year: {year}")
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse fiscal year '{fiscal_year}': {e}")
            return None

    def _determine_period(self, fiscal_period: str, timeframe: str) -> str:
        """
        Determine standardized period from Polygon data.

        Args:
            fiscal_period: Polygon fiscal period (Q1, Q2, Q3, Q4, FY, etc.)
            timeframe: Polygon timeframe ('annual' or 'quarterly')

        Returns:
            Standardized period (FY, Q1, Q2, Q3, Q4)
        """
        if timeframe == "annual":
            return "FY"
        else:
            # Standardize quarterly periods
            period = fiscal_period.upper() if fiscal_period else "Q1"
            # Ensure it's in valid format
            if period in ["Q1", "Q2", "Q3", "Q4"]:
                return period
            elif period == "FY":
                return "FY"
            else:
                logger.debug(f"Unknown period '{period}', defaulting to Q1")
                return "Q1"

    def _extract_filing_date(self, item: dict[str, Any]) -> Any | None:
        """
        Extract filing date from Polygon data.

        Args:
            item: Polygon data item

        Returns:
            Filing date or None
        """
        filing_date = item.get("filing_date")
        if not filing_date:
            # Try alternative field names
            filing_date = item.get("end_date") or item.get("period_end_date")

        if filing_date:
            try:
                return pd.to_datetime(filing_date).date()
            except Exception as e:
                logger.debug(f"Failed to parse filing date '{filing_date}': {e}")

        return None

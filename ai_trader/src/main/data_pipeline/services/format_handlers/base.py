"""
Base Format Handler

Abstract base class for financial data format handlers using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class FormatHandlerConfig:
    """Configuration for format handlers."""
    max_records: Optional[int] = None
    skip_duplicates: bool = True
    validate_data: bool = True


class BaseFormatHandler(ABC):
    """
    Abstract base class for handling different financial data formats.
    
    Implements the Strategy pattern for processing various data sources
    (Polygon, Yahoo, pre-processed, etc.) into a standardized format.
    """
    
    def __init__(
        self,
        metric_extractor: Any,  # Will be MetricExtractionService
        config: Optional[FormatHandlerConfig] = None
    ):
        """
        Initialize the format handler.
        
        Args:
            metric_extractor: Service for extracting and validating metrics
            config: Handler configuration
        """
        self.metric_extractor = metric_extractor
        self.config = config or FormatHandlerConfig()
        self._seen_statements = set()
    
    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """
        Check if this handler can process the given data format.
        
        Args:
            data: Raw data to check
            
        Returns:
            True if this handler can process the data format
        """
        pass
    
    @abstractmethod
    def process(
        self,
        data: Any,
        symbols: List[str],
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Process raw financial data into standardized records.
        
        Args:
            data: Raw financial data in format-specific structure
            symbols: List of symbols this data relates to
            source: Data source name
            
        Returns:
            List of standardized financial records
        """
        pass
    
    def _create_record(
        self,
        symbol: str,
        year: int,
        period: str,
        metrics: Dict[str, Any],
        source: str,
        filing_date: Optional[Any] = None,
        statement_type: str = 'income_statement'
    ) -> Dict[str, Any]:
        """
        Create a standardized financial record.
        
        Args:
            symbol: Stock symbol
            year: Fiscal year
            period: Fiscal period (FY, Q1, Q2, Q3, Q4)
            metrics: Dictionary of financial metrics
            source: Data source
            filing_date: Date of filing
            statement_type: Type of financial statement
            
        Returns:
            Standardized financial record
        """
        return {
            'symbol': symbol,
            'statement_type': statement_type,
            'year': year,
            'period': period,
            'revenue': self.metric_extractor.extract_revenue(metrics),
            'net_income': self.metric_extractor.extract_net_income(metrics),
            'total_assets': self.metric_extractor.extract_total_assets(metrics),
            'total_liabilities': self.metric_extractor.extract_total_liabilities(metrics),
            'operating_cash_flow': self.metric_extractor.extract_operating_cash_flow(metrics),
            'filing_date': self._parse_filing_date(filing_date),
            'gross_profit': self.metric_extractor.extract_gross_profit(metrics),
            'operating_income': self.metric_extractor.extract_operating_income(metrics),
            'eps_basic': self.metric_extractor.extract_eps_basic(metrics),
            'eps_diluted': self.metric_extractor.extract_eps_diluted(metrics),
            'current_assets': self.metric_extractor.extract_current_assets(metrics),
            'current_liabilities': self.metric_extractor.extract_current_liabilities(metrics),
            'stockholders_equity': self.metric_extractor.extract_stockholders_equity(metrics),
            'raw_data': self._clean_raw_data(metrics),
            'source': source,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    
    def _should_skip_duplicate(self, symbol: str, year: int, period: str, statement_type: str = 'income_statement') -> bool:
        """
        Check if this statement combination has already been processed.
        
        Args:
            symbol: Stock symbol
            year: Fiscal year  
            period: Fiscal period
            statement_type: Type of statement
            
        Returns:
            True if should skip as duplicate
        """
        if not self.config.skip_duplicates:
            return False
            
        stmt_key = f"{symbol}_{statement_type}_{year}_{period}"
        if stmt_key in self._seen_statements:
            logger.debug(f"Skipping duplicate statement: {stmt_key}")
            return True
        
        self._seen_statements.add(stmt_key)
        return False
    
    def _parse_filing_date(self, date_value: Any) -> Optional[Any]:
        """
        Parse filing date from various formats.
        
        Args:
            date_value: Date in various formats
            
        Returns:
            Parsed date or None
        """
        if not date_value:
            return None
            
        try:
            # Handle pandas Timestamp
            if hasattr(date_value, 'date'):
                return date_value.date()
            
            # Handle datetime
            if isinstance(date_value, datetime):
                return date_value.date()
            
            # Handle date
            from datetime import date
            if isinstance(date_value, date):
                return date_value
            
            # Try parsing string
            if isinstance(date_value, str):
                import pandas as pd
                return pd.to_datetime(date_value).date()
                
        except Exception as e:
            logger.debug(f"Failed to parse filing date '{date_value}': {e}")
            
        return None
    
    def _clean_raw_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean raw data for JSON serialization.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Cleaned data safe for JSON serialization
        """
        if not data:
            return {}
            
        cleaned = {}
        for key, value in data.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert numpy/pandas types to Python types
            if hasattr(value, 'item'):
                cleaned[key] = value.item()
            elif hasattr(value, 'tolist'):
                cleaned[key] = value.tolist()
            else:
                cleaned[key] = value
                
        return cleaned
    
    def reset_duplicates(self):
        """Reset the duplicate tracking set."""
        self._seen_statements.clear()
        logger.debug(f"{self.__class__.__name__} duplicate tracking reset")
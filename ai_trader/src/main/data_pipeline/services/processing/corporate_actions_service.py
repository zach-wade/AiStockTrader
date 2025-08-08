"""
Corporate Actions Service

Service for common corporate actions operations including date parsing,
frequency mapping, and action type detection.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class CorporateActionsConfig:
    """Configuration for corporate actions service."""
    default_currency: str = 'USD'
    default_dividend_type: str = 'CD'  # Cash Dividend
    default_split_from: float = 1.0
    default_split_to: float = 1.0


class CorporateActionsService:
    """
    Service for processing corporate actions data.
    
    Provides common functionality for date parsing, frequency mapping,
    and action type detection used across different action processors.
    """
    
    # Frequency mappings
    FREQUENCY_TO_INT = {
        'ONE_TIME': 0,
        'ANNUAL': 1,
        'YEARLY': 1,
        'BIANNUAL': 2,
        'SEMIANNUAL': 2,
        'QUARTERLY': 4,
        'MONTHLY': 12,
        'WEEKLY': 52,
        'OTHER': None
    }
    
    INT_TO_FREQUENCY = {
        0: 'ONE_TIME',
        1: 'ANNUAL',
        2: 'BIANNUAL',
        4: 'QUARTERLY',
        12: 'MONTHLY',
        52: 'WEEKLY'
    }
    
    def __init__(self, config: Optional[CorporateActionsConfig] = None):
        """
        Initialize the corporate actions service.
        
        Args:
            config: Service configuration
        """
        self.config = config or CorporateActionsConfig()
    
    def parse_date(self, date_value: Any) -> Optional[datetime]:
        """
        Parse date from various formats.
        
        Args:
            date_value: Date in various formats (string, datetime, etc.)
            
        Returns:
            Parsed datetime with UTC timezone or None
        """
        if not date_value:
            return None
            
        try:
            # Already a datetime
            if isinstance(date_value, datetime):
                return date_value if date_value.tzinfo else date_value.replace(tzinfo=timezone.utc)
            
            # String format
            date_str = str(date_value)
            
            # Try date only format (YYYY-MM-DD)
            if len(date_str) == 10 and '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
            # Try ISO format with time
            if 'T' in date_str:
                # Handle Z suffix for UTC
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                return datetime.fromisoformat(date_str)
            
            # Try other common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_value}': {e}")
            
        return None
    
    def map_frequency_to_int(self, frequency: Any) -> Optional[int]:
        """
        Map frequency string to integer value.
        
        Args:
            frequency: Frequency as string or int
            
        Returns:
            Integer frequency value or None
        """
        if frequency is None:
            return None
            
        # Already an integer
        if isinstance(frequency, (int, float)):
            return int(frequency)
            
        # String mapping
        if isinstance(frequency, str):
            return self.FREQUENCY_TO_INT.get(frequency.upper())
            
        return None
    
    def map_int_to_frequency(self, frequency: Optional[int]) -> Optional[str]:
        """
        Map integer frequency to string representation.
        
        Args:
            frequency: Integer frequency value
            
        Returns:
            String frequency or None
        """
        if frequency is None:
            return None
            
        return self.INT_TO_FREQUENCY.get(frequency, 'OTHER')
    
    def detect_action_type(self, action: Dict[str, Any]) -> Optional[str]:
        """
        Detect the type of corporate action from data fields.
        
        Args:
            action: Corporate action data
            
        Returns:
            Action type ('dividend', 'split', etc.) or None
        """
        # Explicit action_type field
        if 'action_type' in action:
            return action['action_type'].lower()
        
        # Explicit type field
        if 'type' in action:
            return action['type'].lower()
        
        # Detect dividend by fields
        dividend_fields = [
            'ex_dividend_date', 'cash_amount', 'dividend_type',
            'payment_date', 'declaration_date', 'frequency'
        ]
        if any(field in action for field in dividend_fields):
            return 'dividend'
        
        # Detect split by fields
        split_fields = [
            'split_from', 'split_to', 'execution_date', 'split_ratio'
        ]
        if any(field in action for field in split_fields):
            return 'split'
        
        # Could add detection for other types (merger, spinoff, etc.)
        
        return None
    
    def validate_ticker(self, ticker: Any) -> bool:
        """
        Validate ticker symbol.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if valid ticker
        """
        if not ticker:
            return False
            
        ticker_str = str(ticker).strip().upper()
        
        # Basic validation - should be 1-5 uppercase letters
        if not ticker_str or len(ticker_str) > 5:
            return False
            
        # Should contain only letters and possibly dots/dashes for special tickers
        import re
        if not re.match(r'^[A-Z][A-Z0-9.\-]*$', ticker_str):
            return False
            
        return True
    
    def calculate_split_ratio(self, split_from: float, split_to: float) -> float:
        """
        Calculate split ratio.
        
        Args:
            split_from: Split from value (e.g., 1 in "1-for-2")
            split_to: Split to value (e.g., 2 in "1-for-2")
            
        Returns:
            Split ratio (to/from)
        """
        if not split_from or split_from == 0:
            return 1.0
            
        return split_to / split_from
    
    def format_split_description(self, split_from: float, split_to: float) -> str:
        """
        Format split as human-readable description.
        
        Args:
            split_from: Split from value
            split_to: Split to value
            
        Returns:
            Formatted description (e.g., "2-for-1 split")
        """
        # Format as integers if whole numbers
        from_str = str(int(split_from)) if split_from == int(split_from) else str(split_from)
        to_str = str(int(split_to)) if split_to == int(split_to) else str(split_to)
        
        return f"{to_str}-for-{from_str} split"
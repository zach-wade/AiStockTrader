"""
Dividend Processor

Processor for dividend corporate actions.
"""

from typing import Dict, Any, Optional

from main.utils.core import get_logger
from .base import BaseActionProcessor

logger = get_logger(__name__)


class DividendProcessor(BaseActionProcessor):
    """
    Processor for dividend corporate actions.
    
    Handles cash dividends, stock dividends, and special dividends
    with proper field mapping and validation.
    """
    
    def can_process(self, action: Dict[str, Any]) -> bool:
        """
        Check if this processor can handle the given action.
        
        Dividend indicators:
        - action_type == 'dividend'
        - Contains dividend-specific fields
        
        Args:
            action: Corporate action data
            
        Returns:
            True if this is a dividend action
        """
        # Check explicit type
        action_type = self.actions_service.detect_action_type(action)
        if action_type == 'dividend':
            return True
        
        # Check for dividend-specific fields
        dividend_fields = [
            'ex_dividend_date', 'cash_amount', 'dividend_type',
            'payment_date', 'declaration_date'
        ]
        
        return any(field in action for field in dividend_fields)
    
    def process(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process a dividend action into standardized format.
        
        Args:
            action: Raw dividend data
            source: Data source name
            
        Returns:
            Standardized dividend record or None if invalid
        """
        # Validate base fields
        if not self._validate_base_fields(action):
            if not self.config.skip_invalid:
                logger.warning(f"Invalid dividend action: missing required fields")
            return None
        
        # Extract ticker
        ticker = self._extract_ticker(action)
        if not self.actions_service.validate_ticker(ticker):
            logger.debug(f"Invalid ticker: {ticker}")
            if self.config.skip_invalid:
                return None
        
        # Handle pre-transformed data
        if 'action_type' in action and action['action_type'] == 'dividend':
            return self._process_transformed_dividend(action, source)
        else:
            return self._process_raw_dividend(action, source)
    
    def get_action_type(self) -> str:
        """Get the action type this processor handles."""
        return 'dividend'
    
    def _process_raw_dividend(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Process raw dividend data from API.
        
        Args:
            action: Raw dividend data
            source: Data source
            
        Returns:
            Standardized dividend record
        """
        ticker = self._extract_ticker(action)
        
        # Parse dates
        ex_date = self.actions_service.parse_date(
            action.get('ex_dividend_date') or action.get('ex_date')
        )
        payment_date = self.actions_service.parse_date(
            action.get('payment_date') or action.get('pay_date')
        )
        record_date = self.actions_service.parse_date(action.get('record_date'))
        declaration_date = self.actions_service.parse_date(action.get('declaration_date'))
        
        # Extract amount
        cash_amount = self._extract_cash_amount(action)
        
        # Map frequency
        frequency = self.actions_service.map_frequency_to_int(action.get('frequency'))
        
        # Create base record
        record = self._create_base_record(
            ticker=ticker,
            ex_date=ex_date,
            source=source,
            polygon_id=self._extract_polygon_id(action)
        )
        
        # Add dividend-specific fields
        record.update({
            'cash_amount': cash_amount,
            'currency': action.get('currency', self.config.default_currency),
            'dividend_type': action.get('dividend_type', 'CD'),
            'frequency': frequency,
            'payment_date': payment_date,
            'record_date': record_date,
            'declaration_date': declaration_date,
            # For database compatibility (some fields expected to be None for dividends)
            'split_from': None,
            'split_to': None
        })
        
        return record
    
    def _process_transformed_dividend(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Process pre-transformed dividend data.
        
        Args:
            action: Transformed dividend data
            source: Data source
            
        Returns:
            Standardized dividend record
        """
        # Map transformed fields to expected format
        mapped = {
            'ticker': action.get('ticker', action.get('symbol', '')),
            'ex_dividend_date': action.get('ex_date', action.get('ex_dividend_date', action.get('timestamp'))),
            'cash_amount': action.get('amount', action.get('cash_amount', 0)),
            'currency': action.get('currency', self.config.default_currency),
            'dividend_type': action.get('dividend_type', 'CD'),
            'frequency': action.get('frequency'),
            'record_date': action.get('record_date'),
            'payment_date': action.get('pay_date', action.get('payment_date')),
            'declaration_date': action.get('declaration_date'),
            'id': action.get('polygon_id', action.get('id'))
        }
        
        return self._process_raw_dividend(mapped, source)
    
    def _extract_cash_amount(self, action: Dict[str, Any]) -> float:
        """
        Extract and validate cash amount.
        
        Args:
            action: Dividend data
            
        Returns:
            Cash amount as float
        """
        amount = action.get('cash_amount') or action.get('amount') or 0.0
        
        try:
            amount = float(amount)
            
            # Validate reasonable range (0 to $1000 per share)
            if amount < 0 or amount > 1000:
                logger.warning(f"Unusual dividend amount: ${amount}")
                if self.config.validate_data and amount < 0:
                    return 0.0
                    
            return amount
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid cash amount: {amount}")
            return 0.0
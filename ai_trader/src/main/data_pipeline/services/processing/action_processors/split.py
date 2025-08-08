"""
Split Processor

Processor for stock split corporate actions.
"""

from typing import Dict, Any, Optional

from main.utils.core import get_logger
from .base import BaseActionProcessor

logger = get_logger(__name__)


class SplitProcessor(BaseActionProcessor):
    """
    Processor for stock split corporate actions.
    
    Handles regular splits, reverse splits, and stock dividends
    with proper ratio calculation and validation.
    """
    
    def can_process(self, action: Dict[str, Any]) -> bool:
        """
        Check if this processor can handle the given action.
        
        Split indicators:
        - action_type == 'split'
        - Contains split-specific fields
        
        Args:
            action: Corporate action data
            
        Returns:
            True if this is a split action
        """
        # Check explicit type
        action_type = self.actions_service.detect_action_type(action)
        if action_type == 'split':
            return True
        
        # Check for split-specific fields
        split_fields = [
            'split_from', 'split_to', 'execution_date', 'split_ratio'
        ]
        
        return any(field in action for field in split_fields)
    
    def process(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process a split action into standardized format.
        
        Args:
            action: Raw split data
            source: Data source name
            
        Returns:
            Standardized split record or None if invalid
        """
        # Validate base fields
        if not self._validate_base_fields(action):
            if not self.config.skip_invalid:
                logger.warning(f"Invalid split action: missing required fields")
            return None
        
        # Extract ticker
        ticker = self._extract_ticker(action)
        if not self.actions_service.validate_ticker(ticker):
            logger.debug(f"Invalid ticker: {ticker}")
            if self.config.skip_invalid:
                return None
        
        # Handle pre-transformed data
        if 'action_type' in action and action['action_type'] == 'split':
            return self._process_transformed_split(action, source)
        else:
            return self._process_raw_split(action, source)
    
    def get_action_type(self) -> str:
        """Get the action type this processor handles."""
        return 'split'
    
    def _process_raw_split(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Process raw split data from API.
        
        Args:
            action: Raw split data
            source: Data source
            
        Returns:
            Standardized split record
        """
        ticker = self._extract_ticker(action)
        
        # Parse dates
        ex_date = self.actions_service.parse_date(
            action.get('ex_date') or action.get('execution_date')
        )
        
        # Use execution_date as ex_date if ex_date not provided
        if not ex_date:
            ex_date = self.actions_service.parse_date(action.get('execution_date'))
        
        # Extract split values
        split_from, split_to = self._extract_split_values(action)
        
        # Validate split values
        if not self._validate_split_values(split_from, split_to):
            if self.config.skip_invalid:
                return None
        
        # Create base record
        record = self._create_base_record(
            ticker=ticker,
            ex_date=ex_date,
            source=source,
            polygon_id=self._extract_polygon_id(action)
        )
        
        # Add split-specific fields
        record.update({
            'split_from': split_from,
            'split_to': split_to,
            'execution_date': ex_date,  # Use ex_date as execution_date
            # For database compatibility (fields expected to be None for splits)
            'cash_amount': None,
            'currency': None,
            'dividend_type': None,
            'frequency': None,
            'payment_date': None,
            'record_date': None,
            'declaration_date': None
        })
        
        return record
    
    def _process_transformed_split(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Process pre-transformed split data.
        
        Args:
            action: Transformed split data
            source: Data source
            
        Returns:
            Standardized split record
        """
        # Map transformed fields to expected format
        mapped = {
            'ticker': action.get('ticker', action.get('symbol', '')),
            'execution_date': action.get('execution_date', action.get('ex_date', action.get('timestamp'))),
            'split_from': action.get('split_from', 1),
            'split_to': action.get('split_to', 1),
            'ex_date': action.get('ex_date', action.get('execution_date')),
            'id': action.get('polygon_id', action.get('id'))
        }
        
        return self._process_raw_split(mapped, source)
    
    def _extract_split_values(self, action: Dict[str, Any]) -> tuple[float, float]:
        """
        Extract split from and to values.
        
        Args:
            action: Split data
            
        Returns:
            Tuple of (split_from, split_to)
        """
        # Direct fields
        split_from = action.get('split_from')
        split_to = action.get('split_to')
        
        # Try to extract from ratio if provided
        if not split_from or not split_to:
            ratio = action.get('split_ratio') or action.get('ratio')
            if ratio:
                # Ratio might be string like "2:1" or float like 2.0
                if isinstance(ratio, str) and ':' in ratio:
                    parts = ratio.split(':')
                    try:
                        split_to = float(parts[0])
                        split_from = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                elif isinstance(ratio, (int, float)):
                    # Assume ratio means "to" value with from=1
                    split_to = float(ratio)
                    split_from = 1.0
        
        # Convert to float with defaults
        try:
            split_from = float(split_from) if split_from else 1.0
            split_to = float(split_to) if split_to else 1.0
        except (ValueError, TypeError):
            split_from = 1.0
            split_to = 1.0
        
        return split_from, split_to
    
    def _validate_split_values(self, split_from: float, split_to: float) -> bool:
        """
        Validate split values are reasonable.
        
        Args:
            split_from: Split from value
            split_to: Split to value
            
        Returns:
            True if valid split values
        """
        # Both must be positive
        if split_from <= 0 or split_to <= 0:
            logger.warning(f"Invalid split values: {split_from}:{split_to}")
            return False
        
        # Ratio should be reasonable (between 1:100 and 100:1)
        ratio = split_to / split_from
        if ratio < 0.01 or ratio > 100:
            logger.warning(f"Unusual split ratio: {split_from}:{split_to} (ratio={ratio})")
            # Still allow but warn
        
        # Can't both be 1 (no split)
        if split_from == 1.0 and split_to == 1.0:
            logger.debug("Split with 1:1 ratio (no change)")
            return False
        
        return True
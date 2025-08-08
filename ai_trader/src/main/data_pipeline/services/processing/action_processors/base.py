"""
Base Action Processor

Abstract base class for corporate action processors using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class ActionProcessorConfig:
    """Configuration for action processors."""
    validate_data: bool = True
    skip_invalid: bool = True
    default_currency: str = 'USD'


class BaseActionProcessor(ABC):
    """
    Abstract base class for processing different types of corporate actions.
    
    Implements the Strategy pattern for handling various action types
    (dividends, splits, mergers, etc.) with different processing logic.
    """
    
    def __init__(
        self,
        actions_service: Any,  # Will be CorporateActionsService
        config: Optional[ActionProcessorConfig] = None
    ):
        """
        Initialize the action processor.
        
        Args:
            actions_service: Service for common corporate actions operations
            config: Processor configuration
        """
        self.actions_service = actions_service
        self.config = config or ActionProcessorConfig()
    
    @abstractmethod
    def can_process(self, action: Dict[str, Any]) -> bool:
        """
        Check if this processor can handle the given action.
        
        Args:
            action: Corporate action data
            
        Returns:
            True if this processor can handle the action
        """
        pass
    
    @abstractmethod
    def process(
        self,
        action: Dict[str, Any],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process a corporate action into standardized format.
        
        Args:
            action: Raw corporate action data
            source: Data source name
            
        Returns:
            Standardized action record or None if invalid
        """
        pass
    
    @abstractmethod
    def get_action_type(self) -> str:
        """
        Get the type of action this processor handles.
        
        Returns:
            Action type string (e.g., 'dividend', 'split')
        """
        pass
    
    def _create_base_record(
        self,
        ticker: str,
        ex_date: Optional[datetime],
        source: str,
        polygon_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a base corporate action record.
        
        Args:
            ticker: Stock ticker symbol
            ex_date: Ex-date for the action
            source: Data source
            polygon_id: Optional Polygon ID
            
        Returns:
            Base record dictionary
        """
        return {
            'type': self.get_action_type(),
            'ticker': ticker.upper() if ticker else '',
            'action_type': self.get_action_type(),
            'ex_date': ex_date,
            'source': source,
            'polygon_id': polygon_id,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    
    def _validate_base_fields(self, action: Dict[str, Any]) -> bool:
        """
        Validate required base fields.
        
        Args:
            action: Action data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.validate_data:
            return True
            
        # Must have ticker/symbol
        ticker = action.get('ticker') or action.get('symbol')
        if not ticker:
            logger.debug("Action missing ticker/symbol")
            return False
            
        return True
    
    def _extract_ticker(self, action: Dict[str, Any]) -> str:
        """
        Extract ticker symbol from action data.
        
        Args:
            action: Action data
            
        Returns:
            Ticker symbol or empty string
        """
        ticker = action.get('ticker') or action.get('symbol') or ''
        return ticker.upper() if ticker else ''
    
    def _extract_polygon_id(self, action: Dict[str, Any]) -> Optional[str]:
        """
        Extract Polygon ID from action data.
        
        Args:
            action: Action data
            
        Returns:
            Polygon ID or None
        """
        return action.get('polygon_id') or action.get('id')
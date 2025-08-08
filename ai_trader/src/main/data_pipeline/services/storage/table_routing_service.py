"""
Table Routing Service

Handles routing of market data to appropriate tables based on interval
and configuration.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field

from main.utils.core import get_logger
from main.data_pipeline.types import TimeInterval
from main.utils.time.interval_utils import TimeIntervalUtils

logger = get_logger(__name__)


@dataclass
class TableRoutingConfig:
    """Configuration for table routing."""
    table_mapping: Dict[str, str] = field(default_factory=lambda: {
        "1minute": "market_data_1m",
        "1min": "market_data_1m",
        "5minute": "market_data_5m",
        "5min": "market_data_5m",
        "15minute": "market_data_15m",
        "15min": "market_data_15m",
        "30minute": "market_data_30m",
        "30min": "market_data_30m",
        "1hour": "market_data_1h",
        "60minute": "market_data_1h",
        "60min": "market_data_1h",
        "1day": "market_data_1h",  # Daily data goes in hourly table
        "daily": "market_data_1h"
    })
    default_table: str = "market_data_1h"
    normalize_intervals: bool = True


class TableRoutingService:
    """
    Service for routing market data to appropriate database tables.
    
    This service determines which table should receive data based on
    the time interval and configuration.
    """
    
    def __init__(self, config: Optional[TableRoutingConfig] = None):
        """
        Initialize the table routing service.
        
        Args:
            config: Service configuration
        """
        self.config = config or TableRoutingConfig()
        
        # Build reverse mapping for validation
        self._tables_to_intervals: Dict[str, list] = {}
        for interval, table in self.config.table_mapping.items():
            if table not in self._tables_to_intervals:
                self._tables_to_intervals[table] = []
            self._tables_to_intervals[table].append(interval)
        
        logger.info(f"TableRoutingService initialized with {len(self.config.table_mapping)} mappings")
    
    def get_table_for_interval(self, interval: str) -> str:
        """
        Get the appropriate table name for a given interval.
        
        Args:
            interval: Time interval (e.g., '1minute', '5min', '1hour', '1day')
            
        Returns:
            Table name for the interval
        """
        # Normalize interval if configured
        if self.config.normalize_intervals:
            interval = self._normalize_interval(interval)
        
        # Look up table
        table = self.config.table_mapping.get(interval)
        
        if not table:
            logger.warning(
                f"No table mapping for interval '{interval}', "
                f"using default: {self.config.default_table}"
            )
            table = self.config.default_table
        
        return table
    
    def get_table_for_time_interval(self, interval: TimeInterval) -> str:
        """
        Get the appropriate table name for a TimeInterval enum.
        
        Args:
            interval: TimeInterval enum value
            
        Returns:
            Table name for the interval
        """
        # Convert enum to string value
        interval_str = interval.value if hasattr(interval, 'value') else str(interval)
        return self.get_table_for_interval(interval_str)
    
    def get_interval_for_table(self, table_name: str, original_interval: str) -> str:
        """
        Get the canonical interval name for a table.
        
        This is used when storing data to ensure the interval column
        has the correct value for the table.
        
        Args:
            table_name: Name of the table
            original_interval: Original interval requested
            
        Returns:
            Canonical interval name for the table
        """
        # Special handling for daily data in hourly table
        if original_interval in ['1day', 'daily'] and table_name == 'market_data_1h':
            return '1day'
        
        # Map table back to its primary interval
        interval_map = {
            'market_data_1m': '1minute',
            'market_data_5m': '5minute',
            'market_data_15m': '15minute',
            'market_data_30m': '30minute',
            'market_data_1h': '1hour'
        }
        
        return interval_map.get(table_name, original_interval)
    
    def _normalize_interval(self, interval: str) -> str:
        """
        Normalize interval string to standard format.
        
        Args:
            interval: Raw interval string
            
        Returns:
            Normalized interval string
        """
        interval = interval.lower().strip()
        
        # Handle common variations
        replacements = {
            'min': 'minute',
            'mins': 'minute',
            'minutes': 'minute',
            'hr': 'hour',
            'hrs': 'hour',
            'hours': 'hour',
            'd': 'day',
            'days': 'day'
        }
        
        for old, new in replacements.items():
            if interval.endswith(old):
                # Extract number and unit
                for i in range(len(interval)):
                    if not interval[i].isdigit():
                        if i > 0:
                            number = interval[:i]
                            interval = f"{number}{new}"
                        break
        
        return interval
    
    def get_supported_intervals(self) -> list[str]:
        """
        Get list of all supported intervals.
        
        Returns:
            List of interval strings
        """
        return list(self.config.table_mapping.keys())
    
    def get_all_tables(self) -> list[str]:
        """
        Get list of all unique table names.
        
        Returns:
            List of table names
        """
        tables = set(self.config.table_mapping.values())
        tables.add(self.config.default_table)
        return sorted(list(tables))
    
    def validate_interval(self, interval: str) -> bool:
        """
        Check if an interval is valid/supported.
        
        Args:
            interval: Interval to validate
            
        Returns:
            True if interval is supported
        """
        if self.config.normalize_intervals:
            interval = self._normalize_interval(interval)
        return interval in self.config.table_mapping
    
    def get_table_info(self, table_name: str) -> Dict[str, any]:
        """
        Get information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        intervals = self._tables_to_intervals.get(table_name, [])
        
        return {
            'table_name': table_name,
            'supported_intervals': intervals,
            'is_valid': len(intervals) > 0,
            'primary_interval': self.get_interval_for_table(table_name, '')
        }
    
    def update_mapping(self, interval: str, table_name: str):
        """
        Update or add a table mapping.
        
        Args:
            interval: Interval string
            table_name: Table name to map to
        """
        self.config.table_mapping[interval] = table_name
        
        # Update reverse mapping
        if table_name not in self._tables_to_intervals:
            self._tables_to_intervals[table_name] = []
        if interval not in self._tables_to_intervals[table_name]:
            self._tables_to_intervals[table_name].append(interval)
        
        logger.info(f"Updated routing: {interval} -> {table_name}")
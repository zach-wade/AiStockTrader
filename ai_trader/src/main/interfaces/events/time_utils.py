"""
Time utilities for event handling.

These utilities are defined here to avoid circular dependencies 
with the main utils module.
"""

from datetime import datetime, timezone
from typing import Optional


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure a datetime object is timezone-aware and in UTC.
    
    Args:
        dt: Datetime object to convert
        
    Returns:
        Datetime object in UTC timezone, or None if input is None
    """
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        # Assume naive datetime is in UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC if in different timezone
        return dt.astimezone(timezone.utc)
"""
JSON serialization utilities for the AI Trader system.

Provides JSON encoding/decoding with support for common types used in events:
- datetime objects (converted to ISO format)
- Decimal numbers
- Enum values
- Dataclasses and custom objects

This module fills the gap for the to_json/from_json functions referenced
elsewhere in the codebase and provides a consistent serialization approach.
"""

import json
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Union, List
import dataclasses
import uuid


class EventJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles Event-specific types.
    
    Supports:
    - datetime/date objects -> ISO format strings
    - pandas.Timestamp -> ISO format strings
    - Decimal -> float
    - Enum -> value
    - UUID -> string
    - dataclasses -> dict
    - Sets -> lists
    """
    
    def default(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable forms."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        # Handle pandas Timestamp
        elif hasattr(obj, 'isoformat') and hasattr(obj, 'timestamp'):
            # This catches pandas.Timestamp and similar datetime-like objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, **kwargs) -> str:
    """
    Convert object to JSON string using EventJSONEncoder.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps
        
    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=EventJSONEncoder, **kwargs)


def from_json(json_str: str, **kwargs) -> Any:
    """
    Parse JSON string to object.
    
    Args:
        json_str: JSON string to parse
        **kwargs: Additional arguments passed to json.loads
        
    Returns:
        Parsed object (dict, list, etc.)
    """
    return json.loads(json_str, **kwargs)


def parse_iso_datetime(iso_str: str) -> datetime:
    """
    Parse ISO format datetime string.
    
    Args:
        iso_str: ISO format datetime string
        
    Returns:
        datetime object
    """
    # Handle both with and without timezone info
    try:
        # Try parsing with timezone
        return datetime.fromisoformat(iso_str)
    except ValueError:
        # If that fails, try without timezone and assume UTC
        dt = datetime.fromisoformat(iso_str.replace('Z', ''))
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


def dict_to_dataclass(data_dict: Dict[str, Any], dataclass_type: type) -> Any:
    """
    Convert a dictionary to a dataclass instance.
    
    Handles nested dataclasses and type conversions.
    
    Args:
        data_dict: Dictionary of field values
        dataclass_type: The dataclass type to create
        
    Returns:
        Instance of dataclass_type
    """
    if not dataclasses.is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")
    
    field_values = {}
    fields = {f.name: f for f in dataclasses.fields(dataclass_type)}
    
    for field_name, field_value in data_dict.items():
        if field_name in fields:
            field = fields[field_name]
            
            # Handle datetime fields
            if field.type == datetime and isinstance(field_value, str):
                field_values[field_name] = parse_iso_datetime(field_value)
            
            # Handle nested dataclasses
            elif dataclasses.is_dataclass(field.type) and isinstance(field_value, dict):
                field_values[field_name] = dict_to_dataclass(field_value, field.type)
            
            # Handle lists of dataclasses
            elif (hasattr(field.type, '__origin__') and 
                  field.type.__origin__ == list and 
                  len(field.type.__args__) > 0 and
                  dataclasses.is_dataclass(field.type.__args__[0])):
                item_type = field.type.__args__[0]
                field_values[field_name] = [
                    dict_to_dataclass(item, item_type) if isinstance(item, dict) else item
                    for item in field_value
                ]
            
            else:
                field_values[field_name] = field_value
    
    return dataclass_type(**field_values)
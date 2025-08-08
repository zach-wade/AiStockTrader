"""
Shared Mathematical Utilities

Core mathematical operations used across multiple modules to avoid circular dependencies.
These utilities provide safe operations and are used by both feature calculators and monitoring systems.
"""

import numpy as np
import pandas as pd
from typing import Union

# Import logger from core utils to avoid circular dependencies
from main.utils.core import get_logger

logger = get_logger(__name__)


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide avoiding division by zero and handling edge cases.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default_value: Value to return when division is undefined
        
    Returns:
        Result of division or default value
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            
        if isinstance(result, (pd.Series, np.ndarray)):
            result = pd.Series(result) if isinstance(result, np.ndarray) else result
            result = result.replace([np.inf, -np.inf], default_value)
            result = result.fillna(default_value)
        else:
            if np.isnan(result) or np.isinf(result):
                result = default_value
                
        return result
        
    except Exception as e:
        logger.warning(f"Error in safe_divide: {e}")
        return default_value


def safe_log(
    value: Union[float, np.ndarray, pd.Series],
    base: Union[float, None] = None,
    default_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate logarithm avoiding negative/zero inputs.
    
    Args:
        value: Input value(s)
        base: Logarithm base (None for natural log)
        default_value: Value to return for invalid inputs
        
    Returns:
        Logarithm result or default value
    """
    try:
        # Ensure positive values
        if isinstance(value, (pd.Series, np.ndarray)):
            positive_mask = value > 0
            result = pd.Series(index=value.index if isinstance(value, pd.Series) else None,
                             data=default_value)
            
            if positive_mask.any():
                if base is None:
                    result[positive_mask] = np.log(value[positive_mask])
                else:
                    result[positive_mask] = np.log(value[positive_mask]) / np.log(base)
        else:
            if value <= 0:
                return default_value
            result = np.log(value) if base is None else np.log(value) / np.log(base)
            
        return result
        
    except Exception as e:
        logger.warning(f"Error in safe_log: {e}")
        return default_value


def safe_sqrt(
    value: Union[float, np.ndarray, pd.Series],
    default_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate square root avoiding negative inputs.
    
    Args:
        value: Input value(s)
        default_value: Value to return for invalid inputs
        
    Returns:
        Square root result or default value
    """
    try:
        if isinstance(value, (pd.Series, np.ndarray)):
            result = np.sqrt(np.maximum(value, 0))
        else:
            result = np.sqrt(max(value, 0))
            
        return result
        
    except Exception as e:
        logger.warning(f"Error in safe_sqrt: {e}")
        return default_value


def calculate_growth_rate(
    current_value: float,
    previous_value: float,
    time_hours: float,
    default_rate: float = 0.0
) -> float:
    """
    Calculate growth rate per hour.
    
    Args:
        current_value: Current value
        previous_value: Previous value
        time_hours: Time elapsed in hours
        default_rate: Default rate if calculation fails
        
    Returns:
        Growth rate per hour
    """
    if time_hours <= 0:
        return default_rate
    
    try:
        rate = (current_value - previous_value) / time_hours
        return rate
    except Exception as e:
        logger.warning(f"Error calculating growth rate: {e}")
        return default_rate


def format_bytes(bytes_value: float, precision: int = 2) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        precision: Decimal precision
        
    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.{precision}f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.{precision}f} PB"
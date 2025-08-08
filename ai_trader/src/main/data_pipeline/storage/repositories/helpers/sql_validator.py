"""
SQL Validator Helper

Validates SQL components to prevent injection attacks and ensure query safety.
"""

import re
from typing import Dict, List, Set
from main.utils.core import get_logger

logger = get_logger(__name__)

# Whitelist of allowed column names by table
ALLOWED_COLUMNS = {
    'companies': {
        'symbol', 'name', 'description', 'sector', 'industry', 'market_cap',
        'employees', 'founded', 'headquarters', 'website', 'is_active',
        # Layer system columns
        'layer', 'layer_updated_at', 'layer_reason',
        # Scanner metadata column for flexible scanner-specific data
        'scanner_metadata',
        # Note: layer[1-3]_qualified columns removed after migration
        'current_price', 'avg_price', 'price_change', 'volume',
        'created_at', 'updated_at', 'id'
    },
    'market_data_1h': {
        'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close',
        'volume', 'vwap', 'transactions', 'created_at', 'id'
    },
    'features': {
        'symbol', 'timestamp', 'features', 'feature_count', 'metadata',
        'created_at', 'updated_at', 'id'
    },
    'scanner_qualifications': {
        'symbol', 'qualification_date', 'layer', 'qualified',
        'technical_score', 'fundamental_score', 'sentiment_score', 'composite_score',
        'signals', 'metadata', 'created_at', 'id'
    },
    'news_data': {
        'symbol', 'published_at', 'title', 'url', 'summary', 'content',
        'author', 'source', 'sentiment_score', 'relevance_score',
        'keywords', 'entities', 'created_at', 'id'
    },
    'financials_data': {
        'symbol', 'period_end', 'period_type', 'filing_date', 'fiscal_year', 'fiscal_quarter',
        'revenue', 'cost_of_revenue', 'gross_profit', 'operating_expenses', 'operating_income',
        'net_income', 'eps_basic', 'eps_diluted', 'shares_outstanding',
        'total_assets', 'current_assets', 'total_liabilities', 'current_liabilities',
        'total_equity', 'retained_earnings', 'operating_cash_flow', 'investing_cash_flow',
        'financing_cash_flow', 'free_cash_flow', 'profit_margin', 'return_on_assets',
        'return_on_equity', 'debt_to_equity', 'current_ratio', 'quick_ratio',
        'eps_estimate', 'revenue_estimate', 'created_at', 'id'
    },
    'sentiment_data': {
        'symbol', 'timestamp', 'sentiment_score', 'source', 'confidence',
        'volume', 'metadata', 'created_at', 'id'
    },
    'analyst_ratings': {
        'symbol', 'date', 'analyst', 'rating', 'price_target', 'firm',
        'created_at', 'updated_at', 'id'
    },
    'dividends': {
        'symbol', 'ex_date', 'amount', 'type', 'currency',
        'created_at', 'updated_at', 'id'
    },
    'social_sentiment': {
        'symbol', 'timestamp', 'platform', 'mentions', 'sentiment_score',
        'volume', 'metadata', 'created_at', 'id'
    },
    'earnings_guidance': {
        'symbol', 'issued_date', 'period_end', 'metric', 'guidance_value',
        'guidance_low', 'guidance_high', 'created_at', 'updated_at', 'id'
    }
}

# Common safe operators for WHERE clauses
SAFE_OPERATORS = {'=', '!=', '<>', '<', '>', '<=', '>=', 'LIKE', 'ILIKE', 'IN', 'NOT IN', 'IS', 'IS NOT'}

# Pattern for valid column names (alphanumeric + underscore, starting with letter/underscore)
COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


class SQLValidationError(Exception):
    """Raised when SQL validation fails."""
    pass


def validate_column_name(column_name: str) -> bool:
    """
    Validate that a column name is safe for SQL queries.
    
    Args:
        column_name: Column name to validate
        
    Returns:
        True if column name is safe
        
    Raises:
        SQLValidationError: If column name is unsafe
    """
    if not column_name:
        raise SQLValidationError("Column name cannot be empty")
    
    if not isinstance(column_name, str):
        raise SQLValidationError(f"Column name must be string, got {type(column_name)}")
    
    # Check for basic SQL injection patterns
    if not COLUMN_NAME_PATTERN.match(column_name):
        raise SQLValidationError(f"Invalid column name format: {column_name}")
    
    # Check for dangerous keywords
    dangerous_keywords = {
        'drop', 'delete', 'insert', 'update', 'create', 'alter', 'truncate',
        'exec', 'execute', 'union', 'select', 'from', 'where', 'having',
        'group', 'order', 'limit', 'offset', '--', '/*', '*/', ';'
    }
    
    if column_name.lower() in dangerous_keywords:
        raise SQLValidationError(f"Column name contains dangerous keyword: {column_name}")
    
    return True


def validate_table_column(table_name: str, column_name: str) -> bool:
    """
    Validate that a column exists in the whitelist for a specific table.
    
    Args:
        table_name: Name of the database table
        column_name: Column name to validate
        
    Returns:
        True if column is allowed for this table
        
    Raises:
        SQLValidationError: If column is not allowed
    """
    # First validate the column name format
    validate_column_name(column_name)
    
    # Check if table exists in whitelist
    if table_name not in ALLOWED_COLUMNS:
        raise SQLValidationError(f"Table '{table_name}' not in allowed tables")
    
    # Check if column is in whitelist for this table
    if column_name not in ALLOWED_COLUMNS[table_name]:
        raise SQLValidationError(
            f"Column '{column_name}' not allowed for table '{table_name}'"
        )
    
    return True


def sanitize_order_by_columns(table_name: str, columns: List[str]) -> List[str]:
    """
    Sanitize and validate ORDER BY columns.
    
    Args:
        table_name: Name of the database table
        columns: List of column names for ORDER BY
        
    Returns:
        List of validated column names
        
    Raises:
        SQLValidationError: If any column is invalid
    """
    if not columns:
        return []
    
    sanitized = []
    for column in columns:
        # Remove any direction indicators (ASC/DESC) and whitespace
        clean_column = column.strip().split()[0]
        
        # Validate the column
        validate_table_column(table_name, clean_column)
        sanitized.append(clean_column)
    
    return sanitized


def validate_filter_keys(table_name: str, filter_dict: Dict[str, any]) -> Dict[str, any]:
    """
    Validate filter dictionary keys against table column whitelist.
    
    Args:
        table_name: Name of the database table
        filter_dict: Dictionary of filters to validate
        
    Returns:
        Validated filter dictionary
        
    Raises:
        SQLValidationError: If any filter key is invalid
    """
    if not filter_dict:
        return {}
    
    validated = {}
    for key, value in filter_dict.items():
        # Validate the key as a column name
        validate_table_column(table_name, key)
        validated[key] = value
    
    return validated


def build_safe_where_condition(table_name: str, column: str, operator: str = '=') -> str:
    """
    Build a safe WHERE condition with validated components.
    
    Args:
        table_name: Name of the database table
        column: Column name (will be validated)
        operator: SQL operator (must be in safe list)
        
    Returns:
        Safe WHERE condition template
        
    Raises:
        SQLValidationError: If components are invalid
    """
    # Validate column
    validate_table_column(table_name, column)
    
    # Validate operator
    if operator.upper() not in SAFE_OPERATORS:
        raise SQLValidationError(f"Unsafe operator: {operator}")
    
    return f"{column} {operator.upper()}"


def get_allowed_columns(table_name: str) -> Set[str]:
    """
    Get the set of allowed columns for a table.
    
    Args:
        table_name: Name of the database table
        
    Returns:
        Set of allowed column names
        
    Raises:
        SQLValidationError: If table is not in whitelist
    """
    if table_name not in ALLOWED_COLUMNS:
        raise SQLValidationError(f"Table '{table_name}' not in allowed tables")
    
    return ALLOWED_COLUMNS[table_name].copy()


def add_allowed_columns(table_name: str, columns: Set[str]) -> None:
    """
    Add additional allowed columns for a table (for extensibility).
    
    Args:
        table_name: Name of the database table
        columns: Set of column names to add
    """
    if table_name not in ALLOWED_COLUMNS:
        ALLOWED_COLUMNS[table_name] = set()
    
    # Validate each column before adding
    for column in columns:
        validate_column_name(column)
    
    ALLOWED_COLUMNS[table_name].update(columns)
    logger.info(f"Added {len(columns)} columns to table '{table_name}' whitelist")


# Convenience function for logging security events
def log_security_event(event_type: str, details: str, severity: str = "WARNING") -> None:
    """Log security-related events."""
    message = f"SQL_SECURITY_{event_type}: {details}"
    
    if severity == "CRITICAL":
        logger.critical(message)
    elif severity == "ERROR":
        logger.error(message)
    elif severity == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
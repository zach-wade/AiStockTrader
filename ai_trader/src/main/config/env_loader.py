# File: config/env_loader.py

"""
Environment loader for configuration system.

Handles loading of environment variables and environment information.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def ensure_environment_loaded():
    """
    Ensure environment variables are loaded from .env file.
    
    This function loads environment variables from a .env file in the project root
    if it exists. It's safe to call multiple times.
    """
    try:
        # Look for .env file in project root
        # From /Users/zachwade/StockMonitoring/ai_trader/src/main/config/env_loader.py
        # Go up to ai_trader directory (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        env_file = project_root / ".env"
        
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug(f"Environment variables loaded from {env_file}")
        else:
            logger.debug("No .env file found, using system environment variables only")
            
    except Exception as e:
        logger.warning(f"Failed to load environment variables: {e}")


def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information
    """
    try:
        return {
            'environment': os.environ.get('ENVIRONMENT', 'development'),
            'trading_mode': os.environ.get('TRADING_MODE', 'paper'),
            'debug_mode': os.environ.get('DEBUG', 'False').lower() == 'true',
            'has_anthropic_key': 'ANTHROPIC_API_KEY' in os.environ,
            'has_openai_key': 'OPENAI_API_KEY' in os.environ,
            'has_alpha_vantage_key': 'ALPHA_VANTAGE_API_KEY' in os.environ,
            'has_database_url': 'DATABASE_URL' in os.environ,
            'python_path': os.environ.get('PYTHONPATH', ''),
            'total_env_vars': len(os.environ)
        }
    except Exception as e:
        logger.error(f"Failed to get environment info: {e}")
        return {'error': str(e)}


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with optional default.
    
    Args:
        name: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(name, default)


def set_env_var(name: str, value: str):
    """
    Set an environment variable.
    
    Args:
        name: Environment variable name
        value: Value to set
    """
    os.environ[name] = value
    logger.debug(f"Environment variable {name} set")


def is_development() -> bool:
    """
    Check if running in development environment.
    
    Returns:
        True if in development mode
    """
    env = get_env_var('ENVIRONMENT', 'development')
    return env.lower() in ['development', 'dev', 'local']


def is_production() -> bool:
    """
    Check if running in production environment.
    
    Returns:
        True if in production mode
    """
    env = get_env_var('ENVIRONMENT', 'development')
    return env.lower() in ['production', 'prod', 'live']


def validate_required_env_vars(required_vars: list) -> list:
    """
    Validate that required environment variables are present.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        List of missing environment variable names
    """
    missing = []
    for var in required_vars:
        if var not in os.environ:
            missing.append(var)
    
    if missing:
        logger.warning(f"Missing required environment variables: {missing}")
    
    return missing


def get_env_file_path() -> Path:
    """
    Get the path to the .env file.
    
    Returns:
        Path to the .env file
    """
    # Same as in ensure_environment_loaded - 4 levels up to ai_trader dir
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / ".env"


def validate_trading_environment() -> Dict[str, Any]:
    """
    Validate the trading environment configuration.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required API keys
    api_vars = TRADING_API_VARS
    missing_api = validate_required_env_vars(api_vars)
    if missing_api:
        results['valid'] = False
        results['errors'].append(f"Missing API variables: {missing_api}")
    
    # Check database configuration
    db_vars = DATABASE_VARS
    missing_db = validate_required_env_vars(db_vars)
    if missing_db:
        results['warnings'].append(f"Missing database variables: {missing_db}")
    
    return results


# Common environment variable groups
TRADING_API_VARS = [
    'ALPACA_API_KEY',
    'ALPACA_SECRET_KEY',
    'POLYGON_API_KEY'
]

DATABASE_VARS = [
    'DATABASE_URL',
    'REDIS_URL',
    'CACHE_TYPE'
]
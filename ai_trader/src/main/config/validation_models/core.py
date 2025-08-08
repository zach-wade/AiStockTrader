"""
Core configuration validation models (System, Database, API keys).
"""

import os
import re
from typing import Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

# Enums
class Environment(str, Enum):
    """Valid trading environments."""
    PAPER = "paper"
    LIVE = "live"
    TRAINING = "training"
    BACKTEST = "backtest"

def validate_env_var(value: str, var_name: str) -> str:
    """
    Validate environment variable substitution.
    
    Args:
        value: The value to validate (may contain ${VAR_NAME})
        var_name: Name of the variable for error messages
        
    Returns:
        Resolved environment variable value
        
    Raises:
        ValueError: If environment variable is missing or invalid
    """
    if not value:
        raise ValueError(f"{var_name} cannot be empty")
    
    # Check if it's an environment variable reference
    env_pattern = r'\$\{([^}]+)\}'
    match = re.search(env_pattern, value)
    
    if match:
        env_var_name = match.group(1)
        env_value = os.getenv(env_var_name)
        
        if env_value is None:
            raise ValueError(f"Environment variable {env_var_name} is not set for {var_name}")
        
        if not env_value.strip():
            raise ValueError(f"Environment variable {env_var_name} is empty for {var_name}")
        
        # Substitute all environment variables in the string
        def replacer(match):
            env_var = match.group(1)
            env_val = os.getenv(env_var)
            if env_val is None:
                raise ValueError(f"Environment variable {env_var} is not set")
            return env_val
        
        return re.sub(env_pattern, replacer, value)
    
    return value


# API Configuration Models
class AlpacaConfig(BaseModel):
    """Alpaca API configuration with validation."""
    key: str = Field(..., min_length=1, description="Alpaca API key")
    secret: str = Field(..., min_length=1, description="Alpaca secret key")
    base_url: str = Field(default="https://paper-api.alpaca.markets", description="Alpaca API base URL")
    data_url: str = Field(default="https://data.alpaca.markets", description="Alpaca data API URL")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "Alpaca API key")
    
    @field_validator('secret')
    @classmethod
    def validate_secret(cls, v):
        return validate_env_var(v, "Alpaca secret key")
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        if isinstance(v, str):
            v = validate_env_var(v, "Alpaca base URL")
            # Validate that it's a proper URL after substitution
            if not v.startswith(('http://', 'https://')):
                raise ValueError(f"Alpaca base URL must be a valid HTTP/HTTPS URL, got: {v}")
        return v
    
    def get(self, key: str, default: Any = None) -> Any:
        """Backward compatibility method for legacy code."""
        return getattr(self, key, default)

class PolygonConfig(BaseModel):
    """Polygon API configuration with validation."""
    key: str = Field(..., min_length=1, description="Polygon API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "Polygon API key")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Backward compatibility method for legacy code."""
        return getattr(self, key, default)

class AlphaVantageConfig(BaseModel):
    """Alpha Vantage API configuration with validation."""
    key: str = Field(..., min_length=1, description="Alpha Vantage API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "Alpha Vantage API key")

class BenzingaConfig(BaseModel):
    """Benzinga API configuration with validation."""
    key: str = Field(..., min_length=1, description="Benzinga API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "Benzinga API key")

class FinnhubConfig(BaseModel):
    """Finnhub API configuration with validation."""
    key: str = Field(..., min_length=1, description="Finnhub API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "Finnhub API key")

class FredConfig(BaseModel):
    """FRED API configuration with validation."""
    key: str = Field(..., min_length=1, description="FRED API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "FRED API key")

class NewsApiConfig(BaseModel):
    """News API configuration with validation."""
    key: str = Field(..., min_length=1, description="News API key")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_env_var(v, "News API key")

class RedditConfig(BaseModel):
    """Reddit API configuration with validation."""
    client_id: str = Field(..., min_length=1, description="Reddit client ID")
    client_secret: str = Field(..., min_length=1, description="Reddit client secret")
    user_agent: str = Field(..., min_length=1, description="Reddit user agent")
    
    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, v):
        return validate_env_var(v, "Reddit client ID")
    
    @field_validator('client_secret')
    @classmethod
    def validate_client_secret(cls, v):
        return validate_env_var(v, "Reddit client secret")
    
    @field_validator('user_agent')
    @classmethod
    def validate_user_agent(cls, v):
        return validate_env_var(v, "Reddit user agent")

class TwitterConfig(BaseModel):
    """Twitter API configuration with validation."""
    bearer_token: str = Field(..., min_length=1, description="Twitter bearer token")
    api_key: str = Field(..., min_length=1, description="Twitter API key")
    api_secret: str = Field(..., min_length=1, description="Twitter API secret")
    access_token: str = Field(..., min_length=1, description="Twitter access token")
    access_secret: str = Field(..., min_length=1, description="Twitter access secret")
    
    @field_validator('bearer_token')
    @classmethod
    def validate_bearer_token(cls, v):
        return validate_env_var(v, "Twitter bearer token")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        return validate_env_var(v, "Twitter API key")
    
    @field_validator('api_secret')
    @classmethod
    def validate_api_secret(cls, v):
        return validate_env_var(v, "Twitter API secret")
    
    @field_validator('access_token')
    @classmethod
    def validate_access_token(cls, v):
        return validate_env_var(v, "Twitter access token")
    
    @field_validator('access_secret')
    @classmethod
    def validate_access_secret(cls, v):
        return validate_env_var(v, "Twitter access secret")

class ApiKeysConfig(BaseModel):
    """Complete API keys configuration with validation."""
    alpaca: AlpacaConfig
    polygon: Optional[PolygonConfig] = None
    alpha_vantage: Optional[AlphaVantageConfig] = None
    benzinga: Optional[BenzingaConfig] = None
    finnhub: Optional[FinnhubConfig] = None
    fred: Optional[FredConfig] = None
    news_api: Optional[NewsApiConfig] = None
    reddit: Optional[RedditConfig] = None
    twitter: Optional[TwitterConfig] = None
    
    @model_validator(mode='after')
    def validate_required_apis(self):
        """Ensure critical APIs are configured."""
        if not self.alpaca:
            raise ValueError("Alpaca API configuration is required")
        
        # Warn if no secondary data sources are configured
        secondary_sources = [
            self.polygon,
            self.alpha_vantage,
            self.benzinga
        ]
        
        if not any(secondary_sources):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No secondary data sources configured - system will rely solely on Alpaca")
        
        return self
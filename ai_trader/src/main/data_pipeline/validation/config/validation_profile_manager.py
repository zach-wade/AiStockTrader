"""
Validation Profile Manager - Interface Implementation

Manages validation profiles and field mappings implementing IValidationProfileManager interface.
Provides profile-based configuration for validation operations.
"""

from main.utils.core import get_logger
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

# Interface imports
from main.interfaces.validation.config import (
    IValidationProfileManager,
    IFieldMappingConfig
)
from main.interfaces.data_pipeline.validation import ValidationSeverity

# Core imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class ValidationProfileManager:
    """
    Validation profile manager implementation.
    
    Implements IValidationProfileManager interface and provides
    comprehensive profile and field mapping management.
    """
    
    def __init__(
        self,
        field_mapping_config: Optional[IFieldMappingConfig] = None,
        custom_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize validation profile manager.
        
        Args:
            field_mapping_config: Field mapping configuration (DI)
            custom_profiles: Custom profile definitions
        """
        self.field_mapping_config = field_mapping_config
        self.custom_profiles = custom_profiles or {}
        
        # Load default profiles and field mappings
        self._default_profiles = self._load_default_profiles()
        self._field_mappings = self._load_field_mappings()
        
        logger.info("ValidationProfileManager initialized with interface-based architecture")
    
    # IValidationProfileManager interface methods
    async def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get validation profile by name."""
        profile_key = profile_name.upper()
        
        # Check custom profiles first
        if profile_key in self.custom_profiles:
            return self.custom_profiles[profile_key].copy()
        
        # Check default profiles
        if profile_key in self._default_profiles:
            return self._default_profiles[profile_key].copy()
        
        logger.warning(f"Profile '{profile_name}' not found, using LENIENT profile")
        return self._default_profiles['LENIENT'].copy()
    
    async def create_profile(
        self,
        profile_name: str,
        profile_config: Dict[str, Any]
    ) -> None:
        """Create or update a validation profile."""
        # Validate profile configuration
        required_fields = [
            'quality_threshold', 'max_null_percentage', 'severity_threshold'
        ]
        
        for field in required_fields:
            if field not in profile_config:
                raise ValueError(f"Profile configuration missing required field: {field}")
        
        # Validate severity threshold
        if isinstance(profile_config['severity_threshold'], str):
            try:
                profile_config['severity_threshold'] = ValidationSeverity(
                    profile_config['severity_threshold']
                )
            except ValueError:
                logger.warning(f"Invalid severity threshold, using WARNING")
                profile_config['severity_threshold'] = ValidationSeverity.WARNING
        
        # Store the profile
        self.custom_profiles[profile_name.upper()] = profile_config.copy()
        logger.info(f"Created/updated validation profile: {profile_name}")
    
    async def list_profiles(self) -> List[str]:
        """List all available validation profiles."""
        default_profiles = list(self._default_profiles.keys())
        custom_profiles = list(self.custom_profiles.keys())
        return default_profiles + custom_profiles
    
    async def delete_profile(self, profile_name: str) -> bool:
        """Delete a custom validation profile."""
        profile_key = profile_name.upper()
        
        # Cannot delete default profiles
        if profile_key in self._default_profiles:
            logger.warning(f"Cannot delete default profile: {profile_name}")
            return False
        
        if profile_key in self.custom_profiles:
            del self.custom_profiles[profile_key]
            logger.info(f"Deleted validation profile: {profile_name}")
            return True
        
        logger.warning(f"Profile not found for deletion: {profile_name}")
        return False
    
    async def get_field_mapping(
        self,
        source: str,
        target_schema: str = 'standard'
    ) -> Dict[str, str]:
        """Get field mapping for data source."""
        mapping_key = f"{source}_{target_schema}"
        
        if mapping_key in self._field_mappings:
            return self._field_mappings[mapping_key].copy()
        
        # Try source-only mapping
        if source in self._field_mappings:
            return self._field_mappings[source].copy()
        
        logger.warning(f"No field mapping found for {source} -> {target_schema}")
        return {}
    
    async def create_field_mapping(
        self,
        source: str,
        mapping: Dict[str, str],
        target_schema: str = 'standard'
    ) -> None:
        """Create or update field mapping."""
        mapping_key = f"{source}_{target_schema}"
        self._field_mappings[mapping_key] = mapping.copy()
        logger.info(f"Created/updated field mapping: {mapping_key}")
    
    async def validate_profile(self, profile_config: Dict[str, Any]) -> bool:
        """Validate profile configuration."""
        try:
            required_fields = [
                'quality_threshold',
                'max_null_percentage',
                'severity_threshold'
            ]
            
            # Check required fields
            for field in required_fields:
                if field not in profile_config:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate ranges
            if not (0 <= profile_config.get('quality_threshold', 0) <= 100):
                logger.error("quality_threshold must be between 0 and 100")
                return False
            
            if not (0 <= profile_config.get('max_null_percentage', 0) <= 1):
                logger.error("max_null_percentage must be between 0 and 1")
                return False
            
            # Validate severity threshold
            severity = profile_config.get('severity_threshold')
            if isinstance(severity, str):
                try:
                    ValidationSeverity(severity)
                except ValueError:
                    logger.error(f"Invalid severity threshold: {severity}")
                    return False
            elif not isinstance(severity, ValidationSeverity):
                logger.error("severity_threshold must be ValidationSeverity enum or string")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Profile validation failed: {e}")
            return False
    
    # Helper methods
    def _load_default_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load default validation profiles."""
        return {
            'STRICT': {
                'quality_threshold': 80,
                'allow_missing_vwap': False,
                'allow_zero_volume': False,
                'max_price_deviation': 0.5,  # 50% from median
                'require_all_fields': True,
                'max_nan_ratio': 0.1,
                'max_null_percentage': 0.02,
                'aggressive_cleaning': False,
                'min_data_points': 5,
                'allow_weekend_trading': False,
                'allow_future_timestamps': False,
                'severity_threshold': ValidationSeverity.ERROR,
                'timeout_seconds': 30,
                'retry_attempts': 2
            },
            
            'LENIENT': {
                'quality_threshold': 50,
                'allow_missing_vwap': True,
                'allow_zero_volume': True,
                'max_price_deviation': 2.0,  # 200% from median
                'require_all_fields': False,
                'max_nan_ratio': 0.5,
                'max_null_percentage': 0.15,
                'aggressive_cleaning': True,
                'min_data_points': 3,
                'allow_weekend_trading': True,
                'allow_future_timestamps': False,
                'severity_threshold': ValidationSeverity.WARNING,
                'timeout_seconds': 60,
                'retry_attempts': 3
            },
            
            'HISTORICAL': {
                'quality_threshold': 60,
                'allow_missing_vwap': True,
                'allow_zero_volume': True,
                'max_price_deviation': 10.0,  # More lenient for historical data
                'require_all_fields': False,
                'max_nan_ratio': 0.3,
                'max_null_percentage': 0.1,
                'aggressive_cleaning': False,
                'min_data_points': 10,
                'allow_weekend_trading': True,
                'allow_future_timestamps': False,
                'severity_threshold': ValidationSeverity.WARNING,
                'timeout_seconds': 120,  # Historical validation can take longer
                'retry_attempts': 1
            },
            
            'DEVELOPMENT': {
                'quality_threshold': 30,
                'allow_missing_vwap': True,
                'allow_zero_volume': True,
                'max_price_deviation': 5.0,
                'require_all_fields': False,
                'max_nan_ratio': 0.8,
                'max_null_percentage': 0.3,
                'aggressive_cleaning': True,
                'min_data_points': 1,
                'allow_weekend_trading': True,
                'allow_future_timestamps': True,
                'severity_threshold': ValidationSeverity.INFO,
                'timeout_seconds': 10,
                'retry_attempts': 0  # Fast fail in development
            },
            
            'TESTING': {
                'quality_threshold': 0,  # Very permissive for tests
                'allow_missing_vwap': True,
                'allow_zero_volume': True,
                'max_price_deviation': 100.0,
                'require_all_fields': False,
                'max_nan_ratio': 1.0,
                'max_null_percentage': 1.0,
                'aggressive_cleaning': False,
                'min_data_points': 0,
                'allow_weekend_trading': True,
                'allow_future_timestamps': True,
                'severity_threshold': ValidationSeverity.INFO,
                'timeout_seconds': 5,
                'retry_attempts': 0
            }
        }
    
    def _load_field_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load default field mappings."""
        return {
            # Polygon API mappings
            'polygon_standard': {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp',
                'vw': 'vwap',
                'n': 'transactions'
            },
            
            # Alpaca API mappings
            'alpaca_standard': {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp',
                'vw': 'vwap'
            },
            
            # Yahoo Finance mappings
            'yahoo_standard': {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjusted_close'
            },
            
            # News data mappings
            'polygon_news_standard': {
                'title': 'title',
                'published_utc': 'published',
                'tickers': 'symbols',
                'author': 'author',
                'article_url': 'url',
                'description': 'summary'
            },
            
            'benzinga_news_standard': {
                'title': 'title',
                'published': 'published',
                'symbols': 'symbols',
                'author': 'author',
                'url': 'url',
                'teaser': 'summary'
            },
            
            # Fundamentals mappings
            'polygon_fundamentals_standard': {
                'cik': 'cik',
                'fiscal_period': 'period',
                'fiscal_year': 'year',
                'end_date': 'period_end',
                'filing_date': 'filed_date'
            }
        }
    
    def _load_field_mappings_from_file(self, config_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """Load field mappings from configuration file."""
        if not config_path:
            return {}
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Field mapping config file not found: {config_path}")
                return {}
            
            with open(config_file, 'r') as f:
                mappings = json.load(f)
            
            logger.info(f"Loaded field mappings from {config_path}")
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to load field mappings from {config_path}: {e}")
            return {}
    
    def _validate_field_mappings(self, mappings: Dict[str, Dict[str, str]]) -> bool:
        """Validate field mappings configuration."""
        try:
            for mapping_name, mapping in mappings.items():
                if not isinstance(mapping, dict):
                    logger.error(f"Invalid field mapping format for {mapping_name}")
                    return False
                
                for source_field, target_field in mapping.items():
                    if not isinstance(source_field, str) or not isinstance(target_field, str):
                        logger.error(f"Invalid field mapping in {mapping_name}: {source_field} -> {target_field}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Field mapping validation failed: {e}")
            return False
    
    # Public utility methods
    def get_profile_names(self) -> List[str]:
        """Get list of all available profile names."""
        default_names = list(self._default_profiles.keys())
        custom_names = list(self.custom_profiles.keys())
        return default_names + custom_names
    
    def is_default_profile(self, profile_name: str) -> bool:
        """Check if profile is a default profile."""
        return profile_name.upper() in self._default_profiles
    
    def get_mapping_names(self) -> List[str]:
        """Get list of all available field mapping names."""
        return list(self._field_mappings.keys())
    
    def export_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Export all profiles for backup or transfer."""
        return {
            'default_profiles': self._default_profiles.copy(),
            'custom_profiles': self.custom_profiles.copy()
        }
    
    def import_profiles(self, profiles_data: Dict[str, Dict[str, Any]]) -> None:
        """Import profiles from backup or external source."""
        if 'custom_profiles' in profiles_data:
            for name, config in profiles_data['custom_profiles'].items():
                if self._validate_profile_config(config):
                    self.custom_profiles[name.upper()] = config
                else:
                    logger.warning(f"Skipped invalid profile during import: {name}")
        
        logger.info(f"Imported {len(profiles_data.get('custom_profiles', {}))} custom profiles")
    
    def _validate_profile_config(self, config: Dict[str, Any]) -> bool:
        """Validate a single profile configuration."""
        required_fields = ['quality_threshold', 'max_null_percentage']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Basic range checks
        if not (0 <= config.get('quality_threshold', 0) <= 100):
            return False
        
        if not (0 <= config.get('max_null_percentage', 0) <= 1):
            return False
        
        return True


# Factory function
def create_validation_profile_manager(
    field_mapping_config: Optional[IFieldMappingConfig] = None,
    custom_profiles: Optional[Dict[str, Dict[str, Any]]] = None
) -> ValidationProfileManager:
    """
    Factory function to create validation profile manager.
    
    Args:
        field_mapping_config: Field mapping configuration
        custom_profiles: Custom profile definitions
        
    Returns:
        ValidationProfileManager instance
    """
    return ValidationProfileManager(
        field_mapping_config=field_mapping_config,
        custom_profiles=custom_profiles
    )
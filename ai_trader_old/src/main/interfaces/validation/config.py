"""
Validation Framework - Configuration Interfaces

Configuration interfaces for validation framework settings,
profiles, and dynamic configuration management.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import ValidationStage


class ValidationProfile(Enum):
    """Validation profile types."""

    STRICT = "strict"  # Strict validation with low tolerance
    STANDARD = "standard"  # Standard validation rules
    LENIENT = "lenient"  # Lenient validation for historical data
    DEVELOPMENT = "development"  # Development-time validation
    PRODUCTION = "production"  # Production validation
    TESTING = "testing"  # Testing validation
    CUSTOM = "custom"  # Custom profile


class ConfigurationScope(Enum):
    """Configuration scope levels."""

    GLOBAL = "global"  # Global settings
    LAYER = "layer"  # Data layer specific
    DATA_TYPE = "data_type"  # Data type specific
    SOURCE = "source"  # Data source specific
    STAGE = "stage"  # Validation stage specific
    SYMBOL = "symbol"  # Symbol specific


class IValidationConfig(ABC):
    """Interface for validation configuration."""

    @property
    @abstractmethod
    def profile(self) -> ValidationProfile:
        """Validation profile."""
        pass

    @property
    @abstractmethod
    def scope(self) -> ConfigurationScope:
        """Configuration scope."""
        pass

    @abstractmethod
    async def get_setting(self, setting_name: str, default: Any | None = None) -> Any:
        """Get configuration setting."""
        pass

    @abstractmethod
    async def set_setting(self, setting_name: str, value: Any) -> None:
        """Set configuration setting."""
        pass

    @abstractmethod
    async def get_all_settings(self) -> dict[str, Any]:
        """Get all configuration settings."""
        pass

    @abstractmethod
    async def validate_configuration(self) -> tuple[bool, list[str]]:
        """Validate configuration is correct."""
        pass

    @abstractmethod
    async def merge_configuration(
        self, other_config: "IValidationConfig", merge_strategy: str = "override"
    ) -> "IValidationConfig":
        """Merge with another configuration."""
        pass


class IValidationProfileManager(ABC):
    """Interface for validation profile management."""

    @abstractmethod
    async def create_profile(
        self,
        profile_name: str,
        base_profile: ValidationProfile | None = None,
        custom_settings: dict[str, Any] | None = None,
    ) -> str:
        """Create validation profile."""
        pass

    @abstractmethod
    async def get_profile(self, profile_name: str) -> IValidationConfig | None:
        """Get validation profile."""
        pass

    @abstractmethod
    async def update_profile(self, profile_name: str, settings_update: dict[str, Any]) -> bool:
        """Update validation profile."""
        pass

    @abstractmethod
    async def delete_profile(self, profile_name: str) -> bool:
        """Delete validation profile."""
        pass

    @abstractmethod
    async def list_profiles(self) -> list[dict[str, Any]]:
        """List all validation profiles."""
        pass

    @abstractmethod
    async def get_profile_for_context(
        self,
        layer: DataLayer,
        data_type: DataType,
        source: str | None = None,
        stage: ValidationStage | None = None,
    ) -> IValidationConfig:
        """Get profile for validation context."""
        pass


class IFieldMappingConfig(ABC):
    """Interface for field mapping configuration."""

    @abstractmethod
    async def get_field_mapping(self, source: str, data_type: DataType) -> dict[str, str]:
        """Get field mapping for source and data type."""
        pass

    @abstractmethod
    async def set_field_mapping(
        self, source: str, data_type: DataType, field_mapping: dict[str, str]
    ) -> None:
        """Set field mapping for source and data type."""
        pass

    @abstractmethod
    async def get_reverse_mapping(self, source: str, data_type: DataType) -> dict[str, str]:
        """Get reverse field mapping."""
        pass

    @abstractmethod
    async def validate_field_mapping(
        self, source: str, data_type: DataType, sample_data: dict[str, Any] | None = None
    ) -> tuple[bool, list[str]]:
        """Validate field mapping."""
        pass

    @abstractmethod
    async def get_required_fields(self, data_type: DataType, layer: DataLayer) -> list[str]:
        """Get required fields for data type and layer."""
        pass

    @abstractmethod
    async def get_optional_fields(self, data_type: DataType, layer: DataLayer) -> list[str]:
        """Get optional fields for data type and layer."""
        pass


class IValidationRulesConfig(ABC):
    """Interface for validation rules configuration."""

    @abstractmethod
    async def get_rules_for_stage(
        self, stage: ValidationStage, data_type: DataType, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Get validation rules for stage."""
        pass

    @abstractmethod
    async def get_quality_thresholds(
        self, data_type: DataType, layer: DataLayer, profile: ValidationProfile
    ) -> dict[str, float]:
        """Get quality thresholds."""
        pass

    @abstractmethod
    async def get_field_validation_rules(
        self, field_name: str, data_type: DataType, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get validation rules for specific field."""
        pass

    @abstractmethod
    async def get_business_rules(
        self, data_type: DataType, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get business validation rules."""
        pass

    @abstractmethod
    async def update_validation_rules(
        self,
        stage: ValidationStage,
        data_type: DataType,
        rules_update: list[dict[str, Any]],
        source: str | None = None,
    ) -> None:
        """Update validation rules."""
        pass


class IConfigurationProvider(ABC):
    """Interface for configuration provider."""

    @abstractmethod
    async def load_configuration(
        self, config_path: str | None = None, environment: str | None = None
    ) -> IValidationConfig:
        """Load validation configuration."""
        pass

    @abstractmethod
    async def save_configuration(
        self, config: IValidationConfig, config_path: str | None = None
    ) -> None:
        """Save validation configuration."""
        pass

    @abstractmethod
    async def reload_configuration(self) -> None:
        """Reload configuration from source."""
        pass

    @abstractmethod
    async def watch_configuration_changes(self, callback: callable) -> str:
        """Watch for configuration changes."""
        pass

    @abstractmethod
    async def stop_watching(self, watch_id: str) -> None:
        """Stop watching configuration changes."""
        pass


class IDynamicConfigManager(ABC):
    """Interface for dynamic configuration management."""

    @abstractmethod
    async def get_dynamic_setting(
        self, setting_name: str, context: dict[str, Any], default: Any | None = None
    ) -> Any:
        """Get dynamic setting based on context."""
        pass

    @abstractmethod
    async def update_dynamic_setting(
        self,
        setting_name: str,
        value: Any,
        conditions: dict[str, Any] | None = None,
        ttl: timedelta | None = None,
    ) -> None:
        """Update dynamic setting with conditions."""
        pass

    @abstractmethod
    async def create_configuration_rule(
        self, rule_name: str, condition: str, configuration_overrides: dict[str, Any]
    ) -> str:
        """Create configuration rule."""
        pass

    @abstractmethod
    async def evaluate_configuration_rules(self, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate configuration rules for context."""
        pass

    @abstractmethod
    async def get_effective_configuration(
        self, base_config: IValidationConfig, context: dict[str, Any]
    ) -> IValidationConfig:
        """Get effective configuration after applying rules."""
        pass


class IEnvironmentConfig(ABC):
    """Interface for environment-specific configuration."""

    @abstractmethod
    async def get_environment_settings(self, environment: str) -> dict[str, Any]:
        """Get environment-specific settings."""
        pass

    @abstractmethod
    async def set_environment_setting(
        self, environment: str, setting_name: str, value: Any
    ) -> None:
        """Set environment-specific setting."""
        pass

    @abstractmethod
    async def promote_configuration(
        self,
        source_environment: str,
        target_environment: str,
        settings_filter: list[str] | None = None,
    ) -> None:
        """Promote configuration between environments."""
        pass

    @abstractmethod
    async def compare_environments(
        self, environment1: str, environment2: str
    ) -> dict[str, dict[str, Any]]:
        """Compare configuration between environments."""
        pass


class IConfigurationValidator(ABC):
    """Interface for configuration validation."""

    @abstractmethod
    async def validate_profile_settings(
        self, profile_settings: dict[str, Any], profile_type: ValidationProfile
    ) -> tuple[bool, list[str]]:
        """Validate profile settings."""
        pass

    @abstractmethod
    async def validate_field_mappings(
        self, field_mappings: dict[str, dict[str, str]]
    ) -> tuple[bool, list[str]]:
        """Validate field mappings."""
        pass

    @abstractmethod
    async def validate_validation_rules(
        self, validation_rules: list[dict[str, Any]]
    ) -> tuple[bool, list[str]]:
        """Validate validation rules configuration."""
        pass

    @abstractmethod
    async def validate_configuration_consistency(
        self, config: IValidationConfig
    ) -> tuple[bool, list[str]]:
        """Validate configuration consistency."""
        pass

    @abstractmethod
    async def suggest_configuration_improvements(
        self, config: IValidationConfig, usage_patterns: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Suggest configuration improvements."""
        pass


class IConfigurationAuditor(ABC):
    """Interface for configuration auditing."""

    @abstractmethod
    async def log_configuration_change(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
        user: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Log configuration change."""
        pass

    @abstractmethod
    async def get_configuration_history(
        self, setting_name: str | None = None, time_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Get configuration change history."""
        pass

    @abstractmethod
    async def create_configuration_snapshot(
        self, snapshot_name: str, config: IValidationConfig
    ) -> str:
        """Create configuration snapshot."""
        pass

    @abstractmethod
    async def restore_from_snapshot(self, snapshot_id: str) -> IValidationConfig:
        """Restore configuration from snapshot."""
        pass

    @abstractmethod
    async def compare_configurations(
        self, config1: IValidationConfig, config2: IValidationConfig
    ) -> dict[str, dict[str, Any]]:
        """Compare two configurations."""
        pass

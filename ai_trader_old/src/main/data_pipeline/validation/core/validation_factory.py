"""
Validation Framework Factory

Factory for creating validation framework components with proper
dependency injection and configuration management.
"""

# Standard library imports
from typing import Any

# Local imports
# Config imports
from main.config import get_config_manager
from main.config.validation_models import DataPipelineConfig

# Interface imports
from main.interfaces.validation import (
    ICoverageAnalyzer,
    IDataCleaner,
    IDataQualityCalculator,
    IValidationPipeline,
    IValidator,
)
from main.interfaces.validation.config import IValidationConfig, IValidationProfileManager
from main.interfaces.validation.metrics import IValidationMetricsCollector
from main.interfaces.validation.rules import IRuleEngine
from main.interfaces.validation.validators import (
    IFeatureValidator,
    IMarketDataValidator,
    IRecordValidator,
)
from main.utils.core import get_logger

from .validation_pipeline import ValidationPipeline

logger = get_logger(__name__)


class ValidationFactory:
    """
    Factory for creating validation framework components.

    Provides centralized creation of validation components with proper
    dependency injection and configuration management.
    """

    def __init__(
        self,
        profile_manager: IValidationProfileManager | None = None,
        default_profile: str = "standard",
        use_config_manager: bool = True,
    ):
        """
        Initialize the validation factory.

        Args:
            profile_manager: Optional profile manager (for backward compatibility)
            default_profile: Default validation profile to use
            use_config_manager: Whether to use ConfigManager for configuration
        """
        self.profile_manager = profile_manager
        self.default_profile = default_profile
        self.use_config_manager = use_config_manager
        self._component_cache: dict[str, Any] = {}

        # Get config from ConfigManager if enabled
        if use_config_manager:
            try:
                config_manager = get_config_manager()
                config = config_manager.load_config("unified_config")
                if hasattr(config, "orchestrator") and hasattr(
                    config.orchestrator, "data_pipeline"
                ):
                    self.validation_config = config.orchestrator.data_pipeline.validation
                else:
                    self.validation_config = DataPipelineConfig.ValidationConfig()
            except Exception as e:
                logger.warning(f"Failed to load config from ConfigManager: {e}, using defaults")
                self.validation_config = DataPipelineConfig.ValidationConfig()
        else:
            self.validation_config = None

        logger.info(f"Initialized ValidationFactory with profile: {default_profile}")

    def create_validator(
        self, validator_type: str, config: dict[str, Any] | None = None
    ) -> IValidator:
        """Create validator instance."""
        cache_key = f"validator_{validator_type}_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Create validator based on type
        if validator_type == "market_data":
            validator = self._create_market_data_validator(config)
        elif validator_type == "feature":
            validator = self._create_feature_validator(config)
        elif validator_type == "record":
            validator = self._create_record_validator(config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")

        self._component_cache[cache_key] = validator
        return validator

    def create_pipeline(self, config: dict[str, Any] | None = None) -> IValidationPipeline:
        """Create validation pipeline."""
        cache_key = f"pipeline_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Get validation configuration
        validation_config = self._get_validation_config(config)

        # Create required components
        market_data_validator = self._create_market_data_validator(config)
        feature_validator = self._create_feature_validator(config)
        metrics_collector = self._create_metrics_collector(config)
        rule_engine = self._create_rule_engine(config)

        # Create pipeline with dependency injection
        pipeline = ValidationPipeline(
            market_data_validator=market_data_validator,
            feature_validator=feature_validator,
            metrics_collector=metrics_collector,
            rule_engine=rule_engine,
            config=validation_config,
        )

        self._component_cache[cache_key] = pipeline
        return pipeline

    def create_quality_calculator(
        self, config: dict[str, Any] | None = None
    ) -> IDataQualityCalculator:
        """Create data quality calculator."""
        cache_key = f"quality_calculator_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Create quality calculator instance
        from ..quality.data_quality_calculator import DataQualityCalculator

        calculator = DataQualityCalculator(config or {})

        self._component_cache[cache_key] = calculator
        return calculator

    def create_data_cleaner(self, config: dict[str, Any] | None = None) -> IDataCleaner:
        """Create data cleaner."""
        cache_key = f"data_cleaner_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Create data cleaner instance
        from ..quality.data_cleaner import QualityDataCleaner

        cleaner = QualityDataCleaner(config or {})

        self._component_cache[cache_key] = cleaner
        return cleaner

    def create_coverage_analyzer(self, config: dict[str, Any] | None = None) -> ICoverageAnalyzer:
        """Create coverage analyzer."""
        cache_key = f"coverage_analyzer_{hash(str(config))}"

        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        # Create coverage analyzer instance
        from ..coverage.data_coverage_analyzer import DataCoverageAnalyzer

        analyzer = DataCoverageAnalyzer(config or {})

        self._component_cache[cache_key] = analyzer
        return analyzer

    def _create_market_data_validator(
        self, config: dict[str, Any] | None = None
    ) -> IMarketDataValidator:
        """Create market data validator."""
        # Create market data validator instance
        from ..validators.market_data_validator import MarketDataValidator

        # Use ValidationConfig if available, otherwise fall back to dict
        if self.validation_config:
            return MarketDataValidator(self.validation_config)
        return MarketDataValidator(config or {})

    def _create_feature_validator(self, config: dict[str, Any] | None = None) -> IFeatureValidator:
        """Create feature validator."""
        # Create feature validator instance
        from ..validators.feature_validator import FeatureValidator

        # Use ValidationConfig if available, otherwise fall back to dict
        if self.validation_config:
            return FeatureValidator(self.validation_config)
        return FeatureValidator(config or {})

    def _create_record_validator(self, config: dict[str, Any] | None = None) -> IRecordValidator:
        """Create record validator."""
        # Create record validator instance
        from ..validators.record_validator import RecordValidator

        # Use ValidationConfig if available, otherwise fall back to dict
        if self.validation_config:
            return RecordValidator(self.validation_config)
        return RecordValidator(config or {})

    def _create_metrics_collector(
        self, config: dict[str, Any] | None = None
    ) -> IValidationMetricsCollector:
        """Create metrics collector."""
        # Create metrics collector instance
        from ..metrics.validation_metrics import ValidationMetricsCollector

        return ValidationMetricsCollector(config or {})

    def _create_rule_engine(self, config: dict[str, Any] | None = None) -> IRuleEngine:
        """Create rule engine."""
        # Create rule engine instance
        from ..config.validation_rules_engine import ValidationRulesEngine

        return ValidationRulesEngine(config_path=config.get("config_path") if config else None)

    async def _get_validation_config(
        self, config: dict[str, Any] | None = None
    ) -> IValidationConfig:
        """Get validation configuration."""
        profile_name = (
            config.get("profile", self.default_profile) if config else self.default_profile
        )

        validation_config = await self.profile_manager.get_profile(profile_name)
        if not validation_config:
            # Fallback to default profile
            validation_config = await self.profile_manager.get_profile("standard")

        if not validation_config:
            raise ValueError(f"No validation configuration found for profile: {profile_name}")

        return validation_config

    def clear_cache(self) -> None:
        """Clear component cache."""
        self._component_cache.clear()
        logger.info("Cleared validation factory cache")


# Convenience functions for creating common components
async def create_validation_pipeline(
    profile_manager: IValidationProfileManager,
    profile: str = "standard",
    config: dict[str, Any] | None = None,
) -> IValidationPipeline:
    """
    Convenience function to create a validation pipeline.

    Args:
        profile_manager: Profile manager for configuration
        profile: Validation profile to use
        config: Additional configuration options

    Returns:
        Configured validation pipeline
    """
    factory = ValidationFactory(profile_manager, profile)
    return factory.create_pipeline(config)


async def create_unified_validator(
    profile_manager: IValidationProfileManager,
    profile: str = "standard",
    config: dict[str, Any] | None = None,
) -> dict[str, IValidator]:
    """
    Convenience function to create unified validator components.

    Args:
        profile_manager: Profile manager for configuration
        profile: Validation profile to use
        config: Additional configuration options

    Returns:
        Dictionary of validator components
    """
    factory = ValidationFactory(profile_manager, profile)

    return {
        "market_data": factory.create_validator("market_data", config),
        "feature": factory.create_validator("feature", config),
        "record": factory.create_validator("record", config),
        "quality_calculator": factory.create_quality_calculator(config),
        "data_cleaner": factory.create_data_cleaner(config),
        "coverage_analyzer": factory.create_coverage_analyzer(config),
    }

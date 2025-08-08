"""
Validation Stage Factory - Interface Implementation

Factory for creating stage-specific validators using the interface-based
architecture with dependency injection.
"""

from typing import Dict, Any, Optional
from main.utils.core import get_logger

# Interface imports
from main.interfaces.validation import (
    ValidationStage,
    IValidator
)
from main.interfaces.validation.validators import (
    IRecordValidator,
    IMarketDataValidator,
    IFeatureValidator
)
from main.interfaces.validation.rules import IRuleEngine
from main.interfaces.validation.config import IValidationConfig

# Core imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.utils.core import get_logger

logger = get_logger(__name__)


class ValidationStageFactory:
    """
    Factory for creating stage-specific validators.
    
    Creates validator instances configured for specific validation stages
    using dependency injection and interface-based architecture.
    """
    
    def __init__(
        self,
        market_data_validator: IMarketDataValidator,
        feature_validator: IFeatureValidator,
        record_validator: IRecordValidator,
        rule_engine: IRuleEngine,
        config: IValidationConfig
    ):
        """
        Initialize the validation stage factory.
        
        Args:
            market_data_validator: Market data validator implementation
            feature_validator: Feature validator implementation
            record_validator: Record validator implementation
            rule_engine: Rule engine implementation
            config: Validation configuration
        """
        self.market_data_validator = market_data_validator
        self.feature_validator = feature_validator
        self.record_validator = record_validator
        self.rule_engine = rule_engine
        self.config = config
        
        logger.info("Initialized ValidationStageFactory with interface-based architecture")
    
    def create_stage_validator(
        self,
        stage: ValidationStage,
        data_type: DataType,
        layer: DataLayer,
        source: Optional[str] = None
    ) -> IValidator:
        """
        Create a validator for a specific stage.
        
        Args:
            stage: Validation stage
            data_type: Type of data being validated
            layer: Data layer being validated
            source: Optional data source
            
        Returns:
            Configured validator for the stage
            
        Raises:
            ValueError: If unknown validation stage is provided
        """
        if stage == ValidationStage.INGEST:
            return self._create_ingest_validator(data_type, source)
        elif stage == ValidationStage.POST_ETL:
            return self._create_post_etl_validator(data_type, layer)
        elif stage == ValidationStage.FEATURE_READY:
            return self._create_feature_ready_validator(data_type, layer)
        elif stage == ValidationStage.PRE_STORAGE:
            return self._create_pre_storage_validator(data_type, layer)
        elif stage == ValidationStage.POST_STORAGE:
            return self._create_post_storage_validator(data_type, layer)
        else:
            raise ValueError(f"Unknown validation stage: {stage}")
    
    def _create_ingest_validator(
        self,
        data_type: DataType,
        source: Optional[str] = None
    ) -> IValidator:
        """Create validator for ingestion stage."""
        if data_type in [DataType.MARKET_DATA, DataType.FINANCIALS]:
            return StageConfiguredValidator(
                base_validator=self.market_data_validator,
                stage=ValidationStage.INGEST,
                data_type=data_type,
                source=source,
                rule_engine=self.rule_engine
            )
        else:
            return StageConfiguredValidator(
                base_validator=self.record_validator,
                stage=ValidationStage.INGEST,
                data_type=data_type,
                source=source,
                rule_engine=self.rule_engine
            )
    
    def _create_post_etl_validator(
        self,
        data_type: DataType,
        layer: DataLayer
    ) -> IValidator:
        """Create validator for post-ETL stage."""
        if data_type in [DataType.MARKET_DATA, DataType.FINANCIALS]:
            return StageConfiguredValidator(
                base_validator=self.market_data_validator,
                stage=ValidationStage.POST_ETL,
                data_type=data_type,
                rule_engine=self.rule_engine
            )
        else:
            return StageConfiguredValidator(
                base_validator=self.record_validator,
                stage=ValidationStage.POST_ETL,
                data_type=data_type,
                rule_engine=self.rule_engine
            )
    
    def _create_feature_ready_validator(
        self,
        data_type: DataType,
        layer: DataLayer
    ) -> IValidator:
        """Create validator for feature-ready stage."""
        return StageConfiguredValidator(
            base_validator=self.feature_validator,
            stage=ValidationStage.FEATURE_READY,
            data_type=data_type,
            rule_engine=self.rule_engine
        )
    
    def _create_pre_storage_validator(
        self,
        data_type: DataType,
        layer: DataLayer
    ) -> IValidator:
        """Create validator for pre-storage stage."""
        return StageConfiguredValidator(
            base_validator=self.record_validator,
            stage=ValidationStage.PRE_STORAGE,
            data_type=data_type,
            rule_engine=self.rule_engine
        )
    
    def _create_post_storage_validator(
        self,
        data_type: DataType,
        layer: DataLayer
    ) -> IValidator:
        """Create validator for post-storage stage."""
        return StageConfiguredValidator(
            base_validator=self.record_validator,
            stage=ValidationStage.POST_STORAGE,
            data_type=data_type,
            rule_engine=self.rule_engine
        )


class StageConfiguredValidator:
    """
    Wrapper validator that configures a base validator for a specific stage.
    
    This validator wraps a base validator implementation and configures it
    with stage-specific rules and context.
    """
    
    def __init__(
        self,
        base_validator: IValidator,
        stage: ValidationStage,
        data_type: DataType,
        rule_engine: IRuleEngine,
        source: Optional[str] = None
    ):
        """
        Initialize stage-configured validator.
        
        Args:
            base_validator: Base validator implementation
            stage: Validation stage
            data_type: Data type being validated
            rule_engine: Rule engine for getting stage-specific rules
            source: Optional data source
        """
        self.base_validator = base_validator
        self.stage = stage
        self.data_type = data_type
        self.rule_engine = rule_engine
        self.source = source
        
        logger.debug(f"Created {stage.value} validator for {data_type.value}")
    
    async def validate(self, data: Any, context) -> Any:
        """Validate data using base validator with stage configuration."""
        # Add stage-specific context
        stage_context = self._create_stage_context(context)
        
        # Delegate to base validator
        return await self.base_validator.validate(data, stage_context)
    
    async def get_validation_rules(self, context) -> list:
        """Get validation rules for this stage."""
        return await self.base_validator.get_validation_rules(context)
    
    async def is_applicable(self, context) -> bool:
        """Check if validator applies to context."""
        # Check if context matches our stage and data type
        if hasattr(context, 'stage') and context.stage != self.stage:
            return False
        if hasattr(context, 'data_type') and context.data_type != self.data_type:
            return False
        
        return await self.base_validator.is_applicable(context)
    
    def _create_stage_context(self, base_context):
        """Create stage-specific validation context."""
        # This would create a context with stage-specific information
        # For now, pass through the base context
        # TODO: Implement proper context enhancement
        return base_context


# Convenience functions for creating stage validators
def create_ingest_validator(
    market_data_validator: IMarketDataValidator,
    record_validator: IRecordValidator,
    rule_engine: IRuleEngine,
    config: IValidationConfig,
    data_type: DataType = DataType.MARKET_DATA,
    source: Optional[str] = None
) -> IValidator:
    """Create validator for ingestion stage."""
    factory = ValidationStageFactory(
        market_data_validator=market_data_validator,
        feature_validator=None,  # Not needed for ingest
        record_validator=record_validator,
        rule_engine=rule_engine,
        config=config
    )
    
    return factory.create_stage_validator(
        stage=ValidationStage.INGEST,
        data_type=data_type,
        layer=DataLayer.RAW,
        source=source
    )


def create_post_etl_validator(
    market_data_validator: IMarketDataValidator,
    record_validator: IRecordValidator,
    rule_engine: IRuleEngine,
    config: IValidationConfig,
    data_type: DataType = DataType.MARKET_DATA
) -> IValidator:
    """Create validator for post-ETL stage."""
    factory = ValidationStageFactory(
        market_data_validator=market_data_validator,
        feature_validator=None,  # Not needed for post-ETL
        record_validator=record_validator,
        rule_engine=rule_engine,
        config=config
    )
    
    return factory.create_stage_validator(
        stage=ValidationStage.POST_ETL,
        data_type=data_type,
        layer=DataLayer.PROCESSED
    )


def create_feature_ready_validator(
    feature_validator: IFeatureValidator,
    rule_engine: IRuleEngine,
    config: IValidationConfig,
    data_type: DataType = DataType.MARKET_DATA
) -> IValidator:
    """Create validator for feature-ready stage."""
    factory = ValidationStageFactory(
        market_data_validator=None,  # Not needed for features
        feature_validator=feature_validator,
        record_validator=None,  # Not needed for features
        rule_engine=rule_engine,
        config=config
    )
    
    return factory.create_stage_validator(
        stage=ValidationStage.FEATURE_READY,
        data_type=data_type,
        layer=DataLayer.FEATURE
    )
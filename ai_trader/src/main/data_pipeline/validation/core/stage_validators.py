"""
Stage-specific validators for the validation pipeline.

Contains validators for different validation stages: Ingest, PostETL, FeatureReady.
"""

from typing import Any, List
from datetime import datetime
import pandas as pd

from main.interfaces.validation import (
    IValidationResult,
    IValidationContext,
    IValidator,
    ValidationStage
)
from main.interfaces.validation.validators import (
    IMarketDataValidator,
    IFeatureValidator
)
from main.interfaces.validation.rules import IRuleEngine

from main.data_pipeline.core.enums import DataType
from main.data_pipeline.validation.core.validation_types import ValidationResult
from main.utils.core import get_logger

logger = get_logger(__name__)


class IngestValidator:
    """Validator for ingestion stage."""
    
    def __init__(
        self,
        market_data_validator: IMarketDataValidator,
        rule_engine: IRuleEngine
    ):
        self.market_data_validator = market_data_validator
        self.rule_engine = rule_engine
    
    async def validate(
        self,
        data: Any,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate ingestion data."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Get applicable rules
            rules = await self.rule_engine.get_applicable_rules(context)
            
            # Route to appropriate validator based on data type
            if context.data_type == DataType.MARKET_DATA:
                result = await self.market_data_validator.validate_ohlcv_data(
                    data, context.symbol, context
                )
                return result
            
            elif context.data_type == DataType.NEWS:
                # Handle news validation
                metrics['data_type'] = 'news'
                metrics['record_count'] = len(data) if hasattr(data, '__len__') else 1
                
            elif context.data_type == DataType.FINANCIALS:
                # Handle fundamentals validation
                metrics['data_type'] = 'fundamentals'
                metrics['record_count'] = len(data) if hasattr(data, '__len__') else 1
                
            else:
                errors.append(f"Unknown data type for ingestion: {context.data_type}")
            
        except Exception as e:
            errors.append(f"Ingestion validation error: {str(e)}")
        
        passed = len(errors) == 0
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=context.stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=start_time,
            duration_ms=duration_ms
        )
    
    async def get_validation_rules(
        self,
        context: IValidationContext
    ) -> List[str]:
        """Get applicable validation rules."""
        return [f"ingest_{context.data_type.value}_validation"]
    
    async def is_applicable(
        self,
        context: IValidationContext
    ) -> bool:
        """Check if validator applies to context."""
        return context.stage == ValidationStage.INGEST


class PostETLValidator:
    """Validator for post-ETL stage."""
    
    def __init__(
        self,
        market_data_validator: IMarketDataValidator,
        rule_engine: IRuleEngine
    ):
        self.market_data_validator = market_data_validator
        self.rule_engine = rule_engine
    
    async def validate(
        self,
        data: Any,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate post-ETL data."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}
        
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                errors.append(f"Post-ETL data must be a non-empty DataFrame, got {type(data)}")
            else:
                # Validate DataFrame structure and quality
                result = await self.market_data_validator.validate_ohlcv_data(
                    data, context.symbol, context
                )
                return result
                
        except Exception as e:
            errors.append(f"Post-ETL validation error: {str(e)}")
        
        passed = len(errors) == 0
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=context.stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=start_time,
            duration_ms=duration_ms
        )
    
    async def get_validation_rules(
        self,
        context: IValidationContext
    ) -> List[str]:
        """Get applicable validation rules."""
        return [f"post_etl_{context.data_type.value}_validation"]
    
    async def is_applicable(
        self,
        context: IValidationContext
    ) -> bool:
        """Check if validator applies to context."""
        return context.stage == ValidationStage.POST_ETL


class FeatureReadyValidator:
    """Validator for feature-ready stage."""
    
    def __init__(
        self,
        feature_validator: IFeatureValidator,
        rule_engine: IRuleEngine
    ):
        self.feature_validator = feature_validator
        self.rule_engine = rule_engine
    
    async def validate(
        self,
        data: Any,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate feature-ready data."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {}
        
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                errors.append(f"Feature-ready data must be a non-empty DataFrame, got {type(data)}")
            else:
                # Use feature validator
                result = await self.feature_validator.validate_feature_dataframe(
                    data, context
                )
                return result
                
        except Exception as e:
            errors.append(f"Feature-ready validation error: {str(e)}")
        
        passed = len(errors) == 0
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=context.stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=start_time,
            duration_ms=duration_ms
        )
    
    async def get_validation_rules(
        self,
        context: IValidationContext
    ) -> List[str]:
        """Get applicable validation rules."""
        return [f"feature_ready_{context.data_type.value}_validation"]
    
    async def is_applicable(
        self,
        context: IValidationContext
    ) -> bool:
        """Check if validator applies to context."""
        return context.stage == ValidationStage.FEATURE_READY
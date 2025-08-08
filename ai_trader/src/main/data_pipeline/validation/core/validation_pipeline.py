"""
Multi-Stage Validation Pipeline - Interface Implementation

Orchestrates a robust, multi-stage validation system for data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from main.interfaces.validation import (
    IValidationPipeline,
    IValidationResult,
    IValidationContext,
    IValidator,
    ValidationStage,
    ValidationSeverity
)
from main.interfaces.validation.validators import (
    IMarketDataValidator,
    IFeatureValidator
)
from main.interfaces.validation.metrics import IValidationMetricsCollector
from main.interfaces.validation.rules import IRuleEngine
from main.interfaces.validation.config import IValidationConfig

from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.validation.core.validation_types import ValidationResult, ValidationContext
from main.data_pipeline.validation.core.stage_validators import (
    IngestValidator,
    PostETLValidator,
    FeatureReadyValidator
)
from main.utils.core import get_logger

logger = get_logger(__name__)


class ValidationPipeline:
    """
    Main validation pipeline implementation.
    
    Orchestrates multi-stage validation across the data pipeline using
    dependency injection and interface-based architecture.
    """
    
    def __init__(
        self,
        market_data_validator: IMarketDataValidator,
        feature_validator: IFeatureValidator,
        metrics_collector: IValidationMetricsCollector,
        rule_engine: IRuleEngine,
        config: IValidationConfig
    ):
        """
        Initialize the validation pipeline with injected dependencies.
        
        Args:
            market_data_validator: Market data validation implementation
            feature_validator: Feature validation implementation
            metrics_collector: Metrics collection implementation
            rule_engine: Rule engine implementation
            config: Validation configuration
        """
        self.market_data_validator = market_data_validator
        self.feature_validator = feature_validator
        self.metrics_collector = metrics_collector
        self.rule_engine = rule_engine
        self.config = config
        
        # Stage-specific validators
        self._stage_validators: Dict[ValidationStage, IValidator] = {
            ValidationStage.INGEST: IngestValidator(
                market_data_validator=market_data_validator,
                rule_engine=rule_engine
            ),
            ValidationStage.POST_ETL: PostETLValidator(
                market_data_validator=market_data_validator,
                rule_engine=rule_engine
            ),
            ValidationStage.FEATURE_READY: FeatureReadyValidator(
                feature_validator=feature_validator,
                rule_engine=rule_engine
            )
        }
        
        self.results_history: List[IValidationResult] = []
        logger.info("Initialized validation pipeline with interface-based architecture")
    
    async def validate_stage(
        self,
        stage: ValidationStage,
        data: Any,
        layer: DataLayer,
        data_type: DataType,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IValidationResult:
        """Validate data for a specific stage."""
        start_time = datetime.now()
        
        # Create validation context
        context = ValidationContext(
            stage=stage,
            layer=layer,
            data_type=data_type,
            symbol=symbol,
            source=source,
            metadata=metadata
        )
        
        # Get stage-specific validator
        validator = self._stage_validators.get(stage)
        if not validator:
            errors = [f"No validator found for stage: {stage}"]
            return self._create_result(
                stage, False, errors, [], {}, start_time
            )
        
        try:
            # Execute validation
            result = await validator.validate(data, context)
            
            # Record metrics
            await self.metrics_collector.record_validation_metrics(
                stage, result.metadata, context
            )
            
            # Store in history
            self.results_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for stage {stage}: {e}", exc_info=True)
            errors = [f"Validation error: {str(e)}"]
            return self._create_result(
                stage, False, errors, [], {}, start_time
            )
    
    async def validate_batch(
        self,
        stage: ValidationStage,
        data_batch: List[Any],
        layer: DataLayer,
        data_type: DataType,
        symbols: Optional[List[str]] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[IValidationResult]:
        """Validate a batch of data."""
        results = []
        
        for i, data_item in enumerate(data_batch):
            symbol = symbols[i] if symbols and i < len(symbols) else None
            
            result = await self.validate_stage(
                stage=stage,
                data=data_item,
                layer=layer,
                data_type=data_type,
                symbol=symbol,
                source=source,
                metadata=metadata
            )
            results.append(result)
        
        return results
    
    async def get_stage_validators(
        self,
        stage: ValidationStage,
        layer: DataLayer,
        data_type: DataType
    ) -> List[IValidator]:
        """Get validators for a specific stage and context."""
        validator = self._stage_validators.get(stage)
        return [validator] if validator else []
    
    # Convenience methods for specific stages
    async def validate_ingest(
        self,
        data: Any,
        data_type: DataType,
        source: str,
        symbol: Optional[str] = None
    ) -> IValidationResult:
        """Validate raw data at ingestion point."""
        return await self.validate_stage(
            stage=ValidationStage.INGEST,
            data=data,
            layer=DataLayer.RAW,
            data_type=data_type,
            symbol=symbol,
            source=source
        )
    
    async def validate_post_etl(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        symbol: Optional[str] = None,
        expected_freq: str = 'D'
    ) -> IValidationResult:
        """Validate data after ETL transformations."""
        metadata = {'expected_freq': expected_freq}
        
        return await self.validate_stage(
            stage=ValidationStage.POST_ETL,
            data=data,
            layer=DataLayer.PROCESSED,
            data_type=data_type,
            symbol=symbol,
            metadata=metadata
        )
    
    async def validate_feature_ready(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        symbol: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
        min_rows: Optional[int] = None
    ) -> IValidationResult:
        """Validate data before feature calculation."""
        metadata = {}
        if required_columns:
            metadata['required_columns'] = required_columns
        if min_rows:
            metadata['min_rows'] = min_rows
        
        return await self.validate_stage(
            stage=ValidationStage.FEATURE_READY,
            data=data,
            layer=DataLayer.FEATURE,
            data_type=data_type,
            symbol=symbol,
            metadata=metadata
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.results_history:
            return {"message": "No validation results available"}
        
        summary = {
            'total_validation_runs': len(self.results_history),
            'passed_runs': sum(1 for r in self.results_history if r.passed),
            'failed_runs': sum(1 for r in self.results_history if not r.passed),
            'stages': {}
        }
        
        for stage in ValidationStage:
            stage_results = [r for r in self.results_history if r.stage == stage]
            if stage_results:
                summary['stages'][stage.value] = {
                    'total_runs': len(stage_results),
                    'passed_runs': sum(1 for r in stage_results if r.passed),
                    'failed_runs': sum(1 for r in stage_results if not r.passed),
                    'avg_duration_ms': sum(r.duration_ms for r in stage_results) / len(stage_results)
                }
        
        return summary
    
    def clear_history(self) -> None:
        """Clear the validation results history."""
        self.results_history.clear()
        logger.info("Cleared validation history")
    
    def _create_result(
        self,
        stage: ValidationStage,
        passed: bool,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, Any],
        start_time: datetime
    ) -> IValidationResult:
        """Create a validation result."""
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metadata=metrics,  # Map metrics to metadata
            timestamp=start_time,
            duration_ms=duration_ms
        )
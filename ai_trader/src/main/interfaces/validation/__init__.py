"""
Validation Framework Interfaces

This package contains interface definitions for the validation framework,
organized by functional area to enable flexible implementation and testing.

Interface Organization:
- validators.py: Specific validator interfaces (record, feature, market data, etc.)
- quality.py: Data quality assessment and monitoring interfaces
- pipeline.py: Validation pipeline orchestration interfaces
- metrics.py: Validation metrics collection and analysis interfaces
- rules.py: Rule engine and rule management interfaces
- config.py: Configuration and profile management interfaces

Import Guidelines:
- Import specific interfaces directly from their modules
- Use type hints with these interfaces for dependency injection
- Prefer interface types over concrete implementations in signatures

Example Usage:
    from main.interfaces.validation.validators import IRecordValidator, IFeatureValidator
    from main.interfaces.validation.quality import IDataProfiler, IQualityMonitor
    from main.interfaces.validation.pipeline import IValidationWorkflow
    from main.interfaces.validation.metrics import IValidationMetricsCollector
    from main.interfaces.validation.rules import IRuleEngine, IRuleBuilder
    from main.interfaces.validation.config import IValidationConfig, IValidationProfileManager
"""

# Re-export commonly used interfaces for convenience
from main.interfaces.data_pipeline.validation import (
    # Core validation interfaces
    IValidator,
    IValidationPipeline,
    IValidationResult,
    IValidationContext,
    IDataQualityCalculator,
    IDataCleaner,
    ICoverageAnalyzer,
    IValidationMetrics,
    IValidationReporter,
    IValidationRule,
    IRuleEngine,
    IValidationFactory,
    
    # Enums
    ValidationStage,
    ValidationSeverity
)

# Note: Specific interfaces are not re-exported to avoid namespace pollution
# Import them directly from their respective modules as needed

__all__ = [
    # Core interfaces from data_pipeline.validation
    'IValidator',
    'IValidationPipeline', 
    'IValidationResult',
    'IValidationContext',
    'IDataQualityCalculator',
    'IDataCleaner',
    'ICoverageAnalyzer',
    'IValidationMetrics',
    'IValidationReporter',
    'IValidationRule',
    'IRuleEngine',
    'IValidationFactory',
    
    # Enums
    'ValidationStage',
    'ValidationSeverity',
]
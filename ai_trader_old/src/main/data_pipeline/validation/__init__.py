"""
Data Pipeline Validation Framework

This package contains the implementation of the validation framework,
organized by functional area following the interface-based architecture.

Directory Structure:
- core/: Core validation orchestration components
- validators/: Specific validator implementations
- quality/: Data quality assessment and profiling
- metrics/: Validation metrics collection and reporting
- config/: Configuration and rule management
- coverage/: Data coverage analysis
- utils/: Validation utilities and helpers

The validation framework follows interface-based dependency injection patterns
for better testability and flexibility. All components implement interfaces
defined in main.interfaces.validation.

Migration Status: Complete - All components migrated to interface-based architecture.
"""

# Core validation components
# Configuration components
from .config.validation_profile_manager import ValidationProfileManager
from .config.validation_rules_engine import ValidationRulesEngine
from .core.validation_factory import (
    ValidationFactory,
    create_unified_validator,
    create_validation_pipeline,
)
from .core.validation_pipeline import ValidationPipeline
from .core.validation_types import (
    DataQualityError,
    DataValidationError,
    MissingFieldError,
    RuleViolationError,
    ValidationContext,
    ValidationError,
    ValidationResult,
    create_validation_context,
    create_validation_result,
)

# Coverage components
from .coverage.data_coverage_analyzer import DataCoverageAnalyzer
from .metrics.dashboard_generator import DashboardGenerator
from .metrics.prometheus_exporter import PrometheusExporter

# Metrics components
from .metrics.validation_metrics import ValidationMetricsCollector
from .metrics.validation_stats_reporter import (
    ValidationStatsReporter,
    get_validation_stats_reporter,
)
from .quality.data_cleaner import QualityDataCleaner

# Quality components
from .quality.data_quality_calculator import DataQualityCalculator

# Note: ValidationConfig is now in main.config.validation_models
# Import it from there if needed:
# from main.config.validation_models import ValidationConfig
# Utils
from .utils.cache_manager import ValidationCacheManager, get_validation_cache_manager
from .validators.feature_validator import FeatureValidator
from .validators.market_data_validator import MarketDataValidator

# Validators
from .validators.record_validator import RecordValidator

__all__ = [
    # Core
    "ValidationPipeline",
    "ValidationFactory",
    "create_validation_pipeline",
    "create_unified_validator",
    "ValidationResult",
    "ValidationContext",
    "ValidationError",
    "DataValidationError",
    "MissingFieldError",
    "DataQualityError",
    "RuleViolationError",
    "create_validation_result",
    "create_validation_context",
    # Validators
    "RecordValidator",
    "MarketDataValidator",
    "FeatureValidator",
    # Quality
    "DataQualityCalculator",
    "QualityDataCleaner",
    # Metrics
    "ValidationMetricsCollector",
    "PrometheusExporter",
    "DashboardGenerator",
    "ValidationStatsReporter",
    "get_validation_stats_reporter",
    # Coverage
    "DataCoverageAnalyzer",
    # Config
    "ValidationProfileManager",
    "ValidationRulesEngine",
    # Utils
    "ValidationCacheManager",
    "get_validation_cache_manager",
]

"""
Monitoring and orchestrator configuration validation models.
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
from pydantic import BaseModel, Field, confloat, conint, model_validator

logger = logging.getLogger(__name__)

# Custom types
PositiveInt = conint(ge=1)
NonNegativeInt = conint(ge=0)
PositiveFloat = confloat(gt=0.0)
NonNegativeFloat = confloat(ge=0.0)


# Monitoring Configuration Models
class MonitoringConfig(BaseModel):
    """Monitoring configuration with validation."""

    performance: dict[str, Any] = Field(
        default_factory=dict, description="Performance monitoring config"
    )
    alerts: dict[str, Any] = Field(default_factory=dict, description="Alert configuration")
    logging: dict[str, Any] = Field(default_factory=dict, description="Logging configuration")


# Environment-specific overrides
class EnvironmentOverrides(BaseModel):
    """Environment-specific configuration overrides."""

    system: Any | None = None  # Will be SystemConfig from trading.py
    broker: Any | None = None  # Will be BrokerConfig from trading.py
    risk: Any | None = None  # Will be RiskConfig from trading.py
    trading: Any | None = None  # Will be TradingConfig from trading.py
    data: Any | None = None  # Will be DataConfig from features.py


# Main Configuration Model
# Orchestrator Configuration Models
class IntervalsConfig(BaseModel):
    """Orchestrator timing intervals configuration."""

    class DataCollectionConfig(BaseModel):
        """Data collection intervals."""

        market_hours_seconds: PositiveInt = Field(
            default=60, ge=10, le=3600, description="Market hours data collection interval"
        )
        off_hours_seconds: PositiveInt = Field(
            default=300, ge=60, le=7200, description="Off-hours data collection interval"
        )
        realtime_window_minutes: PositiveInt = Field(
            default=5, ge=1, le=60, description="Real-time data window in minutes"
        )

        @model_validator(mode="after")
        def validate_intervals(self):
            """Ensure off-hours interval is not faster than market hours."""
            if self.off_hours_seconds < self.market_hours_seconds:
                raise ValueError("Off-hours interval should be >= market hours interval")
            return self

    class ScannerConfig(BaseModel):
        """Scanner execution intervals."""

        execution_seconds: PositiveInt = Field(
            default=300, ge=30, le=3600, description="Scanner execution interval"
        )
        off_hours_check_seconds: PositiveInt = Field(
            default=600, ge=300, le=7200, description="Off-hours check interval"
        )

    class StrategyConfig(BaseModel):
        """Strategy execution intervals."""

        execution_wait_seconds: PositiveInt = Field(
            default=60, ge=10, le=600, description="Strategy execution wait"
        )
        loop_delay_seconds: PositiveInt = Field(
            default=1, ge=1, le=60, description="Strategy loop delay"
        )

    class RiskMonitoringConfig(BaseModel):
        """Risk monitoring intervals."""

        check_seconds: PositiveInt = Field(
            default=10, ge=1, le=60, description="Risk check interval"
        )
        error_retry_seconds: PositiveInt = Field(
            default=5, ge=1, le=60, description="Error retry interval"
        )

    class PerformanceConfig(BaseModel):
        """Performance tracking intervals."""

        tracking_seconds: PositiveInt = Field(
            default=60, ge=10, le=600, description="Performance tracking interval"
        )
        error_retry_seconds: PositiveInt = Field(
            default=30, ge=5, le=300, description="Error retry interval"
        )

    class HealthCheckConfig(BaseModel):
        """Health check intervals."""

        interval_seconds: PositiveInt = Field(
            default=300, ge=60, le=3600, description="Health check interval"
        )
        error_retry_seconds: PositiveInt = Field(
            default=60, ge=10, le=600, description="Health check error retry"
        )

    class LifecycleConfig(BaseModel):
        """Lifecycle management intervals."""

        daily_cleanup_seconds: PositiveInt = Field(
            default=86400, ge=3600, le=259200, description="Daily cleanup interval"
        )
        error_retry_seconds: PositiveInt = Field(
            default=3600, ge=300, le=43200, description="Lifecycle error retry"
        )

    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk_monitoring: RiskMonitoringConfig = Field(default_factory=RiskMonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)


class LookbackPeriodsConfig(BaseModel):
    """Orchestrator lookback periods configuration."""

    historical_warmup_days: PositiveInt = Field(
        default=30, ge=1, le=365, description="Historical warmup period"
    )
    data_only_mode_days: PositiveInt = Field(
        default=1, ge=1, le=30, description="Data-only mode lookback"
    )
    feature_calculation_days: PositiveInt = Field(
        default=30, ge=1, le=365, description="Feature calculation lookback"
    )


class MarketHoursConfig(BaseModel):
    """Market hours configuration."""

    open_hour_utc: conint(ge=0, le=23) = Field(default=14, description="Market open hour (UTC)")
    open_minute: conint(ge=0, le=59) = Field(default=30, description="Market open minute")
    close_hour_utc: conint(ge=0, le=23) = Field(default=21, description="Market close hour (UTC)")
    close_minute: conint(ge=0, le=59) = Field(default=0, description="Market close minute")

    @model_validator(mode="after")
    def validate_market_hours(self):
        """Ensure market close is after market open."""
        open_minutes = self.open_hour_utc * 60 + self.open_minute
        close_minutes = self.close_hour_utc * 60 + self.close_minute

        if close_minutes <= open_minutes:
            raise ValueError("Market close time must be after market open time")

        # Warn if market hours are unusual
        if (close_minutes - open_minutes) < 360:  # Less than 6 hours
            logger.warning(f"Short market hours: {(close_minutes - open_minutes) / 60:.1f} hours")
        elif (close_minutes - open_minutes) > 780:  # More than 13 hours
            logger.warning(f"Long market hours: {(close_minutes - open_minutes) / 60:.1f} hours")

        return self


class OrchestratorTrainingConfig(BaseModel):
    """Orchestrator training configuration."""

    class LookbackDaysConfig(BaseModel):
        """Training lookback days configuration."""

        default: PositiveInt = Field(
            default=365, ge=30, le=2555, description="Default training lookback days"
        )
        hyperparameter_optimization: PositiveInt = Field(
            default=180, ge=30, le=1095, description="Hyperparameter optimization days"
        )
        default_training: PositiveInt = Field(
            default=60, ge=7, le=365, description="Default training days"
        )
        retraining: PositiveInt = Field(
            default=180, ge=30, le=1095, description="Retraining lookback days"
        )

    class BacktestingConfig(BaseModel):
        """Backtesting configuration."""

        initial_capital: PositiveFloat = Field(
            default=100000.0, ge=1000.0, le=10000000.0, description="Initial capital"
        )
        commission_rate: NonNegativeFloat = Field(
            default=0.001, ge=0.0, le=0.1, description="Commission rate"
        )
        slippage_factor: NonNegativeFloat = Field(
            default=0.0005, ge=0.0, le=0.05, description="Slippage factor"
        )
        retrain_frequency_disable: PositiveInt = Field(
            default=999999, ge=1, description="Retrain frequency disable"
        )

    class DeploymentConfig(BaseModel):
        """Deployment configuration."""

        canary_percentage: confloat(ge=0.0, le=100.0) = Field(
            default=10.0, description="Canary deployment percentage"
        )
        monitoring_period_hours: PositiveInt = Field(
            default=24, ge=1, le=168, description="Monitoring period hours"
        )

    lookback_days: LookbackDaysConfig = Field(default_factory=LookbackDaysConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)


class OrchestratorFeaturesConfig(BaseModel):
    """Orchestrator features configuration."""

    class OrchestratorTechnicalIndicatorsConfig(BaseModel):
        """Orchestrator-specific technical indicators configuration."""

        sma_short_window: PositiveInt = Field(
            default=10, ge=2, le=100, description="SMA short window"
        )
        sma_long_window: PositiveInt = Field(
            default=20, ge=5, le=200, description="SMA long window"
        )
        volume_sma_window: PositiveInt = Field(
            default=10, ge=2, le=100, description="Volume SMA window"
        )

        @model_validator(mode="after")
        def validate_sma_windows(self):
            """Ensure long window is greater than short window."""
            if self.sma_long_window <= self.sma_short_window:
                raise ValueError("SMA long window must be greater than short window")
            return self

    class BatchProcessingConfig(BaseModel):
        """Batch processing configuration."""

        default_batch_size: PositiveInt = Field(
            default=10, ge=1, le=100, description="Default batch size"
        )
        processing_delay_seconds: PositiveInt = Field(
            default=1, ge=0, le=60, description="Processing delay"
        )

    technical_indicators: OrchestratorTechnicalIndicatorsConfig = Field(
        default_factory=OrchestratorTechnicalIndicatorsConfig
    )
    batch_processing: BatchProcessingConfig = Field(default_factory=BatchProcessingConfig)


class CatalystScoringConfig(BaseModel):
    """Catalyst scoring configuration."""

    class ThresholdsConfig(BaseModel):
        """Scoring thresholds configuration."""

        min_final_score: NonNegativeFloat = Field(
            default=3.0, ge=0.0, le=10.0, description="Minimum final score"
        )
        strong_signal_threshold: NonNegativeFloat = Field(
            default=5.0, ge=0.0, le=10.0, description="Strong signal threshold"
        )
        emerging_catalyst_threshold: NonNegativeFloat = Field(
            default=1.5, ge=0.0, le=10.0, description="Emerging catalyst threshold"
        )
        min_layer1_signals: PositiveInt = Field(
            default=2, ge=1, le=10, description="Minimum Layer 1 signals"
        )
        signal_boost_threshold_high: PositiveInt = Field(
            default=5, ge=1, le=20, description="High signal boost threshold"
        )
        signal_boost_threshold_medium: PositiveInt = Field(
            default=3, ge=1, le=10, description="Medium signal boost threshold"
        )

    class ScalingConfig(BaseModel):
        """Scaling configuration."""

        alert_score_multiplier: PositiveFloat = Field(
            default=5.0, ge=1.0, le=20.0, description="Alert score multiplier"
        )
        score_normalization_divisor: PositiveFloat = Field(
            default=10.0, ge=1.0, le=100.0, description="Score normalization divisor"
        )
        emerging_score_base: NonNegativeFloat = Field(
            default=0.4, ge=0.0, le=1.0, description="Emerging score base"
        )
        emerging_score_divisor: PositiveFloat = Field(
            default=20.0, ge=1.0, le=100.0, description="Emerging score divisor"
        )

    class WeightingConfig(BaseModel):
        """Weighting configuration."""

        layer2_confidence_weight: confloat(ge=0.0, le=1.0) = Field(
            default=0.7, description="Layer 2 confidence weight"
        )
        layer1_confidence_weight: confloat(ge=0.0, le=1.0) = Field(
            default=0.3, description="Layer 1 confidence weight"
        )
        high_signal_boost_factor: confloat(ge=1.0, le=2.0) = Field(
            default=1.2, description="High signal boost factor"
        )
        medium_signal_boost_factor: confloat(ge=1.0, le=2.0) = Field(
            default=1.1, description="Medium signal boost factor"
        )

        @model_validator(mode="after")
        def validate_weights(self):
            """Ensure weights sum to 1.0."""
            total = self.layer2_confidence_weight + self.layer1_confidence_weight
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Layer confidence weights must sum to 1.0, got {total}")
            return self

    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)


class DataPipelineConfig(BaseModel):
    """Data pipeline configuration."""

    class ResilienceConfig(BaseModel):
        """Resilience configuration."""

        max_retries: PositiveInt = Field(default=3, ge=0, le=10, description="Maximum retries")
        initial_delay_seconds: PositiveFloat = Field(
            default=1.0, ge=0.1, le=60.0, description="Initial delay seconds"
        )
        backoff_factor: PositiveFloat = Field(
            default=2.0, ge=1.0, le=5.0, description="Backoff factor"
        )
        circuit_breaker_threshold: PositiveInt = Field(
            default=5, ge=1, le=20, description="Circuit breaker threshold"
        )
        recovery_timeout_seconds: PositiveInt = Field(
            default=60, ge=10, le=3600, description="Recovery timeout seconds"
        )

    class ValidationConfig(BaseModel):
        """Data validation configuration."""

        class QualityThresholds(BaseModel):
            """Data quality thresholds."""

            min_quality_score: confloat(ge=0.0, le=100.0) = Field(
                default=80.0, description="Minimum quality score"
            )
            max_nan_ratio: confloat(ge=0.0, le=1.0) = Field(
                default=0.1, description="Maximum NaN ratio"
            )
            max_inf_count: NonNegativeInt = Field(default=0, description="Maximum infinite values")
            max_null_percentage: confloat(ge=0.0, le=1.0) = Field(
                default=0.02, description="Maximum null percentage"
            )
            min_data_points: PositiveInt = Field(
                default=5, ge=1, le=1000, description="Minimum required data points"
            )

        class MarketDataValidation(BaseModel):
            """Market data validation settings."""

            max_price_deviation: confloat(ge=0.0, le=10.0) = Field(
                default=0.5, description="Max price deviation (ratio)"
            )
            allow_zero_volume: bool = Field(default=False, description="Allow zero volume bars")
            allow_missing_vwap: bool = Field(default=True, description="Allow missing VWAP")
            allow_weekend_trading: bool = Field(
                default=False, description="Allow weekend trading data"
            )
            allow_future_timestamps: bool = Field(
                default=False, description="Allow future timestamps"
            )
            required_ohlcv_fields: list[str] = Field(
                default_factory=lambda: ["open", "high", "low", "close", "volume"],
                description="Required OHLCV fields",
            )

        class FeatureValidation(BaseModel):
            """Feature validation settings."""

            min_feature_coverage: confloat(ge=0.0, le=1.0) = Field(
                default=0.8, description="Minimum feature coverage"
            )
            max_correlation: confloat(ge=0.0, le=1.0) = Field(
                default=0.95, description="Max correlation between features"
            )
            max_correlation_threshold: confloat(ge=0.0, le=1.0) = Field(
                default=0.95, description="Max correlation between features (deprecated)"
            )
            detect_feature_drift: bool = Field(
                default=True, description="Enable feature drift detection"
            )
            drift_threshold: confloat(ge=0.0, le=1.0) = Field(
                default=0.1, description="Feature drift threshold"
            )
            high_nan_threshold: confloat(ge=0.0, le=1.0) = Field(
                default=0.5, description="High NaN threshold for features"
            )
            max_nan_ratio: confloat(ge=0.0, le=1.0) = Field(
                default=0.3, description="Maximum NaN ratio allowed"
            )
            max_inf_count: int = Field(
                default=0, ge=0, description="Maximum infinite values allowed"
            )
            distribution_checks: dict[str, Any] = Field(
                default_factory=dict, description="Distribution check constraints"
            )

        class CleaningSettings(BaseModel):
            """Data cleaning settings."""

            aggressive_cleaning: bool = Field(
                default=False, description="Enable aggressive data cleaning"
            )
            interpolation_method: str = Field(
                default="linear", description="Interpolation method for missing values"
            )
            outlier_method: str = Field(default="iqr", description="Outlier detection method")
            outlier_threshold: PositiveFloat = Field(
                default=3.0, ge=1.0, le=10.0, description="Outlier threshold (std devs)"
            )
            field_mappings: dict[str, str] = Field(
                default_factory=dict, description="Field name mappings for data sources"
            )

        class ProfileSettings(BaseModel):
            """Validation profile settings."""

            default_profile: str = Field(
                default="standard", description="Default validation profile"
            )
            fail_fast: bool = Field(default=True, description="Stop on first critical error")
            collect_all_errors: bool = Field(
                default=False, description="Collect all errors before failing"
            )
            cache_validation_results: bool = Field(
                default=True, description="Cache validation results"
            )
            cache_ttl_seconds: PositiveInt = Field(
                default=3600, ge=60, le=86400, description="Cache TTL in seconds"
            )

        quality_thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
        market_data: MarketDataValidation = Field(default_factory=MarketDataValidation)
        features: FeatureValidation = Field(default_factory=FeatureValidation)
        cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
        profile: ProfileSettings = Field(default_factory=ProfileSettings)

    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class OrchestratorConfig(BaseModel):
    """Main orchestrator configuration with validation."""

    intervals: IntervalsConfig = Field(
        default_factory=IntervalsConfig, description="Timing intervals"
    )
    lookback_periods: LookbackPeriodsConfig = Field(
        default_factory=LookbackPeriodsConfig, description="Lookback periods"
    )
    market_hours: MarketHoursConfig = Field(
        default_factory=MarketHoursConfig, description="Market hours"
    )
    training: OrchestratorTrainingConfig = Field(
        default_factory=OrchestratorTrainingConfig, description="Training configuration"
    )
    features: OrchestratorFeaturesConfig = Field(
        default_factory=OrchestratorFeaturesConfig, description="Features configuration"
    )
    catalyst_scoring: CatalystScoringConfig = Field(
        default_factory=CatalystScoringConfig, description="Catalyst scoring"
    )
    data_pipeline: DataPipelineConfig = Field(
        default_factory=DataPipelineConfig, description="Data pipeline"
    )

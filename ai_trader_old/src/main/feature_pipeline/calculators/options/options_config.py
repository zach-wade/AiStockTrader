"""
Options Configuration

Centralized configuration for all options analytics calculators.
Manages options-specific parameters, expiration windows, volume thresholds,
Black-Scholes parameters, and validation settings.
"""

# Standard library imports
from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptionsConfig:
    """Comprehensive configuration for options analytics calculations."""

    # Data Quality Settings
    min_volume: int = 10
    min_open_interest: int = 100
    min_option_price: float = 0.05
    max_option_price: float = 1000.0

    # Expiration Window Analysis
    expiry_windows: list[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])
    primary_expiry: int = 30  # Primary expiration for analysis
    max_days_to_expiry: int = 365
    min_days_to_expiry: int = 1

    # Black-Scholes Model Parameters
    risk_free_rate: float = 0.05  # 5% default risk-free rate
    dividend_yield: float = 0.0  # Default dividend yield
    default_volatility: float = 0.20  # 20% default volatility
    max_volatility: float = 5.0  # Maximum allowed volatility (500%)
    min_volatility: float = 0.01  # Minimum allowed volatility (1%)

    # Volume Flow Detection
    unusual_volume_threshold: float = 2.0  # Multiple of average volume
    block_trade_size: int = 100  # Minimum size for block trades
    large_trade_size: int = 500  # Size for large trades
    unusual_volume_window: int = 20  # Days for average volume calculation

    # Sentiment and Extremes
    put_call_extreme_high: float = 1.5  # High P/C ratio threshold
    put_call_extreme_low: float = 0.67  # Low P/C ratio threshold (1/1.5)
    skew_extreme_threshold: float = 2.0  # Extreme skew threshold
    sentiment_window: int = 5  # Days for sentiment moving average

    # Moneyness Categories
    atm_threshold: float = 0.02  # ±2% for ATM classification
    itm_threshold: float = 0.05  # 5% ITM threshold
    otm_threshold: float = 0.05  # 5% OTM threshold

    # Implied Volatility Analysis
    iv_percentile_window: int = 252  # Days for IV percentile calculation
    iv_rank_window: int = 252  # Days for IV rank calculation
    iv_smoothing_window: int = 5  # Days for IV smoothing
    iv_term_structure_min_expiry: int = 7  # Minimum days for term structure

    # Greeks Calculation
    greeks_spot_shift: float = 0.01  # 1% shift for finite difference Greeks
    greeks_vol_shift: float = 0.01  # 1% vol shift for vega calculation
    greeks_time_shift: float = 1.0  # 1 day shift for theta calculation
    pin_risk_range: float = 0.02  # ±2% range for pin risk analysis

    # Term Structure Analysis
    front_month_cutoff: int = 45  # Days to classify as front month
    back_month_cutoff: int = 90  # Days to classify as back month
    term_structure_points: int = 5  # Number of points in term structure

    # Premium Flow Analysis
    premium_threshold: float = 10000.0  # Minimum premium for flow analysis
    net_flow_window: int = 20  # Days for net flow calculation
    flow_momentum_window: int = 5  # Days for flow momentum

    # Options Chain Filtering
    max_strike_range: float = 0.5  # ±50% from spot for strike filtering
    min_days_to_expiry_filter: int = 3  # Minimum days to expiry to include
    volume_oi_ratio_threshold: float = 0.1  # Min volume/OI ratio for validity

    # Risk Management
    max_delta_exposure: float = 10000.0  # Maximum delta exposure to track
    max_gamma_exposure: float = 1000.0  # Maximum gamma exposure to track
    max_vega_exposure: float = 5000.0  # Maximum vega exposure to track

    # Data Validation
    validate_prices: bool = True  # Enable price validation
    validate_volumes: bool = True  # Enable volume validation
    validate_greeks: bool = True  # Enable Greeks validation
    strict_validation: bool = False  # Strict validation mode

    # Performance Settings
    cache_greeks: bool = True  # Cache Greeks calculations
    parallel_processing: bool = False  # Enable parallel processing
    max_workers: int = 4  # Maximum worker threads

    # Feature Generation Settings
    generate_all_features: bool = True  # Generate all available features
    feature_prefix: str = "opt_"  # Prefix for feature names

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_derived_parameters()

    def _validate_config(self):
        """Validate configuration parameters."""
        errors = []
        warnings = []

        # Validate volume thresholds
        if self.min_volume < 0:
            errors.append("min_volume must be non-negative")
        if self.min_open_interest < 0:
            errors.append("min_open_interest must be non-negative")

        # Validate price ranges
        if self.min_option_price <= 0:
            errors.append("min_option_price must be positive")
        if self.max_option_price <= self.min_option_price:
            errors.append("max_option_price must be greater than min_option_price")

        # Validate expiration windows
        if not self.expiry_windows:
            errors.append("expiry_windows cannot be empty")
        if any(w <= 0 for w in self.expiry_windows):
            errors.append("All expiry_windows must be positive")
        if self.primary_expiry not in self.expiry_windows:
            warnings.append(f"primary_expiry {self.primary_expiry} not in expiry_windows")

        # Validate Black-Scholes parameters
        if not (0 <= self.risk_free_rate <= 1):
            warnings.append("risk_free_rate should be between 0 and 1")
        if not (0 <= self.dividend_yield <= 1):
            warnings.append("dividend_yield should be between 0 and 1")
        if not (self.min_volatility <= self.default_volatility <= self.max_volatility):
            errors.append("default_volatility must be between min_volatility and max_volatility")

        # Validate sentiment thresholds
        if self.put_call_extreme_low >= self.put_call_extreme_high:
            errors.append("put_call_extreme_low must be less than put_call_extreme_high")

        # Validate moneyness thresholds
        if not (0 < self.atm_threshold < 0.1):
            warnings.append("atm_threshold should be between 0 and 0.1 (0-10%)")

        # Log validation results
        if errors:
            error_msg = "OptionsConfig validation errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        if warnings:
            warning_msg = "OptionsConfig validation warnings: " + "; ".join(warnings)
            logger.warning(warning_msg)

    def _setup_derived_parameters(self):
        """Setup derived parameters based on configuration."""
        # Sort expiry windows for consistent processing
        self.expiry_windows = sorted(self.expiry_windows)

        # Calculate derived thresholds
        self.put_call_neutral = 1.0  # Neutral P/C ratio
        self.put_call_range = self.put_call_extreme_high - self.put_call_extreme_low

        # Setup moneyness ranges
        self.moneyness_ranges = {
            "deep_itm": 0.15,  # >15% ITM
            "itm": self.itm_threshold,
            "atm": self.atm_threshold,
            "otm": self.otm_threshold,
            "deep_otm": 0.20,  # >20% OTM
        }

    def get_expiry_config(self, expiry_days: int) -> dict[str, Any]:
        """Get configuration for specific expiry."""
        return {
            "days_to_expiry": expiry_days,
            "is_front_month": expiry_days <= self.front_month_cutoff,
            "is_back_month": expiry_days >= self.back_month_cutoff,
            "weight": self._calculate_expiry_weight(expiry_days),
        }

    def get_moneyness_config(self) -> dict[str, dict[str, float]]:
        """Get moneyness classification configuration."""
        return {
            "thresholds": self.moneyness_ranges,
            "atm_range": (-self.atm_threshold, self.atm_threshold),
            "itm_call_threshold": -self.itm_threshold,
            "otm_call_threshold": self.otm_threshold,
            "itm_put_threshold": self.itm_threshold,
            "otm_put_threshold": -self.otm_threshold,
        }

    def get_volume_config(self) -> dict[str, Any]:
        """Get volume analysis configuration."""
        return {
            "min_volume": self.min_volume,
            "min_open_interest": self.min_open_interest,
            "unusual_threshold": self.unusual_volume_threshold,
            "block_size": self.block_trade_size,
            "large_size": self.large_trade_size,
            "window": self.unusual_volume_window,
        }

    def get_sentiment_config(self) -> dict[str, Any]:
        """Get sentiment analysis configuration."""
        return {
            "extreme_high": self.put_call_extreme_high,
            "extreme_low": self.put_call_extreme_low,
            "neutral": self.put_call_neutral,
            "window": self.sentiment_window,
            "skew_threshold": self.skew_extreme_threshold,
        }

    def get_blackscholes_config(self) -> dict[str, Any]:
        """Get Black-Scholes model configuration."""
        return {
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "default_volatility": self.default_volatility,
            "vol_range": (self.min_volatility, self.max_volatility),
            "greeks_shifts": {
                "spot": self.greeks_spot_shift,
                "vol": self.greeks_vol_shift,
                "time": self.greeks_time_shift,
            },
        }

    def _calculate_expiry_weight(self, expiry_days: int) -> float:
        """Calculate weight for expiry based on time to expiration."""
        if expiry_days <= 0:
            return 0.0

        # Higher weight for nearer expirations
        if expiry_days <= 30:
            return 1.0
        elif expiry_days <= 60:
            return 0.8
        elif expiry_days <= 90:
            return 0.6
        else:
            return 0.4

    def validate_configuration(self) -> dict[str, Any]:
        """Comprehensive configuration validation."""
        validation_result = {"valid": True, "errors": [], "warnings": [], "config_summary": {}}

        try:
            # Re-run validation
            self._validate_config()

            # Generate configuration summary
            validation_result["config_summary"] = {
                "expiry_windows": len(self.expiry_windows),
                "primary_expiry": self.primary_expiry,
                "volume_thresholds": {
                    "min_volume": self.min_volume,
                    "min_oi": self.min_open_interest,
                    "unusual_threshold": self.unusual_volume_threshold,
                },
                "blackscholes_params": {
                    "risk_free_rate": self.risk_free_rate,
                    "dividend_yield": self.dividend_yield,
                    "default_vol": self.default_volatility,
                },
                "sentiment_thresholds": {
                    "pc_high": self.put_call_extreme_high,
                    "pc_low": self.put_call_extreme_low,
                    "skew_extreme": self.skew_extreme_threshold,
                },
            }

        except ValueError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))

        return validation_result

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data_quality": {
                "min_volume": self.min_volume,
                "min_open_interest": self.min_open_interest,
                "min_option_price": self.min_option_price,
                "max_option_price": self.max_option_price,
            },
            "expiry_windows": self.expiry_windows,
            "blackscholes": self.get_blackscholes_config(),
            "volume_analysis": self.get_volume_config(),
            "sentiment": self.get_sentiment_config(),
            "moneyness": self.get_moneyness_config(),
            "validation": {
                "validate_prices": self.validate_prices,
                "validate_volumes": self.validate_volumes,
                "validate_greeks": self.validate_greeks,
                "strict_validation": self.strict_validation,
            },
        }

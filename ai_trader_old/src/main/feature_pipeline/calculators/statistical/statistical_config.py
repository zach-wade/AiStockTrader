"""
Statistical Calculator Configuration

Configuration classes for statistical feature calculators,
providing centralized parameter management and validation.
"""

# Standard library imports
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DistributionType(Enum):
    """Supported distribution types for fitting."""

    NORMAL = "normal"
    T_DISTRIBUTION = "t"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"


class OutlierMethod(Enum):
    """Methods for outlier detection and removal."""

    ZSCORE = "zscore"
    IQR = "iqr"
    MAD = "mad"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"


class EntropyMethod(Enum):
    """Entropy calculation methods."""

    SHANNON = "shannon"
    RENYI = "renyi"
    TSALLIS = "tsallis"
    APPROXIMATE = "approximate"
    SAMPLE = "sample"


@dataclass
class StatisticalConfig:
    """
    Configuration for statistical feature calculators.

    Centralizes all parameters for statistical analysis including
    window sizes, distribution types, entropy parameters, etc.
    """

    # Time windows for rolling calculations
    lookback_periods: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Minimum periods for rolling calculations
    min_periods_ratio: float = 0.5  # As fraction of window size

    # Return calculation parameters
    return_type: str = "simple"  # "simple" or "log"

    # Distribution fitting
    distribution_types: list[str] = field(default_factory=lambda: ["normal", "t", "lognormal"])
    fit_distributions: bool = True
    min_samples_for_fit: int = 30

    # Outlier handling
    remove_outliers: bool = True
    outlier_method: str = "zscore"
    outlier_threshold: float = 3.0

    # Entropy parameters
    calculate_entropy: bool = True
    entropy_bins: int = 10
    entropy_methods: list[str] = field(default_factory=lambda: ["shannon", "normalized"])

    # Autocorrelation parameters
    autocorr_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    partial_autocorr: bool = True

    # Moment calculations
    calculate_moments: bool = True
    max_moment_order: int = 4
    standardize_moments: bool = True

    # Tail analysis
    analyze_tails: bool = True
    tail_percentiles: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])

    # Complexity measures
    calculate_complexity: bool = True
    hurst_min_window: int = 50
    approx_entropy_m: int = 2
    approx_entropy_r: float = 0.2

    # Feature engineering
    create_interaction_features: bool = False
    create_polynomial_features: bool = False
    polynomial_degree: int = 2

    # Performance optimization
    use_parallel: bool = True
    chunk_size: int = 10000
    cache_results: bool = True

    # Validation
    validate_results: bool = True
    max_feature_std: float = 100.0  # Maximum allowed standard deviation

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate lookback periods
        if not self.lookback_periods:
            raise ValueError("lookback_periods cannot be empty")

        if any(p <= 0 for p in self.lookback_periods):
            raise ValueError("All lookback periods must be positive")

        # Validate return type
        if self.return_type not in ["simple", "log"]:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        # Validate outlier method
        valid_outlier_methods = [m.value for m in OutlierMethod]
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"Invalid outlier_method: {self.outlier_method}")

        # Validate distribution types
        valid_distributions = [d.value for d in DistributionType]
        for dist in self.distribution_types:
            if dist not in valid_distributions:
                raise ValueError(f"Invalid distribution type: {dist}")

        # Validate entropy parameters
        if self.entropy_bins <= 0:
            raise ValueError("entropy_bins must be positive")

        # Validate tail percentiles
        for p in self.tail_percentiles:
            if not 0 < p < 0.5:
                raise ValueError(f"Tail percentile {p} must be between 0 and 0.5")

        # Validate complexity parameters
        if self.hurst_min_window < 20:
            raise ValueError("hurst_min_window must be at least 20")

        if self.approx_entropy_m <= 0:
            raise ValueError("approx_entropy_m must be positive")

        if not 0 < self.approx_entropy_r < 1:
            raise ValueError("approx_entropy_r must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "lookback_periods": self.lookback_periods,
            "min_periods_ratio": self.min_periods_ratio,
            "return_type": self.return_type,
            "distribution_types": self.distribution_types,
            "fit_distributions": self.fit_distributions,
            "min_samples_for_fit": self.min_samples_for_fit,
            "remove_outliers": self.remove_outliers,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
            "calculate_entropy": self.calculate_entropy,
            "entropy_bins": self.entropy_bins,
            "entropy_methods": self.entropy_methods,
            "autocorr_lags": self.autocorr_lags,
            "partial_autocorr": self.partial_autocorr,
            "calculate_moments": self.calculate_moments,
            "max_moment_order": self.max_moment_order,
            "standardize_moments": self.standardize_moments,
            "analyze_tails": self.analyze_tails,
            "tail_percentiles": self.tail_percentiles,
            "calculate_complexity": self.calculate_complexity,
            "hurst_min_window": self.hurst_min_window,
            "approx_entropy_m": self.approx_entropy_m,
            "approx_entropy_r": self.approx_entropy_r,
            "create_interaction_features": self.create_interaction_features,
            "create_polynomial_features": self.create_polynomial_features,
            "polynomial_degree": self.polynomial_degree,
            "use_parallel": self.use_parallel,
            "chunk_size": self.chunk_size,
            "cache_results": self.cache_results,
            "validate_results": self.validate_results,
            "max_feature_std": self.max_feature_std,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "StatisticalConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def get_min_periods(self, window: int) -> int:
        """
        Get minimum periods for a given window size.

        Args:
            window: Window size

        Returns:
            Minimum periods required
        """
        return max(1, int(window * self.min_periods_ratio))

    def get_entropy_config(self) -> dict[str, Any]:
        """Get entropy-specific configuration."""
        return {
            "bins": self.entropy_bins,
            "methods": self.entropy_methods,
            "enabled": self.calculate_entropy,
        }

    def get_distribution_config(self) -> dict[str, Any]:
        """Get distribution fitting configuration."""
        return {
            "types": self.distribution_types,
            "enabled": self.fit_distributions,
            "min_samples": self.min_samples_for_fit,
        }

    def get_outlier_config(self) -> dict[str, Any]:
        """Get outlier handling configuration."""
        return {
            "enabled": self.remove_outliers,
            "method": self.outlier_method,
            "threshold": self.outlier_threshold,
        }


@dataclass
class EntropyConfig:
    """Specific configuration for entropy calculations."""

    # Binning parameters
    bins: int = 10
    bin_method: str = "equal_width"  # "equal_width", "equal_frequency", "entropy_optimal"

    # Entropy types
    shannon: bool = True
    renyi: bool = False
    renyi_alpha: float = 2.0
    tsallis: bool = False
    tsallis_q: float = 2.0

    # Approximate and sample entropy
    approximate: bool = True
    approx_m: int = 2
    approx_r: float = 0.2
    sample: bool = False
    sample_m: int = 2
    sample_r: float = 0.2

    # Permutation entropy
    permutation: bool = False
    permutation_order: int = 3

    # Multiscale entropy
    multiscale: bool = False
    multiscale_factors: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass
class MomentConfig:
    """Configuration for moment calculations."""

    # Basic moments
    calculate_raw: bool = True
    calculate_central: bool = True
    calculate_standardized: bool = True

    # Moment orders
    max_order: int = 4

    # L-moments
    calculate_l_moments: bool = False
    l_moment_orders: list[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Partial moments
    calculate_partial: bool = False
    partial_threshold: float = 0.0

    # Co-moments (for multi-asset)
    calculate_co_moments: bool = False
    co_moment_order: int = 2


def create_default_config() -> StatisticalConfig:
    """Create default statistical configuration."""
    return StatisticalConfig()


def create_fast_config() -> StatisticalConfig:
    """Create configuration optimized for speed."""
    return StatisticalConfig(
        lookback_periods=[10, 20, 50],
        distribution_types=["normal"],
        calculate_entropy=False,
        calculate_complexity=False,
        analyze_tails=False,
        use_parallel=True,
        cache_results=True,
    )


def create_comprehensive_config() -> StatisticalConfig:
    """Create configuration for comprehensive analysis."""
    return StatisticalConfig(
        lookback_periods=[5, 10, 20, 50, 100, 200, 500],
        distribution_types=["normal", "t", "lognormal", "gamma", "beta"],
        calculate_entropy=True,
        entropy_methods=["shannon", "normalized", "approximate"],
        calculate_complexity=True,
        analyze_tails=True,
        tail_percentiles=[0.001, 0.01, 0.05, 0.10, 0.25],
        calculate_moments=True,
        max_moment_order=6,
        create_interaction_features=True,
        create_polynomial_features=True,
    )

"""
Correlation Analysis Configuration

Centralized configuration for all correlation analysis calculators including
correlation windows, benchmark symbols, analysis parameters, and thresholds.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis calculators."""

    # Time windows for correlation analysis (in days)
    correlation_windows: list[int] = None

    # Minimum periods required for correlation calculations
    min_periods: int = 15

    # Benchmark symbols for correlation analysis
    benchmark_symbols: list[str] = None

    # Decay parameters for exponential weighting
    decay_halflife: int = 20

    # PCA configuration
    pca_components: int = 5
    pca_variance_threshold: float = 0.95

    # Beta analysis parameters
    beta_lookback_periods: list[int] = None
    beta_regime_threshold: float = 0.02  # 2% volatility threshold

    # Stability analysis parameters
    stability_window: int = 60  # Days
    breakdown_threshold: float = 0.3  # Correlation change threshold
    stability_percentile: float = 0.1  # 10th percentile for stability scoring

    # Lead-lag analysis parameters
    lead_lag_periods: list[int] = None  # Days (negative for leads, positive for lags)
    max_lead_lag: int = 10  # Maximum lead/lag to test

    # Regime analysis parameters
    volatility_lookback: int = 20  # Days for volatility calculation
    volatility_percentiles: list[float] = None  # [low_threshold, high_threshold]
    trend_lookback: int = 20  # Days for trend analysis
    trend_threshold: float = 0.0  # Threshold for up/down trend classification

    # Tail correlation parameters
    tail_percentiles: list[float] = None  # Percentiles for tail analysis
    extreme_threshold: float = 0.05  # 5% threshold for extreme events

    # Network analysis parameters
    network_threshold: float = 0.3  # Minimum correlation for network connections
    centrality_method: str = "degree"  # 'degree', 'betweenness', 'eigenvector'
    stress_correlation_threshold: float = 0.7  # Threshold for stress period detection

    # Cross-sectional analysis parameters
    sector_symbols: dict[str, list[str]] = None
    size_symbols: dict[str, list[str]] = None  # Small, mid, large cap
    style_symbols: dict[str, list[str]] = None  # Growth, value, momentum

    # Statistical parameters
    correlation_significance_level: float = 0.05
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection

    # Performance parameters
    parallel_processing: bool = True
    chunk_size: int = 1000  # For batch processing

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.correlation_windows is None:
            self.correlation_windows = [20, 60, 120]  # 1 month, 3 months, 6 months

        if self.benchmark_symbols is None:
            self.benchmark_symbols = ["SPY", "QQQ", "IWM", "VTI", "XLF", "XLK", "XLE"]

        if self.beta_lookback_periods is None:
            self.beta_lookback_periods = [20, 60, 120, 252]  # Various time horizons

        if self.lead_lag_periods is None:
            self.lead_lag_periods = [-5, -3, -1, 1, 3, 5]  # Days

        if self.volatility_percentiles is None:
            self.volatility_percentiles = [0.33, 0.67]  # Low, medium, high vol regimes

        if self.tail_percentiles is None:
            self.tail_percentiles = [0.05, 0.95]  # 5th and 95th percentiles

        if self.sector_symbols is None:
            self.sector_symbols = {
                "technology": ["XLK", "QQQ", "AAPL", "MSFT", "GOOGL"],
                "financial": ["XLF", "JPM", "BAC", "WFC", "C"],
                "energy": ["XLE", "XOM", "CVX", "COP", "EOG"],
                "healthcare": ["XLV", "JNJ", "PFE", "UNH", "ABBV"],
                "consumer": ["XLY", "AMZN", "TSLA", "HD", "MCD"],
            }

        if self.size_symbols is None:
            self.size_symbols = {
                "large_cap": ["SPY", "VTI", "VOO"],
                "mid_cap": ["MDY", "IJH", "VO"],
                "small_cap": ["IWM", "VB", "IJR"],
            }

        if self.style_symbols is None:
            self.style_symbols = {
                "growth": ["IVW", "VUG", "MTUM"],
                "value": ["IVE", "VTV", "VMOT"],
                "momentum": ["MTUM", "PDP", "QUAL"],
            }

    def get_all_symbols(self) -> list[str]:
        """Get all unique symbols from all categories."""
        all_symbols = set(self.benchmark_symbols)

        # Add sector symbols
        for symbols in self.sector_symbols.values():
            all_symbols.update(symbols)

        # Add size symbols
        for symbols in self.size_symbols.values():
            all_symbols.update(symbols)

        # Add style symbols
        for symbols in self.style_symbols.values():
            all_symbols.update(symbols)

        return sorted(list(all_symbols))

    def validate_configuration(self) -> dict[str, Any]:
        """Validate configuration parameters."""
        validation_results = {"valid": True, "warnings": [], "errors": []}

        # Check correlation windows
        if not self.correlation_windows or min(self.correlation_windows) < 5:
            validation_results["errors"].append("Correlation windows must be at least 5 days")
            validation_results["valid"] = False

        # Check minimum periods
        if self.min_periods < 5:
            validation_results["warnings"].append(
                "Minimum periods less than 5 may produce unreliable correlations"
            )

        # Check PCA components
        if self.pca_components < 1 or self.pca_components > 20:
            validation_results["warnings"].append("PCA components should be between 1 and 20")

        # Check thresholds are in valid ranges
        if not (0 <= self.breakdown_threshold <= 1):
            validation_results["errors"].append("Breakdown threshold must be between 0 and 1")
            validation_results["valid"] = False

        if not (0 <= self.extreme_threshold <= 0.5):
            validation_results["errors"].append("Extreme threshold must be between 0 and 0.5")
            validation_results["valid"] = False

        # Check benchmark symbols
        if not self.benchmark_symbols:
            validation_results["errors"].append("At least one benchmark symbol must be specified")
            validation_results["valid"] = False

        return validation_results

    def get_window_config(self, calculator_type: str) -> dict[str, Any]:
        """Get window configuration for specific calculator type."""
        base_config = {
            "correlation_windows": self.correlation_windows,
            "min_periods": self.min_periods,
        }

        if calculator_type == "rolling":
            return {
                **base_config,
                "benchmark_symbols": self.benchmark_symbols,
                "decay_halflife": self.decay_halflife,
            }

        elif calculator_type == "beta":
            return {
                **base_config,
                "beta_lookback_periods": self.beta_lookback_periods,
                "regime_threshold": self.beta_regime_threshold,
            }

        elif calculator_type == "stability":
            return {
                **base_config,
                "stability_window": self.stability_window,
                "breakdown_threshold": self.breakdown_threshold,
                "stability_percentile": self.stability_percentile,
            }

        elif calculator_type == "lead_lag":
            return {
                **base_config,
                "lead_lag_periods": self.lead_lag_periods,
                "max_lead_lag": self.max_lead_lag,
            }

        elif calculator_type == "pca":
            return {
                **base_config,
                "pca_components": self.pca_components,
                "variance_threshold": self.pca_variance_threshold,
            }

        elif calculator_type == "regime":
            return {
                **base_config,
                "volatility_lookback": self.volatility_lookback,
                "volatility_percentiles": self.volatility_percentiles,
                "trend_lookback": self.trend_lookback,
                "trend_threshold": self.trend_threshold,
            }

        elif calculator_type == "network":
            return {
                **base_config,
                "network_threshold": self.network_threshold,
                "centrality_method": self.centrality_method,
                "stress_threshold": self.stress_correlation_threshold,
            }

        else:
            return base_config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "correlation_windows": self.correlation_windows,
            "min_periods": self.min_periods,
            "benchmark_symbols": self.benchmark_symbols,
            "decay_halflife": self.decay_halflife,
            "pca_components": self.pca_components,
            "pca_variance_threshold": self.pca_variance_threshold,
            "beta_lookback_periods": self.beta_lookback_periods,
            "beta_regime_threshold": self.beta_regime_threshold,
            "stability_window": self.stability_window,
            "breakdown_threshold": self.breakdown_threshold,
            "stability_percentile": self.stability_percentile,
            "lead_lag_periods": self.lead_lag_periods,
            "max_lead_lag": self.max_lead_lag,
            "volatility_lookback": self.volatility_lookback,
            "volatility_percentiles": self.volatility_percentiles,
            "trend_lookback": self.trend_lookback,
            "trend_threshold": self.trend_threshold,
            "tail_percentiles": self.tail_percentiles,
            "extreme_threshold": self.extreme_threshold,
            "network_threshold": self.network_threshold,
            "centrality_method": self.centrality_method,
            "stress_correlation_threshold": self.stress_correlation_threshold,
            "sector_symbols": self.sector_symbols,
            "size_symbols": self.size_symbols,
            "style_symbols": self.style_symbols,
            "correlation_significance_level": self.correlation_significance_level,
            "outlier_threshold": self.outlier_threshold,
            "parallel_processing": self.parallel_processing,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CorrelationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

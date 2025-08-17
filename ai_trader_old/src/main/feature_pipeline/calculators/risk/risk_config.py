"""
Risk Configuration

Comprehensive configuration system for risk metrics calculations including:
- VaR parameters (confidence levels, time horizons, methods)
- Volatility modeling settings (lookback periods, decay factors)
- Drawdown analysis parameters
- Performance metrics settings
- Stress testing configuration
- Tail risk analysis parameters
"""

# Standard library imports
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VaRMethod(Enum):
    """Value at Risk calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    EXTREME_VALUE = "extreme_value"
    CORNISH_FISHER = "cornish_fisher"


class VolatilityMethod(Enum):
    """Volatility calculation methods."""

    HISTORICAL = "historical"
    EWMA = "ewma"
    GARCH = "garch"
    REALIZED = "realized"


class StressTestMethod(Enum):
    """Stress testing methods."""

    HISTORICAL_SCENARIO = "historical_scenario"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC_SHOCK = "parametric_shock"
    TAIL_SCENARIO = "tail_scenario"


@dataclass
class RiskConfig:
    """Configuration for risk metrics calculations."""

    # === VaR Configuration ===
    var_confidence_levels: list[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    var_time_horizons: list[int] = field(default_factory=lambda: [1, 5, 21, 252])  # days
    var_lookback_window: int = 252  # 1 year of daily data
    var_min_observations: int = 60  # Minimum data points required

    # VaR method settings
    var_default_method: VaRMethod = VaRMethod.HISTORICAL
    var_methods_enabled: list[VaRMethod] = field(
        default_factory=lambda: [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]
    )

    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_random_seed: int | None = 42
    mc_confidence_interval: float = 0.95

    # Extreme Value Theory settings
    evt_threshold_percentile: float = 0.95
    evt_block_size: int = 22  # Monthly blocks for block maxima
    evt_min_exceedances: int = 50  # Minimum exceedances for POT

    # Parametric VaR settings
    parametric_distribution: str = "normal"  # normal, t, skewed_t
    parametric_dof: float | None = None  # degrees of freedom for t-distribution

    # === Volatility Configuration ===
    volatility_lookback_window: int = 252  # 1 year
    volatility_min_observations: int = 30
    volatility_annualization_factor: float = 252.0  # Trading days per year

    # EWMA settings
    ewma_decay_factor: float = 0.94  # RiskMetrics standard
    ewma_min_weight: float = 0.001  # Minimum weight for observations

    # Realized volatility settings
    realized_vol_frequency: str = "daily"  # daily, weekly, monthly
    realized_vol_estimator: str = "standard"  # standard, parkinson, garman_klass

    # === Drawdown Configuration ===
    drawdown_lookback_window: int = 252  # 1 year
    drawdown_min_observations: int = 30
    drawdown_recovery_threshold: float = 0.95  # 95% recovery for duration calc

    # === Performance Metrics Configuration ===
    performance_lookback_window: int = 252  # 1 year
    performance_min_observations: int = 60
    performance_risk_free_rate: float = 0.02  # 2% annual
    performance_benchmark_return: float = 0.08  # 8% annual market return

    # Performance ratio settings
    sharpe_annualization: float = 252.0
    sortino_annualization: float = 252.0
    sortino_target_return: float = 0.0  # Target return for downside deviation

    # === Stress Testing Configuration ===
    stress_confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    stress_scenarios: int = 1000
    stress_lookback_window: int = 252  # Historical scenario window

    # Stress shock settings
    stress_shock_sizes: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10])
    stress_shock_correlations: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.8, 1.0])

    # === Tail Risk Configuration ===
    tail_risk_threshold: float = 0.95  # 95th percentile
    tail_risk_min_observations: int = 100
    tail_risk_confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])

    # Hill estimator settings
    hill_estimator_fraction: float = 0.1  # Top 10% of observations
    hill_estimator_min_observations: int = 50

    # === General Configuration ===
    min_return_threshold: float = -0.20  # -20% maximum single-day loss
    max_return_threshold: float = 0.20  # 20% maximum single-day gain

    # Numerical settings
    numerical_precision: int = 6
    convergence_tolerance: float = 1e-8
    max_iterations: int = 1000

    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Data validation settings
    validate_input_data: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations

    # === Risk Limits Configuration ===
    # Portfolio level limits
    portfolio_var_limit_95: float = 0.05  # 5% of portfolio
    portfolio_var_limit_99: float = 0.10  # 10% of portfolio
    portfolio_max_drawdown_limit: float = 0.15  # 15% maximum drawdown

    # Position level limits
    position_var_limit: float = 0.02  # 2% of portfolio per position
    position_concentration_limit: float = 0.20  # 20% max in single position

    # Sector/factor limits
    sector_concentration_limit: float = 0.30  # 30% max in single sector
    factor_exposure_limit: float = 0.25  # 25% max factor exposure

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate confidence levels
        for level in self.var_confidence_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"Invalid confidence level: {level}")

        # Validate time horizons
        for horizon in self.var_time_horizons:
            if horizon <= 0:
                raise ValueError(f"Invalid time horizon: {horizon}")

        # Validate lookback windows
        if self.var_lookback_window <= 0:
            raise ValueError(f"Invalid VaR lookback window: {self.var_lookback_window}")

        if self.volatility_lookback_window <= 0:
            raise ValueError(
                f"Invalid volatility lookback window: {self.volatility_lookback_window}"
            )

        # Validate minimum observations
        if self.var_min_observations <= 0:
            raise ValueError(f"Invalid min observations: {self.var_min_observations}")

        # Validate Monte Carlo settings
        if self.mc_simulations <= 0:
            raise ValueError(f"Invalid MC simulations: {self.mc_simulations}")

        # Validate EWMA decay factor
        if not 0.0 < self.ewma_decay_factor < 1.0:
            raise ValueError(f"Invalid EWMA decay factor: {self.ewma_decay_factor}")

        # Validate risk limits
        if self.portfolio_var_limit_95 <= 0:
            raise ValueError(f"Invalid portfolio VaR limit: {self.portfolio_var_limit_95}")

    def get_var_config(self) -> dict[str, Any]:
        """Get VaR-specific configuration."""
        return {
            "confidence_levels": self.var_confidence_levels,
            "time_horizons": self.var_time_horizons,
            "lookback_window": self.var_lookback_window,
            "min_observations": self.var_min_observations,
            "default_method": self.var_default_method,
            "methods_enabled": self.var_methods_enabled,
            "mc_simulations": self.mc_simulations,
            "mc_random_seed": self.mc_random_seed,
            "evt_threshold_percentile": self.evt_threshold_percentile,
            "evt_block_size": self.evt_block_size,
            "parametric_distribution": self.parametric_distribution,
        }

    def get_volatility_config(self) -> dict[str, Any]:
        """Get volatility-specific configuration."""
        return {
            "lookback_window": self.volatility_lookback_window,
            "min_observations": self.volatility_min_observations,
            "annualization_factor": self.volatility_annualization_factor,
            "ewma_decay_factor": self.ewma_decay_factor,
            "ewma_min_weight": self.ewma_min_weight,
            "realized_vol_frequency": self.realized_vol_frequency,
            "realized_vol_estimator": self.realized_vol_estimator,
        }

    def get_drawdown_config(self) -> dict[str, Any]:
        """Get drawdown-specific configuration."""
        return {
            "lookback_window": self.drawdown_lookback_window,
            "min_observations": self.drawdown_min_observations,
            "recovery_threshold": self.drawdown_recovery_threshold,
        }

    def get_performance_config(self) -> dict[str, Any]:
        """Get performance metrics configuration."""
        return {
            "lookback_window": self.performance_lookback_window,
            "min_observations": self.performance_min_observations,
            "risk_free_rate": self.performance_risk_free_rate,
            "benchmark_return": self.performance_benchmark_return,
            "sharpe_annualization": self.sharpe_annualization,
            "sortino_annualization": self.sortino_annualization,
            "sortino_target_return": self.sortino_target_return,
        }

    def get_stress_test_config(self) -> dict[str, Any]:
        """Get stress testing configuration."""
        return {
            "confidence_levels": self.stress_confidence_levels,
            "scenarios": self.stress_scenarios,
            "lookback_window": self.stress_lookback_window,
            "shock_sizes": self.stress_shock_sizes,
            "shock_correlations": self.stress_shock_correlations,
        }

    def get_tail_risk_config(self) -> dict[str, Any]:
        """Get tail risk configuration."""
        return {
            "threshold": self.tail_risk_threshold,
            "min_observations": self.tail_risk_min_observations,
            "confidence_levels": self.tail_risk_confidence_levels,
            "hill_estimator_fraction": self.hill_estimator_fraction,
            "hill_estimator_min_observations": self.hill_estimator_min_observations,
        }

    def get_limits_config(self) -> dict[str, Any]:
        """Get risk limits configuration."""
        return {
            "portfolio_var_limit_95": self.portfolio_var_limit_95,
            "portfolio_var_limit_99": self.portfolio_var_limit_99,
            "portfolio_max_drawdown_limit": self.portfolio_max_drawdown_limit,
            "position_var_limit": self.position_var_limit,
            "position_concentration_limit": self.position_concentration_limit,
            "sector_concentration_limit": self.sector_concentration_limit,
            "factor_exposure_limit": self.factor_exposure_limit,
        }


def create_default_risk_config() -> RiskConfig:
    """Create default risk configuration."""
    return RiskConfig()


def create_fast_risk_config() -> RiskConfig:
    """Create fast risk configuration with minimal lookback."""
    return RiskConfig(
        var_lookback_window=50,
        volatility_lookback_window=20,
        correlation_lookback_window=50,
        drawdown_lookback_window=50,
        performance_lookback_window=50,
        stress_lookback_window=100,
        var_methods=[VaRMethod.HISTORICAL],
        volatility_method=VolatilityMethod.HISTORICAL,
        tail_risk_min_observations=50,
    )


def create_comprehensive_risk_config() -> RiskConfig:
    """Create comprehensive risk configuration with all features."""
    return RiskConfig(
        var_lookback_window=500,
        volatility_lookback_window=252,
        correlation_lookback_window=252,
        drawdown_lookback_window=252,
        performance_lookback_window=252,
        stress_lookback_window=500,
        var_methods=[VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.CORNISH_FISHER],
        volatility_method=VolatilityMethod.GARCH,
        tail_risk_min_observations=200,
        enable_factor_risk=True,
        enable_stress_testing=True,
    )


def create_conservative_risk_config() -> RiskConfig:
    """Create conservative risk configuration with stricter limits."""
    return RiskConfig(
        var_confidence_levels=[0.95, 0.99],
        var_lookback_window=252,
        volatility_lookback_window=126,
        stress_shock_sizes=[0.10, 0.20, 0.30],
        portfolio_var_limit_95=0.05,
        portfolio_var_limit_99=0.10,
        portfolio_max_drawdown_limit=0.15,
        position_concentration_limit=0.05,
        sector_concentration_limit=0.20,
    )


def create_aggressive_risk_config() -> RiskConfig:
    """Create aggressive risk configuration with looser limits."""
    return RiskConfig(
        var_confidence_levels=[0.90],
        var_lookback_window=50,
        volatility_lookback_window=20,
        stress_shock_sizes=[0.05, 0.10],
        portfolio_var_limit_95=0.15,
        portfolio_var_limit_99=0.25,
        portfolio_max_drawdown_limit=0.30,
        position_concentration_limit=0.20,
        sector_concentration_limit=0.40,
        enable_factor_risk=False,
        enable_stress_testing=False,
    )

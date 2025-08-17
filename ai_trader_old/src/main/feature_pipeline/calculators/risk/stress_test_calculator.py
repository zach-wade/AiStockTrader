"""
Stress Test Calculator

Implements comprehensive stress testing and scenario analysis for
portfolio risk assessment under extreme market conditions.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from main.utils.core import get_logger

from .base_risk import BaseRiskCalculator
from ..helpers import create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class StressTestCalculator(BaseRiskCalculator):
    """
    Calculates stress test scenarios and their potential impacts.

    Features include:
    - Historical stress scenarios
    - Parametric stress tests
    - Monte Carlo simulations
    - Factor-based stress testing
    - Reverse stress testing
    - Scenario aggregation
    - Stress test confidence intervals
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize stress test calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Stress test parameters
        self.historical_scenarios = config.get(
            "historical_scenarios",
            [
                "black_monday_1987",
                "asian_crisis_1997",
                "dot_com_crash_2000",
                "financial_crisis_2008",
                "covid_crash_2020",
            ],
        )

        self.scenario_definitions = {
            "black_monday_1987": {
                "market_shock": -0.22,
                "volatility_spike": 3.0,
                "correlation_increase": 0.3,
                "duration_days": 1,
            },
            "asian_crisis_1997": {
                "market_shock": -0.15,
                "volatility_spike": 2.0,
                "correlation_increase": 0.2,
                "duration_days": 30,
            },
            "dot_com_crash_2000": {
                "market_shock": -0.35,
                "volatility_spike": 2.5,
                "correlation_increase": 0.25,
                "duration_days": 180,
            },
            "financial_crisis_2008": {
                "market_shock": -0.50,
                "volatility_spike": 4.0,
                "correlation_increase": 0.4,
                "duration_days": 365,
            },
            "covid_crash_2020": {
                "market_shock": -0.30,
                "volatility_spike": 5.0,
                "correlation_increase": 0.35,
                "duration_days": 30,
            },
        }

        # Parametric stress parameters
        self.stress_factors = config.get(
            "stress_factors",
            {
                "market": [-0.30, -0.20, -0.10, 0.10, 0.20],
                "volatility": [1.5, 2.0, 3.0, 4.0, 5.0],
                "correlation": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
        )

        # Monte Carlo parameters
        self.n_simulations = config.get("n_simulations", 1000)
        self.simulation_horizon = config.get("simulation_horizon", 20)

        # Reverse stress test parameters
        self.loss_thresholds = config.get("loss_thresholds", [0.10, 0.20, 0.30])

        logger.info("Initialized StressTestCalculator")

    def get_required_columns(self) -> list[str]:
        """Get list of required input columns."""
        return ["close"]

    def get_feature_names(self) -> list[str]:
        """Get list of stress test feature names."""
        features = []

        # Historical scenario impacts
        for scenario in self.historical_scenarios:
            features.extend(
                [f"{scenario}_impact", f"{scenario}_var_impact", f"{scenario}_recovery_time"]
            )

        # Parametric stress test results
        features.extend(
            [
                "market_shock_sensitivity",
                "volatility_shock_impact",
                "correlation_shock_impact",
                "combined_stress_impact",
            ]
        )

        # Monte Carlo stress tests
        features.extend(
            [
                "mc_var_stressed",
                "mc_expected_shortfall_stressed",
                "mc_worst_case_loss",
                "mc_stress_probability",
                "mc_recovery_distribution",
            ]
        )

        # Factor-based stress tests
        features.extend(
            [
                "systematic_risk_exposure",
                "factor_stress_contribution",
                "diversification_under_stress",
                "beta_under_stress",
            ]
        )

        # Reverse stress tests
        for threshold in self.loss_thresholds:
            features.append(f"reverse_stress_{int(threshold*100)}pct")

        # Aggregate stress metrics
        features.extend(
            [
                "stress_test_score",
                "resilience_indicator",
                "vulnerability_index",
                "expected_stress_loss",
                "stress_capital_requirement",
            ]
        )

        return features

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stress test features from return data.

        Args:
            data: DataFrame with return data

        Returns:
            DataFrame with stress test features
        """
        try:
            # Prepare returns data
            returns = self.prepare_returns(data)
            if returns.empty:
                return self._create_empty_features(data.index)

            # Initialize features DataFrame
            features = create_feature_dataframe(returns.index)

            # Calculate historical scenario impacts
            historical_features = self._calculate_historical_scenarios(returns)
            features = pd.concat([features, historical_features], axis=1)

            # Calculate parametric stress tests
            parametric_features = self._calculate_parametric_stress(returns)
            features = pd.concat([features, parametric_features], axis=1)

            # Run Monte Carlo stress simulations
            mc_features = self._calculate_monte_carlo_stress(returns)
            features = pd.concat([features, mc_features], axis=1)

            # Calculate factor-based stress tests
            factor_features = self._calculate_factor_stress(returns)
            features = pd.concat([features, factor_features], axis=1)

            # Perform reverse stress tests
            reverse_features = self._calculate_reverse_stress(returns)
            features = pd.concat([features, reverse_features], axis=1)

            # Calculate aggregate stress metrics
            aggregate_features = self._calculate_aggregate_metrics(features, returns)
            features = pd.concat([features, aggregate_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating stress test features: {e}")
            return self._create_empty_features(data.index)

    def _calculate_historical_scenarios(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate impact of historical stress scenarios."""
        features = pd.DataFrame(index=returns.index)

        # Current portfolio statistics
        current_vol = returns.rolling(window=20).std().iloc[-1]
        current_mean = returns.rolling(window=20).mean().iloc[-1]

        for scenario in self.historical_scenarios:
            if scenario not in self.scenario_definitions:
                continue

            scenario_def = self.scenario_definitions[scenario]

            # Calculate scenario impact
            market_impact = scenario_def["market_shock"]
            vol_multiplier = scenario_def["volatility_spike"]

            # Direct market impact
            features[f"{scenario}_impact"] = market_impact

            # VaR under stress scenario
            stressed_vol = current_vol * vol_multiplier
            stressed_var = stats.norm.ppf(0.05) * stressed_vol + market_impact
            features[f"{scenario}_var_impact"] = stressed_var

            # Estimated recovery time (based on historical patterns)
            recovery_days = scenario_def["duration_days"]
            recovery_rate = -market_impact / recovery_days
            features[f"{scenario}_recovery_time"] = recovery_days

        return features

    def _calculate_parametric_stress(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate parametric stress test results."""
        features = pd.DataFrame(index=returns.index)

        # Current statistics
        current_return = returns.rolling(window=20).mean()
        current_vol = returns.rolling(window=20).std()

        # Market shock sensitivity
        market_shocks = self.stress_factors.get("market", [])
        market_impacts = []

        for shock in market_shocks:
            # Simple linear impact model
            impact = shock + stats.norm.ppf(0.05) * current_vol
            market_impacts.append(impact)

        if market_impacts:
            features["market_shock_sensitivity"] = pd.Series(
                [np.mean([abs(imp) for imp in market_impacts])] * len(returns), index=returns.index
            )

        # Volatility shock impact
        vol_shocks = self.stress_factors.get("volatility", [])
        vol_impacts = []

        for vol_mult in vol_shocks:
            stressed_vol = current_vol * vol_mult
            vol_impact = stats.norm.ppf(0.05) * (stressed_vol - current_vol)
            vol_impacts.append(vol_impact)

        if vol_impacts:
            features["volatility_shock_impact"] = pd.concat(vol_impacts, axis=1).mean(axis=1)

        # Correlation shock impact (simplified)
        corr_shocks = self.stress_factors.get("correlation", [])
        if corr_shocks:
            # Assume higher correlation reduces diversification
            max_corr_impact = max(corr_shocks) * current_vol * 2
            features["correlation_shock_impact"] = max_corr_impact

        # Combined stress impact
        features["combined_stress_impact"] = (
            features.get("market_shock_sensitivity", 0)
            + features.get("volatility_shock_impact", 0)
            + features.get("correlation_shock_impact", 0)
        )

        return features

    def _calculate_monte_carlo_stress(self, returns: pd.Series) -> pd.DataFrame:
        """Run Monte Carlo stress simulations."""
        features = pd.DataFrame(index=returns.index)

        # Historical statistics for simulation
        hist_mean = returns.mean()
        hist_vol = returns.std()

        # Stress parameters (increase vol, negative drift)
        stressed_mean = hist_mean - 2 * hist_vol  # Negative drift
        stressed_vol = hist_vol * 3  # Triple volatility

        # Run simulations
        np.random.seed(42)  # For reproducibility
        simulated_returns = secure_numpy_normal(
            stressed_mean, stressed_vol, (self.n_simulations, self.simulation_horizon)
        )

        # Calculate cumulative returns for each simulation
        cumulative_returns = (1 + simulated_returns).cumprod(axis=1) - 1
        final_returns = cumulative_returns[:, -1]

        # Monte Carlo VaR
        features["mc_var_stressed"] = np.percentile(final_returns, 5)

        # Expected shortfall
        var_threshold = features["mc_var_stressed"].iloc[0]
        tail_returns = final_returns[final_returns <= var_threshold]
        features["mc_expected_shortfall_stressed"] = (
            np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
        )

        # Worst case loss
        features["mc_worst_case_loss"] = np.min(final_returns)

        # Probability of significant loss (>10%)
        features["mc_stress_probability"] = (final_returns < -0.10).mean()

        # Recovery time distribution (simplified)
        recovery_times = []
        for sim in cumulative_returns:
            # Find first period where returns recover to 0
            recovery_idx = np.where(sim > 0)[0]
            recovery_time = recovery_idx[0] if len(recovery_idx) > 0 else self.simulation_horizon
            recovery_times.append(recovery_time)

        features["mc_recovery_distribution"] = np.mean(recovery_times)

        return features

    def _calculate_factor_stress(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate factor-based stress tests."""
        features = pd.DataFrame(index=returns.index)

        # Simplified factor model (market factor only)
        # In practice, you'd have multiple factors

        # Market beta estimation
        market_proxy = returns  # Simplified - normally use market index
        rolling_beta = returns.rolling(window=60).apply(
            lambda x: np.corrcoef(x, market_proxy[-len(x) :])[0, 1]
            * (x.std() / market_proxy[-len(x) :].std()),
            raw=False,
        )

        # Systematic risk exposure
        features["systematic_risk_exposure"] = rolling_beta * returns.std()

        # Factor stress contribution
        market_stress = -0.20  # 20% market drop
        features["factor_stress_contribution"] = rolling_beta * market_stress

        # Diversification under stress (simplified)
        # Assume correlations increase to 0.8 under stress
        stress_correlation = 0.8
        normal_correlation = 0.3

        diversification_benefit = 1 - stress_correlation / (1 - normal_correlation)
        features["diversification_under_stress"] = diversification_benefit

        # Beta under stress (tends to increase)
        features["beta_under_stress"] = rolling_beta * 1.5  # 50% increase

        return features

    def _calculate_reverse_stress(self, returns: pd.Series) -> pd.DataFrame:
        """Perform reverse stress testing."""
        features = pd.DataFrame(index=returns.index)

        # Current statistics
        current_vol = returns.rolling(window=20).std()

        for threshold in self.loss_thresholds:
            # Calculate required shock for given loss threshold
            # Using inverse normal CDF
            z_score = stats.norm.ppf(0.05)  # 5% probability

            # Required shock = threshold / (z_score * volatility)
            required_shock = safe_divide(-threshold, z_score * current_vol)

            features[f"reverse_stress_{int(threshold*100)}pct"] = required_shock

        return features

    def _calculate_aggregate_metrics(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        """Calculate aggregate stress test metrics."""
        aggregate = pd.DataFrame(index=returns.index)

        # Stress test score (weighted average of scenarios)
        scenario_impacts = []

        for scenario in self.historical_scenarios:
            impact_col = f"{scenario}_impact"
            if impact_col in features:
                scenario_impacts.append(abs(features[impact_col]))

        if scenario_impacts:
            aggregate["stress_test_score"] = pd.concat(scenario_impacts, axis=1).mean(axis=1)

        # Resilience indicator (inverse of stress score)
        aggregate["resilience_indicator"] = 1 / (1 + aggregate.get("stress_test_score", 0))

        # Vulnerability index
        vulnerability_components = [
            features.get("mc_stress_probability", 0) * 100,
            abs(features.get("mc_var_stressed", 0)) * 10,
            features.get("systematic_risk_exposure", 0) * 50,
        ]

        aggregate["vulnerability_index"] = np.mean(
            [c for c in vulnerability_components if isinstance(c, (int, float)) or not c.empty]
        )

        # Expected stress loss
        expected_losses = []

        # Weight scenarios by likelihood
        scenario_weights = {
            "black_monday_1987": 0.05,
            "asian_crisis_1997": 0.10,
            "dot_com_crash_2000": 0.15,
            "financial_crisis_2008": 0.20,
            "covid_crash_2020": 0.25,
        }

        for scenario, weight in scenario_weights.items():
            impact_col = f"{scenario}_impact"
            if impact_col in features:
                expected_losses.append(features[impact_col] * weight)

        if expected_losses:
            aggregate["expected_stress_loss"] = sum(expected_losses)

        # Stress capital requirement (buffer needed)
        # Based on worst expected loss plus safety margin
        worst_loss = features.get("mc_worst_case_loss", -0.30)
        aggregate["stress_capital_requirement"] = abs(worst_loss) * 1.2

        return aggregate

    def prepare_returns(self, data: pd.DataFrame) -> pd.Series:
        """Prepare returns data from input DataFrame."""
        if "returns" in data.columns:
            return data["returns"]
        elif "close" in data.columns:
            return data["close"].pct_change().dropna()
        else:
            # Assume single column is returns
            return data.iloc[:, 0]

    def calculate_custom_scenario(
        self, returns: pd.Series, scenario: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate impact of custom stress scenario.

        Args:
            returns: Historical returns
            scenario: Custom scenario definition

        Returns:
            Dictionary with scenario impacts
        """
        results = {}

        # Extract scenario parameters
        market_shock = scenario.get("market_shock", 0)
        vol_multiplier = scenario.get("volatility_spike", 1)

        # Current statistics
        current_vol = returns.std()

        # Calculate impacts
        results["direct_impact"] = market_shock
        results["volatility_impact"] = current_vol * (vol_multiplier - 1)
        results["total_impact"] = market_shock + stats.norm.ppf(0.05) * current_vol * vol_multiplier
        results["var_95_stressed"] = stats.norm.ppf(0.05) * current_vol * vol_multiplier

        return results

"""
Value at Risk (VaR) Based Position Sizing

Implements position sizing algorithms based on Value at Risk calculations
to ensure portfolio risk stays within defined limits.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.models.common import Position
from main.risk_management.portfolio_risk import PortfolioRiskManager
from main.risk_management.risk_calculator import RiskCalculator

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class PositionSizingMethod(Enum):
    """Position sizing methods."""

    EQUAL_VAR = "equal_var"  # Each position contributes equally to VaR
    RISK_PARITY = "risk_parity"  # Risk parity approach
    MAX_VAR = "max_var"  # Maximum VaR per position
    KELLY = "kelly"  # Kelly criterion with VaR constraint


@dataclass
class VaRConstraints:
    """VaR-based risk constraints."""

    max_portfolio_var: float  # Maximum portfolio VaR (% of portfolio)
    max_position_var: float  # Maximum VaR per position (% of portfolio)
    confidence_level: float = 0.95  # VaR confidence level
    time_horizon: int = 1  # Time horizon in days
    var_method: VaRMethod = VaRMethod.HISTORICAL
    lookback_days: int = 252  # Historical data lookback


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation."""

    symbol: str
    recommended_shares: int
    recommended_value: float
    position_var: float
    contribution_to_portfolio_var: float
    confidence: float
    constraints_met: bool
    warnings: list[str]


class VaRPositionSizer:
    """
    Position sizing based on Value at Risk calculations.

    Features:
    - Multiple VaR calculation methods
    - Position sizing to meet VaR constraints
    - Marginal VaR calculations
    - Risk budgeting across positions
    """

    def __init__(
        self,
        risk_calculator: RiskCalculator,
        portfolio_risk_manager: PortfolioRiskManager,
        constraints: VaRConstraints,
    ):
        """
        Initialize VaR position sizer.

        Args:
            risk_calculator: Risk calculation engine
            portfolio_risk_manager: Portfolio risk manager
            constraints: VaR-based constraints
        """
        self.risk_calculator = risk_calculator
        self.portfolio_risk_manager = portfolio_risk_manager
        self.constraints = constraints

        # Cache for performance
        self._var_cache: dict[str, tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)

    async def calculate_position_size(
        self,
        symbol: str,
        current_portfolio: list[Position],
        portfolio_value: float,
        method: PositionSizingMethod = PositionSizingMethod.MAX_VAR,
        target_var_contribution: float | None = None,
    ) -> PositionSizeRecommendation:
        """
        Calculate recommended position size based on VaR constraints.

        Args:
            symbol: Symbol to size position for
            current_portfolio: Current portfolio positions
            portfolio_value: Total portfolio value
            method: Position sizing method
            target_var_contribution: Target VaR contribution (for EQUAL_VAR method)

        Returns:
            Position sizing recommendation
        """
        # Get current price
        current_price = await self._get_current_price(symbol)

        # Calculate individual asset VaR
        asset_var_pct = await self._calculate_asset_var(symbol)

        # Calculate position size based on method
        if method == PositionSizingMethod.MAX_VAR:
            size_recommendation = await self._size_by_max_var(
                symbol, current_price, asset_var_pct, portfolio_value
            )
        elif method == PositionSizingMethod.EQUAL_VAR:
            size_recommendation = await self._size_by_equal_var(
                symbol,
                current_price,
                asset_var_pct,
                portfolio_value,
                current_portfolio,
                target_var_contribution,
            )
        elif method == PositionSizingMethod.RISK_PARITY:
            size_recommendation = await self._size_by_risk_parity(
                symbol, current_price, asset_var_pct, portfolio_value, current_portfolio
            )
        elif method == PositionSizingMethod.KELLY:
            size_recommendation = await self._size_by_kelly_var(
                symbol, current_price, asset_var_pct, portfolio_value
            )
        else:
            raise ValueError(f"Unsupported sizing method: {method}")

        # Check portfolio VaR with new position
        portfolio_var_check = await self._check_portfolio_var_impact(
            symbol, size_recommendation["shares"], current_portfolio, portfolio_value
        )

        # Build recommendation
        return PositionSizeRecommendation(
            symbol=symbol,
            recommended_shares=size_recommendation["shares"],
            recommended_value=size_recommendation["value"],
            position_var=size_recommendation["position_var"],
            contribution_to_portfolio_var=portfolio_var_check["marginal_var"],
            confidence=size_recommendation.get("confidence", 1.0),
            constraints_met=portfolio_var_check["constraints_met"],
            warnings=size_recommendation.get("warnings", [])
            + portfolio_var_check.get("warnings", []),
        )

    async def _calculate_asset_var(self, symbol: str) -> float:
        """
        Calculate VaR for a single asset.

        Returns VaR as percentage of asset value.
        """
        # Check cache
        if symbol in self._var_cache:
            var_value, cache_time = self._var_cache[symbol]
            if datetime.now() - cache_time < self._cache_ttl:
                return var_value

        # Calculate based on method
        if self.constraints.var_method == VaRMethod.HISTORICAL:
            var_pct = await self._calculate_historical_var(symbol)
        elif self.constraints.var_method == VaRMethod.PARAMETRIC:
            var_pct = await self._calculate_parametric_var(symbol)
        elif self.constraints.var_method == VaRMethod.MONTE_CARLO:
            var_pct = await self._calculate_monte_carlo_var(symbol)
        else:
            raise ValueError(f"Unsupported VaR method: {self.constraints.var_method}")

        # Cache result
        self._var_cache[symbol] = (var_pct, datetime.now())

        return var_pct

    async def _calculate_historical_var(self, symbol: str) -> float:
        """Calculate historical VaR."""
        # Get historical returns
        returns = await self.risk_calculator.get_historical_returns(
            symbol, lookback_days=self.constraints.lookback_days
        )

        if returns is None or len(returns) < 30:
            # Not enough data, use default high VaR
            logger.warning(f"Insufficient data for {symbol}, using default VaR")
            return 0.05  # 5% daily VaR

        # Calculate percentile
        percentile = (1 - self.constraints.confidence_level) * 100
        var_return = np.percentile(returns, percentile)

        # Scale to time horizon if needed
        if self.constraints.time_horizon > 1:
            var_return = var_return * np.sqrt(self.constraints.time_horizon)

        return abs(var_return)

    async def _calculate_parametric_var(self, symbol: str) -> float:
        """Calculate parametric (Gaussian) VaR."""
        # Get volatility
        volatility = await self.risk_calculator.calculate_volatility(
            symbol, lookback_days=self.constraints.lookback_days
        )

        # Get z-score for confidence level
        # Third-party imports
        from scipy import stats

        z_score = stats.norm.ppf(1 - self.constraints.confidence_level)

        # Calculate VaR
        var_return = abs(z_score * volatility)

        # Scale to time horizon
        if self.constraints.time_horizon > 1:
            var_return = var_return * np.sqrt(self.constraints.time_horizon)

        return var_return

    async def _calculate_monte_carlo_var(self, symbol: str) -> float:
        """Calculate Monte Carlo VaR."""
        # Get historical data for parameter estimation
        returns = await self.risk_calculator.get_historical_returns(
            symbol, lookback_days=self.constraints.lookback_days
        )

        if returns is None or len(returns) < 30:
            # Fall back to parametric
            return await self._calculate_parametric_var(symbol)

        # Estimate parameters
        mu = np.mean(returns)
        sigma = np.std(returns)

        # Run Monte Carlo simulation
        num_simulations = 10000
        simulated_returns = secure_numpy_normal(
            mu * self.constraints.time_horizon,
            sigma * np.sqrt(self.constraints.time_horizon),
            num_simulations,
        )

        # Calculate VaR
        percentile = (1 - self.constraints.confidence_level) * 100
        var_return = np.percentile(simulated_returns, percentile)

        return abs(var_return)

    async def _size_by_max_var(
        self, symbol: str, current_price: float, asset_var_pct: float, portfolio_value: float
    ) -> dict[str, Any]:
        """Size position based on maximum VaR constraint."""
        # Maximum position value based on VaR constraint
        max_position_value = (self.constraints.max_position_var * portfolio_value) / asset_var_pct

        # Ensure we don't exceed portfolio percentage limits
        max_position_pct = 0.20  # 20% max position size
        max_position_value = min(max_position_value, portfolio_value * max_position_pct)

        # Calculate shares
        shares = int(max_position_value / current_price)
        actual_value = shares * current_price
        position_var = actual_value * asset_var_pct

        warnings = []
        if shares == 0:
            warnings.append("Position size too small given constraints")

        return {
            "shares": shares,
            "value": actual_value,
            "position_var": position_var,
            "warnings": warnings,
        }

    async def _size_by_equal_var(
        self,
        symbol: str,
        current_price: float,
        asset_var_pct: float,
        portfolio_value: float,
        current_portfolio: list[Position],
        target_var_contribution: float | None,
    ) -> dict[str, Any]:
        """Size position for equal VaR contribution across portfolio."""
        # Determine target VaR contribution
        if target_var_contribution is None:
            # Calculate based on number of positions
            num_positions = len(current_portfolio) + 1  # Including new position
            target_var_contribution = self.constraints.max_portfolio_var / num_positions

        # Position value for target VaR contribution
        target_position_value = (target_var_contribution * portfolio_value) / asset_var_pct

        # Calculate shares
        shares = int(target_position_value / current_price)
        actual_value = shares * current_price
        position_var = actual_value * asset_var_pct

        return {
            "shares": shares,
            "value": actual_value,
            "position_var": position_var,
            "confidence": 0.9,  # Equal VaR is approximate
        }

    async def _size_by_risk_parity(
        self,
        symbol: str,
        current_price: float,
        asset_var_pct: float,
        portfolio_value: float,
        current_portfolio: list[Position],
    ) -> dict[str, Any]:
        """Size position using risk parity approach."""
        # Get correlation matrix and volatilities
        symbols = [pos.symbol for pos in current_portfolio] + [symbol]

        # Build covariance matrix
        returns_data = {}
        for sym in symbols:
            returns = await self.risk_calculator.get_historical_returns(
                sym, lookback_days=self.constraints.lookback_days
            )
            if returns is not None:
                returns_data[sym] = returns

        if len(returns_data) < len(symbols):
            # Missing data, fall back to max VaR
            return await self._size_by_max_var(
                symbol, current_price, asset_var_pct, portfolio_value
            )

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov()

        # Risk parity optimization (simplified)
        # Target: each asset contributes equally to portfolio risk
        n_assets = len(symbols)

        # Initial equal weights
        weights = np.ones(n_assets) / n_assets

        # Iterative optimization (simplified)
        for _ in range(10):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib

            # Adjust weights
            weights = weights * (1 / n_assets) / contrib
            weights = weights / weights.sum()

        # Get weight for new symbol
        new_symbol_weight = weights[-1]

        # Calculate position size
        position_value = portfolio_value * new_symbol_weight
        shares = int(position_value / current_price)
        actual_value = shares * current_price
        position_var = actual_value * asset_var_pct

        return {
            "shares": shares,
            "value": actual_value,
            "position_var": position_var,
            "confidence": 0.8,  # Risk parity is model-dependent
        }

    async def _size_by_kelly_var(
        self, symbol: str, current_price: float, asset_var_pct: float, portfolio_value: float
    ) -> dict[str, Any]:
        """Size position using Kelly criterion with VaR constraint."""
        # Get expected return and volatility
        returns = await self.risk_calculator.get_historical_returns(
            symbol, lookback_days=self.constraints.lookback_days
        )

        if returns is None or len(returns) < 60:
            # Not enough data for Kelly
            return await self._size_by_max_var(
                symbol, current_price, asset_var_pct, portfolio_value
            )

        # Calculate Kelly fraction
        mean_return = np.mean(returns)
        variance = np.var(returns)

        # Kelly fraction: f = μ/σ²
        kelly_fraction = mean_return / variance if variance > 0 else 0

        # Apply Kelly fraction with safety factor
        safety_factor = 0.25  # Use 25% of Kelly
        adjusted_kelly = kelly_fraction * safety_factor

        # Ensure Kelly doesn't violate VaR constraint
        kelly_position_value = portfolio_value * adjusted_kelly
        max_var_position_value = (
            self.constraints.max_position_var * portfolio_value
        ) / asset_var_pct

        position_value = min(kelly_position_value, max_var_position_value)

        # Calculate shares
        shares = max(0, int(position_value / current_price))
        actual_value = shares * current_price
        position_var = actual_value * asset_var_pct

        warnings = []
        if kelly_fraction < 0:
            warnings.append("Negative Kelly fraction - negative expected return")
            shares = 0
            actual_value = 0
            position_var = 0

        return {
            "shares": shares,
            "value": actual_value,
            "position_var": position_var,
            "confidence": 0.7,  # Kelly is sensitive to estimation error
            "warnings": warnings,
        }

    async def _check_portfolio_var_impact(
        self, symbol: str, shares: int, current_portfolio: list[Position], portfolio_value: float
    ) -> dict[str, Any]:
        """Check portfolio VaR impact of adding position."""
        # Create portfolio DataFrame with new position
        positions_data = []

        for pos in current_portfolio:
            positions_data.append(
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "current_price": pos.current_price,
                    "market_value": pos.quantity * pos.current_price,
                }
            )

        # Add new position
        current_price = await self._get_current_price(symbol)
        positions_data.append(
            {
                "symbol": symbol,
                "quantity": shares,
                "current_price": current_price,
                "market_value": shares * current_price,
            }
        )

        positions_df = pd.DataFrame(positions_data)

        # Calculate portfolio VaR
        portfolio_var = await self.portfolio_risk_manager.calculate_portfolio_var(
            positions_df,
            confidence_level=self.constraints.confidence_level,
            time_horizon=self.constraints.time_horizon,
        )

        portfolio_var_pct = portfolio_var / portfolio_value

        # Calculate marginal VaR (approximate)
        # Remove new position and recalculate
        positions_df_without = positions_df[positions_df["symbol"] != symbol]

        if not positions_df_without.empty:
            portfolio_var_without = await self.portfolio_risk_manager.calculate_portfolio_var(
                positions_df_without,
                confidence_level=self.constraints.confidence_level,
                time_horizon=self.constraints.time_horizon,
            )
            marginal_var = portfolio_var - portfolio_var_without
        else:
            marginal_var = portfolio_var

        marginal_var_pct = marginal_var / portfolio_value

        # Check constraints
        constraints_met = portfolio_var_pct <= self.constraints.max_portfolio_var

        warnings = []
        if not constraints_met:
            warnings.append(
                f"Portfolio VaR ({portfolio_var_pct:.2%}) exceeds limit ({self.constraints.max_portfolio_var:.2%})"
            )

        return {
            "portfolio_var": portfolio_var_pct,
            "marginal_var": marginal_var_pct,
            "constraints_met": constraints_met,
            "warnings": warnings,
        }

    async def calculate_var_efficient_portfolio(
        self, candidate_symbols: list[str], portfolio_value: float, target_var: float
    ) -> list[PositionSizeRecommendation]:
        """
        Calculate VaR-efficient portfolio allocation.

        Args:
            candidate_symbols: List of symbols to consider
            portfolio_value: Total portfolio value
            target_var: Target portfolio VaR

        Returns:
            List of position recommendations
        """
        # Get returns data for all candidates
        returns_data = {}
        for symbol in candidate_symbols:
            returns = await self.risk_calculator.get_historical_returns(
                symbol, lookback_days=self.constraints.lookback_days
            )
            if returns is not None and len(returns) >= 60:
                returns_data[symbol] = returns

        if not returns_data:
            return []

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        # Optimize for maximum Sharpe ratio with VaR constraint
        # This is a simplified version - in production would use proper optimization
        n_assets = len(returns_data)
        symbols = list(returns_data.keys())

        # Start with equal weights
        weights = np.ones(n_assets) / n_assets

        # Simple optimization loop
        best_weights = weights.copy()
        best_sharpe = -np.inf

        # Try different weight combinations
        for _ in range(100):
            # Random perturbation
            weights = np.random.dirichlet(np.ones(n_assets))

            # Calculate portfolio metrics
            portfolio_return = weights @ expected_returns
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

            # Calculate VaR
            var_pct = abs(stats.norm.ppf(1 - self.constraints.confidence_level)) * portfolio_vol

            if var_pct <= target_var:
                # Calculate Sharpe ratio (assuming 0 risk-free rate)
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights.copy()

        # Convert weights to position recommendations
        recommendations = []

        for i, symbol in enumerate(symbols):
            if best_weights[i] > 0.01:  # At least 1% allocation
                current_price = await self._get_current_price(symbol)
                position_value = portfolio_value * best_weights[i]
                shares = int(position_value / current_price)

                if shares > 0:
                    asset_var_pct = await self._calculate_asset_var(symbol)
                    position_var = shares * current_price * asset_var_pct

                    recommendations.append(
                        PositionSizeRecommendation(
                            symbol=symbol,
                            recommended_shares=shares,
                            recommended_value=shares * current_price,
                            position_var=position_var,
                            contribution_to_portfolio_var=position_var / portfolio_value,
                            confidence=0.7,
                            constraints_met=True,
                            warnings=[],
                        )
                    )

        return recommendations

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # This would connect to market data in production
        # For now, return a placeholder
        return 100.0

    def adjust_for_correlation(
        self,
        base_size: int,
        symbol: str,
        current_portfolio: list[Position],
        correlation_threshold: float = 0.7,
    ) -> int:
        """
        Adjust position size based on correlation with existing positions.

        Args:
            base_size: Base position size in shares
            symbol: Symbol to size
            current_portfolio: Current portfolio positions
            correlation_threshold: Threshold for high correlation

        Returns:
            Adjusted position size
        """
        # This is a simplified version
        # In production, would calculate actual correlations

        # Check for highly correlated positions
        high_correlation_count = 0

        # Reduce size if highly correlated positions exist
        if high_correlation_count > 0:
            reduction_factor = 1 - (0.1 * high_correlation_count)  # 10% reduction per correlation
            adjusted_size = int(base_size * max(0.5, reduction_factor))  # At least 50% of base
        else:
            adjusted_size = base_size

        return adjusted_size

"""
Risk-specific position validation.

This module provides advanced risk validation for positions including VaR,
stress testing, correlation analysis, and portfolio-level risk metrics.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from typing import Any

# Third-party imports
import numpy as np

# Local imports
from main.models.common import Position, PositionSide
from main.trading_engine.core.position_validator import ValidationContext
from main.utils.core import AITraderException, ValidationResult, async_lru_cache
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class RiskValidationError(AITraderException):
    """Raised when risk validation fails."""

    pass


@dataclass(frozen=True)
class RiskLimits:
    """Risk-specific limits for positions."""

    max_var_95: Decimal  # Maximum 95% VaR as % of portfolio
    max_var_99: Decimal  # Maximum 99% VaR as % of portfolio
    max_expected_shortfall: Decimal  # Maximum expected shortfall
    max_beta: Decimal  # Maximum beta to market
    max_correlation: Decimal  # Maximum correlation between positions
    max_volatility: Decimal  # Maximum position volatility
    max_drawdown: Decimal  # Maximum acceptable drawdown
    max_sector_concentration: Decimal  # Maximum sector concentration
    stress_test_scenarios: list[str]  # Required stress test scenarios


@dataclass(frozen=True)
class RiskMetrics:
    """Calculated risk metrics for a position or portfolio."""

    var_95: Decimal  # 95% Value at Risk
    var_99: Decimal  # 99% Value at Risk
    expected_shortfall: Decimal  # Expected shortfall (CVaR)
    volatility: Decimal  # Annualized volatility
    beta: Decimal  # Beta to market
    sharpe_ratio: Decimal  # Sharpe ratio
    max_drawdown: Decimal  # Maximum drawdown
    correlation_matrix: np.ndarray | None = None  # Correlation matrix


@dataclass(frozen=True)
class StressTestResult:
    """Result of a stress test scenario."""

    scenario_name: str
    portfolio_loss: Decimal
    position_losses: dict[str, Decimal]
    passed: bool


@dataclass
class PositionRiskAssessment:
    """Complete risk assessment for a position."""

    position: Position
    risk_metrics: RiskMetrics
    stress_test_results: list[StressTestResult]
    validation_result: ValidationResult
    timestamp: datetime
    assessment_id: str


class PositionRiskValidator:
    """
    Advanced risk validation for positions.

    Performs sophisticated risk analysis including VaR, stress testing,
    correlation analysis, and portfolio optimization constraints.
    """

    def __init__(
        self,
        risk_limits: RiskLimits,
        market_data_provider: Any,  # Interface to get historical data
        metrics_collector: MetricsCollector | None = None,
    ):
        """
        Initialize risk validator.

        Args:
            risk_limits: Risk limit configuration
            market_data_provider: Provider for market data and history
            metrics_collector: Optional metrics collector
        """
        self.limits = risk_limits
        self.market_data = market_data_provider
        self.metrics = metrics_collector

    async def validate_position_risk(
        self, symbol: str, side: PositionSide, quantity: Decimal, context: ValidationContext
    ) -> ValidationResult:
        """
        Validate risk metrics for a new or modified position.

        Args:
            symbol: Symbol for position
            side: Position side
            quantity: Position quantity
            context: Validation context

        Returns:
            Validation result with risk-based errors/warnings
        """
        errors = []
        warnings = []

        try:
            # Get current market data
            current_price = await self._get_current_price(symbol)
            position_value = quantity * current_price

            # Calculate position risk metrics
            position_metrics = await self._calculate_position_metrics(
                symbol, side, quantity, current_price
            )

            # Validate volatility
            if position_metrics.volatility > self.limits.max_volatility:
                errors.append(
                    f"Position volatility {position_metrics.volatility:.1%} exceeds "
                    f"limit {self.limits.max_volatility:.1%}"
                )

            # Validate beta
            if abs(position_metrics.beta) > self.limits.max_beta:
                errors.append(
                    f"Position beta {position_metrics.beta:.2f} exceeds "
                    f"limit {self.limits.max_beta:.2f}"
                )

            # Calculate portfolio metrics with new position
            portfolio_metrics = await self._calculate_portfolio_metrics(
                context.existing_positions, symbol, side, quantity, current_price
            )

            # Validate VaR
            portfolio_value = context.account_info.portfolio_value
            if portfolio_value > 0:
                var_95_pct = (portfolio_metrics.var_95 / portfolio_value) * 100
                var_99_pct = (portfolio_metrics.var_99 / portfolio_value) * 100

                if var_95_pct > self.limits.max_var_95:
                    errors.append(
                        f"Portfolio 95% VaR {var_95_pct:.1f}% exceeds "
                        f"limit {self.limits.max_var_95:.1f}%"
                    )

                if var_99_pct > self.limits.max_var_99:
                    errors.append(
                        f"Portfolio 99% VaR {var_99_pct:.1f}% exceeds "
                        f"limit {self.limits.max_var_99:.1f}%"
                    )

                # Warnings for approaching limits
                if var_95_pct > self.limits.max_var_95 * Decimal("0.8"):
                    warnings.append(f"Portfolio 95% VaR {var_95_pct:.1f}% approaching limit")

            # Validate expected shortfall
            if portfolio_metrics.expected_shortfall > self.limits.max_expected_shortfall:
                errors.append(
                    f"Expected shortfall {portfolio_metrics.expected_shortfall:.1%} exceeds "
                    f"limit {self.limits.max_expected_shortfall:.1%}"
                )

            # Check correlations
            if portfolio_metrics.correlation_matrix is not None:
                max_correlation = self._check_correlations(
                    portfolio_metrics.correlation_matrix, context.existing_positions, symbol
                )
                if max_correlation > self.limits.max_correlation:
                    errors.append(
                        f"Maximum position correlation {max_correlation:.2f} exceeds "
                        f"limit {self.limits.max_correlation:.2f}"
                    )

            # Run stress tests
            stress_results = await self._run_stress_tests(
                context.existing_positions, symbol, side, quantity, current_price
            )

            for result in stress_results:
                if not result.passed:
                    errors.append(
                        f"Failed stress test '{result.scenario_name}': loss {result.portfolio_loss:.1%}"
                    )

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            warnings.append(f"Unable to fully validate risk: {e}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    async def validate_portfolio_risk(self, context: ValidationContext) -> ValidationResult:
        """
        Validate overall portfolio risk.

        Args:
            context: Validation context

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        try:
            # Calculate current portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(context.existing_positions)

            portfolio_value = context.account_info.portfolio_value
            if portfolio_value > 0:
                # Check VaR limits
                var_95_pct = (portfolio_metrics.var_95 / portfolio_value) * 100
                var_99_pct = (portfolio_metrics.var_99 / portfolio_value) * 100

                if var_95_pct > self.limits.max_var_95:
                    errors.append(
                        f"Portfolio 95% VaR {var_95_pct:.1f}% exceeds limit {self.limits.max_var_95:.1f}%"
                    )

                if var_99_pct > self.limits.max_var_99:
                    errors.append(
                        f"Portfolio 99% VaR {var_99_pct:.1f}% exceeds limit {self.limits.max_var_99:.1f}%"
                    )

            # Check drawdown
            if portfolio_metrics.max_drawdown > self.limits.max_drawdown:
                errors.append(
                    f"Portfolio drawdown {portfolio_metrics.max_drawdown:.1%} exceeds "
                    f"limit {self.limits.max_drawdown:.1%}"
                )

            # Check sector concentration
            sector_concentration = await self._calculate_sector_concentration(
                context.existing_positions
            )

            for sector, concentration in sector_concentration.items():
                if concentration > self.limits.max_sector_concentration:
                    errors.append(
                        f"Sector {sector} concentration {concentration:.1%} exceeds "
                        f"limit {self.limits.max_sector_concentration:.1%}"
                    )

        except Exception as e:
            logger.error(f"Error validating portfolio risk: {e}")
            warnings.append(f"Unable to fully validate portfolio risk: {e}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    async def _calculate_position_metrics(
        self, symbol: str, side: PositionSide, quantity: Decimal, current_price: Decimal
    ) -> RiskMetrics:
        """Calculate risk metrics for a single position."""
        # Get historical prices
        prices = await self._get_historical_prices(symbol, lookback_days=252)
        returns = np.diff(np.log(prices))

        # Calculate volatility
        volatility = Decimal(str(np.std(returns) * np.sqrt(252)))

        # Calculate beta (simplified - against SPY)
        market_prices = await self._get_historical_prices("SPY", lookback_days=252)
        market_returns = np.diff(np.log(market_prices))

        if len(returns) == len(market_returns):
            beta = Decimal(str(np.cov(returns, market_returns)[0, 1] / np.var(market_returns)))
        else:
            beta = Decimal("1.0")

        # Calculate VaR
        position_value = float(quantity * current_price)
        var_95 = Decimal(str(abs(np.percentile(returns, 5) * position_value)))
        var_99 = Decimal(str(abs(np.percentile(returns, 1) * position_value)))

        # Calculate expected shortfall (CVaR)
        worst_returns = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = Decimal(str(abs(np.mean(worst_returns))))

        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = Decimal(str(np.mean(returns) / np.std(returns) * np.sqrt(252)))

        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = Decimal(str(abs(np.min(drawdown))))

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
        )

    async def _calculate_portfolio_metrics(
        self,
        positions: list[Position],
        new_symbol: str | None = None,
        new_side: PositionSide | None = None,
        new_quantity: Decimal | None = None,
        new_price: Decimal | None = None,
    ) -> RiskMetrics:
        """Calculate risk metrics for entire portfolio."""
        # Build position list including potential new position
        symbols = [p.symbol for p in positions]
        weights = []

        total_value = sum(p.market_value for p in positions)

        if new_symbol and new_quantity and new_price:
            symbols.append(new_symbol)
            new_value = new_quantity * new_price
            total_value += new_value

        if total_value == 0:
            # Return zero metrics for empty portfolio
            return RiskMetrics(
                var_95=Decimal("0"),
                var_99=Decimal("0"),
                expected_shortfall=Decimal("0"),
                volatility=Decimal("0"),
                beta=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
            )

        # Calculate weights
        for position in positions:
            weights.append(float(position.market_value / total_value))

        if new_symbol and new_quantity and new_price:
            weights.append(float(new_value / total_value))

        # Get returns for all symbols
        all_returns = []
        for symbol in symbols:
            prices = await self._get_historical_prices(symbol, lookback_days=252)
            returns = np.diff(np.log(prices))
            all_returns.append(returns)

        # Align returns to same length
        min_length = min(len(r) for r in all_returns)
        all_returns = [r[-min_length:] for r in all_returns]
        returns_matrix = np.array(all_returns).T

        # Calculate portfolio returns
        weights_array = np.array(weights)
        portfolio_returns = returns_matrix @ weights_array

        # Calculate portfolio metrics
        portfolio_volatility = Decimal(str(np.std(portfolio_returns) * np.sqrt(252)))
        portfolio_var_95 = Decimal(
            str(abs(np.percentile(portfolio_returns, 5) * float(total_value)))
        )
        portfolio_var_99 = Decimal(
            str(abs(np.percentile(portfolio_returns, 1) * float(total_value)))
        )

        worst_returns = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]
        portfolio_es = Decimal(str(abs(np.mean(worst_returns))))

        # Portfolio beta
        market_prices = await self._get_historical_prices("SPY", lookback_days=252)
        market_returns = np.diff(np.log(market_prices))[-min_length:]
        portfolio_beta = Decimal(
            str(np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns))
        )

        # Sharpe ratio
        portfolio_sharpe = Decimal(
            str(np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252))
        )

        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        portfolio_max_dd = Decimal(str(abs(np.min(drawdown))))

        # Correlation matrix
        correlation_matrix = np.corrcoef(all_returns) if len(all_returns) > 1 else None

        return RiskMetrics(
            var_95=portfolio_var_95,
            var_99=portfolio_var_99,
            expected_shortfall=portfolio_es,
            volatility=portfolio_volatility,
            beta=portfolio_beta,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=portfolio_max_dd,
            correlation_matrix=correlation_matrix,
        )

    def _check_correlations(
        self, correlation_matrix: np.ndarray, positions: list[Position], new_symbol: str
    ) -> Decimal:
        """Check maximum correlation in portfolio."""
        # Extract upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = np.triu(correlation_matrix, k=1)
        max_correlation = np.max(np.abs(upper_triangle))

        return Decimal(str(max_correlation))

    async def _run_stress_tests(
        self,
        positions: list[Position],
        new_symbol: str,
        new_side: PositionSide,
        new_quantity: Decimal,
        new_price: Decimal,
    ) -> list[StressTestResult]:
        """Run stress test scenarios."""
        results = []

        for scenario in self.limits.stress_test_scenarios:
            if scenario == "market_crash_10pct":
                # 10% market crash scenario
                portfolio_loss = Decimal("0")
                position_losses = {}

                for position in positions:
                    loss = position.market_value * Decimal("0.10")
                    portfolio_loss += loss
                    position_losses[position.symbol] = loss

                # Add new position loss
                new_loss = (new_quantity * new_price) * Decimal("0.10")
                portfolio_loss += new_loss
                position_losses[new_symbol] = new_loss

                # Pass if loss is less than 15% of portfolio
                portfolio_value = sum(p.market_value for p in positions) + (
                    new_quantity * new_price
                )
                passed = portfolio_loss < portfolio_value * Decimal("0.15")

                results.append(
                    StressTestResult(
                        scenario_name=scenario,
                        portfolio_loss=portfolio_loss,
                        position_losses=position_losses,
                        passed=passed,
                    )
                )

            elif scenario == "volatility_spike":
                # Volatility spike scenario (2x normal volatility)
                # Simplified: use 2x daily volatility as potential loss
                portfolio_metrics = await self._calculate_portfolio_metrics(
                    positions, new_symbol, new_side, new_quantity, new_price
                )

                daily_vol = portfolio_metrics.volatility / Decimal(str(np.sqrt(252)))
                portfolio_value = sum(p.market_value for p in positions) + (
                    new_quantity * new_price
                )
                potential_loss = portfolio_value * daily_vol * 2

                passed = potential_loss < portfolio_value * Decimal("0.05")

                results.append(
                    StressTestResult(
                        scenario_name=scenario,
                        portfolio_loss=potential_loss,
                        position_losses={},
                        passed=passed,
                    )
                )

        return results

    async def _calculate_sector_concentration(
        self, positions: list[Position]
    ) -> dict[str, Decimal]:
        """Calculate concentration by sector."""
        sector_values: dict[str, Decimal] = {}
        total_value = sum(p.market_value for p in positions)

        if total_value == 0:
            return {}

        for position in positions:
            # Get sector from market data provider
            sector = await self._get_symbol_sector(position.symbol)
            if sector not in sector_values:
                sector_values[sector] = Decimal("0")
            sector_values[sector] += position.market_value

        # Convert to percentages
        sector_concentration = {
            sector: value / total_value for sector, value in sector_values.items()
        }

        return sector_concentration

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        market_data = await self.market_data.get_market_data(symbol)
        return market_data.last

    @async_lru_cache(maxsize=100)
    async def _get_historical_prices(self, symbol: str, lookback_days: int) -> np.ndarray:
        """Get historical prices for symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        prices = await self.market_data.get_historical_prices(symbol, start_date, end_date)

        return np.array([float(p) for p in prices])

    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        # This would typically query a reference data service
        # For now, return a default
        return "Unknown"

"""
Position Limit Circuit Breaker

Monitors position count and concentration limits to prevent overexposure
and ensure proper risk diversification.

Created: 2025-07-15
"""

# Standard library imports
from datetime import datetime
import logging
from typing import Any

# Third-party imports
import numpy as np

from ..config import BreakerConfig
from ..registry import BaseBreaker
from ..types import BreakerMetrics, BreakerType, MarketConditions

logger = logging.getLogger(__name__)


class PositionLimitBreaker(BaseBreaker):
    """
    Circuit breaker for position limit protection.

    Monitors:
    - Total position count
    - Position concentration (single position size)
    - Sector/industry concentration
    - Leverage exposure
    """

    def __init__(self, breaker_type: BreakerType, config: BreakerConfig):
        """Initialize position limit breaker."""
        super().__init__(breaker_type, config)

        # Configuration
        self.max_positions = config.max_positions
        self.max_position_size = config.max_position_size
        self.max_sector_concentration = config.config.get(
            "max_sector_concentration", 0.30
        )  # 30% per sector
        self.max_long_exposure = config.config.get("max_long_exposure", 1.0)  # 100% long exposure
        self.max_short_exposure = config.config.get("max_short_exposure", 0.5)  # 50% short exposure

        # Warning thresholds
        self.position_warning_threshold = int(self.max_positions * 0.9)  # 90% of max
        self.concentration_warning_threshold = self.max_position_size * 0.8  # 80% of max

        # State tracking
        self.position_history = []
        self.concentration_violations = 0
        self.max_concentration_seen = 0.0

        logger.info(
            f"Position limit breaker initialized - max positions: {self.max_positions}, max size: {self.max_position_size:.2%}"
        )

    async def check(
        self, portfolio_value: float, positions: dict[str, Any], market_conditions: MarketConditions
    ) -> bool:
        """
        Check if position limit breaker should trip.

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            market_conditions: Current market conditions

        Returns:
            True if breaker should trip
        """
        if not self.is_enabled():
            return False

        # Update position history
        self.position_history.append(
            {
                "timestamp": datetime.now(),
                "position_count": len(positions),
                "positions": positions.copy(),
            }
        )

        # Keep only recent history
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]

        # Check position count limit
        if len(positions) > self.max_positions:
            self.logger.error(
                f"Position count limit exceeded: {len(positions)} > {self.max_positions}"
            )
            return True

        # Check position concentration
        if await self._check_position_concentration(positions, portfolio_value):
            return True

        # Check sector concentration
        if await self._check_sector_concentration(positions, portfolio_value):
            return True

        # Check exposure limits
        if await self._check_exposure_limits(positions, portfolio_value):
            return True

        return False

    async def check_warning_conditions(
        self, portfolio_value: float, positions: dict[str, Any], market_conditions: MarketConditions
    ) -> bool:
        """
        Check if position limit breaker should be in warning state.

        Returns:
            True if breaker should be in warning state
        """
        if not self.is_enabled():
            return False

        # Warning if approaching position count limit
        if len(positions) >= self.position_warning_threshold:
            self.logger.warning(f"Approaching position count limit: {len(positions)}")
            return True

        # Warning if approaching concentration limit
        max_concentration = self._calculate_max_concentration(positions, portfolio_value)
        if max_concentration > self.concentration_warning_threshold:
            self.logger.warning(f"Approaching concentration limit: {max_concentration:.2%}")
            return True

        return False

    async def _check_position_concentration(
        self, positions: dict[str, Any], portfolio_value: float
    ) -> bool:
        """Check if any single position exceeds concentration limits."""
        if not positions or portfolio_value <= 0:
            return False

        total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
        if total_value <= 0:
            return False

        for symbol, pos in positions.items():
            position_value = abs(pos.get("market_value", 0))
            position_pct = position_value / total_value

            # Update max concentration seen
            if position_pct > self.max_concentration_seen:
                self.max_concentration_seen = position_pct

            if position_pct > self.max_position_size:
                self.concentration_violations += 1
                self.logger.error(
                    f"Position concentration limit exceeded for {symbol}: {position_pct:.2%} > {self.max_position_size:.2%}"
                )
                return True

        return False

    async def _check_sector_concentration(
        self, positions: dict[str, Any], portfolio_value: float
    ) -> bool:
        """Check if any sector concentration exceeds limits."""
        if not positions or portfolio_value <= 0:
            return False

        # Group positions by sector
        sector_exposure = {}
        total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())

        if total_value <= 0:
            return False

        for symbol, pos in positions.items():
            sector = pos.get("sector", "Unknown")
            position_value = abs(pos.get("market_value", 0))

            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += position_value

        # Check sector concentration limits
        for sector, exposure in sector_exposure.items():
            sector_pct = exposure / total_value

            if sector_pct > self.max_sector_concentration:
                self.logger.error(
                    f"Sector concentration limit exceeded for {sector}: {sector_pct:.2%} > {self.max_sector_concentration:.2%}"
                )
                return True

        return False

    async def _check_exposure_limits(
        self, positions: dict[str, Any], portfolio_value: float
    ) -> bool:
        """Check if long/short exposure limits are exceeded."""
        if not positions or portfolio_value <= 0:
            return False

        long_exposure = 0
        short_exposure = 0

        for symbol, pos in positions.items():
            position_value = pos.get("market_value", 0)

            if position_value > 0:
                long_exposure += position_value
            else:
                short_exposure += abs(position_value)

        # Calculate exposure as percentage of portfolio
        long_exposure_pct = long_exposure / portfolio_value
        short_exposure_pct = short_exposure / portfolio_value

        # Check long exposure limit
        if long_exposure_pct > self.max_long_exposure:
            self.logger.error(
                f"Long exposure limit exceeded: {long_exposure_pct:.2%} > {self.max_long_exposure:.2%}"
            )
            return True

        # Check short exposure limit
        if short_exposure_pct > self.max_short_exposure:
            self.logger.error(
                f"Short exposure limit exceeded: {short_exposure_pct:.2%} > {self.max_short_exposure:.2%}"
            )
            return True

        return False

    def _calculate_max_concentration(
        self, positions: dict[str, Any], portfolio_value: float
    ) -> float:
        """Calculate the maximum position concentration."""
        if not positions or portfolio_value <= 0:
            return 0.0

        total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
        if total_value <= 0:
            return 0.0

        max_concentration = 0.0
        for pos in positions.values():
            position_value = abs(pos.get("market_value", 0))
            concentration = position_value / total_value
            max_concentration = max(max_concentration, concentration)

        return max_concentration

    def get_metrics(self) -> BreakerMetrics:
        """Get current position limit metrics."""
        metrics = BreakerMetrics()

        if self.position_history:
            latest = self.position_history[-1]
            metrics.position_count = latest["position_count"]

            # Calculate max position size from latest positions
            if latest["positions"]:
                total_value = sum(
                    abs(pos.get("market_value", 0)) for pos in latest["positions"].values()
                )
                if total_value > 0:
                    max_pos_value = max(
                        abs(pos.get("market_value", 0)) for pos in latest["positions"].values()
                    )
                    metrics.max_position_size = max_pos_value / total_value

        return metrics

    def get_position_statistics(self) -> dict[str, Any]:
        """Get detailed position statistics."""
        if not self.position_history:
            return {
                "current_position_count": 0,
                "max_position_count": 0,
                "avg_position_count": 0,
                "max_concentration": 0.0,
                "avg_concentration": 0.0,
                "concentration_violations": self.concentration_violations,
                "sector_analysis": {},
                "exposure_analysis": {},
            }

        latest = self.position_history[-1]
        positions = latest["positions"]

        # Position count statistics
        position_counts = [ph["position_count"] for ph in self.position_history]

        # Concentration analysis
        if positions:
            total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
            concentrations = []
            if total_value > 0:
                concentrations = [
                    abs(pos.get("market_value", 0)) / total_value for pos in positions.values()
                ]
        else:
            concentrations = []

        # Sector analysis
        sector_exposure = {}
        if positions and total_value > 0:
            for symbol, pos in positions.items():
                sector = pos.get("sector", "Unknown")
                position_value = abs(pos.get("market_value", 0))

                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += position_value / total_value

        # Exposure analysis
        long_exposure = sum(
            pos.get("market_value", 0)
            for pos in positions.values()
            if pos.get("market_value", 0) > 0
        )
        short_exposure = sum(
            abs(pos.get("market_value", 0))
            for pos in positions.values()
            if pos.get("market_value", 0) < 0
        )

        return {
            "current_position_count": len(positions),
            "max_position_count": max(position_counts) if position_counts else 0,
            "avg_position_count": np.mean(position_counts) if position_counts else 0,
            "max_concentration": max(concentrations) if concentrations else 0.0,
            "avg_concentration": np.mean(concentrations) if concentrations else 0.0,
            "concentration_violations": self.concentration_violations,
            "sector_analysis": dict(sector_exposure),
            "exposure_analysis": {
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "net_exposure": long_exposure - short_exposure,
                "gross_exposure": long_exposure + short_exposure,
            },
        }

    def get_diversification_analysis(self) -> dict[str, Any]:
        """Analyze portfolio diversification."""
        if not self.position_history:
            return {"insufficient_data": True}

        latest = self.position_history[-1]
        positions = latest["positions"]

        if not positions:
            return {"no_positions": True}

        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
        if total_value <= 0:
            return {"invalid_positions": True}

        hhi = 0
        for pos in positions.values():
            weight = abs(pos.get("market_value", 0)) / total_value
            hhi += weight**2

        # Sector diversification
        sector_weights = {}
        for symbol, pos in positions.items():
            sector = pos.get("sector", "Unknown")
            weight = abs(pos.get("market_value", 0)) / total_value

            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weight

        sector_hhi = sum(weight**2 for weight in sector_weights.values())

        # Calculate diversification scores (lower HHI = better diversification)
        position_diversification = 1 - hhi  # 0 to 1 scale
        sector_diversification = 1 - sector_hhi

        return {
            "position_hhi": hhi,
            "sector_hhi": sector_hhi,
            "position_diversification_score": position_diversification,
            "sector_diversification_score": sector_diversification,
            "effective_positions": int(1 / hhi) if hhi > 0 else 0,
            "effective_sectors": int(1 / sector_hhi) if sector_hhi > 0 else 0,
            "sector_count": len(sector_weights),
            "largest_sector_weight": max(sector_weights.values()) if sector_weights else 0,
        }

    def get_risk_contribution_analysis(self) -> dict[str, Any]:
        """Analyze risk contribution of positions."""
        if not self.position_history:
            return {"insufficient_data": True}

        latest = self.position_history[-1]
        positions = latest["positions"]

        if not positions:
            return {"no_positions": True}

        # Calculate risk contributions (simplified using position size as proxy)
        total_value = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
        if total_value <= 0:
            return {"invalid_positions": True}

        risk_contributions = {}
        for symbol, pos in positions.items():
            weight = abs(pos.get("market_value", 0)) / total_value
            volatility = pos.get("volatility", 0.2)  # Default 20% volatility if not provided

            # Risk contribution = weight * volatility (simplified)
            risk_contribution = weight * volatility
            risk_contributions[symbol] = {
                "weight": weight,
                "volatility": volatility,
                "risk_contribution": risk_contribution,
            }

        # Sort by risk contribution
        sorted_risks = sorted(
            risk_contributions.items(), key=lambda x: x[1]["risk_contribution"], reverse=True
        )

        # Top risk contributors
        top_contributors = sorted_risks[:5]
        total_risk = sum(rc["risk_contribution"] for _, rc in risk_contributions.items())

        return {
            "total_risk": total_risk,
            "top_risk_contributors": [
                {
                    "symbol": symbol,
                    "risk_contribution": rc["risk_contribution"],
                    "risk_percentage": (
                        rc["risk_contribution"] / total_risk * 100 if total_risk > 0 else 0
                    ),
                    "weight": rc["weight"],
                    "volatility": rc["volatility"],
                }
                for symbol, rc in top_contributors
            ],
            "risk_concentration": (
                max(rc["risk_contribution"] for _, rc in risk_contributions.items()) / total_risk
                if total_risk > 0
                else 0
            ),
        }

    def get_info(self) -> dict[str, Any]:
        """Get breaker information including position limit-specific details."""
        base_info = super().get_info()

        position_stats = self.get_position_statistics()
        diversification_analysis = self.get_diversification_analysis()
        risk_analysis = self.get_risk_contribution_analysis()

        base_info.update(
            {
                "max_positions": self.max_positions,
                "max_position_size": self.max_position_size,
                "max_sector_concentration": self.max_sector_concentration,
                "max_long_exposure": self.max_long_exposure,
                "max_short_exposure": self.max_short_exposure,
                "position_warning_threshold": self.position_warning_threshold,
                "concentration_warning_threshold": self.concentration_warning_threshold,
                "max_concentration_seen": self.max_concentration_seen,
                "concentration_violations": self.concentration_violations,
                "current_stats": position_stats,
                "diversification_analysis": diversification_analysis,
                "risk_analysis": risk_analysis,
            }
        )

        return base_info

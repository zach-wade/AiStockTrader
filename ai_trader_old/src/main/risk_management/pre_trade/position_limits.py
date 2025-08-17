"""
Position size limits
Created: 2025-06-16
"""

"""Position size and exposure limit checks."""
# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.utils.cache import CacheType

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of position limits."""

    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    CORRELATION_EXPOSURE = "correlation_exposure"
    BETA_EXPOSURE = "beta_exposure"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class PositionLimit:
    """Position limit definition."""

    limit_type: LimitType
    limit_value: float
    current_value: float
    max_allowed: float
    units: str

    @property
    def utilization(self) -> float:
        """Calculate limit utilization percentage."""
        return (self.current_value / self.max_allowed * 100) if self.max_allowed > 0 else 0

    @property
    def remaining(self) -> float:
        """Calculate remaining capacity."""
        return self.max_allowed - self.current_value

    @property
    def is_breached(self) -> bool:
        """Check if limit is breached."""
        return self.current_value >= self.max_allowed


class PositionLimitChecker:
    """Check and enforce position limits."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # Limit configurations
        self.limits = {
            "max_position_pct": config.get("max_position_pct", 0.10),  # 10% max per position
            "max_sector_pct": config.get("max_sector_pct", 0.30),  # 30% max per sector
            "max_correlation": config.get("max_correlation", 0.70),  # 70% max correlation
            "max_beta_exposure": config.get("max_beta_exposure", 1.5),  # 1.5x market beta
            "max_concentration": config.get("max_concentration", 0.50),  # 50% in top 5 positions
            "min_liquidity_ratio": config.get("min_liquidity_ratio", 0.01),  # 1% of ADV
            "max_volatility_units": config.get("max_volatility_units", 20),  # 20 vol units
        }

        # Sector mappings
        self.sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer",
            "TSLA": "Consumer",
            "WMT": "Consumer",
            "JPM": "Financial",
            "BAC": "Financial",
            "GS": "Financial",
            "XOM": "Energy",
            "CVX": "Energy",
            "COP": "Energy",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "UNH": "Healthcare",
        }

        # Market data cache
        self.cache = get_global_cache()
        self.correlation_matrix = None
        self.last_update = None

        self._lock = asyncio.Lock()

    async def check_pre_trade_limits(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        portfolio_value: float,
        current_positions: dict[str, Any],
    ) -> tuple[bool, list[PositionLimit]]:
        """Check all position limits before trade."""
        async with self._lock:
            limits_checked = []
            all_passed = True

            # 1. Position size limit
            position_limit = await self._check_position_size(
                symbol, quantity, price, portfolio_value
            )
            limits_checked.append(position_limit)
            if position_limit.is_breached:
                all_passed = False

            # 2. Sector exposure limit
            sector_limit = await self._check_sector_exposure(
                symbol, quantity, price, portfolio_value, current_positions
            )
            limits_checked.append(sector_limit)
            if sector_limit.is_breached:
                all_passed = False

            # 3. Correlation exposure
            correlation_limit = await self._check_correlation_exposure(symbol, current_positions)
            limits_checked.append(correlation_limit)
            if correlation_limit.is_breached:
                all_passed = False

            # 4. Beta exposure
            beta_limit = await self._check_beta_exposure(
                symbol, quantity, price, portfolio_value, current_positions
            )
            limits_checked.append(beta_limit)
            if beta_limit.is_breached:
                all_passed = False

            # 5. Concentration limit
            concentration_limit = await self._check_concentration(
                symbol, quantity, price, portfolio_value, current_positions
            )
            limits_checked.append(concentration_limit)
            if concentration_limit.is_breached:
                all_passed = False

            # 6. Liquidity check
            liquidity_limit = await self._check_liquidity(symbol, quantity)
            limits_checked.append(liquidity_limit)
            if liquidity_limit.is_breached:
                all_passed = False

            # 7. Volatility-adjusted size
            vol_limit = await self._check_volatility_adjusted_size(
                symbol, quantity, price, portfolio_value
            )
            limits_checked.append(vol_limit)
            if vol_limit.is_breached:
                all_passed = False

            # Log limit checks
            for limit in limits_checked:
                if limit.utilization > 80:
                    logger.warning(
                        f"Limit approaching: {limit.limit_type.value} at {limit.utilization:.1f}%"
                    )

            return all_passed, limits_checked

    async def _check_position_size(
        self, symbol: str, quantity: int, price: float, portfolio_value: float
    ) -> PositionLimit:
        """Check single position size limit."""
        position_value = quantity * price
        position_pct = position_value / portfolio_value

        return PositionLimit(
            limit_type=LimitType.POSITION_SIZE,
            limit_value=position_pct,
            current_value=position_pct,
            max_allowed=self.limits["max_position_pct"],
            units="portfolio %",
        )

    async def _check_sector_exposure(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: dict[str, Any],
    ) -> PositionLimit:
        """Check sector exposure limit."""
        sector = self.sector_map.get(symbol, "Other")

        # Calculate current sector exposure
        sector_value = 0
        for pos_symbol, position in current_positions.items():
            if self.sector_map.get(pos_symbol, "Other") == sector:
                sector_value += position.market_value

        # Add new position
        new_position_value = quantity * price
        total_sector_value = sector_value + new_position_value
        sector_pct = total_sector_value / portfolio_value

        return PositionLimit(
            limit_type=LimitType.SECTOR_EXPOSURE,
            limit_value=sector_pct,
            current_value=sector_pct,
            max_allowed=self.limits["max_sector_pct"],
            units=f"{sector} %",
        )

    async def _check_correlation_exposure(
        self, symbol: str, current_positions: dict[str, Any]
    ) -> PositionLimit:
        """Check correlation with existing positions."""
        if not current_positions:
            return PositionLimit(
                limit_type=LimitType.CORRELATION_EXPOSURE,
                limit_value=0,
                current_value=0,
                max_allowed=self.limits["max_correlation"],
                units="correlation",
            )

        # Get correlation data
        correlations = await self._get_correlations(symbol, list(current_positions.keys()))

        # Calculate weighted average correlation
        total_weight = 0
        weighted_corr = 0

        for pos_symbol, position in current_positions.items():
            if pos_symbol in correlations:
                weight = position.market_value
                weighted_corr += correlations[pos_symbol] * weight
                total_weight += weight

        avg_correlation = weighted_corr / total_weight if total_weight > 0 else 0

        return PositionLimit(
            limit_type=LimitType.CORRELATION_EXPOSURE,
            limit_value=abs(avg_correlation),
            current_value=abs(avg_correlation),
            max_allowed=self.limits["max_correlation"],
            units="avg correlation",
        )

    async def _check_beta_exposure(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: dict[str, Any],
    ) -> PositionLimit:
        """Check portfolio beta exposure."""
        # Get beta for new position
        symbol_beta = await self._get_beta(symbol)

        # Calculate current portfolio beta
        portfolio_beta = 0
        total_value = 0

        for pos_symbol, position in current_positions.items():
            pos_beta = await self._get_beta(pos_symbol)
            pos_value = position.market_value
            portfolio_beta += pos_beta * pos_value
            total_value += pos_value

        # Add new position
        new_value = quantity * price
        portfolio_beta += symbol_beta * new_value
        total_value += new_value

        weighted_beta = portfolio_beta / total_value if total_value > 0 else 0

        return PositionLimit(
            limit_type=LimitType.BETA_EXPOSURE,
            limit_value=weighted_beta,
            current_value=weighted_beta,
            max_allowed=self.limits["max_beta_exposure"],
            units="portfolio beta",
        )

    async def _check_concentration(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: dict[str, Any],
    ) -> PositionLimit:
        """Check position concentration (top 5 positions)."""
        # Get all position values
        position_values = []

        for position in current_positions.values():
            position_values.append(position.market_value)

        # Add new position
        new_value = quantity * price
        position_values.append(new_value)

        # Sort and get top 5
        position_values.sort(reverse=True)
        top_5_value = sum(position_values[:5])
        concentration = top_5_value / portfolio_value

        return PositionLimit(
            limit_type=LimitType.CONCENTRATION,
            limit_value=concentration,
            current_value=concentration,
            max_allowed=self.limits["max_concentration"],
            units="top 5 concentration",
        )

    async def _check_liquidity(self, symbol: str, quantity: int) -> PositionLimit:
        """Check position liquidity vs average daily volume."""
        # Get average daily volume
        adv = await self._get_average_daily_volume(symbol)

        if adv == 0:
            return PositionLimit(
                limit_type=LimitType.LIQUIDITY,
                limit_value=float("inf"),
                current_value=float("inf"),
                max_allowed=self.limits["min_liquidity_ratio"],
                units="% of ADV",
            )

        # Calculate position as percentage of ADV
        liquidity_ratio = quantity / adv

        return PositionLimit(
            limit_type=LimitType.LIQUIDITY,
            limit_value=liquidity_ratio,
            current_value=liquidity_ratio,
            max_allowed=self.limits["min_liquidity_ratio"],
            units="% of ADV",
        )

    async def _check_volatility_adjusted_size(
        self, symbol: str, quantity: int, price: float, portfolio_value: float
    ) -> PositionLimit:
        """Check volatility-adjusted position size."""
        # Get volatility
        volatility = await self._get_volatility(symbol)

        # Calculate volatility units
        position_value = quantity * price
        vol_units = (position_value / portfolio_value) / volatility if volatility > 0 else 0

        return PositionLimit(
            limit_type=LimitType.VOLATILITY_ADJUSTED,
            limit_value=vol_units,
            current_value=vol_units,
            max_allowed=self.limits["max_volatility_units"],
            units="volatility units",
        )

    async def _get_correlations(self, symbol: str, other_symbols: list[str]) -> dict[str, float]:
        """Get correlations between symbol and other symbols."""
        # In production, calculate from historical data
        # For now, return mock correlations
        correlations = {}

        for other in other_symbols:
            if symbol == other:
                correlations[other] = 1.0
            elif self.sector_map.get(symbol) == self.sector_map.get(other):
                correlations[other] = 0.7  # High correlation within sector
            else:
                correlations[other] = 0.3  # Lower correlation across sectors

        return correlations

    async def _get_beta(self, symbol: str) -> float:
        """Get stock beta."""
        # In production, calculate from historical data
        # For now, return sector-based estimates
        betas = {
            "Technology": 1.2,
            "Financial": 1.1,
            "Consumer": 0.9,
            "Energy": 1.3,
            "Healthcare": 0.8,
            "Other": 1.0,
        }

        sector = self.sector_map.get(symbol, "Other")
        return betas.get(sector, 1.0)

    async def _get_average_daily_volume(self, symbol: str) -> float:
        """Get average daily volume."""
        # In production, calculate from historical data
        # For now, return estimates
        adv_estimates = {
            "AAPL": 80_000_000,
            "MSFT": 25_000_000,
            "GOOGL": 20_000_000,
            "AMZN": 3_000_000,
            "TSLA": 70_000_000,
        }

        return adv_estimates.get(symbol, 10_000_000)

    async def _get_volatility(self, symbol: str) -> float:
        """Get annualized volatility."""
        # In production, calculate from historical data
        # For now, return estimates
        vol_estimates = {"AAPL": 0.25, "MSFT": 0.22, "GOOGL": 0.28, "AMZN": 0.30, "TSLA": 0.60}

        return vol_estimates.get(symbol, 0.30)

    async def update_market_data(self, market_data: dict[str, pd.DataFrame]):
        """Update market data for limit calculations."""
        async with self._lock:
            # Store market data in cache
            for symbol, data in market_data.items():
                cache_key = f"market_data:{symbol}"
                # Convert DataFrame to dict for caching
                data_dict = data.to_dict() if hasattr(data, "to_dict") else data
                await self.cache.set(CacheType.QUOTES, cache_key, data_dict, 300)  # 5 min TTL
            self.last_update = datetime.now()

            # Update correlation matrix
            if market_data:
                returns_data = {}
                for symbol, df in market_data.items():
                    if "close" in df.columns:
                        returns_data[symbol] = df["close"].pct_change().dropna()

                if returns_data:
                    returns_df = pd.DataFrame(returns_data)
                    self.correlation_matrix = returns_df.corr()

    def adjust_position_size(self, proposed_quantity: int, limits: list[PositionLimit]) -> int:
        """Adjust position size to comply with limits."""
        if all(not limit.is_breached for limit in limits):
            return proposed_quantity

        # Find the most restrictive limit
        adjustment_factors = []

        for limit in limits:
            if limit.is_breached and limit.max_allowed > 0:
                # Calculate how much we need to reduce
                reduction_factor = limit.max_allowed / limit.current_value
                adjustment_factors.append(reduction_factor)

        if adjustment_factors:
            # Use the most restrictive factor
            min_factor = min(adjustment_factors)
            adjusted_quantity = int(
                proposed_quantity * min_factor * 0.95
            )  # 95% to ensure compliance

            logger.info(f"Position size adjusted from {proposed_quantity} to {adjusted_quantity}")
            return max(0, adjusted_quantity)

        return proposed_quantity

    async def get_limit_summary(
        self, portfolio_value: float, current_positions: dict[str, Any]
    ) -> dict[str, Any]:
        """Get summary of all current limit utilizations."""
        summary = {"timestamp": datetime.now(), "limits": {}, "warnings": [], "breaches": []}

        # Check each type of limit
        # Position concentration
        if current_positions:
            position_values = [pos.market_value for pos in current_positions.values()]
            top_5_value = sum(sorted(position_values, reverse=True)[:5])
            concentration = top_5_value / portfolio_value

            summary["limits"]["concentration"] = {
                "current": concentration,
                "limit": self.limits["max_concentration"],
                "utilization": concentration / self.limits["max_concentration"] * 100,
            }

            if concentration > self.limits["max_concentration"] * 0.8:
                summary["warnings"].append(f"High concentration: {concentration:.1%}")

        # Sector exposures
        sector_exposures = {}
        for symbol, position in current_positions.items():
            sector = self.sector_map.get(symbol, "Other")
            sector_exposures[sector] = (
                sector_exposures.get(sector, 0) + position.market_value / portfolio_value
            )

        for sector, exposure in sector_exposures.items():
            summary["limits"][f"sector_{sector}"] = {
                "current": exposure,
                "limit": self.limits["max_sector_pct"],
                "utilization": exposure / self.limits["max_sector_pct"] * 100,
            }

            if exposure > self.limits["max_sector_pct"] * 0.8:
                summary["warnings"].append(f"High {sector} exposure: {exposure:.1%}")

        return summary

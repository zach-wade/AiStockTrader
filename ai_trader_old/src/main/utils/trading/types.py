"""
Trading Universe Types

Data structures and enums for trading universe management.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Third-party imports
import pandas as pd


class UniverseType(Enum):
    """Types of trading universes."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    SCREENED = "screened"
    INDEX_BASED = "index_based"
    SECTOR_BASED = "sector_based"
    MARKET_CAP_BASED = "market_cap_based"
    CUSTOM = "custom"


class FilterCriteria(Enum):
    """Filter criteria for universe construction."""

    MARKET_CAP = "market_cap"
    VOLUME = "volume"
    PRICE = "price"
    SECTOR = "sector"
    INDUSTRY = "industry"
    EXCHANGE = "exchange"
    COUNTRY = "country"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    BETA = "beta"
    DIVIDEND_YIELD = "dividend_yield"
    PE_RATIO = "pe_ratio"
    CUSTOM = "custom"


@dataclass
class Filter:
    """Universe filter definition."""

    criteria: FilterCriteria
    operator: str  # 'gt', 'lt', 'eq', 'ne', 'in', 'not_in', 'between'
    value: float | int | str | list[Any]
    weight: float = 1.0
    enabled: bool = True

    def apply(self, data: pd.Series) -> pd.Series:
        """Apply filter to data series."""
        if not self.enabled:
            return pd.Series(True, index=data.index)

        if self.operator == "gt":
            return data > self.value
        elif self.operator == "lt":
            return data < self.value
        elif self.operator == "eq":
            return data == self.value
        elif self.operator == "ne":
            return data != self.value
        elif self.operator == "in":
            return data.isin(self.value)
        elif self.operator == "not_in":
            return ~data.isin(self.value)
        elif self.operator == "between":
            return (data >= self.value[0]) & (data <= self.value[1])
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class UniverseConfig:
    """Configuration for universe construction."""

    name: str
    universe_type: UniverseType
    filters: list[Filter] = field(default_factory=list)
    max_symbols: int | None = None
    min_symbols: int | None = None
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    ranking_criteria: str | None = None
    ranking_ascending: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "universe_type": self.universe_type.value,
            "filters": [
                {
                    "criteria": f.criteria.value,
                    "operator": f.operator,
                    "value": f.value,
                    "weight": f.weight,
                    "enabled": f.enabled,
                }
                for f in self.filters
            ],
            "max_symbols": self.max_symbols,
            "min_symbols": self.min_symbols,
            "rebalance_frequency": self.rebalance_frequency,
            "ranking_criteria": self.ranking_criteria,
            "ranking_ascending": self.ranking_ascending,
        }


@dataclass
class UniverseSnapshot:
    """Snapshot of a universe at a point in time."""

    timestamp: datetime
    symbols: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbols": list(self.symbols),
            "symbol_count": len(self.symbols),
            "metadata": self.metadata,
        }

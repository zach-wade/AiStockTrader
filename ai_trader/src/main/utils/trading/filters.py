"""
Trading Universe Filters

Utility functions for creating and managing trading universe filters.
"""

from typing import List, Optional
from .types import Filter, FilterCriteria


def create_market_cap_filter(min_cap: float, max_cap: Optional[float] = None) -> Filter:
    """Create a market cap filter."""
    if max_cap is None:
        return Filter(FilterCriteria.MARKET_CAP, 'gt', min_cap)
    else:
        return Filter(FilterCriteria.MARKET_CAP, 'between', [min_cap, max_cap])


def create_volume_filter(min_volume: float) -> Filter:
    """Create a volume filter."""
    return Filter(FilterCriteria.VOLUME, 'gt', min_volume)


def create_sector_filter(sectors: List[str], exclude: bool = False) -> Filter:
    """Create a sector filter."""
    operator = 'not_in' if exclude else 'in'
    return Filter(FilterCriteria.SECTOR, operator, sectors)


def create_exchange_filter(exchanges: List[str]) -> Filter:
    """Create an exchange filter."""
    return Filter(FilterCriteria.EXCHANGE, 'in', exchanges)


def create_price_range_filter(min_price: float, max_price: float) -> Filter:
    """Create a price range filter."""
    return Filter(FilterCriteria.PRICE, 'between', [min_price, max_price])


def create_liquidity_filter(min_liquidity: float) -> Filter:
    """Create a liquidity filter."""
    return Filter(FilterCriteria.LIQUIDITY, 'gt', min_liquidity)


def create_volatility_filter(min_volatility: float, max_volatility: Optional[float] = None) -> Filter:
    """Create a volatility filter."""
    if max_volatility is None:
        return Filter(FilterCriteria.VOLATILITY, 'gt', min_volatility)
    else:
        return Filter(FilterCriteria.VOLATILITY, 'between', [min_volatility, max_volatility])


def create_beta_filter(min_beta: float, max_beta: Optional[float] = None) -> Filter:
    """Create a beta filter."""
    if max_beta is None:
        return Filter(FilterCriteria.BETA, 'gt', min_beta)
    else:
        return Filter(FilterCriteria.BETA, 'between', [min_beta, max_beta])


def create_dividend_yield_filter(min_yield: float, max_yield: Optional[float] = None) -> Filter:
    """Create a dividend yield filter."""
    if max_yield is None:
        return Filter(FilterCriteria.DIVIDEND_YIELD, 'gt', min_yield)
    else:
        return Filter(FilterCriteria.DIVIDEND_YIELD, 'between', [min_yield, max_yield])


def create_pe_ratio_filter(min_pe: float, max_pe: Optional[float] = None) -> Filter:
    """Create a PE ratio filter."""
    if max_pe is None:
        return Filter(FilterCriteria.PE_RATIO, 'gt', min_pe)
    else:
        return Filter(FilterCriteria.PE_RATIO, 'between', [min_pe, max_pe])


def create_industry_filter(industries: List[str], exclude: bool = False) -> Filter:
    """Create an industry filter."""
    operator = 'not_in' if exclude else 'in'
    return Filter(FilterCriteria.INDUSTRY, operator, industries)


def create_country_filter(countries: List[str], exclude: bool = False) -> Filter:
    """Create a country filter."""
    operator = 'not_in' if exclude else 'in'
    return Filter(FilterCriteria.COUNTRY, operator, countries)


# Common filter presets
def create_large_cap_filters() -> List[Filter]:
    """Create filters for large cap stocks."""
    return [
        create_market_cap_filter(10_000_000_000),  # > $10B
        create_volume_filter(1_000_000),  # > 1M volume
        create_price_range_filter(5.0, 1000.0)  # $5 - $1000
    ]


def create_high_volume_filters() -> List[Filter]:
    """Create filters for high volume stocks."""
    return [
        create_volume_filter(5_000_000),  # > 5M volume
        create_price_range_filter(1.0, 1000.0)  # $1 - $1000
    ]


def create_growth_filters() -> List[Filter]:
    """Create filters for growth stocks."""
    return [
        create_pe_ratio_filter(15.0, 50.0),  # PE between 15-50
        create_market_cap_filter(1_000_000_000),  # > $1B
        create_volume_filter(500_000)  # > 500K volume
    ]


def create_dividend_filters() -> List[Filter]:
    """Create filters for dividend stocks."""
    return [
        create_dividend_yield_filter(2.0, 10.0),  # Yield between 2-10%
        create_market_cap_filter(1_000_000_000),  # > $1B
        create_pe_ratio_filter(5.0, 25.0)  # PE between 5-25
    ]


def create_value_filters() -> List[Filter]:
    """Create filters for value stocks."""
    return [
        create_pe_ratio_filter(5.0, 20.0),  # PE between 5-20
        create_market_cap_filter(500_000_000),  # > $500M
        create_volume_filter(100_000)  # > 100K volume
    ]
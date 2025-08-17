# File: backtesting/engine/cost_model.py

"""
Trading Cost Model for Backtesting Engine.

Provides comprehensive cost modeling including:
- Commission fees (fixed, percentage, tiered)
- Slippage estimation
- Market impact modeling
- Borrowing costs for shorts
- Exchange fees
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

# Local imports
from main.models.common import OrderSide, OrderType
from main.utils.core import ErrorHandlingMixin, get_logger

logger = get_logger(__name__)


class CostType(Enum):
    """Types of trading costs."""

    COMMISSION = "commission"
    SLIPPAGE = "slippage"
    MARKET_IMPACT = "market_impact"
    BORROWING = "borrowing"
    EXCHANGE_FEE = "exchange_fee"
    REGULATORY_FEE = "regulatory_fee"


@dataclass
class CostComponents:
    """Breakdown of trading costs."""

    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    borrowing_cost: float = 0.0
    exchange_fee: float = 0.0
    regulatory_fee: float = 0.0

    @property
    def total_cost(self) -> float:
        """Calculate total trading cost."""
        return (
            self.commission
            + self.slippage
            + self.market_impact
            + self.borrowing_cost
            + self.exchange_fee
            + self.regulatory_fee
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "commission": self.commission,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "borrowing_cost": self.borrowing_cost,
            "exchange_fee": self.exchange_fee,
            "regulatory_fee": self.regulatory_fee,
            "total": self.total_cost,
        }


class CommissionModel:
    """Base commission model."""

    def calculate(self, quantity: int, price: float, order_side: OrderSide) -> float:
        """
        Calculate commission for a trade.

        This is an abstract method that must be implemented by subclasses.

        Args:
            quantity: Number of shares
            price: Price per share
            order_side: Buy or sell

        Returns:
            Commission amount in dollars
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class FixedCommission(CommissionModel):
    """Fixed commission per trade."""

    def __init__(self, commission_per_trade: float = 1.0):
        self.commission_per_trade = commission_per_trade

    def calculate(self, quantity: int, price: float, order_side: OrderSide) -> float:
        """Fixed commission regardless of trade size."""
        return self.commission_per_trade


class PercentageCommission(CommissionModel):
    """Percentage-based commission."""

    def __init__(self, rate: float = 0.001, min_commission: float = 1.0):
        self.rate = rate
        self.min_commission = min_commission

    def calculate(self, quantity: int, price: float, order_side: OrderSide) -> float:
        """Commission as percentage of trade value."""
        trade_value = quantity * price
        commission = trade_value * self.rate
        return max(commission, self.min_commission)


class TieredCommission(CommissionModel):
    """Tiered commission based on trade size."""

    def __init__(self, tiers: dict[int, float], min_commission: float = 1.0):
        """
        Args:
            tiers: Dict mapping quantity thresholds to commission rates
                   e.g., {0: 0.01, 1000: 0.005, 10000: 0.001}
        """
        self.tiers = sorted(tiers.items(), reverse=True)
        self.min_commission = min_commission

    def calculate(self, quantity: int, price: float, order_side: OrderSide) -> float:
        """Commission based on quantity tier."""
        for threshold, rate in self.tiers:
            if quantity >= threshold:
                commission = quantity * rate
                return max(commission, self.min_commission)

        # Default to highest rate if no tier matched
        commission = quantity * self.tiers[-1][1]
        return max(commission, self.min_commission)


class SlippageModel:
    """Base slippage model."""

    def calculate(
        self, quantity: int, price: float, order_side: OrderSide, spread: float, volatility: float
    ) -> float:
        """
        Calculate slippage cost.

        This is an abstract method that must be implemented by subclasses.

        Args:
            quantity: Number of shares
            price: Price per share
            order_side: Buy or sell
            spread: Bid-ask spread
            volatility: Market volatility

        Returns:
            Slippage cost in dollars
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class FixedSlippage(SlippageModel):
    """Fixed slippage in basis points."""

    def __init__(self, slippage_bps: float = 5.0):
        self.slippage_bps = slippage_bps

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Fixed slippage as percentage of trade value."""
        trade_value = quantity * price
        return trade_value * (self.slippage_bps / 10000)


class SpreadSlippage(SlippageModel):
    """Slippage based on bid-ask spread."""

    def __init__(self, spread_fraction: float = 0.5):
        """
        Args:
            spread_fraction: Fraction of spread paid (0.5 = half spread)
        """
        self.spread_fraction = spread_fraction

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Slippage as fraction of spread."""
        trade_value = quantity * price
        spread_cost = spread * self.spread_fraction
        return trade_value * spread_cost


class VolatilitySlippage(SlippageModel):
    """Slippage based on volatility and order size."""

    def __init__(self, impact_coefficient: float = 0.1):
        self.impact_coefficient = impact_coefficient

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Slippage increases with volatility and order size."""
        trade_value = quantity * price

        # Basic volatility-based slippage
        vol_slippage = volatility * self.impact_coefficient

        # Add spread component
        spread_component = spread * 0.5

        total_slippage = vol_slippage + spread_component
        return trade_value * total_slippage


class LinearSlippage(SlippageModel):
    """Linear slippage model based on order size."""

    def __init__(self, size_coefficient: float = 0.00001):
        """
        Args:
            size_coefficient: Slippage per share as fraction of price
        """
        self.size_coefficient = size_coefficient

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Linear slippage proportional to order size."""
        # Slippage increases linearly with quantity
        slippage_pct = self.size_coefficient * quantity
        trade_value = quantity * price
        return trade_value * slippage_pct


class SquareRootSlippage(SlippageModel):
    """Square root slippage model (Almgren-Chriss style)."""

    def __init__(self, impact_coefficient: float = 0.0001, daily_volume: int = 1000000):
        """
        Args:
            impact_coefficient: Market impact coefficient
            daily_volume: Assumed average daily volume
        """
        self.impact_coefficient = impact_coefficient
        self.daily_volume = daily_volume

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Square root slippage based on participation rate."""
        # Calculate participation rate
        participation_rate = quantity / self.daily_volume

        # Square root impact
        slippage_pct = self.impact_coefficient * math.sqrt(participation_rate) * volatility

        # Add spread component
        spread_cost = spread * 0.5

        total_slippage = slippage_pct + spread_cost
        trade_value = quantity * price
        return trade_value * total_slippage


class AdaptiveSlippage(SlippageModel):
    """Adaptive slippage model that combines multiple factors."""

    def __init__(
        self,
        base_spread_fraction: float = 0.5,
        volatility_multiplier: float = 2.0,
        size_penalty: float = 0.00001,
        urgency_factor: float = 1.0,
    ):
        """
        Args:
            base_spread_fraction: Base fraction of spread to pay
            volatility_multiplier: How much volatility increases slippage
            size_penalty: Additional slippage per share
            urgency_factor: Multiplier for urgent orders (1.0 = normal, 2.0 = urgent)
        """
        self.base_spread_fraction = base_spread_fraction
        self.volatility_multiplier = volatility_multiplier
        self.size_penalty = size_penalty
        self.urgency_factor = urgency_factor

    def calculate(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        spread: float = 0.01,
        volatility: float = 0.02,
    ) -> float:
        """Adaptive slippage considering market conditions."""
        trade_value = quantity * price

        # Base spread cost
        spread_cost = spread * self.base_spread_fraction

        # Volatility adjustment
        vol_adjustment = volatility * self.volatility_multiplier

        # Size penalty
        size_impact = self.size_penalty * quantity

        # Combine all factors
        total_slippage = (spread_cost + vol_adjustment + size_impact) * self.urgency_factor

        return trade_value * total_slippage


class MarketImpactModel:
    """Model for permanent market impact."""

    def __init__(
        self,
        linear_coefficient: float = 0.0001,
        square_root_coefficient: float = 0.001,
        daily_volume_fraction: float = 0.01,
    ):
        """
        Args:
            linear_coefficient: Linear impact factor
            square_root_coefficient: Square root impact factor
            daily_volume_fraction: Assumed fraction of daily volume
        """
        self.linear_coefficient = linear_coefficient
        self.square_root_coefficient = square_root_coefficient
        self.daily_volume_fraction = daily_volume_fraction

    def calculate(
        self, quantity: int, price: float, order_side: OrderSide, avg_daily_volume: int = 1000000
    ) -> float:
        """Calculate permanent market impact."""
        trade_value = quantity * price

        # Participation rate (fraction of daily volume)
        participation_rate = quantity / avg_daily_volume

        # Linear impact
        linear_impact = self.linear_coefficient * participation_rate

        # Square root impact (Almgren model)
        sqrt_impact = self.square_root_coefficient * math.sqrt(participation_rate)

        # Total impact
        total_impact = linear_impact + sqrt_impact

        # Direction matters - buying pushes price up, selling down
        if order_side == OrderSide.BUY:
            impact_cost = trade_value * total_impact
        else:
            impact_cost = trade_value * total_impact * 0.5  # Selling has less impact

        return impact_cost


class CostModel(ErrorHandlingMixin):
    """
    Comprehensive cost model for backtesting.

    Combines multiple cost components to provide realistic trading costs.
    """

    def __init__(
        self,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        market_impact_model: MarketImpactModel | None = None,
        borrowing_rate: float = 0.02,  # 2% annual
        exchange_fee_rate: float = 0.00005,  # 0.5 bps
        regulatory_fee_rate: float = 0.00002,
    ):  # 0.2 bps
        """
        Initialize cost model with component models.

        Args:
            commission_model: Model for commission calculation
            slippage_model: Model for slippage calculation
            market_impact_model: Model for market impact
            borrowing_rate: Annual rate for short positions
            exchange_fee_rate: Exchange fee as fraction of trade value
            regulatory_fee_rate: Regulatory fee as fraction of trade value
        """
        self.commission_model = commission_model or PercentageCommission()
        self.slippage_model = slippage_model or FixedSlippage()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.borrowing_rate = borrowing_rate
        self.exchange_fee_rate = exchange_fee_rate
        self.regulatory_fee_rate = regulatory_fee_rate

        # Cost calculation statistics
        self.total_costs = CostComponents()
        self.trade_count = 0

        logger.info("CostModel initialized")

    def calculate_trade_cost(
        self,
        quantity: int,
        price: float,
        order_side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        spread: float = 0.01,
        volatility: float = 0.02,
        avg_daily_volume: int = 1000000,
        holding_days: int = 0,
    ) -> CostComponents:
        """
        Calculate all cost components for a trade.

        Args:
            quantity: Number of shares
            price: Execution price
            order_side: Buy or sell
            order_type: Market or limit order
            spread: Bid-ask spread as fraction
            volatility: Asset volatility
            avg_daily_volume: Average daily volume
            holding_days: Days position held (for borrowing costs)

        Returns:
            CostComponents with breakdown of all costs
        """
        costs = CostComponents()
        trade_value = quantity * price

        # Commission
        costs.commission = self.commission_model.calculate(quantity, price, order_side)

        # Slippage (market orders only)
        if order_type == OrderType.MARKET:
            costs.slippage = self.slippage_model.calculate(
                quantity, price, order_side, spread, volatility
            )

        # Market impact
        costs.market_impact = self.market_impact_model.calculate(
            quantity, price, order_side, avg_daily_volume
        )

        # Borrowing costs (short positions only)
        if order_side == OrderSide.SELL and holding_days > 0:
            daily_rate = self.borrowing_rate / 365
            costs.borrowing_cost = trade_value * daily_rate * holding_days

        # Exchange fees
        costs.exchange_fee = trade_value * self.exchange_fee_rate

        # Regulatory fees
        costs.regulatory_fee = trade_value * self.regulatory_fee_rate

        # Update statistics
        self._update_statistics(costs)

        logger.debug(f"Trade cost calculated: {costs.to_dict()}")

        return costs

    def _update_statistics(self, costs: CostComponents):
        """Update cumulative cost statistics."""
        self.total_costs.commission += costs.commission
        self.total_costs.slippage += costs.slippage
        self.total_costs.market_impact += costs.market_impact
        self.total_costs.borrowing_cost += costs.borrowing_cost
        self.total_costs.exchange_fee += costs.exchange_fee
        self.total_costs.regulatory_fee += costs.regulatory_fee
        self.trade_count += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get cost model statistics."""
        avg_cost = self.total_costs.total_cost / self.trade_count if self.trade_count > 0 else 0

        return {
            "total_costs": self.total_costs.to_dict(),
            "trade_count": self.trade_count,
            "average_cost_per_trade": avg_cost,
            "cost_breakdown": {
                "commission_pct": (
                    self.total_costs.commission / self.total_costs.total_cost * 100
                    if self.total_costs.total_cost > 0
                    else 0
                ),
                "slippage_pct": (
                    self.total_costs.slippage / self.total_costs.total_cost * 100
                    if self.total_costs.total_cost > 0
                    else 0
                ),
                "market_impact_pct": (
                    self.total_costs.market_impact / self.total_costs.total_cost * 100
                    if self.total_costs.total_cost > 0
                    else 0
                ),
            },
        }

    def reset_statistics(self):
        """Reset cost statistics."""
        self.total_costs = CostComponents()
        self.trade_count = 0
        logger.info("Cost model statistics reset")


def create_default_cost_model() -> CostModel:
    """Create cost model with default settings."""
    return CostModel(
        commission_model=PercentageCommission(rate=0.001, min_commission=1.0),
        slippage_model=SpreadSlippage(spread_fraction=0.5),
        market_impact_model=MarketImpactModel(),
    )


def create_zero_cost_model() -> CostModel:
    """Create cost model with zero costs (for testing)."""
    return CostModel(
        commission_model=FixedCommission(0),
        slippage_model=FixedSlippage(0),
        market_impact_model=MarketImpactModel(0, 0, 0),
        borrowing_rate=0,
        exchange_fee_rate=0,
        regulatory_fee_rate=0,
    )


# Broker-Specific Commission Structures


def create_interactive_brokers_cost_model() -> CostModel:
    """Create cost model for Interactive Brokers pricing."""
    # IB uses tiered pricing based on volume
    ib_tiers = {
        0: 0.0035,  # $0.0035 per share for first 300,000 shares
        300000: 0.002,  # $0.002 per share for next 3,700,000 shares
        4000000: 0.0015,  # $0.0015 per share for next 20,000,000 shares
        24000000: 0.001,  # $0.001 per share above 24,000,000 shares
    }

    return CostModel(
        commission_model=TieredCommission(tiers=ib_tiers, min_commission=0.35),
        slippage_model=SpreadSlippage(spread_fraction=0.5),
        market_impact_model=MarketImpactModel(),
        borrowing_rate=0.0159,  # 1.59% for tier 1
        exchange_fee_rate=0.00003,  # Typical exchange fees
        regulatory_fee_rate=0.0000229,  # SEC fee
    )


def create_td_ameritrade_cost_model() -> CostModel:
    """Create cost model for TD Ameritrade (now Schwab) pricing."""
    return CostModel(
        commission_model=FixedCommission(commission_per_trade=0.0),  # Zero commission
        slippage_model=SpreadSlippage(spread_fraction=0.5),
        market_impact_model=MarketImpactModel(),
        borrowing_rate=0.0975,  # 9.75% margin rate
        exchange_fee_rate=0.00003,
        regulatory_fee_rate=0.0000229,
    )


def create_robinhood_cost_model() -> CostModel:
    """Create cost model for Robinhood pricing."""
    return CostModel(
        commission_model=FixedCommission(commission_per_trade=0.0),  # Zero commission
        slippage_model=FixedSlippage(slippage_bps=10.0),  # Higher slippage due to PFOF
        market_impact_model=MarketImpactModel(),
        borrowing_rate=0.12,  # 12% margin rate
        exchange_fee_rate=0.00003,
        regulatory_fee_rate=0.0000229,
    )


def create_alpaca_cost_model() -> CostModel:
    """Create cost model for Alpaca pricing."""
    return CostModel(
        commission_model=FixedCommission(commission_per_trade=0.0),  # Zero commission
        slippage_model=SpreadSlippage(spread_fraction=0.5),
        market_impact_model=MarketImpactModel(),
        borrowing_rate=0.055,  # 5.5% margin rate
        exchange_fee_rate=0.00003,
        regulatory_fee_rate=0.0000229,
    )


def create_institutional_cost_model() -> CostModel:
    """Create cost model for institutional trading."""
    # Institutional traders get better rates but pay commissions
    inst_tiers = {
        0: 0.001,  # $0.001 per share for first 1M shares
        1000000: 0.0008,  # $0.0008 per share for next 9M shares
        10000000: 0.0005,  # $0.0005 per share above 10M shares
    }

    return CostModel(
        commission_model=TieredCommission(tiers=inst_tiers, min_commission=0.5),
        slippage_model=AdaptiveSlippage(
            base_spread_fraction=0.3,  # Better spreads
            volatility_multiplier=1.5,
            size_penalty=0.000005,
            urgency_factor=1.0,
        ),
        market_impact_model=MarketImpactModel(
            linear_coefficient=0.00005,  # Lower impact
            square_root_coefficient=0.0005,
            daily_volume_fraction=0.02,
        ),
        borrowing_rate=0.005,  # 0.5% institutional rate
        exchange_fee_rate=0.00002,  # Lower exchange fees
        regulatory_fee_rate=0.0000229,
    )


def create_high_frequency_cost_model() -> CostModel:
    """Create cost model for high-frequency trading."""
    return CostModel(
        commission_model=PercentageCommission(rate=0.00002, min_commission=0.01),  # Very low rates
        slippage_model=LinearSlippage(size_coefficient=0.000001),  # Minimal slippage
        market_impact_model=MarketImpactModel(
            linear_coefficient=0.00001,
            square_root_coefficient=0.0001,
            daily_volume_fraction=0.001,  # Very small participation
        ),
        borrowing_rate=0.003,  # Prime rate
        exchange_fee_rate=0.00001,  # Rebates possible
        regulatory_fee_rate=0.0000229,
    )


def get_broker_cost_model(broker_name: str) -> CostModel:
    """Get cost model for a specific broker."""
    broker_models = {
        "interactive_brokers": create_interactive_brokers_cost_model,
        "ib": create_interactive_brokers_cost_model,
        "td_ameritrade": create_td_ameritrade_cost_model,
        "schwab": create_td_ameritrade_cost_model,
        "robinhood": create_robinhood_cost_model,
        "alpaca": create_alpaca_cost_model,
        "institutional": create_institutional_cost_model,
        "hft": create_high_frequency_cost_model,
        "default": create_default_cost_model,
        "zero": create_zero_cost_model,
    }

    creator = broker_models.get(broker_name.lower(), create_default_cost_model)
    return creator()

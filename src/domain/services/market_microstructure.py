"""
Market Microstructure - Domain service for market impact and slippage modeling
"""

import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from ..entities.order import OrderSide, OrderType
from ..value_objects.price import Price
from ..value_objects.quantity import Quantity


class MarketImpactModel(Enum):
    """Types of market impact models"""

    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    FIXED = "fixed"


@dataclass
class SlippageConfig:
    """Configuration for slippage calculation"""

    base_bid_ask_bps: Decimal  # Base bid-ask difference in basis points
    impact_coefficient: Decimal  # Market impact coefficient
    volatility_multiplier: Decimal = Decimal("1.0")
    add_randomness: bool = True
    random_factor_range: tuple[Decimal, Decimal] = (Decimal("0.8"), Decimal("1.2"))

    def __post_init__(self) -> None:
        """Validate slippage configuration"""
        if self.base_bid_ask_bps < 0:
            raise ValueError("Base bid-ask difference cannot be negative")
        if self.impact_coefficient < 0:
            raise ValueError("Impact coefficient cannot be negative")
        if self.volatility_multiplier < 0:
            raise ValueError("Volatility multiplier cannot be negative")


class IMarketMicrostructure(ABC):
    """Interface for market microstructure models"""

    @abstractmethod
    def calculate_execution_price(
        self, base_price: Price, side: OrderSide, quantity: Quantity, order_type: OrderType
    ) -> Price:
        """Calculate expected execution price with slippage"""
        pass

    @abstractmethod
    def calculate_market_impact(
        self, price: Price, quantity: Quantity, average_volume: Quantity | None = None
    ) -> Decimal:
        """Calculate market impact in basis points"""
        pass


class LinearImpactModel(IMarketMicrostructure):
    """Linear market impact model"""

    def __init__(self, config: SlippageConfig) -> None:
        self.config = config
        self._rng = secrets.SystemRandom()

    def calculate_execution_price(
        self, base_price: Price, side: OrderSide, quantity: Quantity, order_type: OrderType
    ) -> Price:
        """Calculate execution price with linear impact"""
        # Market orders get full slippage
        if order_type != OrderType.MARKET:
            return base_price

        # Calculate base bid-ask difference
        bid_ask_diff = base_price.value * self.config.base_bid_ask_bps / Decimal("10000")

        # Calculate market impact
        impact = self.calculate_market_impact(base_price, quantity)
        impact_amount = base_price.value * impact / Decimal("10000")

        # Add randomness if configured
        if self.config.add_randomness:
            random_factor = self._get_random_factor()
            bid_ask_diff *= random_factor
            impact_amount *= random_factor

        # Apply slippage based on side
        if side == OrderSide.BUY:
            # Buyers pay more (cross the bid-ask + impact)
            execution_price = base_price.value + bid_ask_diff + impact_amount
        else:
            # Sellers receive less
            execution_price = base_price.value - bid_ask_diff - impact_amount

        return Price(max(execution_price, Decimal("0.01")))

    def calculate_market_impact(
        self, price: Price, quantity: Quantity, average_volume: Quantity | None = None
    ) -> Decimal:
        """Calculate linear market impact in basis points"""
        # Simple linear impact based on quantity
        # In reality, this would consider average daily volume
        base_impact = self.config.impact_coefficient * abs(quantity.value) / Decimal("1000")

        # Scale by volatility
        scaled_impact = base_impact * self.config.volatility_multiplier

        return scaled_impact

    def _get_random_factor(self) -> Decimal:
        """Generate cryptographically secure random factor for realistic simulation.

        Creates a random multiplier within the configured range to simulate
        natural variation in execution quality. Uses cryptographically secure
        randomness for better statistical properties.

        Returns:
            Decimal: Random factor between min and max range (e.g., 0.8-1.2).

        Note:
            Using secrets.SystemRandom ensures high-quality randomness suitable
            for financial simulations.
        """
        min_factor, max_factor = self.config.random_factor_range
        random_value = self._rng.random()  # Cryptographically secure random value
        return min_factor + (max_factor - min_factor) * Decimal(str(random_value))


class SquareRootImpactModel(IMarketMicrostructure):
    """Square-root market impact model (more realistic for large orders)"""

    def __init__(self, config: SlippageConfig) -> None:
        self.config = config
        self._rng = secrets.SystemRandom()

    def calculate_execution_price(
        self, base_price: Price, side: OrderSide, quantity: Quantity, order_type: OrderType
    ) -> Price:
        """Calculate execution price with square-root impact"""
        # Market orders get full slippage
        if order_type != OrderType.MARKET:
            return base_price

        # Calculate base bid-ask difference
        bid_ask_diff = base_price.value * self.config.base_bid_ask_bps / Decimal("10000")

        # Calculate market impact
        impact = self.calculate_market_impact(base_price, quantity)
        impact_amount = base_price.value * impact / Decimal("10000")

        # Add randomness if configured
        if self.config.add_randomness:
            random_factor = self._get_random_factor()
            bid_ask_diff *= random_factor
            impact_amount *= random_factor

        # Apply slippage based on side
        if side == OrderSide.BUY:
            execution_price = base_price.value + bid_ask_diff + impact_amount
        else:
            execution_price = base_price.value - bid_ask_diff - impact_amount

        return Price(max(execution_price, Decimal("0.01")))

    def calculate_market_impact(
        self, price: Price, quantity: Quantity, average_volume: Quantity | None = None
    ) -> Decimal:
        """Calculate square-root market impact in basis points"""
        # Square-root impact (Almgren-Chriss model simplified)
        # For large orders, square root grows slower than linear
        quantity_factor = abs(quantity.value) ** Decimal("0.5")
        base_impact = self.config.impact_coefficient * quantity_factor / Decimal("100")

        # Scale by volatility
        scaled_impact = base_impact * self.config.volatility_multiplier

        return scaled_impact

    def _get_random_factor(self) -> Decimal:
        """Generate cryptographically secure random factor for realistic simulation.

        Creates a random multiplier within the configured range to simulate
        natural variation in execution quality. Uses cryptographically secure
        randomness for better statistical properties.

        Returns:
            Decimal: Random factor between min and max range (e.g., 0.8-1.2).

        Note:
            Using secrets.SystemRandom ensures high-quality randomness suitable
            for financial simulations.
        """
        min_factor, max_factor = self.config.random_factor_range
        random_value = self._rng.random()  # Cryptographically secure random value
        return min_factor + (max_factor - min_factor) * Decimal(str(random_value))


class MarketMicrostructureFactory:
    """Factory for creating market microstructure models"""

    @staticmethod
    def create(model_type: MarketImpactModel, config: SlippageConfig) -> IMarketMicrostructure:
        """Create appropriate model based on type"""
        if model_type == MarketImpactModel.LINEAR:
            return LinearImpactModel(config)
        elif model_type == MarketImpactModel.SQUARE_ROOT:
            return SquareRootImpactModel(config)
        else:
            raise ValueError(f"Unsupported market impact model: {model_type}")

    @staticmethod
    def create_default(model_type: MarketImpactModel) -> IMarketMicrostructure:
        """Create model with default configuration"""
        return MarketMicrostructureFactory.create(
            model_type=model_type, config=DEFAULT_EQUITY_CONFIG
        )


# Default configurations
DEFAULT_EQUITY_CONFIG = SlippageConfig(
    base_bid_ask_bps=Decimal("2"),  # 2 basis points
    impact_coefficient=Decimal("0.1"),
    volatility_multiplier=Decimal("1.0"),
)

DEFAULT_FOREX_CONFIG = SlippageConfig(
    base_bid_ask_bps=Decimal("0.5"),  # Tighter bid-ask in FX
    impact_coefficient=Decimal("0.05"),
    volatility_multiplier=Decimal("0.5"),
)

"""
Entry price determination for outcome classification.

This module determines accurate entry prices for trades based on:
- Market microstructure analysis
- Execution timing
- Spread considerations
- Volume-weighted pricing
"""

# Standard library imports
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger, timer
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


class PricingMethod(Enum):
    """Entry price determination methods."""

    CLOSE = "close"  # Use close price
    OPEN = "open"  # Use open price
    MIDPOINT = "midpoint"  # Bid-ask midpoint
    VWAP = "vwap"  # Volume-weighted average price
    TWAP = "twap"  # Time-weighted average price
    ARRIVAL = "arrival"  # Arrival price with market impact
    ADAPTIVE = "adaptive"  # Adaptive based on conditions


@dataclass
class PriceContext:
    """Market context for price determination."""

    timestamp: datetime
    symbol: str

    # Basic prices
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None

    # Order book data
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None

    # Volume data
    volume: Optional[float] = None
    dollar_volume: Optional[float] = None
    trade_count: Optional[int] = None

    # Market conditions
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    market_hours: bool = True

    # Execution context
    order_size: Optional[float] = None
    order_side: Optional[str] = None  # 'buy' or 'sell'
    urgency: Optional[str] = None  # 'low', 'medium', 'high'


@dataclass
class EntryPriceResult:
    """Result of entry price determination."""

    symbol: str
    timestamp: datetime
    entry_price: float
    pricing_method: PricingMethod
    confidence: float  # 0.0 to 1.0

    # Price components
    base_price: float
    market_impact: float = 0.0
    spread_cost: float = 0.0
    timing_adjustment: float = 0.0

    # Context used
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    execution_quality: Optional[float] = None

    # Alternative prices for comparison
    alternative_prices: Dict[str, float] = field(default_factory=dict)


class EntryPriceDeterminer(ErrorHandlingMixin):
    """
    Determines realistic entry prices for outcome classification.

    Features:
    - Multiple pricing methodologies
    - Market microstructure awareness
    - Execution cost modeling
    - Liquidity impact analysis
    - Adaptive pricing based on conditions
    """

    def __init__(self, db_pool: DatabasePool):
        """Initialize entry price determiner."""
        super().__init__()
        self.db_pool = db_pool

        # Configuration
        self._default_method = PricingMethod.ADAPTIVE
        self._spread_impact_factor = 0.5  # How much of spread to pay
        self._market_impact_model = "sqrt"  # "linear" or "sqrt"

        # Market impact parameters
        self._impact_coefficients = {"small_cap": 0.001, "mid_cap": 0.0005, "large_cap": 0.0002}

        # Liquidity thresholds
        self._liquidity_thresholds = {
            "high": 1000000,  # $1M+ daily volume
            "medium": 100000,  # $100K+ daily volume
            "low": 10000,  # $10K+ daily volume
        }

    @timer
    async def determine_entry_price(
        self,
        symbol: str,
        timestamp: datetime,
        order_size: Optional[float] = None,
        order_side: str = "buy",
        method: Optional[PricingMethod] = None,
        urgency: str = "medium",
    ) -> EntryPriceResult:
        """
        Determine entry price for a trade.

        Args:
            symbol: Trading symbol
            timestamp: Entry timestamp
            order_size: Order size in shares
            order_side: 'buy' or 'sell'
            method: Pricing method to use
            urgency: Execution urgency level

        Returns:
            Entry price result
        """
        with self._handle_error("determining entry price"):
            # Get market context
            context = await self._get_price_context(
                symbol, timestamp, order_size, order_side, urgency
            )

            # Select pricing method
            if method is None:
                method = await self._select_pricing_method(context)

            # Calculate entry price
            price_result = await self._calculate_entry_price(context, method)

            # Validate result
            await self._validate_price_result(price_result, context)

            # Record metrics
            record_metric(
                "entry_price_determiner.price_determined",
                1,
                tags={"symbol": symbol, "method": method.value, "side": order_side},
            )

            logger.debug(
                f"Determined entry price for {symbol}: {price_result.entry_price:.4f} "
                f"using {method.value} method"
            )

            return price_result

    async def determine_batch_entry_prices(
        self,
        requests: List[Tuple[str, datetime, Optional[float], str]],  # symbol, timestamp, size, side
        method: Optional[PricingMethod] = None,
        max_concurrent: int = 20,
    ) -> List[EntryPriceResult]:
        """
        Determine entry prices for multiple requests in batch.

        Args:
            requests: List of (symbol, timestamp, order_size, order_side) tuples
            method: Pricing method to use
            max_concurrent: Maximum concurrent requests

        Returns:
            List of entry price results
        """
        with self._handle_error("determining batch entry prices"):
            results = []

            # Process in batches
            for i in range(0, len(requests), max_concurrent):
                batch = requests[i : i + max_concurrent]

                # Process batch concurrently
                batch_results = await asyncio.gather(
                    *[
                        self.determine_entry_price(
                            symbol, timestamp, order_size, order_side, method
                        )
                        for symbol, timestamp, order_size, order_side in batch
                    ],
                    return_exceptions=True,
                )

                # Collect results
                for request, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error determining price for {request[0]}: {result}")
                    else:
                        results.append(result)

            logger.info(f"Determined entry prices for {len(results)}/{len(requests)} requests")

            return results

    async def _get_price_context(
        self,
        symbol: str,
        timestamp: datetime,
        order_size: Optional[float],
        order_side: str,
        urgency: str,
    ) -> PriceContext:
        """Get comprehensive price context."""
        context = PriceContext(
            timestamp=timestamp,
            symbol=symbol,
            order_size=order_size,
            order_side=order_side,
            urgency=urgency,
        )

        async with self.db_pool.acquire() as conn:
            # Get basic OHLCV data
            ohlcv_query = """
                SELECT open, high, low, close, volume
                FROM market_data
                WHERE symbol = $1 AND timestamp = $2
            """

            ohlcv_row = await conn.fetchrow(ohlcv_query, symbol, timestamp)

            if ohlcv_row:
                context.open_price = float(ohlcv_row["open"])
                context.high_price = float(ohlcv_row["high"])
                context.low_price = float(ohlcv_row["low"])
                context.close_price = float(ohlcv_row["close"])
                context.volume = float(ohlcv_row["volume"]) if ohlcv_row["volume"] else 0
                context.dollar_volume = context.volume * context.close_price

            # Get order book data if available
            book_query = """
                SELECT bid_price, ask_price, bid_size, ask_size
                FROM order_book_snapshots
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $2)))
                LIMIT 1
            """

            window_start = timestamp - timedelta(minutes=5)
            window_end = timestamp + timedelta(minutes=5)

            book_row = await conn.fetchrow(book_query, symbol, window_start, window_end)

            if book_row:
                context.bid_price = float(book_row["bid_price"])
                context.ask_price = float(book_row["ask_price"])
                context.bid_size = float(book_row["bid_size"]) if book_row["bid_size"] else 0
                context.ask_size = float(book_row["ask_size"]) if book_row["ask_size"] else 0
                context.spread = context.ask_price - context.bid_price

            # Calculate derived metrics
            await self._calculate_derived_context(context, conn)

        return context

    async def _calculate_derived_context(self, context: PriceContext, conn) -> None:
        """Calculate derived context metrics."""
        # Calculate volatility
        if context.close_price:
            vol_query = """
                SELECT close
                FROM market_data
                WHERE symbol = $1
                AND timestamp <= $2
                ORDER BY timestamp DESC
                LIMIT 20
            """

            vol_rows = await conn.fetch(vol_query, context.symbol, context.timestamp)

            if len(vol_rows) > 1:
                prices = [float(row["close"]) for row in vol_rows]
                returns = []
                for i in range(len(prices) - 1):
                    ret = (prices[i] - prices[i + 1]) / prices[i + 1]
                    returns.append(ret)

                if returns:
                    context.volatility = float(np.std(returns))

        # Calculate liquidity score
        if context.dollar_volume:
            if context.dollar_volume >= self._liquidity_thresholds["high"]:
                context.liquidity_score = 1.0
            elif context.dollar_volume >= self._liquidity_thresholds["medium"]:
                context.liquidity_score = 0.7
            elif context.dollar_volume >= self._liquidity_thresholds["low"]:
                context.liquidity_score = 0.4
            else:
                context.liquidity_score = 0.1

        # Check market hours
        hour = context.timestamp.hour
        context.market_hours = 9 <= hour <= 16  # Simplified market hours

    async def _select_pricing_method(self, context: PriceContext) -> PricingMethod:
        """Select optimal pricing method based on context."""
        # High liquidity and normal hours -> use midpoint or VWAP
        if (
            context.liquidity_score
            and context.liquidity_score > 0.7
            and context.market_hours
            and context.bid_price
            and context.ask_price
        ):
            return PricingMethod.MIDPOINT

        # Low liquidity -> use close price
        if context.liquidity_score and context.liquidity_score < 0.4:
            return PricingMethod.CLOSE

        # High urgency -> use arrival price
        if context.urgency == "high":
            return PricingMethod.ARRIVAL

        # Large orders -> use VWAP
        if context.order_size and context.volume and context.order_size > 0.05 * context.volume:
            return PricingMethod.VWAP

        # Default to close price
        return PricingMethod.CLOSE

    async def _calculate_entry_price(
        self, context: PriceContext, method: PricingMethod
    ) -> EntryPriceResult:
        """Calculate entry price using specified method."""
        if method == PricingMethod.CLOSE:
            return await self._calculate_close_price(context)
        elif method == PricingMethod.OPEN:
            return await self._calculate_open_price(context)
        elif method == PricingMethod.MIDPOINT:
            return await self._calculate_midpoint_price(context)
        elif method == PricingMethod.VWAP:
            return await self._calculate_vwap_price(context)
        elif method == PricingMethod.TWAP:
            return await self._calculate_twap_price(context)
        elif method == PricingMethod.ARRIVAL:
            return await self._calculate_arrival_price(context)
        else:
            # Default to close
            return await self._calculate_close_price(context)

    async def _calculate_close_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price using close price."""
        if not context.close_price:
            raise ValueError("Close price not available")

        return EntryPriceResult(
            symbol=context.symbol,
            timestamp=context.timestamp,
            entry_price=context.close_price,
            pricing_method=PricingMethod.CLOSE,
            confidence=0.9,
            base_price=context.close_price,
            market_conditions={
                "volume": context.volume,
                "liquidity_score": context.liquidity_score,
            },
        )

    async def _calculate_open_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price using open price."""
        if not context.open_price:
            raise ValueError("Open price not available")

        return EntryPriceResult(
            symbol=context.symbol,
            timestamp=context.timestamp,
            entry_price=context.open_price,
            pricing_method=PricingMethod.OPEN,
            confidence=0.8,
            base_price=context.open_price,
            market_conditions={
                "volume": context.volume,
                "liquidity_score": context.liquidity_score,
            },
        )

    async def _calculate_midpoint_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price using bid-ask midpoint."""
        if not context.bid_price or not context.ask_price:
            # Fallback to close price
            return await self._calculate_close_price(context)

        midpoint = (context.bid_price + context.ask_price) / 2
        spread_cost = context.spread * self._spread_impact_factor

        # Adjust for order side
        if context.order_side == "buy":
            entry_price = midpoint + spread_cost / 2
        else:
            entry_price = midpoint - spread_cost / 2

        return EntryPriceResult(
            symbol=context.symbol,
            timestamp=context.timestamp,
            entry_price=entry_price,
            pricing_method=PricingMethod.MIDPOINT,
            confidence=0.95,
            base_price=midpoint,
            spread_cost=spread_cost,
            market_conditions={
                "spread": context.spread,
                "bid_size": context.bid_size,
                "ask_size": context.ask_size,
            },
        )

    async def _calculate_vwap_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price using VWAP."""
        # Get intraday data for VWAP calculation
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT close, volume
                FROM market_data
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """

            start_of_day = context.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)

            rows = await conn.fetch(query, context.symbol, start_of_day, context.timestamp)

            if not rows:
                # Fallback to close price
                return await self._calculate_close_price(context)

            # Calculate VWAP
            total_volume = 0
            total_pv = 0

            for row in rows:
                price = float(row["close"])
                volume = float(row["volume"]) if row["volume"] else 0

                total_pv += price * volume
                total_volume += volume

            if total_volume == 0:
                return await self._calculate_close_price(context)

            vwap = total_pv / total_volume

            # Add market impact for large orders
            market_impact = 0.0
            if context.order_size and context.volume:
                participation_rate = context.order_size / context.volume
                market_impact = self._calculate_market_impact(participation_rate, context)

            entry_price = vwap + market_impact

            return EntryPriceResult(
                symbol=context.symbol,
                timestamp=context.timestamp,
                entry_price=entry_price,
                pricing_method=PricingMethod.VWAP,
                confidence=0.85,
                base_price=vwap,
                market_impact=market_impact,
                market_conditions={"vwap_periods": len(rows), "total_volume": total_volume},
            )

    async def _calculate_twap_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price using TWAP."""
        # Get intraday data for TWAP calculation
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT close
                FROM market_data
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """

            start_of_day = context.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)

            rows = await conn.fetch(query, context.symbol, start_of_day, context.timestamp)

            if not rows:
                return await self._calculate_close_price(context)

            # Calculate TWAP
            prices = [float(row["close"]) for row in rows]
            twap = sum(prices) / len(prices)

            return EntryPriceResult(
                symbol=context.symbol,
                timestamp=context.timestamp,
                entry_price=twap,
                pricing_method=PricingMethod.TWAP,
                confidence=0.8,
                base_price=twap,
                market_conditions={"twap_periods": len(prices)},
            )

    async def _calculate_arrival_price(self, context: PriceContext) -> EntryPriceResult:
        """Calculate entry price with arrival price model."""
        base_price = context.close_price or context.bid_price or context.ask_price

        if not base_price:
            raise ValueError("No base price available")

        # Calculate market impact
        market_impact = 0.0
        spread_cost = 0.0

        if context.order_size and context.volume:
            participation_rate = context.order_size / context.volume
            market_impact = self._calculate_market_impact(participation_rate, context)

        # Add spread cost for urgent orders
        if context.spread:
            spread_cost = context.spread * 0.7  # Pay most of the spread

        # Adjust for order side
        if context.order_side == "buy":
            entry_price = base_price + market_impact + spread_cost / 2
        else:
            entry_price = base_price - market_impact - spread_cost / 2

        return EntryPriceResult(
            symbol=context.symbol,
            timestamp=context.timestamp,
            entry_price=entry_price,
            pricing_method=PricingMethod.ARRIVAL,
            confidence=0.9,
            base_price=base_price,
            market_impact=market_impact,
            spread_cost=spread_cost,
            market_conditions={
                "urgency": context.urgency,
                "participation_rate": context.order_size / context.volume if context.volume else 0,
            },
        )

    def _calculate_market_impact(self, participation_rate: float, context: PriceContext) -> float:
        """Calculate market impact based on order size."""
        # Determine market cap tier (simplified)
        if context.dollar_volume and context.dollar_volume > 10000000:  # $10M+
            coefficient = self._impact_coefficients["large_cap"]
        elif context.dollar_volume and context.dollar_volume > 1000000:  # $1M+
            coefficient = self._impact_coefficients["mid_cap"]
        else:
            coefficient = self._impact_coefficients["small_cap"]

        # Apply impact model
        if self._market_impact_model == "sqrt":
            impact = coefficient * np.sqrt(participation_rate)
        else:  # linear
            impact = coefficient * participation_rate

        # Scale by volatility if available
        if context.volatility:
            impact *= 1 + context.volatility * 10  # Volatility adjustment

        # Convert to price impact
        base_price = context.close_price or 100  # Default price
        return impact * base_price

    async def _validate_price_result(self, result: EntryPriceResult, context: PriceContext) -> None:
        """Validate price result for reasonableness."""
        # Check if price is within reasonable bounds
        if context.high_price and context.low_price:
            if not (context.low_price <= result.entry_price <= context.high_price):
                # If outside daily range, check if it's reasonable
                daily_range = context.high_price - context.low_price
                max_deviation = daily_range * 0.1  # 10% of daily range

                if (
                    result.entry_price < context.low_price - max_deviation
                    or result.entry_price > context.high_price + max_deviation
                ):

                    logger.warning(
                        f"Entry price {result.entry_price:.4f} outside reasonable range "
                        f"[{context.low_price:.4f}, {context.high_price:.4f}] for {context.symbol}"
                    )

                    # Adjust confidence
                    result.confidence *= 0.5

        # Set alternative prices for comparison
        if context.close_price:
            result.alternative_prices["close"] = context.close_price
        if context.open_price:
            result.alternative_prices["open"] = context.open_price
        if context.bid_price and context.ask_price:
            result.alternative_prices["midpoint"] = (context.bid_price + context.ask_price) / 2

    def get_pricing_statistics(self, results: List[EntryPriceResult]) -> Dict[str, Any]:
        """Get statistics about pricing results."""
        if not results:
            return {}

        prices = [r.entry_price for r in results]
        methods = [r.pricing_method.value for r in results]
        confidences = [r.confidence for r in results]

        return {
            "total_prices": len(prices),
            "mean_price": float(np.mean(prices)),
            "std_price": float(np.std(prices)),
            "min_price": float(np.min(prices)),
            "max_price": float(np.max(prices)),
            "mean_confidence": float(np.mean(confidences)),
            "method_distribution": {method: methods.count(method) for method in set(methods)},
            "high_confidence_ratio": sum(1 for c in confidences if c > 0.8) / len(confidences),
        }

    def configure_impact_model(
        self,
        impact_coefficients: Optional[Dict[str, float]] = None,
        impact_model: str = "sqrt",
        spread_impact_factor: float = 0.5,
    ) -> None:
        """Configure market impact model parameters."""
        if impact_coefficients:
            self._impact_coefficients.update(impact_coefficients)

        self._market_impact_model = impact_model
        self._spread_impact_factor = spread_impact_factor

        logger.info(
            f"Configured impact model: {impact_model}, " f"spread factor: {spread_impact_factor}"
        )

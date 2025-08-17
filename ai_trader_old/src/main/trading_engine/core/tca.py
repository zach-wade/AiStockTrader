# Standard library imports
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# Third-party imports
import numpy as np
import pandas as pd


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExecutionData:
    """Container for trade execution data"""

    symbol: str
    side: OrderSide
    order_type: OrderType
    intended_price: float
    execution_price: float
    intended_quantity: float
    executed_quantity: float
    timestamp: datetime
    order_id: str

    # Market data at execution
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    market_volume: float

    # Additional metadata
    venue: str | None = None
    algo_strategy: str | None = None
    urgency: str | None = "NORMAL"  # URGENT, NORMAL, PASSIVE


@dataclass
class TCAMetrics:
    """Comprehensive transaction cost metrics"""

    # Core costs
    spread_cost: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float

    # Cost breakdown in bps
    spread_cost_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    opportunity_cost_bps: float
    total_cost_bps: float

    # Execution quality metrics
    execution_shortfall: float
    implementation_shortfall: float
    reversion_pnl: float  # 10min post-trade price movement
    participation_rate: float

    # Price benchmarks
    arrival_price: float
    vwap: float
    twap: float
    close_price: float

    # Additional analytics
    fill_rate: float
    avg_fill_price: float
    price_improvement: float
    effective_spread: float
    realized_spread: float

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class TransactionCostAnalyzer:
    """
    Comprehensive Transaction Cost Analysis system
    """

    def __init__(self, commission_rate: float = 0.0005):
        """
        Initialize TCA system

        Args:
            commission_rate: Commission rate (default: 5bps)
        """
        self.commission_rate = commission_rate
        self.execution_history: list[ExecutionData] = []
        self.tca_results: list[TCAMetrics] = []
        self.cache = get_global_cache()

    def analyze_execution(
        self, execution: ExecutionData, market_data: pd.DataFrame | None = None
    ) -> TCAMetrics:
        """
        Analyze a single trade execution

        Args:
            execution: Trade execution data
            market_data: Optional intraday market data around execution

        Returns:
            TCAMetrics object with cost analysis
        """
        # Calculate spread cost
        spread = execution.ask_price - execution.bid_price
        mid_price = (execution.bid_price + execution.ask_price) / 2

        if execution.side == OrderSide.BUY:
            spread_cost = (execution.execution_price - mid_price) * execution.executed_quantity
        else:
            spread_cost = (mid_price - execution.execution_price) * execution.executed_quantity

        spread_cost_bps = (spread_cost / (mid_price * execution.executed_quantity)) * 10000

        # Calculate market impact
        if execution.side == OrderSide.BUY:
            price_move = execution.execution_price - execution.intended_price
        else:
            price_move = execution.intended_price - execution.execution_price

        market_impact = price_move * execution.executed_quantity
        market_impact_bps = (
            market_impact / (execution.intended_price * execution.executed_quantity)
        ) * 10000

        # Calculate timing cost (if market data available)
        timing_cost = 0
        timing_cost_bps = 0

        if market_data is not None and len(market_data) > 0:
            # Get arrival price (price when order was placed)
            arrival_price = self._get_arrival_price(execution, market_data)

            # Calculate VWAP during execution period
            vwap = self._calculate_vwap(
                market_data, execution.timestamp, execution.timestamp + timedelta(minutes=5)
            )

            # Timing cost: difference between execution and arrival price
            if execution.side == OrderSide.BUY:
                timing_cost = (
                    execution.execution_price - arrival_price
                ) * execution.executed_quantity
            else:
                timing_cost = (
                    arrival_price - execution.execution_price
                ) * execution.executed_quantity

            timing_cost_bps = (timing_cost / (arrival_price * execution.executed_quantity)) * 10000
        else:
            arrival_price = execution.intended_price
            vwap = execution.execution_price

        # Calculate opportunity cost
        unfilled_quantity = execution.intended_quantity - execution.executed_quantity
        opportunity_cost = 0

        if unfilled_quantity > 0:
            # Estimate cost of not filling full order
            if execution.side == OrderSide.BUY:
                # Assume price continues up for unfilled portion
                opportunity_cost = unfilled_quantity * spread * 0.5  # Conservative estimate
            else:
                # Assume price continues down for unfilled portion
                opportunity_cost = unfilled_quantity * spread * 0.5

        opportunity_cost_bps = (
            (opportunity_cost / (mid_price * execution.intended_quantity)) * 10000
            if execution.intended_quantity > 0
            else 0
        )

        # Calculate total cost
        total_cost = spread_cost + market_impact + timing_cost + opportunity_cost
        total_cost_bps = (
            spread_cost_bps + market_impact_bps + timing_cost_bps + opportunity_cost_bps
        )

        # Calculate execution shortfall
        if execution.side == OrderSide.BUY:
            execution_shortfall = (
                execution.execution_price - execution.intended_price
            ) * execution.executed_quantity
        else:
            execution_shortfall = (
                execution.intended_price - execution.execution_price
            ) * execution.executed_quantity

        # Calculate implementation shortfall
        implementation_shortfall = execution_shortfall + opportunity_cost

        # Calculate effective and realized spread
        effective_spread = 2 * abs(execution.execution_price - mid_price)
        realized_spread = effective_spread  # Would need future price for true realized spread

        # Calculate fill rate
        fill_rate = (
            execution.executed_quantity / execution.intended_quantity
            if execution.intended_quantity > 0
            else 0
        )

        # Calculate participation rate
        participation_rate = (
            execution.executed_quantity / execution.market_volume
            if execution.market_volume > 0
            else 0
        )

        # Price improvement
        if execution.side == OrderSide.BUY:
            price_improvement = max(0, execution.ask_price - execution.execution_price)
        else:
            price_improvement = max(0, execution.execution_price - execution.bid_price)

        # Create TCA metrics
        metrics = TCAMetrics(
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            spread_cost_bps=spread_cost_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            opportunity_cost_bps=opportunity_cost_bps,
            total_cost_bps=total_cost_bps,
            execution_shortfall=execution_shortfall,
            implementation_shortfall=implementation_shortfall,
            reversion_pnl=0,  # Would need post-trade data
            participation_rate=participation_rate,
            arrival_price=arrival_price,
            vwap=vwap,
            twap=mid_price,  # Simplified
            close_price=0,  # Would need EOD data
            fill_rate=fill_rate,
            avg_fill_price=execution.execution_price,
            price_improvement=price_improvement,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            metadata={
                "symbol": execution.symbol,
                "side": execution.side.value,
                "order_type": execution.order_type.value,
                "venue": execution.venue,
                "algo_strategy": execution.algo_strategy,
                "urgency": execution.urgency,
            },
        )

        # Store results
        self.execution_history.append(execution)
        self.tca_results.append(metrics)

        return metrics

    def _get_arrival_price(self, execution: ExecutionData, market_data: pd.DataFrame) -> float:
        """Get price when order was placed"""
        # For simplicity, using intended price as arrival price
        # In practice, would look up actual market price at order time
        return execution.intended_price

    def _calculate_vwap(
        self, market_data: pd.DataFrame, start_time: datetime, end_time: datetime
    ) -> float:
        """Calculate VWAP over a time period"""
        # Simplified VWAP calculation
        # In practice, would use actual volume-weighted prices
        if "price" in market_data.columns and "volume" in market_data.columns:
            mask = (market_data.index >= start_time) & (market_data.index <= end_time)
            data_slice = market_data[mask]
            if len(data_slice) > 0:
                return (data_slice["price"] * data_slice["volume"]).sum() / data_slice[
                    "volume"
                ].sum()

        return 0

    def batch_analyze(self, executions: list[ExecutionData]) -> pd.DataFrame:
        """
        Analyze multiple executions and return summary statistics

        Args:
            executions: List of execution data

        Returns:
            DataFrame with TCA metrics for all executions
        """
        results = []

        for execution in executions:
            metrics = self.analyze_execution(execution)

            result_dict = {
                "symbol": execution.symbol,
                "side": execution.side.value,
                "timestamp": execution.timestamp,
                "executed_qty": execution.executed_quantity,
                "execution_price": execution.execution_price,
                "spread_cost_bps": metrics.spread_cost_bps,
                "market_impact_bps": metrics.market_impact_bps,
                "timing_cost_bps": metrics.timing_cost_bps,
                "total_cost_bps": metrics.total_cost_bps,
                "fill_rate": metrics.fill_rate,
                "price_improvement": metrics.price_improvement,
            }

            results.append(result_dict)

        return pd.DataFrame(results)

    def get_summary_statistics(
        self, lookback_days: int | None = None, symbol: str | None = None
    ) -> dict:
        """
        Get summary statistics for transaction costs

        Args:
            lookback_days: Number of days to look back
            symbol: Filter by symbol

        Returns:
            Dictionary with summary statistics
        """
        # Filter results
        filtered_results = self.tca_results

        if lookback_days:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            filtered_results = [r for r in filtered_results if r.timestamp >= cutoff_date]

        if symbol:
            filtered_results = [r for r in filtered_results if r.metadata.get("symbol") == symbol]

        if not filtered_results:
            return {"error": "No data available for the specified filters"}

        # Calculate statistics
        total_costs = [r.total_cost_bps for r in filtered_results]
        spread_costs = [r.spread_cost_bps for r in filtered_results]
        market_impacts = [r.market_impact_bps for r in filtered_results]
        fill_rates = [r.fill_rate for r in filtered_results]

        summary = {
            "period": f"Last {lookback_days} days" if lookback_days else "All time",
            "symbol": symbol or "All symbols",
            "trade_count": len(filtered_results),
            "avg_total_cost_bps": np.mean(total_costs),
            "median_total_cost_bps": np.median(total_costs),
            "std_total_cost_bps": np.std(total_costs),
            "avg_spread_cost_bps": np.mean(spread_costs),
            "avg_market_impact_bps": np.mean(market_impacts),
            "avg_fill_rate": np.mean(fill_rates),
            "cost_breakdown": {
                "spread": (
                    np.mean(spread_costs) / np.mean(total_costs) * 100
                    if np.mean(total_costs) > 0
                    else 0
                ),
                "market_impact": (
                    np.mean(market_impacts) / np.mean(total_costs) * 100
                    if np.mean(total_costs) > 0
                    else 0
                ),
            },
            "worst_executions": sorted(total_costs, reverse=True)[:5],
            "best_executions": sorted(total_costs)[:5],
        }

        return summary

    def identify_cost_drivers(self) -> dict:
        """
        Identify main drivers of transaction costs
        """
        if not self.tca_results:
            return {"error": "No execution data available"}

        # Group by various factors
        venue_costs = defaultdict(list)
        time_costs = defaultdict(list)
        size_costs = defaultdict(list)
        urgency_costs = defaultdict(list)

        for i, result in enumerate(self.tca_results):
            execution = self.execution_history[i]

            # By venue
            if execution.venue:
                venue_costs[execution.venue].append(result.total_cost_bps)

            # By time of day
            hour = execution.timestamp.hour
            time_costs[f"{hour:02d}:00"].append(result.total_cost_bps)

            # By order size
            size_bucket = self._get_size_bucket(execution.executed_quantity)
            size_costs[size_bucket].append(result.total_cost_bps)

            # By urgency
            urgency_costs[execution.urgency].append(result.total_cost_bps)

        # Calculate averages
        analysis = {
            "by_venue": {venue: np.mean(costs) for venue, costs in venue_costs.items()},
            "by_time": {time: np.mean(costs) for time, costs in time_costs.items()},
            "by_size": {size: np.mean(costs) for size, costs in size_costs.items()},
            "by_urgency": {urgency: np.mean(costs) for urgency, costs in urgency_costs.items()},
            "recommendations": self._generate_recommendations(
                venue_costs, time_costs, size_costs, urgency_costs
            ),
        }

        return analysis

    def _get_size_bucket(self, quantity: float) -> str:
        """Categorize order size"""
        if quantity < 100:
            return "Small"
        elif quantity < 1000:
            return "Medium"
        elif quantity < 10000:
            return "Large"
        else:
            return "Block"

    def _generate_recommendations(
        self, venue_costs, time_costs, size_costs, urgency_costs
    ) -> list[str]:
        """Generate actionable recommendations based on cost analysis"""
        recommendations = []

        # Venue recommendations
        if venue_costs:
            best_venue = min(venue_costs.items(), key=lambda x: np.mean(x[1]))
            worst_venue = max(venue_costs.items(), key=lambda x: np.mean(x[1]))
            if best_venue[0] != worst_venue[0]:
                recommendations.append(
                    f"Route more orders to {best_venue[0]} (avg cost: {np.mean(best_venue[1]):.1f}bps) "
                    f"instead of {worst_venue[0]} (avg cost: {np.mean(worst_venue[1]):.1f}bps)"
                )

        # Time recommendations
        if time_costs:
            best_time = min(time_costs.items(), key=lambda x: np.mean(x[1]))
            worst_time = max(time_costs.items(), key=lambda x: np.mean(x[1]))
            recommendations.append(
                f"Best execution times around {best_time[0]} (avg cost: {np.mean(best_time[1]):.1f}bps)"
            )

        # Size recommendations
        if size_costs and "Block" in size_costs:
            block_avg = np.mean(size_costs["Block"])
            if block_avg > 50:  # More than 50bps for block trades
                recommendations.append(
                    "Consider using algorithmic execution for block trades to reduce market impact"
                )

        return recommendations

    def calculate_slippage_metrics(
        self, intended_trades: list[dict], actual_trades: list[dict]
    ) -> dict:
        """
        Calculate detailed slippage metrics

        Args:
            intended_trades: List of intended trade details
            actual_trades: List of actual execution details

        Returns:
            Dictionary with slippage analysis
        """
        total_slippage = 0
        slippage_by_symbol = defaultdict(list)

        for intended, actual in zip(intended_trades, actual_trades):
            symbol = intended["symbol"]

            if intended["side"] == "BUY":
                slippage = (actual["price"] - intended["price"]) * actual["quantity"]
            else:
                slippage = (intended["price"] - actual["price"]) * actual["quantity"]

            slippage_bps = (slippage / (intended["price"] * actual["quantity"])) * 10000

            total_slippage += slippage
            slippage_by_symbol[symbol].append(slippage_bps)

        return {
            "total_slippage": total_slippage,
            "avg_slippage_bps": np.mean(
                [s for sublist in slippage_by_symbol.values() for s in sublist]
            ),
            "slippage_by_symbol": {
                symbol: {
                    "avg_bps": np.mean(slippages),
                    "std_bps": np.std(slippages),
                    "count": len(slippages),
                }
                for symbol, slippages in slippage_by_symbol.items()
            },
        }

    def export_report(self, filepath: str, format: str = "json"):
        """
        Export TCA report

        Args:
            filepath: Output file path
            format: Output format ('json' or 'csv')
        """
        if format == "json":
            report = {
                "summary": self.get_summary_statistics(),
                "cost_drivers": self.identify_cost_drivers(),
                "execution_count": len(self.execution_history),
                "generated_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

        elif format == "csv":
            df = self.batch_analyze(self.execution_history)
            df.to_csv(filepath, index=False)


class SmartOrderRouter:
    """
    Smart Order Router to minimize transaction costs
    """

    def __init__(self, tca_analyzer: TransactionCostAnalyzer):
        """
        Initialize Smart Order Router

        Args:
            tca_analyzer: TCA analyzer instance
        """
        self.tca = tca_analyzer
        self.venue_scores = {}
        self.routing_rules = {}

    def update_venue_scores(self):
        """Update venue scores based on historical TCA data"""
        cost_drivers = self.tca.identify_cost_drivers()

        if "by_venue" in cost_drivers:
            # Lower cost = higher score
            max_cost = max(cost_drivers["by_venue"].values()) if cost_drivers["by_venue"] else 1

            for venue, avg_cost in cost_drivers["by_venue"].items():
                self.venue_scores[venue] = 1 - (avg_cost / max_cost)

    def route_order(
        self, symbol: str, side: OrderSide, quantity: float, urgency: str = "NORMAL"
    ) -> dict[str, float]:
        """
        Determine optimal routing for an order

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            urgency: Order urgency level

        Returns:
            Dictionary of venue -> quantity allocations
        """
        self.update_venue_scores()

        if not self.venue_scores:
            return {"DEFAULT": quantity}

        # For urgent orders, use fastest venue regardless of cost
        if urgency == "URGENT":
            # In practice, would have latency data
            return {list(self.venue_scores.keys())[0]: quantity}

        # For normal orders, distribute based on scores
        total_score = sum(self.venue_scores.values())
        allocations = {}

        for venue, score in self.venue_scores.items():
            allocation = quantity * (score / total_score)
            if allocation > 0:
                allocations[venue] = allocation

        return allocations

    def optimize_execution_schedule(
        self, symbol: str, quantity: float, time_horizon: int
    ) -> list[dict]:
        """
        Create optimal execution schedule to minimize market impact

        Args:
            symbol: Trading symbol
            quantity: Total quantity to execute
            time_horizon: Time horizon in minutes

        Returns:
            List of execution instructions
        """
        cost_drivers = self.tca.identify_cost_drivers()

        # Get best execution times
        time_costs = cost_drivers.get("by_time", {})
        if not time_costs:
            # Default TWAP schedule
            slices = time_horizon // 5  # 5-minute intervals
            qty_per_slice = quantity / slices

            return [
                {
                    "time_offset": i * 5,
                    "quantity": qty_per_slice,
                    "type": "LIMIT",
                    "urgency": "PASSIVE",
                }
                for i in range(slices)
            ]

        # Optimize based on historical costs
        sorted_times = sorted(time_costs.items(), key=lambda x: x[1])
        best_times = [t[0] for t in sorted_times[:3]]  # Top 3 best times

        # Allocate more quantity to better execution times
        schedule = []
        for i, time_str in enumerate(best_times):
            weight = 1 / (i + 1)  # Decreasing weights
            schedule.append(
                {
                    "preferred_time": time_str,
                    "quantity": quantity
                    * weight
                    / sum(1 / (j + 1) for j in range(len(best_times))),
                    "type": "LIMIT",
                    "urgency": "NORMAL",
                }
            )

        return schedule


# Example usage
if __name__ == "__main__":
    # Initialize TCA system
    tca = TransactionCostAnalyzer(commission_rate=0.0005)

    # Example execution data
    execution = ExecutionData(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        intended_price=150.00,
        execution_price=150.05,
        intended_quantity=1000,
        executed_quantity=950,
        timestamp=datetime.now(),
        order_id="ORD123456",
        bid_price=149.98,
        ask_price=150.02,
        bid_size=500,
        ask_size=800,
        market_volume=1000000,
        venue="NASDAQ",
        algo_strategy="VWAP",
        urgency="NORMAL",
    )

    # Analyze execution
    metrics = tca.analyze_execution(execution)

    print("Transaction Cost Analysis:")
    print("-" * 50)
    print(f"Total Cost: {metrics.total_cost_bps:.2f} bps (${metrics.total_cost:.2f})")
    print(f"  - Spread Cost: {metrics.spread_cost_bps:.2f} bps")
    print(f"  - Market Impact: {metrics.market_impact_bps:.2f} bps")
    print(f"  - Timing Cost: {metrics.timing_cost_bps:.2f} bps")
    print(f"  - Opportunity Cost: {metrics.opportunity_cost_bps:.2f} bps")
    print(f"\nFill Rate: {metrics.fill_rate:.1%}")
    print(f"Price Improvement: ${metrics.price_improvement:.4f}")
    print(f"Effective Spread: ${metrics.effective_spread:.4f}")

    # Get summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print("-" * 50)

    # Simulate more executions for statistics
    for i in range(10):
        sim_execution = ExecutionData(
            symbol="AAPL",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            intended_price=150.00 + secure_numpy_normal(0, 0.1),
            execution_price=150.00 + secure_numpy_normal(0.05, 0.1),
            intended_quantity=1000,
            executed_quantity=950 + np.secure_randint(-50, 50),
            timestamp=datetime.now() - timedelta(hours=i),
            order_id=f"ORD{i:06d}",
            bid_price=149.98,
            ask_price=150.02,
            bid_size=500,
            ask_size=800,
            market_volume=1000000,
            venue="NASDAQ" if i % 3 == 0 else "NYSE",
            urgency="URGENT" if i % 4 == 0 else "NORMAL",
        )
        tca.analyze_execution(sim_execution)

    summary = tca.get_summary_statistics()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    # Identify cost drivers
    print("\n" + "=" * 50)
    print("Cost Driver Analysis:")
    print("-" * 50)

    drivers = tca.identify_cost_drivers()
    for category, data in drivers.items():
        if category != "recommendations":
            print(f"\n{category}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {v:.2f} bps")

    print("\nRecommendations:")
    for rec in drivers.get("recommendations", []):
        print(f"  • {rec}")

    # Smart Order Router example
    print("\n" + "=" * 50)
    print("Smart Order Router:")
    print("-" * 50)

    router = SmartOrderRouter(tca)
    allocations = router.route_order(
        symbol="AAPL", side=OrderSide.BUY, quantity=10000, urgency="NORMAL"
    )

    print("Order Routing Allocations:")
    for venue, qty in allocations.items():
        print(f"  {venue}: {qty:.0f} shares")

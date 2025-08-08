import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy import stats

from .base_hft_strategy import BaseHFTStrategy

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Point-in-time order book state"""
    timestamp: datetime
    symbol: str
    bids: List[tuple[float, float]]  # [(price, size), ...]
    asks: List[tuple[float, float]]  # [(price, size), ...]
    bid_depth: float  # Total bid volume
    ask_depth: float  # Total ask volume
    spread: float
    mid_price: float
    imbalance: float  # -1 to 1 (negative = selling pressure)


@dataclass
class MicrostructureSignal:
    """Signal based on order book dynamics"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    signal_type: str  # 'imbalance', 'sweep', 'iceberg', 'spread'
    confidence: float
    expected_move: float  # Expected price move in bps
    time_horizon: int  # Expected holding period in seconds
    entry_price: float
    metadata: Dict = field(default_factory=dict)


class MicrostructureAlphaStrategy(BaseHFTStrategy):
    """
    High-frequency strategy that trades on order book dynamics and microstructure patterns,
    refactored to be an event-driven component of the HFT Engine.
    """
    
    def __init__(self, config: Dict, strategy_specific_config: Dict):
        # REFACTOR: Calls the new BaseHFTStrategy __init__
        super().__init__(config, strategy_specific_config)
        self.name = "microstructure_alpha"
        
        # --- All parameters and state are preserved from your original file ---
        # Thresholds
        self.min_imbalance = self.params.get('min_imbalance', 0.65)
        self.min_spread_bps = self.params.get('min_spread_bps', 2)
        self.iceberg_threshold = self.params.get('iceberg_threshold', 0.8)
        self.sweep_size_multiplier = self.params.get('sweep_size_multiplier', 3)
        
        # Holding periods (seconds)
        self.holding_periods = self.params.get('holding_periods', {
            'imbalance': 30, 'sweep': 10, 'iceberg': 60, 'spread': 20
        })
        
        # Risk parameters
        self.max_position_per_symbol = self.params.get('max_position_per_symbol', 0.02)
        self.stop_loss_bps = self.params.get('stop_loss_bps', 10)
        
        # In-memory state tracking
        self.order_books: Dict[str, Deque[OrderBookSnapshot]] = defaultdict(lambda: deque(maxlen=1000))
        self.trade_flow: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=5000))
        
        logger.info("MicrostructureAlphaStrategy initialized for HFT Engine.")
    
    # REFACTOR: This is the new primary entry point for order book logic
    async def on_orderbook_update(self, symbol: str, orderbook_data: Dict) -> List[Dict]:
        """Handles a new order book event and generates signals if patterns are detected."""
        snapshot = self._parse_and_store_snapshot(symbol, orderbook_data)
        if not snapshot:
            return []

        snapshots_history = list(self.order_books[symbol])
        if len(snapshots_history) < 50: # Need sufficient history for analysis
            return []

        try:
            # Analyze all potential microstructure patterns concurrently
            signal_checks = [
                self._check_order_imbalance(symbol, snapshots_history),
                self._check_sweep_detection(symbol, snapshots_history),
                self._check_iceberg_orders(symbol, snapshots_history),
                self._check_spread_dynamics(symbol, snapshots_history)
            ]
            potential_signals = await asyncio.gather(*signal_checks)
            
            # Filter out nulls and choose the best signal based on confidence
            valid_signals = [s for s in potential_signals if s is not None and s.action != 'HOLD']
            if not valid_signals:
                return []

            best_signal = max(valid_signals, key=lambda s: s.confidence)
            
            # Format the signal into an execution-ready order dictionary
            return [self._format_signal(best_signal)]

        except Exception as e:
            logger.error(f"Error generating microstructure signal for {symbol}: {e}", exc_info=True)
        
        return []

    # REFACTOR: This is the new entry point for trade logic
    async def on_trade_update(self, symbol: str, trade_data: Dict) -> List[Dict]:
        """Processes new trade data to inform analyses like sweep detection."""
        self.trade_flow[symbol].append({
            'timestamp': trade_data.get('timestamp', datetime.now(timezone.utc)),
            'price': trade_data.get('price'),
            'size': trade_data.get('size'),
            'side': trade_data.get('side')
        })
        # This strategy doesn't trade directly on individual trades, but other HFT
        # strategies might, so we return an empty list of signals.
        return []

    # --- All internal logic methods are preserved from your original file ---

    def _parse_and_store_snapshot(self, symbol: str, data: Dict) -> Optional[OrderBookSnapshot]:
        """Helper to parse raw data and store a rich OrderBookSnapshot object."""
        try:
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            if not bids or not asks: return None
            
            bid_depth = sum(size for _, size in bids)
            ask_depth = sum(size for _, size in asks)
            mid_price = (asks[0][0] + bids[0][0]) / 2
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

            snapshot = OrderBookSnapshot(
                timestamp=datetime.fromisoformat(data['timestamp']),
                symbol=symbol, bids=bids, asks=asks, bid_depth=bid_depth, ask_depth=ask_depth,
                spread=(asks[0][0] - bids[0][0]), mid_price=mid_price, imbalance=imbalance
            )
            self.order_books[symbol].append(snapshot)
            return snapshot
        except Exception as e:
            logger.error(f"Failed to parse orderbook snapshot for {symbol}: {e}")
            return None

    async def _check_order_imbalance(self, symbol: str, snapshots: List[OrderBookSnapshot]) -> Optional[MicrostructureSignal]:
        current = snapshots[-1]
        weighted_imbalance = self._calculate_weighted_imbalance(current)
        
        if abs(weighted_imbalance) > self.min_imbalance and self._check_persistent_imbalance(snapshots[-30:]):
            return MicrostructureSignal(
                symbol=symbol, action='BUY' if weighted_imbalance > 0 else 'SELL', signal_type='imbalance',
                confidence=min(abs(weighted_imbalance), 0.95),
                expected_move=self._predict_price_move(weighted_imbalance, current.spread),
                time_horizon=self.holding_periods['imbalance'], entry_price=current.mid_price,
                metadata={'weighted_imbalance': weighted_imbalance}
            )
        return None

    def _calculate_weighted_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        mid, weighted_bid_volume, weighted_ask_volume = snapshot.mid_price, 0, 0
        for price, size in snapshot.bids[:10]:
            weighted_bid_volume += size * (1 / (1 + abs(price - mid) / mid))
        for price, size in snapshot.asks[:10]:
            weighted_ask_volume += size * (1 / (1 + abs(price - mid) / mid))
        total_volume = weighted_bid_volume + weighted_ask_volume
        return (weighted_bid_volume - weighted_ask_volume) / total_volume if total_volume > 0 else 0.0

    def _check_persistent_imbalance(self, snapshots: List[OrderBookSnapshot], threshold: float = 0.5) -> bool:
        if len(snapshots) < 10: return False
        significant_count = sum(1 for s in snapshots if abs(s.imbalance) > threshold)
        return significant_count > len(snapshots) * 0.7

    async def _check_sweep_detection(self, symbol: str, snapshots: List[OrderBookSnapshot]) -> Optional[MicrostructureSignal]:
        if len(snapshots) < 2: return None
        current, previous = snapshots[-1], snapshots[-2]
        
        bid_sweep = previous.bid_depth > 0 and (previous.bid_depth - current.bid_depth) / previous.bid_depth > 0.5
        ask_sweep = previous.ask_depth > 0 and (previous.ask_depth - current.ask_depth) / previous.ask_depth > 0.5
        price_move = abs(current.mid_price - previous.mid_price) / previous.mid_price
        
        if (bid_sweep or ask_sweep) and price_move > 0.0005:
            recent_trades = list(self.trade_flow[symbol])[-100:]
            if len(recent_trades) > 10:
                avg_size = np.mean([t['size'] for t in recent_trades[:-10]])
                if any(t['size'] > avg_size * self.sweep_size_multiplier for t in recent_trades[-10:]):
                    return MicrostructureSignal(
                        symbol=symbol, action='SELL' if bid_sweep else 'BUY', signal_type='sweep',
                        confidence=0.8, expected_move=price_move * 10000,
                        time_horizon=self.holding_periods['sweep'], entry_price=current.mid_price,
                        metadata={'price_impact_bps': price_move * 10000}
                    )
        return None

    async def _check_iceberg_orders(self, symbol: str, snapshots: List[OrderBookSnapshot]) -> Optional[MicrostructureSignal]:
        # This is a complex detection logic, preserved as is from your file.
        # It looks for price levels that consistently reload after being executed against.
        if len(snapshots) < 50: return None
        price_levels = defaultdict(list)
        for snapshot in snapshots[-50:]:
            for price, size in snapshot.bids[:3]: price_levels[('bid', price)].append(size)
            for price, size in snapshot.asks[:3]: price_levels[('ask', price)].append(size)

        for (side, price), sizes in price_levels.items():
            if len(sizes) > 10:
                changes = np.diff(sizes)
                reloads = sum(1 for i in range(len(changes)-1) if changes[i] < 0 and changes[i+1] > 0)
                if reloads > len(changes) * 0.3:
                    return MicrostructureSignal(
                        symbol=symbol, action='BUY' if side == 'bid' else 'SELL',
                        signal_type='iceberg', confidence=0.75, expected_move=10,
                        time_horizon=self.holding_periods['iceberg'], entry_price=snapshots[-1].mid_price,
                        metadata={'iceberg_price': price}
                    )
        return None

    async def _check_spread_dynamics(self, symbol: str, snapshots: List[OrderBookSnapshot]) -> Optional[MicrostructureSignal]:
        # This logic for liquidity provision is preserved as is.
        if len(snapshots) < 20 or not snapshots[-1].bids or not snapshots[-1].asks: return None
        current = snapshots[-1]
        spread_bps = (current.asks[0][0] - current.bids[0][0]) / current.mid_price * 10000
        historical_spreads = [s.spread / s.mid_price * 10000 for s in snapshots[-20:] if s.mid_price > 0]
        if not historical_spreads: return None
        
        avg_spread, spread_std = np.mean(historical_spreads), np.std(historical_spreads)
        if spread_std > 0 and spread_bps > avg_spread + 2 * spread_std:
            return MicrostructureSignal(
                symbol=symbol, action='HOLD', signal_type='spread_widening', confidence=0.6,
                expected_move=spread_bps/2, time_horizon=self.holding_periods['spread'],
                entry_price=current.mid_price, metadata={'spread_zscore': (spread_bps - avg_spread) / spread_std}
            )
        return None

    def _predict_price_move(self, imbalance: float, spread: float) -> float:
        base_move = abs(imbalance) * 20  # In basis points
        spread_adjustment = max(0.5, 1 - spread / 0.001)
        return base_move * spread_adjustment

    def _format_signal(self, signal: MicrostructureSignal) -> Dict:
        """Formats the internal signal into an execution-ready order dictionary."""
        base_size = self.params.get('base_position_size', 0.01)
        size_multipliers = {'imbalance': 1.0, 'sweep': 0.5, 'iceberg': 1.2, 'spread': 0.8}
        size = base_size * size_multipliers.get(signal.signal_type, 1.0) * signal.confidence
        
        return {
            'symbol': signal.symbol,
            'action': signal.action,
            'size_percent': min(size, self.max_position_per_symbol),
            'signal_type': f"{self.name}:{signal.signal_type}",
            'confidence': signal.confidence,
            'time_horizon_sec': signal.time_horizon,
            'entry_price': signal.entry_price,
            'stop_loss_bps': self.stop_loss_bps,
            'take_profit_bps': signal.expected_move,
            'metadata': signal.metadata
        }
"""
Trading Metrics Calculator

Handles all trading-specific performance calculations.
"""

from typing import List, Tuple
import numpy as np

from ..models.trade_record import TradeRecord


class TradingMetricsCalculator:
    """Trading-specific performance calculations."""
    
    @staticmethod
    def win_rate(trades: List[TradeRecord]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[TradeRecord]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def average_win_loss(trades: List[TradeRecord]) -> Tuple[float, float]:
        """Calculate average win and loss."""
        winning_trades = [trade.pnl for trade in trades if trade.pnl > 0]
        losing_trades = [trade.pnl for trade in trades if trade.pnl < 0]
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        return avg_win, avg_loss
    
    @staticmethod
    def largest_win_loss(trades: List[TradeRecord]) -> Tuple[float, float]:
        """Calculate largest win and loss."""
        if not trades:
            return 0.0, 0.0
        
        pnls = [trade.pnl for trade in trades]
        return max(pnls), min(pnls)
    
    @staticmethod
    def execution_metrics(trades: List[TradeRecord]) -> dict:
        """Calculate execution-related metrics."""
        if not trades:
            return {
                'avg_execution_time': 0.0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_fees': 0.0
            }
        
        return {
            'avg_execution_time': np.mean([trade.execution_time_ms for trade in trades]),
            'total_commission': sum(trade.commission for trade in trades),
            'total_slippage': sum(trade.slippage for trade in trades),
            'total_fees': sum(trade.fees for trade in trades)
        }
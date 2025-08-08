#!/usr/bin/env python
"""
Simple Backtest Runner - Direct strategy evaluation without BacktestEngine
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from ai_trader.models.strategies.mean_reversion import MeanReversionStrategy
from ai_trader.models.strategies.correlation_strategy import CorrelationStrategy

from ai_trader.utils.database import DatabasePool
from main.config.config_manager import ModularConfigManager as ConfigManager
from ai_trader.utils.core import setup_logging
from sqlalchemy import text

# Setup logging
logger = setup_logging(__name__)

load_dotenv()


class SimpleBacktestRunner:
    """Simplified backtest runner that works with actual database"""
    
    def __init__(self):
        # Load configuration
        config_manager = ConfigManager()
        self.config = config_manager.load_config('unified_config.yaml')
        
        # Initialize database
        db_user = os.getenv('DB_USER', 'zachwade')
        db_password = os.getenv('DB_PASSWORD', '')
        if db_password:
            db_url = f"postgresql://{db_user}:{db_password}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'ai_trader')}"
        else:
            db_url = f"postgresql://{db_user}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'ai_trader')}"
        
        self.db_pool = DatabasePool()
        self.db_pool.initialize(database_url=db_url)
        
        # Results storage
        self.results = {}
        
    async def fetch_top_symbols(self, limit: int = 10) -> List[str]:
        """Get top symbols by recent volume"""
        with self.db_pool.get_session() as session:
            query = text("""
                SELECT symbol, AVG(volume) as avg_volume
                FROM market_data 
                WHERE timestamp > NOW() - INTERVAL '30 days'
                  AND interval = '1minute'
                GROUP BY symbol 
                HAVING COUNT(*) > 1000
                ORDER BY avg_volume DESC 
                LIMIT :limit
            """)
            result = session.execute(query, {'limit': limit})
            rows = result.fetchall()
            return [row[0] for row in rows]
    
    async def fetch_symbol_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        with self.db_pool.get_session() as session:
            query = text("""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = :symbol 
                  AND timestamp > NOW() - INTERVAL :days
                  AND interval = '1minute'
                ORDER BY timestamp
            """)
            result = session.execute(query, {
                'symbol': symbol,
                'days': f'{days} days'
            })
            
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            return df
    
    def calculate_simple_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized metrics (assuming minute data, ~252 trading days)
        minutes_per_year = 252 * 6.5 * 60  # 252 days, 6.5 hours, 60 minutes
        n_periods = len(returns)
        annualization_factor = minutes_per_year / n_periods if n_periods > 0 else 1
        
        annual_return = (1 + total_return) ** annualization_factor - 1
        annual_vol = returns.std() * np.sqrt(annualization_factor)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_periods': n_periods
        }
    
    async def run_backtest(self):
        """Run simple backtest on top symbols"""
        logger.info("Starting simple backtest...")
        
        # Get top symbols
        symbols = await self.fetch_top_symbols(limit=5)
        logger.info(f"Testing on symbols: {symbols}")
        
        if not symbols:
            logger.error("No symbols found")
            return
        
        # Test on each symbol
        for symbol in symbols:
            logger.info(f"\nTesting {symbol}...")
            
            # Fetch data
            data = await self.fetch_symbol_data(symbol, days=7)  # Just 1 week for testing
            
            if data.empty:
                logger.warning(f"No data for {symbol}")
                continue
                
            logger.info(f"Loaded {len(data)} data points for {symbol}")
            
            # Simple momentum strategy simulation
            # Buy when price > 20-period SMA, sell when below
            data['sma20'] = data['close'].rolling(20).mean()
            data['signal'] = (data['close'] > data['sma20']).astype(int)
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate equity curve
            initial_capital = 100000
            data['equity'] = initial_capital * (1 + data['strategy_returns']).cumprod()
            
            # Calculate metrics
            metrics = self.calculate_simple_metrics(data['equity'].dropna())
            
            self.results[symbol] = {
                'metrics': metrics,
                'data_points': len(data),
                'first_date': data.index[0],
                'last_date': data.index[-1]
            }
            
            # Log results
            logger.info(f"Results for {symbol}:")
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save backtest results"""
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"simple_backtest_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        logger.info(f"\nResults saved to {results_file}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*50)
        
        for symbol, result in self.results.items():
            metrics = result['metrics']
            logger.info(f"\n{symbol}:")
            logger.info(f"  Data Points: {result['data_points']}")
            logger.info(f"  Period: {result['first_date']} to {result['last_date']}")
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")


async def main():
    """Run the simple backtest"""
    try:
        runner = SimpleBacktestRunner()
        await runner.run_backtest()
    finally:
        if 'runner' in locals():
            runner.db_pool.dispose()


if __name__ == "__main__":
    asyncio.run(main())
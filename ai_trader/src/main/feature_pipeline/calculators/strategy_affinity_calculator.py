"""
Strategy Affinity Calculator

Calculates and stores strategy affinity scores for symbols to support Layer 1.5 filtering.
Analyzes historical price data and technical indicators to determine how well each symbol
aligns with different trading strategies: momentum, mean_reversion, breakout, sentiment.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.repositories.market_data_repository import MarketDataRepository
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


class StrategyAffinityCalculator:
    """
    Calculates strategy affinity scores for symbols based on historical performance
    with different trading strategies.
    """
    
    def __init__(self, config: Any, db_adapter: IAsyncDatabase):
        """
        Initialize the calculator.
        
        Args:
            config: Application configuration
            db_adapter: Database adapter for data access
        """
        self.config = config
        self.db_adapter = db_adapter
        self.market_data_repo = MarketDataRepository(db_adapter)
        
        # Load configuration parameters
        self.params = self.config.get('strategy_affinity', {})
        self.lookback_days = self.params.get('lookback_days', 90)
        self.min_data_points = self.params.get('min_data_points', 50)
        self.momentum_period = self.params.get('momentum_period', 20)
        self.mean_reversion_period = self.params.get('mean_reversion_period', 14)
        self.breakout_period = self.params.get('breakout_period', 30)
        
    async def calculate_affinities_batch(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate strategy affinity scores for a batch of symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary mapping symbols to their affinity scores
        """
        logger.info(f"🎯 Calculating strategy affinities for {len(symbols)} symbols...")
        
        results = {}
        
        # Process symbols in smaller batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: symbols {i+1}-{min(i+batch_size, len(symbols))}")
            
            batch_results = await self._process_symbol_batch(batch_symbols)
            results.update(batch_results)
            
        logger.info(f"✅ Completed affinity calculations for {len(results)} symbols")
        return results
    
    async def _process_symbol_batch(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Process a batch of symbols concurrently."""
        tasks = [self._calculate_symbol_affinities(symbol) for symbol in symbols]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for symbol, result in zip(symbols, batch_results):
            if isinstance(result, dict):
                results[symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Error calculating affinities for {symbol}: {result}")
                # Provide default scores for failed calculations
                results[symbol] = self._get_default_affinities()
            else:
                logger.warning(f"Unexpected result type for {symbol}: {type(result)}")
                results[symbol] = self._get_default_affinities()
                
        return results
    
    async def _calculate_symbol_affinities(self, symbol: str) -> Dict[str, float]:
        """
        Calculate strategy affinity scores for a single symbol.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with affinity scores for each strategy
        """
        try:
            # Get historical market data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.lookback_days)
            
            market_data = await self.market_data_repo.get_by_date_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not market_data or len(market_data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(market_data) if market_data else 0} points")
                return self._get_default_affinities()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(market_data)
            df = df.sort_values('timestamp')
            
            # Convert decimal columns to float for pandas operations
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            # Calculate strategy affinities
            affinities = {
                'momentum_affinity': self._calculate_momentum_affinity(df),
                'mean_reversion_affinity': self._calculate_mean_reversion_affinity(df),
                'breakout_affinity': self._calculate_breakout_affinity(df),
                'sentiment_affinity': self._calculate_sentiment_affinity(df)
            }
            
            # Normalize scores to 0-1 range
            affinities = self._normalize_affinities(affinities)
            
            logger.debug(f"Affinities for {symbol}: {affinities}")
            return affinities
            
        except Exception as e:
            logger.error(f"Error calculating affinities for {symbol}: {e}", exc_info=True)
            return self._get_default_affinities()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Volatility indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_momentum_affinity(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum strategy affinity based on trend following indicators.
        
        Higher scores indicate stronger momentum characteristics:
        - Consistent price trends
        - Moving average alignment
        - MACD momentum signals
        """
        try:
            # Price momentum (trend consistency)
            momentum_returns = df['returns'].rolling(window=self.momentum_period).sum()
            positive_momentum_periods = (momentum_returns > 0.02).sum()  # >2% gains
            momentum_consistency = positive_momentum_periods / len(momentum_returns.dropna())
            
            # Moving average alignment (upward bias)
            ma_alignment = ((df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])).sum()
            ma_alignment_score = ma_alignment / len(df.dropna())
            
            # MACD momentum strength
            macd_positive = (df['macd'] > df['macd_signal']).sum()
            macd_score = macd_positive / len(df.dropna())
            
            # Volume confirmation (higher volume on up days)
            up_days = df['returns'] > 0
            volume_confirmation = df.loc[up_days, 'volume_ratio'].mean()
            volume_score = min(volume_confirmation / 1.5, 1.0) if not pd.isna(volume_confirmation) else 0.5
            
            # Combine scores with weights
            momentum_affinity = (
                momentum_consistency * 0.35 +
                ma_alignment_score * 0.25 +
                macd_score * 0.25 +
                volume_score * 0.15
            )
            
            return float(np.clip(momentum_affinity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating momentum affinity: {e}")
            return 0.5
    
    def _calculate_mean_reversion_affinity(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion strategy affinity based on oscillating indicators.
        
        Higher scores indicate stronger mean reversion characteristics:
        - RSI oscillations between overbought/oversold
        - Bollinger Band bounces
        - Price reversals from extremes
        """
        try:
            # RSI mean reversion signals
            rsi_oversold = (df['rsi'] < 30).sum()
            rsi_overbought = (df['rsi'] > 70).sum()
            rsi_extremes = (rsi_oversold + rsi_overbought) / len(df.dropna())
            
            # Bollinger Band mean reversion
            bb_lower_touches = (df['bb_position'] < 0.1).sum()  # Near lower band
            bb_upper_touches = (df['bb_position'] > 0.9).sum()  # Near upper band
            bb_extremes = (bb_lower_touches + bb_upper_touches) / len(df.dropna())
            
            # Price reversals (high volatility with return to mean)
            high_vol_periods = df['volatility'] > df['volatility'].quantile(0.7)
            reversals = 0
            if high_vol_periods.sum() > 0:
                # Look for price returning to SMA after high volatility
                for idx in df[high_vol_periods].index:
                    if idx + 5 < len(df):
                        future_prices = df.loc[idx:idx+5, 'close']
                        sma_value = df.loc[idx, 'sma_20']
                        if not pd.isna(sma_value):
                            if any(abs(price - sma_value) / sma_value < 0.02 for price in future_prices):
                                reversals += 1
                reversal_score = reversals / high_vol_periods.sum()
            else:
                reversal_score = 0.0
            
            # Low trending periods (sideways movement)
            trending_strength = abs(df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            sideways_score = 1.0 - min(trending_strength / 0.20, 1.0)  # Penalize strong trends
            
            # Combine scores
            mean_reversion_affinity = (
                rsi_extremes * 0.30 +
                bb_extremes * 0.25 +
                reversal_score * 0.25 +
                sideways_score * 0.20
            )
            
            return float(np.clip(mean_reversion_affinity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion affinity: {e}")
            return 0.5
    
    def _calculate_breakout_affinity(self, df: pd.DataFrame) -> float:
        """
        Calculate breakout strategy affinity based on volatility and range expansion.
        
        Higher scores indicate stronger breakout characteristics:
        - Volatility expansion patterns
        - Range breakouts with volume
        - Consolidation followed by expansion
        """
        try:
            # Volatility expansion (increasing volatility over time)
            vol_trend = np.polyfit(range(len(df['volatility'].dropna())), 
                                 df['volatility'].dropna(), 1)[0]
            vol_expansion_score = min(vol_trend / 0.001, 1.0) if vol_trend > 0 else 0.0
            
            # Range breakouts
            rolling_high = df['high'].rolling(window=self.breakout_period).max()
            rolling_low = df['low'].rolling(window=self.breakout_period).min()
            
            # Count breakouts above recent highs with volume confirmation
            high_breakouts = 0
            low_breakouts = 0
            
            for i in range(self.breakout_period, len(df)):
                if (df.iloc[i]['high'] > rolling_high.iloc[i-1] and 
                    df.iloc[i]['volume_ratio'] > 1.5):
                    high_breakouts += 1
                elif (df.iloc[i]['low'] < rolling_low.iloc[i-1] and 
                      df.iloc[i]['volume_ratio'] > 1.5):
                    low_breakouts += 1
            
            breakout_frequency = (high_breakouts + low_breakouts) / (len(df) - self.breakout_period)
            breakout_score = min(breakout_frequency / 0.1, 1.0)  # Normalize to reasonable frequency
            
            # Consolidation patterns (periods of low volatility followed by expansion)
            consolidation_periods = 0
            window = 10
            
            for i in range(window, len(df) - window):
                # Check for low volatility period
                pre_vol = df['volatility'].iloc[i-window:i].mean()
                post_vol = df['volatility'].iloc[i:i+window].mean()
                
                if (pre_vol < df['volatility'].quantile(0.3) and 
                    post_vol > df['volatility'].quantile(0.7)):
                    consolidation_periods += 1
            
            consolidation_score = min(consolidation_periods / 10, 1.0)
            
            # Volume spike patterns
            volume_spikes = (df['volume_ratio'] > 2.0).sum()
            volume_spike_score = min(volume_spikes / (len(df) * 0.05), 1.0)  # 5% of days
            
            # Combine scores
            breakout_affinity = (
                vol_expansion_score * 0.25 +
                breakout_score * 0.35 +
                consolidation_score * 0.25 +
                volume_spike_score * 0.15
            )
            
            return float(np.clip(breakout_affinity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating breakout affinity: {e}")
            return 0.5
    
    def _calculate_sentiment_affinity(self, df: pd.DataFrame) -> float:
        """
        Calculate sentiment strategy affinity based on price/volume patterns
        that typically respond to news and sentiment.
        
        Higher scores indicate stronger sentiment-driven characteristics:
        - Gap up/down patterns
        - High volume surprise moves
        - Correlation with broader market sentiment
        """
        try:
            # Gap analysis (overnight sentiment reactions)
            df['prev_close'] = df['close'].shift(1)
            df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
            
            significant_gaps = (abs(df['gap']) > 0.02).sum()  # >2% gaps
            gap_score = min(significant_gaps / (len(df) * 0.1), 1.0)  # 10% of days
            
            # Volume surprise moves (sentiment-driven volume spikes)
            volume_surprises = ((df['volume_ratio'] > 2.0) & (abs(df['returns']) > 0.03)).sum()
            volume_surprise_score = min(volume_surprises / (len(df) * 0.05), 1.0)
            
            # Intraday volatility (high-low range expansion)
            df['intraday_range'] = (df['high'] - df['low']) / df['open']
            high_range_days = (df['intraday_range'] > df['intraday_range'].quantile(0.8)).sum()
            range_score = high_range_days / len(df.dropna())
            
            # Price momentum after volume spikes (sentiment follow-through)
            momentum_after_spikes = 0
            spike_days = df['volume_ratio'] > 1.8
            
            for idx in df[spike_days].index:
                if idx + 3 < len(df):
                    # Check for continued momentum in next 3 days
                    future_returns = df.loc[idx+1:idx+3, 'returns']
                    if len(future_returns) > 0:
                        avg_return = future_returns.mean()
                        initial_return = df.loc[idx, 'returns']
                        
                        # Same direction momentum
                        if (initial_return > 0 and avg_return > 0) or (initial_return < 0 and avg_return < 0):
                            momentum_after_spikes += 1
            
            momentum_score = momentum_after_spikes / spike_days.sum() if spike_days.sum() > 0 else 0.0
            
            # Combine scores
            sentiment_affinity = (
                gap_score * 0.30 +
                volume_surprise_score * 0.25 +
                range_score * 0.25 +
                momentum_score * 0.20
            )
            
            return float(np.clip(sentiment_affinity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment affinity: {e}")
            return 0.5
    
    def _normalize_affinities(self, affinities: Dict[str, float]) -> Dict[str, float]:
        """Normalize affinity scores to ensure they sum to reasonable values."""
        # Ensure all scores are between 0 and 1
        normalized = {}
        for strategy, score in affinities.items():
            normalized[strategy] = float(np.clip(score, 0.0, 1.0))
        
        return normalized
    
    def _get_default_affinities(self) -> Dict[str, float]:
        """Return default affinity scores for symbols with insufficient data."""
        return {
            'momentum_affinity': 0.5,
            'mean_reversion_affinity': 0.5,
            'breakout_affinity': 0.5,
            'sentiment_affinity': 0.5
        }
    
    async def save_affinities_to_database(self, affinities: Dict[str, Dict[str, float]]) -> int:
        """
        Save calculated affinity scores to the database.
        
        Args:
            affinities: Dictionary mapping symbols to their affinity scores
            
        Returns:
            Number of records saved
        """
        if not affinities:
            logger.warning("No affinities to save")
            return 0
        
        import json
        timestamp = datetime.now(timezone.utc)
        records = []
        
        for symbol, scores in affinities.items():
            # Determine best strategy
            best_strategy = max(scores.items(), key=lambda x: x[1])[0].replace('_affinity', '')
            
            # Create composite scores JSON
            composite_scores = {
                'momentum': scores['momentum_affinity'],
                'mean_reversion': scores['mean_reversion_affinity'],
                'breakout': scores['breakout_affinity'],
                'sentiment': scores['sentiment_affinity'],
                'best_strategy': best_strategy,
                'calculated_at': timestamp.isoformat()
            }
            
            record = {
                'symbol': symbol,
                'timestamp': timestamp,
                'momentum_affinity': scores['momentum_affinity'],
                'mean_reversion_affinity': scores['mean_reversion_affinity'],
                'breakout_affinity': scores['breakout_affinity'],
                'sentiment_affinity': scores['sentiment_affinity'],
                'best_strategy': best_strategy,
                'composite_scores_json': json.dumps(composite_scores)
            }
            records.append(record)
        
        # Bulk insert records
        if records:
            query = """
                INSERT INTO strategy_affinity_scores (
                    symbol, timestamp, momentum_affinity, mean_reversion_affinity,
                    breakout_affinity, sentiment_affinity, best_strategy, composite_scores_json
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8
                )
                ON CONFLICT (symbol, timestamp) 
                DO UPDATE SET
                    momentum_affinity = EXCLUDED.momentum_affinity,
                    mean_reversion_affinity = EXCLUDED.mean_reversion_affinity,
                    breakout_affinity = EXCLUDED.breakout_affinity,
                    sentiment_affinity = EXCLUDED.sentiment_affinity,
                    best_strategy = EXCLUDED.best_strategy,
                    composite_scores_json = EXCLUDED.composite_scores_json
            """
            
            await self.db_adapter.execute_many(query, records)
            logger.info(f"💾 Saved {len(records)} strategy affinity records to database")
            
        return len(records)


async def calculate_and_store_strategy_affinities(symbols: List[str], config=None) -> int:
    """
    Convenience function to calculate and store strategy affinities for a list of symbols.
    
    Args:
        symbols: List of symbols to analyze
        config: Optional config override
        
    Returns:
        Number of records saved
    """
    if config is None:
        config = get_config()
    
    # Import here to avoid circular dependencies
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        calculator = StrategyAffinityCalculator(config, db_adapter)
        affinities = await calculator.calculate_affinities_batch(symbols)
        records_saved = await calculator.save_affinities_to_database(affinities)
        
        logger.info(f"✅ Strategy affinity calculation completed: {records_saved} records saved")
        return records_saved
        
    finally:
        if hasattr(db_adapter, 'close'):
            await db_adapter.close()


if __name__ == "__main__":
    import sys
    
    async def main():
        """Command line interface for strategy affinity calculation."""
        if len(sys.argv) < 2:
            print("Usage: python strategy_affinity_calculator.py <symbol1> [symbol2] ...")
            print("   or: python strategy_affinity_calculator.py --layer1")
            sys.exit(1)
        
        if sys.argv[1] == "--layer1":
            # Calculate for all Layer 1 symbols
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            config = get_config()
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(config)
            
            try:
                # Get Layer 1 qualified symbols
                query = """
                    SELECT symbol FROM companies 
                    WHERE layer >= 1 
                    AND is_active = true
                    ORDER BY liquidity_score DESC
                    LIMIT 500
                """
                rows = await db_adapter.fetch_all(query)
                symbols = [row['symbol'] for row in rows]
                
                print(f"Calculating affinities for {len(symbols)} Layer 1 symbols...")
                records_saved = await calculate_and_store_strategy_affinities(symbols, config)
                print(f"Completed: {records_saved} records saved")
                
            finally:
                await db_adapter.close()
        else:
            # Calculate for specified symbols
            symbols = sys.argv[1:]
            records_saved = await calculate_and_store_strategy_affinities(symbols)
            print(f"Completed: {records_saved} records saved for symbols: {', '.join(symbols)}")
    
    asyncio.run(main())
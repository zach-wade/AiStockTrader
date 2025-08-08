"""
Regime Adaptive Trading Strategy

Dynamically adapts trading behavior based on detected market regimes
(trending, mean-reverting, volatile, calm) using regime detection models.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from enum import Enum

from .base_strategy import BaseStrategy, Signal
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONING = "transitioning"


class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Adaptive strategy that changes behavior based on detected market regime.
    Uses different sub-strategies optimized for each regime type.
    """
    
    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = 'regime_adaptive'
        
        # Load strategy configuration
        strategy_conf = self.config.get('strategies', {}).get(self.name, {})
        
        # Regime detection parameters
        self.regime_lookback = strategy_conf.get('regime_lookback', 20)
        self.regime_threshold = strategy_conf.get('regime_threshold', 0.7)
        self.transition_buffer = strategy_conf.get('transition_buffer', 3)
        
        # Sub-strategy parameters for each regime
        self.regime_strategies = {
            MarketRegime.BULL_TREND: {
                'type': 'momentum',
                'entry_threshold': 0.6,
                'exit_threshold': 0.3,
                'position_size': 0.15,
                'stop_loss': 0.05
            },
            MarketRegime.BEAR_TREND: {
                'type': 'short_momentum',
                'entry_threshold': 0.6,
                'exit_threshold': 0.3,
                'position_size': 0.10,
                'stop_loss': 0.03
            },
            MarketRegime.MEAN_REVERTING: {
                'type': 'mean_reversion',
                'entry_threshold': 2.0,  # Z-score
                'exit_threshold': 0.5,
                'position_size': 0.12,
                'stop_loss': 0.04
            },
            MarketRegime.HIGH_VOLATILITY: {
                'type': 'volatility_breakout',
                'entry_threshold': 0.7,
                'exit_threshold': 0.4,
                'position_size': 0.08,
                'stop_loss': 0.06
            },
            MarketRegime.LOW_VOLATILITY: {
                'type': 'range_trading',
                'entry_threshold': 0.8,
                'exit_threshold': 0.2,
                'position_size': 0.10,
                'stop_loss': 0.02
            }
        }
        
        # Override with config if provided
        custom_regime_strategies = strategy_conf.get('regime_strategies', {})
        for regime, params in custom_regime_strategies.items():
            if regime in self.regime_strategies:
                self.regime_strategies[regime].update(params)
        
        # Regime history for smoothing
        self.regime_history: List[MarketRegime] = []
        self.current_regime: Optional[MarketRegime] = None
        
    def get_required_feature_sets(self) -> List[str]:
        """Specify required feature sets."""
        return ['technical', 'market_regime', 'volatility', 'microstructure']
    
    async def generate_signals(self,
                              symbol: str,
                              features: pd.DataFrame,
                              current_position: Optional[Dict]) -> List[Signal]:
        """
        Generate trading signals based on current market regime.
        """
        if features.empty or len(features) < self.regime_lookback:
            return []
        
        try:
            # Detect current market regime
            regime = self._detect_regime(features)
            if regime is None:
                return []
            
            # Update regime tracking
            self._update_regime_tracking(regime)
            
            # Check if regime is stable or transitioning
            if self._is_regime_transitioning():
                # During transitions, reduce or close positions
                return self._handle_regime_transition(symbol, current_position)
            
            # Generate signals based on current regime
            regime_strategy = self.regime_strategies.get(self.current_regime)
            if not regime_strategy:
                return []
            
            # Apply regime-specific strategy
            signal = self._apply_regime_strategy(
                symbol,
                features,
                self.current_regime,
                regime_strategy,
                current_position
            )
            
            return [signal] if signal else []
            
        except Exception as e:
            logger.error(f"Error in regime adaptive strategy for {symbol}: {e}")
            return []
    
    def _detect_regime(self, features: pd.DataFrame) -> Optional[MarketRegime]:
        """Detect current market regime from features."""
        try:
            latest = features.iloc[-1]
            recent = features.iloc[-self.regime_lookback:]
            
            # Primary regime from features if available
            if 'market_regime' in latest:
                regime_str = latest['market_regime']
                regime_map = {
                    'bullish': MarketRegime.BULL_TREND,
                    'bearish': MarketRegime.BEAR_TREND,
                    'neutral': MarketRegime.MEAN_REVERTING,
                    'volatile': MarketRegime.HIGH_VOLATILITY
                }
                if regime_str in regime_map:
                    return regime_map[regime_str]
            
            # Fallback to manual detection
            return self._detect_regime_manually(recent)
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None
    
    def _detect_regime_manually(self, features: pd.DataFrame) -> MarketRegime:
        """Manually detect regime from price and volatility patterns."""
        # Calculate trend strength
        returns = features['close'].pct_change().dropna()
        cum_return = (1 + returns).cumprod().iloc[-1] - 1
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        avg_volatility = 0.15  # Assume 15% annual vol as baseline
        
        # Calculate efficiency ratio (trending vs choppy)
        price_change = features['close'].iloc[-1] - features['close'].iloc[0]
        path_length = np.abs(features['close'].diff()).sum()
        efficiency_ratio = abs(price_change) / path_length if path_length > 0 else 0
        
        # Determine regime
        if volatility > avg_volatility * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < avg_volatility * 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif efficiency_ratio > 0.3:  # Strong trend
            if cum_return > 0.05:
                return MarketRegime.BULL_TREND
            elif cum_return < -0.05:
                return MarketRegime.BEAR_TREND
        else:
            return MarketRegime.MEAN_REVERTING
    
    def _update_regime_tracking(self, new_regime: MarketRegime):
        """Update regime history and current regime."""
        self.regime_history.append(new_regime)
        
        # Keep only recent history
        if len(self.regime_history) > self.transition_buffer * 2:
            self.regime_history = self.regime_history[-self.transition_buffer * 2:]
        
        # Update current regime if stable
        if len(self.regime_history) >= self.transition_buffer:
            recent_regimes = self.regime_history[-self.transition_buffer:]
            # Check if regime is stable
            if all(r == recent_regimes[0] for r in recent_regimes):
                if self.current_regime != recent_regimes[0]:
                    logger.info(f"Regime change detected: {self.current_regime} -> {recent_regimes[0]}")
                self.current_regime = recent_regimes[0]
    
    def _is_regime_transitioning(self) -> bool:
        """Check if market is transitioning between regimes."""
        if len(self.regime_history) < self.transition_buffer:
            return True
        
        recent_regimes = self.regime_history[-self.transition_buffer:]
        unique_regimes = set(recent_regimes)
        
        return len(unique_regimes) > 1
    
    def _handle_regime_transition(self,
                                 symbol: str,
                                 current_position: Optional[Dict]) -> List[Signal]:
        """Handle regime transitions by reducing or closing positions."""
        if not current_position:
            return []
        
        # During transitions, exit positions with lower confidence
        logger.info(f"Regime transitioning - considering position exit for {symbol}")
        
        exit_confidence = 0.6  # Lower confidence during transitions
        
        if current_position.get('direction') == 'long':
            return [Signal(
                symbol=symbol,
                direction='sell',
                confidence=exit_confidence,
                metadata={
                    'reason': 'regime_transition',
                    'strategy': self.name
                }
            )]
        elif current_position.get('direction') == 'short':
            return [Signal(
                symbol=symbol,
                direction='buy',
                confidence=exit_confidence,
                metadata={
                    'reason': 'regime_transition',
                    'strategy': self.name
                }
            )]
        
        return []
    
    def _apply_regime_strategy(self,
                              symbol: str,
                              features: pd.DataFrame,
                              regime: MarketRegime,
                              strategy_params: Dict[str, Any],
                              current_position: Optional[Dict]) -> Optional[Signal]:
        """Apply regime-specific trading strategy."""
        strategy_type = strategy_params['type']
        latest = features.iloc[-1]
        
        if strategy_type == 'momentum':
            return self._momentum_signal(
                symbol, features, strategy_params, current_position, 'long'
            )
        
        elif strategy_type == 'short_momentum':
            return self._momentum_signal(
                symbol, features, strategy_params, current_position, 'short'
            )
        
        elif strategy_type == 'mean_reversion':
            return self._mean_reversion_signal(
                symbol, features, strategy_params, current_position
            )
        
        elif strategy_type == 'volatility_breakout':
            return self._volatility_breakout_signal(
                symbol, features, strategy_params, current_position
            )
        
        elif strategy_type == 'range_trading':
            return self._range_trading_signal(
                symbol, features, strategy_params, current_position
            )
        
        return None
    
    def _momentum_signal(self,
                        symbol: str,
                        features: pd.DataFrame,
                        params: Dict[str, Any],
                        current_position: Optional[Dict],
                        bias: str = 'long') -> Optional[Signal]:
        """Generate momentum-based signal."""
        latest = features.iloc[-1]
        
        # Calculate momentum score
        momentum_score = 0.0
        momentum_factors = 0
        
        # Price momentum
        if 'momentum_score' in latest:
            momentum_score += latest['momentum_score']
            momentum_factors += 1
        
        # RSI momentum
        if 'rsi_14' in latest:
            if bias == 'long' and latest['rsi_14'] > 50:
                momentum_score += (latest['rsi_14'] - 50) / 50
            elif bias == 'short' and latest['rsi_14'] < 50:
                momentum_score += (50 - latest['rsi_14']) / 50
            momentum_factors += 1
        
        # MACD momentum
        if 'macd_signal' in latest:
            if (bias == 'long' and latest['macd_signal'] > 0) or \
               (bias == 'short' and latest['macd_signal'] < 0):
                momentum_score += abs(latest['macd_signal'])
                momentum_factors += 1
        
        if momentum_factors == 0:
            return None
        
        # Average momentum score
        momentum_score /= momentum_factors
        
        # Generate signal if threshold met
        if momentum_score > params['entry_threshold']:
            direction = 'buy' if bias == 'long' else 'sell'
            
            # Check if already positioned correctly
            if current_position:
                if (current_position.get('direction') == 'long' and direction == 'buy') or \
                   (current_position.get('direction') == 'short' and direction == 'sell'):
                    return None
            
            confidence = min(momentum_score, 1.0)
            
            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                size=params['position_size'],
                metadata={
                    'strategy': self.name,
                    'regime': self.current_regime.value,
                    'sub_strategy': params['type'],
                    'momentum_score': momentum_score,
                    'stop_loss': params['stop_loss']
                }
            )
        
        return None
    
    def _mean_reversion_signal(self,
                              symbol: str,
                              features: pd.DataFrame,
                              params: Dict[str, Any],
                              current_position: Optional[Dict]) -> Optional[Signal]:
        """Generate mean reversion signal."""
        latest = features.iloc[-1]
        recent = features.iloc[-20:]  # 20-period lookback
        
        # Calculate z-score
        current_price = latest['close']
        mean_price = recent['close'].mean()
        std_price = recent['close'].std()
        
        if std_price == 0:
            return None
        
        z_score = (current_price - mean_price) / std_price
        
        # Generate signal based on z-score
        if abs(z_score) > params['entry_threshold']:
            # Overbought/oversold
            direction = 'sell' if z_score > 0 else 'buy'
            
            # Check current position
            if current_position:
                if (current_position.get('direction') == 'long' and direction == 'buy') or \
                   (current_position.get('direction') == 'short' and direction == 'sell'):
                    return None
            
            confidence = min(abs(z_score) / 3.0, 1.0)  # Scale confidence
            
            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                size=params['position_size'],
                metadata={
                    'strategy': self.name,
                    'regime': self.current_regime.value,
                    'sub_strategy': params['type'],
                    'z_score': z_score,
                    'stop_loss': params['stop_loss']
                }
            )
        
        # Exit signal if z-score reverts
        elif current_position and abs(z_score) < params['exit_threshold']:
            direction = 'sell' if current_position.get('direction') == 'long' else 'buy'
            
            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=0.8,
                metadata={
                    'strategy': self.name,
                    'reason': 'mean_reversion_exit',
                    'z_score': z_score
                }
            )
        
        return None
    
    def _volatility_breakout_signal(self,
                                   symbol: str,
                                   features: pd.DataFrame,
                                   params: Dict[str, Any],
                                   current_position: Optional[Dict]) -> Optional[Signal]:
        """Generate volatility breakout signal."""
        latest = features.iloc[-1]
        
        # Check for volatility expansion
        if 'atr_14' in latest and 'atr_14' in features.columns:
            recent_atr = features['atr_14'].iloc[-20:].mean()
            current_atr = latest['atr_14']
            
            if current_atr > recent_atr * 1.5:  # Volatility expansion
                # Look for directional breakout
                if 'bollinger_upper' in latest and latest['close'] > latest['bollinger_upper']:
                    direction = 'buy'
                elif 'bollinger_lower' in latest and latest['close'] < latest['bollinger_lower']:
                    direction = 'sell'
                else:
                    return None
                
                # Check position
                if current_position:
                    if (current_position.get('direction') == 'long' and direction == 'buy') or \
                       (current_position.get('direction') == 'short' and direction == 'sell'):
                        return None
                
                confidence = params['entry_threshold']
                
                return Signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    size=params['position_size'],
                    metadata={
                        'strategy': self.name,
                        'regime': self.current_regime.value,
                        'sub_strategy': params['type'],
                        'atr_ratio': current_atr / recent_atr,
                        'stop_loss': params['stop_loss']
                    }
                )
        
        return None
    
    def _range_trading_signal(self,
                             symbol: str,
                             features: pd.DataFrame,
                             params: Dict[str, Any],
                             current_position: Optional[Dict]) -> Optional[Signal]:
        """Generate range trading signal for low volatility regimes."""
        latest = features.iloc[-1]
        recent = features.iloc[-20:]
        
        # Identify range boundaries
        resistance = recent['high'].max()
        support = recent['low'].min()
        current_price = latest['close']
        
        range_size = resistance - support
        if range_size == 0:
            return None
        
        # Position within range
        position_in_range = (current_price - support) / range_size
        
        # Generate signals at range extremes
        if position_in_range > 0.8:  # Near resistance
            direction = 'sell'
        elif position_in_range < 0.2:  # Near support
            direction = 'buy'
        else:
            return None
        
        # Check current position
        if current_position:
            if (current_position.get('direction') == 'long' and direction == 'buy') or \
               (current_position.get('direction') == 'short' and direction == 'sell'):
                return None
        
        confidence = params['entry_threshold']
        
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            size=params['position_size'],
            metadata={
                'strategy': self.name,
                'regime': self.current_regime.value,
                'sub_strategy': params['type'],
                'range_position': position_in_range,
                'support': support,
                'resistance': resistance,
                'stop_loss': params['stop_loss']
            }
        )
    
    def get_position_size(self, signal: Signal, features: pd.DataFrame) -> float:
        """Get position size based on regime and volatility."""
        if signal.size is not None:
            # Adjust for current volatility
            if 'volatility_zscore' in features.iloc[-1]:
                vol_z = features.iloc[-1]['volatility_zscore']
                if abs(vol_z) > 2:
                    # High volatility - reduce size
                    return signal.size * 0.7
                elif abs(vol_z) < 0.5:
                    # Low volatility - can increase size slightly
                    return signal.size * 1.1
            
            return signal.size
        
        # Default sizing
        return 0.1
"""
Volume Indicators Calculator

Specialized calculator for volume-based technical indicators including:
- OBV (On Balance Volume)
- A/D Line (Accumulation/Distribution)
- Chaikin Money Flow
- Volume ROC (Rate of Change)
- PVT (Price Volume Trend)
- VWAP (Volume Weighted Average Price)
- Volume Profile indicators
- Force Index
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from .base_technical import BaseTechnicalCalculator


class VolumeIndicatorsCalculator(BaseTechnicalCalculator):
    """Calculator for volume-based indicators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize volume indicators calculator."""
        super().__init__(config)
        
        # OBV parameters
        self.obv_momentum_period = self.config.get('obv_momentum_period', 20)
        
        # A/D Line parameters
        self.ad_momentum_period = self.config.get('ad_momentum_period', 20)
        
        # Chaikin Money Flow parameters
        self.cmf_fast_period = self.config.get('cmf_fast_period', 3)
        self.cmf_slow_period = self.config.get('cmf_slow_period', 10)
        
        # Volume ROC parameters
        self.volume_roc_period = self.config.get('volume_roc_period', 10)
        
        # Volume Profile parameters
        self.volume_ma_periods = self.config.get('volume_ma_periods', [20, 50])
        
        # Force Index parameters
        self.force_index_ema_period = self.config.get('force_index_ema_period', 13)
    
    def get_feature_names(self) -> List[str]:
        """Return list of volume indicator feature names."""
        feature_names = []
        
        # Volume indicators
        feature_names.extend([
            'obv', 'obv_momentum', 'ad_line', 'ad_momentum', 'chaikin_money_flow',
            'volume_roc', 'pvt', 'vwap', 'price_to_vwap'
        ])
        
        # Volume Profile indicators for multiple periods
        for period in self.volume_ma_periods:
            feature_names.extend([
                f'volume_ma_{period}', f'relative_volume_{period}'
            ])
        
        # Force Index
        feature_names.extend([
            'force_index', 'force_index_ema'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicator features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_feature_dataframe(data.index)
            
            # Calculate volume indicators for each symbol
            for symbol in data['symbol'].unique():
                mask = data['symbol'] == symbol
                symbol_data = data[mask].copy()
                
                if len(symbol_data) < 50:
                    continue
                
                # Calculate OBV indicators
                features = self._calculate_obv_indicators(features, symbol_data, mask)
                
                # Calculate A/D Line indicators
                features = self._calculate_ad_indicators(features, symbol_data, mask)
                
                # Calculate Chaikin Money Flow
                features = self._calculate_cmf_indicators(features, symbol_data, mask)
                
                # Calculate Volume ROC and PVT
                features = self._calculate_volume_momentum(features, symbol_data, mask)
                
                # Calculate VWAP indicators
                features = self._calculate_vwap_indicators(features, symbol_data, mask)
                
                # Calculate Volume Profile indicators
                features = self._calculate_volume_profile(features, symbol_data, mask)
                
                # Calculate Force Index
                features = self._calculate_force_index(features, symbol_data, mask)
            
            return features
            
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            return self.create_feature_dataframe(data.index)
    
    def _calculate_obv_indicators(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate OBV indicators."""
        # OBV (On Balance Volume)
        obv = talib.OBV(
            symbol_data['close'].values,
            symbol_data['volume'].values
        )
        features.loc[mask, 'obv'] = obv
        
        # OBV momentum
        obv_series = pd.Series(obv, index=symbol_data.index)
        features.loc[mask, 'obv_momentum'] = obv_series.pct_change(self.obv_momentum_period)
        
        return features
    
    def _calculate_ad_indicators(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate A/D Line indicators."""
        # AD (Accumulation/Distribution)
        ad = talib.AD(
            symbol_data['high'].values,
            symbol_data['low'].values,
            symbol_data['close'].values,
            symbol_data['volume'].values
        )
        features.loc[mask, 'ad_line'] = ad
        
        # A/D momentum
        ad_series = pd.Series(ad, index=symbol_data.index)
        features.loc[mask, 'ad_momentum'] = ad_series.pct_change(self.ad_momentum_period)
        
        return features
    
    def _calculate_cmf_indicators(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate Chaikin Money Flow indicators."""
        # CMF (Chaikin Money Flow)
        adl = talib.ADOSC(
            symbol_data['high'].values,
            symbol_data['low'].values,
            symbol_data['close'].values,
            symbol_data['volume'].values,
            fastperiod=self.cmf_fast_period,
            slowperiod=self.cmf_slow_period
        )
        features.loc[mask, 'chaikin_money_flow'] = adl
        
        return features
    
    def _calculate_volume_momentum(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate Volume ROC and PVT indicators."""
        # Volume Rate of Change
        volume_series = pd.Series(symbol_data['volume'].values, index=symbol_data.index)
        vroc = volume_series.pct_change(self.volume_roc_period)
        features.loc[mask, 'volume_roc'] = vroc
        
        # Price Volume Trend
        price_change_pct = symbol_data['close'].pct_change()
        pvt = (price_change_pct * symbol_data['volume']).cumsum()
        features.loc[mask, 'pvt'] = pvt
        
        return features
    
    def _calculate_vwap_indicators(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate VWAP indicators."""
        # Volume-Weighted Average Price (VWAP)
        typical_price = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
        vwap = (typical_price * symbol_data['volume']).cumsum() / symbol_data['volume'].cumsum()
        features.loc[mask, 'vwap'] = vwap
        
        # Price to VWAP ratio
        price_to_vwap = (symbol_data['close'] / vwap).fillna(1.0)
        features.loc[mask, 'price_to_vwap'] = price_to_vwap
        
        return features
    
    def _calculate_volume_profile(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate Volume Profile indicators."""
        # Volume Profile for different periods
        for period in self.volume_ma_periods:
            volume_ma = symbol_data['volume'].rolling(period, min_periods=1).mean()
            features.loc[mask, f'volume_ma_{period}'] = volume_ma
            
            # Relative volume (current volume vs average volume)
            relative_volume = (symbol_data['volume'] / volume_ma).fillna(1.0)
            features.loc[mask, f'relative_volume_{period}'] = relative_volume
        
        return features
    
    def _calculate_force_index(self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Calculate Force Index indicators."""
        # Force Index (price change * volume)
        price_change = symbol_data['close'].diff()
        force_index = price_change * symbol_data['volume']
        features.loc[mask, 'force_index'] = force_index
        
        # Force Index EMA
        force_index_ema = force_index.ewm(span=self.force_index_ema_period).mean()
        features.loc[mask, 'force_index_ema'] = force_index_ema
        
        return features
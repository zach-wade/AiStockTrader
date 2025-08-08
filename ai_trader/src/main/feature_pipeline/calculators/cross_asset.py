"""
Cross-Asset Correlation Calculator

Calculates features based on relationships between different assets including:
- Inter-market correlations (stocks, bonds, commodities, currencies)
- Sector rotation indicators
- Risk-on/Risk-off asset relationships
- Lead-lag relationships
- Pair trading opportunities
- Market divergence signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA

from .base_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetConfig:
    """Configuration for cross-asset calculations"""
    # Correlation windows
    corr_windows: List[int] = None
    
    # Market indices to track
    market_indices: List[str] = None
    
    # Sector ETFs
    sector_etfs: List[str] = None
    
    # Safe haven assets
    safe_havens: List[str] = None
    
    # Risk assets
    risk_assets: List[str] = None
    
    # Lead-lag analysis
    lag_periods: List[int] = None
    
    # PCA components
    n_components: int = 5
    
    def __post_init__(self):
        if self.corr_windows is None:
            self.corr_windows = [20, 60, 120]
        if self.market_indices is None:
            self.market_indices = ['SPY', 'QQQ', 'IWM', 'DIA']
        if self.sector_etfs is None:
            self.sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU']
        if self.safe_havens is None:
            self.safe_havens = ['GLD', 'TLT', 'UUP', 'VIX']
        if self.risk_assets is None:
            self.risk_assets = ['HYG', 'EEM', 'ARKK', 'IWM']
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 5, 10]


class CrossAssetCalculator(BaseFeatureCalculator):
    """Calculator for cross-asset correlation features"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.cross_config = CrossAssetConfig(
            **config.get('cross_asset', {}) if config else {}
        )
        self.market_data = {}  # Cache for other asset data
    
    def set_market_data(self, symbol: str, data: pd.DataFrame):
        """Add market data for cross-asset calculations"""
        self.market_data[symbol] = data
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-asset features.
        
        Args:
            data: DataFrame with target asset OHLCV data
            
        Returns:
            DataFrame with cross-asset features
        """
        try:
            # Validate inputs
            if not self.validate_inputs(data):
                logger.error("Invalid input data for cross-asset calculation")
                return self._create_empty_features(data.index if not data.empty else pd.DatetimeIndex([]))
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            if processed_data.empty:
                logger.error("Data preprocessing failed for cross-asset analysis")
                return self._create_empty_features(data.index if not data.empty else pd.DatetimeIndex([]))
            
            features = pd.DataFrame(index=processed_data.index)
            feature_count = 0
            
            # Market beta and correlations
            try:
                features = self._add_market_correlations(processed_data, features)
                market_count = len(self.cross_config.market_indices) * (len(self.cross_config.corr_windows) * 2 + 1) + 2
                feature_count += market_count
                logger.debug(f"Added {market_count} market correlation features")
            except Exception as e:
                logger.warning(f"Failed to add market correlation features: {e}")
            
            # Sector exposures
            try:
                features = self._add_sector_analysis(processed_data, features)
                sector_count = len(self.cross_config.sector_etfs) + 6  # correlations + aggregates
                feature_count += sector_count
                logger.debug(f"Added {sector_count} sector analysis features")
            except Exception as e:
                logger.warning(f"Failed to add sector analysis features: {e}")
            
            # Risk on/off indicators
            try:
                features = self._add_risk_sentiment(processed_data, features)
                feature_count += 7  # risk sentiment features
                logger.debug("Added risk sentiment features")
            except Exception as e:
                logger.warning(f"Failed to add risk sentiment features: {e}")
            
            # Inter-market relationships
            try:
                features = self._add_intermarket_features(processed_data, features)
                feature_count += 5  # inter-market features
                logger.debug("Added inter-market relationship features")
            except Exception as e:
                logger.warning(f"Failed to add inter-market features: {e}")
            
            # Lead-lag relationships
            try:
                features = self._add_lead_lag_features(processed_data, features)
                feature_count += 7  # lead-lag features
                logger.debug("Added lead-lag relationship features")
            except Exception as e:
                logger.warning(f"Failed to add lead-lag features: {e}")
            
            # Market divergence
            try:
                features = self._add_divergence_features(processed_data, features)
                feature_count += 4  # divergence features
                logger.debug("Added market divergence features")
            except Exception as e:
                logger.warning(f"Failed to add divergence features: {e}")
            
            # Principal components
            if len(self.market_data) >= self.cross_config.n_components:
                try:
                    features = self._add_pca_features(processed_data, features)
                    pca_count = self.cross_config.n_components + 2  # components + explained variance + loading
                    feature_count += pca_count
                    logger.debug(f"Added {pca_count} PCA features")
                except Exception as e:
                    logger.warning(f"Failed to add PCA features: {e}")
            else:
                logger.debug(f"Insufficient market data for PCA: {len(self.market_data)} < {self.cross_config.n_components}")
            
            # Pair trading opportunities
            try:
                features = self._add_pair_trading_features(processed_data, features)
                feature_count += 8  # pair trading features
                logger.debug("Added pair trading features")
            except Exception as e:
                logger.warning(f"Failed to add pair trading features: {e}")
            
            # Postprocess features
            features = self.postprocess_features(features)
            
            if features.empty:
                logger.error("All cross-asset feature calculations failed")
                return self._create_empty_features(data.index if not data.empty else pd.DatetimeIndex([]))
            
            logger.info(f"Successfully calculated {len(features.columns)} cross-asset features")
            
        except Exception as e:
            logger.error(f"Critical error in cross-asset feature calculation: {e}")
            return self._create_empty_features(data.index if not data.empty else pd.DatetimeIndex([]))
            
        return features
    
    def _create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """Create empty feature DataFrame with proper column names"""
        try:
            feature_names = self.get_feature_names()
            empty_features = pd.DataFrame(
                data=0,  # Fill with zeros instead of NaN
                index=index,
                columns=feature_names
            )
            logger.warning(f"Created empty cross-asset features DataFrame with {len(feature_names)} columns")
            return empty_features
        except Exception as e:
            logger.error(f"Failed to create empty cross-asset features DataFrame: {e}")
            return pd.DataFrame(index=index)
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required = ['close']
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
            
        if len(data) < max(self.cross_config.corr_windows):
            logger.warning(f"Insufficient data for cross-asset calculation")
            return False
            
        return True
    
    def get_required_columns(self) -> List[str]:
        """Return list of required input columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_feature_names(self) -> List[str]:
        """Return list of all cross-asset feature names"""
        feature_names = []
        
        # Market correlations and betas
        for index_symbol in self.cross_config.market_indices:
            feature_names.append(f'rel_strength_{index_symbol}')
            for window in self.cross_config.corr_windows:
                feature_names.extend([
                    f'corr_{index_symbol}_{window}d',
                    f'beta_{index_symbol}_{window}d'
                ])
        
        # Market correlation aggregates
        feature_names.extend(['avg_market_correlation', 'market_correlation_dispersion'])
        
        # Sector analysis
        for sector in self.cross_config.sector_etfs:
            feature_names.append(f'sector_corr_{sector}')
        feature_names.extend([
            'dominant_sector', 'max_sector_corr', 'sector_dispersion',
            'sector_rotation_score', 'tech_financial_spread', 'defensive_cyclical_spread'
        ])
        
        # Risk sentiment
        feature_names.extend([
            'safe_haven_correlation', 'risk_asset_correlation', 'risk_sentiment_score',
            'risk_on', 'risk_off', 'vix_correlation', 'fear_gauge'
        ])
        
        # Inter-market relationships
        feature_names.extend([
            'bond_stock_correlation', 'flight_to_quality', 'dollar_correlation',
            'commodity_sensitivity', 'emerging_markets_corr'
        ])
        
        # Lead-lag relationships
        feature_names.extend([
            'best_leader', 'leader_lag', 'leader_correlation', 'leader_signal',
            'best_follower', 'follower_lag', 'follower_correlation'
        ])
        
        # Divergence features
        feature_names.extend([
            'return_divergence', 'positive_divergence', 'negative_divergence', 'breadth_divergence'
        ])
        
        # PCA features
        for i in range(1, self.cross_config.n_components + 1):
            feature_names.append(f'market_pc_{i}')
        feature_names.extend(['pca_explained_variance', 'market_factor_loading'])
        
        # Pair trading features
        feature_names.extend([
            'best_pair_symbol', 'best_pair_correlation', 'pair_spread', 'pair_spread_zscore',
            'pair_trade_long', 'pair_trade_short', 'pair_cointegration_pvalue', 'is_cointegrated'
        ])
        
        return feature_names
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for cross-asset calculation"""
        try:
            if data.empty:
                logger.warning("Empty data provided to cross-asset calculator")
                return data
            
            # Ensure required columns exist
            if 'close' not in data.columns:
                logger.error("Missing required 'close' column for cross-asset analysis")
                return pd.DataFrame()
            
            # Clean the data
            processed_data = data.copy()
            
            # Handle missing values in close prices
            processed_data['close'] = processed_data['close'].fillna(method='ffill')
            
            # Remove any remaining NaN rows
            processed_data = processed_data.dropna(subset=['close'])
            
            # Ensure we have enough data for cross-asset analysis
            min_required = max(self.cross_config.corr_windows) + 10
            if len(processed_data) < min_required:
                logger.warning(f"Insufficient data for cross-asset analysis: {len(processed_data)} < {min_required}")
                return pd.DataFrame()
            
            # Validate price data
            if (processed_data['close'] <= 0).any():
                logger.warning("Found non-positive prices, filtering out")
                processed_data = processed_data[processed_data['close'] > 0]
            
            # Handle other OHLC columns if present
            ohlc_cols = ['open', 'high', 'low']
            available_ohlc = [col for col in ohlc_cols if col in processed_data.columns]
            
            if available_ohlc:
                processed_data[available_ohlc] = processed_data[available_ohlc].fillna(method='ffill')
                # Ensure OHLC consistency and positive prices
                for col in available_ohlc:
                    processed_data[col] = processed_data[col].clip(lower=0.01)
            
            # Handle volume if present
            if 'volume' in processed_data.columns:
                processed_data['volume'] = processed_data['volume'].fillna(0)
                processed_data['volume'] = processed_data['volume'].clip(lower=0)
            
            # Clean market data cache if needed
            cleaned_market_data = {}
            for symbol, market_df in self.market_data.items():
                if market_df is not None and not market_df.empty and 'close' in market_df.columns:
                    # Clean market data similarly
                    clean_market = market_df.copy()
                    clean_market['close'] = clean_market['close'].fillna(method='ffill')
                    clean_market = clean_market.dropna(subset=['close'])
                    
                    if len(clean_market) > 10:  # Minimum data requirement
                        cleaned_market_data[symbol] = clean_market
                    else:
                        logger.debug(f"Insufficient data for {symbol}, excluding from cross-asset analysis")
                else:
                    logger.debug(f"Invalid market data for {symbol}, excluding from analysis")
            
            self.market_data = cleaned_market_data
            
            logger.info(f"Preprocessed cross-asset data: {len(processed_data)} rows, {len(self.market_data)} market assets")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data for cross-asset analysis: {e}")
            return pd.DataFrame()
    
    def postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Postprocess cross-asset features"""
        try:
            if features.empty:
                logger.warning("No cross-asset features to postprocess")
                return features
            
            processed_features = features.copy()
            
            # Handle infinite values
            processed_features = processed_features.replace([np.inf, -np.inf], np.nan)
            
            # Handle specific feature types
            
            # Correlation values should be in [-1, 1]
            corr_cols = [col for col in processed_features.columns if 'corr' in col and 'pvalue' not in col]
            for col in corr_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(-1, 1)
            
            # Beta values should be reasonable (clip extreme values)
            beta_cols = [col for col in processed_features.columns if col.startswith('beta_')]
            for col in beta_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(-10, 10)
            
            # P-values should be in [0, 1]
            pvalue_cols = [col for col in processed_features.columns if 'pvalue' in col]
            for col in pvalue_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(0, 1)
            
            # Z-scores should be reasonable
            zscore_cols = [col for col in processed_features.columns if 'zscore' in col]
            for col in zscore_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(-10, 10)
            
            # Relative strength ratios should be positive and reasonable
            rel_strength_cols = [col for col in processed_features.columns if col.startswith('rel_strength_')]
            for col in rel_strength_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(0.1, 10)
            
            # Spread values should be reasonable
            spread_cols = [col for col in processed_features.columns if 'spread' in col and 'zscore' not in col]
            for col in spread_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(-5, 5)
            
            # PCA explained variance should be in [0, 1]
            if 'pca_explained_variance' in processed_features.columns:
                processed_features['pca_explained_variance'] = processed_features['pca_explained_variance'].clip(0, 1)
            
            # Principal components should be standardized (reasonable bounds)
            pc_cols = [col for col in processed_features.columns if col.startswith('market_pc_')]
            for col in pc_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(-5, 5)
            
            # Market factor loading should be reasonable
            if 'market_factor_loading' in processed_features.columns:
                processed_features['market_factor_loading'] = processed_features['market_factor_loading'].clip(-2, 2)
            
            # Handle categorical features (convert to numeric if needed)
            categorical_cols = ['dominant_sector', 'best_leader', 'best_follower', 'best_pair_symbol']
            for col in categorical_cols:
                if col in processed_features.columns:
                    # Convert to string and then to category codes for numerical processing
                    try:
                        processed_features[col] = pd.Categorical(processed_features[col].astype(str)).codes
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning(f"Failed to convert column {col} to categorical codes: {e}")
                        processed_features[col] = 0  # Default if conversion fails
            
            # Ensure binary indicators are 0 or 1
            binary_cols = [
                'risk_on', 'risk_off', 'flight_to_quality', 'positive_divergence',
                'negative_divergence', 'breadth_divergence', 'pair_trade_long',
                'pair_trade_short', 'is_cointegrated'
            ]
            for col in binary_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].fillna(0).astype(int).clip(0, 1)
            
            # Forward fill NaN values for continuity
            processed_features = processed_features.fillna(method='ffill')
            
            # Fill any remaining NaN with appropriate defaults
            # Correlations default to 0 (no correlation)
            corr_default_cols = corr_cols + ['avg_market_correlation', 'market_correlation_dispersion']
            for col in corr_default_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].fillna(0)
            
            # Other features default to 0
            processed_features = processed_features.fillna(0)
            
            # Final validation
            if processed_features.isnull().any().any():
                logger.warning("Some NaN values remain in cross-asset features")
            
            if np.isinf(processed_features.values).any():
                logger.warning("Some infinite values remain in cross-asset features")
                processed_features = processed_features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Postprocessed {len(processed_features.columns)} cross-asset features")
            return processed_features
            
        except Exception as e:
            logger.error(f"Error postprocessing cross-asset features: {e}")
            return features  # Return original features if postprocessing fails
    
    def _add_market_correlations(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add correlations with major market indices"""
        returns = data['close'].pct_change()
        
        for index_symbol in self.cross_config.market_indices:
            if index_symbol in self.market_data:
                index_data = self.market_data[index_symbol]
                index_returns = index_data['close'].pct_change()
                
                # Align data
                aligned_returns, aligned_index = returns.align(index_returns, join='inner')
                
                for window in self.cross_config.corr_windows:
                    # Rolling correlation
                    corr = aligned_returns.rolling(window).corr(aligned_index)
                    features[f'corr_{index_symbol}_{window}d'] = corr.reindex(data.index)
                    
                    # Rolling beta
                    covariance = aligned_returns.rolling(window).cov(aligned_index)
                    index_variance = aligned_index.rolling(window).var()
                    beta = covariance / index_variance
                    features[f'beta_{index_symbol}_{window}d'] = beta.reindex(data.index)
                
                # Relative strength
                cum_returns = (1 + returns).cumprod()
                cum_index_returns = (1 + index_returns).cumprod()
                features[f'rel_strength_{index_symbol}'] = (
                    cum_returns / cum_index_returns
                ).reindex(data.index)
        
        # Average market correlation
        corr_cols = [col for col in features.columns if col.startswith('corr_')]
        if corr_cols:
            features['avg_market_correlation'] = features[corr_cols].mean(axis=1)
            features['market_correlation_dispersion'] = features[corr_cols].std(axis=1)
        
        return features
    
    def _add_sector_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector rotation and exposure features"""
        returns = data['close'].pct_change()
        
        sector_correlations = {}
        
        for sector in self.cross_config.sector_etfs:
            if sector in self.market_data:
                sector_data = self.market_data[sector]
                sector_returns = sector_data['close'].pct_change()
                
                # Align data
                aligned_returns, aligned_sector = returns.align(sector_returns, join='inner')
                
                # Calculate correlation
                corr = aligned_returns.rolling(60).corr(aligned_sector)
                sector_correlations[sector] = corr.reindex(data.index)
                features[f'sector_corr_{sector}'] = sector_correlations[sector]
        
        if sector_correlations:
            # Find dominant sector
            sector_df = pd.DataFrame(sector_correlations)
            features['dominant_sector'] = sector_df.idxmax(axis=1)
            features['max_sector_corr'] = sector_df.max(axis=1)
            features['sector_dispersion'] = sector_df.std(axis=1)
            
            # Sector rotation score
            features['sector_rotation_score'] = features['sector_dispersion'] * features['max_sector_corr']
            
            # Technology vs Financials spread (risk indicator)
            if 'XLK' in sector_correlations and 'XLF' in sector_correlations:
                features['tech_financial_spread'] = (
                    sector_correlations['XLK'] - sector_correlations['XLF']
                )
            
            # Defensive vs Cyclical
            defensive = ['XLP', 'XLU', 'XLV']  # Staples, Utilities, Healthcare
            cyclical = ['XLY', 'XLF', 'XLI']  # Discretionary, Financials, Industrials
            
            defensive_corr = pd.DataFrame({s: sector_correlations.get(s, 0) for s in defensive}).mean(axis=1)
            cyclical_corr = pd.DataFrame({s: sector_correlations.get(s, 0) for s in cyclical}).mean(axis=1)
            
            features['defensive_cyclical_spread'] = defensive_corr - cyclical_corr
        
        return features
    
    def _add_risk_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add risk-on/risk-off sentiment indicators"""
        returns = data['close'].pct_change()
        
        # Safe haven correlations
        safe_haven_corrs = []
        for asset in self.cross_config.safe_havens:
            if asset in self.market_data:
                asset_returns = self.market_data[asset]['close'].pct_change()
                aligned_returns, aligned_asset = returns.align(asset_returns, join='inner')
                corr = aligned_returns.rolling(20).corr(aligned_asset)
                safe_haven_corrs.append(corr.reindex(data.index))
        
        if safe_haven_corrs:
            features['safe_haven_correlation'] = pd.concat(safe_haven_corrs, axis=1).mean(axis=1)
        
        # Risk asset correlations
        risk_asset_corrs = []
        for asset in self.cross_config.risk_assets:
            if asset in self.market_data:
                asset_returns = self.market_data[asset]['close'].pct_change()
                aligned_returns, aligned_asset = returns.align(asset_returns, join='inner')
                corr = aligned_returns.rolling(20).corr(aligned_asset)
                risk_asset_corrs.append(corr.reindex(data.index))
        
        if risk_asset_corrs:
            features['risk_asset_correlation'] = pd.concat(risk_asset_corrs, axis=1).mean(axis=1)
        
        # Risk sentiment score
        if 'safe_haven_correlation' in features and 'risk_asset_correlation' in features:
            features['risk_sentiment_score'] = (
                features['risk_asset_correlation'] - features['safe_haven_correlation']
            )
            features['risk_on'] = (features['risk_sentiment_score'] > 0.3).astype(int)
            features['risk_off'] = (features['risk_sentiment_score'] < -0.3).astype(int)
        
        # VIX correlation if available
        if 'VIX' in self.market_data:
            vix_returns = self.market_data['VIX']['close'].pct_change()
            aligned_returns, aligned_vix = returns.align(vix_returns, join='inner')
            features['vix_correlation'] = aligned_returns.rolling(20).corr(aligned_vix).reindex(data.index)
            features['fear_gauge'] = features['vix_correlation'] * -1  # Negative correlation is fearful
        
        return features
    
    def _add_intermarket_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add inter-market relationship features"""
        # Bond-Stock correlation
        if 'TLT' in self.market_data and 'SPY' in self.market_data:
            tlt_returns = self.market_data['TLT']['close'].pct_change()
            spy_returns = self.market_data['SPY']['close'].pct_change()
            aligned_tlt, aligned_spy = tlt_returns.align(spy_returns, join='inner')
            bond_stock_corr = aligned_tlt.rolling(60).corr(aligned_spy)
            features['bond_stock_correlation'] = bond_stock_corr.reindex(data.index)
            
            # Flight to quality indicator
            features['flight_to_quality'] = (
                (bond_stock_corr < -0.3) & (aligned_spy.rolling(5).mean() < 0)
            ).astype(int).reindex(data.index)
        
        # Dollar strength impact
        if 'UUP' in self.market_data:
            dollar_returns = self.market_data['UUP']['close'].pct_change()
            returns = data['close'].pct_change()
            aligned_returns, aligned_dollar = returns.align(dollar_returns, join='inner')
            features['dollar_correlation'] = aligned_returns.rolling(20).corr(aligned_dollar).reindex(data.index)
        
        # Commodity correlation
        commodity_symbols = ['GLD', 'USO', 'DBA']  # Gold, Oil, Agriculture
        commodity_corrs = []
        
        for commodity in commodity_symbols:
            if commodity in self.market_data:
                commodity_returns = self.market_data[commodity]['close'].pct_change()
                returns = data['close'].pct_change()
                aligned_returns, aligned_commodity = returns.align(commodity_returns, join='inner')
                corr = aligned_returns.rolling(20).corr(aligned_commodity)
                commodity_corrs.append(corr.reindex(data.index))
        
        if commodity_corrs:
            features['commodity_sensitivity'] = pd.concat(commodity_corrs, axis=1).mean(axis=1)
        
        # Emerging markets correlation
        if 'EEM' in self.market_data:
            eem_returns = self.market_data['EEM']['close'].pct_change()
            returns = data['close'].pct_change()
            aligned_returns, aligned_eem = returns.align(eem_returns, join='inner')
            features['emerging_markets_corr'] = aligned_returns.rolling(20).corr(aligned_eem).reindex(data.index)
        
        return features
    
    def _add_lead_lag_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add lead-lag relationship features"""
        returns = data['close'].pct_change()
        
        # Check which assets lead/lag
        lead_lag_scores = {}
        
        for symbol in self.market_data:
            other_returns = self.market_data[symbol]['close'].pct_change()
            aligned_returns, aligned_other = returns.align(other_returns, join='inner')
            
            # Calculate cross-correlations at different lags
            max_corr = 0
            best_lag = 0
            
            for lag in self.cross_config.lag_periods:
                # Positive lag: other leads
                corr_pos = aligned_returns.rolling(20).corr(aligned_other.shift(lag))
                # Negative lag: we lead
                corr_neg = aligned_returns.rolling(20).corr(aligned_other.shift(-lag))
                
                if abs(corr_pos.mean()) > abs(max_corr):
                    max_corr = corr_pos.mean()
                    best_lag = lag
                    
                if abs(corr_neg.mean()) > abs(max_corr):
                    max_corr = corr_neg.mean()
                    best_lag = -lag
            
            if abs(max_corr) > 0.3:  # Significant correlation
                lead_lag_scores[symbol] = (best_lag, max_corr)
        
        # Find strongest leader and follower
        if lead_lag_scores:
            leaders = [(s, lag, corr) for s, (lag, corr) in lead_lag_scores.items() if lag > 0]
            followers = [(s, lag, corr) for s, (lag, corr) in lead_lag_scores.items() if lag < 0]
            
            if leaders:
                best_leader = max(leaders, key=lambda x: abs(x[2]))
                features['best_leader'] = best_leader[0]
                features['leader_lag'] = best_leader[1]
                features['leader_correlation'] = best_leader[2]
                
                # Get leading indicator signal
                leader_returns = self.market_data[best_leader[0]]['close'].pct_change()
                features['leader_signal'] = leader_returns.shift(best_leader[1]).reindex(data.index)
            
            if followers:
                best_follower = max(followers, key=lambda x: abs(x[2]))
                features['best_follower'] = best_follower[0]
                features['follower_lag'] = abs(best_follower[1])
                features['follower_correlation'] = best_follower[2]
        
        return features
    
    def _add_divergence_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market divergence features"""
        close = data['close']
        returns = close.pct_change()
        
        # Price vs Market divergence
        if 'SPY' in self.market_data:
            spy_close = self.market_data['SPY']['close']
            spy_returns = spy_close.pct_change()
            
            # Cumulative returns
            cum_returns = (1 + returns).cumprod()
            cum_spy_returns = (1 + spy_returns).cumprod()
            
            # Aligned data
            aligned_cum, aligned_spy_cum = cum_returns.align(cum_spy_returns, join='inner')
            
            # Divergence score
            rolling_return = aligned_cum.pct_change(20)
            rolling_spy_return = aligned_spy_cum.pct_change(20)
            
            features['return_divergence'] = (rolling_return - rolling_spy_return).reindex(data.index)
            
            # Divergence direction
            features['positive_divergence'] = (
                (rolling_return > 0) & (rolling_spy_return < 0)
            ).astype(int).reindex(data.index)
            
            features['negative_divergence'] = (
                (rolling_return < 0) & (rolling_spy_return > 0)
            ).astype(int).reindex(data.index)
        
        # Breadth divergence
        if len(self.market_data) > 10:
            # Calculate market breadth
            advances = 0
            for symbol in list(self.market_data.keys())[:50]:  # Limit to 50 symbols
                symbol_returns = self.market_data[symbol]['close'].pct_change()
                aligned_returns, aligned_symbol = returns.align(symbol_returns, join='inner')
                advances += (aligned_symbol > 0).astype(int)
            
            breadth_pct = advances / min(len(self.market_data), 50)
            
            # Breadth vs Price divergence
            price_direction = (returns.rolling(5).mean() > 0).astype(int)
            breadth_direction = (breadth_pct > 0.5).astype(int)
            
            features['breadth_divergence'] = (
                price_direction != breadth_direction
            ).astype(int).reindex(data.index)
        
        return features
    
    def _add_pca_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add principal component features from market returns"""
        returns = data['close'].pct_change()
        
        # Collect returns from multiple assets
        returns_matrix = [returns]
        symbols = ['target']
        
        for symbol in list(self.market_data.keys())[:20]:  # Limit to 20 assets
            asset_returns = self.market_data[symbol]['close'].pct_change()
            aligned_returns, aligned_asset = returns.align(asset_returns, join='inner')
            returns_matrix.append(aligned_asset)
            symbols.append(symbol)
        
        # Create returns DataFrame
        returns_df = pd.concat(returns_matrix, axis=1)
        returns_df.columns = symbols
        
        # Remove NaN rows
        returns_df = returns_df.dropna()
        
        if len(returns_df) > self.cross_config.n_components * 2:
            # Standardize returns
            returns_std = (returns_df - returns_df.mean()) / returns_df.std()
            
            # Fit PCA
            pca = PCA(n_components=self.cross_config.n_components)
            components = pca.fit_transform(returns_std)
            
            # Create component DataFrame
            component_df = pd.DataFrame(
                components,
                index=returns_df.index,
                columns=[f'market_pc_{i+1}' for i in range(self.cross_config.n_components)]
            )
            
            # Add to features
            for col in component_df.columns:
                features[col] = component_df[col].reindex(data.index)
            
            # Explained variance
            features['pca_explained_variance'] = sum(pca.explained_variance_ratio_)
            
            # Loading on first component (market factor)
            features['market_factor_loading'] = pca.components_[0][0]  # Target asset loading
        
        return features
    
    def _add_pair_trading_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add pair trading opportunity features"""
        close = data['close']
        returns = close.pct_change()
        
        # Find best correlated pairs
        high_corr_pairs = []
        
        for symbol in self.market_data:
            other_close = self.market_data[symbol]['close']
            other_returns = other_close.pct_change()
            
            # Align data
            aligned_close, aligned_other = close.align(other_close, join='inner')
            aligned_returns, aligned_other_returns = returns.align(other_returns, join='inner')
            
            # Calculate correlation
            corr = aligned_returns.rolling(60).corr(aligned_other_returns).mean()
            
            if corr > 0.7:  # High correlation threshold
                high_corr_pairs.append((symbol, corr))
        
        if high_corr_pairs:
            # Sort by correlation
            high_corr_pairs.sort(key=lambda x: x[1], reverse=True)
            best_pair = high_corr_pairs[0][0]
            
            features['best_pair_symbol'] = best_pair
            features['best_pair_correlation'] = high_corr_pairs[0][1]
            
            # Calculate spread
            pair_close = self.market_data[best_pair]['close']
            aligned_close, aligned_pair = close.align(pair_close, join='inner')
            
            # Normalize prices
            norm_close = aligned_close / aligned_close.iloc[0]
            norm_pair = aligned_pair / aligned_pair.iloc[0]
            
            # Spread
            spread = norm_close - norm_pair
            
            # Z-score of spread
            spread_mean = spread.rolling(60).mean()
            spread_std = spread.rolling(60).std()
            spread_zscore = (spread - spread_mean) / spread_std
            
            features['pair_spread'] = spread.reindex(data.index)
            features['pair_spread_zscore'] = spread_zscore.reindex(data.index)
            
            # Trading signals
            features['pair_trade_long'] = (spread_zscore < -2).astype(int).reindex(data.index)
            features['pair_trade_short'] = (spread_zscore > 2).astype(int).reindex(data.index)
            
            # Cointegration test (simplified)
            from statsmodels.tsa.stattools import coint
            try:
                score, pvalue, _ = coint(aligned_close.dropna(), aligned_pair.dropna())
                features['pair_cointegration_pvalue'] = pvalue
                features['is_cointegrated'] = (pvalue < 0.05).astype(int)
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not calculate cointegration: {e}")
                pass
        
        return features
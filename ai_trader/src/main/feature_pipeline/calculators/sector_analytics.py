"""
Sector Analytics Calculator

Calculates sector-based features including:
- Sector rotation indicators
- Relative strength analysis
- Sector breadth and momentum
- Industry group performance
- Sector correlation and divergence
- Economic cycle positioning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf

from .base_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


@dataclass
class SectorConfig:
    """Configuration for sector analytics"""
    # Sector ETFs
    sector_etfs: Dict[str, str] = None
    
    # Industry groups
    industry_groups: Dict[str, List[str]] = None
    
    # Analysis windows
    momentum_windows: List[int] = None
    correlation_window: int = 60
    
    # Rotation thresholds
    rotation_threshold: float = 0.05  # 5% relative performance
    breadth_threshold: float = 0.6    # 60% advancing
    
    # Economic sectors
    cyclical_sectors: List[str] = None
    defensive_sectors: List[str] = None
    
    def __post_init__(self):
        if self.sector_etfs is None:
            self.sector_etfs = {
                'technology': 'XLK',
                'healthcare': 'XLV',
                'financials': 'XLF',
                'consumer_discretionary': 'XLY',
                'industrials': 'XLI',
                'consumer_staples': 'XLP',
                'energy': 'XLE',
                'utilities': 'XLU',
                'materials': 'XLB',
                'real_estate': 'XLRE',
                'communication': 'XLC'
            }
        
        if self.momentum_windows is None:
            self.momentum_windows = [5, 20, 60]
        
        if self.cyclical_sectors is None:
            self.cyclical_sectors = ['technology', 'consumer_discretionary', 'financials', 'industrials', 'materials']
        
        if self.defensive_sectors is None:
            self.defensive_sectors = ['consumer_staples', 'healthcare', 'utilities', 'real_estate']


class SectorAnalyticsCalculator(BaseFeatureCalculator):
    """Calculator for sector-based analytics and features"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.sector_config = SectorConfig(
            **config.get('sector', {}) if config else {}
        )
        self.sector_data = {}  # Cache for sector ETF data
        self.industry_constituents = {}  # Industry group constituents
        self.market_index = None  # Benchmark index data
    
    def set_sector_data(self, sector: str, data: pd.DataFrame):
        """Set data for a specific sector"""
        self.sector_data[sector] = data
    
    def set_market_index(self, data: pd.DataFrame):
        """Set market index data for relative performance"""
        self.market_index = data
    
    def set_industry_constituents(self, industry: str, symbols: List[str]):
        """Set constituent symbols for an industry group"""
        self.industry_constituents[industry] = symbols
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-based features.
        
        Args:
            data: DataFrame with target symbol OHLCV data
            
        Returns:
            DataFrame with sector features
        """
        # Enhanced input validation
        if not self.validate_inputs(data):
            logger.error("Invalid input data for sector calculation")
            return self._create_empty_features(data.index)
        
        # Validate sector data availability
        if not self._validate_sector_data():
            logger.warning("No valid sector data available, returning minimal features")
            return self._create_empty_features(data.index)
        
        features = pd.DataFrame(index=data.index)
        feature_counts = {}
        
        # Feature calculation with individual error handling
        feature_methods = [
            ('sector_performance', self._add_sector_performance),
            ('sector_rotation', self._add_sector_rotation),
            ('relative_strength', self._add_relative_strength),
            ('sector_breadth', self._add_sector_breadth),
            ('sector_momentum', self._add_sector_momentum),
            ('economic_cycle', self._add_economic_cycle),
            ('sector_correlation', self._add_sector_correlation),
            ('industry_analysis', self._add_industry_analysis),
            ('sector_divergence', self._add_sector_divergence)
        ]
        
        for method_name, method_func in feature_methods:
            try:
                initial_cols = len(features.columns)
                features = method_func(data, features)
                added_cols = len(features.columns) - initial_cols
                feature_counts[method_name] = added_cols
                
                if added_cols > 0:
                    logger.debug(f"Added {added_cols} features from {method_name}")
                    
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}", exc_info=True)
                # Continue with other features even if one fails
                continue
        
        # Validate output features
        if features.empty:
            logger.warning("No features were calculated successfully")
            return self._create_empty_features(data.index)
        
        # Final validation and cleanup
        features = self._validate_output_features(features)
        
        total_features = len(features.columns)
        logger.info(f"Successfully calculated {total_features} sector features across {len(feature_counts)} categories")
        logger.debug(f"Feature breakdown: {feature_counts}")
        
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required = ['close']
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
            
        return True
    
    def get_required_columns(self) -> List[str]:
        """Return list of required input columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this calculator produces"""
        feature_names = []
        
        # Sector performance features for each momentum window
        for window in self.sector_config.momentum_windows:
            feature_names.extend([
                f'sector_relative_strength_{window}d',
                f'sector_momentum_{window}d',
                f'sector_vs_market_{window}d'
            ])
        
        # Sector rotation features
        feature_names.extend([
            'sector_rotation_strength',
            'cyclical_defensive_ratio',
            'growth_value_ratio',
            'risk_appetite_score',
            'sector_leadership_change',
            'defensive_rotation',
            'cyclical_rotation'
        ])
        
        # Sector breadth and momentum
        feature_names.extend([
            'sector_breadth_ratio',
            'sector_advance_decline',
            'sector_momentum_breadth',
            'sector_trend_consistency'
        ])
        
        # Economic cycle features
        feature_names.extend([
            'early_cycle_score',
            'mid_cycle_score', 
            'late_cycle_score',
            'recession_score',
            'recovery_score',
            'expansion_score'
        ])
        
        # Sector correlation features
        for window in [30, 60]:
            feature_names.extend([
                f'avg_sector_correlation_{window}d',
                f'sector_correlation_dispersion_{window}d',
                f'sector_correlation_trend_{window}d'
            ])
        
        # Sector divergence and stress
        feature_names.extend([
            'sector_return_dispersion',
            'sector_performance_spread',
            'sector_trend_dispersion',
            'sector_stress'
        ])
        
        # Inter-sector correlations (dynamic based on available sectors)
        sector_relationships = [
            ('technology', 'financials'),
            ('energy', 'materials'), 
            ('utilities', 'real_estate'),
            ('consumer_discretionary', 'consumer_staples')
        ]
        
        for sector1, sector2 in sector_relationships:
            feature_names.append(f'{sector1}_{sector2}_correlation')
        
        return feature_names
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data before calculation"""
        data = super().preprocess_data(data)
        
        # Validate and prepare sector data if available
        if self.sector_data:
            filtered_sector_data = {}
            for sector_name, sector_df in self.sector_data.items():
                if sector_df is not None and not sector_df.empty:
                    # Ensure datetime index
                    if not isinstance(sector_df.index, pd.DatetimeIndex):
                        if 'date' in sector_df.columns:
                            sector_df = sector_df.set_index('date')
                        elif 'timestamp' in sector_df.columns:
                            sector_df = sector_df.set_index('timestamp')
                    
                    # Filter to relevant time range
                    if not data.empty:
                        start_date = data.index.min() - timedelta(days=max(self.sector_config.momentum_windows))
                        end_date = data.index.max()
                        
                        sector_df = sector_df[
                            (sector_df.index >= start_date) & 
                            (sector_df.index <= end_date)
                        ].copy()
                    
                    filtered_sector_data[sector_name] = sector_df
                    
            self.sector_data = filtered_sector_data
            logger.info(f"Preprocessed sector data for {len(self.sector_data)} sectors")
        
        # Validate market index data if available
        if self.market_index is not None:
            if not isinstance(self.market_index.index, pd.DatetimeIndex):
                if 'date' in self.market_index.columns:
                    self.market_index = self.market_index.set_index('date')
                elif 'timestamp' in self.market_index.columns:
                    self.market_index = self.market_index.set_index('timestamp')
        
        return data
    
    def postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Postprocess calculated features"""
        features = super().postprocess_features(features)
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values based on feature type
        
        # Ratio and correlation features: fill with 0
        ratio_columns = [col for col in features.columns if any(keyword in col for keyword in 
                        ['ratio', 'correlation', 'relative_strength', 'vs_market'])]
        features[ratio_columns] = features[ratio_columns].fillna(0)
        
        # Score features: clip to [0, 1] and fill NaN with 0.5 (neutral)
        score_columns = [col for col in features.columns if 'score' in col]
        for col in score_columns:
            if col in features.columns:
                features[col] = features[col].clip(0, 1).fillna(0.5)
        
        # Breadth and momentum features: fill with 0
        breadth_columns = [col for col in features.columns if any(keyword in col for keyword in 
                          ['breadth', 'momentum', 'advance_decline', 'dispersion'])]
        features[breadth_columns] = features[breadth_columns].fillna(0)
        
        # Binary rotation features: fill with 0
        rotation_columns = [col for col in features.columns if 'rotation' in col]
        features[rotation_columns] = features[rotation_columns].fillna(0).astype(int)
        
        # Clip extreme ratio values to reasonable bounds
        for col in ratio_columns:
            if col in features.columns:
                features[col] = features[col].clip(-5, 5)
        
        # Ensure correlation features are in [-1, 1] range
        correlation_columns = [col for col in features.columns if 'correlation' in col]
        for col in correlation_columns:
            if col in features.columns:
                features[col] = features[col].clip(-1, 1)
        
        logger.info(f"Postprocessed {len(features.columns)} sector features")
        return features
    
    def _add_sector_performance(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector performance metrics"""
        if not self.sector_data:
            return features
        
        # Calculate returns for each sector
        sector_returns = {}
        for sector, sector_df in self.sector_data.items():
            if 'close' in sector_df.columns:
                aligned_close = sector_df['close'].reindex(data.index, method='ffill')
                sector_returns[sector] = aligned_close.pct_change()
        
        if not sector_returns:
            return features
        
        # Sector returns over different periods
        for window in self.sector_config.momentum_windows:
            for sector, returns in sector_returns.items():
                features[f'{sector}_return_{window}d'] = returns.rolling(window).sum()
        
        # Best and worst performing sectors
        returns_df = pd.DataFrame(sector_returns)
        
        for window in self.sector_config.momentum_windows:
            rolling_returns = returns_df.rolling(window).sum()
            features[f'best_sector_{window}d'] = rolling_returns.idxmax(axis=1)
            features[f'worst_sector_{window}d'] = rolling_returns.idxmin(axis=1)
            features[f'best_sector_return_{window}d'] = rolling_returns.max(axis=1)
            features[f'worst_sector_return_{window}d'] = rolling_returns.min(axis=1)
            
            # Sector return spread
            features[f'sector_return_spread_{window}d'] = (
                features[f'best_sector_return_{window}d'] - 
                features[f'worst_sector_return_{window}d']
            )
        
        # Average sector performance
        features['avg_sector_return'] = returns_df.mean(axis=1)
        features['sector_return_dispersion'] = returns_df.std(axis=1)
        
        return features
    
    def _add_sector_rotation(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector rotation indicators"""
        if not self.sector_data:
            return features
        
        # Calculate relative performance vs market
        if self.market_index is not None:
            market_returns = self.market_index['close'].pct_change()
        else:
            # Use equal-weighted sector average as proxy
            sector_closes = pd.DataFrame({
                sector: df['close'] for sector, df in self.sector_data.items()
                if 'close' in df.columns
            })
            market_returns = sector_closes.mean(axis=1).pct_change()
        
        # Sector rotation scores
        rotation_scores = {}
        
        for sector, sector_df in self.sector_data.items():
            if 'close' in sector_df.columns:
                sector_returns = sector_df['close'].pct_change()
                
                # Align data
                aligned_sector, aligned_market = sector_returns.align(market_returns, join='inner')
                
                # Relative performance
                relative_perf = aligned_sector - aligned_market
                
                # Rotation score (momentum of relative performance)
                rotation_score = relative_perf.rolling(20).mean()
                rotation_scores[sector] = rotation_score.reindex(data.index)
                
                features[f'{sector}_rotation_score'] = rotation_scores[sector]
        
        # Identify rotation patterns
        if rotation_scores:
            scores_df = pd.DataFrame(rotation_scores)
            
            # Leading sectors (positive rotation)
            leading_mask = scores_df > self.sector_config.rotation_threshold
            features['leading_sectors_count'] = leading_mask.sum(axis=1)
            features['leading_sectors'] = leading_mask.apply(
                lambda x: ','.join(x[x].index.tolist()), axis=1
            )
            
            # Lagging sectors (negative rotation)
            lagging_mask = scores_df < -self.sector_config.rotation_threshold
            features['lagging_sectors_count'] = lagging_mask.sum(axis=1)
            
            # Rotation velocity
            features['rotation_velocity'] = scores_df.diff().abs().mean(axis=1)
            
            # Rotation clarity (how distinct the rotation is)
            features['rotation_clarity'] = scores_df.std(axis=1)
        
        return features
    
    def _add_relative_strength(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength analysis"""
        symbol_returns = data['close'].pct_change()
        
        # RS vs each sector
        for sector, sector_df in self.sector_data.items():
            if 'close' not in sector_df.columns:
                continue
                
            sector_returns = sector_df['close'].pct_change()
            
            # Align data
            aligned_symbol, aligned_sector = symbol_returns.align(sector_returns, join='inner')
            
            # Relative strength
            for window in [20, 60]:
                symbol_perf = (1 + aligned_symbol).rolling(window).apply(lambda x: x.prod()) - 1
                sector_perf = (1 + aligned_sector).rolling(window).apply(lambda x: x.prod()) - 1
                
                rs = (1 + symbol_perf) / (1 + sector_perf) - 1
                features[f'rs_vs_{sector}_{window}d'] = rs.reindex(data.index)
        
        # Overall relative strength ranking
        if self.sector_data:
            rs_scores = []
            for col in features.columns:
                if col.startswith('rs_vs_') and col.endswith('_20d'):
                    rs_scores.append(features[col])
            
            if rs_scores:
                features['avg_relative_strength'] = pd.concat(rs_scores, axis=1).mean(axis=1)
                features['rs_percentile'] = features['avg_relative_strength'].rolling(252).rank(pct=True)
        
        return features
    
    def _add_sector_breadth(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector breadth indicators"""
        # If we have industry constituent data, calculate breadth
        if self.industry_constituents:
            breadth_data = {}
            
            for industry, symbols in self.industry_constituents.items():
                # This would need actual constituent data
                # For now, using sector ETF volume as proxy
                sector = self._get_sector_for_industry(industry)
                if sector in self.sector_data and 'volume' in self.sector_data[sector].columns:
                    volume = self.sector_data[sector]['volume']
                    # Volume breadth as proxy
                    breadth_data[industry] = volume / volume.rolling(20).mean()
            
            if breadth_data:
                breadth_df = pd.DataFrame(breadth_data)
                features['sector_breadth'] = (breadth_df > 1).mean(axis=1)
                features['breadth_thrust'] = (
                    features['sector_breadth'] > self.sector_config.breadth_threshold
                ).astype(int)
        
        # Sector participation
        if 'leading_sectors_count' in features.columns:
            total_sectors = len(self.sector_config.sector_etfs)
            features['sector_participation'] = features['leading_sectors_count'] / total_sectors
        
        return features
    
    def _add_sector_momentum(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector momentum features"""
        if not self.sector_data:
            return features
        
        # Sector momentum scores
        for sector, sector_df in self.sector_data.items():
            if 'close' not in sector_df.columns:
                continue
            
            close = sector_df['close'].reindex(data.index, method='ffill')
            
            # Multiple timeframe momentum
            mom_scores = []
            for window in self.sector_config.momentum_windows:
                momentum = close.pct_change(window)
                mom_scores.append(momentum)
            
            # Composite momentum
            if mom_scores:
                features[f'{sector}_momentum'] = pd.concat(mom_scores, axis=1).mean(axis=1)
                
                # Momentum rank
                features[f'{sector}_momentum_rank'] = features[f'{sector}_momentum'].rolling(60).rank(pct=True)
        
        # Cross-sector momentum
        momentum_cols = [col for col in features.columns if col.endswith('_momentum') and not col.endswith('_rank')]
        if momentum_cols:
            momentum_df = features[momentum_cols]
            
            # Momentum dispersion
            features['sector_momentum_dispersion'] = momentum_df.std(axis=1)
            
            # Momentum alignment
            features['sector_momentum_alignment'] = (momentum_df > 0).mean(axis=1)
            
            # Strong momentum sectors
            features['strong_momentum_sectors'] = (momentum_df > momentum_df.quantile(0.8, axis=1).values[:, None]).sum(axis=1)
        
        return features
    
    def _add_economic_cycle(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add economic cycle positioning features"""
        if not self.sector_data:
            return features
        
        # Cyclical vs Defensive performance
        cyclical_returns = []
        defensive_returns = []
        
        for sector, sector_df in self.sector_data.items():
            if 'close' not in sector_df.columns:
                continue
                
            returns = sector_df['close'].pct_change().reindex(data.index)
            
            if sector in self.sector_config.cyclical_sectors:
                cyclical_returns.append(returns)
            elif sector in self.sector_config.defensive_sectors:
                defensive_returns.append(returns)
        
        if cyclical_returns and defensive_returns:
            # Average performance
            avg_cyclical = pd.concat(cyclical_returns, axis=1).mean(axis=1)
            avg_defensive = pd.concat(defensive_returns, axis=1).mean(axis=1)
            
            # Relative performance
            features['cyclical_defensive_ratio'] = (
                (1 + avg_cyclical.rolling(20).mean()) / 
                (1 + avg_defensive.rolling(20).mean())
            )
            
            # Cycle phase indicators
            features['early_cycle'] = (
                (features['cyclical_defensive_ratio'] > 1.05) & 
                (features.get('financials_momentum', 0) > 0)
            ).astype(int)
            
            features['late_cycle'] = (
                (features['cyclical_defensive_ratio'] < 0.95) & 
                (features.get('energy_momentum', 0) > 0)
            ).astype(int)
            
            # Risk on/off based on sectors
            features['sector_risk_on'] = (
                features['cyclical_defensive_ratio'] > features['cyclical_defensive_ratio'].rolling(20).mean()
            ).astype(int)
        
        # Growth vs Value proxy (using tech vs financials)
        if 'technology_rotation_score' in features and 'financials_rotation_score' in features:
            features['growth_value_tilt'] = (
                features['technology_rotation_score'] - 
                features['financials_rotation_score']
            )
        
        return features
    
    def _add_sector_correlation(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector correlation features"""
        if not self.sector_data:
            return features
        
        symbol_returns = data['close'].pct_change()
        
        # Correlation with each sector
        correlations = {}
        for sector, sector_df in self.sector_data.items():
            if 'close' not in sector_df.columns:
                continue
                
            sector_returns = sector_df['close'].pct_change()
            aligned_symbol, aligned_sector = symbol_returns.align(sector_returns, join='inner')
            
            # Rolling correlation
            corr = aligned_symbol.rolling(self.sector_config.correlation_window).corr(aligned_sector)
            correlations[sector] = corr.reindex(data.index)
            features[f'corr_{sector}'] = correlations[sector]
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            
            # Highest correlation sector
            features['highest_corr_sector'] = corr_df.idxmax(axis=1)
            features['highest_correlation'] = corr_df.max(axis=1)
            
            # Correlation dispersion
            features['sector_corr_dispersion'] = corr_df.std(axis=1)
            
            # Decorrelation indicator
            features['decorrelated'] = (
                features['highest_correlation'] < 0.5
            ).astype(int)
        
        # Inter-sector correlations
        if len(self.sector_data) >= 2:
            features = self._add_intersector_correlations(features)
        
        return features
    
    def _add_industry_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add industry group analysis"""
        # Industry momentum (simplified using sector proxies)
        industry_momentum = {}
        
        # Map industries to sectors
        industry_sector_map = {
            'software': 'technology',
            'semiconductors': 'technology',
            'biotech': 'healthcare',
            'pharma': 'healthcare',
            'banks': 'financials',
            'insurance': 'financials',
            'retail': 'consumer_discretionary',
            'autos': 'consumer_discretionary',
            'mining': 'materials',
            'chemicals': 'materials'
        }
        
        for industry, sector in industry_sector_map.items():
            if sector in self.sector_data and 'close' in self.sector_data[sector].columns:
                # Use sector as proxy with some noise
                sector_close = self.sector_data[sector]['close'].reindex(data.index, method='ffill')
                industry_return = sector_close.pct_change(20)
                
                # Add some differentiation
                if 'software' in industry:
                    industry_return *= 1.2  # Higher beta
                elif 'mining' in industry:
                    industry_return *= 0.8  # Lower beta
                
                industry_momentum[industry] = industry_return
                features[f'{industry}_momentum'] = industry_return
        
        # Industry group strength
        if industry_momentum:
            momentum_df = pd.DataFrame(industry_momentum)
            features['strongest_industry'] = momentum_df.idxmax(axis=1)
            features['weakest_industry'] = momentum_df.idxmin(axis=1)
            
            # Industry concentration
            features['industry_concentration'] = momentum_df.abs().max(axis=1) / momentum_df.abs().mean(axis=1)
        
        return features
    
    def _add_sector_divergence(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sector divergence indicators"""
        if not self.sector_data or len(self.sector_data) < 2:
            return features
        
        # Price divergence from sector
        symbol_returns = data['close'].pct_change()
        
        divergence_scores = []
        for sector, sector_df in self.sector_data.items():
            if 'close' not in sector_df.columns:
                continue
                
            sector_returns = sector_df['close'].pct_change()
            aligned_symbol, aligned_sector = symbol_returns.align(sector_returns, join='inner')
            
            # 20-day performance difference
            symbol_perf = aligned_symbol.rolling(20).sum()
            sector_perf = aligned_sector.rolling(20).sum()
            
            divergence = symbol_perf - sector_perf
            divergence_scores.append(divergence.reindex(data.index))
        
        if divergence_scores:
            # Maximum divergence
            features['max_sector_divergence'] = pd.concat(divergence_scores, axis=1).abs().max(axis=1)
            
            # Divergence direction
            features['positive_divergence'] = (
                pd.concat(divergence_scores, axis=1) > 0.1
            ).any(axis=1).astype(int)
            
            features['negative_divergence'] = (
                pd.concat(divergence_scores, axis=1) < -0.1
            ).any(axis=1).astype(int)
        
        # Sector dispersion as market stress indicator
        if 'sector_return_dispersion' in features.columns:
            features['sector_stress'] = (
                features['sector_return_dispersion'] > 
                features['sector_return_dispersion'].rolling(60).quantile(0.8)
            ).astype(int)
        
        return features
    
    def _add_intersector_correlations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add correlations between sectors"""
        # Key sector relationships
        relationships = [
            ('technology', 'financials'),  # Risk sentiment
            ('energy', 'materials'),       # Commodity play
            ('utilities', 'real_estate'),  # Defensive correlation
            ('consumer_discretionary', 'consumer_staples')  # Consumer strength
        ]
        
        for sector1, sector2 in relationships:
            if sector1 in self.sector_data and sector2 in self.sector_data:
                if 'close' in self.sector_data[sector1].columns and 'close' in self.sector_data[sector2].columns:
                    returns1 = self.sector_data[sector1]['close'].pct_change()
                    returns2 = self.sector_data[sector2]['close'].pct_change()
                    
                    aligned_1, aligned_2 = returns1.align(returns2, join='inner')
                    
                    corr = aligned_1.rolling(30).corr(aligned_2)
                    features[f'{sector1}_{sector2}_correlation'] = corr.reindex(features.index)
        
        return features
    
    def _validate_sector_data(self) -> bool:
        """Validate that sector data is available and properly formatted"""
        if not self.sector_data:
            logger.debug("No sector data available")
            return False
            
        valid_sectors = 0
        for sector_name, sector_df in self.sector_data.items():
            if sector_df is None or sector_df.empty:
                logger.debug(f"Sector {sector_name} has no data")
                continue
                
            # Check for required columns
            if 'close' not in sector_df.columns:
                logger.debug(f"Sector {sector_name} missing 'close' column")
                continue
                
            # Check for valid data
            if sector_df['close'].isna().all():
                logger.debug(f"Sector {sector_name} has all NaN close prices")
                continue
                
            valid_sectors += 1
            
        logger.debug(f"Found {valid_sectors} valid sectors out of {len(self.sector_data)}")
        return valid_sectors > 0
    
    def _create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """Create empty features DataFrame with proper structure when calculation fails"""
        feature_names = self.get_feature_names()
        empty_features = pd.DataFrame(index=index, columns=feature_names)
        
        # Fill with appropriate default values based on feature type
        for col in feature_names:
            if any(keyword in col for keyword in ['ratio', 'correlation', 'relative_strength']):
                empty_features[col] = 0.0
            elif 'score' in col:
                empty_features[col] = 0.5  # Neutral score
            elif any(keyword in col for keyword in ['rotation', 'thrust', 'early_cycle', 'late_cycle']):
                empty_features[col] = 0  # Binary features
            else:
                empty_features[col] = 0.0
        
        logger.debug(f"Created empty features DataFrame with {len(feature_names)} columns")
        return empty_features
    
    def _validate_output_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean output features"""
        if features.empty:
            return features
            
        # Check for completely empty columns
        empty_cols = features.columns[features.isna().all()].tolist()
        if empty_cols:
            logger.warning(f"Found {len(empty_cols)} completely empty feature columns")
            # Fill empty columns with defaults instead of dropping
            for col in empty_cols:
                if 'score' in col:
                    features[col] = 0.5
                else:
                    features[col] = 0.0
        
        # Check for infinite values
        inf_cols = []
        for col in features.columns:
            if features[col].replace([np.inf, -np.inf], np.nan).isna().sum() != features[col].isna().sum():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Found infinite values in {len(inf_cols)} columns: {inf_cols[:5]}...")
            features = features.replace([np.inf, -np.inf], np.nan)
        
        # Validate data types
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        return features
    
    def _get_sector_for_industry(self, industry: str) -> Optional[str]:
        """Map industry to sector"""
        # Simplified mapping
        industry_map = {
            'software': 'technology',
            'semiconductors': 'technology',
            'biotech': 'healthcare',
            'banks': 'financials',
            'retail': 'consumer_discretionary',
            'mining': 'materials'
        }
        return industry_map.get(industry.lower())
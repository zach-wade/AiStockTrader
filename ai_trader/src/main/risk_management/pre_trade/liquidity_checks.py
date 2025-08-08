"""
Liquidity verification
Created: 2025-06-16
"""

"""
Liquidity checks to ensure positions can be exited.

Validates that position sizes are appropriate for the
liquidity available in the market.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from main.config.config_manager import get_config
from main.interfaces.database import IAsyncDatabase
from main.utils.cache import MemoryBackend
from main.utils.cache import CacheType

logger = logging.getLogger(__name__)


class LiquidityChecker:
    """Validates liquidity before entering positions."""
    
    def __init__(self, config: Any, database: IAsyncDatabase):
        """Initialize liquidity checker."""
        self.config = config
        self.database = database
        
        # Liquidity parameters
        self.min_daily_volume = config.get('liquidity.min_daily_volume', 100000)
        self.max_position_pct = config.get('liquidity.max_position_pct_adv', 0.01)  # 1% of ADV
        self.max_spread_pct = config.get('liquidity.max_spread_pct', 0.005)  # 0.5%
        self.min_market_cap = config.get('liquidity.min_market_cap', 1e9)  # $1B
        
        # Liquidity scoring weights
        self.volume_weight = config.get('liquidity.volume_weight', 0.4)
        self.spread_weight = config.get('liquidity.spread_weight', 0.3)
        self.depth_weight = config.get('liquidity.depth_weight', 0.3)
        
        # Cache for liquidity data
        self.cache = get_global_cache()
        self.cache_ttl = config.get('liquidity.cache_ttl_seconds', 300)  # 5 minutes
    
    async def check_liquidity(self, symbol: str, quantity: int, 
                            side: str = 'buy') -> Dict[str, Any]:
        """
        Check if a position has sufficient liquidity.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            
        Returns:
            Dictionary with liquidity check results
        """
        # Get liquidity data
        liquidity_data = await self._get_liquidity_data(symbol)
        
        if not liquidity_data:
            return {
                'passed': False,
                'reason': 'No liquidity data available',
                'score': 0
            }
        
        # Run liquidity checks
        checks = {
            'volume_check': self._check_volume(quantity, liquidity_data),
            'spread_check': self._check_spread(liquidity_data),
            'market_cap_check': self._check_market_cap(liquidity_data),
            'depth_check': self._check_market_depth(quantity, liquidity_data, side),
            'impact_check': self._check_market_impact(quantity, liquidity_data)
        }
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(checks, liquidity_data)
        
        # Overall pass/fail
        passed = all(check['passed'] for check in checks.values())
        
        # Get failure reasons
        failure_reasons = [
            f"{name}: {check['reason']}" 
            for name, check in checks.items() 
            if not check['passed']
        ]
        
        return {
            'passed': passed,
            'score': liquidity_score,
            'checks': checks,
            'failure_reasons': failure_reasons,
            'liquidity_data': liquidity_data,
            'recommendations': self._get_recommendations(quantity, liquidity_data, checks)
        }
    
    async def _get_liquidity_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get liquidity data for a symbol."""
        # Check cache first
        cache_key = f"liquidity:{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        cached_data = await self.cache.get(CacheType.CUSTOM, cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get volume data (last 20 days)
            volume_data = await self.database.fetch(
                """
                SELECT 
                    AVG(volume) as avg_volume,
                    AVG(dollar_volume) as avg_dollar_volume,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY volume) as volume_q25,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY volume) as volume_q75,
                    COUNT(*) as trading_days
                FROM market_data
                WHERE symbol = $1 
                AND timestamp >= CURRENT_DATE - INTERVAL '20 days'
                AND volume > 0
                """,
                symbol
            )
            
            # Get spread data (last 5 days)
            spread_data = await self.database.fetch_one(
                """
                SELECT 
                    AVG((ask - bid) / ((ask + bid) / 2)) as avg_spread_pct,
                    MAX((ask - bid) / ((ask + bid) / 2)) as max_spread_pct,
                    AVG(ask - bid) as avg_spread
                FROM market_data
                WHERE symbol = $1 
                AND timestamp >= CURRENT_DATE - INTERVAL '5 days'
                AND bid > 0 AND ask > 0
                """,
                symbol
            )
            
            # Get latest quote
            latest_quote = await self.database.fetch_one(
                """
                SELECT price, bid, ask, volume
                FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol
            )
            
            # Get market cap (simplified - would use fundamental data in production)
            market_cap = await self._estimate_market_cap(symbol)
            
            if volume_data and latest_quote:
                liquidity_data = {
                    'symbol': symbol,
                    'avg_volume': volume_data[0]['avg_volume'],
                    'avg_dollar_volume': volume_data[0]['avg_dollar_volume'],
                    'volume_q25': volume_data[0]['volume_q25'],
                    'volume_q75': volume_data[0]['volume_q75'],
                    'trading_days': volume_data[0]['trading_days'],
                    'current_price': latest_quote['price'],
                    'bid': latest_quote['bid'],
                    'ask': latest_quote['ask'],
                    'spread': latest_quote['ask'] - latest_quote['bid'] if latest_quote['ask'] and latest_quote['bid'] else 0,
                    'spread_pct': spread_data['avg_spread_pct'] if spread_data else 0,
                    'market_cap': market_cap,
                    'liquidity_score': 0  # Will be calculated
                }
                
                # Cache the data
                await self.cache.set(CacheType.CUSTOM, cache_key, liquidity_data, self.cache_ttl)
                
                return liquidity_data
            
        except Exception as e:
            logger.error(f"Failed to get liquidity data for {symbol}: {e}")
        
        return None
    
    def _check_volume(self, quantity: int, liquidity_data: Dict) -> Dict[str, Any]:
        """Check if volume is sufficient."""
        avg_volume = liquidity_data['avg_volume']
        
        # Check minimum volume requirement
        if avg_volume < self.min_daily_volume:
            return {
                'passed': False,
                'reason': f'Volume too low: {avg_volume:,.0f} < {self.min_daily_volume:,.0f}',
                'avg_volume': avg_volume
            }
        
        # Check position size vs volume
        position_pct = quantity / avg_volume
        if position_pct > self.max_position_pct:
            return {
                'passed': False,
                'reason': f'Position too large: {position_pct:.1%} of ADV',
                'position_pct': position_pct,
                'max_allowed': int(avg_volume * self.max_position_pct)
            }
        
        return {
            'passed': True,
            'avg_volume': avg_volume,
            'position_pct': position_pct,
            'volume_score': min(1.0, avg_volume / 1e6)  # Score based on 1M volume
        }
    
    def _check_spread(self, liquidity_data: Dict) -> Dict[str, Any]:
        """Check if spread is acceptable."""
        spread_pct = liquidity_data['spread_pct']
        
        if spread_pct > self.max_spread_pct:
            return {
                'passed': False,
                'reason': f'Spread too wide: {spread_pct:.3%} > {self.max_spread_pct:.3%}',
                'spread_pct': spread_pct
            }
        
        return {
            'passed': True,
            'spread_pct': spread_pct,
            'spread_score': 1 - (spread_pct / self.max_spread_pct)
        }
    
    def _check_market_cap(self, liquidity_data: Dict) -> Dict[str, Any]:
        """Check if market cap is sufficient."""
        market_cap = liquidity_data.get('market_cap', 0)
        
        if market_cap < self.min_market_cap:
            return {
                'passed': False,
                'reason': f'Market cap too small: ${market_cap/1e9:.1f}B < ${self.min_market_cap/1e9:.1f}B',
                'market_cap': market_cap
            }
        
        return {
            'passed': True,
            'market_cap': market_cap,
            'market_cap_score': min(1.0, market_cap / 10e9)  # Score based on $10B
        }
    
    def _check_market_depth(self, quantity: int, liquidity_data: Dict, 
                          side: str) -> Dict[str, Any]:
        """Check market depth (simplified without order book)."""
        # Estimate based on volume distribution
        avg_volume = liquidity_data['avg_volume']
        volume_q25 = liquidity_data['volume_q25']
        volume_q75 = liquidity_data['volume_q75']
        
        # Rough estimate of available liquidity
        # In low volume periods, expect 25% of average
        estimated_depth = volume_q25 * 0.1  # 10% of lower quartile volume
        
        if quantity > estimated_depth:
            return {
                'passed': False,
                'reason': f'Insufficient market depth: {quantity} > {estimated_depth:.0f}',
                'estimated_depth': estimated_depth
            }
        
        # Depth score based on volume consistency
        volume_consistency = 1 - (volume_q75 - volume_q25) / avg_volume if avg_volume > 0 else 0
        
        return {
            'passed': True,
            'estimated_depth': estimated_depth,
            'depth_score': volume_consistency
        }
    
    def _check_market_impact(self, quantity: int, liquidity_data: Dict) -> Dict[str, Any]:
        """Estimate market impact of the trade."""
        avg_volume = liquidity_data['avg_volume']
        current_price = liquidity_data['current_price']
        
        # Simple market impact model
        # Impact = constant * (quantity / ADV) ^ power
        impact_constant = 0.1  # 10 bps for 1% of ADV
        impact_power = 0.5  # Square root model
        
        participation_rate = quantity / avg_volume
        estimated_impact_bps = impact_constant * (participation_rate ** impact_power) * 10000
        estimated_impact_dollars = current_price * quantity * estimated_impact_bps / 10000
        
        # Check if impact is acceptable (< 50 bps)
        max_impact_bps = 50
        
        if estimated_impact_bps > max_impact_bps:
            return {
                'passed': False,
                'reason': f'Market impact too high: {estimated_impact_bps:.0f} bps',
                'impact_bps': estimated_impact_bps,
                'impact_dollars': estimated_impact_dollars
            }
        
        return {
            'passed': True,
            'impact_bps': estimated_impact_bps,
            'impact_dollars': estimated_impact_dollars,
            'impact_score': 1 - (estimated_impact_bps / max_impact_bps)
        }
    
    def _calculate_liquidity_score(self, checks: Dict, 
                                 liquidity_data: Dict) -> float:
        """Calculate overall liquidity score (0-100)."""
        scores = []
        weights = []
        
        # Volume score
        if 'volume_score' in checks['volume_check']:
            scores.append(checks['volume_check']['volume_score'])
            weights.append(self.volume_weight)
        
        # Spread score
        if 'spread_score' in checks['spread_check']:
            scores.append(checks['spread_check']['spread_score'])
            weights.append(self.spread_weight)
        
        # Depth score
        if 'depth_score' in checks['depth_check']:
            scores.append(checks['depth_check']['depth_score'])
            weights.append(self.depth_weight)
        
        # Market cap bonus
        if 'market_cap_score' in checks['market_cap_check']:
            scores.append(checks['market_cap_check']['market_cap_score'])
            weights.append(0.1)  # Small bonus for large caps
        
        if scores:
            weighted_score = np.average(scores, weights=weights)
            return round(weighted_score * 100, 1)
        
        return 0
    
    def _get_recommendations(self, quantity: int, liquidity_data: Dict, 
                           checks: Dict) -> List[str]:
        """Get recommendations for improving execution."""
        recommendations = []
        
        # Volume-based recommendations
        if not checks['volume_check']['passed']:
            max_allowed = checks['volume_check'].get('max_allowed', 0)
            if max_allowed > 0:
                recommendations.append(f"Reduce position size to {max_allowed:,} shares")
                recommendations.append("Consider splitting order across multiple days")
        
        # Spread recommendations
        if not checks['spread_check']['passed']:
            recommendations.append("Use limit orders to avoid crossing wide spread")
            recommendations.append("Trade during high volume periods (open/close)")
        
        # Impact recommendations
        if not checks['impact_check']['passed']:
            recommendations.append("Use VWAP or TWAP algorithm")
            recommendations.append("Consider iceberg orders")
            
            # Calculate suggested split
            avg_volume = liquidity_data['avg_volume']
            suggested_size = int(avg_volume * self.max_position_pct * 0.5)  # 50% of max
            num_orders = int(np.ceil(quantity / suggested_size))
            recommendations.append(f"Split into {num_orders} orders of {suggested_size:,} shares")
        
        # General recommendations based on score
        score = self._calculate_liquidity_score(checks, liquidity_data)
        if score < 50:
            recommendations.append("Consider alternative stocks with better liquidity")
        elif score < 70:
            recommendations.append("Monitor execution carefully")
        
        return recommendations
    
    async def _estimate_market_cap(self, symbol: str) -> float:
        """Estimate market cap (simplified)."""
        # In production, this would query fundamental data
        # For now, use a simple estimation based on price and typical shares
        
        latest_price = await self.database.fetch_one(
            "SELECT price FROM market_data WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1",
            symbol
        )
        
        if latest_price:
            # Rough estimate: assume 100M shares for unknown stocks
            estimated_shares = 100_000_000
            return latest_price['price'] * estimated_shares
        
        return 0
    
    async def check_portfolio_liquidity(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Check liquidity for entire portfolio."""
        results = {}
        total_score = 0
        illiquid_positions = []
        
        for symbol, position in positions.items():
            quantity = abs(position.get('quantity', 0))
            if quantity > 0:
                check_result = await self.check_liquidity(symbol, quantity)
                results[symbol] = check_result
                
                total_score += check_result['score']
                
                if not check_result['passed']:
                    illiquid_positions.append(symbol)
        
        avg_score = total_score / len(positions) if positions else 0
        
        return {
            'average_liquidity_score': avg_score,
            'illiquid_positions': illiquid_positions,
            'position_results': results,
            'portfolio_liquid': len(illiquid_positions) == 0
        }
    
    def estimate_execution_time(self, symbol: str, quantity: int, 
                              liquidity_data: Dict) -> Dict[str, Any]:
        """Estimate time required to execute order without impact."""
        avg_volume = liquidity_data['avg_volume']
        
        # Assume we can safely trade 1% of volume per 5-minute bar
        safe_rate = avg_volume * 0.01 / 78  # 78 five-minute bars in a day
        
        # Time to execute
        bars_needed = quantity / safe_rate
        minutes_needed = bars_needed * 5
        
        return {
            'estimated_minutes': round(minutes_needed, 1),
            'estimated_bars': round(bars_needed, 1),
            'safe_execution_rate': round(safe_rate),
            'full_day_executable': quantity <= avg_volume * self.max_position_pct
        }
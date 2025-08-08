"""
Volume Scanner Example

This example demonstrates how to create a custom volume scanner
that identifies stocks with unusual volume patterns.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from main.scanners.base_scanner import BaseScanner
from main.config.config_manager import get_config
from main.data_providers.alpaca.market_data import AlpacaMarketClient
from main.utils.resilience.recovery_manager import get_global_recovery_manager
import structlog

logger = structlog.get_logger(__name__)


class VolumeScanner(BaseScanner):
    """
    Custom volume scanner that identifies:
    - Volume spikes (current volume > X times average)
    - Accumulation patterns
    - Distribution patterns
    - Volume dry-ups
    """
    
    def __init__(
        self, 
        config, 
        volume_multiplier: float = 2.0,
        lookback_days: int = 20
    ):
        super().__init__(config)
        self.volume_multiplier = volume_multiplier
        self.lookback_days = lookback_days
        
        # Initialize market data client
        recovery_manager = get_global_recovery_manager()
        self.market_client = AlpacaMarketClient(config, recovery_manager)
    
    async def scan(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan symbols for volume patterns."""
        logger.info(
            "Starting volume scan",
            symbol_count=len(symbols),
            multiplier=self.volume_multiplier
        )
        
        results = {
            'volume_spikes': [],
            'accumulation': [],
            'distribution': [],
            'volume_dryup': [],
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'symbols_scanned': len(symbols),
                'lookback_days': self.lookback_days,
                'volume_multiplier': self.volume_multiplier
            }
        }
        
        # Process symbols in batches
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_results = await self._process_batch(batch)
            
            # Categorize results
            for symbol, analysis in batch_results.items():
                if analysis['has_volume_spike']:
                    results['volume_spikes'].append({
                        'symbol': symbol,
                        'volume_ratio': analysis['volume_ratio'],
                        'current_volume': analysis['current_volume'],
                        'avg_volume': analysis['avg_volume'],
                        'price_change': analysis['price_change']
                    })
                
                if analysis['accumulation_score'] > 0.7:
                    results['accumulation'].append({
                        'symbol': symbol,
                        'score': analysis['accumulation_score'],
                        'pattern': analysis['pattern']
                    })
                
                if analysis['distribution_score'] > 0.7:
                    results['distribution'].append({
                        'symbol': symbol,
                        'score': analysis['distribution_score'],
                        'pattern': analysis['pattern']
                    })
                
                if analysis['is_volume_dryup']:
                    results['volume_dryup'].append({
                        'symbol': symbol,
                        'volume_ratio': analysis['volume_ratio'],
                        'days_below_average': analysis['days_below_average']
                    })
        
        logger.info(
            "Volume scan complete",
            spikes=len(results['volume_spikes']),
            accumulation=len(results['accumulation']),
            distribution=len(results['distribution']),
            dryups=len(results['volume_dryup'])
        )
        
        return results
    
    async def _process_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Process a batch of symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                analysis = await self._analyze_volume(symbol)
                results[symbol] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        return results
    
    async def _analyze_volume(self, symbol: str) -> Dict[str, Any]:
        """Analyze volume patterns for a single symbol."""
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 5)
        
        bars = await self.market_client.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe='1Day'
        )
        
        if not bars or len(bars) < self.lookback_days:
            return self._empty_analysis()
        
        # Calculate metrics
        volumes = [bar.volume for bar in bars]
        closes = [bar.close for bar in bars]
        
        # Current and average volume
        current_volume = volumes[-1]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Price change
        price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
        
        # Volume patterns
        accumulation_score = self._calculate_accumulation_score(
            volumes, closes
        )
        distribution_score = self._calculate_distribution_score(
            volumes, closes
        )
        
        # Volume dryup detection
        days_below_average = sum(
            1 for v in volumes[-5:] if v < avg_volume * 0.5
        )
        
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'has_volume_spike': volume_ratio > self.volume_multiplier,
            'price_change': price_change,
            'accumulation_score': accumulation_score,
            'distribution_score': distribution_score,
            'is_volume_dryup': days_below_average >= 3,
            'days_below_average': days_below_average,
            'pattern': self._identify_pattern(volumes, closes)
        }
    
    def _calculate_accumulation_score(
        self, 
        volumes: List[float], 
        closes: List[float]
    ) -> float:
        """Calculate accumulation score (0-1)."""
        if len(volumes) < 5:
            return 0.0
        
        score = 0.0
        
        # Check for increasing volume on up days
        for i in range(1, min(5, len(volumes))):
            if closes[-i] > closes[-i-1] and volumes[-i] > volumes[-i-1]:
                score += 0.2
        
        # Check for decreasing volume on down days
        for i in range(1, min(5, len(volumes))):
            if closes[-i] < closes[-i-1] and volumes[-i] < volumes[-i-1]:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_distribution_score(
        self, 
        volumes: List[float], 
        closes: List[float]
    ) -> float:
        """Calculate distribution score (0-1)."""
        if len(volumes) < 5:
            return 0.0
        
        score = 0.0
        
        # Check for increasing volume on down days
        for i in range(1, min(5, len(volumes))):
            if closes[-i] < closes[-i-1] and volumes[-i] > volumes[-i-1]:
                score += 0.2
        
        # Check for decreasing volume on up days
        for i in range(1, min(5, len(volumes))):
            if closes[-i] > closes[-i-1] and volumes[-i] < volumes[-i-1]:
                score += 0.1
        
        return min(score, 1.0)
    
    def _identify_pattern(
        self, 
        volumes: List[float], 
        closes: List[float]
    ) -> str:
        """Identify the primary volume pattern."""
        if len(volumes) < 5:
            return "insufficient_data"
        
        # Calculate recent trends
        recent_volumes = volumes[-5:]
        recent_closes = closes[-5:]
        
        avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
        historical_avg = sum(volumes[:-5]) / len(volumes[:-5])
        
        # Identify patterns
        if avg_recent_volume > historical_avg * 1.5:
            if recent_closes[-1] > recent_closes[0]:
                return "bullish_volume_expansion"
            else:
                return "bearish_volume_expansion"
        elif avg_recent_volume < historical_avg * 0.5:
            return "volume_contraction"
        else:
            return "normal_volume"
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            'current_volume': 0,
            'avg_volume': 0,
            'volume_ratio': 0,
            'has_volume_spike': False,
            'price_change': 0,
            'accumulation_score': 0,
            'distribution_score': 0,
            'is_volume_dryup': False,
            'days_below_average': 0,
            'pattern': 'no_data'
        }


async def main():
    """Example usage of volume scanner."""
    
    # Load configuration
    config = get_config(config_name='prod', environment='dev')
    
    # Create scanner
    scanner = VolumeScanner(
        config,
        volume_multiplier=2.5,
        lookback_days=20
    )
    
    # Example symbols to scan
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    print("=== Volume Scanner Example ===\n")
    print(f"Scanning {len(symbols)} symbols...")
    print(f"Volume spike threshold: {scanner.volume_multiplier}x average")
    print(f"Lookback period: {scanner.lookback_days} days\n")
    
    # Run scan
    results = await scanner.scan(symbols)
    
    # Display results
    print("=== Scan Results ===\n")
    
    if results['volume_spikes']:
        print("Volume Spikes Detected:")
        for spike in results['volume_spikes']:
            print(f"  {spike['symbol']}: {spike['volume_ratio']:.1f}x average "
                  f"(Price change: {spike['price_change']*100:.1f}%)")
    else:
        print("No volume spikes detected")
    
    print()
    
    if results['accumulation']:
        print("Accumulation Patterns:")
        for acc in results['accumulation']:
            print(f"  {acc['symbol']}: Score {acc['score']:.2f} - {acc['pattern']}")
    else:
        print("No accumulation patterns detected")
    
    print()
    
    if results['distribution']:
        print("Distribution Patterns:")
        for dist in results['distribution']:
            print(f"  {dist['symbol']}: Score {dist['score']:.2f} - {dist['pattern']}")
    else:
        print("No distribution patterns detected")
    
    print()
    
    if results['volume_dryup']:
        print("Volume Dry-ups:")
        for dryup in results['volume_dryup']:
            print(f"  {dryup['symbol']}: {dryup['days_below_average']} days below average")
    else:
        print("No volume dry-ups detected")
    
    print(f"\n=== Scan Metadata ===")
    print(f"Timestamp: {results['scan_metadata']['timestamp']}")
    print(f"Symbols scanned: {results['scan_metadata']['symbols_scanned']}")


if __name__ == "__main__":
    asyncio.run(main())
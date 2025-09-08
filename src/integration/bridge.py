"""
Bridge Module - Integrates old AI trader components with new clean architecture
This allows us to use the complete trading system from ai_trader_old while
maintaining the clean architecture of the new system.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

# Add old system to path
OLD_SYSTEM_PATH = Path(__file__).parent.parent.parent / "ai_trader_old" / "src"
sys.path.insert(0, str(OLD_SYSTEM_PATH))

# Set environment for old system
os.environ["DATA_LAKE_PATH"] = str(Path(__file__).parent.parent.parent / "ai_trader_old" / "data_lake")


class TradingBridge:
    """Bridge between old and new trading systems."""
    
    def __init__(self):
        """Initialize the bridge with lazy imports."""
        self._technical_scanner = None
        self._volume_scanner = None
        self._ml_strategy = None
        self._feature_engine = None
        self._initialized = False
    
    def initialize(self):
        """Initialize old system components."""
        if self._initialized:
            return
        
        try:
            # Import scanners
            from main.scanners.catalysts.technical_scanner import TechnicalScanner
            from main.scanners.catalysts.volume_scanner import VolumeScanner
            
            # Import strategies
            from main.models.strategies.ml_momentum import MLMomentumStrategy
            from main.models.strategies.mean_reversion import MeanReversionStrategy
            
            # Import feature engine (if available)
            try:
                from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
                self._feature_engine = UnifiedFeatureEngine()
            except ImportError:
                print("⚠️ Feature engine not available, using basic features")
            
            # Initialize scanners
            self._technical_scanner = TechnicalScanner()
            self._volume_scanner = VolumeScanner()
            
            print("✅ Trading bridge initialized successfully")
            self._initialized = True
            
        except ImportError as e:
            print(f"❌ Failed to initialize bridge: {e}")
            print("Using fallback simple scanners")
            self._initialized = False
    
    def scan_technical(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Run technical scanner on symbols."""
        if not self._initialized:
            self.initialize()
        
        if self._technical_scanner:
            try:
                results = self._technical_scanner.scan(symbols)
                return self._convert_scan_results(results)
            except Exception as e:
                print(f"Scanner error: {e}")
        
        # Fallback to simple scanning
        return self._simple_technical_scan(symbols)
    
    def scan_volume(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Run volume scanner on symbols."""
        if not self._initialized:
            self.initialize()
        
        if self._volume_scanner:
            try:
                results = self._volume_scanner.scan(symbols)
                return self._convert_scan_results(results)
            except Exception as e:
                print(f"Volume scanner error: {e}")
        
        return []
    
    def _convert_scan_results(self, results: Any) -> List[Dict[str, Any]]:
        """Convert old system scan results to new format."""
        signals = []
        
        if hasattr(results, '__iter__'):
            for result in results:
                signal = {
                    'symbol': getattr(result, 'symbol', 'UNKNOWN'),
                    'signal_type': getattr(result, 'signal_type', 'UNKNOWN'),
                    'confidence': float(getattr(result, 'confidence', 0.5)),
                    'timestamp': datetime.now(),
                    'metadata': {}
                }
                
                # Add any additional fields
                if hasattr(result, 'reason'):
                    signal['metadata']['reason'] = result.reason
                if hasattr(result, 'price'):
                    signal['metadata']['price'] = float(result.price)
                    
                signals.append(signal)
        
        return signals
    
    def _simple_technical_scan(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fallback simple technical scanner."""
        signals = []
        
        try:
            from paper_trading_alpaca import AlpacaMarketData
            market_data = AlpacaMarketData()
            
            for symbol in symbols:
                try:
                    price = market_data.get_latest_price(symbol)
                    
                    # Simple momentum check (would need historical data for real RSI)
                    # This is just a placeholder
                    if price < Decimal("100"):
                        signals.append({
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'confidence': 0.6,
                            'timestamp': datetime.now(),
                            'metadata': {
                                'reason': 'Low price potential bounce',
                                'price': float(price)
                            }
                        })
                    
                except Exception as e:
                    print(f"Error scanning {symbol}: {e}")
                    
        except ImportError:
            print("❌ AlpacaMarketData not available")
        
        return signals
    
    def get_ml_predictions(self, symbols: List[str]) -> Dict[str, float]:
        """Get ML model predictions for symbols."""
        predictions = {}
        
        # Placeholder - would load actual ML models
        for symbol in symbols:
            # Random prediction for now
            import random
            predictions[symbol] = random.uniform(0.3, 0.8)
        
        return predictions


class SignalAggregator:
    """Aggregates signals from multiple sources."""
    
    def __init__(self, bridge: TradingBridge):
        self.bridge = bridge
        self.signals = []
    
    def collect_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Collect signals from all available scanners."""
        all_signals = []
        
        # Technical signals
        tech_signals = self.bridge.scan_technical(symbols)
        for signal in tech_signals:
            signal['source'] = 'technical'
        all_signals.extend(tech_signals)
        
        # Volume signals
        vol_signals = self.bridge.scan_volume(symbols)
        for signal in vol_signals:
            signal['source'] = 'volume'
        all_signals.extend(vol_signals)
        
        # ML predictions
        ml_predictions = self.bridge.get_ml_predictions(symbols)
        for symbol, confidence in ml_predictions.items():
            if confidence > 0.65:  # Only high confidence
                all_signals.append({
                    'symbol': symbol,
                    'signal_type': 'BUY' if confidence > 0.7 else 'HOLD',
                    'confidence': confidence,
                    'source': 'ml_model',
                    'timestamp': datetime.now(),
                    'metadata': {'ml_score': confidence}
                })
        
        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_signals
    
    def get_top_signals(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N signals by confidence."""
        signals = self.collect_signals(['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN'])
        return signals[:n]


def test_bridge():
    """Test the bridge functionality."""
    print("Testing Trading Bridge...")
    
    bridge = TradingBridge()
    bridge.initialize()
    
    # Test scanners
    symbols = ['AAPL', 'MSFT', 'NVDA']
    
    print(f"\nScanning {symbols}...")
    signals = bridge.scan_technical(symbols)
    
    if signals:
        print(f"Found {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal['symbol']}: {signal['signal_type']} (confidence: {signal['confidence']:.2f})")
    else:
        print("No signals found")
    
    # Test aggregator
    print("\nTesting Signal Aggregator...")
    aggregator = SignalAggregator(bridge)
    top_signals = aggregator.get_top_signals(3)
    
    print(f"Top {len(top_signals)} signals:")
    for signal in top_signals:
        print(f"  {signal['symbol']}: {signal['signal_type']} from {signal['source']} (confidence: {signal['confidence']:.2f})")


if __name__ == "__main__":
    test_bridge()
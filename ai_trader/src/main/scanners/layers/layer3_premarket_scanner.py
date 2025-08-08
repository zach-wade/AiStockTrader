"""
Layer 3: Pre-Market Scanner
Real-time scanner that runs 4:00 AM - 9:30 AM ET
Filters ~100-200 catalyst symbols down to ~10-30 trading candidates
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta, time
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy import text
from collections import defaultdict

from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.repositories import get_repository_factory
from main.interfaces.repositories import ICompanyRepository
from main.data_pipeline.ingestion.clients.polygon_market_client import PolygonMarketClient
from main.config.config_manager import get_config
from main.utils.cache import get_global_cache, CacheType
from main.data_pipeline.core.enums import DataLayer
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.interfaces.events import IEventBus

logger = logging.getLogger(__name__)


class Layer3PreMarketScanner:
    """
    Real-time pre-market scanner that identifies the best trading candidates.
    Focuses on relative volume (RVOL), price movement, and catalyst strength.
    """
    
    def __init__(self, config: Any = None, event_bus: IEventBus = None):
        if config is None:
            config = get_config()
        self.config = config
        
        # Initialize event publisher for layer qualification events
        self.event_publisher = ScannerEventPublisher(event_bus) if event_bus else None
        
        # Initialize database and clients
        from main.utils.database import DatabasePool
        
        self.db_pool = DatabasePool()
        self.db_pool.initialize()
        
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        self.company_repo: ICompanyRepository = repo_factory.create_company_repository(self.db_adapter)
        
        # Initialize market data client - Polygon only per architecture
        try:
            self.polygon_client = PolygonMarketClient(config)
            self.use_polygon = True
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            raise RuntimeError("Polygon client is required for market data")
        
        # Pre-market thresholds
        self.min_rvol = 2.0  # Minimum 2x average volume
        self.min_price_change = 0.02  # Minimum 2% price movement
        self.min_premarket_volume = 10000  # Minimum pre-market shares traded
        self.min_catalyst_score = 3.0  # From Layer 2
        
        # RVOL calculation parameters
        self.rvol_lookback_days = 20  # 20-day average volume
        self.rvol_time_buckets = 5  # 5-minute buckets for intraday RVOL
        
        # Real-time tracking
        self.premarket_data = {}  # Symbol -> latest pre-market data
        self.cache = get_global_cache()  # Use global cache for RVOL patterns
        self.last_update_time = {}  # Symbol -> last update timestamp
        
    def is_premarket_hours(self) -> bool:
        """Check if currently in pre-market hours (4:00 AM - 9:30 AM ET)"""
        now = datetime.now(timezone.utc)
        et_now = now.astimezone(timezone(timedelta(hours=-5)))  # ET timezone
        
        # Pre-market: 4:00 AM - 9:30 AM ET
        premarket_start = time(4, 0)
        premarket_end = time(9, 30)
        
        return premarket_start <= et_now.time() <= premarket_end
    
    async def run(self, input_symbols: List[str]) -> List[str]:
        """
        Run Layer 3 scanning and return filtered symbols.
        Wrapper around scan_premarket for funnel compatibility.
        """
        result = await self.scan_premarket(input_symbols)
        return result.get('final_watchlist', [])
    
    async def scan_premarket(self, input_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan pre-market activity for Layer 2 qualified symbols.
        This should run continuously during pre-market hours.
        """
        logger.info("Starting Layer 3 Pre-Market Scanner...")
        start_time = datetime.now(timezone.utc)
        
        # Check if we're in pre-market hours
        if not self.is_premarket_hours():
            logger.warning("Not in pre-market hours (4:00 AM - 9:30 AM ET)")
            # For testing, continue anyway
        
        try:
            # Step 1: Get input universe (Layer 2 qualified symbols)
            if input_symbols:
                universe = input_symbols
                logger.info(f"Using {len(universe)} symbols from pipeline input")
            else:
                # Fallback to database query if no input provided (for standalone runs)
                universe = await self._get_layer2_universe()
                logger.info(f"Using {len(universe)} symbols from database (standalone mode)")
            
            # Store input symbols for database update
            self._last_input_symbols = universe
            
            if not universe:
                logger.warning("No input symbols for Layer 3 pre-market scanner")
                return {
                    'final_watchlist': [],
                    'qualified_symbols': [],
                    'scan_metadata': {'reason': 'no_input_symbols'}
                }
            
            # Step 2: Load historical volume patterns for RVOL calculation
            await self._load_rvol_baselines(universe)
            
            # Step 3: Get real-time pre-market data
            premarket_results = await self._scan_realtime_data(universe)
            
            # Step 4: Calculate RVOL and filter
            qualified_symbols = await self._apply_premarket_filters(premarket_results)
            
            # Step 5: Rank by opportunity score
            ranked_symbols = self._rank_opportunities(qualified_symbols)
            
            # Step 6: Update database
            await self._update_database(ranked_symbols)
            
            # Step 7: Generate report
            return self._generate_report(
                start_time=start_time,
                universe=universe,
                premarket_results=premarket_results,
                qualified_symbols=ranked_symbols
            )
            
        except Exception as e:
            logger.error(f"Error in Layer 3 pre-market scan: {e}")
            raise
    
    async def _get_layer2_universe(self) -> List[str]:
        """Get Layer 2 qualified symbols from database."""
        query = text("""
            SELECT symbol, catalyst_score
            FROM companies 
            WHERE layer >= 2 
            AND is_active = true
            ORDER BY catalyst_score DESC
        """)
        
        def execute_query(session):
            result = session.execute(query)
            return [(row.symbol, row.catalyst_score) for row in result]
        
        results = await self.db_adapter.run_sync(execute_query)
        
        # Store catalyst scores for later use
        self.catalyst_scores = {symbol: score for symbol, score in results}
        
        return [symbol for symbol, _ in results]
    
    async def _load_rvol_baselines(self, symbols: List[str]):
        """Load historical volume patterns for RVOL calculation."""
        logger.info("Loading RVOL baselines...")
        
        # Query for average volume by time of day
        query = text(f"""
            WITH volume_by_time AS (
                SELECT 
                    symbol,
                    EXTRACT(hour FROM timestamp AT TIME ZONE 'America/New_York') as hour,
                    EXTRACT(minute FROM timestamp AT TIME ZONE 'America/New_York') as minute,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as std_volume
                FROM market_data
                WHERE symbol = ANY(:symbols)
                AND timestamp >= NOW() - INTERVAL '{self.rvol_lookback_days} days'
                AND EXTRACT(hour FROM timestamp AT TIME ZONE 'America/New_York') BETWEEN 4 AND 16
                GROUP BY symbol, hour, minute
            )
            SELECT * FROM volume_by_time
            ORDER BY symbol, hour, minute
        """)
        
        def execute_query(session):
            result = session.execute(query, {
                'symbols': symbols
            })
            return [row._mapping for row in result]
        
        volume_data = await self.db_adapter.run_sync(execute_query)
        
        # Organize by symbol and time
        symbols_processed = set()
        for row in volume_data:
            symbol = row['symbol']
            symbols_processed.add(symbol)
            
            time_key = f"{int(row['hour']):02d}:{int(row['minute']):02d}"
            rvol_data = {
                'avg_volume': float(row['avg_volume'] or 0),
                'std_volume': float(row['std_volume'] or 0)
            }
            
            # Store in cache with symbol+time_key as cache key
            cache_key = f"{symbol}:{time_key}"
            # Use centralized TTL for metrics
            metrics_ttl = self.config.get('cache', {}).get('ttl', {}).get('metrics', 3600)
            await self.cache.set(CacheType.METRICS, cache_key, rvol_data, ttl_seconds=metrics_ttl)
        
        logger.info(f"Loaded RVOL baselines for {len(symbols_processed)} symbols")
    
    async def _scan_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time pre-market data for symbols."""
        logger.info("Fetching real-time pre-market data...")
        results = {}
        
        # Batch symbols for efficiency
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            try:
                if self.use_polygon:
                    # Polygon has better pre-market data
                    batch_data = await self._get_polygon_premarket_data(batch)
                else:
                    # Fallback to Alpaca
                    batch_data = await self._get_alpaca_premarket_data(batch)
                
                results.update(batch_data)
                
            except Exception as e:
                logger.error(f"Error fetching batch {i//batch_size}: {e}")
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"Fetched pre-market data for {len(results)} symbols")
        return results
    
    async def _get_alpaca_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get pre-market data from Alpaca."""
        data = {}
        
        # Get minute bars for pre-market (or recent data for testing)
        now = datetime.now(timezone.utc)
        
        # For testing outside pre-market, get last hour of data
        if not self.is_premarket_hours():
            start = now - timedelta(hours=1)
        else:
            start = now.replace(hour=8, minute=0, second=0, microsecond=0)  # 4 AM ET
        
        for symbol in symbols:
            try:
                # Get bars as DataFrame from Polygon
                bars_df = await self.polygon_client.fetch_market_data(
                    symbol=symbol,
                    start=start,
                    end=now,
                    timeframe='1Min'
                )
                
                if bars_df is not None and not bars_df.empty:
                    # Calculate pre-market metrics from DataFrame
                    pm_volume = bars_df['volume'].sum()
                    
                    # Calculate VWAP
                    if pm_volume > 0:
                        pm_vwap = (bars_df['volume'] * bars_df['close']).sum() / pm_volume
                    else:
                        pm_vwap = bars_df['close'].mean()
                    
                    pm_high = bars_df['high'].max()
                    pm_low = bars_df['low'].min()
                    pm_close = bars_df['close'].iloc[-1]
                    
                    # Get previous day close
                    prev_close = await self._get_previous_close(symbol)
                    
                    data[symbol] = {
                        'timestamp': now,
                        'price': pm_close,
                        'volume': pm_volume,
                        'vwap': pm_vwap,
                        'high': pm_high,
                        'low': pm_low,
                        'prev_close': prev_close,
                        'price_change': (pm_close - prev_close) / prev_close if prev_close else 0,
                        'spread': 0.01,  # Default spread for testing
                        'bars_count': len(bars_df)
                    }
                    
            except Exception as e:
                logger.debug(f"Error getting pre-market data for {symbol}: {e}")
        
        return data
    
    async def _get_polygon_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get pre-market data from Polygon (better pre-market coverage)."""
        # Implementation would use Polygon's aggregate bars API
        # This is a placeholder - actual implementation would depend on Polygon client
        return await self._get_alpaca_premarket_data(symbols)
    
    async def _get_previous_close(self, symbol: str) -> float:
        """Get previous trading day's closing price."""
        query = text("""
            SELECT close
            FROM market_data
            WHERE symbol = :symbol
            AND timestamp < DATE_TRUNC('day', NOW())
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        def execute_query(session):
            result = session.execute(query, {'symbol': symbol})
            row = result.first()
            return float(row.close) if row else 0.0
        
        return await self.db_adapter.run_sync(execute_query)
    
    async def _calculate_rvol(self, symbol: str, current_volume: float, current_time: datetime) -> float:
        """Calculate relative volume compared to historical average."""
        # Get time bucket
        hour = current_time.hour
        minute = (current_time.minute // 5) * 5  # 5-minute buckets
        time_key = f"{hour:02d}:{minute:02d}"
        
        # Get baseline from cache
        cache_key = f"{symbol}:{time_key}"
        baseline = await self.cache.get(CacheType.METRICS, cache_key) or {}
        avg_volume = baseline.get('avg_volume', 0)
        
        if avg_volume > 0:
            return current_volume / avg_volume
        
        return 0.0
    
    async def _apply_premarket_filters(self, premarket_data: Dict[str, Dict]) -> List[Dict]:
        """Apply pre-market filters and calculate scores."""
        qualified = []
        
        for symbol, data in premarket_data.items():
            # Calculate RVOL
            rvol = await self._calculate_rvol(symbol, data['volume'], data['timestamp'])
            
            # Apply filters
            if (rvol >= self.min_rvol and
                abs(data['price_change']) >= self.min_price_change and
                data['volume'] >= self.min_premarket_volume):
                
                # Calculate pre-market score
                pm_score = self._calculate_premarket_score(symbol, data, rvol)
                
                qualified.append({
                    'symbol': symbol,
                    'rvol': rvol,
                    'price_change': data['price_change'],
                    'volume': data['volume'],
                    'price': data['price'],
                    'vwap': data['vwap'],
                    'spread': data['spread'],
                    'premarket_score': pm_score,
                    'catalyst_score': self.catalyst_scores.get(symbol, 0),
                    'data': data
                })
        
        return qualified
    
    def _calculate_premarket_score(self, symbol: str, data: Dict, rvol: float) -> float:
        """Calculate pre-market opportunity score."""
        score = 0.0
        
        # RVOL component (0-5 points)
        rvol_score = min(5.0, rvol)
        score += rvol_score
        
        # Price movement component (0-3 points)
        price_move = abs(data['price_change'])
        price_score = min(3.0, price_move * 100)  # 1 point per 1%
        score += price_score
        
        # Volume component (0-2 points)
        volume_score = min(2.0, data['volume'] / 100000)  # 1 point per 100k shares
        score += volume_score
        
        # Catalyst strength bonus (from Layer 2)
        catalyst_score = self.catalyst_scores.get(symbol, 0)
        score += catalyst_score * 0.5  # Half weight
        
        # Price/VWAP relationship (0-1 point)
        if data['vwap'] > 0:
            vwap_score = 1.0 if data['price'] > data['vwap'] else 0.5
            score += vwap_score
        
        return score
    
    def _rank_opportunities(self, qualified: List[Dict]) -> List[Dict]:
        """Rank opportunities by composite score."""
        # Calculate final scores
        for item in qualified:
            # Combine pre-market score with catalyst score
            item['final_score'] = (
                item['premarket_score'] * 0.7 +  # 70% weight on pre-market activity
                item['catalyst_score'] * 0.3      # 30% weight on catalysts
            )
        
        # Sort by final score
        qualified.sort(key=lambda x: x['final_score'], reverse=True)
        
        return qualified
    
    async def _update_database(self, qualified_symbols: List[Dict]):
        """Update Layer 3 (ACTIVE) qualification status in database."""
        if not qualified_symbols:
            return
        
        try:
            # Extract qualified symbols and scores
            qualified_list = [s['symbol'] for s in qualified_symbols]
            premarket_scores = {s['symbol']: s.get('premarket_score', 0.0) for s in qualified_symbols}
            
            # Get all symbols that were evaluated (from instance variable if available)
            all_input_symbols = getattr(self, '_last_input_symbols', qualified_list)
            
            # Update qualified symbols to Layer 3
            qualified_count = 0
            promoted_count = 0
            
            for symbol_data in qualified_symbols:
                symbol = symbol_data['symbol']
                
                # Get current layer for the symbol
                current = await self.company_repo.get_by_symbol(symbol)
                if current:
                    current_layer = current.get('layer', 0)
                    
                    # Update to Layer 3 (ACTIVE)
                    result = await self.company_repo.update_layer(
                        symbol=symbol,
                        layer=DataLayer.ACTIVE,
                        metadata={
                            'premarket_score': premarket_scores.get(symbol, 0.0),
                            'rvol': symbol_data.get('rvol', 0.0),
                            'price_change': symbol_data.get('price_change', 0.0),
                            'source': 'layer3_scanner',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    if result.success:
                        qualified_count += 1
                        
                        # Publish event based on whether this is qualification or promotion
                        if self.event_publisher:
                            if current_layer < 3:
                                # This is a promotion to Layer 3
                                await self.event_publisher.publish_symbol_promoted(
                                    symbol=symbol,
                                    from_layer=DataLayer(current_layer),
                                    to_layer=DataLayer.ACTIVE,
                                    promotion_reason="High pre-market activity detected",
                                    metrics={
                                        'premarket_score': premarket_scores.get(symbol, 0.0),
                                        'rvol': symbol_data.get('rvol', 0.0),
                                        'price_change': symbol_data.get('price_change', 0.0)
                                    }
                                )
                                promoted_count += 1
                            else:
                                # Re-qualification at same layer
                                await self.event_publisher.publish_symbol_qualified(
                                    symbol=symbol,
                                    layer=DataLayer.ACTIVE,
                                    qualification_reason="High pre-market activity detected",
                                    metrics={
                                        'premarket_score': premarket_scores.get(symbol, 0.0),
                                        'rvol': symbol_data.get('rvol', 0.0)
                                    }
                                )
                    else:
                        logger.warning(f"Failed to update layer for {symbol}: {result.errors}")
            
            # Optionally downgrade non-qualified symbols
            non_qualified = set(all_input_symbols) - set(qualified_list)
            for symbol in non_qualified:
                current = await self.company_repo.get_by_symbol(symbol)
                if current and current.get('layer', 0) == 3:
                    # Downgrade from Layer 3 if no longer qualified
                    await self.company_repo.update_layer(
                        symbol=symbol,
                        layer=DataLayer.CATALYST,  # Downgrade to Layer 2
                        metadata={'reason': 'No significant pre-market activity'}
                    )
            
            logger.info(
                f"✅ Updated Layer 3 qualifications: "
                f"{qualified_count} qualified ({promoted_count} promoted)"
            )
                
        except Exception as e:
            logger.error(f"Error updating Layer 3 qualifications: {e}", exc_info=True)
            # Don't fail the scan if qualification update fails
        
        # Original update logic for additional fields like rvol
        # This can be kept for backward compatibility or additional data
        update_query = text("""
            UPDATE companies 
            SET rvol = :rvol
            WHERE symbol = :symbol
        """)
        
        def execute_updates(session):
            # Clear old qualifications
            session.execute(clear_query)
            
            # Update new qualifications
            for item in qualified_symbols:
                session.execute(update_query, {
                    'symbol': item['symbol'],
                    'score': item['final_score'],
                    'rvol': item['rvol']
                })
            
            session.commit()
        
        await self.db_adapter.run_sync(execute_updates)
        logger.info(f"Updated Layer 3 qualification for {len(qualified_symbols)} symbols")
    
    def _generate_report(self, start_time: datetime, universe: List[str],
                        premarket_results: Dict, qualified_symbols: List[Dict]) -> Dict:
        """Generate comprehensive pre-market scan report."""
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Calculate statistics
        rvol_distribution = [s['rvol'] for s in qualified_symbols]
        price_changes = [s['price_change'] for s in qualified_symbols]
        
        report = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'market_time': 'pre-market' if self.is_premarket_hours() else 'outside-hours',
            'input_count': len(universe),
            'scanned_count': len(premarket_results),
            'output_count': len(qualified_symbols),
            'statistics': {
                'average_rvol': np.mean(rvol_distribution) if rvol_distribution else 0,
                'max_rvol': max(rvol_distribution) if rvol_distribution else 0,
                'average_price_change': np.mean([abs(p) for p in price_changes]) if price_changes else 0,
                'bullish_count': sum(1 for p in price_changes if p > 0),
                'bearish_count': sum(1 for p in price_changes if p < 0),
                'total_premarket_volume': sum(s['volume'] for s in qualified_symbols)
            },
            'top_opportunities': [
                {
                    'symbol': s['symbol'],
                    'final_score': s['final_score'],
                    'rvol': s['rvol'],
                    'price_change': s['price_change'],
                    'volume': s['volume'],
                    'catalyst_score': s['catalyst_score']
                }
                for s in qualified_symbols[:10]  # Top 10
            ]
        }
        
        return report


async def run_premarket_scan():
    """Run the Layer 3 pre-market scan."""
    # Environment variables are loaded by config_manager
    from config import get_config
    
    config = get_config()
    scanner = Layer3PreMarketScanner(config)
    
    logger.info("=" * 60)
    logger.info("Starting Layer 3 Pre-Market Scanner")
    logger.info("=" * 60)
    
    results = await scanner.scan_premarket()
    
    # Save results
    import json
    from pathlib import Path
    
    output_dir = Path('data/universe/layer3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"premarket_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Found {results['output_count']} pre-market opportunities")
    
    # Print top opportunities
    if results['output_count'] > 0:
        logger.info("\nTop Pre-Market Opportunities:")
        for i, opp in enumerate(results['top_opportunities'][:5], 1):
            logger.info(
                f"{i}. {opp['symbol']}: "
                f"Score={opp['final_score']:.2f}, "
                f"RVOL={opp['rvol']:.1f}x, "
                f"Change={opp['price_change']*100:.1f}%"
            )
    
    return results


if __name__ == "__main__":
    asyncio.run(run_premarket_scan())
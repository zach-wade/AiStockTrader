"""
Layer 1.5: Strategy Affinity Filter
Bridge between static universe and dynamic catalyst search using pre-computed affinity scores.

Purpose: Filter symbols by strategy-regime compatibility to optimize strategy selection.
Reduces Layer 1 universe (~1,500 symbols) to regime-compatible subset (~500 symbols).
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path

from main.interfaces.database import IAsyncDatabase
from main.feature_pipeline.calculators.market_regime import MarketRegimeCalculator
from main.feature_pipeline.calculators.strategy_affinity_calculator import StrategyAffinityCalculator
from main.config.config_manager import get_config
from main.data_pipeline.core.enums import DataLayer
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.interfaces.events import IEventBus
from main.data_pipeline.storage.repositories.company_repository import CompanyRepository

logger = logging.getLogger(__name__)


class Layer1_5_StrategyAffinityFilter:
    """
    Bridge between static universe and dynamic catalyst search.
    Filters symbols by strategy-regime compatibility using pre-computed affinity scores.
    """
    
    def __init__(self, config: Any, db_adapter: IAsyncDatabase, event_bus: IEventBus = None):
        """
        Initializes the Layer 1.5 filter.

        Args:
            config: The main application configuration object.
            db_adapter: An instance of IAsyncDatabase for all DB interactions.
            event_bus: Optional event bus for publishing layer qualification events.
        """
        self.config = config
        self.db_adapter = db_adapter
        
        # Initialize event publisher and company repository
        from main.data_pipeline.storage.repositories import get_repository_factory
        factory = get_repository_factory()
        self.event_publisher = ScannerEventPublisher(event_bus) if event_bus else None
        self.company_repository = factory.create_company_repository(self.db_adapter)
        self.market_regime_analytics = MarketRegimeCalculator(config)
        self.strategy_affinity_calculator = StrategyAffinityCalculator(config, db_adapter)
        
        # Load all parameters from the configuration object
        self.params = self.config.get('layer1_5', {})
        self.min_affinity_score = self.params.get('min_affinity_score', 0.5)
        self.auto_calculate_affinities = self.params.get('auto_calculate_affinities', True)
        self.max_batch_size = self.params.get('affinity_batch_size', 100)
        self.max_output_symbols = self.params.get('max_output_symbols', 500)
        self.enable_regime_filtering = self.params.get('enable_regime_filtering', True)
        
        # The strategy-regime compatibility matrix is now loaded from config
        self.regime_strategy_preferences = self.params.get('regime_strategy_preferences', {
            'unknown': ['momentum', 'breakout', 'mean_reversion', 'sentiment']
        })
        self.fallback_strategy = self.params.get('fallback_strategy', 'momentum')
        
        self.output_dir = Path(self.config.get('paths.universe_dir', 'data/universe')) / 'layer1_5'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, input_symbols: List[str]) -> List[str]:
        """
        Filters the input universe by strategy-regime compatibility.

        Args:
            input_symbols: The list of liquid symbols passed from Layer 1.
            
        Returns:
            A refined list of symbols qualified for Layer 2 catalyst scanning.
        """
        logger.info(f"🎯 Starting Layer 1.5 Strategy Affinity Filter for {len(input_symbols)} symbols...")
        if not input_symbols:
            logger.warning("Input symbol list is empty. Aborting Layer 1.5 filter.")
            return []

        # The rest of your run method is perfectly implemented and remains the same.
        # It correctly calls the helper methods in sequence.
        start_time = datetime.now(timezone.utc)
        
        try:
            current_regime = await self._get_current_market_regime()
            affinity_scores = await self._load_affinity_scores(input_symbols)
            
            # Identify symbols without affinity scores and calculate them if enabled
            symbols_with_scores = set(affinity_scores.keys())
            symbols_without_scores = [symbol for symbol in input_symbols if symbol not in symbols_with_scores]
            
            if symbols_without_scores:
                logger.info(f"📊 Found {len(symbols_without_scores)} symbols without affinity scores")
                if self.auto_calculate_affinities:
                    # Calculate missing affinities
                    calculated_affinities = await self._calculate_missing_affinities(symbols_without_scores)
                    # Merge calculated affinities with existing ones
                    affinity_scores.update(calculated_affinities)
                    logger.info(f"📈 Total symbols with affinity scores: {len(affinity_scores)}")
                else:
                    logger.info("⏭️  Auto-calculation disabled, skipping symbols without scores")
            
            qualified_symbols = self._apply_strategy_affinity_filter(
                input_symbols, current_regime, affinity_scores
            )
            final_symbols = self._rank_and_limit_symbols(qualified_symbols)
            
            await self._save_layer1_5_output(final_symbols, current_regime, affinity_scores)
            
            # Update layer qualifications for qualified symbols (Layer 1.5 doesn't have its own layer value,
            # but we can track metadata for these symbols)
            await self._update_layer_metadata(final_symbols, affinity_scores)
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✅ Layer 1.5 Filter: {len(input_symbols)} -> {len(final_symbols)} symbols in {duration:.2f}s")
            
            return final_symbols
            
        except Exception as e:
            logger.error(f"Error in Layer 1.5 Strategy Affinity Filter: {e}", exc_info=True)
            return input_symbols # Fallback: return input universe on error
    
    async def _update_layer_metadata(self, symbols: List[str], affinity_scores: Dict[str, Dict[str, Any]]):
        """Update metadata for Layer 1.5 qualified symbols.
        
        Layer 1.5 doesn't change the layer value, but tracks strategy affinity metadata.
        """
        try:
            for symbol in symbols:
                # Get the affinity data for this symbol
                affinity_data = affinity_scores.get(symbol, {})
                
                # Get current company data
                current = await self.company_repository.get_company(symbol)
                if not current:
                    logger.warning(f"Symbol {symbol} not found in database")
                    continue
                
                # Build scanner metadata structure
                import json
                existing_metadata = current.get('scanner_metadata', {})
                if isinstance(existing_metadata, str):
                    try:
                        existing_metadata = json.loads(existing_metadata)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing_metadata = {}
                elif not isinstance(existing_metadata, dict):
                    existing_metadata = {}
                
                # Update layer 1.5 specific metadata
                existing_metadata['layer1_5'] = {
                    'qualified': True,
                    'strategy_affinity': affinity_data,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Update company metadata (not layer)
                result = await self.company_repository.update_company_metadata(
                    symbol=symbol,
                    metadata={'scanner_metadata': json.dumps(existing_metadata)}
                )
                
                if result.success and self.event_publisher:
                    # Get current layer for event
                    current_layer = current.get('layer', 1)
                    
                    # Publish metadata update event
                    await self.event_publisher.publish_symbol_qualified(
                        symbol=symbol,
                        layer=current_layer,
                        qualification_reason="Symbol has high strategy affinity",
                        metrics={
                            'affinity_score': affinity_data.get('total_score', 0),
                            'strategy_affinity': True
                        }
                    )
            
            logger.info(f"Updated Layer 1.5 metadata for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating layer metadata: {e}", exc_info=True)
    
    async def _load_layer1_universe(self) -> List[str]:
        """Load Layer 1 qualified symbols from database."""
        query = """
            SELECT symbol 
            FROM companies 
            WHERE layer >= 1 
            AND is_active = true
            ORDER BY liquidity_score DESC
        """
        
        try:
            rows = await self.db_adapter.fetch_all(query)
            return [row['symbol'] for row in rows]
        except Exception as e:
            logger.error(f"Error loading Layer 1 universe: {e}")
            return []
    
    async def _get_current_market_regime(self) -> str:
        """Determine current market regime using MarketRegimeAnalytics."""
        if not self.enable_regime_filtering:
            logger.info("Regime filtering disabled - using 'unknown' regime")
            return 'unknown'
        
        try:
            regime = await self.market_regime_analytics.get_current_market_regime()
            return regime
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}. Using 'unknown'")
            return 'unknown'
    
    async def _load_affinity_scores(self, universe: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load pre-computed affinity scores from database."""
        if not universe:
            return {}
        
        # Load most recent affinity scores
        query = """
            SELECT 
                symbol,
                momentum_affinity,
                mean_reversion_affinity,
                breakout_affinity,
                sentiment_affinity,
                best_strategy,
                composite_scores_json,
                timestamp
            FROM strategy_affinity_scores 
            WHERE symbol = ANY($1)
            AND timestamp >= $2
            ORDER BY symbol, timestamp DESC
        """
        
        # Get scores from last 7 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
        
        try:
            # Execute query directly using asyncpg
            async with self.db_adapter._pool.acquire() as conn:
                rows = await conn.fetch(query, universe, cutoff_date)
            
            # Get most recent score per symbol
            affinity_data = {}
            for row in rows:
                symbol = row['symbol']
                if symbol not in affinity_data:  # First (most recent) record for this symbol
                    affinity_data[symbol] = {
                        'momentum_affinity': row['momentum_affinity'] or 0.0,
                        'mean_reversion_affinity': row['mean_reversion_affinity'] or 0.0,
                        'breakout_affinity': row['breakout_affinity'] or 0.0,
                        'sentiment_affinity': row['sentiment_affinity'] or 0.0,
                        'best_strategy': row['best_strategy'],
                        'composite_scores': json.loads(row['composite_scores_json']) if row['composite_scores_json'] else {},
                        'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None
                    }
            
            return affinity_data
        except Exception as e:
            logger.error(f"Error loading affinity scores: {e}")
            return {}
    
    async def _calculate_missing_affinities(self, symbols_without_scores: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculate strategy affinities for symbols that don't have recent scores."""
        if not symbols_without_scores or not self.auto_calculate_affinities:
            return {}
        
        logger.info(f"🔄 Calculating strategy affinities for {len(symbols_without_scores)} symbols...")
        
        calculated_affinities = {}
        
        # Process symbols in batches to avoid overwhelming the system
        for i in range(0, len(symbols_without_scores), self.max_batch_size):
            batch = symbols_without_scores[i:i + self.max_batch_size]
            logger.info(f"   Processing batch {i//self.max_batch_size + 1}: {len(batch)} symbols")
            
            try:
                # Calculate affinities for this batch
                batch_affinities = await self.strategy_affinity_calculator.calculate_affinities_batch(batch)
                
                if batch_affinities:
                    # Save to database for future use
                    await self.strategy_affinity_calculator.save_affinities_to_database(batch_affinities)
                    
                    # Convert to the format expected by the filter
                    for symbol, scores in batch_affinities.items():
                        calculated_affinities[symbol] = {
                            'momentum_affinity': scores['momentum_affinity'],
                            'mean_reversion_affinity': scores['mean_reversion_affinity'],
                            'breakout_affinity': scores['breakout_affinity'],
                            'sentiment_affinity': scores['sentiment_affinity'],
                            'best_strategy': max(scores.items(), key=lambda x: x[1])[0].replace('_affinity', ''),
                            'timestamp': datetime.now(timezone.utc)
                        }
                        
                    logger.info(f"   ✅ Successfully calculated affinities for {len(batch_affinities)} symbols")
                else:
                    logger.warning(f"   ⚠️ No affinities calculated for batch starting at index {i}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error calculating affinities for batch starting at index {i}: {e}")
                continue
        
        logger.info(f"🔄 Completed affinity calculation: {len(calculated_affinities)} symbols processed")
        return calculated_affinities
    
    def _apply_strategy_affinity_filter(
        self, 
        universe: List[str], 
        current_regime: str, 
        affinity_scores: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply strategy affinity filtering based on current market regime."""
        qualified_symbols = []
        
        # Get preferred strategies for current regime
        preferred_strategies = self.regime_strategy_preferences.get(current_regime, ['momentum'])
        logger.info(f"Preferred strategies for {current_regime}: {preferred_strategies}")
        
        stats = {
            'total_processed': 0,
            'with_affinity_scores': 0,
            'regime_qualified': 0,
            'affinity_qualified': 0
        }
        
        for symbol in universe:
            stats['total_processed'] += 1
            
            # Check if we have affinity scores for this symbol
            if symbol not in affinity_scores:
                # No affinity scores available - skip
                continue
                
            stats['with_affinity_scores'] += 1
            symbol_affinities = affinity_scores[symbol]
            
            # Strategy-regime compatibility analysis
            regime_compatibility = self._calculate_regime_compatibility(
                symbol_affinities, preferred_strategies
            )
            
            if regime_compatibility['is_compatible']:
                stats['regime_qualified'] += 1
                
                # Check if any strategy meets minimum affinity threshold
                if regime_compatibility['max_affinity_score'] >= self.min_affinity_score:
                    stats['affinity_qualified'] += 1
                    
                    qualified_symbols.append({
                        'symbol': symbol,
                        'regime_compatibility': regime_compatibility,
                        'affinity_scores': symbol_affinities,
                        'composite_score': regime_compatibility['max_affinity_score'],
                        'best_strategy': regime_compatibility['best_strategy'],
                        'selection_reason': regime_compatibility['selection_reason']
                    })
        
        logger.info(f"Affinity filter statistics: {stats}")
        return qualified_symbols
    
    def _calculate_regime_compatibility(
        self, 
        symbol_affinities: Dict[str, Any], 
        preferred_strategies: List[str]
    ) -> Dict[str, Any]:
        """Calculate regime compatibility for a symbol."""
        strategy_scores = []
        
        # Check each preferred strategy
        for strategy in preferred_strategies:
            affinity_key = f'{strategy}_affinity'
            if affinity_key in symbol_affinities:
                affinity_score = symbol_affinities[affinity_key]
                if affinity_score > 0:
                    strategy_scores.append((strategy, affinity_score))
        
        # If no preferred strategies have good scores, try fallback
        if not strategy_scores:
            fallback_key = f'{self.fallback_strategy}_affinity'
            if fallback_key in symbol_affinities:
                fallback_score = symbol_affinities[fallback_key]
                if fallback_score > 0:
                    strategy_scores.append((self.fallback_strategy, fallback_score))
        
        # Determine best strategy and compatibility
        if strategy_scores:
            # Sort by affinity score
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            best_strategy, max_score = strategy_scores[0]
            
            return {
                'is_compatible': True,
                'best_strategy': best_strategy,
                'max_affinity_score': max_score,
                'available_strategies': strategy_scores,
                'selection_reason': f'{best_strategy} affinity: {max_score:.3f}'
            }
        else:
            return {
                'is_compatible': False,
                'best_strategy': None,
                'max_affinity_score': 0.0,
                'available_strategies': [],
                'selection_reason': 'No viable strategy affinities'
            }
    
    def _rank_and_limit_symbols(
        self, 
        qualified_symbols: List[Dict[str, Any]]
    ) -> List[str]:
        """Rank symbols by composite score and limit to max output."""
        # Sort by composite score (descending)
        qualified_symbols.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Limit to max output
        limited_symbols = qualified_symbols[:self.max_output_symbols]
        
        # Extract symbol names
        return [item['symbol'] for item in limited_symbols]
    
    async def _save_layer1_5_output(
        self, 
        final_symbols: List[str], 
        current_regime: str, 
        affinity_scores: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save Layer 1.5 output to standardized location."""
        timestamp = datetime.now(timezone.utc)
        
        # Create output data
        output_data = {
            'timestamp': timestamp.isoformat(),
            'layer': '1.5',
            'description': 'Strategy affinity filtered symbols',
            'symbol_count': len(final_symbols),
            'symbols': final_symbols,
            'filter_criteria': {
                'market_regime': current_regime,
                'min_affinity_score': self.min_affinity_score,
                'max_output_symbols': self.max_output_symbols,
                'regime_filtering_enabled': self.enable_regime_filtering
            },
            'market_context': {
                'regime': current_regime,
                'preferred_strategies': self.regime_strategy_preferences.get(current_regime, [])
            }
        }
        
        # Save to JSON file
        output_file = self.output_dir / 'layer1_5_universe.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        # Also save timestamped version
        timestamped_file = self.output_dir / f'layer1_5_universe_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(timestamped_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Saved Layer 1.5 output to {output_file}")
    
    def _generate_filter_statistics(
        self, 
        input_universe: List[str], 
        output_symbols: List[str], 
        current_regime: str
    ) -> Dict[str, Any]:
        """Generate statistics about the filtering process."""
        filter_rate = len(output_symbols) / len(input_universe) if input_universe else 0
        
        stats = {
            'input_count': len(input_universe),
            'output_count': len(output_symbols),
            'filter_rate': filter_rate,
            'reduction_rate': 1 - filter_rate,
            'market_regime': current_regime,
            'preferred_strategies': self.regime_strategy_preferences.get(current_regime, []),
            'filter_parameters': {
                'min_affinity_score': self.min_affinity_score,
                'max_output_symbols': self.max_output_symbols,
                'regime_filtering_enabled': self.enable_regime_filtering
            }
        }
        
        return stats


# Factory function for easy integration
def create_layer1_5_strategy_affinity_filter(config: Any) -> Layer1_5_StrategyAffinityFilter:
    """Factory function to create Layer1_5_StrategyAffinityFilter."""
    return Layer1_5_StrategyAffinityFilter(config)


async def run_layer1_5_filter():
    """Standalone execution for Layer 1.5 Strategy Affinity Filter."""
    # Environment variables are loaded by config_manager
    from config import get_config
    from main.utils.database import DatabasePool
    
    config = get_config()
    
    # Initialize database pool
    db_pool = DatabasePool()
    db_pool.initialize(config=config)
    
    try:
        filter_layer = Layer1_5_StrategyAffinityFilter(config)
        
        logger.info("=" * 60)
        logger.info("Starting Layer 1.5 Strategy Affinity Filter")
        logger.info("=" * 60)
        
        qualified_symbols = await filter_layer.run()
        
        logger.info("=" * 60)
        logger.info("Layer 1.5 Filter Completed Successfully")
        logger.info(f"Output: {len(qualified_symbols)} qualified symbols")
        
        if qualified_symbols:
            logger.info("Top 10 qualified symbols:")
            for symbol in qualified_symbols[:10]:
                logger.info(f"  {symbol}")
        
        logger.info("=" * 60)
        
        return qualified_symbols
        
    except Exception as e:
        logger.error(f"Fatal error in Layer 1.5 filter: {e}", exc_info=True)
        raise
    finally:
        # Close database pool if it has a close method
        if hasattr(db_pool, 'close'):
            await db_pool.close()
        elif hasattr(db_pool, 'dispose'):
            db_pool.dispose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_layer1_5_filter())
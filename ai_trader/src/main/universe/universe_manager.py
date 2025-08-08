# File: universe/universe_manager.py

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from main.config.config_manager import get_config
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories import ICompanyRepository
from main.data_pipeline.storage.database_factory import DatabaseFactory

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Orchestrates the Layer 0 → Layer 1 → Layer 2 → Layer 3 qualification system.
    Provides unified interface for universe management and qualification.
    """
    
    def __init__(self, config=None):
        """Initialize the universe manager."""
        if config is None:
            config = get_config()
        
        self.config = config
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(config)
        
        # Initialize repository using factory pattern
        from main.data_pipeline.storage.repositories import get_repository_factory
        repo_factory = get_repository_factory()
        self.company_repository = repo_factory.create_company_repository(self.db_adapter)
        self._layer0_scanner = None  # Lazy initialization to avoid import issues
        self._db_initialized = False
    
    async def _ensure_db_initialized(self):
        """Ensure the database connection pool is initialized."""
        if not self._db_initialized:
            logger.info("Initializing database connection pool...")
            await self.db_adapter.initialize()
            self._db_initialized = True
            logger.info("Database connection pool initialized")
    
    async def populate_universe(self) -> Dict[str, Any]:
        """
        Run Layer 0 scan to populate the initial universe.
        
        Returns:
            Dictionary with population results
        """
        logger.info("🚀 Starting universe population with Layer 0 scan...")
        start_time = datetime.now(timezone.utc)
        
        # Ensure database is initialized
        await self._ensure_db_initialized()
        
        try:
            # Lazy import and initialize Layer 0 scanner to avoid import issues
            if self._layer0_scanner is None:
                from main.scanners.layers.layer0_static_universe import Layer0StaticUniverseScanner
                self._layer0_scanner = Layer0StaticUniverseScanner(self.config)
            
            # Run Layer 0 scan which will populate companies table
            assets = await self._layer0_scanner.run()
            
            # Get universe statistics
            stats = await self._get_universe_stats()
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                'success': True,
                'duration_seconds': duration,
                'assets_discovered': len(assets),
                'companies_in_db': stats['active_companies'],
                'universe_stats': stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✅ Universe population complete in {duration:.2f}s")
            logger.info(f"📊 Active companies: {stats['active_companies']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Universe population failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_qualified_symbols(self, layer: str = "0", limit: Optional[int] = None) -> List[str]:
        """
        Get symbols qualified for a specific layer.
        
        Layer qualification is hierarchical:
        - Layer 0: All scanned symbols
        - Layer 1: Passed Layer 1 filters (subset of Layer 0)
        - Layer 2: Passed Layer 2 filters (subset of Layer 1)
        - Layer 3: Passed Layer 3 filters (subset of Layer 2)
        
        When requesting Layer N, returns all symbols that qualify for Layer N or higher.
        
        Args:
            layer: Layer level ('0', '1', '2', '3')
            limit: Optional limit on number of symbols
            
        Returns:
            List of qualified symbols
        """
        # Ensure database is initialized before querying
        await self._ensure_db_initialized()
        
        try:
            layer_num = int(layer)
            
            # Use the new layer column system with hierarchical behavior
            # Layer N returns all symbols at layer N or higher
            if layer_num == 0:
                # Layer 0: All active symbols (BASIC layer and above)
                symbols = await self.company_repository.get_symbols_above_layer(
                    min_layer=0,
                    is_active=True
                )
            else:
                # Layer 1+: Use hierarchical approach (layer N includes all symbols >= N)
                symbols = await self.company_repository.get_symbols_above_layer(
                    min_layer=layer_num,
                    is_active=True
                )
            
            # Apply limit if specified
            if limit and len(symbols) > limit:
                symbols = symbols[:limit]
                
            logger.info(f"Retrieved {len(symbols)} symbols for layer {layer_num} (limit: {limit or 'none'})")
            return symbols
            
        except ValueError:
            logger.error(f"Invalid layer: {layer}")
            return []
        except Exception as e:
            logger.error(f"Error getting qualified symbols for layer {layer}: {e}")
            return []
    
    async def get_universe_for_backfill(self, preferred_layer: str = "0", 
                                       fallback_limit: Optional[int] = None) -> List[str]:
        """
        Get universe symbols optimized for backfill operations.
        
        Args:
            preferred_layer: Preferred qualification layer 
            fallback_limit: Maximum symbols if specified, None for unlimited
            
        Returns:
            List of symbols for backfill
        """
        logger.info(f"Getting universe for backfill, preferred layer: {preferred_layer}, limit: {fallback_limit}")
        
        # Try to get symbols from the preferred layer
        symbols = await self.get_qualified_symbols(preferred_layer, limit=fallback_limit)
        
        if symbols:
            logger.info(f"✅ Found {len(symbols)} symbols from layer {preferred_layer}")
            return symbols
        
        # Fallback to Layer 0 (all active) with same limit
        logger.warning(f"No symbols found for layer {preferred_layer}, falling back to active companies")
        symbols = await self.get_qualified_symbols("0", limit=fallback_limit)
        
        if symbols:
            logger.info(f"✅ Fallback: Found {len(symbols)} active companies")
            return symbols
        
        # Ultimate fallback - log warning and return empty list
        logger.error("❌ No symbols found in database. Run universe population first.")
        return []
    
    async def qualify_layer1(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run Layer 1 liquidity qualification.
        
        Args:
            force_refresh: Whether to re-qualify all symbols
            
        Returns:
            Qualification results
        """
        try:
            from main.scanners.layers.layer1_liquidity_filter import Layer1LiquidityFilter
            from main.data_pipeline.historical.manager import HistoricalManager
            
            logger.info("Starting Layer 1 liquidity qualification...")
            
            # Initialize historical manager for liquidity filter
            historical_manager = HistoricalManager(self.config)
            
            # Initialize Layer 1 filter
            layer1_filter = Layer1LiquidityFilter(self.config, historical_manager)
            
            # Get all active companies
            active_companies = await self.company_repository.get_active_companies()
            active_symbols = [company['symbol'] for company in active_companies]
            
            logger.info(f"Qualifying {len(active_symbols)} active symbols for Layer 1...")
            
            # Run liquidity filter
            qualified_symbols = await layer1_filter.run(active_symbols)
            
            # Update database with qualification results
            current_time = datetime.utcnow()
            
            # Import DataLayer enum
            from main.data_pipeline.core.enums import DataLayer
            
            # Update qualified companies to Layer 1
            if qualified_symbols:
                for symbol in qualified_symbols:
                    await self.company_repository.update_layer(
                        symbol=symbol,
                        layer=DataLayer.LIQUID,
                        metadata={
                            'source': 'universe_manager',
                            'timestamp': current_time.isoformat()
                        }
                    )
            
            # Update non-qualified companies to Layer 0
            non_qualified_symbols = list(set(active_symbols) - set(qualified_symbols))
            if non_qualified_symbols and force_refresh:
                for symbol in non_qualified_symbols:
                    await self.company_repository.update_layer(
                        symbol=symbol,
                        layer=DataLayer.BASIC,
                        metadata={
                            'source': 'universe_manager',
                            'timestamp': current_time.isoformat()
                        }
                    )
            
            logger.info(f"Layer 1 qualification complete: {len(qualified_symbols)} qualified out of {len(active_symbols)}")
            
            return {
                'success': True,
                'total_symbols': len(active_symbols),
                'qualified_symbols': len(qualified_symbols),
                'qualification_rate': len(qualified_symbols) / len(active_symbols) * 100 if active_symbols else 0,
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Error during Layer 1 qualification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_universe_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive universe statistics.
        
        Returns:
            Dictionary with universe statistics
        """
        stats = await self._get_universe_stats()
        
        # Add some derived metrics
        total = stats['total_companies']
        active = stats['active_companies']
        
        stats['active_percentage'] = (active / total * 100) if total and total > 0 else 0
        stats['layer0_percentage'] = (stats.get('layer0_count', 0) / active * 100) if active and active > 0 else 0
        stats['layer1_percentage'] = (stats.get('layer1_count', 0) / active * 100) if active and active > 0 else 0
        stats['layer2_percentage'] = (stats.get('layer2_count', 0) / active * 100) if active and active > 0 else 0
        stats['layer3_percentage'] = (stats.get('layer3_count', 0) / active * 100) if active and active > 0 else 0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the universe system.
        
        Returns:
            Health check results
        """
        try:
            stats = await self._get_universe_stats()
            
            # Check if we have a reasonable number of companies
            min_expected_companies = 1000  # US market should have thousands of symbols
            has_sufficient_companies = stats['active_companies'] >= min_expected_companies
            
            # Check if database is accessible
            test_symbols = await self.get_qualified_symbols("0", limit=1)
            database_accessible = len(test_symbols) > 0 or stats['total_companies'] == 0
            
            health_status = {
                'healthy': has_sufficient_companies and database_accessible,
                'companies_count': stats['active_companies'],
                'has_sufficient_companies': has_sufficient_companies,
                'database_accessible': database_accessible,
                'universe_stats': stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if health_status['healthy']:
                logger.info(f"✅ Universe health check passed. {stats['active_companies']} active companies")
            else:
                logger.warning(f"⚠️ Universe health issues detected. Run universe population if needed.")
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Universe health check failed: {e}")
            return {
                'healthy': False,
                'companies_count': 0,
                'has_sufficient_companies': False,
                'database_accessible': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_universe_stats(self) -> Dict[str, Any]:
        """
        Get universe statistics using the existing repository methods.
        
        Returns:
            Dictionary with universe statistics
        """
        # Ensure database is initialized
        await self._ensure_db_initialized()
        
        # Use repository's counting methods for efficiency instead of loading all records
        total_companies = await self.company_repository.get_record_count()
        
        # Count active companies using simple query to get accurate count
        active_companies = await self.company_repository.get_active_companies(simple_query=True)
        active_count = len(active_companies)
        
        # Count layer qualified companies using new layer column
        # Get counts for each layer
        layer0_count = await self.company_repository.get_record_count_filtered(
            filter_dict={'layer': 0, 'is_active': True}
        )
        layer1_count = await self.company_repository.get_record_count_filtered(
            filter_dict={'layer': 1, 'is_active': True}
        )
        layer2_count = await self.company_repository.get_record_count_filtered(
            filter_dict={'layer': 2, 'is_active': True}
        )
        layer3_count = await self.company_repository.get_record_count_filtered(
            filter_dict={'layer': 3, 'is_active': True}
        )
        
        stats = {
            'total_companies': total_companies,
            'active_companies': active_count,
            'layer0_count': layer0_count,
            'layer1_count': layer1_count,
            'layer2_count': layer2_count,
            'layer3_count': layer3_count
        }
        
        return stats
    
    async def close(self):
        """Close repository connections."""
        await self.db_adapter.close()
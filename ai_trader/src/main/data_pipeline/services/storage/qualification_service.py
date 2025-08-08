"""
Qualification Service for Scanner Layer System

Handles symbol qualification checks for determining data storage policies
based on scanner layer assignments.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio

from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class QualificationConfig:
    """Configuration for qualification service."""
    cache_ttl_seconds: int = 3600  # 1 hour default
    default_retention_days: int = 30
    minute_data_requires_layer: int = 1  # Minimum layer for minute data
    enable_caching: bool = True


@dataclass
class SymbolQualification:
    """Symbol qualification details."""
    symbol: str
    layer_qualified: int
    retention_days: int
    allows_minute_data: bool
    is_active: bool = True
    last_updated: Optional[datetime] = None
    
    def should_store_interval(self, interval: str) -> bool:
        """
        Determine if data should be stored for this interval.
        
        Args:
            interval: Time interval (e.g., '1minute', '1hour', '1day')
            
        Returns:
            True if data should be stored
        """
        # Always store hourly and daily data
        if interval in ['1hour', '1day']:
            return True
        
        # Minute intervals require qualification
        if 'minute' in interval.lower():
            return self.allows_minute_data
        
        # Default to storing
        return True


class QualificationService:
    """
    Service for managing symbol qualifications and scanner layer checks.
    
    This service determines which symbols qualify for different types
    of data storage based on their scanner layer assignments.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[QualificationConfig] = None
    ):
        """
        Initialize the qualification service.
        
        Args:
            db_adapter: Database adapter for queries
            config: Service configuration
        """
        self.db_adapter = db_adapter
        self.config = config or QualificationConfig()
        
        # Cache for qualifications
        self._cache: Dict[str, SymbolQualification] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_lock = asyncio.Lock()
        
        # Batch loading for efficiency
        self._pending_lookups: set = set()
        self._batch_lock = asyncio.Lock()
    
    async def get_qualification(self, symbol: str) -> SymbolQualification:
        """
        Get qualification for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            SymbolQualification with layer and policy information
        """
        symbol = symbol.upper()
        
        # Check cache first
        if self.config.enable_caching:
            cached = await self._get_from_cache(symbol)
            if cached:
                return cached
        
        # Query database
        qualification = await self._query_qualification(symbol)
        
        # Update cache
        if self.config.enable_caching:
            await self._update_cache(symbol, qualification)
        
        return qualification
    
    async def get_qualifications_batch(
        self,
        symbols: list[str]
    ) -> Dict[str, SymbolQualification]:
        """
        Get qualifications for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their qualifications
        """
        results = {}
        uncached_symbols = []
        
        # Check cache for each symbol
        for symbol in symbols:
            symbol = symbol.upper()
            if self.config.enable_caching:
                cached = await self._get_from_cache(symbol)
                if cached:
                    results[symbol] = cached
                else:
                    uncached_symbols.append(symbol)
            else:
                uncached_symbols.append(symbol)
        
        # Batch query for uncached symbols
        if uncached_symbols:
            batch_results = await self._query_qualifications_batch(uncached_symbols)
            results.update(batch_results)
            
            # Update cache
            if self.config.enable_caching:
                for symbol, qual in batch_results.items():
                    await self._update_cache(symbol, qual)
        
        return results
    
    async def _get_from_cache(self, symbol: str) -> Optional[SymbolQualification]:
        """Get qualification from cache if valid."""
        async with self._cache_lock:
            if symbol not in self._cache:
                return None
            
            # Check if cache entry is expired
            cache_time = self._cache_timestamps.get(symbol)
            if not cache_time:
                return None
            
            age = (datetime.now(timezone.utc) - cache_time).total_seconds()
            if age > self.config.cache_ttl_seconds:
                # Expired - remove from cache
                del self._cache[symbol]
                del self._cache_timestamps[symbol]
                return None
            
            return self._cache[symbol]
    
    async def _update_cache(self, symbol: str, qualification: SymbolQualification):
        """Update cache with new qualification."""
        async with self._cache_lock:
            self._cache[symbol] = qualification
            self._cache_timestamps[symbol] = datetime.now(timezone.utc)
    
    async def _query_qualification(self, symbol: str) -> SymbolQualification:
        """
        Query database for symbol qualification.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            SymbolQualification
        """
        try:
            # Try to use stored function if it exists
            query = """
            SELECT 
                symbol,
                COALESCE(layer, 0) as layer_qualified,
                CASE 
                    WHEN layer = 3 THEN 1825  -- Layer 3 (ACTIVE): 5 years
                    WHEN layer = 2 THEN 730   -- Layer 2 (CATALYST): 2 years  
                    WHEN layer = 1 THEN 365   -- Layer 1 (LIQUID): 1 year
                    ELSE 30                   -- Layer 0 (BASIC): 30 days
                END as retention_days,
                COALESCE(layer >= 1, FALSE) as allows_minute_data,
                COALESCE(is_active, TRUE) as is_active
            FROM companies
            WHERE symbol = $1
            """
            
            result = await self.db_adapter.fetch_one(query, {'symbol': symbol})
            
            if result:
                return SymbolQualification(
                    symbol=symbol,
                    layer_qualified=result['layer_qualified'],
                    retention_days=result['retention_days'],
                    allows_minute_data=result['allows_minute_data'],
                    is_active=result['is_active'],
                    last_updated=datetime.now(timezone.utc)
                )
            else:
                # Symbol not found - return default qualification
                logger.debug(f"Symbol {symbol} not found in companies table, using defaults")
                return SymbolQualification(
                    symbol=symbol,
                    layer_qualified=0,
                    retention_days=self.config.default_retention_days,
                    allows_minute_data=False,
                    is_active=False,
                    last_updated=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Error querying qualification for {symbol}: {e}")
            # Return safe defaults on error
            return SymbolQualification(
                symbol=symbol,
                layer_qualified=0,
                retention_days=self.config.default_retention_days,
                allows_minute_data=False,
                is_active=False,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _query_qualifications_batch(
        self,
        symbols: list[str]
    ) -> Dict[str, SymbolQualification]:
        """
        Query qualifications for multiple symbols in one query.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of symbol to qualification
        """
        if not symbols:
            return {}
        
        try:
            # Create parameter placeholders
            placeholders = ', '.join(f'${i+1}' for i in range(len(symbols)))
            
            query = f"""
            SELECT 
                symbol,
                COALESCE(layer, 0) as layer_qualified,
                CASE 
                    WHEN layer = 3 THEN 1825  -- Layer 3 (ACTIVE): 5 years
                    WHEN layer = 2 THEN 730   -- Layer 2 (CATALYST): 2 years  
                    WHEN layer = 1 THEN 365   -- Layer 1 (LIQUID): 1 year
                    ELSE 30                   -- Layer 0 (BASIC): 30 days
                END as retention_days,
                COALESCE(layer >= 1, FALSE) as allows_minute_data,
                COALESCE(is_active, TRUE) as is_active
            FROM companies
            WHERE symbol IN ({placeholders})
            """
            
            # Execute with positional parameters
            params = {str(i+1): symbol for i, symbol in enumerate(symbols)}
            rows = await self.db_adapter.fetch_all(query, params)
            
            results = {}
            found_symbols = set()
            
            for row in rows:
                symbol = row['symbol']
                found_symbols.add(symbol)
                results[symbol] = SymbolQualification(
                    symbol=symbol,
                    layer_qualified=row['layer_qualified'],
                    retention_days=row['retention_days'],
                    allows_minute_data=row['allows_minute_data'],
                    is_active=row['is_active'],
                    last_updated=datetime.now(timezone.utc)
                )
            
            # Add defaults for symbols not found
            for symbol in symbols:
                if symbol not in found_symbols:
                    results[symbol] = SymbolQualification(
                        symbol=symbol,
                        layer_qualified=0,
                        retention_days=self.config.default_retention_days,
                        allows_minute_data=False,
                        is_active=False,
                        last_updated=datetime.now(timezone.utc)
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch qualification query: {e}")
            # Return defaults for all symbols on error
            return {
                symbol: SymbolQualification(
                    symbol=symbol,
                    layer_qualified=0,
                    retention_days=self.config.default_retention_days,
                    allows_minute_data=False,
                    is_active=False,
                    last_updated=datetime.now(timezone.utc)
                )
                for symbol in symbols
            }
    
    async def clear_cache(self):
        """Clear the qualification cache."""
        async with self._cache_lock:
            self._cache.clear()
            self._cache_timestamps.clear()
        logger.info("Qualification cache cleared")
    
    async def refresh_qualification(self, symbol: str) -> SymbolQualification:
        """
        Force refresh qualification for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Fresh SymbolQualification from database
        """
        symbol = symbol.upper()
        
        # Remove from cache
        async with self._cache_lock:
            self._cache.pop(symbol, None)
            self._cache_timestamps.pop(symbol, None)
        
        # Query fresh data
        return await self.get_qualification(symbol)
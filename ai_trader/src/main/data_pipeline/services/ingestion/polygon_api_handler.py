"""
Polygon API Handler

Shared service for all Polygon API operations.
Handles pagination, rate limiting, and error handling.
"""

from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from datetime import datetime
import asyncio
import aiohttp

from main.utils.core import get_logger, ensure_utc
from main.data_pipeline.core.enums import DataLayer
from main.utils.layer_utils import (
    get_api_rate_limit, get_layer_batch_size, 
    get_layer_max_concurrent, get_layer_cache_ttl
)
from main.utils.layer_metrics import LayerMetricsCollector
from main.data_pipeline.ingestion.clients.base_client import FetchResult, ClientConfig


class PolygonApiHandler:
    """
    Centralized handler for Polygon API operations.
    
    Eliminates duplication across Polygon clients by providing:
    - Common configuration and setup
    - Pagination handling
    - Error response handling
    - Batch processing
    - Response validation
    """
    
    # Layer-based configuration is now handled by layer_utils
    # No more hardcoded tier limits - use get_api_rate_limit(layer) instead
    
    def __init__(self, base_url: str = 'https://api.polygon.io'):
        """Initialize the API handler."""
        self.base_url = base_url
        self.logger = get_logger(__name__)
    
    async def paginate(
        self,
        client: Any,
        endpoint: str,
        params: Dict[str, Any],
        max_pages: int = 10
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """
        Handle pagination for any Polygon endpoint.
        
        Args:
            client: The client with fetch method
            endpoint: API endpoint
            params: Query parameters
            max_pages: Maximum pages to fetch
            
        Yields:
            Lists of results from each page
        """
        pages_fetched = 0
        next_url = None
        
        while pages_fetched < max_pages:
            # Use next URL if available
            if next_url:
                result = await self._fetch_next_page(client, next_url)
            else:
                result = await client.fetch(endpoint, params)
            
            if not result.success:
                if pages_fetched == 0:
                    self.logger.error(f"Failed to fetch first page: {result.error}")
                break
            
            if result.data:
                yield result.data
                pages_fetched += 1
            
            # Check for next page
            next_url = getattr(client, '_next_url', None)
            if not next_url or not result.data:
                break
            
            # Small delay between pages
            if pages_fetched < max_pages:
                await asyncio.sleep(0.1)
    
    async def batch_fetch(
        self,
        fetch_fn: Callable,
        items: List[Any],
        batch_size: int = 50,
        max_concurrent: int = 5
    ) -> Dict[Any, FetchResult]:
        """
        Process items in batches with concurrency control.
        
        Args:
            fetch_fn: Function to fetch data for an item
            items: Items to process
            batch_size: Items per batch
            max_concurrent: Max concurrent requests
            
        Returns:
            Dictionary mapping items to results
        """
        results = {}
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create semaphore for this batch
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            
            for item in batch:
                task = self._fetch_with_semaphore(
                    semaphore, fetch_fn, item
                )
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[item] = FetchResult(
                        success=False,
                        error=str(result)
                    )
                else:
                    results[item] = result
            
            # Delay between batches
            if i + batch_size < len(items):
                await asyncio.sleep(0.5)
        
        self.logger.info(f"Processed {len(items)} items in batches")
        return results
    
    async def _fetch_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        fetch_fn: Callable,
        item: Any
    ) -> FetchResult:
        """Execute fetch with semaphore."""
        async with semaphore:
            return await fetch_fn(item)
    
    async def _fetch_next_page(self, client: Any, next_url: str) -> FetchResult:
        """Fetch next page from pagination URL."""
        if next_url.startswith(self.base_url):
            url_parts = next_url.replace(self.base_url, '').split('?')
            
            if len(url_parts) == 2:
                endpoint = url_parts[0].lstrip('/')
                
                # Parse query params
                params = {}
                for param in url_parts[1].split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key] = value
                
                return await client.fetch(endpoint, params, use_cache=False)
        
        return FetchResult(success=False, error="Invalid pagination URL")
    
    def parse_polygon_error(self, response_data: Dict[str, Any]) -> str:
        """
        Parse error message from Polygon response.
        
        Args:
            response_data: Response data from Polygon
            
        Returns:
            Formatted error message
        """
        status = response_data.get('status', 'ERROR')
        message = response_data.get('message', 'Unknown error')
        error = response_data.get('error', '')
        
        if error:
            return f"{status}: {message} - {error}"
        return f"{status}: {message}"
    
    def validate_polygon_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Validate Polygon API response.
        
        Args:
            response_data: Response data
            
        Returns:
            True if valid response
        """
        return response_data.get('status') == 'OK' and 'results' in response_data
    
    def create_polygon_config(
        self,
        api_key: str,
        layer: DataLayer = DataLayer.BASIC,
        config: Optional[ClientConfig] = None,
        cache_ttl_seconds: Optional[int] = None
    ) -> ClientConfig:
        """
        Create standardized ClientConfig for Polygon clients.
        
        Args:
            api_key: Polygon API key
            layer: Data layer for configuration
            config: Optional existing config to modify
            cache_ttl_seconds: Optional cache TTL override
            
        Returns:
            Configured ClientConfig instance
        """
        # Get layer-based rate limit
        rate_limit = get_api_rate_limit(layer)
        
        # Track rate limit configuration
        max_rate_limit = get_api_rate_limit(DataLayer.ACTIVE)  # Maximum possible rate
        usage_percent = (rate_limit / max_rate_limit * 100) if max_rate_limit > 0 else 0
        LayerMetricsCollector.record_rate_limit_usage(
            layer,
            usage_percent,
            int(rate_limit * 60),  # calls per minute
            int(max_rate_limit * 60)  # max calls per minute
        )
        
        if config is None:
            config = ClientConfig(
                api_key=api_key,
                base_url=self.base_url,
                rate_limit_per_second=rate_limit,
                max_retries=3,
                timeout_seconds=30,
                cache_ttl_seconds=cache_ttl_seconds or get_layer_cache_ttl(layer)
            )
        else:
            # Update existing config
            config.api_key = api_key
            config.base_url = config.base_url or self.base_url
            config.rate_limit_per_second = rate_limit
            if cache_ttl_seconds is not None:
                config.cache_ttl_seconds = cache_ttl_seconds
        
        return config
    
    def get_standard_headers(self, api_key: str) -> Dict[str, str]:
        """
        Get standard headers for Polygon API requests.
        
        Args:
            api_key: Polygon API key
            
        Returns:
            Dictionary of headers
        """
        return {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    async def validate_http_response(self, response: aiohttp.ClientResponse) -> bool:
        """
        Validate HTTP response from Polygon API.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if valid response
        """
        if response.status != 200:
            return False
        
        try:
            data = await response.json()
            return self.validate_polygon_response(data)
        except Exception:
            return False
    
    async def parse_polygon_response(self, response: aiohttp.ClientResponse) -> List[Dict[str, Any]]:
        """
        Parse Polygon API response into standardized format.
        
        Args:
            response: HTTP response object
            
        Returns:
            List of results from response
        """
        try:
            data = await response.json()
            results = data.get('results', [])
            return results
        except Exception as e:
            self.logger.error(f"Error parsing Polygon response: {e}")
            return []
    
    def build_date_params(
        self,
        start_date: datetime,
        end_date: datetime,
        date_field: str = 'timestamp'
    ) -> Dict[str, str]:
        """
        Build common date range parameters for Polygon API.
        
        Args:
            start_date: Start date
            end_date: End date
            date_field: Field name for date filtering
            
        Returns:
            Dictionary of date parameters
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)
        
        return {
            f'{date_field}.gte': start_date.strftime('%Y-%m-%d'),
            f'{date_field}.lte': end_date.strftime('%Y-%m-%d')
        }
    
    async def fetch_with_pagination(
        self,
        client: Any,
        endpoint: str,
        params: Dict[str, Any],
        limit: Optional[int] = None,
        max_pages: int = 10
    ) -> FetchResult[List[Dict[str, Any]]]:
        """
        Fetch data with automatic pagination handling.
        
        Args:
            client: Client instance with fetch method
            endpoint: API endpoint
            params: Query parameters
            limit: Maximum records to fetch (None for unlimited)
            max_pages: Maximum pages to fetch
            
        Returns:
            FetchResult with aggregated results
        """
        all_results = []
        pages_fetched = 0
        
        try:
            async for page_results in self.paginate(client, endpoint, params, max_pages):
                all_results.extend(page_results)
                pages_fetched += 1
                
                # Check limit
                if limit and len(all_results) >= limit:
                    all_results = all_results[:limit]
                    break
            
            return FetchResult(
                success=True,
                data=all_results,
                metadata={
                    'pages_fetched': pages_fetched,
                    'total_records': len(all_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching with pagination: {e}")
            return FetchResult(
                success=False,
                error=str(e),
                data=all_results if all_results else None
            )
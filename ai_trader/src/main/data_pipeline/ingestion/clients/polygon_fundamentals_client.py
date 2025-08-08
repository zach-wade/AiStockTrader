"""
Polygon Fundamentals Client - Refactored

Simplified client for fetching financial statements from Polygon.io API.
Uses PolygonApiHandler for common functionality.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from main.utils.core import get_logger, ensure_utc
from main.utils.monitoring import timer, record_metric, MetricType
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.services.ingestion import MetricExtractionService
from main.data_pipeline.services.ingestion.polygon_api_handler import PolygonApiHandler
from .base_client import BaseIngestionClient, ClientConfig, FetchResult


class PolygonFundamentalsClient(BaseIngestionClient[List[Dict[str, Any]]]):
    """
    Simplified client for fetching financial statements from Polygon.io.
    
    Delegates common functionality to PolygonApiHandler.
    """
    
    def __init__(
        self,
        api_key: str,
        layer: DataLayer = DataLayer.BASIC,
        config: Optional[ClientConfig] = None,
        metric_extractor: Optional[MetricExtractionService] = None
    ):
        """Initialize the Polygon fundamentals client."""
        self.api_handler = PolygonApiHandler()
        
        # Create config using handler with layer-based configuration
        # Financials use custom cache TTL regardless of layer
        config = self.api_handler.create_polygon_config(
            api_key=api_key,
            layer=layer,
            config=config,
            cache_ttl_seconds=3600  # Cache financials for 1 hour
        )
        
        super().__init__(config)
        self.layer = layer
        self.metric_extractor = metric_extractor or MetricExtractionService()
        
        self.logger = get_logger(__name__)
        self.logger.info(f"PolygonFundamentalsClient initialized with layer: {layer.name}")
    
    def get_base_url(self) -> str:
        """Get the base URL for Polygon API."""
        return self.config.base_url
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return self.api_handler.get_standard_headers(self.config.api_key)
    
    async def validate_response(self, response) -> bool:
        """Validate Polygon API response."""
        return await self.api_handler.validate_http_response(response)
    
    async def parse_response(self, response) -> List[Dict[str, Any]]:
        """Parse Polygon API response into standardized format."""
        results = await self.api_handler.parse_polygon_response(response)
        
        parsed_filings = []
        for record in results:
            normalized = self._normalize_filing(record)
            if normalized and self._validate_financial_data(normalized):
                parsed_filings.append(normalized)
        
        return parsed_filings
    
    async def fetch_financials(
        self,
        symbol: str,
        timeframe: str = 'quarterly',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 20
    ) -> FetchResult[List[Dict[str, Any]]]:
        """Fetch financial statements for a symbol."""
        endpoint = "vX/reference/financials"
        
        # Build params
        params = {
            'ticker': symbol.upper(),
            'timeframe': timeframe,
            'limit': str(min(limit, 100)),
            'order': 'desc',
            'sort': 'filing_date'
        }
        
        # Add date filters if provided
        if start_date:
            params['filing_date.gte'] = ensure_utc(start_date).strftime('%Y-%m-%d')
        if end_date:
            params['filing_date.lte'] = ensure_utc(end_date).strftime('%Y-%m-%d')
        
        # Track API call performance
        with timer("polygon.fundamentals.fetch", tags={"symbol": symbol, "timeframe": timeframe}):
            # Use handler for pagination
            result = await self.api_handler.fetch_with_pagination(
                self, endpoint, params, limit=limit, max_pages=5
            )
        
        if result.success and result.data:
            # Add symbol to each record
            for record in result.data:
                record['symbol'] = symbol
            
            # Track filings fetched
            record_metric("polygon.fundamentals.filings", len(result.data), MetricType.COUNTER,
                         tags={"symbol": symbol, "timeframe": timeframe})
            
            # Track filing recency
            if result.data:
                latest = result.data[0]
                if 'filing_date' in latest:
                    days_old = (datetime.now(timezone.utc) - latest['filing_date']).days
                    gauge("polygon.fundamentals.filing_age_days", days_old,
                          tags={"symbol": symbol, "timeframe": timeframe})
        else:
            # Track API errors
            record_metric("polygon.api.errors", 1, MetricType.COUNTER,
                         tags={"data_type": "fundamentals", "symbol": symbol,
                               "error": result.error or "unknown"})
        
        return result
    
    async def fetch_quarterly(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 8
    ) -> FetchResult[List[Dict[str, Any]]]:
        """Fetch quarterly financial statements."""
        return await self.fetch_financials(
            symbol=symbol,
            timeframe='quarterly',
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    async def fetch_annual(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 5
    ) -> FetchResult[List[Dict[str, Any]]]:
        """Fetch annual financial statements."""
        return await self.fetch_financials(
            symbol=symbol,
            timeframe='annual',
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = 'quarterly',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent: int = 5
    ) -> Dict[str, FetchResult[List[Dict[str, Any]]]]:
        """Fetch financials for multiple symbols using the handler."""
        async def fetch_symbol(symbol: str) -> FetchResult:
            return await self.fetch_financials(
                symbol, timeframe, start_date, end_date
            )
        
        # Track batch operation
        batch_start = datetime.now()
        gauge("polygon.fundamentals.batch_symbols", len(symbols),
              tags={"timeframe": timeframe})
        
        # Use handler's batch_fetch
        results = await self.api_handler.batch_fetch(
            fetch_symbol,
            symbols,
            batch_size=50,
            max_concurrent=max_concurrent
        )
        
        # Calculate batch metrics
        batch_duration = (datetime.now() - batch_start).total_seconds()
        successful = sum(1 for r in results.values() if r.success)
        total_filings = sum(len(r.data) for r in results.values() if r.success and r.data)
        
        # Record batch metrics
        record_metric("polygon.fundamentals.batch_duration", batch_duration, MetricType.HISTOGRAM,
                     tags={"timeframe": timeframe, "symbols": len(symbols)})
        record_metric("polygon.fundamentals.batch_success_rate", successful / len(symbols) if symbols else 0,
                     MetricType.GAUGE, tags={"timeframe": timeframe})
        record_metric("polygon.fundamentals.batch_total_filings", total_filings, MetricType.COUNTER,
                     tags={"timeframe": timeframe})
        
        if len(symbols) - successful > 0:
            self.logger.warning(f"Fundamentals batch fetch failed for {len(symbols) - successful}/{len(symbols)} symbols")
        
        return results
    
    def _normalize_filing(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a financial filing to standard format."""
        try:
            # Extract metadata
            filing_date = datetime.fromisoformat(
                record.get('filing_date', '').replace('Z', '+00:00')
            )
            
            fiscal_year = record.get('fiscal_year')
            fiscal_period = record.get('fiscal_period', '')
            timeframe = record.get('timeframe', 'quarterly')
            
            # Extract financials
            financials = record.get('financials', {})
            metrics = self._extract_metrics(financials)
            
            # Build normalized record
            return {
                'symbol': record.get('ticker', '').upper(),
                'fiscal_year': fiscal_year,
                'fiscal_period': self._determine_period(fiscal_period, timeframe),
                'filing_date': filing_date,
                'timeframe': timeframe,
                'cik': record.get('cik'),
                'company_name': record.get('company_name'),
                'source_filing_url': record.get('source_filing_url'),
                'source_filing_file_url': record.get('source_filing_file_url'),
                **metrics,
                'raw_data': financials
            }
        except Exception as e:
            self.logger.debug(f"Error normalizing filing: {e}")
            raise
    
    def _extract_metrics(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics from nested structure."""
        with timer("polygon.fundamentals.metric_extraction"):
            metrics = {}
            
            # Income statement metrics
            income_statement = financials.get('income_statement', {})
            metrics['revenue'] = self._extract_value(
                income_statement.get('revenues', income_statement.get('revenue'))
            )
            metrics['net_income'] = self._extract_value(
                income_statement.get('net_income_loss', income_statement.get('net_income'))
            )
            metrics['eps_basic'] = self._extract_value(
                income_statement.get('basic_earnings_per_share')
            )
            metrics['eps_diluted'] = self._extract_value(
                income_statement.get('diluted_earnings_per_share')
            )
            
            # Balance sheet metrics
            balance_sheet = financials.get('balance_sheet', {})
            metrics['total_assets'] = self._extract_value(balance_sheet.get('assets'))
            metrics['total_liabilities'] = self._extract_value(balance_sheet.get('liabilities'))
            metrics['stockholders_equity'] = self._extract_value(
                balance_sheet.get('equity', balance_sheet.get('stockholders_equity'))
            )
            
            # Cash flow metrics
            cash_flow = financials.get('cash_flow_statement', {})
            metrics['operating_cash_flow'] = self._extract_value(
                cash_flow.get('net_cash_flow_from_operating_activities')
            )
            
            # Track metric completeness
            extracted_count = sum(1 for v in metrics.values() if v is not None)
            total_count = len(metrics)
            if total_count > 0:
                completeness = extracted_count / total_count
                record_metric("polygon.fundamentals.metric_completeness", completeness, MetricType.GAUGE)
            
            # Use metric extractor for additional calculations if available
            if self.metric_extractor:
                with timer("polygon.fundamentals.ratio_calculation"):
                    metrics.update(self.metric_extractor.extract_ratios(metrics))
        
        return metrics
    
    def _extract_value(self, *fields) -> Optional[float]:
        """Extract numeric value from Polygon's field format."""
        for field in fields:
            if field is not None:
                if isinstance(field, dict):
                    value = field.get('value')
                    if value is not None:
                        return float(value)
                elif isinstance(field, (int, float)):
                    return float(field)
        return None
    
    def _determine_period(self, fiscal_period: str, timeframe: str) -> str:
        """Determine the period string (Q1-Q4 or FY)."""
        if timeframe == 'annual':
            return 'FY'
        
        # Map various quarter formats to standard Q1-Q4
        fiscal_period_upper = fiscal_period.upper()
        if 'Q1' in fiscal_period_upper or 'FIRST' in fiscal_period_upper:
            return 'Q1'
        elif 'Q2' in fiscal_period_upper or 'SECOND' in fiscal_period_upper:
            return 'Q2'
        elif 'Q3' in fiscal_period_upper or 'THIRD' in fiscal_period_upper:
            return 'Q3'
        elif 'Q4' in fiscal_period_upper or 'FOURTH' in fiscal_period_upper:
            return 'Q4'
        
        return fiscal_period
    
    def _validate_financial_data(self, data: Dict[str, Any]) -> bool:
        """Validate that financial data has required fields."""
        required_fields = ['fiscal_year', 'fiscal_period', 'filing_date']
        
        # Check required fields exist
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Check at least one financial metric exists
        metric_fields = ['revenue', 'net_income', 'total_assets', 'operating_cash_flow']
        return any(data.get(field) is not None for field in metric_fields)
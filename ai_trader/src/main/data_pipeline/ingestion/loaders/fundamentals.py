"""
Fundamentals bulk loader for efficient backfill operations.

This module provides optimized bulk loading for financial statements,
using PostgreSQL COPY command and efficient batching strategies.
Uses Strategy pattern with format handlers for different data sources.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, date

from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig, BulkLoadResult
from main.utils.core import get_logger
from .base import BaseBulkLoader

# Import directly to avoid circular import through __init__
from main.data_pipeline.ingestion.factories.fundamentals_format_factory import (
    FundamentalsFormatFactory,
    FundamentalsFormatFactoryConfig
)

logger = get_logger(__name__)


class FundamentalsBulkLoader(BaseBulkLoader[Dict[str, Any]]):
    """
    Optimized bulk loader for fundamentals/financial statement data.
    
    Uses format handlers to process data from multiple sources (Polygon, Yahoo, pre-processed)
    and loads efficiently using PostgreSQL COPY operations.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        format_factory: Optional[FundamentalsFormatFactory] = None,
        archive: Optional[Any] = None,
        config: Optional[BulkLoadConfig] = None
    ):
        """
        Initialize fundamentals bulk loader.
        
        Args:
            db_adapter: Database adapter for operations
            format_factory: Factory for selecting format handlers
            archive: Optional archive for cold storage
            config: Bulk loading configuration
        """
        super().__init__(
            db_adapter=db_adapter,
            archive=archive,
            config=config,
            data_type="fundamentals"
        )
        
        # Initialize format factory if not provided
        if not format_factory:
            factory_config = FundamentalsFormatFactoryConfig()
            format_factory = FundamentalsFormatFactory(factory_config)
        
        self.format_factory = format_factory
        
        logger.info(
            f"FundamentalsBulkLoader initialized with format factory"
        )
    
    async def load(
        self,
        data: List[Dict[str, Any]],
        symbols: List[str],
        source: str = "yahoo",
        **kwargs
    ) -> BulkLoadResult:
        """
        Load financial statement data efficiently using bulk operations.
        
        Args:
            data: List of financial statement records or raw data
            symbols: List of symbols this data relates to
            source: Data source name ('polygon', 'yahoo', etc.)
            **kwargs: Additional parameters
            
        Returns:
            BulkLoadResult with operation details
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)
        
        if not data:
            result.success = True
            result.skip_reason = "No data provided"
            return result
        
        try:
            # Get appropriate format handler
            handler = self.format_factory.get_handler(data, source)
            if not handler:
                error_msg = f"No suitable format handler found for data type: {type(data)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                return result
            
            # Process data using the handler
            logger.debug(f"Processing data with {handler.__class__.__name__}")
            prepared_records = handler.process(data, symbols, source)
            
            if not prepared_records:
                logger.info("No valid financial statement records to load")
                result.success = True
                result.skip_reason = "No valid records after processing"
                return result
            
            logger.debug(f"Prepared {len(prepared_records)} financial records")
            
            # Add to buffer
            self._add_to_buffer(prepared_records)
            for symbol in symbols:
                self._symbols_in_buffer.add(symbol.upper())
            
            # Check if we should flush
            if self._should_flush():
                flush_result = await self._flush_buffer()
                result.records_loaded = flush_result.records_loaded
                result.records_failed = flush_result.records_failed
                result.symbols_processed = flush_result.symbols_processed
                result.load_time_seconds = flush_result.load_time_seconds
                result.archive_time_seconds = flush_result.archive_time_seconds
                result.errors = flush_result.errors
                result.success = flush_result.success
            else:
                # Data is buffered, will be written later
                result.success = True
                result.records_loaded = len(prepared_records)
                result.metadata["buffered"] = True
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to load fundamentals: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
    
    def _estimate_record_size(self, record: Dict[str, Any]) -> int:
        """Estimate size of a fundamentals record."""
        # Base size for standard fields
        size = 300
        
        # Add size for raw_data if present
        if 'raw_data' in record:
            size += len(json.dumps(record['raw_data']))
        
        return size
    
    async def _load_to_database(self, records: List[Dict[str, Any]]) -> int:
        """
        Load financial statement records to database using COPY.
        
        Args:
            records: Financial records to load
            
        Returns:
            Number of records loaded
        """
        if not records:
            return 0
        
        # Columns matching the enhanced financials_data schema
        columns = [
            'symbol', 'statement_type', 'year', 'period', 'revenue',
            'net_income', 'total_assets', 'total_liabilities', 'operating_cash_flow',
            'filing_date', 'gross_profit', 'operating_income', 'eps_basic', 'eps_diluted',
            'current_assets', 'current_liabilities', 'stockholders_equity',
            'source', 'raw_data', 'created_at', 'updated_at'
        ]
        
        # Convert records to tuples for COPY
        copy_records = []
        for record in records:
            copy_record = (
                record['symbol'],
                record.get('statement_type', 'income_statement'),
                record['year'],
                record['period'],
                record.get('revenue'),
                record.get('net_income'),
                record.get('total_assets'),
                record.get('total_liabilities'),
                record.get('operating_cash_flow'),
                self._format_date_for_copy(record.get('filing_date')),
                record.get('gross_profit'),
                record.get('operating_income'),
                record.get('eps_basic'),
                record.get('eps_diluted'),
                record.get('current_assets'),
                record.get('current_liabilities'),
                record.get('stockholders_equity'),
                record['source'],
                json.dumps(record.get('raw_data', {})),
                record['created_at'],
                record['updated_at']
            )
            copy_records.append(copy_record)
        
        async with self.db_adapter.acquire() as conn:
            try:
                # Create temp table
                await conn.execute("DROP TABLE IF EXISTS temp_financials")
                await conn.execute(
                    "CREATE TEMP TABLE temp_financials (LIKE financials_data INCLUDING ALL)"
                )
                
                # Use COPY to load data
                await conn.copy_records_to_table(
                    'temp_financials',
                    records=copy_records,
                    columns=columns
                )
                
                # UPSERT from temp table
                upsert_sql = """
                INSERT INTO financials_data
                SELECT * FROM temp_financials
                ON CONFLICT (symbol, statement_type, year, period)
                DO UPDATE SET
                    revenue = EXCLUDED.revenue,
                    net_income = EXCLUDED.net_income,
                    total_assets = EXCLUDED.total_assets,
                    total_liabilities = EXCLUDED.total_liabilities,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    filing_date = EXCLUDED.filing_date,
                    gross_profit = EXCLUDED.gross_profit,
                    operating_income = EXCLUDED.operating_income,
                    eps_basic = EXCLUDED.eps_basic,
                    eps_diluted = EXCLUDED.eps_diluted,
                    current_assets = EXCLUDED.current_assets,
                    current_liabilities = EXCLUDED.current_liabilities,
                    stockholders_equity = EXCLUDED.stockholders_equity,
                    source = EXCLUDED.source,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = EXCLUDED.updated_at
                """
                
                result = await conn.execute(upsert_sql)
                
                # Clean up
                await conn.execute("DROP TABLE temp_financials")
                
                # Extract count
                if result and result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        return int(parts[2])
                
                return len(records)
                
            except Exception as e:
                logger.warning(f"COPY failed for fundamentals: {e}, falling back to INSERT")
                # Fall back to INSERT method
                return await self._load_with_insert(records)
    
    async def _load_with_insert(self, records: List[Dict[str, Any]]) -> int:
        """Fallback INSERT method for loading fundamentals."""
        batch_size = 50  # Smaller batches for fundamentals
        total_loaded = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Build parameterized insert
            placeholders = []
            values = []
            param_count = 1
            
            for record in batch:
                # 21 parameters per record
                params = []
                for j in range(21):
                    params.append(f"${param_count + j}")
                placeholder = f"({','.join(params)})"
                placeholders.append(placeholder)
                
                values.extend([
                    record['symbol'],
                    record.get('statement_type', 'income_statement'),
                    record['year'],
                    record['period'],
                    record.get('revenue'),
                    record.get('net_income'),
                    record.get('total_assets'),
                    record.get('total_liabilities'),
                    record.get('operating_cash_flow'),
                    record.get('filing_date'),
                    record.get('gross_profit'),
                    record.get('operating_income'),
                    record.get('eps_basic'),
                    record.get('eps_diluted'),
                    record.get('current_assets'),
                    record.get('current_liabilities'),
                    record.get('stockholders_equity'),
                    record['source'],
                    record.get('raw_data', {}),  # asyncpg handles JSON
                    record['created_at'],
                    record['updated_at']
                ])
                
                param_count += 21
            
            sql = f"""
            INSERT INTO financials_data (
                symbol, statement_type, year, period, revenue,
                net_income, total_assets, total_liabilities, operating_cash_flow,
                filing_date, gross_profit, operating_income, eps_basic, eps_diluted,
                current_assets, current_liabilities, stockholders_equity,
                source, raw_data, created_at, updated_at
            )
            VALUES {','.join(placeholders)}
            ON CONFLICT (symbol, statement_type, year, period)
            DO UPDATE SET
                revenue = EXCLUDED.revenue,
                net_income = EXCLUDED.net_income,
                total_assets = EXCLUDED.total_assets,
                total_liabilities = EXCLUDED.total_liabilities,
                operating_cash_flow = EXCLUDED.operating_cash_flow,
                filing_date = EXCLUDED.filing_date,
                gross_profit = EXCLUDED.gross_profit,
                operating_income = EXCLUDED.operating_income,
                eps_basic = EXCLUDED.eps_basic,
                eps_diluted = EXCLUDED.eps_diluted,
                current_assets = EXCLUDED.current_assets,
                current_liabilities = EXCLUDED.current_liabilities,
                stockholders_equity = EXCLUDED.stockholders_equity,
                source = EXCLUDED.source,
                raw_data = EXCLUDED.raw_data,
                updated_at = EXCLUDED.updated_at
            """
            
            async with self.db_adapter.acquire() as conn:
                result = await conn.execute(sql, *values)
                if result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        total_loaded += int(parts[2])
        
        return total_loaded
    
    def _format_date_for_copy(self, date_value: Any) -> Optional[date]:
        """Format date field for COPY operation."""
        if date_value is None:
            return None
        
        if isinstance(date_value, date):
            return date_value
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, str):
            try:
                return datetime.strptime(date_value, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                return None
        
        return None
    
    async def _archive_records(self, records: List[Dict[str, Any]]) -> None:
        """Archive financial records to cold storage."""
        if not self.archive or not records:
            return
        
        # Group by year and period for archiving
        from collections import defaultdict
        period_groups = defaultdict(list)
        
        for record in records:
            key = f"{record['year']}_{record['period']}"
            period_groups[key].append(record)
        
        # Archive each group
        for period_key, group_records in period_groups.items():
            try:
                # Extract year and period from key
                year, period = period_key.split('_')
                
                # Create archive metadata
                metadata = {
                    'data_type': 'fundamentals',
                    'year': year,
                    'period': period,
                    'record_count': len(group_records),
                    'symbols': list(set(
                        record.get('symbol', '') 
                        for record in group_records if record.get('symbol')
                    )),
                    'metrics': {
                        'has_revenue': sum(1 for r in group_records if r.get('revenue') is not None),
                        'has_eps': sum(1 for r in group_records if r.get('eps_basic') is not None),
                        'has_net_income': sum(1 for r in group_records if r.get('net_income') is not None)
                    }
                }
                
                # Create RawDataRecord for archive
                from main.data_pipeline.storage.archive import RawDataRecord
                
                record = RawDataRecord(
                    source=group_records[0].get('source', 'polygon'),
                    data_type='fundamentals',
                    symbol=period_key,  # Use year_period as identifier
                    timestamp=datetime.now(timezone.utc),
                    data={'financials': group_records},
                    metadata=metadata
                )
                
                # Use archive's async save method
                await self.archive.save_raw_record_async(record)
                
                logger.debug(
                    f"Archived {len(group_records)} financial records for {period_key} "
                    f"({len(metadata['symbols'])} unique symbols)"
                )
            except Exception as e:
                logger.error(f"Failed to archive fundamentals for {period_key}: {e}")
    
    def reset_metrics(self):
        """Reset metrics including format handler duplicates."""
        super().reset_metrics()
        if self.format_factory:
            self.format_factory.reset_all_handlers()
        logger.info("Fundamentals bulk loader metrics reset")
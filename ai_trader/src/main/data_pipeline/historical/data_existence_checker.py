"""
Data Existence Checker Service

Checks what data actually exists in hot and cold storage.
Queries databases and archives to verify data presence.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from main.utils.core import get_logger, ensure_utc
from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.archive import DataArchive
# Repository imports removed - initialized via factory in constructor
from main.data_pipeline.services.storage import TableRoutingService
from main.data_pipeline.types import DataType, TimeInterval


@dataclass
class DataExistenceInfo:
    """Information about data existence."""
    timestamp: datetime
    exists: bool
    source: str  # 'hot', 'cold', or 'missing'
    record_count: int = 0
    data_quality: str = "unknown"  # 'good', 'partial', 'poor'


class DataExistenceChecker:
    """
    Service for checking data existence across storage layers.
    
    Queries hot storage (database) and cold storage (archive)
    to determine what data actually exists.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: Optional[DataArchive] = None,
        market_data_repo: Optional[Any] = None,
        news_repo: Optional[Any] = None,
        table_routing: Optional[TableRoutingService] = None
    ):
        """
        Initialize the data existence checker.
        
        Args:
            db_adapter: Database adapter for hot storage
            archive: Optional archive for cold storage
            market_data_repo: Market data repository
            news_repo: News repository
            table_routing: Table routing service
        """
        self.db_adapter = db_adapter
        self.archive = archive
        
        # Initialize repositories using factory if not provided
        if market_data_repo is None or news_repo is None:
            from main.data_pipeline.storage.repositories import get_repository_factory
            factory = get_repository_factory()
            self.market_data_repo = market_data_repo or factory.create_market_data_repository(db_adapter)
            self.news_repo = news_repo or factory.create_news_repository(db_adapter)
        else:
            self.market_data_repo = market_data_repo
            self.news_repo = news_repo
            
        self.table_routing = table_routing or TableRoutingService()
        
        self.logger = get_logger(__name__)
        
        # Cache for existence checks
        self._existence_cache: Dict[str, DataExistenceInfo] = {}
    
    async def get_actual_data_points(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
        check_archive: bool = True
    ) -> List[DataExistenceInfo]:
        """
        Get actual data points that exist in storage.
        
        Args:
            symbol: Symbol to check
            data_type: Type of data
            interval: Time interval
            start_date: Start date
            end_date: End date
            check_archive: Whether to check cold storage
            
        Returns:
            List of data existence information
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)
        
        self.logger.debug(
            f"Checking existence for {symbol} {data_type.value} {interval.value}"
        )
        
        # Check hot storage first
        hot_data = await self._query_hot_storage(
            symbol, data_type, interval, start_date, end_date
        )
        
        # Check cold storage if requested
        cold_data = []
        if check_archive and self.archive:
            cold_data = await self._query_cold_storage(
                symbol, data_type, interval, start_date, end_date
            )
        
        # Merge results (hot storage takes precedence)
        return self._merge_existence_data(hot_data, cold_data)
    
    async def _query_hot_storage(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime
    ) -> List[DataExistenceInfo]:
        """Query hot storage (database) for data existence."""
        existence_data = []
        
        try:
            if data_type == DataType.MARKET_DATA:
                existence_data = await self._check_market_data_hot(
                    symbol, interval, start_date, end_date
                )
            elif data_type == DataType.NEWS:
                existence_data = await self._check_news_hot(
                    symbol, start_date, end_date
                )
            else:
                self.logger.warning(f"Unsupported data type for hot storage: {data_type}")
                
        except Exception as e:
            self.logger.error(f"Error querying hot storage: {e}")
        
        return existence_data
    
    async def _query_cold_storage(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime
    ) -> List[DataExistenceInfo]:
        """Query cold storage (archive) for data existence."""
        existence_data = []
        
        try:
            # Query archive for raw records
            records = await self.archive.query_raw_records(
                source='polygon',  # Primary source
                data_type=data_type.value,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Process archive records
            for record in records:
                if record.data and 'data' in record.data:
                    record_count = len(record.data['data'])
                    
                    existence_data.append(DataExistenceInfo(
                        timestamp=record.timestamp,
                        exists=True,
                        source='cold',
                        record_count=record_count,
                        data_quality=self._assess_data_quality(record_count)
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error querying cold storage: {e}")
        
        return existence_data
    
    async def _check_market_data_hot(
        self,
        symbol: str,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime
    ) -> List[DataExistenceInfo]:
        """Check market data existence in hot storage."""
        # Get appropriate table
        table_name = self.table_routing.get_table_for_interval(interval.value)
        
        query = f"""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour_bucket,
                COUNT(*) as record_count,
                MIN(timestamp) as min_time,
                MAX(timestamp) as max_time
            FROM {table_name}
            WHERE symbol = %s
            AND timestamp >= %s
            AND timestamp <= %s
            GROUP BY DATE_TRUNC('hour', timestamp)
            ORDER BY hour_bucket
        """
        
        existence_data = []
        
        try:
            rows = await self.db_adapter.fetch_all(query, symbol, start_date, end_date)
            
            for row in rows:
                existence_data.append(DataExistenceInfo(
                    timestamp=row['hour_bucket'],
                    exists=True,
                    source='hot',
                    record_count=row['record_count'],
                    data_quality=self._assess_data_quality(row['record_count'])
                ))
                
        except Exception as e:
            self.logger.error(f"Error checking market data hot storage: {e}")
        
        return existence_data
    
    async def _check_news_hot(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[DataExistenceInfo]:
        """Check news data existence in hot storage."""
        query = """
            SELECT 
                DATE_TRUNC('day', published_at) as day_bucket,
                COUNT(*) as record_count
            FROM news_data
            WHERE symbol = %s
            AND published_at >= %s
            AND published_at <= %s
            GROUP BY DATE_TRUNC('day', published_at)
            ORDER BY day_bucket
        """
        
        existence_data = []
        
        try:
            rows = await self.db_adapter.fetch_all(query, symbol, start_date, end_date)
            
            for row in rows:
                existence_data.append(DataExistenceInfo(
                    timestamp=row['day_bucket'],
                    exists=True,
                    source='hot',
                    record_count=row['record_count'],
                    data_quality=self._assess_news_quality(row['record_count'])
                ))
                
        except Exception as e:
            self.logger.error(f"Error checking news hot storage: {e}")
        
        return existence_data
    
    def _merge_existence_data(
        self,
        hot_data: List[DataExistenceInfo],
        cold_data: List[DataExistenceInfo]
    ) -> List[DataExistenceInfo]:
        """
        Merge hot and cold storage data.
        Hot storage takes precedence over cold storage.
        """
        # Create timestamp index for hot data
        hot_timestamps = {info.timestamp: info for info in hot_data}
        
        # Start with hot data
        merged_data = list(hot_data)
        
        # Add cold data that's not in hot storage
        for cold_info in cold_data:
            if cold_info.timestamp not in hot_timestamps:
                merged_data.append(cold_info)
        
        # Sort by timestamp
        merged_data.sort(key=lambda x: x.timestamp)
        
        return merged_data
    
    def _assess_data_quality(self, record_count: int) -> str:
        """Assess data quality based on record count."""
        if record_count == 0:
            return "missing"
        elif record_count < 5:
            return "poor"
        elif record_count < 50:
            return "partial"
        else:
            return "good"
    
    def _assess_news_quality(self, record_count: int) -> str:
        """Assess news data quality (different thresholds)."""
        if record_count == 0:
            return "missing"
        elif record_count < 2:
            return "poor"
        elif record_count < 10:
            return "partial"
        else:
            return "good"
    
    def get_existence_summary(
        self,
        existence_data: List[DataExistenceInfo]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for existence data.
        
        Args:
            existence_data: List of existence info
            
        Returns:
            Summary statistics
        """
        if not existence_data:
            return {
                "total_points": 0,
                "exists_count": 0,
                "missing_count": 0,
                "sources": {},
                "quality": {}
            }
        
        total_points = len(existence_data)
        exists_count = sum(1 for info in existence_data if info.exists)
        missing_count = total_points - exists_count
        
        # Count by source
        sources = {}
        for info in existence_data:
            sources[info.source] = sources.get(info.source, 0) + 1
        
        # Count by quality
        quality = {}
        for info in existence_data:
            quality[info.data_quality] = quality.get(info.data_quality, 0) + 1
        
        return {
            "total_points": total_points,
            "exists_count": exists_count,
            "missing_count": missing_count,
            "coverage_percentage": (exists_count / total_points * 100) if total_points > 0 else 0,
            "sources": sources,
            "quality": quality
        }
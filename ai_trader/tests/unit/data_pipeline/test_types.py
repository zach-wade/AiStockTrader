"""
Unit tests for data pipeline type definitions.
"""

import pytest
from datetime import datetime, timezone, timedelta

from main.data_pipeline.types import (
    DataType,
    DataSource,
    DataPipelineStatus,
    TimeInterval,
    RawDataRecord,
    ProcessedDataRecord,
    DataPipelineResult,
    DataQualityMetrics,
    DataRequest,
    StreamConfig,
    GapType,
    GapInfo,
    SymbolGaps,
    BackfillParams,
    BackfillSummary
)


class TestEnums:
    """Test enum definitions."""
    
    def test_data_type_enum(self):
        """Test DataType enum values."""
        assert DataType.MARKET_DATA.value == "market_data"
        assert DataType.NEWS.value == "news"
        assert DataType.SOCIAL_SENTIMENT.value == "social_sentiment"
        assert DataType.CORPORATE_ACTIONS.value == "corporate_actions"
        assert DataType.OPTIONS.value == "options"
        assert DataType.FUNDAMENTALS.value == "fundamentals"
        assert DataType.INSIDER.value == "insider"
        assert DataType.ANALYST.value == "analyst"
        assert DataType.REALTIME.value == "realtime"
    
    def test_data_source_enum(self):
        """Test DataSource enum values."""
        assert DataSource.ALPACA.value == "alpaca"
        assert DataSource.POLYGON.value == "polygon"
        assert DataSource.YAHOO.value == "yahoo"
        assert DataSource.REDDIT.value == "reddit"
        assert DataSource.BENZINGA.value == "benzinga"
        assert DataSource.FINNHUB.value == "finnhub"
        assert DataSource.IEX.value == "iex"
    
    def test_data_pipeline_status_enum(self):
        """Test DataPipelineStatus enum values."""
        assert DataPipelineStatus.SUCCESS.value == "success"
        assert DataPipelineStatus.PARTIAL.value == "partial"
        assert DataPipelineStatus.FAILED.value == "failed"
    
    def test_time_interval_enum(self):
        """Test TimeInterval enum values."""
        assert TimeInterval.MINUTE_1.value == "1min"
        assert TimeInterval.MINUTE_5.value == "5min"
        assert TimeInterval.HOUR_1.value == "1hour"
        assert TimeInterval.DAY_1.value == "1day"
        assert TimeInterval.WEEK_1.value == "1week"
    
    def test_gap_type_enum(self):
        """Test GapType enum values."""
        assert GapType.WEEKEND.value == "weekend"
        assert GapType.HOLIDAY.value == "holiday"
        assert GapType.AFTER_HOURS.value == "after_hours"
        assert GapType.DATA_MISSING.value == "data_missing"
        assert GapType.SYSTEM_OUTAGE.value == "system_outage"


class TestRawDataRecord:
    """Test RawDataRecord dataclass."""
    
    def test_raw_data_record_creation(self):
        """Test creating a RawDataRecord."""
        timestamp = datetime.now(timezone.utc)
        data = {'open': 150.0, 'close': 151.0}
        metadata = {'api_version': '2'}
        
        record = RawDataRecord(
            source='alpaca',
            data_type='market_data',
            symbol='AAPL',
            timestamp=timestamp,
            data=data,
            metadata=metadata
        )
        
        assert record.source == 'alpaca'
        assert record.data_type == 'market_data'
        assert record.symbol == 'AAPL'
        assert record.timestamp == timestamp
        assert record.data == data
        assert record.metadata == metadata
    
    def test_raw_data_record_to_dict(self):
        """Test converting RawDataRecord to dictionary."""
        timestamp = datetime.now(timezone.utc)
        record = RawDataRecord(
            source='polygon',
            data_type='news',
            symbol='MSFT',
            timestamp=timestamp,
            data={'headline': 'Test news'},
            metadata={'source_id': '123'}
        )
        
        record_dict = record.to_dict()
        
        assert record_dict['source'] == 'polygon'
        assert record_dict['data_type'] == 'news'
        assert record_dict['symbol'] == 'MSFT'
        assert record_dict['timestamp'] == timestamp.isoformat()
        assert record_dict['data'] == {'headline': 'Test news'}
        assert record_dict['metadata'] == {'source_id': '123'}


class TestProcessedDataRecord:
    """Test ProcessedDataRecord dataclass."""
    
    def test_processed_data_record_creation(self):
        """Test creating a ProcessedDataRecord."""
        timestamp = datetime.now(timezone.utc)
        features = {'sma_20': 150.5, 'rsi': 65.0}
        raw_refs = ['ref_1', 'ref_2']
        metadata = {'processing_version': '1.0'}
        
        record = ProcessedDataRecord(
            symbol='AAPL',
            timestamp=timestamp,
            features=features,
            raw_data_refs=raw_refs,
            processing_metadata=metadata
        )
        
        assert record.symbol == 'AAPL'
        assert record.timestamp == timestamp
        assert record.features == features
        assert record.raw_data_refs == raw_refs
        assert record.processing_metadata == metadata
    
    def test_processed_data_record_to_dict(self):
        """Test converting ProcessedDataRecord to dictionary."""
        timestamp = datetime.now(timezone.utc)
        record = ProcessedDataRecord(
            symbol='GOOGL',
            timestamp=timestamp,
            features={'volume_ratio': 1.2},
            raw_data_refs=['ref_3']
        )
        
        record_dict = record.to_dict()
        
        assert record_dict['symbol'] == 'GOOGL'
        assert record_dict['timestamp'] == timestamp.isoformat()
        assert record_dict['features'] == {'volume_ratio': 1.2}
        assert record_dict['raw_data_refs'] == ['ref_3']
        assert record_dict['processing_metadata'] == {}


class TestDataPipelineResult:
    """Test DataPipelineResult dataclass."""
    
    def test_data_pipeline_result_creation(self):
        """Test creating a DataPipelineResult."""
        result = DataPipelineResult(
            status=DataPipelineStatus.SUCCESS,
            records_processed=100,
            records_failed=5,
            errors=['Error 1', 'Error 2'],
            metadata={'duration': 10.5}
        )
        
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.records_processed == 100
        assert result.records_failed == 5
        assert result.errors == ['Error 1', 'Error 2']
        assert result.metadata == {'duration': 10.5}
    
    def test_data_pipeline_result_success_rate(self):
        """Test success rate calculation."""
        # All successful
        result1 = DataPipelineResult(
            status=DataPipelineStatus.SUCCESS,
            records_processed=100,
            records_failed=0
        )
        assert result1.success_rate == 1.0
        
        # Partial success
        result2 = DataPipelineResult(
            status=DataPipelineStatus.PARTIAL,
            records_processed=80,
            records_failed=20
        )
        assert result2.success_rate == 0.8
        
        # No records
        result3 = DataPipelineResult(
            status=DataPipelineStatus.FAILED,
            records_processed=0,
            records_failed=0
        )
        assert result3.success_rate == 0.0


class TestDataQualityMetrics:
    """Test DataQualityMetrics dataclass."""
    
    def test_data_quality_metrics_creation(self):
        """Test creating DataQualityMetrics."""
        metrics = DataQualityMetrics(
            completeness=0.95,
            timeliness=0.90,
            accuracy=0.98,
            consistency=0.92,
            metadata={'checked_fields': 10}
        )
        
        assert metrics.completeness == 0.95
        assert metrics.timeliness == 0.90
        assert metrics.accuracy == 0.98
        assert metrics.consistency == 0.92
        assert metrics.metadata == {'checked_fields': 10}
    
    def test_overall_quality_calculation(self):
        """Test overall quality score calculation."""
        metrics = DataQualityMetrics(
            completeness=0.80,
            timeliness=0.90,
            accuracy=0.95,
            consistency=0.85
        )
        
        expected = (0.80 + 0.90 + 0.95 + 0.85) / 4.0
        assert metrics.overall_quality == expected


class TestGapInfo:
    """Test GapInfo dataclass."""
    
    def test_gap_info_creation(self):
        """Test creating GapInfo."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=2)
        
        gap = GapInfo(
            start_date=start,
            end_date=end,
            gap_type=GapType.WEEKEND,
            interval=TimeInterval.HOUR_1,
            size_minutes=120,
            size_points=2,
            severity=0.3,
            metadata={'reason': 'market closed'}
        )
        
        assert gap.start_date == start
        assert gap.end_date == end
        assert gap.gap_type == GapType.WEEKEND
        assert gap.interval == TimeInterval.HOUR_1
        assert gap.size_minutes == 120
        assert gap.size_points == 2
        assert gap.severity == 0.3
    
    def test_gap_duration(self):
        """Test gap duration calculation."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=3, minutes=30)
        
        gap = GapInfo(
            start_date=start,
            end_date=end,
            gap_type=GapType.DATA_MISSING,
            interval=TimeInterval.MINUTE_5,
            size_minutes=210,
            size_points=42,
            severity=0.9
        )
        
        assert gap.duration == timedelta(hours=3, minutes=30)
    
    def test_is_critical(self):
        """Test critical gap detection."""
        # High severity gap
        gap1 = GapInfo(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(hours=1),
            gap_type=GapType.HOLIDAY,
            interval=TimeInterval.DAY_1,
            size_minutes=60,
            size_points=1,
            severity=0.85
        )
        assert gap1.is_critical() is True
        
        # Data missing gap
        gap2 = GapInfo(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(minutes=30),
            gap_type=GapType.DATA_MISSING,
            interval=TimeInterval.MINUTE_1,
            size_minutes=30,
            size_points=30,
            severity=0.5
        )
        assert gap2.is_critical() is True
        
        # Non-critical gap
        gap3 = GapInfo(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(hours=2),
            gap_type=GapType.WEEKEND,
            interval=TimeInterval.HOUR_1,
            size_minutes=120,
            size_points=2,
            severity=0.3
        )
        assert gap3.is_critical() is False


class TestSymbolGaps:
    """Test SymbolGaps dataclass."""
    
    def test_symbol_gaps_creation(self):
        """Test creating SymbolGaps."""
        now = datetime.now(timezone.utc)
        gap1 = GapInfo(
            start_date=now,
            end_date=now + timedelta(hours=1),
            gap_type=GapType.DATA_MISSING,
            interval=TimeInterval.MINUTE_5,
            size_minutes=60,
            size_points=12,
            severity=0.9
        )
        gap2 = GapInfo(
            start_date=now + timedelta(days=1),
            end_date=now + timedelta(days=1, hours=2),
            gap_type=GapType.WEEKEND,
            interval=TimeInterval.HOUR_1,
            size_minutes=120,
            size_points=2,
            severity=0.2
        )
        
        symbol_gaps = SymbolGaps(
            symbol='AAPL',
            interval=TimeInterval.MINUTE_5,
            analysis_date=now,
            total_gaps=2,
            gaps=[gap1, gap2],
            coverage_pct=0.95,
            data_quality_score=0.92
        )
        
        assert symbol_gaps.symbol == 'AAPL'
        assert symbol_gaps.total_gaps == 2
        assert len(symbol_gaps.gaps) == 2
        assert symbol_gaps.coverage_pct == 0.95
    
    def test_get_critical_gaps(self):
        """Test getting critical gaps."""
        now = datetime.now(timezone.utc)
        critical_gap = GapInfo(
            start_date=now,
            end_date=now + timedelta(hours=1),
            gap_type=GapType.DATA_MISSING,
            interval=TimeInterval.MINUTE_1,
            size_minutes=60,
            size_points=60,
            severity=0.9
        )
        normal_gap = GapInfo(
            start_date=now + timedelta(days=1),
            end_date=now + timedelta(days=1, hours=1),
            gap_type=GapType.AFTER_HOURS,
            interval=TimeInterval.MINUTE_1,
            size_minutes=60,
            size_points=60,
            severity=0.3
        )
        
        symbol_gaps = SymbolGaps(
            symbol='MSFT',
            interval=TimeInterval.MINUTE_1,
            analysis_date=now,
            total_gaps=2,
            gaps=[critical_gap, normal_gap]
        )
        
        critical_gaps = symbol_gaps.get_critical_gaps()
        assert len(critical_gaps) == 1
        assert critical_gaps[0] == critical_gap
    
    def test_get_gaps_by_type(self):
        """Test getting gaps by type."""
        now = datetime.now(timezone.utc)
        weekend_gap = GapInfo(
            start_date=now,
            end_date=now + timedelta(days=2),
            gap_type=GapType.WEEKEND,
            interval=TimeInterval.DAY_1,
            size_minutes=2880,
            size_points=2,
            severity=0.2
        )
        holiday_gap = GapInfo(
            start_date=now + timedelta(days=7),
            end_date=now + timedelta(days=8),
            gap_type=GapType.HOLIDAY,
            interval=TimeInterval.DAY_1,
            size_minutes=1440,
            size_points=1,
            severity=0.2
        )
        
        symbol_gaps = SymbolGaps(
            symbol='GOOGL',
            interval=TimeInterval.DAY_1,
            analysis_date=now,
            total_gaps=2,
            gaps=[weekend_gap, holiday_gap]
        )
        
        weekend_gaps = symbol_gaps.get_gaps_by_type(GapType.WEEKEND)
        assert len(weekend_gaps) == 1
        assert weekend_gaps[0] == weekend_gap
        
        holiday_gaps = symbol_gaps.get_gaps_by_type(GapType.HOLIDAY)
        assert len(holiday_gaps) == 1
        assert holiday_gaps[0] == holiday_gap
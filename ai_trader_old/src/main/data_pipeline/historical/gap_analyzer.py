"""
Gap Analyzer Service

Analyzes data gaps by comparing expected vs actual data points.
Identifies, merges, and calculates gap characteristics.
"""

# Standard library imports
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# Local imports
from main.data_pipeline.types import DataType, GapInfo, TimeInterval
from main.utils.core import get_logger

from .data_existence_checker import DataExistenceInfo
from .timeline_analyzer import DataPointInfo


@dataclass
class GapSegment:
    """Information about a data gap segment."""

    start_time: datetime
    end_time: datetime
    expected_points: int
    missing_points: int
    gap_type: str  # 'complete', 'partial', 'quality'
    severity: str  # 'low', 'medium', 'high', 'critical'


class GapAnalyzer:
    """
    Service for analyzing data gaps.

    Compares expected data timeline with actual data existence
    to identify, characterize, and prioritize gaps.
    """

    def __init__(self):
        """Initialize the gap analyzer."""
        self.logger = get_logger(__name__)

    def identify_gaps(
        self,
        expected_timeline: list[DataPointInfo],
        actual_data: list[DataExistenceInfo],
        data_type: DataType,
        interval: TimeInterval,
    ) -> list[GapSegment]:
        """
        Identify gaps by comparing expected vs actual data.

        Args:
            expected_timeline: Expected data points
            actual_data: Actual data existence info
            data_type: Type of data being analyzed
            interval: Time interval

        Returns:
            List of identified gap segments
        """
        if not expected_timeline:
            return []

        self.logger.debug(
            f"Analyzing gaps: {len(expected_timeline)} expected, "
            f"{len(actual_data)} actual points"
        )

        # Create lookup for actual data
        actual_lookup = self._create_existence_lookup(actual_data)

        # Find missing and poor quality points
        gap_points = []
        for expected in expected_timeline:
            existence = actual_lookup.get(expected.timestamp)

            if not existence:
                # Completely missing
                gap_points.append((expected.timestamp, "complete"))
            elif not existence.exists:
                # Marked as not existing
                gap_points.append((expected.timestamp, "complete"))
            elif existence.data_quality in ["poor", "missing"]:
                # Poor quality data
                gap_points.append((expected.timestamp, "quality"))
            elif existence.data_quality == "partial":
                # Partial data
                gap_points.append((expected.timestamp, "partial"))

        # Convert gap points to segments
        gap_segments = self._create_gap_segments(gap_points, interval)

        # Merge consecutive gaps
        merged_segments = self._merge_consecutive_gaps(gap_segments, interval)

        # Calculate gap characteristics
        analyzed_segments = self._analyze_gap_characteristics(merged_segments, data_type, interval)

        return analyzed_segments

    def _create_existence_lookup(
        self, actual_data: list[DataExistenceInfo]
    ) -> dict[datetime, DataExistenceInfo]:
        """Create timestamp lookup for actual data."""
        lookup = {}

        for data_point in actual_data:
            # For hourly aggregated data, match to closest hour
            hour_key = data_point.timestamp.replace(minute=0, second=0, microsecond=0)
            lookup[hour_key] = data_point

            # Also add exact timestamp
            lookup[data_point.timestamp] = data_point

        return lookup

    def _create_gap_segments(
        self, gap_points: list[tuple[datetime, str]], interval: TimeInterval
    ) -> list[GapSegment]:
        """Convert gap points to initial segments."""
        segments = []
        interval_minutes = self._get_interval_minutes(interval)

        for timestamp, gap_type in gap_points:
            # Each point represents one missing interval
            end_time = timestamp + timedelta(minutes=interval_minutes)

            segments.append(
                GapSegment(
                    start_time=timestamp,
                    end_time=end_time,
                    expected_points=1,
                    missing_points=1,
                    gap_type=gap_type,
                    severity="medium",  # Will be recalculated
                )
            )

        return segments

    def _merge_consecutive_gaps(
        self, segments: list[GapSegment], interval: TimeInterval
    ) -> list[GapSegment]:
        """Merge consecutive gap segments."""
        if not segments:
            return []

        # Sort by start time
        segments.sort(key=lambda x: x.start_time)

        merged = []
        current_segment = segments[0]
        interval_minutes = self._get_interval_minutes(interval)
        merge_threshold = timedelta(minutes=interval_minutes * 2)  # Allow small gaps

        for next_segment in segments[1:]:
            # Check if segments can be merged
            gap_between = next_segment.start_time - current_segment.end_time

            if gap_between <= merge_threshold and current_segment.gap_type == next_segment.gap_type:
                # Merge segments
                current_segment = GapSegment(
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time,
                    expected_points=current_segment.expected_points + next_segment.expected_points,
                    missing_points=current_segment.missing_points + next_segment.missing_points,
                    gap_type=current_segment.gap_type,
                    severity=current_segment.severity,
                )
            else:
                # Start new segment
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)

        return merged

    def _analyze_gap_characteristics(
        self, segments: list[GapSegment], data_type: DataType, interval: TimeInterval
    ) -> list[GapSegment]:
        """Analyze and set gap characteristics."""
        for segment in segments:
            # Calculate gap duration
            duration = segment.end_time - segment.start_time

            # Determine severity based on gap size and data type
            segment.severity = self._calculate_gap_severity(
                duration, segment.missing_points, data_type, interval
            )

        return segments

    def _calculate_gap_severity(
        self, duration: timedelta, missing_points: int, data_type: DataType, interval: TimeInterval
    ) -> str:
        """Calculate gap severity based on characteristics."""
        duration_hours = duration.total_seconds() / 3600

        if data_type == DataType.MARKET_DATA:
            if interval == TimeInterval.ONE_DAY:
                # Daily data gaps
                if missing_points >= 5:  # Week or more
                    return "critical"
                elif missing_points >= 3:  # 3+ days
                    return "high"
                elif missing_points >= 1:  # 1-2 days
                    return "medium"
                else:
                    return "low"
            elif duration_hours >= 24:  # Full day or more
                return "critical"
            elif duration_hours >= 6:  # Half day
                return "high"
            elif duration_hours >= 2:  # Multiple hours
                return "medium"
            else:
                return "low"

        elif data_type == DataType.NEWS:
            # News gaps are less critical
            if duration.days >= 7:  # Week or more
                return "high"
            elif duration.days >= 3:  # 3+ days
                return "medium"
            else:
                return "low"

        elif duration_hours >= 24:
            return "high"
        elif duration_hours >= 6:
            return "medium"
        else:
            return "low"

    def _get_interval_minutes(self, interval: TimeInterval) -> int:
        """Convert interval to minutes."""
        interval_map = {
            TimeInterval.ONE_MIN: 1,
            TimeInterval.FIVE_MIN: 5,
            TimeInterval.FIFTEEN_MIN: 15,
            TimeInterval.THIRTY_MIN: 30,
            TimeInterval.ONE_HOUR: 60,
            TimeInterval.ONE_DAY: 1440,
        }
        return interval_map.get(interval, 60)

    def convert_to_gap_info(
        self, segments: list[GapSegment], symbol: str, data_type: DataType, interval: TimeInterval
    ) -> list[GapInfo]:
        """Convert gap segments to GapInfo objects."""
        gap_infos = []

        for segment in segments:
            gap_info = GapInfo(
                symbol=symbol,
                data_type=data_type,
                interval=interval,
                start_date=segment.start_time,
                end_date=segment.end_time,
                expected_records=segment.expected_points,
                missing_records=segment.missing_points,
                priority_score=self._calculate_priority_score(segment),
                gap_reason=segment.gap_type,
                estimated_fill_time=self._estimate_fill_time(segment),
            )
            gap_infos.append(gap_info)

        return gap_infos

    def _calculate_priority_score(self, segment: GapSegment) -> float:
        """Calculate priority score for a gap segment."""
        # Base score from severity
        severity_scores = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}

        base_score = severity_scores.get(segment.severity, 2.0)

        # Adjust based on gap type
        type_multiplier = {"complete": 1.0, "partial": 0.7, "quality": 0.5}

        multiplier = type_multiplier.get(segment.gap_type, 1.0)

        # Adjust based on gap size
        size_factor = min(segment.missing_points / 10.0, 2.0)  # Cap at 2x

        return base_score * multiplier * (1 + size_factor)

    def _estimate_fill_time(self, segment: GapSegment) -> int:
        """Estimate time to fill gap in minutes."""
        # Base time per missing point
        base_time = 30  # 30 seconds per point

        # Scale with gap size (larger gaps take longer per point)
        scale_factor = 1 + (segment.missing_points / 100.0)

        return int(base_time * segment.missing_points * scale_factor / 60)  # Convert to minutes

    def get_gap_statistics(self, segments: list[GapSegment]) -> dict[str, Any]:
        """Get statistics about identified gaps."""
        if not segments:
            return {
                "total_gaps": 0,
                "total_missing_points": 0,
                "severity_distribution": {},
                "type_distribution": {},
            }

        total_missing = sum(s.missing_points for s in segments)

        # Count by severity
        severity_dist = defaultdict(int)
        for segment in segments:
            severity_dist[segment.severity] += 1

        # Count by type
        type_dist = defaultdict(int)
        for segment in segments:
            type_dist[segment.gap_type] += 1

        return {
            "total_gaps": len(segments),
            "total_missing_points": total_missing,
            "severity_distribution": dict(severity_dist),
            "type_distribution": dict(type_dist),
            "average_gap_size": total_missing / len(segments) if segments else 0,
        }

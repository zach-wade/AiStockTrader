"""
Validation Framework - Data Quality Interfaces

Additional data quality interfaces for advanced quality assessment,
profiling, and monitoring capabilities.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import IDataQualityCalculator, IValidationContext


class QualityDimension(Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"


class IQualityProfile(ABC):
    """Interface for data quality profiles."""

    @property
    @abstractmethod
    def profile_name(self) -> str:
        """Profile name."""
        pass

    @property
    @abstractmethod
    def quality_thresholds(self) -> dict[QualityDimension, float]:
        """Quality thresholds for each dimension."""
        pass

    @property
    @abstractmethod
    def required_dimensions(self) -> list[QualityDimension]:
        """Required quality dimensions for this profile."""
        pass

    @abstractmethod
    async def get_dimension_weight(self, dimension: QualityDimension) -> float:
        """Get weight for quality dimension in overall score."""
        pass

    @abstractmethod
    async def is_quality_acceptable(
        self, quality_scores: dict[QualityDimension, float]
    ) -> tuple[bool, list[str]]:
        """Check if quality scores meet profile requirements."""
        pass


class IDataProfiler(ABC):
    """Interface for comprehensive data profiling."""

    @abstractmethod
    async def profile_dataset(
        self, data: pd.DataFrame, context: IValidationContext
    ) -> dict[str, Any]:
        """Generate comprehensive data profile."""
        pass

    @abstractmethod
    async def profile_column(
        self, data: pd.Series, column_name: str, context: IValidationContext
    ) -> dict[str, Any]:
        """Profile a single column."""
        pass

    @abstractmethod
    async def detect_data_types(
        self, data: pd.DataFrame, context: IValidationContext
    ) -> dict[str, str]:
        """Detect optimal data types for columns."""
        pass

    @abstractmethod
    async def analyze_distributions(
        self,
        data: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        context: IValidationContext | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze statistical distributions of numeric columns."""
        pass

    @abstractmethod
    async def analyze_patterns(
        self,
        data: pd.DataFrame,
        text_columns: list[str] | None = None,
        context: IValidationContext | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze patterns in text columns."""
        pass

    @abstractmethod
    async def detect_anomalies(
        self, data: pd.DataFrame, context: IValidationContext
    ) -> dict[str, list[Any]]:
        """Detect anomalies in data."""
        pass


class IQualityMonitor(ABC):
    """Interface for continuous quality monitoring."""

    @abstractmethod
    async def start_monitoring(
        self,
        data_source: str,
        layer: DataLayer,
        data_type: DataType,
        monitoring_config: dict[str, Any],
    ) -> str:
        """Start quality monitoring for a data source."""
        pass

    @abstractmethod
    async def stop_monitoring(self, monitor_id: str) -> bool:
        """Stop quality monitoring."""
        pass

    @abstractmethod
    async def get_monitoring_status(self, monitor_id: str) -> dict[str, Any]:
        """Get monitoring status."""
        pass

    @abstractmethod
    async def record_quality_measurement(
        self,
        monitor_id: str,
        quality_scores: dict[QualityDimension, float],
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record quality measurement."""
        pass

    @abstractmethod
    async def get_quality_trends(
        self,
        monitor_id: str,
        time_range: tuple[datetime, datetime],
        dimensions: list[QualityDimension] | None = None,
    ) -> dict[str, list[tuple[datetime, float]]]:
        """Get quality trends over time."""
        pass

    @abstractmethod
    async def check_quality_alerts(
        self, monitor_id: str, current_scores: dict[QualityDimension, float]
    ) -> list[dict[str, Any]]:
        """Check for quality alerts based on thresholds."""
        pass


class IQualityComparator(ABC):
    """Interface for quality comparison operations."""

    @abstractmethod
    async def compare_quality_profiles(
        self,
        profile1: dict[str, Any],
        profile2: dict[str, Any],
        comparison_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two data quality profiles."""
        pass

    @abstractmethod
    async def compare_datasets(
        self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, context: IValidationContext
    ) -> dict[str, Any]:
        """Compare quality between two datasets."""
        pass

    @abstractmethod
    async def track_quality_regression(
        self,
        baseline_scores: dict[QualityDimension, float],
        current_scores: dict[QualityDimension, float],
        regression_threshold: float = 0.05,
    ) -> tuple[bool, dict[QualityDimension, float]]:
        """Track quality regression from baseline."""
        pass

    @abstractmethod
    async def identify_quality_improvements(
        self,
        historical_scores: list[dict[QualityDimension, float]],
        current_scores: dict[QualityDimension, float],
    ) -> dict[str, Any]:
        """Identify quality improvements over time."""
        pass


class IQualityEnhancer(ABC):
    """Interface for quality enhancement recommendations."""

    @abstractmethod
    async def analyze_quality_issues(
        self,
        data: pd.DataFrame,
        quality_scores: dict[QualityDimension, float],
        context: IValidationContext,
    ) -> list[dict[str, Any]]:
        """Analyze quality issues and their root causes."""
        pass

    @abstractmethod
    async def recommend_quality_improvements(
        self, quality_issues: list[dict[str, Any]], context: IValidationContext
    ) -> list[dict[str, Any]]:
        """Recommend quality improvement actions."""
        pass

    @abstractmethod
    async def estimate_improvement_impact(
        self,
        current_scores: dict[QualityDimension, float],
        proposed_actions: list[dict[str, Any]],
        context: IValidationContext,
    ) -> dict[QualityDimension, float]:
        """Estimate impact of proposed quality improvements."""
        pass

    @abstractmethod
    async def prioritize_improvements(
        self,
        improvement_recommendations: list[dict[str, Any]],
        business_impact_weights: dict[QualityDimension, float],
    ) -> list[dict[str, Any]]:
        """Prioritize quality improvement recommendations."""
        pass


class IQualityReportGenerator(ABC):
    """Interface for quality report generation."""

    @abstractmethod
    async def generate_quality_summary(
        self, quality_scores: dict[QualityDimension, float], context: IValidationContext
    ) -> dict[str, Any]:
        """Generate quality summary report."""
        pass

    @abstractmethod
    async def generate_detailed_quality_report(
        self,
        data_profile: dict[str, Any],
        quality_scores: dict[QualityDimension, float],
        quality_issues: list[dict[str, Any]],
        context: IValidationContext,
    ) -> dict[str, Any]:
        """Generate detailed quality report."""
        pass

    @abstractmethod
    async def generate_quality_dashboard_data(
        self, monitor_ids: list[str], time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """Generate data for quality dashboard."""
        pass

    @abstractmethod
    async def export_quality_metrics(
        self,
        quality_data: dict[str, Any],
        export_format: str = "json",
        include_metadata: bool = True,
    ) -> str:
        """Export quality metrics in specified format."""
        pass


class IAdvancedQualityCalculator(IDataQualityCalculator):
    """Extended interface for advanced quality calculations."""

    @abstractmethod
    async def calculate_dimensional_quality(
        self, data: Any, dimension: QualityDimension, context: IValidationContext
    ) -> float:
        """Calculate quality score for specific dimension."""
        pass

    @abstractmethod
    async def calculate_weighted_quality_score(
        self,
        dimensional_scores: dict[QualityDimension, float],
        weights: dict[QualityDimension, float],
    ) -> float:
        """Calculate weighted overall quality score."""
        pass

    @abstractmethod
    async def calculate_quality_confidence(
        self, data: Any, quality_scores: dict[QualityDimension, float], context: IValidationContext
    ) -> float:
        """Calculate confidence level in quality assessment."""
        pass

    @abstractmethod
    async def benchmark_quality(
        self,
        current_scores: dict[QualityDimension, float],
        benchmark_scores: dict[QualityDimension, float],
    ) -> dict[str, Any]:
        """Benchmark quality against reference scores."""
        pass

    @abstractmethod
    async def calculate_data_lineage_quality(
        self, lineage_chain: list[dict[str, Any]], context: IValidationContext
    ) -> dict[str, Any]:
        """Calculate quality impact through data lineage."""
        pass

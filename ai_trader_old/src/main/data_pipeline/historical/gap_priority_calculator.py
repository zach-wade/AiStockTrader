"""
Gap Priority Calculator

Strategy class for calculating priorities for data gaps based on
various factors like data type, layer, recency, and size.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

# Local imports
from main.data_pipeline.types import DataType


@dataclass
class PriorityConfig:
    """Configuration for priority calculation."""

    # Data type base priorities (lower = higher priority)
    data_type_priorities: dict[DataType, int] = None

    # Layer priority multipliers (higher multiplier = higher priority)
    layer_multipliers: dict[int, float] = None

    # Recency decay factor (days old factor)
    recency_decay_factor: float = 365.0

    # Size thresholds for priority boost
    large_gap_threshold: int = 100
    large_gap_boost: float = 0.8

    # Critical period (days) for highest priority
    critical_period_days: int = 7

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.data_type_priorities is None:
            self.data_type_priorities = {
                DataType.MARKET_DATA: 1,
                DataType.NEWS: 2,
                DataType.FINANCIALS: 3,
                DataType.CORPORATE_ACTIONS: 4,
                DataType.SOCIAL_SENTIMENT: 5,
            }

        if self.layer_multipliers is None:
            self.layer_multipliers = {
                1: 1.0,  # Layer 1: Highest priority
                2: 0.75,  # Layer 2: Medium priority
                3: 0.5,  # Layer 3: Lower priority
                4: 0.25,  # Layer 4+: Lowest priority
            }


class PriorityStrategy(Enum):
    """Priority calculation strategies."""

    STANDARD = "standard"
    RECENCY_FOCUSED = "recency_focused"
    SIZE_FOCUSED = "size_focused"
    LAYER_FOCUSED = "layer_focused"


class GapPriorityCalculator:
    """
    Calculates priorities for data gaps using configurable strategies.

    Lower priority values indicate higher importance (priority 1 is highest).
    """

    def __init__(
        self,
        config: PriorityConfig | None = None,
        strategy: PriorityStrategy = PriorityStrategy.STANDARD,
    ):
        """
        Initialize the priority calculator.

        Args:
            config: Priority calculation configuration
            strategy: Priority calculation strategy to use
        """
        self.config = config or PriorityConfig()
        self.strategy = strategy

    def calculate_priority(
        self,
        data_type: DataType,
        layer: int,
        start_date: datetime,
        end_date: datetime,
        gap_size: int = 1,
    ) -> int:
        """
        Calculate priority for a data gap.

        Args:
            data_type: Type of data
            layer: Symbol layer (1-3+)
            start_date: Gap start date
            end_date: Gap end date
            gap_size: Number of missing records

        Returns:
            Priority value (lower = higher priority)
        """
        if self.strategy == PriorityStrategy.STANDARD:
            return self._calculate_standard_priority(
                data_type, layer, start_date, end_date, gap_size
            )
        elif self.strategy == PriorityStrategy.RECENCY_FOCUSED:
            return self._calculate_recency_focused_priority(
                data_type, layer, start_date, end_date, gap_size
            )
        elif self.strategy == PriorityStrategy.SIZE_FOCUSED:
            return self._calculate_size_focused_priority(
                data_type, layer, start_date, end_date, gap_size
            )
        elif self.strategy == PriorityStrategy.LAYER_FOCUSED:
            return self._calculate_layer_focused_priority(
                data_type, layer, start_date, end_date, gap_size
            )
        else:
            return self._calculate_standard_priority(
                data_type, layer, start_date, end_date, gap_size
            )

    def _calculate_standard_priority(
        self,
        data_type: DataType,
        layer: int,
        start_date: datetime,
        end_date: datetime,
        gap_size: int,
    ) -> int:
        """
        Standard priority calculation balancing all factors.

        Priority = base_priority * (1 / layer_multiplier) * recency_factor * size_factor
        """
        # Base priority from data type
        base_priority = self.config.data_type_priorities.get(data_type, 10)

        # Layer adjustment
        layer_mult = self.config.layer_multipliers.get(
            layer, self.config.layer_multipliers.get(4, 0.25)
        )
        layer_factor = 1.0 / layer_mult if layer_mult > 0 else 4.0

        # Recency factor (more recent = lower factor = higher priority)
        days_old = (datetime.now(UTC) - end_date).days
        recency_factor = 1.0 + (days_old / self.config.recency_decay_factor)

        # Size factor (larger gaps get priority boost)
        size_factor = 1.0
        if gap_size >= self.config.large_gap_threshold:
            size_factor = self.config.large_gap_boost

        # Critical period boost
        if days_old <= self.config.critical_period_days:
            recency_factor *= 0.5  # Double the priority for critical period

        # Calculate final priority
        priority = base_priority * layer_factor * recency_factor * size_factor

        return max(1, int(priority))

    def _calculate_recency_focused_priority(
        self,
        data_type: DataType,
        layer: int,
        start_date: datetime,
        end_date: datetime,
        gap_size: int,
    ) -> int:
        """
        Priority calculation heavily weighted towards recent data.
        """
        # Base priority from data type
        base_priority = self.config.data_type_priorities.get(data_type, 10)

        # Strong recency factor
        days_old = (datetime.now(UTC) - end_date).days

        # Exponential decay for older data
        recency_factor = 1.0 + (days_old**1.5 / self.config.recency_decay_factor)

        # Minimal layer adjustment
        layer_mult = self.config.layer_multipliers.get(layer, 0.5)
        layer_factor = 1.0 / layer_mult if layer_mult > 0 else 2.0

        # Calculate priority
        priority = base_priority * recency_factor * (layer_factor**0.5)

        return max(1, int(priority))

    def _calculate_size_focused_priority(
        self,
        data_type: DataType,
        layer: int,
        start_date: datetime,
        end_date: datetime,
        gap_size: int,
    ) -> int:
        """
        Priority calculation weighted towards larger gaps.
        """
        # Base priority from data type
        base_priority = self.config.data_type_priorities.get(data_type, 10)

        # Strong size factor
        size_factor = max(0.1, 1.0 - (gap_size / 1000.0))

        # Moderate recency factor
        days_old = (datetime.now(UTC) - end_date).days
        recency_factor = 1.0 + (days_old / (self.config.recency_decay_factor * 2))

        # Layer adjustment
        layer_mult = self.config.layer_multipliers.get(layer, 0.5)
        layer_factor = 1.0 / layer_mult if layer_mult > 0 else 2.0

        # Calculate priority
        priority = base_priority * size_factor * recency_factor * layer_factor

        return max(1, int(priority))

    def _calculate_layer_focused_priority(
        self,
        data_type: DataType,
        layer: int,
        start_date: datetime,
        end_date: datetime,
        gap_size: int,
    ) -> int:
        """
        Priority calculation heavily weighted by symbol layer.
        """
        # Base priority from data type
        base_priority = self.config.data_type_priorities.get(data_type, 10)

        # Strong layer factor
        if layer == 1:
            layer_factor = 1.0
        elif layer == 2:
            layer_factor = 3.0
        elif layer == 3:
            layer_factor = 6.0
        else:
            layer_factor = 10.0

        # Moderate recency factor
        days_old = (datetime.now(UTC) - end_date).days
        recency_factor = 1.0 + (days_old / (self.config.recency_decay_factor * 1.5))

        # Calculate priority
        priority = base_priority * layer_factor * recency_factor

        return max(1, int(priority))

    def compare_priorities(self, priority1: int, priority2: int) -> int:
        """
        Compare two priority values.

        Args:
            priority1: First priority value
            priority2: Second priority value

        Returns:
            -1 if priority1 is higher, 1 if priority2 is higher, 0 if equal
        """
        if priority1 < priority2:
            return -1
        elif priority1 > priority2:
            return 1
        else:
            return 0

    def is_high_priority(self, priority: int) -> bool:
        """
        Check if a priority value indicates high priority.

        Args:
            priority: Priority value to check

        Returns:
            True if high priority (value <= 10)
        """
        return priority <= 10

    def is_critical_priority(self, priority: int) -> bool:
        """
        Check if a priority value indicates critical priority.

        Args:
            priority: Priority value to check

        Returns:
            True if critical priority (value <= 3)
        """
        return priority <= 3

    def get_priority_label(self, priority: int) -> str:
        """
        Get a human-readable label for a priority value.

        Args:
            priority: Priority value

        Returns:
            Priority label string
        """
        if priority <= 3:
            return "CRITICAL"
        elif priority <= 10:
            return "HIGH"
        elif priority <= 25:
            return "MEDIUM"
        elif priority <= 50:
            return "LOW"
        else:
            return "MINIMAL"

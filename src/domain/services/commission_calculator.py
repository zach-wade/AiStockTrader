"""
Commission Calculator - Domain service for calculating trading commissions.

This module provides commission calculation services for trading operations.
It supports multiple commission structures including per-share, percentage-based,
fixed, and tiered pricing models commonly used by brokers.

The commission calculator follows the Strategy pattern, allowing different
calculation methods to be used interchangeably based on the broker's fee structure.
This design enables easy addition of new commission models without modifying
existing code.

Key Components:
    - CommissionType: Enumeration of supported commission structures
    - CommissionSchedule: Configuration for commission calculation
    - ICommissionCalculator: Abstract interface for calculation strategies
    - Concrete calculators: Per-share and percentage implementations
    - CommissionCalculatorFactory: Creates appropriate calculator instances

Supported Commission Types:
    - PER_SHARE: Fixed rate per share traded
    - PERCENTAGE: Percentage of trade value
    - FIXED: Flat fee per trade (future implementation)
    - TIERED: Volume-based pricing tiers (future implementation)

Design Patterns:
    - Strategy Pattern: Different commission calculation algorithms
    - Factory Pattern: Creates calculators based on commission type
    - Dependency Injection: Calculators injected into services that need them

Example:
    >>> from decimal import Decimal
    >>> from domain.services.commission_calculator import (
    ...     CommissionSchedule, CommissionType, CommissionCalculatorFactory
    ... )
    >>> from domain.value_objects import Quantity, Money
    >>>
    >>> # Create a per-share commission schedule
    >>> schedule = CommissionSchedule(
    ...     commission_type=CommissionType.PER_SHARE,
    ...     rate=Decimal("0.005"),  # $0.005 per share
    ...     minimum=Decimal("1.00"),  # $1 minimum
    ...     maximum=Decimal("5.00")   # $5 maximum
    ... )
    >>>
    >>> calculator = CommissionCalculatorFactory.create(schedule)
    >>> commission = calculator.calculate(Quantity(100))
    >>> print(f"Commission: ${commission.amount}")

Note:
    Commission calculations are crucial for accurate P&L reporting.
    Always ensure commission is properly tracked and allocated to positions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from ..value_objects.money import Money
from ..value_objects.quantity import Quantity


class CommissionType(Enum):
    """Types of commission structures.

    Enumeration of supported commission calculation methods used by brokers.

    Values:
        PER_SHARE: Fixed amount per share traded (e.g., $0.01/share)
        PERCENTAGE: Percentage of total trade value (e.g., 0.1%)
        FIXED: Flat fee per trade regardless of size (future)
        TIERED: Volume-based pricing with different rates per tier (future)
    """

    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    TIERED = "tiered"


@dataclass
class CommissionSchedule:
    """Configuration for commission calculation.

    Defines the parameters for calculating commissions including the type,
    rate, and any minimum/maximum constraints.

    Attributes:
        commission_type: The type of commission structure to use.
        rate: The commission rate. Interpretation depends on type:
            - PER_SHARE: Dollar amount per share (e.g., 0.01 for $0.01/share)
            - PERCENTAGE: Percentage as decimal (e.g., 0.001 for 0.1%)
        minimum: Minimum commission amount (default $0).
        maximum: Maximum commission cap (optional).

    Example:
        >>> # $0.005 per share, $1 minimum, $10 maximum
        >>> schedule = CommissionSchedule(
        ...     commission_type=CommissionType.PER_SHARE,
        ...     rate=Decimal("0.005"),
        ...     minimum=Decimal("1.00"),
        ...     maximum=Decimal("10.00")
        ... )
    """

    commission_type: CommissionType
    rate: Decimal  # Per-share rate or percentage
    minimum: Decimal = Decimal("0")
    maximum: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate commission schedule.

        Ensures commission parameters are valid and consistent.

        Raises:
            ValueError: If rate is negative.
            ValueError: If minimum is negative.
            ValueError: If maximum is less than minimum.
        """
        if self.rate < 0:
            raise ValueError("Commission rate cannot be negative")
        if self.minimum < 0:
            raise ValueError("Minimum commission cannot be negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("Maximum commission must be greater than minimum")


class ICommissionCalculator(ABC):
    """Interface for commission calculation strategies.

    Abstract base class defining the interface for all commission calculators.
    Concrete implementations provide specific calculation algorithms based on
    the commission type.

    This interface follows the Strategy pattern, allowing different commission
    calculation methods to be used interchangeably.
    """

    @abstractmethod
    def calculate(self, quantity: Quantity, price: Money | None = None) -> Money:
        """Calculate commission for a trade.

        Args:
            quantity: Number of shares being traded.
            price: Total trade value (required for percentage-based commissions).

        Returns:
            Money: Calculated commission amount.

        Raises:
            ValueError: If required parameters are missing for the calculation type.
        """
        pass


class PerShareCommissionCalculator(ICommissionCalculator):
    """Calculate commission based on number of shares.

    Implements per-share commission pricing where traders pay a fixed amount
    for each share traded. This is common for retail brokers and active traders.

    Attributes:
        schedule: Commission schedule with rate and limits.

    Example:
        >>> schedule = CommissionSchedule(
        ...     commission_type=CommissionType.PER_SHARE,
        ...     rate=Decimal("0.01"),  # $0.01 per share
        ...     minimum=Decimal("1.00")
        ... )
        >>> calculator = PerShareCommissionCalculator(schedule)
        >>> commission = calculator.calculate(Quantity(50))
        >>> # Commission = max($1.00, 50 * $0.01) = $1.00 (minimum applies)
    """

    def __init__(self, schedule: CommissionSchedule) -> None:
        """Initialize with commission schedule.

        Args:
            schedule: Commission configuration with per-share rate.
        """
        self.schedule = schedule

    def calculate(self, quantity: Quantity, price: Money | None = None) -> Money:
        """Calculate per-share commission.

        Computes commission as rate × quantity, subject to minimum and maximum limits.

        Args:
            quantity: Number of shares being traded (absolute value used).
            price: Not used for per-share calculations (optional).

        Returns:
            Money: Commission amount after applying min/max constraints.

        Formula:
            Commission = max(minimum, min(maximum, rate × |quantity|))

        Example:
            >>> calculator.calculate(Quantity(100))  # 100 shares
            >>> # If rate=$0.005, min=$1, max=$5:
            >>> # Raw: 100 * 0.005 = $0.50
            >>> # After minimum: max($1, $0.50) = $1.00
        """
        commission = abs(quantity.value) * self.schedule.rate

        # Apply minimum
        commission = max(commission, self.schedule.minimum)

        # Apply maximum if set
        if self.schedule.maximum is not None:
            commission = min(commission, self.schedule.maximum)

        return Money(commission)


class PercentageCommissionCalculator(ICommissionCalculator):
    """Calculate commission as percentage of trade value.

    Implements percentage-based commission pricing where traders pay a percentage
    of the total trade value. Common for international markets and some brokers.

    Attributes:
        schedule: Commission schedule with percentage rate and limits.

    Example:
        >>> schedule = CommissionSchedule(
        ...     commission_type=CommissionType.PERCENTAGE,
        ...     rate=Decimal("0.1"),  # 0.1% of trade value
        ...     minimum=Decimal("5.00")
        ... )
        >>> calculator = PercentageCommissionCalculator(schedule)
        >>> trade_value = Money(Decimal("10000"))
        >>> commission = calculator.calculate(Quantity(100), trade_value)
        >>> # Commission = 0.1% of $10,000 = $10.00
    """

    def __init__(self, schedule: CommissionSchedule) -> None:
        """Initialize with commission schedule.

        Args:
            schedule: Commission configuration with percentage rate.
        """
        self.schedule = schedule

    def calculate(self, quantity: Quantity, price: Money | None = None) -> Money:
        """Calculate percentage-based commission.

        Computes commission as a percentage of total trade value,
        subject to minimum and maximum limits.

        Args:
            quantity: Number of shares being traded.
            price: Total trade value or price per share. Required for this
                calculation type.

        Returns:
            Money: Commission amount after applying min/max constraints.

        Raises:
            ValueError: If price is not provided.

        Formula:
            Trade Value = |quantity| × price
            Commission = max(minimum, min(maximum, trade_value × rate / 100))

        Example:
            >>> # 100 shares at $50/share, 0.1% rate
            >>> calculator.calculate(Quantity(100), Money(Decimal("50")))
            >>> # Trade value: 100 * $50 = $5,000
            >>> # Commission: $5,000 * 0.1% = $5.00

        Note:
            The rate is divided by 100 to convert percentage to decimal.
        """
        if price is None:
            raise ValueError("Price required for percentage commission calculation")

        trade_value = abs(quantity.value) * price.amount
        commission = trade_value * self.schedule.rate / Decimal("100")

        # Apply minimum
        commission = max(commission, self.schedule.minimum)

        # Apply maximum if set
        if self.schedule.maximum is not None:
            commission = min(commission, self.schedule.maximum)

        return Money(commission)


class CommissionCalculatorFactory:
    """Factory for creating commission calculators.

    Creates appropriate commission calculator instances based on the commission
    type specified in the schedule. This factory encapsulates the creation logic
    and makes it easy to add new commission types.

    Design Pattern:
        Factory Method - Centralizes object creation and enables easy extension
        for new commission types without modifying client code.
    """

    @staticmethod
    def create(schedule: CommissionSchedule) -> ICommissionCalculator:
        """Create appropriate calculator based on commission type.

        Factory method that returns the correct calculator implementation
        based on the commission type in the schedule.

        Args:
            schedule: Commission schedule specifying type and parameters.

        Returns:
            ICommissionCalculator: Concrete calculator for the specified type.

        Raises:
            ValueError: If commission type is not supported.

        Example:
            >>> schedule = CommissionSchedule(
            ...     commission_type=CommissionType.PER_SHARE,
            ...     rate=Decimal("0.01")
            ... )
            >>> calculator = CommissionCalculatorFactory.create(schedule)
            >>> # Returns PerShareCommissionCalculator instance

        Note:
            To add a new commission type:
            1. Add the type to CommissionType enum
            2. Create a new calculator class implementing ICommissionCalculator
            3. Add a case to this factory method
        """
        if schedule.commission_type == CommissionType.PER_SHARE:
            return PerShareCommissionCalculator(schedule)
        elif schedule.commission_type == CommissionType.PERCENTAGE:
            return PercentageCommissionCalculator(schedule)
        else:
            raise ValueError(f"Unsupported commission type: {schedule.commission_type}")


# Default commission schedules
DEFAULT_RETAIL_SCHEDULE = CommissionSchedule(
    commission_type=CommissionType.PER_SHARE,
    rate=Decimal("0.01"),
    minimum=Decimal("1.00"),
    maximum=Decimal("5.00"),
)
# Default commission schedule for retail traders.
# Typical retail broker commission structure:
# - $0.01 per share
# - $1.00 minimum per trade
# - $5.00 maximum per trade
# This structure is common for discount brokers catering to individual traders.

DEFAULT_INSTITUTIONAL_SCHEDULE = CommissionSchedule(
    commission_type=CommissionType.PER_SHARE, rate=Decimal("0.005"), minimum=Decimal("0.35")
)
"""Default commission schedule for institutional traders.

Typical institutional/professional commission structure:
- $0.005 per share
- $0.35 minimum per trade
- No maximum (uncapped)

This structure is common for institutional brokers and professional trading firms
with higher volumes and better rates.
"""

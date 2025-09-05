"""Financial Safety Service - Provides safe mathematical operations for financial calculations."""

from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, Overflow
from typing import TypeVar

T = TypeVar("T")


class FinancialSafety:
    """Service for safe financial calculations with overflow and division protection."""

    # Maximum values for financial calculations
    MAX_DECIMAL = Decimal("999999999999999999.99")  # 18 digits before decimal, 2 after
    MIN_DECIMAL = -MAX_DECIMAL
    ZERO_THRESHOLD = Decimal("0.0000000001")  # Values smaller than this are treated as zero

    @classmethod
    def safe_divide(
        cls, numerator: Decimal, denominator: Decimal, default: Decimal | None = None
    ) -> Decimal | None:
        """
        Safely divide two decimals with zero division protection.

        Args:
            numerator: The dividend
            denominator: The divisor
            default: Value to return if division by zero (None if not specified)

        Returns:
            Result of division or default value if division by zero
        """
        try:
            # Check for zero denominator
            if abs(denominator) < cls.ZERO_THRESHOLD:
                return default

            result = numerator / denominator

            # Check for overflow
            if abs(result) > cls.MAX_DECIMAL:
                return cls.MAX_DECIMAL if result > 0 else cls.MIN_DECIMAL

            return result

        except (DivisionByZero, InvalidOperation, Overflow):
            return default

    @classmethod
    def safe_multiply(
        cls, value1: Decimal, value2: Decimal, default: Decimal | None = None
    ) -> Decimal:
        """
        Safely multiply two decimals with overflow protection.

        Args:
            value1: First value
            value2: Second value
            default: Value to return on overflow (MAX_DECIMAL if not specified)

        Returns:
            Result of multiplication or capped value on overflow
        """
        try:
            result = value1 * value2

            # Check for overflow
            if abs(result) > cls.MAX_DECIMAL:
                if default is not None:
                    return default
                return cls.MAX_DECIMAL if result > 0 else cls.MIN_DECIMAL

            return result

        except (InvalidOperation, Overflow):
            if default is not None:
                return default
            # Determine sign for capped value
            if (value1 > 0 and value2 > 0) or (value1 < 0 and value2 < 0):
                return cls.MAX_DECIMAL
            else:
                return cls.MIN_DECIMAL

    @classmethod
    def safe_add(cls, value1: Decimal, value2: Decimal, default: Decimal | None = None) -> Decimal:
        """
        Safely add two decimals with overflow protection.

        Args:
            value1: First value
            value2: Second value
            default: Value to return on overflow (MAX_DECIMAL if not specified)

        Returns:
            Result of addition or capped value on overflow
        """
        try:
            result = value1 + value2

            # Check for overflow
            if result > cls.MAX_DECIMAL:
                return default if default is not None else cls.MAX_DECIMAL
            elif result < cls.MIN_DECIMAL:
                return default if default is not None else cls.MIN_DECIMAL

            return result

        except (InvalidOperation, Overflow):
            if default is not None:
                return default
            # Determine direction for capped value
            if value1 > 0 and value2 > 0:
                return cls.MAX_DECIMAL
            else:
                return cls.MIN_DECIMAL

    @classmethod
    def safe_subtract(
        cls, value1: Decimal, value2: Decimal, default: Decimal | None = None
    ) -> Decimal:
        """
        Safely subtract two decimals with overflow protection.

        Args:
            value1: Value to subtract from
            value2: Value to subtract
            default: Value to return on overflow (MAX_DECIMAL if not specified)

        Returns:
            Result of subtraction or capped value on overflow
        """
        try:
            result = value1 - value2

            # Check for overflow
            if result > cls.MAX_DECIMAL:
                return default if default is not None else cls.MAX_DECIMAL
            elif result < cls.MIN_DECIMAL:
                return default if default is not None else cls.MIN_DECIMAL

            return result

        except (InvalidOperation, Overflow):
            if default is not None:
                return default
            # Determine direction for capped value
            if value1 > 0 and value2 < 0:
                return cls.MAX_DECIMAL
            else:
                return cls.MIN_DECIMAL

    @classmethod
    def safe_percentage(
        cls, part: Decimal, whole: Decimal, default: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Safely calculate percentage (part/whole * 100).

        Args:
            part: The partial value
            whole: The total value
            default: Value to return if whole is zero

        Returns:
            Percentage value or default if whole is zero
        """
        ratio = cls.safe_divide(part, whole, default=Decimal("0"))
        if ratio is None:
            return default

        return cls.safe_multiply(ratio, Decimal("100"), default=default)

    @classmethod
    def safe_ratio(
        cls,
        numerator: Decimal,
        denominator: Decimal,
        max_ratio: Decimal = Decimal("999.99"),
        default: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Safely calculate a ratio with maximum capping.

        Args:
            numerator: The dividend
            denominator: The divisor
            max_ratio: Maximum ratio to return (caps extreme values)
            default: Value to return if denominator is zero

        Returns:
            Ratio value capped at max_ratio or default
        """
        result = cls.safe_divide(numerator, denominator, default=default)

        if result is None:
            return default

        # Cap at maximum ratio
        if abs(result) > max_ratio:
            return max_ratio if result > 0 else -max_ratio

        return result

    @classmethod
    def safe_round(cls, value: Decimal, decimal_places: int = 2) -> Decimal:
        """
        Safely round a decimal value.

        Args:
            value: Value to round
            decimal_places: Number of decimal places

        Returns:
            Rounded value
        """
        try:
            quantizer = Decimal(10) ** -decimal_places
            return value.quantize(quantizer, rounding=ROUND_HALF_UP)
        except (InvalidOperation, Overflow):
            # If rounding fails, return original value
            return value

    @classmethod
    def is_near_zero(cls, value: Decimal) -> bool:
        """
        Check if a value is effectively zero (within threshold).

        Args:
            value: Value to check

        Returns:
            True if value is near zero
        """
        return abs(value) < cls.ZERO_THRESHOLD

    @classmethod
    def validate_financial_value(
        cls,
        value: Decimal,
        min_value: Decimal | None = None,
        max_value: Decimal | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate a financial value is within acceptable bounds.

        Args:
            value: Value to validate
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN or Inf
        if not value.is_finite():
            return False, "Value is not finite (NaN or Inf)"

        # Check against absolute limits
        if abs(value) > cls.MAX_DECIMAL:
            return False, f"Value exceeds maximum limit of {cls.MAX_DECIMAL}"

        # Check custom bounds
        if min_value is not None and value < min_value:
            return False, f"Value {value} is below minimum {min_value}"

        if max_value is not None and value > max_value:
            return False, f"Value {value} is above maximum {max_value}"

        return True, None

    @classmethod
    def clamp_value(
        cls,
        value: Decimal,
        min_value: Decimal | None = None,
        max_value: Decimal | None = None,
    ) -> Decimal:
        """
        Clamp a value within specified bounds.

        Args:
            value: Value to clamp
            min_value: Minimum value (uses MIN_DECIMAL if not specified)
            max_value: Maximum value (uses MAX_DECIMAL if not specified)

        Returns:
            Clamped value
        """
        min_val = min_value if min_value is not None else cls.MIN_DECIMAL
        max_val = max_value if max_value is not None else cls.MAX_DECIMAL

        if value < min_val:
            return min_val
        elif value > max_val:
            return max_val
        else:
            return value

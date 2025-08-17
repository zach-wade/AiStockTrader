"""Money value object for representing monetary values with currency."""

# Standard library imports
from decimal import ROUND_HALF_UP, Decimal
from typing import Self

from ..constants import CURRENCY_CODE_LENGTH


class Money:
    """Immutable value object representing money with currency and precision."""

    def __init__(self, amount: Decimal | float | int | str, currency: str = "USD") -> None:
        """Initialize Money with amount and currency.

        Args:
            amount: The monetary amount (converted to Decimal)
            currency: ISO 4217 currency code (default: USD)

        Raises:
            ValueError: If currency is invalid
        """
        if not isinstance(amount, Decimal):
            amount = Decimal(str(amount))

        self._amount = amount
        self._currency = currency.upper()

        if len(self._currency) != CURRENCY_CODE_LENGTH:
            raise ValueError(f"Invalid currency code: {currency}")

    @property
    def amount(self) -> Decimal:
        """Get the decimal amount."""
        return self._amount

    @property
    def currency(self) -> str:
        """Get the currency code."""
        return self._currency

    def add(self, other: Self) -> Self:
        """Add two money values.

        Args:
            other: Another Money instance

        Returns:
            New Money instance with sum

        Raises:
            ValueError: If currencies don't match
        """
        if not isinstance(other, Money):
            raise TypeError(f"Cannot add Money and {type(other)}")
        if self._currency != other._currency:
            raise ValueError(f"Cannot add {self._currency} and {other._currency}")

        return Money(self._amount + other._amount, self._currency)

    def subtract(self, other: Self) -> Self:
        """Subtract another money value.

        Args:
            other: Another Money instance

        Returns:
            New Money instance with difference

        Raises:
            ValueError: If currencies don't match
        """
        if not isinstance(other, Money):
            raise TypeError(f"Cannot subtract {type(other)} from Money")
        if self._currency != other._currency:
            raise ValueError(f"Cannot subtract {other._currency} from {self._currency}")

        return Money(self._amount - other._amount, self._currency)

    def multiply(self, factor: Decimal | float | int) -> Self:
        """Multiply money by a factor.

        Args:
            factor: Multiplication factor

        Returns:
            New Money instance with product
        """
        if not isinstance(factor, Decimal):
            factor = Decimal(str(factor))

        return Money(self._amount * factor, self._currency)

    def divide(self, divisor: Decimal | float | int) -> Self:
        """Divide money by a divisor.

        Args:
            divisor: Division factor

        Returns:
            New Money instance with quotient

        Raises:
            ValueError: If divisor is zero
        """
        if not isinstance(divisor, Decimal):
            divisor = Decimal(str(divisor))

        if divisor == 0:
            raise ValueError("Cannot divide by zero")

        return Money(self._amount / divisor, self._currency)

    def round(self, decimal_places: int = 2) -> Self:
        """Round to specified decimal places.

        Args:
            decimal_places: Number of decimal places

        Returns:
            New Money instance with rounded amount
        """
        quantizer = Decimal(10) ** -decimal_places
        rounded = self._amount.quantize(quantizer, rounding=ROUND_HALF_UP)
        return Money(rounded, self._currency)

    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self._amount > 0

    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self._amount < 0

    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self._amount == 0

    def format(self, include_currency: bool = True, decimal_places: int = 2) -> str:
        """Format money for display.

        Args:
            include_currency: Whether to include currency symbol
            decimal_places: Number of decimal places to show

        Returns:
            Formatted string representation
        """
        # Round for display
        display_amount = self.round(decimal_places)._amount

        # Format with thousands separator
        formatted = f"{display_amount:,.{decimal_places}f}"

        if include_currency:
            if self._currency == "USD":
                return f"${formatted}"
            else:
                return f"{formatted} {self._currency}"

        return formatted

    def __eq__(self, other: object) -> bool:
        """Check equality with another Money instance."""
        if not isinstance(other, Money):
            return False
        return self._amount == other._amount and self._currency == other._currency

    def __lt__(self, other: Self) -> bool:
        """Check if less than another Money instance."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money and {type(other)}")
        if self._currency != other._currency:
            raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
        return self._amount < other._amount

    def __le__(self, other: Self) -> bool:
        """Check if less than or equal to another Money instance."""
        return self == other or self < other

    def __gt__(self, other: Self) -> bool:
        """Check if greater than another Money instance."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money and {type(other)}")
        if self._currency != other._currency:
            raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
        return self._amount > other._amount

    def __ge__(self, other: Self) -> bool:
        """Check if greater than or equal to another Money instance."""
        return self == other or self > other

    def __neg__(self) -> Self:
        """Negate the money amount."""
        return Money(-self._amount, self._currency)

    def __abs__(self) -> Self:
        """Get absolute value of money."""
        return Money(abs(self._amount), self._currency)

    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        return hash((self._amount, self._currency))

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return f"Money({self._amount}, '{self._currency}')"

    def __str__(self) -> str:
        """Get string representation for display."""
        return self.format()

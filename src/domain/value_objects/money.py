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

        return type(self)(self._amount + other._amount, self._currency)

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

        return type(self)(self._amount - other._amount, self._currency)

    def multiply(self, factor: Decimal | float | int) -> Self:
        """Multiply money by a factor.

        Args:
            factor: Multiplication factor

        Returns:
            New Money instance with product
        """
        if not isinstance(factor, Decimal):
            factor = Decimal(str(factor))

        return type(self)(self._amount * factor, self._currency)

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

        return type(self)(self._amount / divisor, self._currency)

    def round(self, decimal_places: int = 2) -> Self:
        """Round to specified decimal places.

        Args:
            decimal_places: Number of decimal places

        Returns:
            New Money instance with rounded amount
        """
        quantizer = Decimal(10) ** -decimal_places
        rounded = self._amount.quantize(quantizer, rounding=ROUND_HALF_UP)
        return type(self)(rounded, self._currency)

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

    def __lt__(self, other: Self | Decimal | int | float) -> bool:
        """Check if less than another Money instance or numeric value."""
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
            return self._amount < other._amount
        if isinstance(other, (Decimal, int, float)):
            return self._amount < Decimal(str(other))
        raise TypeError(f"Cannot compare Money and {type(other)}")

    def __le__(self, other: Self | Decimal | int | float) -> bool:
        """Check if less than or equal to another Money instance or numeric value."""
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
            return self._amount <= other._amount
        if isinstance(other, (Decimal, int, float)):
            return self._amount <= Decimal(str(other))
        raise TypeError(f"Cannot compare Money and {type(other)}")

    def __gt__(self, other: Self | Decimal | int | float) -> bool:
        """Check if greater than another Money instance or numeric value."""
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
            return self._amount > other._amount
        if isinstance(other, (Decimal, int, float)):
            return self._amount > Decimal(str(other))
        raise TypeError(f"Cannot compare Money and {type(other)}")

    def __ge__(self, other: Self | Decimal | int | float) -> bool:
        """Check if greater than or equal to another Money instance or numeric value."""
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot compare {self._currency} and {other._currency}")
            return self._amount >= other._amount
        if isinstance(other, (Decimal, int, float)):
            return self._amount >= Decimal(str(other))
        raise TypeError(f"Cannot compare Money and {type(other)}")

    def __neg__(self) -> Self:
        """Negate the money amount."""
        return type(self)(-self._amount, self._currency)

    def __abs__(self) -> Self:
        """Get absolute value of money."""
        return type(self)(abs(self._amount), self._currency)

    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        return hash((self._amount, self._currency))

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return f"Money({self._amount}, '{self._currency}')"

    def __str__(self) -> str:
        """Get string representation for display."""
        return self.format()

    def __add__(self, other: Self | Decimal | float | int) -> Self:
        """Add Money to another Money or numeric value.

        Args:
            other: Money instance or numeric value to add

        Returns:
            New Money instance with sum

        Raises:
            ValueError: If currencies don't match (for Money)
        """
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot add {self._currency} and {other._currency}")
            return type(self)(self._amount + other._amount, self._currency)
        elif isinstance(other, (Decimal, float, int)):
            # Allow adding numeric values directly (treated as same currency)
            if not isinstance(other, Decimal):
                other = Decimal(str(other))
            return type(self)(self._amount + other, self._currency)
        else:
            raise TypeError(f"Cannot add Money and {type(other)}")

    def __radd__(self, other: Decimal | float | int) -> Self:
        """Right-side addition (when Money is on the right).

        Args:
            other: Numeric value to add

        Returns:
            New Money instance with sum
        """
        if isinstance(other, (Decimal, float, int)):
            return self.__add__(other)
        raise TypeError(f"Cannot add {type(other)} and Money")

    def __sub__(self, other: Self | Decimal | float | int) -> Self:
        """Subtract Money or numeric value from this Money.

        Args:
            other: Money instance or numeric value to subtract

        Returns:
            New Money instance with difference

        Raises:
            ValueError: If currencies don't match (for Money)
        """
        if isinstance(other, Money):
            if self._currency != other._currency:
                raise ValueError(f"Cannot subtract {other._currency} from {self._currency}")
            return type(self)(self._amount - other._amount, self._currency)
        elif isinstance(other, (Decimal, float, int)):
            if not isinstance(other, Decimal):
                other = Decimal(str(other))
            return type(self)(self._amount - other, self._currency)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Money")

    def __rsub__(self, other: Decimal | float | int) -> Self:
        """Right-side subtraction (when Money is on the right).

        Args:
            other: Numeric value to subtract from

        Returns:
            New Money instance with difference
        """
        if isinstance(other, (Decimal, float, int)):
            if not isinstance(other, Decimal):
                other = Decimal(str(other))
            return type(self)(other - self._amount, self._currency)
        raise TypeError(f"Cannot subtract Money from {type(other)}")

    def __iadd__(self, other: Self | Decimal | float | int) -> Self:
        """In-place addition (+=).

        Note: Returns new instance since Money is immutable.

        Args:
            other: Money instance or numeric value to add

        Returns:
            New Money instance with sum
        """
        return self.__add__(other)

    def __isub__(self, other: Self | Decimal | float | int) -> Self:
        """In-place subtraction (-=).

        Note: Returns new instance since Money is immutable.

        Args:
            other: Money instance or numeric value to subtract

        Returns:
            New Money instance with difference
        """
        return self.__sub__(other)

    def __truediv__(self, other: Self | Decimal | float | int) -> Self | Decimal:
        """Division operator (/).

        Args:
            other: Money instance or numeric value to divide by

        Returns:
            New Money instance if dividing by a scalar,
            or Decimal if dividing Money by Money (ratio)

        Raises:
            ValueError: If divisor is zero
        """
        if isinstance(other, Money):
            # Money / Money = ratio (Decimal)
            if self._currency != other._currency:
                raise ValueError(f"Cannot divide {self._currency} by {other._currency}")
            if other._amount == 0:
                raise ValueError("Cannot divide by zero")
            return self._amount / other._amount
        elif isinstance(other, (Decimal, float, int)):
            # Money / number = Money
            if not isinstance(other, Decimal):
                other = Decimal(str(other))
            if other == 0:
                raise ValueError("Cannot divide by zero")
            return type(self)(self._amount / other, self._currency)
        else:
            raise TypeError(f"Cannot divide Money by {type(other)}")

    def __rtruediv__(self, other: Decimal | float | int) -> Decimal:
        """Right-side division (when Money is on the right).

        Args:
            other: Numeric value to be divided

        Returns:
            Decimal result of division

        Raises:
            ValueError: If Money amount is zero
        """
        if self._amount == 0:
            raise ValueError("Cannot divide by zero")
        if isinstance(other, (Decimal, float, int)):
            if not isinstance(other, Decimal):
                other = Decimal(str(other))
            return other / self._amount
        raise TypeError(f"Cannot divide {type(other)} by Money")

    def __mul__(self, other: Decimal | float | int) -> Self:
        """Multiplication operator (*).

        Args:
            other: Numeric value to multiply by

        Returns:
            New Money instance with product
        """
        if not isinstance(other, (Decimal, float, int)):
            raise TypeError(f"Cannot multiply Money by {type(other)}")
        if not isinstance(other, Decimal):
            other = Decimal(str(other))
        return type(self)(self._amount * other, self._currency)

    def __rmul__(self, other: Decimal | float | int) -> Self:
        """Right-side multiplication (when Money is on the right).

        Args:
            other: Numeric value to multiply

        Returns:
            New Money instance with product
        """
        return self.__mul__(other)

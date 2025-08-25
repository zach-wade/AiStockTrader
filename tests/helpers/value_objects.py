"""Value object test helpers and fixtures."""

from decimal import Decimal

from src.domain.value_objects import Money, Price, Quantity, Symbol, Timezone


def test_symbol(ticker: str = "AAPL") -> Symbol:
    """Create a test Symbol instance."""
    return Symbol(ticker)


def test_quantity(amount: str | int | Decimal = "100") -> Quantity:
    """Create a test Quantity instance."""
    return Quantity(Decimal(str(amount)))


def test_price(amount: str | float | Decimal = "100.00") -> Price:
    """Create a test Price instance."""
    return Price(Decimal(str(amount)))


def test_money(amount: str | float | Decimal = "1000.00", currency: str = "USD") -> Money:
    """Create a test Money instance."""
    return Money(Decimal(str(amount)), currency)


def test_timezone(tz: str = "America/New_York") -> Timezone:
    """Create a test Timezone instance."""
    return Timezone(tz)


# Common test values
DEFAULT_SYMBOL = test_symbol("AAPL")
DEFAULT_QUANTITY = test_quantity(100)
DEFAULT_PRICE = test_price("150.00")
DEFAULT_MONEY = test_money("15000.00")
DEFAULT_TIMEZONE = test_timezone("America/New_York")

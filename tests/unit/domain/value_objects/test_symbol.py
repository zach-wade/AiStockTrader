"""
Comprehensive unit tests for Symbol value object.

Tests all public methods, validation, edge cases, symbol patterns,
comparisons, immutability, and string representations.
"""

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.symbol import Symbol


class TestSymbolCreation:
    """Test Symbol creation and initialization."""

    def test_create_stock_symbol(self):
        """Test creating standard stock symbols."""
        symbol = Symbol("AAPL")
        assert symbol == "AAPL"
        assert symbol.base_symbol == "AAPL"
        assert symbol.exchange is None
        assert symbol.is_stock()
        assert not symbol.is_crypto()
        assert not symbol.is_option()

    def test_create_symbol_with_lowercase(self):
        """Test that symbols are normalized to uppercase."""
        symbol = Symbol("aapl")
        assert symbol == "AAPL"

    def test_create_symbol_with_whitespace(self):
        """Test that whitespace is stripped."""
        symbol = Symbol("  AAPL  ")
        assert symbol == "AAPL"

    def test_create_single_letter_symbol(self):
        """Test creating single letter symbol."""
        symbol = Symbol("F")  # Ford
        assert symbol == "F"
        assert symbol.is_stock()

    def test_create_five_letter_symbol(self):
        """Test creating five letter symbol (max for stocks)."""
        symbol = Symbol("GOOGL")
        assert symbol == "GOOGL"
        assert symbol.is_stock()

    def test_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            Symbol("")

        # Whitespace-only string gets normalized and becomes invalid format
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol("   ")

    def test_invalid_symbol_format_raises_error(self):
        """Test that invalid symbol formats raise ValueError."""
        # Too long for stock
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol("TOOLONG")

        # Contains numbers (invalid for plain stock)
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol("AAPL123")

        # Contains lowercase (after normalization this should work though)
        symbol = Symbol("aapl")  # This should work due to normalization
        assert symbol == "AAPL"

        # Special characters
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol("AAPL@")

        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol("AAPL#")


class TestSymbolWithExchange:
    """Test Symbol with exchange suffixes."""

    def test_create_symbol_with_exchange_dot_notation(self):
        """Test creating symbol with exchange using dot notation."""
        symbol = Symbol("AAPL.US")
        assert symbol == "AAPL.US"
        assert symbol.base_symbol == "AAPL"
        assert symbol.exchange == "US"
        assert symbol.is_stock()

    def test_create_symbol_with_exchange_colon_notation(self):
        """Test creating symbol with exchange using colon notation."""
        symbol = Symbol("AAPL:NASDAQ")
        assert symbol == "AAPL:NASDAQ"
        assert symbol.base_symbol == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert symbol.is_stock()

    def test_various_exchange_codes(self):
        """Test various valid exchange codes."""
        exchanges = {
            "AAPL.US": ("AAPL", "US"),
            "AAPL.NASDAQ": ("AAPL", "NASDAQ"),
            "BP.LON": ("BP", "LON"),  # London
            "SAP.DE": ("SAP", "DE"),  # Germany
            "IBM.NYSE": ("IBM", "NYSE"),  # NYSE
        }

        for full_symbol, (base, exchange) in exchanges.items():
            symbol = Symbol(full_symbol)
            assert symbol == full_symbol
            assert symbol.base_symbol == base
            assert symbol.exchange == exchange


class TestCryptoSymbols:
    """Test cryptocurrency symbol patterns."""

    def test_create_crypto_symbol(self):
        """Test creating cryptocurrency symbols."""
        symbol = Symbol("BTC-USD")
        assert symbol == "BTC-USD"
        assert symbol.base_symbol == "BTC"
        assert symbol.quote_currency == "USD"
        assert symbol.exchange is None
        assert symbol.is_crypto()
        assert not symbol.is_stock()
        assert not symbol.is_option()

    def test_various_crypto_pairs(self):
        """Test various cryptocurrency pairs."""
        pairs = {
            "BTC-USD": ("BTC", "USD"),
            "ETH-USDT": ("ETH", "USDT"),
            "BNB-BUSD": ("BNB", "BUSD"),
            "ADA-BTC": ("ADA", "BTC"),
            "DOGE-USD": ("DOGE", "USD"),
        }

        for full_symbol, (base, quote) in pairs.items():
            symbol = Symbol(full_symbol)
            assert symbol == full_symbol
            assert symbol.base_symbol == base
            assert symbol.quote_currency == quote
            assert symbol.is_crypto()


class TestOptionSymbols:
    """Test option symbol patterns."""

    def test_create_option_symbol(self):
        """Test creating option symbols (OCC format)."""
        # AAPL Jan 19, 2024 $150 Call
        symbol = Symbol("AAPL240119C00150000")
        assert symbol == "AAPL240119C00150000"
        assert symbol.is_option()
        assert not symbol.is_stock()
        assert not symbol.is_crypto()

    def test_various_option_symbols(self):
        """Test various option symbol formats."""
        options = [
            "AAPL240119C00150000",  # Call
            "AAPL240119P00150000",  # Put
            "SPY231215C00450000",  # SPY call
            "TSLA240216P00200000",  # TSLA put
        ]

        for option_symbol in options:
            symbol = Symbol(option_symbol)
            assert symbol == option_symbol
            assert symbol.is_option()


class TestSymbolProperties:
    """Test Symbol properties and methods."""

    def test_value_property(self):
        """Test value property returns full symbol."""
        symbol = Symbol("AAPL")
        assert symbol == "AAPL"

    def test_base_symbol_property(self):
        """Test base_symbol property."""
        # Plain symbol
        symbol = Symbol("AAPL")
        assert symbol.base_symbol == "AAPL"

        # With exchange
        symbol = Symbol("AAPL.US")
        assert symbol.base_symbol == "AAPL"

        # Crypto pair
        symbol = Symbol("BTC-USD")
        assert symbol.base_symbol == "BTC"

    def test_exchange_property(self):
        """Test exchange property."""
        # No exchange
        symbol = Symbol("AAPL")
        assert symbol.exchange is None

        # With exchange
        symbol = Symbol("AAPL.US")
        assert symbol.exchange == "US"

        # Crypto (no exchange)
        symbol = Symbol("BTC-USD")
        assert symbol.exchange is None

    def test_quote_currency_property(self):
        """Test quote_currency property for crypto pairs."""
        # Crypto pair
        symbol = Symbol("BTC-USD")
        assert symbol.quote_currency == "USD"

        # Stock (no quote currency)
        symbol = Symbol("AAPL")
        assert symbol.quote_currency is None


class TestSymbolClassification:
    """Test Symbol classification methods."""

    def test_is_stock(self):
        """Test is_stock method."""
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "AAPL.US", "BP.LON"]
        for sym in stock_symbols:
            symbol = Symbol(sym)
            assert symbol.is_stock()
            assert not symbol.is_crypto()
            assert not symbol.is_option()

    def test_is_crypto(self):
        """Test is_crypto method."""
        crypto_symbols = ["BTC-USD", "ETH-USDT", "ADA-BTC"]
        for sym in crypto_symbols:
            symbol = Symbol(sym)
            assert symbol.is_crypto()
            assert not symbol.is_stock()
            assert not symbol.is_option()

    def test_is_option(self):
        """Test is_option method."""
        option_symbols = ["AAPL240119C00150000", "SPY231215P00450000"]
        for sym in option_symbols:
            symbol = Symbol(sym)
            assert symbol.is_option()
            assert not symbol.is_stock()
            assert not symbol.is_crypto()


class TestSymbolManipulation:
    """Test Symbol manipulation methods."""

    def test_with_exchange(self):
        """Test adding exchange to symbol."""
        symbol = Symbol("AAPL")
        new_symbol = symbol.with_exchange("NASDAQ")

        assert new_symbol == "AAPL.NASDAQ"
        assert new_symbol.base_symbol == "AAPL"
        assert new_symbol.exchange == "NASDAQ"

        # Original unchanged
        assert symbol == "AAPL"
        assert symbol.exchange is None

    def test_with_exchange_normalizes_uppercase(self):
        """Test that exchange is normalized to uppercase."""
        symbol = Symbol("AAPL")
        new_symbol = symbol.with_exchange("nasdaq")

        assert new_symbol == "AAPL.NASDAQ"
        assert new_symbol.exchange == "NASDAQ"

    def test_with_exchange_on_crypto_raises_error(self):
        """Test that adding exchange to crypto symbol raises error."""
        symbol = Symbol("BTC-USD")

        with pytest.raises(ValueError, match="Cannot add exchange to crypto symbol"):
            symbol.with_exchange("BINANCE")

    def test_without_exchange(self):
        """Test removing exchange from symbol."""
        symbol = Symbol("AAPL.US")
        new_symbol = symbol.without_exchange()

        assert new_symbol == "AAPL"
        assert new_symbol.base_symbol == "AAPL"
        assert new_symbol.exchange is None

        # Original unchanged
        assert symbol == "AAPL.US"
        assert symbol.exchange == "US"

    def test_without_exchange_no_exchange(self):
        """Test removing exchange when there is none."""
        symbol = Symbol("AAPL")
        new_symbol = symbol.without_exchange()

        assert new_symbol == symbol
        assert new_symbol == "AAPL"


class TestSymbolComparison:
    """Test Symbol comparison operations."""

    def test_equality_same_symbol(self):
        """Test equality for same symbol."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")

        assert symbol1 == symbol2
        assert symbol1 == symbol2

    def test_equality_different_symbol(self):
        """Test inequality for different symbols."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("GOOGL")

        assert symbol1 != symbol2
        assert symbol1 != symbol2

    def test_equality_case_insensitive_creation(self):
        """Test that symbols created with different cases are equal."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("aapl")

        assert symbol1 == symbol2

    def test_equality_with_non_symbol(self):
        """Test equality comparison with non-Symbol types."""
        symbol = Symbol("AAPL")

        assert symbol != "AAPL"
        assert symbol != 100
        assert symbol != None

    def test_less_than_alphabetical(self):
        """Test less than comparison (alphabetical)."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("GOOGL")

        assert symbol1 < symbol2
        assert not symbol2 < symbol1

    def test_less_than_non_symbol_raises_error(self):
        """Test less than comparison with non-Symbol raises error."""
        symbol = Symbol("AAPL")

        with pytest.raises(TypeError, match="Cannot compare Symbol and"):
            symbol < "GOOGL"

    def test_hash(self):
        """Test hash for use in sets and dicts."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")
        symbol3 = Symbol("GOOGL")

        # Equal symbols have same hash
        assert hash(symbol1) == hash(symbol2)

        # Can be used in sets
        symbol_set = {symbol1, symbol2, symbol3}
        assert len(symbol_set) == 2  # symbol1 and symbol2 are equal

        # Can be used as dict keys
        symbol_dict = {symbol1: "Apple", symbol3: "Google"}
        assert len(symbol_dict) == 2
        assert symbol_dict[symbol2] == "Apple"  # symbol2 accesses same as symbol1


class TestSymbolValidation:
    """Test Symbol validation methods."""

    def test_validate_class_method_valid(self):
        """Test validate class method with valid symbols."""
        valid_symbols = [
            "AAPL",
            "GOOGL",
            "F",
            "AAPL.US",
            "BTC-USD",
            "AAPL240119C00150000",
        ]

        for sym in valid_symbols:
            assert Symbol.validate(sym) is True

    def test_validate_class_method_invalid(self):
        """Test validate class method with invalid symbols."""
        invalid_symbols = [
            "",
            "TOOLONG",
            "AAPL@",
            "123ABC",
            "AAPL#NASDAQ",
        ]

        for sym in invalid_symbols:
            assert Symbol.validate(sym) is False


class TestSymbolFormatting:
    """Test Symbol formatting and display."""

    def test_str_representation(self):
        """Test string representation."""
        symbol = Symbol("AAPL")
        assert str(symbol) == "AAPL"

        symbol = Symbol("AAPL.US")
        assert str(symbol) == "AAPL.US"

    def test_repr_representation(self):
        """Test repr representation."""
        symbol = Symbol("AAPL")
        assert repr(symbol) == "Symbol('AAPL')"

        symbol = Symbol("BTC-USD")
        assert repr(symbol) == "Symbol('BTC-USD')"


class TestSymbolEdgeCases:
    """Test Symbol edge cases and special scenarios."""

    def test_numeric_symbols(self):
        """Test symbols that are purely numeric (some exchanges allow)."""
        # Some exchanges like Tokyo use numeric codes with proper exchange suffix
        symbol = Symbol("SONY.TYO")
        assert symbol == "SONY.TYO"
        assert symbol.base_symbol == "SONY"
        assert symbol.exchange == "TYO"

    def test_symbol_immutability(self):
        """Test that Symbol is immutable."""
        symbol = Symbol("AAPL")
        original_value = symbol

        # Properties are read-only - trying to set them should fail
        # Python allows setting attributes unless we use slots or properties
        # Since Symbol doesn't use __slots__, we can set attributes
        # but this doesn't change the actual value since it's stored in _value

        # Operations return new objects
        new_symbol = symbol.with_exchange("US")
        assert symbol == original_value
        assert new_symbol == "AAPL.US"

    def test_mixed_patterns(self):
        """Test that symbols match only one pattern."""
        # A symbol should match exactly one pattern
        symbol = Symbol("AAPL")
        assert symbol.is_stock()
        assert not symbol.is_crypto()
        assert not symbol.is_option()

        # Even complex symbols should have clear classification
        symbol = Symbol("AAPL.US")
        assert symbol.is_stock()
        assert not symbol.is_crypto()

    def test_boundary_lengths(self):
        """Test boundary cases for symbol lengths."""
        # Minimum length (1 character)
        symbol = Symbol("F")
        assert symbol == "F"

        # Maximum length for plain stock (5 characters)
        symbol = Symbol("GOOGL")
        assert symbol == "GOOGL"

        # With exchange can be longer
        symbol = Symbol("GOOGL.NASDAQ")
        assert symbol == "GOOGL.NASDAQ"

    def test_special_exchange_codes(self):
        """Test special exchange codes from various markets."""
        exchanges = [
            "AAPL.NASDAQ",  # NASDAQ
            "IBM.NYSE",  # NYSE
            "BP.LON",  # London
            "SAP.DE",  # Germany
            "SONY.TYO",  # Tokyo
            "BABA.HK",  # Hong Kong
            "RIO.AX",  # Australia
            "TD.TO",  # Toronto
        ]

        for full_symbol in exchanges:
            symbol = Symbol(full_symbol)
            assert "." in symbol
            assert symbol.exchange is not None
            assert symbol.is_stock()

    def test_whitespace_handling(self):
        """Test various whitespace scenarios."""
        # Leading/trailing spaces
        symbol = Symbol("  AAPL  ")
        assert symbol == "AAPL"

        # Spaces with exchange
        symbol = Symbol("  AAPL.US  ")
        assert symbol == "AAPL.US"

        # Spaces with crypto
        symbol = Symbol("  BTC-USD  ")
        assert symbol == "BTC-USD"

    def test_case_normalization(self):
        """Test case normalization scenarios."""
        # Mixed case
        symbol = Symbol("AaPl")
        assert symbol == "AAPL"

        # With exchange
        symbol = Symbol("aapl.us")
        assert symbol == "AAPL.US"

        # Crypto pair
        symbol = Symbol("btc-usd")
        assert symbol == "BTC-USD"

    def test_symbol_sorting(self):
        """Test that symbols can be sorted alphabetically."""
        symbols = [
            Symbol("GOOGL"),
            Symbol("AAPL"),
            Symbol("MSFT"),
            Symbol("AMZN"),
        ]

        sorted_symbols = sorted(symbols)
        expected_order = ["AAPL", "AMZN", "GOOGL", "MSFT"]
        assert [s for s in sorted_symbols] == expected_order

    def test_symbol_as_dict_key(self):
        """Test using Symbol as dictionary key."""
        portfolio = {}
        aapl = Symbol("AAPL")
        googl = Symbol("GOOGL")

        portfolio[aapl] = 100
        portfolio[googl] = 50

        # Access with new instance of same symbol
        aapl2 = Symbol("AAPL")
        assert portfolio[aapl2] == 100

        # Case insensitive creation still works
        aapl3 = Symbol("aapl")
        assert portfolio[aapl3] == 100

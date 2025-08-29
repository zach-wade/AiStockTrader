"""
Comprehensive unit tests for base value object classes.

Tests the ValueObject and ComparableValueObject base classes.
"""

import pytest

from src.domain.value_objects.base import ComparableValueObject, ValueObject


class ConcreteValueObject(ValueObject):
    """Concrete implementation of ValueObject for testing."""

    __slots__ = ("_value", "_name")

    def __init__(self, value: int, name: str):
        """Initialize concrete value object."""
        super().__setattr__("_value", value)
        super().__setattr__("_name", name)

    @property
    def value(self) -> int:
        """Get the value."""
        return self._value

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ConcreteValueObject):
            return False
        return self._value == other._value and self._name == other._name

    def __hash__(self) -> int:
        """Get hash."""
        return hash((self._value, self._name))

    def __repr__(self) -> str:
        """Get string representation."""
        return f"ConcreteValueObject(value={self._value}, name='{self._name}')"


class ConcreteComparableValueObject(ComparableValueObject):
    """Concrete implementation of ComparableValueObject for testing."""

    __slots__ = ("_value",)

    def __init__(self, value: int):
        """Initialize comparable value object."""
        super().__setattr__("_value", value)

    @property
    def value(self) -> int:
        """Get the value."""
        return self._value

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ConcreteComparableValueObject):
            return False
        return self._value == other._value

    def __lt__(self, other: "ConcreteComparableValueObject") -> bool:
        """Check if less than."""
        if not isinstance(other, ConcreteComparableValueObject):
            return NotImplemented
        return self._value < other._value

    def __hash__(self) -> int:
        """Get hash."""
        return hash(self._value)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"ConcreteComparableValueObject(value={self._value})"


class TestValueObject:
    """Test suite for ValueObject base class."""

    def test_initialization(self):
        """Test value object initialization."""
        vo = ConcreteValueObject(42, "test")
        assert vo.value == 42
        assert vo.name == "test"

    def test_equality(self):
        """Test value object equality."""
        vo1 = ConcreteValueObject(42, "test")
        vo2 = ConcreteValueObject(42, "test")
        vo3 = ConcreteValueObject(43, "test")
        vo4 = ConcreteValueObject(42, "other")

        assert vo1 == vo2
        assert vo1 != vo3
        assert vo1 != vo4
        assert vo1 != 42  # Different type
        assert vo1 != "test"  # Different type

    def test_hashability(self):
        """Test value object hashability."""
        vo1 = ConcreteValueObject(42, "test")
        vo2 = ConcreteValueObject(42, "test")
        vo3 = ConcreteValueObject(43, "test")

        # Equal objects should have same hash
        assert hash(vo1) == hash(vo2)

        # Can be used in sets
        value_set = {vo1, vo2, vo3}
        assert len(value_set) == 2  # vo1 and vo2 are equal

        # Can be used as dict keys
        value_dict = {vo1: "first", vo2: "second", vo3: "third"}
        assert len(value_dict) == 2
        assert value_dict[vo1] == "second"  # vo2 overwrote vo1

    def test_repr(self):
        """Test value object string representation."""
        vo = ConcreteValueObject(42, "test")
        repr_str = repr(vo)

        assert "ConcreteValueObject" in repr_str
        assert "42" in repr_str
        assert "test" in repr_str

    def test_immutability_modify_existing(self):
        """Test that existing attributes cannot be modified."""
        vo = ConcreteValueObject(42, "test")

        with pytest.raises(AttributeError, match="Cannot modify immutable"):
            vo._value = 100

        with pytest.raises(AttributeError, match="Cannot modify immutable"):
            vo._name = "changed"

    def test_immutability_add_new(self):
        """Test that new attributes cannot be added."""
        vo = ConcreteValueObject(42, "test")

        with pytest.raises(AttributeError, match="Cannot add new attribute"):
            vo.new_attr = "value"

        with pytest.raises(AttributeError, match="Cannot add new attribute"):
            vo._new_private = 123

    def test_slots_enforcement(self):
        """Test that __slots__ prevents __dict__ creation."""
        vo = ConcreteValueObject(42, "test")

        # Should not have __dict__
        assert not hasattr(vo, "__dict__")

        # Should only have defined slots
        assert hasattr(vo, "_value")
        assert hasattr(vo, "_name")

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        vo1 = ConcreteValueObject(42, "test1")
        vo2 = ConcreteValueObject(100, "test2")

        assert vo1.value == 42
        assert vo2.value == 100
        assert vo1.name == "test1"
        assert vo2.name == "test2"

        # They should be different objects
        assert vo1 is not vo2
        assert vo1 != vo2


class TestComparableValueObject:
    """Test suite for ComparableValueObject base class."""

    def test_initialization(self):
        """Test comparable value object initialization."""
        cvo = ConcreteComparableValueObject(42)
        assert cvo.value == 42

    def test_equality(self):
        """Test comparable value object equality."""
        cvo1 = ConcreteComparableValueObject(42)
        cvo2 = ConcreteComparableValueObject(42)
        cvo3 = ConcreteComparableValueObject(43)

        assert cvo1 == cvo2
        assert cvo1 != cvo3
        assert cvo1 != 42  # Different type

    def test_less_than(self):
        """Test less than comparison."""
        cvo1 = ConcreteComparableValueObject(10)
        cvo2 = ConcreteComparableValueObject(20)
        cvo3 = ConcreteComparableValueObject(30)

        assert cvo1 < cvo2
        assert cvo2 < cvo3
        assert cvo1 < cvo3
        assert not cvo2 < cvo1
        assert not cvo3 < cvo1

        # Equal values
        cvo4 = ConcreteComparableValueObject(20)
        assert not cvo2 < cvo4
        assert not cvo4 < cvo2

    def test_greater_than(self):
        """Test greater than comparison (provided by @total_ordering)."""
        cvo1 = ConcreteComparableValueObject(10)
        cvo2 = ConcreteComparableValueObject(20)
        cvo3 = ConcreteComparableValueObject(30)

        assert cvo2 > cvo1
        assert cvo3 > cvo2
        assert cvo3 > cvo1
        assert not cvo1 > cvo2
        assert not cvo1 > cvo3

    def test_less_than_or_equal(self):
        """Test less than or equal comparison (provided by @total_ordering)."""
        cvo1 = ConcreteComparableValueObject(10)
        cvo2 = ConcreteComparableValueObject(20)
        cvo3 = ConcreteComparableValueObject(20)

        assert cvo1 <= cvo2
        assert cvo2 <= cvo3
        assert cvo3 <= cvo2
        assert not cvo2 <= cvo1

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison (provided by @total_ordering)."""
        cvo1 = ConcreteComparableValueObject(10)
        cvo2 = ConcreteComparableValueObject(20)
        cvo3 = ConcreteComparableValueObject(20)

        assert cvo2 >= cvo1
        assert cvo2 >= cvo3
        assert cvo3 >= cvo2
        assert not cvo1 >= cvo2

    def test_sorting(self):
        """Test that comparable value objects can be sorted."""
        values = [
            ConcreteComparableValueObject(30),
            ConcreteComparableValueObject(10),
            ConcreteComparableValueObject(20),
            ConcreteComparableValueObject(15),
            ConcreteComparableValueObject(25),
        ]

        sorted_values = sorted(values)

        assert sorted_values[0].value == 10
        assert sorted_values[1].value == 15
        assert sorted_values[2].value == 20
        assert sorted_values[3].value == 25
        assert sorted_values[4].value == 30

    def test_min_max(self):
        """Test min/max operations on comparable value objects."""
        values = [
            ConcreteComparableValueObject(30),
            ConcreteComparableValueObject(10),
            ConcreteComparableValueObject(20),
        ]

        assert min(values).value == 10
        assert max(values).value == 30

    def test_hashability(self):
        """Test comparable value object hashability."""
        cvo1 = ConcreteComparableValueObject(42)
        cvo2 = ConcreteComparableValueObject(42)
        cvo3 = ConcreteComparableValueObject(43)

        # Equal objects should have same hash
        assert hash(cvo1) == hash(cvo2)

        # Can be used in sets
        value_set = {cvo1, cvo2, cvo3}
        assert len(value_set) == 2

        # Can be used as dict keys
        value_dict = {cvo1: "first", cvo2: "second", cvo3: "third"}
        assert len(value_dict) == 2

    def test_immutability(self):
        """Test that comparable value objects are immutable."""
        cvo = ConcreteComparableValueObject(42)

        with pytest.raises(AttributeError, match="Cannot modify immutable"):
            cvo._value = 100

        with pytest.raises(AttributeError, match="Cannot add new attribute"):
            cvo.new_attr = "value"

    def test_comparison_with_incompatible_type(self):
        """Test comparison with incompatible types."""
        cvo = ConcreteComparableValueObject(42)

        # Equality should return False for different types
        assert cvo != 42
        assert cvo != "42"
        assert cvo != None

        # Ordering comparisons should raise TypeError
        with pytest.raises(TypeError):
            cvo < 42

        with pytest.raises(TypeError):
            cvo > "42"

        with pytest.raises(TypeError):
            cvo <= None

    def test_comparison_chain(self):
        """Test chained comparisons."""
        cvo1 = ConcreteComparableValueObject(10)
        cvo2 = ConcreteComparableValueObject(20)
        cvo3 = ConcreteComparableValueObject(30)

        # Chained comparisons should work
        assert cvo1 < cvo2 < cvo3
        assert cvo1 < cvo2 <= ConcreteComparableValueObject(20)
        assert cvo3 > cvo2 > cvo1
        assert cvo3 >= ConcreteComparableValueObject(30) >= cvo2


class TestValueObjectInheritance:
    """Test inheritance patterns for value objects."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that abstract base classes cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ValueObject()

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComparableValueObject()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteValueObject(ValueObject):
            __slots__ = ("_value",)

            def __init__(self, value):
                super().__setattr__("_value", value)

            # Missing __eq__, __hash__, __repr__

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteValueObject(42)

    def test_comparable_subclass_must_implement_lt(self):
        """Test that comparable subclasses must implement __lt__."""

        class IncompleteComparable(ComparableValueObject):
            __slots__ = ("_value",)

            def __init__(self, value):
                super().__setattr__("_value", value)

            def __eq__(self, other):
                return self._value == other._value

            def __hash__(self):
                return hash(self._value)

            def __repr__(self):
                return f"IncompleteComparable({self._value})"

            # Missing __lt__

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteComparable(42)


class TestValueObjectEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_slots(self):
        """Test value object with no data."""

        class EmptyValueObject(ValueObject):
            __slots__ = ()

            def __eq__(self, other):
                return isinstance(other, EmptyValueObject)

            def __hash__(self):
                return hash(EmptyValueObject)

            def __repr__(self):
                return "EmptyValueObject()"

        vo1 = EmptyValueObject()
        vo2 = EmptyValueObject()

        assert vo1 == vo2
        assert hash(vo1) == hash(vo2)
        assert repr(vo1) == "EmptyValueObject()"

    def test_complex_nested_value_object(self):
        """Test value object containing other value objects."""

        class Address(ValueObject):
            __slots__ = ("_street", "_city")

            def __init__(self, street: str, city: str):
                super().__setattr__("_street", street)
                super().__setattr__("_city", city)

            @property
            def street(self):
                return self._street

            @property
            def city(self):
                return self._city

            def __eq__(self, other):
                if not isinstance(other, Address):
                    return False
                return self._street == other._street and self._city == other._city

            def __hash__(self):
                return hash((self._street, self._city))

            def __repr__(self):
                return f"Address(street='{self._street}', city='{self._city}')"

        class Person(ValueObject):
            __slots__ = ("_name", "_address")

            def __init__(self, name: str, address: Address):
                super().__setattr__("_name", name)
                super().__setattr__("_address", address)

            @property
            def name(self):
                return self._name

            @property
            def address(self):
                return self._address

            def __eq__(self, other):
                if not isinstance(other, Person):
                    return False
                return self._name == other._name and self._address == other._address

            def __hash__(self):
                return hash((self._name, self._address))

            def __repr__(self):
                return f"Person(name='{self._name}', address={self._address})"

        addr1 = Address("123 Main St", "Springfield")
        addr2 = Address("123 Main St", "Springfield")

        person1 = Person("John Doe", addr1)
        person2 = Person("John Doe", addr2)

        assert person1 == person2
        assert hash(person1) == hash(person2)

        # Nested immutability
        with pytest.raises(AttributeError):
            person1._address._street = "456 Elm St"

    def test_value_object_with_none_values(self):
        """Test value object handling None values."""

        class NullableValueObject(ValueObject):
            __slots__ = ("_value",)

            def __init__(self, value):
                super().__setattr__("_value", value)

            @property
            def value(self):
                return self._value

            def __eq__(self, other):
                if not isinstance(other, NullableValueObject):
                    return False
                return self._value == other._value

            def __hash__(self):
                return hash(self._value) if self._value is not None else 0

            def __repr__(self):
                return f"NullableValueObject(value={self._value})"

        vo1 = NullableValueObject(None)
        vo2 = NullableValueObject(None)
        vo3 = NullableValueObject(42)

        assert vo1 == vo2
        assert vo1 != vo3
        assert hash(vo1) == hash(vo2)

    def test_value_object_copy_behavior(self):
        """Test that value objects behave correctly with copy operations."""
        import copy

        vo = ConcreteValueObject(42, "test")

        # Shallow copy
        vo_copy = copy.copy(vo)
        assert vo == vo_copy
        assert vo is not vo_copy

        # Deep copy
        vo_deepcopy = copy.deepcopy(vo)
        assert vo == vo_deepcopy
        assert vo is not vo_deepcopy

        # Copies should also be immutable
        with pytest.raises(AttributeError):
            vo_copy._value = 100

        with pytest.raises(AttributeError):
            vo_deepcopy._value = 100

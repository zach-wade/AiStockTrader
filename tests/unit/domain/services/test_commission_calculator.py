"""
Unit tests for commission calculator domain service
"""

from decimal import Decimal

import pytest

from src.domain.services.commission_calculator import (
    CommissionCalculatorFactory,
    CommissionSchedule,
    CommissionType,
    PercentageCommissionCalculator,
    PerShareCommissionCalculator,
)
from src.domain.value_objects.money import Money
from src.domain.value_objects.quantity import Quantity


class TestCommissionSchedule:
    """Test commission schedule validation"""

    def test_valid_schedule(self):
        """Test creating valid commission schedule"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=Decimal("10.00"),
        )
        assert schedule.rate == Decimal("0.01")
        assert schedule.minimum == Decimal("1.00")
        assert schedule.maximum == Decimal("10.00")

    def test_negative_rate_raises_error(self):
        """Test that negative rate raises ValueError"""
        with pytest.raises(ValueError, match="rate cannot be negative"):
            CommissionSchedule(
                commission_type=CommissionType.PER_SHARE,
                rate=Decimal("-0.01"),
                minimum=Decimal("1.00"),
            )

    def test_negative_minimum_raises_error(self):
        """Test that negative minimum raises ValueError"""
        with pytest.raises(ValueError, match="Minimum commission cannot be negative"):
            CommissionSchedule(
                commission_type=CommissionType.PER_SHARE,
                rate=Decimal("0.01"),
                minimum=Decimal("-1.00"),
            )

    def test_maximum_less_than_minimum_raises_error(self):
        """Test that maximum < minimum raises ValueError"""
        with pytest.raises(ValueError, match="Maximum commission must be greater than minimum"):
            CommissionSchedule(
                commission_type=CommissionType.PER_SHARE,
                rate=Decimal("0.01"),
                minimum=Decimal("10.00"),
                maximum=Decimal("5.00"),
            )


class TestPerShareCommissionCalculator:
    """Test per-share commission calculation"""

    @pytest.fixture
    def calculator(self):
        """Create calculator with standard schedule"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=Decimal("10.00"),
        )
        return PerShareCommissionCalculator(schedule)

    def test_basic_calculation(self, calculator):
        """Test basic per-share calculation"""
        quantity = Quantity(Decimal("100"))
        commission = calculator.calculate(quantity)

        # 100 shares * $0.01 = $1.00
        assert commission.amount == Decimal("1.00")

    def test_minimum_commission_applied(self, calculator):
        """Test minimum commission is applied"""
        quantity = Quantity(Decimal("50"))
        commission = calculator.calculate(quantity)

        # 50 shares * $0.01 = $0.50, but minimum is $1.00
        assert commission.amount == Decimal("1.00")

    def test_maximum_commission_applied(self, calculator):
        """Test maximum commission is applied"""
        quantity = Quantity(Decimal("2000"))
        commission = calculator.calculate(quantity)

        # 2000 shares * $0.01 = $20.00, but maximum is $10.00
        assert commission.amount == Decimal("10.00")

    def test_negative_quantity_uses_absolute_value(self, calculator):
        """Test that negative quantities use absolute value"""
        quantity = Quantity(Decimal("-100"))
        commission = calculator.calculate(quantity)

        # |-100| shares * $0.01 = $1.00
        assert commission.amount == Decimal("1.00")


class TestPercentageCommissionCalculator:
    """Test percentage-based commission calculation"""

    @pytest.fixture
    def calculator(self):
        """Create calculator with percentage schedule"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PERCENTAGE,
            rate=Decimal("0.1"),  # 0.1%
            minimum=Decimal("1.00"),
            maximum=Decimal("50.00"),
        )
        return PercentageCommissionCalculator(schedule)

    def test_basic_percentage_calculation(self, calculator):
        """Test basic percentage calculation"""
        quantity = Quantity(Decimal("100"))
        price = Money(Decimal("50.00"))  # $50 per share
        commission = calculator.calculate(quantity, price)

        # Trade value: 100 * $50 = $5000
        # Commission: $5000 * 0.1% = $5.00
        assert commission.amount == Decimal("5.00")

    def test_requires_price(self, calculator):
        """Test that price is required for percentage calculation"""
        quantity = Quantity(Decimal("100"))

        with pytest.raises(ValueError, match="Price required for percentage commission"):
            calculator.calculate(quantity)

    def test_minimum_percentage_commission(self, calculator):
        """Test minimum commission is applied to percentage calculation"""
        quantity = Quantity(Decimal("10"))
        price = Money(Decimal("10.00"))  # $10 per share
        commission = calculator.calculate(quantity, price)

        # Trade value: 10 * $10 = $100
        # Commission: $100 * 0.1% = $0.10, but minimum is $1.00
        assert commission.amount == Decimal("1.00")

    def test_maximum_percentage_commission(self, calculator):
        """Test maximum commission is applied to percentage calculation"""
        quantity = Quantity(Decimal("1000"))
        price = Money(Decimal("100.00"))  # $100 per share
        commission = calculator.calculate(quantity, price)

        # Trade value: 1000 * $100 = $100,000
        # Commission: $100,000 * 0.1% = $100.00, but maximum is $50.00
        assert commission.amount == Decimal("50.00")


class TestCommissionCalculatorFactory:
    """Test commission calculator factory"""

    def test_create_per_share_calculator(self):
        """Test factory creates per-share calculator"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE, rate=Decimal("0.01"), minimum=Decimal("1.00")
        )

        calculator = CommissionCalculatorFactory.create(schedule)
        assert isinstance(calculator, PerShareCommissionCalculator)

    def test_create_percentage_calculator(self):
        """Test factory creates percentage calculator"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PERCENTAGE,
            rate=Decimal("0.001"),
            minimum=Decimal("1.00"),
        )

        calculator = CommissionCalculatorFactory.create(schedule)
        assert isinstance(calculator, PercentageCommissionCalculator)

    def test_unsupported_type_raises_error(self):
        """Test unsupported commission type raises error"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.TIERED,  # Not yet implemented
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
        )

        with pytest.raises(ValueError, match="Unsupported commission type"):
            CommissionCalculatorFactory.create(schedule)


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions"""

    def test_zero_quantity_commission(self):
        """Test commission calculation with zero quantity"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE, rate=Decimal("0.01"), minimum=Decimal("1.00")
        )
        calculator = PerShareCommissionCalculator(schedule)

        quantity = Quantity(Decimal("0"))
        commission = calculator.calculate(quantity)

        # Zero shares should still have minimum commission
        assert commission.amount == Decimal("1.00")

    def test_very_large_quantity(self):
        """Test commission calculation with very large quantity"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=Decimal("100.00"),
        )
        calculator = PerShareCommissionCalculator(schedule)

        quantity = Quantity(Decimal("1000000"))  # 1 million shares
        commission = calculator.calculate(quantity)

        # Should be capped at maximum
        assert commission.amount == Decimal("100.00")

    def test_precise_decimal_calculation(self):
        """Test commission calculation with precise decimal values"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.0035"),  # $0.0035 per share
            minimum=Decimal("0.35"),
        )
        calculator = PerShareCommissionCalculator(schedule)

        quantity = Quantity(Decimal("123.456"))
        commission = calculator.calculate(quantity)

        expected = Decimal("123.456") * Decimal("0.0035")
        assert commission.amount == max(expected, Decimal("0.35"))

    def test_percentage_with_zero_price(self):
        """Test percentage commission with zero price"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PERCENTAGE,
            rate=Decimal("0.001"),
            minimum=Decimal("1.00"),
        )
        calculator = PercentageCommissionCalculator(schedule)

        quantity = Quantity(Decimal("100"))
        price = Money(Decimal("0"))
        commission = calculator.calculate(quantity, price)

        # Zero trade value should still have minimum commission
        assert commission.amount == Decimal("1.00")

    def test_fractional_percentage_rate(self):
        """Test percentage commission with fractional rate"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PERCENTAGE,
            rate=Decimal("0.0025"),  # 0.0025%
            minimum=Decimal("0.01"),
        )
        calculator = PercentageCommissionCalculator(schedule)

        quantity = Quantity(Decimal("1000"))
        price = Money(Decimal("100"))
        commission = calculator.calculate(quantity, price)

        # Trade value: 1000 * 100 = 100,000
        # Commission: 100,000 * 0.0025% = 2.50
        assert commission.amount == Decimal("2.50")

    def test_schedule_with_no_maximum(self):
        """Test commission schedule with no maximum cap"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=None,  # No maximum
        )
        calculator = PerShareCommissionCalculator(schedule)

        quantity = Quantity(Decimal("10000"))
        commission = calculator.calculate(quantity)

        # 10000 * 0.01 = 100, no cap
        assert commission.amount == Decimal("100.00")

    def test_minimum_equals_maximum(self):
        """Test when minimum equals maximum (fixed commission)"""
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("5.00"),
            maximum=Decimal("5.00"),
        )
        calculator = PerShareCommissionCalculator(schedule)

        # Any quantity should result in $5 commission
        for qty in [10, 100, 1000]:
            commission = calculator.calculate(Quantity(Decimal(str(qty))))
            assert commission.amount == Decimal("5.00")


class TestDefaultSchedules:
    """Test default commission schedules"""

    def test_default_retail_schedule(self):
        """Test default retail commission schedule"""
        from src.domain.services.commission_calculator import DEFAULT_RETAIL_SCHEDULE

        assert DEFAULT_RETAIL_SCHEDULE.commission_type == CommissionType.PER_SHARE
        assert DEFAULT_RETAIL_SCHEDULE.rate == Decimal("0.01")
        assert DEFAULT_RETAIL_SCHEDULE.minimum == Decimal("1.00")
        assert DEFAULT_RETAIL_SCHEDULE.maximum == Decimal("5.00")

        # Test calculation with default retail schedule
        calculator = CommissionCalculatorFactory.create(DEFAULT_RETAIL_SCHEDULE)
        commission = calculator.calculate(Quantity(Decimal("100")))
        assert commission.amount == Decimal("1.00")  # Minimum applies

    def test_default_institutional_schedule(self):
        """Test default institutional commission schedule"""
        from src.domain.services.commission_calculator import DEFAULT_INSTITUTIONAL_SCHEDULE

        assert DEFAULT_INSTITUTIONAL_SCHEDULE.commission_type == CommissionType.PER_SHARE
        assert DEFAULT_INSTITUTIONAL_SCHEDULE.rate == Decimal("0.005")
        assert DEFAULT_INSTITUTIONAL_SCHEDULE.minimum == Decimal("0.35")
        assert DEFAULT_INSTITUTIONAL_SCHEDULE.maximum is None

        # Test calculation with default institutional schedule
        calculator = CommissionCalculatorFactory.create(DEFAULT_INSTITUTIONAL_SCHEDULE)
        commission = calculator.calculate(Quantity(Decimal("1000")))
        assert commission.amount == Decimal("5.00")  # 1000 * 0.005

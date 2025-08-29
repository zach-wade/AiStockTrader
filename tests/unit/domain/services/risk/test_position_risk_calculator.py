"""
Comprehensive Tests for Position Risk Calculator Service
======================================================

Tests for the PositionRiskCalculator domain service that handles
individual position risk analysis and metrics.
"""

from decimal import Decimal

import pytest

from src.domain.entities.position import Position
from src.domain.services.risk.position_risk_calculator import PositionRiskCalculator
from src.domain.value_objects import Money, Price, Quantity


class TestCalculatePositionRisk:
    """Test position risk calculation functionality."""

    def test_calculate_position_risk_open_long_position(self):
        """Test risk calculation for open long position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),
            commission=Money(Decimal("5")),
        )
        position.realized_pnl = Money(Decimal("200"))  # From partial closes
        position.stop_loss_price = Price(Decimal("145"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("160"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Verify position value: 100 * 160 = 16000
        assert metrics["position_value"] == Money(Decimal("16000"))

        # Verify unrealized P&L: 100 * (160 - 150) = 1000
        assert metrics["unrealized_pnl"] == Money(Decimal("1000"))

        # Verify realized P&L
        assert metrics["realized_pnl"] == Money(Decimal("200"))

        # Verify total P&L: realized + unrealized - commission = 200 + 1000 - 5 = 1195
        assert metrics["total_pnl"] == Money(Decimal("1195"))

        # Verify return percentage: 1195 / (100 * 150) * 100 = 7.9666...%
        expected_return = (Decimal("1195") / Decimal("15000")) * Decimal("100")
        assert abs(metrics["return_pct"] - expected_return) < Decimal("0.01")

        # Verify risk amount: 100 * |160 - 145| = 1500
        assert metrics["risk_amount"] == Money(Decimal("1500"))

        # Verify position was updated with current price
        assert position.current_price == current_price

    def test_calculate_position_risk_open_short_position(self):
        """Test risk calculation for open short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )
        position.stop_loss_price = Price(Decimal("155"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("140"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Verify position value: abs(-100) * 140 = 14000
        assert metrics["position_value"] == Money(Decimal("14000"))

        # Verify unrealized P&L: 100 * (150 - 140) = 1000 profit
        assert metrics["unrealized_pnl"] == Money(Decimal("1000"))

        # Verify total P&L includes unrealized
        assert metrics["total_pnl"] == Money(Decimal("1000"))

        # Verify risk amount: 100 * |140 - 155| = 1500
        assert metrics["risk_amount"] == Money(Decimal("1500"))

    def test_calculate_position_risk_closed_position(self):
        """Test risk calculation for closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        position.close_position(Price(Decimal("160")), Money(Decimal("5")))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("170"))  # Shouldn't matter for closed position

        metrics = calculator.calculate_position_risk(position, current_price)

        # Closed position metrics
        assert metrics["position_value"] == Money(Decimal("0"))
        assert metrics["unrealized_pnl"] == Money(Decimal("0"))
        assert metrics["total_pnl"] == position.realized_pnl
        assert metrics["return_pct"] == Decimal("0")
        assert metrics["risk_amount"] == Money(Decimal("0"))

    def test_calculate_position_risk_no_stop_loss(self):
        """Test risk calculation when no stop loss is set."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        # No stop loss set

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("160"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Risk amount should be zero when no stop loss
        assert metrics["risk_amount"] == Money(Decimal("0"))

        # Other metrics should still be calculated
        assert metrics["position_value"] == Money(Decimal("16000"))
        assert metrics["unrealized_pnl"] == Money(Decimal("1000"))

    def test_calculate_position_risk_no_current_price_previously(self):
        """Test risk calculation when position has no previous current price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        # Position has no current price set initially
        assert position.current_price is None

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("160"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Should calculate normally after setting price
        assert metrics["position_value"] == Money(Decimal("16000"))
        assert metrics["unrealized_pnl"] == Money(Decimal("1000"))

    def test_calculate_position_risk_fractional_shares(self):
        """Test risk calculation with fractional shares."""
        position = Position.open_position(
            symbol="BRK.A", quantity=Quantity(Decimal("0.5")), entry_price=Price(Decimal("500000"))
        )
        position.stop_loss_price = Price(Decimal("480000"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("520000"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Position value: 0.5 * 520000 = 260000
        assert metrics["position_value"] == Money(Decimal("260000"))

        # Unrealized P&L: 0.5 * (520000 - 500000) = 10000
        assert metrics["unrealized_pnl"] == Money(Decimal("10000"))

        # Risk amount: 0.5 * |520000 - 480000| = 20000
        assert metrics["risk_amount"] == Money(Decimal("20000"))

    def test_calculate_position_risk_negative_price_difference(self):
        """Test risk calculation when position is at a loss."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )
        position.stop_loss_price = Price(Decimal("140"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("145"))  # Position at loss

        metrics = calculator.calculate_position_risk(position, current_price)

        # Position value: 100 * 145 = 14500
        assert metrics["position_value"] == Money(Decimal("14500"))

        # Unrealized P&L: 100 * (145 - 150) = -500 loss
        assert metrics["unrealized_pnl"] == Money(Decimal("-500"))

        # Total P&L should include loss
        assert metrics["total_pnl"] == Money(Decimal("-500"))

        # Risk amount: 100 * |145 - 140| = 500
        assert metrics["risk_amount"] == Money(Decimal("500"))

    def test_calculate_position_risk_with_partial_realized_pnl(self):
        """Test risk calculation with existing realized P&L from partial closes."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("150")), entry_price=Price(Decimal("100"))
        )

        # Simulate partial close
        position.reduce_position(
            quantity=Quantity(Decimal("50")),
            exit_price=Price(Decimal("110")),
            commission=Money(Decimal("3")),
        )

        # Remaining 100 shares at entry price 100
        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("120"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Position value: 100 * 120 = 12000
        assert metrics["position_value"] == Money(Decimal("12000"))

        # Unrealized P&L on remaining shares: 100 * (120 - 100) = 2000
        assert metrics["unrealized_pnl"] == Money(Decimal("2000"))

        # Should include previous realized P&L
        assert metrics["realized_pnl"] == position.realized_pnl

        # Total P&L = realized + unrealized - commission
        expected_total = position.realized_pnl + Money(Decimal("2000")) - position.commission_paid
        assert metrics["total_pnl"] == expected_total

    def test_calculate_position_risk_high_precision(self):
        """Test risk calculation maintains high precision."""
        position = Position.open_position(
            symbol="PRECISE",
            quantity=Quantity(Decimal("123.456789")),
            entry_price=Price(Decimal("98.765432")),
        )
        position.stop_loss_price = Price(Decimal("95.123456"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("101.234567"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Verify calculations maintain precision
        expected_position_value = Decimal("123.456789") * Decimal("101.234567")
        assert metrics["position_value"] == Money(expected_position_value)

        expected_unrealized = Decimal("123.456789") * (Decimal("101.234567") - Decimal("98.765432"))
        assert metrics["unrealized_pnl"] == Money(expected_unrealized)

        expected_risk = Decimal("123.456789") * abs(Decimal("101.234567") - Decimal("95.123456"))
        assert metrics["risk_amount"] == Money(expected_risk)


class TestCalculatePositionRiskReward:
    """Test risk/reward ratio calculation functionality."""

    def test_calculate_position_risk_reward_basic(self):
        """Test basic risk/reward ratio calculation."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("95"))
        take_profit = Price(Decimal("110"))

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 100 - 95 = 5
        # Reward: 110 - 100 = 10
        # Ratio: 10 / 5 = 2.0
        assert ratio == Decimal("2.0")

    def test_calculate_position_risk_reward_unfavorable(self):
        """Test risk/reward ratio for unfavorable trade setup."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("90"))  # 10 point risk
        take_profit = Price(Decimal("105"))  # 5 point reward

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 10, Reward: 5, Ratio: 0.5 (unfavorable)
        assert ratio == Decimal("0.5")

    def test_calculate_position_risk_reward_excellent(self):
        """Test risk/reward ratio for excellent trade setup."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("98"))  # 2 point risk
        take_profit = Price(Decimal("108"))  # 8 point reward

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 2, Reward: 8, Ratio: 4.0 (excellent)
        assert ratio == Decimal("4.0")

    def test_calculate_position_risk_reward_short_setup(self):
        """Test risk/reward ratio for short position setup."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("105"))  # Stop above for short
        take_profit = Price(Decimal("85"))  # Profit target below

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: |100 - 105| = 5
        # Reward: |85 - 100| = 15
        # Ratio: 15 / 5 = 3.0
        assert ratio == Decimal("3.0")

    def test_calculate_position_risk_reward_equal_risk_reward(self):
        """Test risk/reward ratio when risk equals reward."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("95"))
        take_profit = Price(Decimal("105"))

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 5, Reward: 5, Ratio: 1.0 (breakeven risk profile)
        assert ratio == Decimal("1.0")

    def test_calculate_position_risk_reward_zero_risk(self):
        """Test risk/reward ratio calculation with zero risk."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100"))
        stop_loss = Price(Decimal("100"))  # No risk (entry = stop)
        take_profit = Price(Decimal("110"))

        with pytest.raises(ValueError, match="Risk cannot be zero"):
            calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

    def test_calculate_position_risk_reward_high_precision(self):
        """Test risk/reward calculation with high precision values."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("123.456789"))
        stop_loss = Price(Decimal("121.111111"))
        take_profit = Price(Decimal("127.888888"))

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 123.456789 - 121.111111 = 2.345678
        # Reward: 127.888888 - 123.456789 = 4.432099
        # Ratio: 4.432099 / 2.345678
        expected_ratio = Decimal("4.432099") / Decimal("2.345678")
        assert abs(ratio - expected_ratio) < Decimal("0.000001")

    def test_calculate_position_risk_reward_fractional_prices(self):
        """Test risk/reward calculation with fractional price movements."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("50.25"))
        stop_loss = Price(Decimal("49.75"))  # 0.50 risk
        take_profit = Price(Decimal("51.75"))  # 1.50 reward

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 0.50, Reward: 1.50, Ratio: 3.0
        assert ratio == Decimal("3.0")

    def test_calculate_position_risk_reward_very_small_movements(self):
        """Test risk/reward calculation with very small price movements."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("99.99"))  # 0.01 risk
        take_profit = Price(Decimal("100.03"))  # 0.03 reward

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 0.01, Reward: 0.03, Ratio: 3.0
        assert ratio == Decimal("3.0")

    def test_calculate_position_risk_reward_large_values(self):
        """Test risk/reward calculation with large price values."""
        calculator = PositionRiskCalculator()

        entry_price = Price(Decimal("50000"))
        stop_loss = Price(Decimal("48000"))  # 2000 risk
        take_profit = Price(Decimal("55000"))  # 5000 reward

        ratio = calculator.calculate_position_risk_reward(entry_price, stop_loss, take_profit)

        # Risk: 2000, Reward: 5000, Ratio: 2.5
        assert ratio == Decimal("2.5")


class TestPositionRiskCalculatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_calculate_position_risk_extreme_values(self):
        """Test position risk calculation with extreme values."""
        position = Position.open_position(
            symbol="EXTREME",
            quantity=Quantity(Decimal("0.000001")),  # Very small quantity
            entry_price=Price(Decimal("1000000")),  # Very large price
        )

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("1100000"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Should handle extreme values without error
        expected_value = Decimal("0.000001") * Decimal("1100000")
        assert metrics["position_value"] == Money(expected_value)

        expected_pnl = Decimal("0.000001") * (Decimal("1100000") - Decimal("1000000"))
        assert metrics["unrealized_pnl"] == Money(expected_pnl)

    def test_calculate_position_risk_maintains_precision(self):
        """Test that calculations maintain decimal precision throughout."""
        position = Position.open_position(
            symbol="PRECISION",
            quantity=Quantity(Decimal("33.333333")),
            entry_price=Price(Decimal("66.666666")),
        )
        position.realized_pnl = Money(Decimal("11.111111"))
        position.commission_paid = Money(Decimal("2.222222"))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("77.777777"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Verify precision is maintained in complex calculations
        expected_position_value = Decimal("33.333333") * Decimal("77.777777")
        assert metrics["position_value"] == Money(expected_position_value)

        expected_unrealized = Decimal("33.333333") * (Decimal("77.777777") - Decimal("66.666666"))
        assert metrics["unrealized_pnl"] == Money(expected_unrealized)

    def test_risk_calculator_thread_safety(self):
        """Test that risk calculator is stateless and thread-safe."""
        calculator = PositionRiskCalculator()

        # Create multiple positions
        position1 = Position.open_position("AAPL", Quantity(Decimal("100")), Price(Decimal("150")))
        position2 = Position.open_position("MSFT", Quantity(Decimal("50")), Price(Decimal("300")))

        # Calculate risks simultaneously (simulating concurrent access)
        metrics1 = calculator.calculate_position_risk(position1, Price(Decimal("160")))
        metrics2 = calculator.calculate_position_risk(position2, Price(Decimal("310")))

        # Verify independent calculations
        assert metrics1["position_value"] == Money(Decimal("16000"))
        assert metrics2["position_value"] == Money(Decimal("15500"))

        # Verify no state contamination
        assert position1.current_price.value == Decimal("160")
        assert position2.current_price.value == Decimal("310")

    def test_calculate_position_risk_default_values(self):
        """Test that metrics have appropriate default values."""
        position = Position.open_position(
            symbol="DEFAULT", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("50"))
        )
        # Close position to test defaults
        position.close_position(Price(Decimal("55")))

        calculator = PositionRiskCalculator()
        current_price = Price(Decimal("60"))

        metrics = calculator.calculate_position_risk(position, current_price)

        # Verify default values for closed position
        assert metrics["position_value"] == Money(Decimal("0"))
        assert metrics["unrealized_pnl"] == Money(Decimal("0"))
        assert metrics["return_pct"] == Decimal("0")
        assert metrics["risk_amount"] == Money(Decimal("0"))

        # Realized P&L should still be present
        assert metrics["realized_pnl"] != Money(Decimal("0"))
        assert metrics["total_pnl"] == metrics["realized_pnl"]

    def test_risk_reward_boundary_conditions(self):
        """Test risk/reward calculation at boundary conditions."""
        calculator = PositionRiskCalculator()

        # Minimum possible difference
        entry = Price(Decimal("100.01"))
        stop = Price(Decimal("100.00"))
        profit = Price(Decimal("100.02"))

        ratio = calculator.calculate_position_risk_reward(entry, stop, profit)

        # Risk: 0.01, Reward: 0.01, Ratio: 1.0
        assert ratio == Decimal("1.0")

        # Test with very small stop loss difference
        stop_tiny = Price(Decimal("100.001"))
        profit_large = Price(Decimal("110"))

        ratio_large = calculator.calculate_position_risk_reward(entry, stop_tiny, profit_large)

        # Should handle very small denominators correctly
        expected = Decimal("9.99") / Decimal("0.009")  # Approximately 1110
        assert abs(ratio_large - expected) < Decimal("1")  # Within reasonable precision

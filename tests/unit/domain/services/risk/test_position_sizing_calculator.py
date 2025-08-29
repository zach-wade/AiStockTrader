"""Comprehensive tests for PositionSizingCalculator domain service.

This module provides comprehensive test coverage for the PositionSizingCalculator service,
focusing on Kelly Criterion calculations and position sizing optimization.
Critical for production safety as this service determines optimal position sizes for real money.
"""

# Standard library imports
from decimal import Decimal

import pytest

# Local imports
from src.domain.services.risk.position_sizing_calculator import PositionSizingCalculator
from src.domain.value_objects.money import Money


class TestPositionSizingCalculator:
    """Comprehensive test suite for PositionSizingCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create PositionSizingCalculator instance."""
        return PositionSizingCalculator()

    # Basic Kelly Criterion tests
    def test_calculate_kelly_criterion_basic_profitable(self, calculator):
        """Test basic Kelly calculation with profitable scenario."""
        win_probability = Decimal("0.6")  # 60% win rate
        win_amount = Money(Decimal("100.00"))  # $100 average win
        loss_amount = Money(Decimal("50.00"))  # $50 average loss

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (p * b - q) / b = (0.6 * 2 - 0.4) / 2 = 0.4
        # But capped at 0.25
        assert result == Decimal("0.25")  # Should be capped at 25%

    def test_calculate_kelly_criterion_exact_formula(self, calculator):
        """Test Kelly formula calculation without cap."""
        win_probability = Decimal("0.55")  # 55% win rate
        win_amount = Money(Decimal("100.00"))  # $100 average win
        loss_amount = Money(Decimal("100.00"))  # $100 average loss (1:1 ratio)

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (p * b - q) / b = (0.55 * 1 - 0.45) / 1 = 0.1
        assert result == Decimal("0.1")  # 10% position size

    def test_calculate_kelly_criterion_unfavorable_odds(self, calculator):
        """Test Kelly calculation with unfavorable odds (negative result)."""
        win_probability = Decimal("0.4")  # 40% win rate
        win_amount = Money(Decimal("50.00"))  # $50 average win
        loss_amount = Money(Decimal("100.00"))  # $100 average loss

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.4 * 0.5 - 0.6) / 0.5 = -0.8
        # Should return negative value indicating unfavorable odds
        assert result == Decimal("-0.8")

    def test_calculate_kelly_criterion_breakeven(self, calculator):
        """Test Kelly calculation at breakeven point."""
        win_probability = Decimal("0.5")  # 50% win rate
        win_amount = Money(Decimal("100.00"))  # $100 average win
        loss_amount = Money(Decimal("100.00"))  # $100 average loss

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.5 * 1 - 0.5) / 1 = 0
        assert result == Decimal("0")

    # Cap testing
    def test_calculate_kelly_criterion_cap_at_25_percent(self, calculator):
        """Test that Kelly result is capped at 25% for safety."""
        win_probability = Decimal("0.8")  # 80% win rate
        win_amount = Money(Decimal("200.00"))  # High win amount
        loss_amount = Money(Decimal("50.00"))  # Low loss amount

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly would be high, but should be capped at 25%
        assert result == Decimal("0.25")

    def test_calculate_kelly_criterion_under_cap(self, calculator):
        """Test Kelly result under 25% cap (should not be capped)."""
        win_probability = Decimal("0.52")  # 52% win rate
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("110.00"))  # Slightly unfavorable ratio

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.52 * (100/110) - 0.48) / (100/110) ≈ -0.054
        assert result < Decimal("0.25")
        assert result < Decimal("0")  # Should be negative

    # Edge case tests for win_probability
    def test_calculate_kelly_criterion_zero_win_probability_error(self, calculator):
        """Test Kelly calculation with zero win probability (should raise error)."""
        win_probability = Decimal("0")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_one_win_probability_error(self, calculator):
        """Test Kelly calculation with 100% win probability (should raise error)."""
        win_probability = Decimal("1")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_negative_win_probability_error(self, calculator):
        """Test Kelly calculation with negative win probability (should raise error)."""
        win_probability = Decimal("-0.1")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_over_one_win_probability_error(self, calculator):
        """Test Kelly calculation with win probability > 1 (should raise error)."""
        win_probability = Decimal("1.5")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    # Edge case tests for amounts
    def test_calculate_kelly_criterion_zero_win_amount_error(self, calculator):
        """Test Kelly calculation with zero win amount (should raise error)."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("0.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_zero_loss_amount_error(self, calculator):
        """Test Kelly calculation with zero loss amount (should raise error)."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("0.00"))

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_negative_win_amount_error(self, calculator):
        """Test Kelly calculation with negative win amount (should raise error)."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("-100.00"))
        loss_amount = Money(Decimal("50.00"))

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    def test_calculate_kelly_criterion_negative_loss_amount_error(self, calculator):
        """Test Kelly calculation with negative loss amount (should raise error)."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("-50.00"))

        with pytest.raises(ValueError, match="Win and loss amounts must be positive"):
            calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

    # Precision and boundary tests
    def test_calculate_kelly_criterion_very_small_amounts(self, calculator):
        """Test Kelly calculation with very small monetary amounts."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("0.01"))  # 1 cent
        loss_amount = Money(Decimal("0.01"))  # 1 cent

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.6 * 1 - 0.4) / 1 = 0.2
        assert result == Decimal("0.2")

    def test_calculate_kelly_criterion_very_large_amounts(self, calculator):
        """Test Kelly calculation with very large monetary amounts."""
        win_probability = Decimal("0.55")
        win_amount = Money(Decimal("1000000.00"))  # $1M
        loss_amount = Money(Decimal("1000000.00"))  # $1M

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.55 * 1 - 0.45) / 1 = 0.1
        assert result == Decimal("0.1")

    def test_calculate_kelly_criterion_high_precision_probability(self, calculator):
        """Test Kelly calculation with high precision probability."""
        win_probability = Decimal("0.5555555555555555")  # High precision
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("90.00"))

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should handle high precision calculations
        assert isinstance(result, Decimal)
        assert result != Decimal("0")

    def test_calculate_kelly_criterion_extreme_win_loss_ratios(self, calculator):
        """Test Kelly calculation with extreme win/loss ratios."""
        win_probability = Decimal("0.51")  # Slight edge
        win_amount = Money(Decimal("1000.00"))  # Large wins
        loss_amount = Money(Decimal("1.00"))  # Small losses

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should be capped at 25% due to high ratio
        assert result == Decimal("0.25")

    # Mathematical edge cases
    def test_calculate_kelly_criterion_near_boundary_probability(self, calculator):
        """Test Kelly calculation with probabilities near boundaries."""
        # Test near 0
        win_probability = Decimal("0.001")  # 0.1% win rate
        win_amount = Money(Decimal("1000.00"))  # Need high ratio to compensate
        loss_amount = Money(Decimal("1.00"))

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.001 * 1000 - 0.999) / 1000 = 0.001 / 1000 = 0.000001
        assert result == Decimal("0.000001")

    def test_calculate_kelly_criterion_near_one_probability(self, calculator):
        """Test Kelly calculation with probability near 1."""
        win_probability = Decimal("0.999")  # 99.9% win rate
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("1000.00"))  # Large losses

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Kelly = (0.999 * 0.1 - 0.001) / 0.1 = 0.989
        # Should be capped at 25%
        assert result == Decimal("0.25")

    # Financial precision tests
    def test_calculate_kelly_criterion_decimal_precision(self, calculator):
        """Test that Kelly calculations maintain decimal precision."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("33.33"))
        loss_amount = Money(Decimal("66.67"))

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should maintain precision without floating point errors
        assert isinstance(result, Decimal)
        # Kelly = (0.6 * (33.33/66.67) - 0.4) / (33.33/66.67) ≈ 0.0999
        expected = (Decimal("0.6") * (Decimal("33.33") / Decimal("66.67")) - Decimal("0.4")) / (
            Decimal("33.33") / Decimal("66.67")
        )
        assert abs(result - expected) < Decimal("0.0001")  # Small tolerance for precision

    # Real-world scenario tests
    def test_calculate_kelly_criterion_conservative_strategy(self, calculator):
        """Test Kelly calculation for conservative trading strategy."""
        win_probability = Decimal("0.52")  # 52% win rate
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("105.00"))  # Risk slightly more than gain

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should result in small position size for conservative strategy
        assert result < Decimal("0.1")

    def test_calculate_kelly_criterion_aggressive_strategy(self, calculator):
        """Test Kelly calculation for aggressive trading strategy."""
        win_probability = Decimal("0.65")  # 65% win rate
        win_amount = Money(Decimal("150.00"))
        loss_amount = Money(Decimal("75.00"))  # 2:1 reward/risk ratio

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should result in higher position size but capped at 25%
        assert result == Decimal("0.25")

    def test_calculate_kelly_criterion_martingale_like_scenario(self, calculator):
        """Test Kelly calculation for martingale-like scenario (high loss, low win rate)."""
        win_probability = Decimal("0.49")  # 49% win rate
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("100.00"))  # 1:1 ratio

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Should be negative (unfavorable)
        # Kelly = (0.49 * 1 - 0.51) / 1 = -0.02
        assert result == Decimal("-0.02")

    # Stateless and thread safety tests
    def test_calculator_is_stateless(self, calculator):
        """Test that calculator is stateless and thread-safe."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("50.00"))

        # Run multiple calculations to ensure no state is maintained
        results = []
        for _ in range(5):
            result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)
            results.append(result)

        # All results should be identical
        assert all(r == results[0] for r in results)
        assert all(r == Decimal("0.25") for r in results)  # Expected result

    def test_concurrent_kelly_calculations_isolation(self, calculator):
        """Test that concurrent calculations don't interfere."""
        # Different parameter sets
        params1 = (Decimal("0.6"), Money(Decimal("100.00")), Money(Decimal("50.00")))
        params2 = (Decimal("0.4"), Money(Decimal("50.00")), Money(Decimal("100.00")))

        result1 = calculator.calculate_kelly_criterion(*params1)
        result2 = calculator.calculate_kelly_criterion(*params2)

        # Results should be independent and different
        assert result1 != result2
        assert result1 == Decimal("0.25")  # First should be capped
        assert result2 == Decimal("-0.8")  # Second should be negative

    # Performance test
    def test_kelly_calculation_performance(self, calculator):
        """Test Kelly calculation performance with many iterations."""
        win_probability = Decimal("0.55")
        win_amount = Money(Decimal("100.00"))
        loss_amount = Money(Decimal("90.00"))

        # Run many calculations to ensure performance is reasonable
        results = []
        for i in range(100):
            # Vary parameters slightly to prevent optimization
            prob = win_probability + Decimal(str(i * 0.0001))
            result = calculator.calculate_kelly_criterion(prob, win_amount, loss_amount)
            results.append(result)

        # Should complete without performance issues
        assert len(results) == 100
        assert all(isinstance(r, Decimal) for r in results)

    # Documentation example tests
    def test_kelly_criterion_documentation_examples(self, calculator):
        """Test Kelly criterion with examples from documentation."""
        # Conservative example (0-0.10 range)
        result_conservative = calculator.calculate_kelly_criterion(
            Decimal("0.51"), Money(Decimal("100.00")), Money(Decimal("100.00"))
        )
        assert Decimal("0") < result_conservative <= Decimal("0.10")

        # Aggressive example (0.10-0.25 range)
        result_aggressive = calculator.calculate_kelly_criterion(
            Decimal("0.7"), Money(Decimal("100.00")), Money(Decimal("50.00"))
        )
        assert result_aggressive == Decimal("0.25")  # Should be capped

    def test_kelly_formula_components_verification(self, calculator):
        """Test that Kelly formula components are calculated correctly."""
        win_probability = Decimal("0.6")
        win_amount = Money(Decimal("120.00"))
        loss_amount = Money(Decimal("80.00"))

        result = calculator.calculate_kelly_criterion(win_probability, win_amount, loss_amount)

        # Manually verify components:
        # p = 0.6, q = 0.4, b = 120/80 = 1.5
        # Kelly = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = 0.333...
        # Capped at 0.25
        expected = min(
            (
                win_probability * (win_amount.amount / loss_amount.amount)
                - (Decimal("1") - win_probability)
            )
            / (win_amount.amount / loss_amount.amount),
            Decimal("0.25"),
        )
        assert result == expected

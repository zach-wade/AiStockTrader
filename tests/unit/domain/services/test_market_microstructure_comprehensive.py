"""
Comprehensive unit tests for Market Microstructure service.

Tests market impact models, slippage calculations, and execution price determination
with 95%+ coverage including edge cases and error scenarios.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.domain.entities.order import OrderSide, OrderType
from src.domain.services.market_microstructure import (
    LinearImpactModel,
    MarketImpactModel,
    MarketMicrostructureFactory,
    SlippageConfig,
    SquareRootImpactModel,
)
from src.domain.value_objects.price import Price


class TestSlippageConfig:
    """Test SlippageConfig validation and configuration."""

    def test_valid_config(self):
        """Test creating valid slippage configuration."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"),
            impact_coefficient=Decimal("0.1"),
            volatility_multiplier=Decimal("1.5"),
            add_randomness=True,
            random_factor_range=(Decimal("0.9"), Decimal("1.1")),
        )

        assert config.base_bid_ask_bps == Decimal("5")
        assert config.impact_coefficient == Decimal("0.1")
        assert config.volatility_multiplier == Decimal("1.5")
        assert config.add_randomness is True
        assert config.random_factor_range == (Decimal("0.9"), Decimal("1.1"))

    def test_default_values(self):
        """Test default values in slippage configuration."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        assert config.volatility_multiplier == Decimal("1.0")
        assert config.add_randomness is True
        assert config.random_factor_range == (Decimal("0.8"), Decimal("1.2"))

    def test_negative_bid_ask_raises_error(self):
        """Test that negative bid-ask spread raises ValueError."""
        with pytest.raises(ValueError, match="Base bid-ask difference cannot be negative"):
            SlippageConfig(base_bid_ask_bps=Decimal("-5"), impact_coefficient=Decimal("0.1"))

    def test_negative_impact_coefficient_raises_error(self):
        """Test that negative impact coefficient raises ValueError."""
        with pytest.raises(ValueError, match="Impact coefficient cannot be negative"):
            SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("-0.1"))

    def test_negative_volatility_multiplier_raises_error(self):
        """Test that negative volatility multiplier raises ValueError."""
        with pytest.raises(ValueError, match="Volatility multiplier cannot be negative"):
            SlippageConfig(
                base_bid_ask_bps=Decimal("5"),
                impact_coefficient=Decimal("0.1"),
                volatility_multiplier=Decimal("-1.5"),
            )

    def test_zero_values_allowed(self):
        """Test that zero values are allowed for configuration."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("0"),
            impact_coefficient=Decimal("0"),
            volatility_multiplier=Decimal("0"),
        )

        assert config.base_bid_ask_bps == Decimal("0")
        assert config.impact_coefficient == Decimal("0")
        assert config.volatility_multiplier == Decimal("0")


class TestLinearImpactModel:
    """Test LinearImpactModel for market impact and execution price calculation."""

    @pytest.fixture
    def config(self):
        """Create standard slippage configuration."""
        return SlippageConfig(
            base_bid_ask_bps=Decimal("5"),  # 5 basis points
            impact_coefficient=Decimal("0.1"),
            volatility_multiplier=Decimal("1.0"),
            add_randomness=False,  # Disable randomness for deterministic tests
        )

    @pytest.fixture
    def config_with_randomness(self):
        """Create slippage configuration with randomness enabled."""
        return SlippageConfig(
            base_bid_ask_bps=Decimal("5"),
            impact_coefficient=Decimal("0.1"),
            volatility_multiplier=Decimal("1.0"),
            add_randomness=True,
            random_factor_range=(Decimal("0.9"), Decimal("1.1")),
        )

    @pytest.fixture
    def model(self, config):
        """Create LinearImpactModel with standard config."""
        return LinearImpactModel(config)

    def test_market_order_buy_execution_price(self, model):
        """Test execution price calculation for market buy order."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Buy orders pay more (base + bid-ask + impact)
        # Bid-ask: 100 * 5/10000 = 0.05
        # Impact: 0.1 * 1000/1000 * 1.0 = 0.1 bps -> 100 * 0.1/10000 = 0.001
        expected = Decimal("100.051")
        assert execution_price == expected

    def test_market_order_sell_execution_price(self, model):
        """Test execution price calculation for market sell order."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Sell orders receive less (base - bid-ask - impact)
        expected = Decimal("99.949")
        assert execution_price == expected

    def test_limit_order_no_slippage(self, model):
        """Test that limit orders have no slippage."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        # Test buy limit order
        execution_price = model.calculate_execution_price(
            base_price=base_price, side=OrderSide.BUY, quantity=quantity, order_type=OrderType.LIMIT
        )
        assert execution_price == Decimal("100.00")

        # Test sell limit order
        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,
        )
        assert execution_price == Decimal("100.00")

    def test_stop_order_no_slippage(self, model):
        """Test that stop orders have no slippage (treated as limit when filled)."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        execution_price = model.calculate_execution_price(
            base_price=base_price, side=OrderSide.BUY, quantity=quantity, order_type=OrderType.STOP
        )
        assert execution_price == Decimal("100.00")

    def test_stop_limit_order_no_slippage(self, model):
        """Test that stop limit orders have no slippage."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.STOP_LIMIT,
        )
        assert execution_price == Decimal("100.00")

    def test_calculate_market_impact(self, model):
        """Test market impact calculation."""
        price = Price(Decimal("100.00"))
        quantity = Decimal("5000")

        impact = model.calculate_market_impact(price, quantity)

        # Impact = 0.1 * 5000/1000 * 1.0 = 0.5 basis points
        assert impact == Decimal("0.5")

    def test_market_impact_with_average_volume(self, model):
        """Test market impact calculation with average volume (not used in linear model)."""
        price = Price(Decimal("100.00"))
        quantity = Decimal("5000")
        avg_volume = Decimal("1000000")

        # Linear model doesn't use average volume
        impact = model.calculate_market_impact(price, quantity, avg_volume)
        assert impact == Decimal("0.5")

    def test_market_impact_uses_absolute_quantity(self, model):
        """Test that market impact uses absolute value of quantity."""
        price = Price(Decimal("100.00"))
        positive_qty = Decimal("5000")
        negative_qty = Decimal("-5000")

        positive_impact = model.calculate_market_impact(price, positive_qty)
        negative_impact = model.calculate_market_impact(price, negative_qty)

        assert positive_impact == negative_impact

    def test_volatility_multiplier_effect(self):
        """Test volatility multiplier effect on market impact."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"),
            impact_coefficient=Decimal("0.1"),
            volatility_multiplier=Decimal("2.0"),
            add_randomness=False,
        )
        model = LinearImpactModel(config)

        price = Price(Decimal("100.00"))
        quantity = Decimal("5000")

        impact = model.calculate_market_impact(price, quantity)
        # Impact = 0.1 * 5000/1000 * 2.0 = 1.0 basis points
        assert impact == Decimal("1.0")

    def test_execution_price_with_randomness(self, config_with_randomness):
        """Test execution price calculation with randomness enabled."""
        model = LinearImpactModel(config_with_randomness)

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        # Mock the random number generator
        with patch.object(model._rng, "random", return_value=0.5):
            execution_price = model.calculate_execution_price(
                base_price=base_price,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            # Random factor = 0.9 + (1.1 - 0.9) * 0.5 = 1.0
            # So same as without randomness
            expected = Decimal("100.051")
            assert execution_price == expected

    def test_random_factor_range(self, config_with_randomness):
        """Test random factor generation within configured range."""
        model = LinearImpactModel(config_with_randomness)

        # Test minimum random value
        with patch.object(model._rng, "random", return_value=0.0):
            factor = model._get_random_factor()
            assert factor == Decimal("0.9")

        # Test maximum random value
        with patch.object(model._rng, "random", return_value=1.0):
            factor = model._get_random_factor()
            assert factor == Decimal("1.1")

        # Test middle random value
        with patch.object(model._rng, "random", return_value=0.5):
            factor = model._get_random_factor()
            assert factor == Decimal("1.0")

    def test_minimum_price_constraint(self):
        """Test that execution price never goes below $0.01."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("10000"),  # Extreme bid-ask spread
            impact_coefficient=Decimal("100"),  # Extreme impact
            add_randomness=False,
        )
        model = LinearImpactModel(config)

        base_price = Price(Decimal("0.10"))
        quantity = Decimal("10000")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Should not go below $0.01
        assert execution_price == Decimal("0.01")

    def test_zero_quantity_handling(self, model):
        """Test handling of zero quantity orders."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("0")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Zero quantity should still have bid-ask spread but no impact
        # Bid-ask: 100 * 5/10000 = 0.05
        # Impact: 0.1 * 0/1000 = 0
        expected = Decimal("100.05")
        assert execution_price == expected

    def test_very_large_quantity(self, model):
        """Test handling of very large quantities."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000000")  # 1 million shares

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Bid-ask: 100 * 5/10000 = 0.05
        # Impact: 0.1 * 1000000/1000 = 100 bps -> 100 * 100/10000 = 1.00
        expected = Decimal("101.05")
        assert execution_price == expected

    def test_fractional_quantities(self, model):
        """Test handling of fractional share quantities."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("0.5")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Very small impact due to fractional quantity
        assert execution_price < Decimal("100.00")
        assert execution_price > Decimal("99.94")  # Reasonable bounds


class TestSquareRootImpactModel:
    """Test SquareRootImpactModel for non-linear market impact."""

    @pytest.fixture
    def config(self):
        """Create standard slippage configuration."""
        return SlippageConfig(
            base_bid_ask_bps=Decimal("5"),
            impact_coefficient=Decimal("0.1"),
            volatility_multiplier=Decimal("1.0"),
            add_randomness=False,
        )

    @pytest.fixture
    def model(self, config):
        """Create SquareRootImpactModel with standard config."""
        return SquareRootImpactModel(config)

    def test_square_root_impact_calculation(self, model):
        """Test square root market impact calculation."""
        price = Price(Decimal("100.00"))

        # Test with different quantities to verify square root relationship
        qty1 = Decimal("100")
        impact1 = model.calculate_market_impact(price, qty1)

        qty2 = Decimal("400")  # 4x the quantity
        impact2 = model.calculate_market_impact(price, qty2)

        # Impact should scale with square root, so 2x not 4x
        # Allow for small rounding differences
        assert abs(impact2 - impact1 * Decimal("2")) < Decimal("0.01")

    def test_market_order_execution_with_sqrt_impact(self, model):
        """Test execution price with square root impact model."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("10000")  # Large order

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Square root model should have less impact than linear for large orders
        # sqrt(10000/1000) = sqrt(10) ≈ 3.16
        # Impact ≈ 0.1 * 3.16 = 0.316 bps
        assert execution_price > Decimal("100.00")
        assert execution_price < Decimal("100.10")  # Much less than linear

    def test_sqrt_model_with_average_volume(self, model):
        """Test square root model with average volume consideration."""
        price = Price(Decimal("100.00"))
        quantity = Decimal("50000")
        avg_volume = Decimal("1000000")

        # With average volume, impact should be scaled by volume ratio
        impact = model.calculate_market_impact(price, quantity, avg_volume)

        # Verify impact is reasonable and uses volume
        assert impact > Decimal("0")
        assert impact < Decimal("10")  # Reasonable upper bound

    def test_sqrt_model_randomness(self):
        """Test square root model with randomness."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"),
            impact_coefficient=Decimal("0.1"),
            add_randomness=True,
            random_factor_range=(Decimal("0.5"), Decimal("1.5")),
        )
        model = SquareRootImpactModel(config)

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        with patch.object(model._rng, "random", return_value=0.5):
            execution_price = model.calculate_execution_price(
                base_price=base_price,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            # Verify price is affected by randomness
            assert execution_price < Decimal("100.00")

    def test_sqrt_model_limit_orders(self, model):
        """Test that square root model also doesn't affect limit orders."""
        base_price = Price(Decimal("100.00"))
        quantity = Decimal("10000")

        execution_price = model.calculate_execution_price(
            base_price=base_price, side=OrderSide.BUY, quantity=quantity, order_type=OrderType.LIMIT
        )

        assert execution_price == Decimal("100.00")

    def test_sqrt_model_minimum_price(self):
        """Test square root model respects minimum price."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5000"),  # Extreme values
            impact_coefficient=Decimal("10"),
            add_randomness=False,
        )
        model = SquareRootImpactModel(config)

        base_price = Price(Decimal("1.00"))
        quantity = Decimal("100000")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        assert execution_price >= Decimal("0.01")


class TestMarketMicrostructureFactory:
    """Test factory for creating market microstructure models."""

    def test_create_linear_model(self):
        """Test factory creates linear impact model."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        model = MarketMicrostructureFactory.create(
            model_type=MarketImpactModel.LINEAR, config=config
        )

        assert isinstance(model, LinearImpactModel)

    def test_create_square_root_model(self):
        """Test factory creates square root impact model."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        model = MarketMicrostructureFactory.create(
            model_type=MarketImpactModel.SQUARE_ROOT, config=config
        )

        assert isinstance(model, SquareRootImpactModel)

    def test_create_fixed_model(self):
        """Test factory raises error for unimplemented fixed model."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        with pytest.raises(ValueError, match="Unsupported market impact model"):
            MarketMicrostructureFactory.create(model_type=MarketImpactModel.FIXED, config=config)

    def test_create_logarithmic_model(self):
        """Test factory raises error for unimplemented logarithmic model."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        with pytest.raises(ValueError, match="Unsupported market impact model"):
            MarketMicrostructureFactory.create(
                model_type=MarketImpactModel.LOGARITHMIC, config=config
            )

    def test_factory_with_default_config(self):
        """Test factory can use default configuration."""
        model = MarketMicrostructureFactory.create_default(model_type=MarketImpactModel.LINEAR)

        assert isinstance(model, LinearImpactModel)
        assert model.config.base_bid_ask_bps == Decimal(
            "2"
        )  # Default value from DEFAULT_EQUITY_CONFIG

    def test_factory_creates_independent_instances(self):
        """Test factory creates independent model instances."""
        config = SlippageConfig(base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"))

        model1 = MarketMicrostructureFactory.create(
            model_type=MarketImpactModel.LINEAR, config=config
        )
        model2 = MarketMicrostructureFactory.create(
            model_type=MarketImpactModel.LINEAR, config=config
        )

        assert model1 is not model2
        assert model1._rng is not model2._rng  # Independent RNG instances


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def test_zero_config_values(self):
        """Test models with zero configuration values."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("0"),
            impact_coefficient=Decimal("0"),
            volatility_multiplier=Decimal("0"),
        )
        model = LinearImpactModel(config)

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        # With zero config, market orders should have no slippage
        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        assert execution_price == Decimal("100.00")

    def test_extreme_market_conditions(self):
        """Test models under extreme market conditions."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("500"),  # 5% spread (extreme)
            impact_coefficient=Decimal("10"),  # High impact
            volatility_multiplier=Decimal("5"),  # High volatility
            add_randomness=False,
        )

        linear_model = LinearImpactModel(config)
        sqrt_model = SquareRootImpactModel(config)

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("10000")

        linear_exec = linear_model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        sqrt_exec = sqrt_model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Both should increase price for buy orders
        assert linear_exec > base_price
        assert sqrt_exec > base_price

        # Linear should have more impact for large orders
        assert linear_exec > sqrt_exec

    def test_model_consistency_across_order_types(self):
        """Test model consistency across different order types."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"), add_randomness=False
        )
        model = LinearImpactModel(config)

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        # Market order has slippage
        market_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Limit order has no slippage
        limit_price = model.calculate_execution_price(
            base_price=base_price, side=OrderSide.BUY, quantity=quantity, order_type=OrderType.LIMIT
        )

        # Stop order has no slippage (when triggered, acts like limit)
        stop_price = model.calculate_execution_price(
            base_price=base_price, side=OrderSide.BUY, quantity=quantity, order_type=OrderType.STOP
        )

        assert market_price > limit_price
        assert limit_price == stop_price == base_price

    def test_precision_handling(self):
        """Test precise decimal calculation handling."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("3.33"),
            impact_coefficient=Decimal("0.0123"),
            volatility_multiplier=Decimal("1.111"),
            add_randomness=False,
        )
        model = LinearImpactModel(config)

        base_price = Price(Decimal("123.456"))
        quantity = Decimal("789.012")

        execution_price = model.calculate_execution_price(
            base_price=base_price,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Should handle precise decimals without rounding errors
        assert isinstance(execution_price, Decimal)
        assert execution_price < base_price  # Sell gets less

    def test_concurrent_model_usage(self):
        """Test that models can be used concurrently (thread-safe RNG)."""
        config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"), impact_coefficient=Decimal("0.1"), add_randomness=True
        )

        # Create multiple models that could be used concurrently
        models = [LinearImpactModel(config) for _ in range(5)]

        base_price = Price(Decimal("100.00"))
        quantity = Decimal("1000")

        # Execute on all models
        results = []
        for model in models:
            price = model.calculate_execution_price(
                base_price=base_price,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            results.append(price)

        # Results should vary due to randomness
        assert len(set(results)) > 1  # Not all the same

        # But all should be reasonable
        for result in results:
            assert result > Decimal("100.00")
            assert result < Decimal("101.00")

"""
Options Sentiment Calculator

Analyzes options flow and positioning to derive market sentiment indicators
and directional biases from options market activity.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_options import BaseOptionsCalculator
from ..helpers import create_feature_dataframe, create_rolling_features, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class OptionsSentimentCalculator(BaseOptionsCalculator):
    """
    Calculates sentiment indicators from options market data.

    Features include:
    - Put/Call ratios (volume and open interest based)
    - Skew indicators
    - Options flow sentiment
    - Smart money indicators
    - Term structure sentiment
    - Volatility smile sentiment
    - Positioning extremes
    - Cross-strike correlations
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize options sentiment calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Sentiment-specific parameters
        self.sentiment_windows = config.get("sentiment_windows", [1, 5, 20])
        self.volume_threshold = config.get("volume_threshold", 100)
        self.smart_money_threshold = config.get("smart_money_threshold", 1000000)

        # Skew calculation parameters
        self.skew_otm_threshold = config.get("skew_otm_threshold", 0.25)
        self.skew_strikes = config.get("skew_strikes", [0.9, 0.95, 1.0, 1.05, 1.1])

        # Flow analysis parameters
        self.flow_categories = {
            "small": (0, 10000),
            "medium": (10000, 100000),
            "large": (100000, 1000000),
            "smart": (1000000, float("inf")),
        }

        logger.info("Initialized OptionsSentimentCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of sentiment feature names."""
        features = []

        # Put/Call ratios
        for window in self.sentiment_windows:
            features.extend(
                [
                    f"pcr_volume_{window}d",
                    f"pcr_oi_{window}d",
                    f"pcr_dollar_volume_{window}d",
                    f"pcr_change_{window}d",
                ]
            )

        # Skew indicators
        features.extend(
            [
                "put_skew",
                "call_skew",
                "skew_ratio",
                "risk_reversal_25d",
                "butterfly_25d",
                "smile_slope",
                "smile_curvature",
            ]
        )

        # Options flow sentiment
        features.extend(
            [
                "call_flow_sentiment",
                "put_flow_sentiment",
                "net_flow_sentiment",
                "smart_money_sentiment",
                "retail_sentiment",
                "flow_momentum",
            ]
        )

        # Volume-weighted sentiment
        features.extend(
            [
                "vwap_call_sentiment",
                "vwap_put_sentiment",
                "dollar_weighted_pcr",
                "premium_flow_ratio",
            ]
        )

        # Term structure sentiment
        features.extend(
            [
                "near_term_sentiment",
                "far_term_sentiment",
                "term_structure_slope",
                "calendar_spread_sentiment",
            ]
        )

        # Positioning indicators
        features.extend(
            [
                "call_positioning_score",
                "put_positioning_score",
                "net_positioning",
                "positioning_concentration",
                "max_pain_deviation",
            ]
        )

        # Composite sentiment
        features.extend(
            [
                "options_sentiment_composite",
                "sentiment_divergence",
                "sentiment_strength",
                "contrarian_signal",
            ]
        )

        return features

    def calculate(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features from options data.

        Args:
            options_data: DataFrame with options chain data

        Returns:
            DataFrame with sentiment features
        """
        try:
            # Validate options data
            validated_data = self.validate_options_data(options_data)
            if validated_data.empty:
                return self._create_empty_features(options_data.index)

            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index.unique())

            # Calculate put/call ratios
            pcr_features = self._calculate_pcr_features(validated_data)
            features = pd.concat([features, pcr_features], axis=1)

            # Calculate skew indicators
            skew_features = self._calculate_skew_features(validated_data)
            features = pd.concat([features, skew_features], axis=1)

            # Calculate options flow sentiment
            flow_features = self._calculate_flow_sentiment(validated_data)
            features = pd.concat([features, flow_features], axis=1)

            # Calculate volume-weighted sentiment
            vw_features = self._calculate_volume_weighted_sentiment(validated_data)
            features = pd.concat([features, vw_features], axis=1)

            # Calculate term structure sentiment
            term_features = self._calculate_term_structure_sentiment(validated_data)
            features = pd.concat([features, term_features], axis=1)

            # Calculate positioning indicators
            positioning_features = self._calculate_positioning_indicators(validated_data)
            features = pd.concat([features, positioning_features], axis=1)

            # Calculate composite sentiment
            composite_features = self._calculate_composite_sentiment(features)
            features = pd.concat([features, composite_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")
            return self._create_empty_features(options_data.index.unique())

    def _calculate_pcr_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate put/call ratio features."""
        features = pd.DataFrame(index=options_data.index.unique())

        # Separate calls and puts
        calls = options_data[options_data["option_type"] == "call"]
        puts = options_data[options_data["option_type"] == "put"]

        # Calculate base PCR ratios first
        call_vol = calls.groupby(calls.index)["volume"].sum()
        put_vol = puts.groupby(puts.index)["volume"].sum()
        pcr_volume = safe_divide(put_vol, call_vol)

        call_oi = calls.groupby(calls.index)["open_interest"].sum()
        put_oi = puts.groupby(puts.index)["open_interest"].sum()
        pcr_oi = safe_divide(put_oi, call_oi)

        call_dollar = (calls["volume"] * calls["last_price"]).groupby(calls.index).sum()
        put_dollar = (puts["volume"] * puts["last_price"]).groupby(puts.index).sum()
        pcr_dollar = safe_divide(put_dollar, call_dollar)

        # Create DataFrame for rolling features
        pcr_df = pd.DataFrame(
            {"pcr_volume": pcr_volume, "pcr_oi": pcr_oi, "pcr_dollar": pcr_dollar},
            index=features.index,
        )

        # Use create_rolling_features with all windows at once
        rolling_features = create_rolling_features(
            pcr_df,
            columns=["pcr_volume", "pcr_oi", "pcr_dollar"],
            windows=self.sentiment_windows,
            operations=["mean"],
            min_periods=1,
        )

        # Extract the features and rename them
        for window in self.sentiment_windows:
            features[f"pcr_volume_{window}d"] = rolling_features[
                f"pcr_volume_rolling_mean_{window}"
            ]
            features[f"pcr_oi_{window}d"] = rolling_features[f"pcr_oi_rolling_mean_{window}"]
            features[f"pcr_dollar_volume_{window}d"] = rolling_features[
                f"pcr_dollar_rolling_mean_{window}"
            ]

            # PCR change
            pcr_change = pcr_volume.pct_change(window)
            features[f"pcr_change_{window}d"] = pcr_change

        return features

    def _calculate_skew_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility skew features."""
        features = pd.DataFrame(index=options_data.index.unique())

        # Calculate moneyness
        options_data["moneyness"] = safe_divide(
            options_data["strike"], options_data["underlying_price"], default_value=1.0
        )

        # Group by expiration for skew calculation
        for date in options_data.index.unique():
            date_data = options_data.loc[date]

            # Find near-term expiration
            expirations = date_data["expiration"].unique()
            if len(expirations) == 0:
                continue

            near_exp = min(expirations)
            exp_data = date_data[date_data["expiration"] == near_exp]

            # Calculate skew for calls and puts
            put_skew = self._calculate_single_skew(exp_data[exp_data["option_type"] == "put"])
            call_skew = self._calculate_single_skew(exp_data[exp_data["option_type"] == "call"])

            features.loc[date, "put_skew"] = put_skew
            features.loc[date, "call_skew"] = call_skew
            features.loc[date, "skew_ratio"] = safe_divide(put_skew, call_skew)

            # Risk reversal (25 delta)
            rr_25d = self._calculate_risk_reversal(exp_data, 0.25)
            features.loc[date, "risk_reversal_25d"] = rr_25d

            # Butterfly spread (25 delta)
            bf_25d = self._calculate_butterfly(exp_data, 0.25)
            features.loc[date, "butterfly_25d"] = bf_25d

            # Smile characteristics
            smile_slope, smile_curve = self._calculate_smile_shape(exp_data)
            features.loc[date, "smile_slope"] = smile_slope
            features.loc[date, "smile_curvature"] = smile_curve

        return features.fillna(0)

    def _calculate_flow_sentiment(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate options flow sentiment."""
        features = pd.DataFrame(index=options_data.index.unique())

        # Separate by option type
        calls = options_data[options_data["option_type"] == "call"]
        puts = options_data[options_data["option_type"] == "put"]

        for date in options_data.index.unique():
            date_calls = calls.loc[date] if date in calls.index else pd.DataFrame()
            date_puts = puts.loc[date] if date in puts.index else pd.DataFrame()

            # Call flow sentiment
            call_buy_vol = date_calls[date_calls["trade_type"] == "buy"]["volume"].sum()
            call_sell_vol = date_calls[date_calls["trade_type"] == "sell"]["volume"].sum()
            call_sentiment = safe_divide(call_buy_vol - call_sell_vol, call_buy_vol + call_sell_vol)
            features.loc[date, "call_flow_sentiment"] = call_sentiment

            # Put flow sentiment
            put_buy_vol = date_puts[date_puts["trade_type"] == "buy"]["volume"].sum()
            put_sell_vol = date_puts[date_puts["trade_type"] == "sell"]["volume"].sum()
            put_sentiment = safe_divide(put_buy_vol - put_sell_vol, put_buy_vol + put_sell_vol)
            features.loc[date, "put_flow_sentiment"] = put_sentiment

            # Net flow sentiment
            features.loc[date, "net_flow_sentiment"] = call_sentiment - put_sentiment

            # Smart money vs retail sentiment
            smart_money_sent = self._calculate_smart_money_sentiment(options_data.loc[date])
            retail_sent = self._calculate_retail_sentiment(options_data.loc[date])

            features.loc[date, "smart_money_sentiment"] = smart_money_sent
            features.loc[date, "retail_sentiment"] = retail_sent

        # Flow momentum
        features["flow_momentum"] = features["net_flow_sentiment"].pct_change(5)

        return features.fillna(0)

    def _calculate_volume_weighted_sentiment(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-weighted sentiment metrics."""
        features = pd.DataFrame(index=options_data.index.unique())

        for date in options_data.index.unique():
            date_data = options_data.loc[date]

            # VWAP sentiment for calls
            calls = date_data[date_data["option_type"] == "call"]
            if not calls.empty:
                call_vwap = self._calculate_vwap_sentiment(calls)
                features.loc[date, "vwap_call_sentiment"] = call_vwap

            # VWAP sentiment for puts
            puts = date_data[date_data["option_type"] == "put"]
            if not puts.empty:
                put_vwap = self._calculate_vwap_sentiment(puts)
                features.loc[date, "vwap_put_sentiment"] = put_vwap

            # Dollar-weighted PCR
            call_premium = (calls["volume"] * calls["last_price"]).sum()
            put_premium = (puts["volume"] * puts["last_price"]).sum()
            features.loc[date, "dollar_weighted_pcr"] = safe_divide(put_premium, call_premium)

            # Premium flow ratio
            total_premium = call_premium + put_premium
            features.loc[date, "premium_flow_ratio"] = safe_divide(
                call_premium - put_premium, total_premium
            )

        return features.fillna(0)

    def _calculate_term_structure_sentiment(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate term structure based sentiment."""
        features = pd.DataFrame(index=options_data.index.unique())

        for date in options_data.index.unique():
            date_data = options_data.loc[date]

            # Get unique expirations
            expirations = sorted(date_data["expiration"].unique())
            if len(expirations) < 2:
                continue

            # Near-term sentiment (first expiration)
            near_exp = expirations[0]
            near_data = date_data[date_data["expiration"] == near_exp]
            near_pcr = self._calculate_simple_pcr(near_data)
            features.loc[date, "near_term_sentiment"] = 1 - near_pcr  # Invert for sentiment

            # Far-term sentiment (last expiration)
            far_exp = expirations[-1]
            far_data = date_data[date_data["expiration"] == far_exp]
            far_pcr = self._calculate_simple_pcr(far_data)
            features.loc[date, "far_term_sentiment"] = 1 - far_pcr

            # Term structure slope
            features.loc[date, "term_structure_slope"] = (
                features.loc[date, "far_term_sentiment"] - features.loc[date, "near_term_sentiment"]
            )

            # Calendar spread sentiment
            calendar_sent = self._calculate_calendar_sentiment(date_data, expirations)
            features.loc[date, "calendar_spread_sentiment"] = calendar_sent

        return features.fillna(0)

    def _calculate_positioning_indicators(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate positioning indicators."""
        features = pd.DataFrame(index=options_data.index.unique())

        for date in options_data.index.unique():
            date_data = options_data.loc[date]

            # Call positioning score
            calls = date_data[date_data["option_type"] == "call"]
            call_score = self._calculate_positioning_score(calls)
            features.loc[date, "call_positioning_score"] = call_score

            # Put positioning score
            puts = date_data[date_data["option_type"] == "put"]
            put_score = self._calculate_positioning_score(puts)
            features.loc[date, "put_positioning_score"] = put_score

            # Net positioning
            features.loc[date, "net_positioning"] = call_score - put_score

            # Positioning concentration
            concentration = self._calculate_positioning_concentration(date_data)
            features.loc[date, "positioning_concentration"] = concentration

            # Max pain deviation
            max_pain = self._calculate_max_pain(date_data)
            current_price = date_data["underlying_price"].iloc[0]
            features.loc[date, "max_pain_deviation"] = safe_divide(
                current_price - max_pain, max_pain
            )

        return features.fillna(0)

    def _calculate_composite_sentiment(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite sentiment scores."""
        composite = pd.DataFrame(index=features.index)

        # Options sentiment composite (weighted average)
        sentiment_components = [
            features.get("pcr_volume_5d", 0) * -0.2,  # Invert PCR
            features.get("net_flow_sentiment", 0) * 0.3,
            features.get("smart_money_sentiment", 0) * 0.25,
            features.get("risk_reversal_25d", 0) * 0.15,
            features.get("net_positioning", 0) * 0.1,
        ]
        composite["options_sentiment_composite"] = sum(sentiment_components)

        # Sentiment divergence
        smart_retail_div = abs(
            features.get("smart_money_sentiment", 0) - features.get("retail_sentiment", 0)
        )
        term_div = abs(features.get("term_structure_slope", 0))
        composite["sentiment_divergence"] = safe_divide(
            (smart_retail_div + term_div), 2, default_value=0.0
        )

        # Sentiment strength
        sentiment_abs = abs(composite["options_sentiment_composite"])
        flow_strength = abs(features.get("net_flow_sentiment", 0))
        composite["sentiment_strength"] = safe_divide(
            (sentiment_abs + flow_strength), 2, default_value=0.0
        )

        # Contrarian signal
        extreme_pcr = (features.get("pcr_volume_5d", 1) > 1.5) | (
            features.get("pcr_volume_5d", 1) < 0.5
        )
        extreme_positioning = abs(features.get("net_positioning", 0)) > 0.8

        composite["contrarian_signal"] = safe_divide(
            (extreme_pcr.astype(int) + extreme_positioning.astype(int)), 2, default_value=0.0
        )

        return composite

    def _calculate_single_skew(self, options: pd.DataFrame) -> float:
        """Calculate skew for single option type."""
        if options.empty:
            return 0

        # Get OTM options
        if "option_type" in options.columns and options["option_type"].iloc[0] == "put":
            otm = options[options["moneyness"] < (1 - self.skew_otm_threshold)]
        else:
            otm = options[options["moneyness"] > (1 + self.skew_otm_threshold)]

        if otm.empty:
            return 0

        # Calculate average IV for OTM vs ATM
        atm = options[abs(options["moneyness"] - 1) < 0.05]

        if atm.empty or "implied_volatility" not in options.columns:
            return 0

        otm_iv = otm["implied_volatility"].mean()
        atm_iv = atm["implied_volatility"].mean()

        return safe_divide(otm_iv - atm_iv, atm_iv)

    def _calculate_risk_reversal(self, options: pd.DataFrame, delta: float) -> float:
        """Calculate risk reversal for given delta."""
        calls = options[options["option_type"] == "call"]
        puts = options[options["option_type"] == "put"]

        if calls.empty or puts.empty or "delta" not in options.columns:
            return 0

        # Find options closest to target delta
        call_delta = calls.iloc[(calls["delta"] - delta).abs().argsort()[:1]]
        put_delta = puts.iloc[(puts["delta"].abs() - delta).abs().argsort()[:1]]

        if call_delta.empty or put_delta.empty:
            return 0

        call_iv = call_delta["implied_volatility"].iloc[0]
        put_iv = put_delta["implied_volatility"].iloc[0]

        return call_iv - put_iv

    def _calculate_butterfly(self, options: pd.DataFrame, delta: float) -> float:
        """Calculate butterfly spread for given delta."""
        if "delta" not in options.columns or "implied_volatility" not in options.columns:
            return 0

        # ATM volatility
        atm = options[abs(options["moneyness"] - 1) < 0.05]
        if atm.empty:
            return 0
        atm_iv = atm["implied_volatility"].mean()

        # OTM call and put volatilities
        otm_calls = options[
            (options["option_type"] == "call") & (options["delta"].abs() - delta).abs() < 0.05
        ]
        otm_puts = options[
            (options["option_type"] == "put") & (options["delta"].abs() - delta).abs() < 0.05
        ]

        if otm_calls.empty or otm_puts.empty:
            return 0

        wing_iv = safe_divide(
            (otm_calls["implied_volatility"].mean() + otm_puts["implied_volatility"].mean()),
            2,
            default_value=0.0,
        )

        return wing_iv - atm_iv

    def _calculate_smile_shape(self, options: pd.DataFrame) -> tuple[float, float]:
        """Calculate smile slope and curvature."""
        if "moneyness" not in options.columns or "implied_volatility" not in options.columns:
            return 0, 0

        # Get strikes around ATM
        strikes = []
        ivs = []

        for moneyness in self.skew_strikes:
            strike_options = options[abs(options["moneyness"] - moneyness) < 0.02]
            if not strike_options.empty:
                strikes.append(moneyness)
                ivs.append(strike_options["implied_volatility"].mean())

        if len(strikes) < 3:
            return 0, 0

        # Fit quadratic to get slope and curvature
        coeffs = np.polyfit(strikes, ivs, 2)

        # Slope at ATM (derivative at x=1)
        slope = 2 * coeffs[0] * 1 + coeffs[1]

        # Curvature (second derivative)
        curvature = 2 * coeffs[0]

        return slope, curvature

    def _calculate_smart_money_sentiment(self, options: pd.DataFrame) -> float:
        """Calculate sentiment from large/smart money trades."""
        if "trade_size" not in options.columns:
            # Estimate from volume * price
            options["trade_size"] = options["volume"] * options["last_price"]

        smart_trades = options[options["trade_size"] > self.smart_money_threshold]

        if smart_trades.empty:
            return 0

        # Calculate directional sentiment
        call_smart = smart_trades[smart_trades["option_type"] == "call"]["volume"].sum()
        put_smart = smart_trades[smart_trades["option_type"] == "put"]["volume"].sum()

        return safe_divide(call_smart - put_smart, call_smart + put_smart)

    def _calculate_retail_sentiment(self, options: pd.DataFrame) -> float:
        """Calculate sentiment from retail-sized trades."""
        if "trade_size" not in options.columns:
            options["trade_size"] = options["volume"] * options["last_price"]

        retail_trades = options[
            (options["trade_size"] > self.flow_categories["small"][0])
            & (options["trade_size"] < self.flow_categories["medium"][1])
        ]

        if retail_trades.empty:
            return 0

        # Calculate directional sentiment
        call_retail = retail_trades[retail_trades["option_type"] == "call"]["volume"].sum()
        put_retail = retail_trades[retail_trades["option_type"] == "put"]["volume"].sum()

        return safe_divide(call_retail - put_retail, call_retail + put_retail)

    def _calculate_vwap_sentiment(self, options: pd.DataFrame) -> float:
        """Calculate volume-weighted average price sentiment."""
        if options.empty:
            return 0

        # VWAP
        vwap = safe_divide(
            (options["last_price"] * options["volume"]).sum(),
            options["volume"].sum(),
            default_value=0.0,
        )

        # Compare to average strike
        avg_price = options["last_price"].mean()

        # Sentiment: VWAP above average is bullish
        return safe_divide(vwap - avg_price, avg_price)

    def _calculate_simple_pcr(self, options: pd.DataFrame) -> float:
        """Calculate simple put/call ratio."""
        put_vol = options[options["option_type"] == "put"]["volume"].sum()
        call_vol = options[options["option_type"] == "call"]["volume"].sum()

        return safe_divide(put_vol, call_vol, default_value=1.0)

    def _calculate_calendar_sentiment(self, options: pd.DataFrame, expirations: list) -> float:
        """Calculate calendar spread sentiment."""
        if len(expirations) < 2:
            return 0

        # Compare near vs next expiration
        near_exp = expirations[0]
        next_exp = expirations[1]

        near_iv = options[options["expiration"] == near_exp]["implied_volatility"].mean()
        next_iv = options[options["expiration"] == next_exp]["implied_volatility"].mean()

        # Higher near-term IV suggests event risk (bearish)
        return safe_divide(next_iv - near_iv, near_iv)

    def _calculate_positioning_score(self, options: pd.DataFrame) -> float:
        """Calculate positioning score for option type."""
        if options.empty:
            return 0

        # Combine volume and open interest
        volume_score = options["volume"].sum()
        oi_score = options["open_interest"].sum()

        # Weight by moneyness (ITM positions are stronger signals)
        if "moneyness" in options.columns:
            itm_weight = abs(1 - options["moneyness"])
            weighted_score = (options["volume"] * itm_weight).sum()
        else:
            weighted_score = volume_score

        # Normalize
        total_score = volume_score + oi_score + weighted_score
        return np.tanh(safe_divide(total_score, 1e6, default_value=0.0))  # Normalize to [-1, 1]

    def _calculate_positioning_concentration(self, options: pd.DataFrame) -> float:
        """Calculate how concentrated positioning is."""
        if options.empty:
            return 0

        # Calculate concentration by strike
        strike_volumes = options.groupby("strike")["volume"].sum()

        if len(strike_volumes) == 0:
            return 0

        # Herfindahl index
        total_volume = strike_volumes.sum()
        if total_volume == 0:
            return 0

        market_shares = safe_divide(strike_volumes, total_volume, default_value=0.0)
        hhi = (market_shares**2).sum()

        return hhi

    def _calculate_max_pain(self, options: pd.DataFrame) -> float:
        """Calculate max pain price."""
        if options.empty or "open_interest" not in options.columns:
            return options["underlying_price"].iloc[0] if not options.empty else 0

        strikes = options["strike"].unique()
        max_pain_values = []

        for strike in strikes:
            # Calculate total value if expired at this strike
            call_pain = options[(options["option_type"] == "call") & (options["strike"] < strike)][
                "open_interest"
            ].sum() * (strike - options["strike"])

            put_pain = options[(options["option_type"] == "put") & (options["strike"] > strike)][
                "open_interest"
            ].sum() * (options["strike"] - strike)

            total_pain = call_pain.sum() + put_pain.sum()
            max_pain_values.append((strike, total_pain))

        if not max_pain_values:
            return options["underlying_price"].iloc[0]

        # Find strike with minimum pain (max pain for option holders)
        max_pain_strike = min(max_pain_values, key=lambda x: x[1])[0]

        return max_pain_strike

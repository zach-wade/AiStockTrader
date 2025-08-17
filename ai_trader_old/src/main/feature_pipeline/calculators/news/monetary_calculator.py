"""
Monetary Impact Calculator

Analyzes monetary and financial impact from news articles, including
price targets, analyst estimates, earnings guidance, and other quantitative
financial information mentioned in news.
"""

# Standard library imports
import re
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_news import BaseNewsCalculator
from ..helpers import create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MonetaryImpactCalculator(BaseNewsCalculator):
    """
    Extracts and analyzes monetary information from news articles.

    Features include:
    - Price target extraction and analysis
    - Earnings estimate changes
    - Revenue guidance impact
    - Analyst rating changes with monetary implications
    - Deal values and M&A impacts
    - Dividend and buyback announcements
    - Monetary policy implications
    - Quantitative financial metrics mentioned
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize monetary impact calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Monetary-specific configuration
        self.price_target_windows = config.get("price_target_windows", [1, 7, 30])
        self.estimate_change_threshold = config.get("estimate_change_threshold", 0.05)

        # Regular expressions for monetary extraction
        self.monetary_patterns = {
            "price_target": r"price\s+target\s+(?:of\s+)?(?:\$|USD)?\s*(\d+(?:\.\d+)?)",
            "earnings_estimate": r"earnings\s+(?:estimate|expected|forecast)\s+(?:of\s+)?(?:\$|USD)?\s*(\d+(?:\.\d+)?)",
            "revenue_estimate": r"revenue\s+(?:estimate|expected|forecast)\s+(?:of\s+)?(?:\$|USD)?\s*(\d+(?:\.\d+)?)\s*(?:billion|million)?",
            "percentage_change": r"(?:up|down|increase|decrease|rise|fall)\s+(\d+(?:\.\d+)?)\s*(?:%|percent)",
            "deal_value": r"(?:deal|transaction|acquisition)\s+(?:valued?\s+at|worth)\s+(?:\$|USD)?\s*(\d+(?:\.\d+)?)\s*(?:billion|million)?",
            "dividend": r"dividend\s+(?:of\s+)?(?:\$|USD)?\s*(\d+(?:\.\d+)?)\s*(?:per\s+share)?",
            "buyback": r"buyback\s+(?:of\s+)?(?:\$|USD)?\s*(\d+(?:\.\d+)?)\s*(?:billion|million)?",
        }

        # Analyst action mappings
        self.analyst_actions = {
            "upgrade": {"weight": 1.0, "impact": 0.3},
            "downgrade": {"weight": -1.0, "impact": -0.3},
            "initiate": {"weight": 0.5, "impact": 0.1},
            "reiterate": {"weight": 0.2, "impact": 0.05},
            "raise": {"weight": 0.8, "impact": 0.2},
            "lower": {"weight": -0.8, "impact": -0.2},
        }

        # Rating mappings
        self.rating_scores = {
            "strong buy": 1.0,
            "buy": 0.7,
            "outperform": 0.5,
            "hold": 0.0,
            "neutral": 0.0,
            "underperform": -0.5,
            "sell": -0.7,
            "strong sell": -1.0,
        }

        logger.info("Initialized MonetaryImpactCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of monetary impact feature names."""
        features = []

        # Price target features
        for window in self.price_target_windows:
            features.extend(
                [
                    f"price_target_mean_{window}d",
                    f"price_target_std_{window}d",
                    f"price_target_change_{window}d",
                    f"price_target_consensus_{window}d",
                    f"price_target_count_{window}d",
                ]
            )

        # Earnings and revenue estimates
        features.extend(
            [
                "earnings_estimate_mean",
                "earnings_estimate_change",
                "earnings_revision_direction",
                "earnings_revision_magnitude",
                "revenue_estimate_mean",
                "revenue_estimate_change",
                "revenue_revision_direction",
            ]
        )

        # Analyst actions
        features.extend(
            [
                "analyst_upgrade_score",
                "analyst_downgrade_score",
                "analyst_net_score",
                "analyst_action_intensity",
                "rating_average",
                "rating_change",
            ]
        )

        # Deal and corporate actions
        features.extend(
            [
                "deal_value_total",
                "deal_count",
                "avg_deal_size",
                "dividend_announced",
                "dividend_change",
                "buyback_value",
                "corporate_action_score",
            ]
        )

        # Monetary policy and macro
        features.extend(
            [
                "rate_hike_probability",
                "rate_cut_probability",
                "monetary_policy_score",
                "inflation_mentions",
                "gdp_growth_mentions",
            ]
        )

        # Composite scores
        features.extend(
            [
                "monetary_impact_score",
                "estimate_revision_score",
                "analyst_consensus_score",
                "quantitative_news_ratio",
            ]
        )

        return features

    def calculate(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monetary impact features from news data.

        Args:
            news_data: DataFrame with news articles

        Returns:
            DataFrame with monetary impact features
        """
        try:
            # Validate and prepare data
            validated_data = self.validate_and_prepare_data(news_data)
            if validated_data.empty:
                return self._create_empty_features(news_data.index)

            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)

            # Extract monetary information
            monetary_data = self._extract_monetary_data(validated_data)

            # Calculate price target features
            price_features = self._calculate_price_target_features(monetary_data)
            features = pd.concat([features, price_features], axis=1)

            # Calculate estimate features
            estimate_features = self._calculate_estimate_features(monetary_data)
            features = pd.concat([features, estimate_features], axis=1)

            # Calculate analyst action features
            analyst_features = self._calculate_analyst_features(validated_data, monetary_data)
            features = pd.concat([features, analyst_features], axis=1)

            # Calculate deal and corporate action features
            corporate_features = self._calculate_corporate_features(monetary_data)
            features = pd.concat([features, corporate_features], axis=1)

            # Calculate monetary policy features
            policy_features = self._calculate_policy_features(validated_data)
            features = pd.concat([features, policy_features], axis=1)

            # Calculate composite scores
            composite_features = self._calculate_composite_scores(features, monetary_data)
            features = pd.concat([features, composite_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating monetary impact features: {e}")
            return self._create_empty_features(news_data.index)

    def _extract_monetary_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Extract monetary values from news text."""
        monetary_data = news_data.copy()

        # Initialize extraction columns
        monetary_data["price_targets"] = None
        monetary_data["earnings_estimates"] = None
        monetary_data["revenue_estimates"] = None
        monetary_data["percentage_changes"] = None
        monetary_data["deal_values"] = None
        monetary_data["dividends"] = None
        monetary_data["buybacks"] = None

        for idx, row in news_data.iterrows():
            text = f"{row.get('headline', '')} {row.get('content', '')}"

            # Extract various monetary values
            monetary_data.at[idx, "price_targets"] = self._extract_values(
                text, self.monetary_patterns["price_target"]
            )
            monetary_data.at[idx, "earnings_estimates"] = self._extract_values(
                text, self.monetary_patterns["earnings_estimate"]
            )
            monetary_data.at[idx, "revenue_estimates"] = self._extract_values(
                text, self.monetary_patterns["revenue_estimate"]
            )
            monetary_data.at[idx, "percentage_changes"] = self._extract_values(
                text, self.monetary_patterns["percentage_change"]
            )
            monetary_data.at[idx, "deal_values"] = self._extract_values(
                text, self.monetary_patterns["deal_value"]
            )
            monetary_data.at[idx, "dividends"] = self._extract_values(
                text, self.monetary_patterns["dividend"]
            )
            monetary_data.at[idx, "buybacks"] = self._extract_values(
                text, self.monetary_patterns["buyback"]
            )

        return monetary_data

    def _calculate_price_target_features(self, monetary_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price target related features."""
        features = pd.DataFrame(index=monetary_data.index)

        for window in self.price_target_windows:
            window_data = self.filter_by_time_window(monetary_data, window * 24)

            # Extract all price targets in window
            all_targets = []
            for targets in window_data["price_targets"]:
                if targets:
                    all_targets.extend(targets)

            if all_targets:
                # Calculate statistics
                features[f"price_target_mean_{window}d"] = np.mean(all_targets)
                features[f"price_target_std_{window}d"] = np.std(all_targets)
                features[f"price_target_count_{window}d"] = len(all_targets)

                # Calculate consensus (inverse of coefficient of variation)
                cv = safe_divide(np.std(all_targets), np.mean(all_targets))
                features[f"price_target_consensus_{window}d"] = 1 / (1 + cv)

                # Calculate change from previous window
                if window == self.price_target_windows[0]:
                    features[f"price_target_change_{window}d"] = 0
                else:
                    prev_window_idx = self.price_target_windows.index(window) - 1
                    prev_window = self.price_target_windows[prev_window_idx]
                    prev_mean = features.get(f"price_target_mean_{prev_window}d", 0)
                    curr_mean = features[f"price_target_mean_{window}d"]
                    features[f"price_target_change_{window}d"] = safe_divide(
                        curr_mean - prev_mean, prev_mean
                    )
            else:
                # Fill with zeros if no targets found
                features[f"price_target_mean_{window}d"] = 0
                features[f"price_target_std_{window}d"] = 0
                features[f"price_target_count_{window}d"] = 0
                features[f"price_target_consensus_{window}d"] = 0
                features[f"price_target_change_{window}d"] = 0

        return features

    def _calculate_estimate_features(self, monetary_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate earnings and revenue estimate features."""
        features = pd.DataFrame(index=monetary_data.index)

        # Process earnings estimates
        earnings_values = []
        for estimates in monetary_data["earnings_estimates"]:
            if estimates:
                earnings_values.extend(estimates)

        if earnings_values:
            features["earnings_estimate_mean"] = np.mean(earnings_values)
            features["earnings_estimate_change"] = self._calculate_estimate_change(earnings_values)
            features["earnings_revision_direction"] = np.sign(features["earnings_estimate_change"])
            features["earnings_revision_magnitude"] = abs(features["earnings_estimate_change"])
        else:
            features["earnings_estimate_mean"] = 0
            features["earnings_estimate_change"] = 0
            features["earnings_revision_direction"] = 0
            features["earnings_revision_magnitude"] = 0

        # Process revenue estimates
        revenue_values = []
        for estimates in monetary_data["revenue_estimates"]:
            if estimates:
                revenue_values.extend(estimates)

        if revenue_values:
            features["revenue_estimate_mean"] = np.mean(revenue_values)
            features["revenue_estimate_change"] = self._calculate_estimate_change(revenue_values)
            features["revenue_revision_direction"] = np.sign(features["revenue_estimate_change"])
        else:
            features["revenue_estimate_mean"] = 0
            features["revenue_estimate_change"] = 0
            features["revenue_revision_direction"] = 0

        return features

    def _calculate_analyst_features(
        self, news_data: pd.DataFrame, monetary_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate analyst action and rating features."""
        features = pd.DataFrame(index=news_data.index)

        # Initialize scores
        upgrade_score = 0
        downgrade_score = 0
        total_actions = 0
        ratings = []

        for idx, row in news_data.iterrows():
            text = row.get("headline", "").lower()

            # Check for analyst actions
            for action, params in self.analyst_actions.items():
                if action in text:
                    weight = params["weight"]
                    impact = params["impact"]

                    if weight > 0:
                        upgrade_score += abs(weight * impact)
                    else:
                        downgrade_score += abs(weight * impact)

                    total_actions += 1

            # Extract ratings
            for rating, score in self.rating_scores.items():
                if rating in text:
                    ratings.append(score)

        # Calculate features
        features["analyst_upgrade_score"] = upgrade_score
        features["analyst_downgrade_score"] = downgrade_score
        features["analyst_net_score"] = upgrade_score - downgrade_score
        features["analyst_action_intensity"] = total_actions

        if ratings:
            features["rating_average"] = np.mean(ratings)
            features["rating_change"] = ratings[-1] - ratings[0] if len(ratings) > 1 else 0
        else:
            features["rating_average"] = 0
            features["rating_change"] = 0

        return features

    def _calculate_corporate_features(self, monetary_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate deal and corporate action features."""
        features = pd.DataFrame(index=monetary_data.index)

        # Deal values
        all_deals = []
        for deals in monetary_data["deal_values"]:
            if deals:
                all_deals.extend(deals)

        if all_deals:
            features["deal_value_total"] = sum(all_deals)
            features["deal_count"] = len(all_deals)
            features["avg_deal_size"] = np.mean(all_deals)
        else:
            features["deal_value_total"] = 0
            features["deal_count"] = 0
            features["avg_deal_size"] = 0

        # Dividends
        all_dividends = []
        for divs in monetary_data["dividends"]:
            if divs:
                all_dividends.extend(divs)

        if all_dividends:
            features["dividend_announced"] = 1
            features["dividend_change"] = self._calculate_dividend_change(all_dividends)
        else:
            features["dividend_announced"] = 0
            features["dividend_change"] = 0

        # Buybacks
        all_buybacks = []
        for buybacks in monetary_data["buybacks"]:
            if buybacks:
                all_buybacks.extend(buybacks)

        features["buyback_value"] = sum(all_buybacks) if all_buybacks else 0

        # Corporate action score
        features["corporate_action_score"] = (
            features["deal_count"] * 0.3
            + features["dividend_announced"] * 0.3
            + (features["buyback_value"] > 0) * 0.4
        )

        return features

    def _calculate_policy_features(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate monetary policy related features."""
        features = pd.DataFrame(index=news_data.index)

        # Keywords for policy detection
        rate_hike_keywords = ["rate hike", "raise rates", "tightening", "hawkish"]
        rate_cut_keywords = ["rate cut", "lower rates", "easing", "dovish"]
        inflation_keywords = ["inflation", "cpi", "pce", "price pressure"]
        gdp_keywords = ["gdp", "growth", "economic expansion", "recession"]

        # Count mentions
        rate_hike_count = 0
        rate_cut_count = 0
        inflation_count = 0
        gdp_count = 0

        for idx, row in news_data.iterrows():
            text = row.get("headline", "").lower() + " " + row.get("content", "").lower()

            # Count keyword occurrences
            for keyword in rate_hike_keywords:
                if keyword in text:
                    rate_hike_count += 1

            for keyword in rate_cut_keywords:
                if keyword in text:
                    rate_cut_count += 1

            for keyword in inflation_keywords:
                if keyword in text:
                    inflation_count += 1

            for keyword in gdp_keywords:
                if keyword in text:
                    gdp_count += 1

        # Calculate probabilities and scores
        total_policy_mentions = rate_hike_count + rate_cut_count

        features["rate_hike_probability"] = safe_divide(rate_hike_count, total_policy_mentions)
        features["rate_cut_probability"] = safe_divide(rate_cut_count, total_policy_mentions)
        features["monetary_policy_score"] = (
            features["rate_hike_probability"] - features["rate_cut_probability"]
        )
        features["inflation_mentions"] = inflation_count
        features["gdp_growth_mentions"] = gdp_count

        return features

    def _calculate_composite_scores(
        self, features: pd.DataFrame, monetary_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite monetary impact scores."""
        composite = pd.DataFrame(index=features.index)

        # Monetary impact score
        impact_components = [
            features.get("price_target_change_7d", 0) * 0.3,
            features.get("earnings_revision_direction", 0) * 0.2,
            features.get("analyst_net_score", 0) * 0.2,
            features.get("corporate_action_score", 0) * 0.15,
            features.get("monetary_policy_score", 0) * 0.15,
        ]
        composite["monetary_impact_score"] = sum(impact_components)

        # Estimate revision score
        composite["estimate_revision_score"] = (
            features.get("earnings_revision_magnitude", 0) * 0.6
            + features.get("revenue_revision_direction", 0) * 0.4
        )

        # Analyst consensus score
        composite["analyst_consensus_score"] = (
            features.get("rating_average", 0) * 0.5
            + features.get("price_target_consensus_7d", 0) * 0.5
        )

        # Quantitative news ratio
        total_news = len(monetary_data)
        quant_news = sum(
            [
                (monetary_data["price_targets"].notna()).sum(),
                (monetary_data["earnings_estimates"].notna()).sum(),
                (monetary_data["deal_values"].notna()).sum(),
            ]
        )
        composite["quantitative_news_ratio"] = safe_divide(quant_news, total_news)

        return composite

    def _extract_values(self, text: str, pattern: str) -> list[float]:
        """Extract numerical values using regex pattern."""
        matches = re.findall(pattern, text, re.IGNORECASE)
        values = []

        for match in matches:
            try:
                # Convert to float
                value = float(match)

                # Handle million/billion multipliers if in text
                if "billion" in text.lower():
                    value *= 1e9
                elif "million" in text.lower():
                    value *= 1e6

                values.append(value)
            except ValueError:
                continue

        return values if values else None

    def _calculate_estimate_change(self, estimates: list[float]) -> float:
        """Calculate percentage change in estimates."""
        if len(estimates) < 2:
            return 0

        # Sort by time (assuming later estimates are more recent)
        # In practice, you'd want to track timestamps
        old_estimate = estimates[0]
        new_estimate = estimates[-1]

        return safe_divide(new_estimate - old_estimate, old_estimate)

    def _calculate_dividend_change(self, dividends: list[float]) -> float:
        """Calculate dividend change."""
        if len(dividends) < 2:
            return 0

        # Compare most recent to previous
        return safe_divide(dividends[-1] - dividends[-2], dividends[-2])

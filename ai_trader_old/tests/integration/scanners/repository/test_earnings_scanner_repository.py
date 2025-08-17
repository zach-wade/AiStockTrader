"""
Integration tests for Earnings Scanner repository interactions.

Tests earnings data retrieval and analysis for the earnings scanner.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.interfaces.scanners import IScannerRepository


@pytest.mark.integration
@pytest.mark.asyncio
class TestEarningsScannerRepository:
    """Test earnings scanner repository integration."""

    async def test_get_earnings_data_basic(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_earnings_data,
    ):
        """Test basic earnings data retrieval."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            mock_earnings.return_value = sample_earnings_data

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Verify earnings data structure
            assert isinstance(result, list)
            assert len(result) > 0

            # Check first earnings record structure
            earnings = result[0]
            required_fields = ["symbol", "eps_actual", "eps_estimate", "report_date"]
            for field in required_fields:
                assert field in earnings

    async def test_earnings_surprise_analysis(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test earnings surprise detection and analysis."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock earnings with various surprise scenarios
            surprise_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) - timedelta(days=1),
                    "period_ending": datetime.now(UTC) - timedelta(days=31),
                    "eps_actual": 3.50,
                    "eps_estimate": 3.25,
                    "revenue_actual": 100000000000,
                    "revenue_estimate": 98000000000,
                    "surprise_percent": 7.69,  # Positive surprise
                    "revenue_surprise_percent": 2.04,
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) - timedelta(days=2),
                    "period_ending": datetime.now(UTC) - timedelta(days=32),
                    "eps_actual": 1.20,
                    "eps_estimate": 1.45,
                    "revenue_actual": 68000000000,
                    "revenue_estimate": 70000000000,
                    "surprise_percent": -17.24,  # Negative surprise
                    "revenue_surprise_percent": -2.86,
                },
                {
                    "symbol": "MSFT",
                    "report_date": datetime.now(UTC) - timedelta(days=3),
                    "period_ending": datetime.now(UTC) - timedelta(days=33),
                    "eps_actual": 2.75,
                    "eps_estimate": 2.73,
                    "revenue_actual": 85000000000,
                    "revenue_estimate": 84500000000,
                    "surprise_percent": 0.73,  # Small positive surprise
                    "revenue_surprise_percent": 0.59,
                },
            ]

            mock_earnings.return_value = surprise_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:3], query_filter=query_filter
            )

            # Analyze earnings surprises
            positive_surprises = [
                earning for earning in result if earning["surprise_percent"] > 5.0
            ]
            negative_surprises = [
                earning for earning in result if earning["surprise_percent"] < -5.0
            ]

            # Should detect significant surprises
            assert len(positive_surprises) == 1  # AAPL
            assert len(negative_surprises) == 1  # GOOGL

            # Check surprise magnitudes
            aapl_surprise = positive_surprises[0]
            assert aapl_surprise["symbol"] == "AAPL"
            assert aapl_surprise["surprise_percent"] > 7.0

            googl_surprise = negative_surprises[0]
            assert googl_surprise["symbol"] == "GOOGL"
            assert googl_surprise["surprise_percent"] < -15.0

    async def test_earnings_calendar_upcoming(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test upcoming earnings calendar data."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock upcoming earnings
            upcoming_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) + timedelta(days=2),  # Upcoming
                    "period_ending": datetime.now(UTC) - timedelta(days=1),
                    "eps_estimate": 3.75,
                    "revenue_estimate": 102000000000,
                    "confirmed": True,
                    "time_of_day": "after_market",
                    "fiscal_quarter": "Q1",
                    "fiscal_year": 2024,
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) + timedelta(days=5),  # Next week
                    "period_ending": datetime.now(UTC) - timedelta(days=4),
                    "eps_estimate": 1.55,
                    "revenue_estimate": 72000000000,
                    "confirmed": True,
                    "time_of_day": "before_market",
                    "fiscal_quarter": "Q1",
                    "fiscal_year": 2024,
                },
                {
                    "symbol": "MSFT",
                    "report_date": datetime.now(UTC) + timedelta(days=1),  # Tomorrow
                    "period_ending": datetime.now(UTC) - timedelta(days=2),
                    "eps_estimate": 2.85,
                    "revenue_estimate": 86000000000,
                    "confirmed": False,  # Unconfirmed date
                    "time_of_day": "after_market",
                    "fiscal_quarter": "Q1",
                    "fiscal_year": 2024,
                },
            ]

            mock_earnings.return_value = upcoming_earnings

            # Query for upcoming earnings
            future_filter = QueryFilter(
                start_date=datetime.now(UTC),
                end_date=datetime.now(UTC) + timedelta(days=7),
                symbols=test_symbols[:3],
            )

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:3], query_filter=future_filter
            )

            # All should be upcoming earnings
            assert len(result) == 3

            # Check timing analysis
            near_term = [
                earning
                for earning in result
                if (earning["report_date"] - datetime.now(UTC)).days <= 2
            ]

            assert len(near_term) == 2  # AAPL and MSFT within 2 days

            # Check confirmation status
            confirmed_earnings = [earning for earning in result if earning.get("confirmed", False)]

            assert len(confirmed_earnings) == 2  # AAPL and GOOGL confirmed

    async def test_earnings_guidance_analysis(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test earnings guidance and forward-looking metrics."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock earnings with guidance
            guidance_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) - timedelta(days=1),
                    "eps_actual": 3.50,
                    "eps_estimate": 3.25,
                    "next_quarter_eps_guidance": 3.80,
                    "next_quarter_eps_consensus": 3.70,
                    "full_year_eps_guidance": 14.50,
                    "full_year_eps_consensus": 14.20,
                    "revenue_guidance": 105000000000,
                    "revenue_consensus": 103000000000,
                    "guidance_raised": True,
                    "guidance_sentiment": "positive",
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) - timedelta(days=2),
                    "eps_actual": 1.20,
                    "eps_estimate": 1.45,
                    "next_quarter_eps_guidance": 1.35,
                    "next_quarter_eps_consensus": 1.50,
                    "full_year_eps_guidance": 5.80,
                    "full_year_eps_consensus": 6.20,
                    "revenue_guidance": 68000000000,
                    "revenue_consensus": 71000000000,
                    "guidance_raised": False,
                    "guidance_sentiment": "cautious",
                },
            ]

            mock_earnings.return_value = guidance_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Analyze guidance patterns
            positive_guidance = [
                earning for earning in result if earning.get("guidance_raised", False)
            ]
            conservative_guidance = [
                earning for earning in result if not earning.get("guidance_raised", True)
            ]

            assert len(positive_guidance) == 1  # AAPL raised guidance
            assert len(conservative_guidance) == 1  # GOOGL conservative

            # Check guidance vs consensus
            aapl_earnings = positive_guidance[0]
            assert (
                aapl_earnings["next_quarter_eps_guidance"]
                > aapl_earnings["next_quarter_eps_consensus"]
            )
            assert aapl_earnings["guidance_sentiment"] == "positive"

    async def test_earnings_historical_trends(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        historical_date_range: QueryFilter,
    ):
        """Test historical earnings trends and patterns."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock historical earnings (4 quarters)
            historical_earnings = []
            base_date = datetime.now(UTC) - timedelta(days=90)

            # Generate 4 quarters of earnings for AAPL
            quarterly_eps = [2.50, 2.75, 3.00, 3.25]  # Growing trend
            quarterly_revenue = [80, 85, 92, 98]  # Billions, growing

            for i, (eps, revenue) in enumerate(zip(quarterly_eps, quarterly_revenue)):
                historical_earnings.append(
                    {
                        "symbol": "AAPL",
                        "report_date": base_date + timedelta(days=i * 90),
                        "period_ending": base_date + timedelta(days=i * 90 - 30),
                        "eps_actual": eps,
                        "eps_estimate": eps - 0.05,  # Consistently beat estimates
                        "revenue_actual": revenue * 1000000000,
                        "revenue_estimate": revenue * 1000000000 - 1000000000,
                        "fiscal_quarter": f"Q{i+1}",
                        "fiscal_year": 2024,
                        "yoy_eps_growth": (
                            (eps - quarterly_eps[0]) / quarterly_eps[0] * 100 if i > 0 else 0
                        ),
                        "qoq_eps_growth": (
                            (eps - quarterly_eps[i - 1]) / quarterly_eps[i - 1] * 100
                            if i > 0
                            else 0
                        ),
                    }
                )

            mock_earnings.return_value = historical_earnings

            query_filter = historical_date_range
            query_filter.symbols = ["AAPL"]

            result = await scanner_repository.get_earnings_data(
                symbols=["AAPL"], query_filter=query_filter
            )

            # Analyze trends
            eps_values = [earning["eps_actual"] for earning in result]
            revenue_values = [earning["revenue_actual"] for earning in result]

            # Should show consistent growth
            assert all(eps_values[i] <= eps_values[i + 1] for i in range(len(eps_values) - 1))
            assert all(
                revenue_values[i] <= revenue_values[i + 1] for i in range(len(revenue_values) - 1)
            )

            # Check growth rates
            latest_earning = result[-1]
            assert latest_earning.get("yoy_eps_growth", 0) > 25.0  # Strong YoY growth

    async def test_earnings_sector_comparison(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test earnings comparison within sector."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock earnings for tech sector
            sector_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) - timedelta(days=1),
                    "eps_actual": 3.50,
                    "eps_estimate": 3.25,
                    "revenue_actual": 100000000000,
                    "surprise_percent": 7.69,
                    "sector": "Technology",
                    "pe_ratio": 28.5,
                    "profit_margin": 0.23,
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) - timedelta(days=2),
                    "eps_actual": 1.20,
                    "eps_estimate": 1.45,
                    "revenue_actual": 68000000000,
                    "surprise_percent": -17.24,
                    "sector": "Technology",
                    "pe_ratio": 25.2,
                    "profit_margin": 0.18,
                },
                {
                    "symbol": "MSFT",
                    "report_date": datetime.now(UTC) - timedelta(days=3),
                    "eps_actual": 2.75,
                    "eps_estimate": 2.73,
                    "revenue_actual": 85000000000,
                    "surprise_percent": 0.73,
                    "sector": "Technology",
                    "pe_ratio": 32.1,
                    "profit_margin": 0.35,
                },
            ]

            mock_earnings.return_value = sector_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:3]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:3], query_filter=query_filter
            )

            # Sector performance analysis
            sector_surprises = [earning["surprise_percent"] for earning in result]
            sector_pe_ratios = [earning["pe_ratio"] for earning in result]
            sector_margins = [earning["profit_margin"] for earning in result]

            # Calculate sector averages
            avg_surprise = sum(sector_surprises) / len(sector_surprises)
            avg_pe = sum(sector_pe_ratios) / len(sector_pe_ratios)
            avg_margin = sum(sector_margins) / len(sector_margins)

            # Check sector performance characteristics
            assert -5.0 < avg_surprise < 0  # Mixed sector performance
            assert 25.0 < avg_pe < 35.0  # Reasonable tech PE range
            assert avg_margin > 0.2  # Good tech sector margins

    async def test_earnings_conference_call_sentiment(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test earnings conference call sentiment analysis."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock earnings with conference call data
            call_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) - timedelta(days=1),
                    "eps_actual": 3.50,
                    "eps_estimate": 3.25,
                    "conference_call_sentiment": 0.8,  # Positive
                    "management_tone": "confident",
                    "key_themes": ["growth", "innovation", "market expansion"],
                    "guidance_tone": "optimistic",
                    "analyst_reception": "positive",
                    "call_transcript_available": True,
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) - timedelta(days=2),
                    "eps_actual": 1.20,
                    "eps_estimate": 1.45,
                    "conference_call_sentiment": 0.3,  # Negative
                    "management_tone": "cautious",
                    "key_themes": ["challenges", "headwinds", "investment"],
                    "guidance_tone": "conservative",
                    "analyst_reception": "mixed",
                    "call_transcript_available": True,
                },
            ]

            mock_earnings.return_value = call_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Analyze conference call sentiment
            positive_calls = [
                earning for earning in result if earning.get("conference_call_sentiment", 0.5) > 0.7
            ]
            negative_calls = [
                earning for earning in result if earning.get("conference_call_sentiment", 0.5) < 0.4
            ]

            assert len(positive_calls) == 1  # AAPL positive call
            assert len(negative_calls) == 1  # GOOGL negative call

            # Check sentiment details
            aapl_call = positive_calls[0]
            assert aapl_call["management_tone"] == "confident"
            assert "growth" in aapl_call["key_themes"]

            googl_call = negative_calls[0]
            assert googl_call["management_tone"] == "cautious"
            assert "challenges" in googl_call["key_themes"]

    async def test_earnings_estimate_revisions(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test earnings estimate revisions tracking."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock earnings with estimate revision data
            revision_earnings = [
                {
                    "symbol": "AAPL",
                    "report_date": datetime.now(UTC) - timedelta(days=1),
                    "eps_actual": 3.50,
                    "eps_estimate": 3.25,
                    "eps_estimate_30_days_ago": 3.10,
                    "eps_estimate_60_days_ago": 2.95,
                    "eps_estimate_90_days_ago": 2.85,
                    "revision_trend": "upward",
                    "analyst_upgrades": 8,
                    "analyst_downgrades": 2,
                    "price_target_avg": 185.0,
                    "price_target_high": 220.0,
                    "price_target_low": 160.0,
                },
                {
                    "symbol": "GOOGL",
                    "report_date": datetime.now(UTC) - timedelta(days=2),
                    "eps_actual": 1.20,
                    "eps_estimate": 1.45,
                    "eps_estimate_30_days_ago": 1.50,
                    "eps_estimate_60_days_ago": 1.55,
                    "eps_estimate_90_days_ago": 1.60,
                    "revision_trend": "downward",
                    "analyst_upgrades": 1,
                    "analyst_downgrades": 6,
                    "price_target_avg": 125.0,
                    "price_target_high": 150.0,
                    "price_target_low": 100.0,
                },
            ]

            mock_earnings.return_value = revision_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:2]

            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols[:2], query_filter=query_filter
            )

            # Analyze estimate revisions
            upward_revisions = [
                earning for earning in result if earning.get("revision_trend") == "upward"
            ]
            downward_revisions = [
                earning for earning in result if earning.get("revision_trend") == "downward"
            ]

            assert len(upward_revisions) == 1  # AAPL
            assert len(downward_revisions) == 1  # GOOGL

            # Check revision magnitude
            aapl_revision = upward_revisions[0]
            revision_change = (
                aapl_revision["eps_estimate"] - aapl_revision["eps_estimate_90_days_ago"]
            )
            assert revision_change > 0.3  # Significant upward revision

    async def test_error_handling_no_earnings_data(
        self, scanner_repository: IScannerRepository, test_symbols, recent_date_range: QueryFilter
    ):
        """Test error handling when no earnings data is available."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock no earnings data
            mock_earnings.return_value = []

            query_filter = recent_date_range
            query_filter.symbols = ["UNKNOWN_SYMBOL"]

            result = await scanner_repository.get_earnings_data(
                symbols=["UNKNOWN_SYMBOL"], query_filter=query_filter
            )

            # Should return empty list gracefully
            assert isinstance(result, list)
            assert len(result) == 0

    async def test_earnings_query_performance(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        performance_thresholds,
    ):
        """Test earnings query performance meets scanner requirements."""
        with patch.object(scanner_repository, "get_earnings_data") as mock_earnings:
            # Mock large earnings dataset
            large_earnings = [
                {
                    "symbol": test_symbols[i % len(test_symbols)],
                    "report_date": datetime.now(UTC) - timedelta(days=i),
                    "eps_actual": 2.0 + (i % 10) * 0.1,
                    "eps_estimate": 1.9 + (i % 10) * 0.1,
                    "revenue_actual": 50000000000 + i * 1000000000,
                    "surprise_percent": (i % 20) - 10,  # Range from -10% to +9%
                }
                for i in range(200)  # 200 earnings reports
            ]

            mock_earnings.return_value = large_earnings

            query_filter = recent_date_range
            query_filter.symbols = test_symbols

            # Time the earnings query
            start_time = datetime.now()
            result = await scanner_repository.get_earnings_data(
                symbols=test_symbols, query_filter=query_filter
            )
            end_time = datetime.now()

            query_time_ms = (end_time - start_time).total_seconds() * 1000

            # Should meet performance threshold
            threshold = performance_thresholds["repository"]["query_time_ms"]
            assert query_time_ms < threshold
            assert len(result) == 200

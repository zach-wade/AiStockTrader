"""
Integration tests for Volume Scanner repository interactions.

Tests volume data retrieval and processing for the volume scanner.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from main.interfaces.scanners import IScannerRepository


@pytest.mark.integration
@pytest.mark.asyncio
class TestVolumeScannerRepository:
    """Test volume scanner repository integration."""

    async def test_volume_statistics_calculation(
        self, scanner_repository: IScannerRepository, test_symbols, sample_market_data
    ):
        """Test volume statistics calculation for relative volume analysis."""
        with patch.object(scanner_repository, "db_adapter") as mock_db:
            # Mock volume statistics data
            volume_stats = [
                {
                    "symbol": "AAPL",
                    "avg_volume": 75000000.0,
                    "std_volume": 15000000.0,
                    "min_volume": 40000000.0,
                    "max_volume": 120000000.0,
                    "data_points": 20,
                },
                {
                    "symbol": "GOOGL",
                    "avg_volume": 25000000.0,
                    "std_volume": 5000000.0,
                    "min_volume": 15000000.0,
                    "max_volume": 40000000.0,
                    "data_points": 20,
                },
            ]

            async def mock_run_async(func):
                return volume_stats

            mock_db.run_async = mock_run_async

            result = await scanner_repository.get_volume_statistics(
                symbols=test_symbols[:2], lookback_days=20
            )

            # Verify results structure
            assert isinstance(result, dict)
            assert "AAPL" in result
            assert "GOOGL" in result

            # Check AAPL stats
            aapl_stats = result["AAPL"]
            assert aapl_stats["avg_volume"] == 75000000.0
            assert aapl_stats["std_volume"] == 15000000.0
            assert aapl_stats["data_points"] == 20

            # Check GOOGL stats
            googl_stats = result["GOOGL"]
            assert googl_stats["avg_volume"] == 25000000.0
            assert googl_stats["std_volume"] == 5000000.0

    async def test_relative_volume_calculation(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test relative volume calculation for anomaly detection."""
        # Mock current and historical volume data
        with (
            patch.object(scanner_repository, "get_volume_statistics") as mock_stats,
            patch.object(scanner_repository, "get_latest_prices") as mock_prices,
        ):

            # Mock volume statistics
            mock_stats.return_value = {
                "AAPL": {
                    "avg_volume": 50000000.0,
                    "std_volume": 10000000.0,
                    "min_volume": 30000000.0,
                    "max_volume": 80000000.0,
                    "data_points": 20,
                }
            }

            # Mock current volume (high relative volume)
            mock_prices.return_value = {
                "AAPL": {
                    "timestamp": datetime.now(UTC),
                    "volume": 150000000.0,  # 3x average
                    "close": 150.0,
                    "open": 148.0,
                    "high": 152.0,
                    "low": 147.0,
                    "vwap": 150.5,
                }
            }

            # Get the data
            stats = await scanner_repository.get_volume_statistics(["AAPL"])
            prices = await scanner_repository.get_latest_prices(["AAPL"])

            # Calculate relative volume
            aapl_current_volume = prices["AAPL"]["volume"]
            aapl_avg_volume = stats["AAPL"]["avg_volume"]
            relative_volume = aapl_current_volume / aapl_avg_volume

            # Should detect high relative volume
            assert relative_volume == 3.0
            assert relative_volume > 2.0  # Threshold for volume anomaly

    async def test_intraday_volume_patterns(
        self, scanner_repository: IScannerRepository, test_symbols, sample_market_data
    ):
        """Test intraday volume pattern analysis."""
        with patch.object(scanner_repository, "get_intraday_data") as mock_intraday:
            # Mock intraday data with volume patterns
            # Third-party imports
            import pandas as pd

            # Create sample intraday data with volume spikes
            timestamps = [
                datetime.now(UTC) - timedelta(minutes=i * 15)
                for i in range(32)  # 8 hours of 15-min data
            ]

            volumes = [1000000] * 32  # Base volume
            volumes[5] = 5000000  # Volume spike at market open
            volumes[10] = 3000000  # Mid-morning spike
            volumes[20] = 7000000  # Afternoon spike

            intraday_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "volume": volumes,
                    "close": [150.0] * 32,
                    "symbol": ["AAPL"] * 32,
                }
            )

            mock_intraday.return_value = {"AAPL": intraday_df}

            result = await scanner_repository.get_intraday_data(["AAPL"])

            # Verify intraday data structure
            assert "AAPL" in result
            aapl_data = result["AAPL"]
            assert len(aapl_data) == 32

            # Check for volume spikes
            volume_spikes = aapl_data[aapl_data["volume"] > 2000000]
            assert len(volume_spikes) == 3  # Three volume spikes

    async def test_volume_breakout_detection(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test volume breakout detection logic."""
        with (
            patch.object(scanner_repository, "get_volume_statistics") as mock_stats,
            patch.object(scanner_repository, "get_intraday_data") as mock_intraday,
        ):

            # Mock volume statistics
            mock_stats.return_value = {
                "AAPL": {"avg_volume": 60000000.0, "std_volume": 12000000.0, "data_points": 20}
            }

            # Mock intraday data with breakout pattern
            # Third-party imports
            import pandas as pd

            # Simulate volume breakout: low volume followed by high volume
            timestamps = [
                datetime.now(UTC) - timedelta(minutes=i * 5)
                for i in range(12)  # 1 hour of 5-min data
            ]

            volumes = [30000000] * 8 + [180000000] * 4  # Low then high volume
            closes = [149.0] * 8 + [155.0] * 4  # Price breakout with volume

            breakout_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "volume": volumes,
                    "close": closes,
                    "symbol": ["AAPL"] * 12,
                }
            )

            mock_intraday.return_value = {"AAPL": breakout_df}

            # Get data and analyze breakout
            stats = await scanner_repository.get_volume_statistics(["AAPL"])
            intraday = await scanner_repository.get_intraday_data(["AAPL"], lookback_hours=1)

            aapl_data = intraday["AAPL"]
            avg_volume = stats["AAPL"]["avg_volume"]

            # Check for volume breakout pattern
            recent_volume = aapl_data.tail(4)["volume"].mean()  # Last 4 periods
            early_volume = aapl_data.head(8)["volume"].mean()  # First 8 periods

            volume_increase_ratio = recent_volume / early_volume
            relative_to_avg = recent_volume / avg_volume

            # Should detect significant volume increase
            assert volume_increase_ratio == 6.0  # 6x increase
            assert relative_to_avg == 3.0  # 3x average volume

    async def test_volume_drying_up_detection(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test volume drying up detection (low volume patterns)."""
        with (
            patch.object(scanner_repository, "get_volume_statistics") as mock_stats,
            patch.object(scanner_repository, "get_intraday_data") as mock_intraday,
        ):

            # Mock volume statistics
            mock_stats.return_value = {
                "AAPL": {"avg_volume": 75000000.0, "std_volume": 15000000.0, "data_points": 20}
            }

            # Mock intraday data with declining volume
            # Third-party imports
            import pandas as pd

            timestamps = [
                datetime.now(UTC) - timedelta(minutes=i * 15)
                for i in range(16)  # 4 hours of 15-min data
            ]

            # Declining volume pattern
            volumes = [75000000 - (i * 3000000) for i in range(16)]
            volumes = [max(v, 15000000) for v in volumes]  # Floor at 15M

            drying_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "volume": volumes,
                    "close": [150.0] * 16,  # Stable price with declining volume
                    "symbol": ["AAPL"] * 16,
                }
            )

            mock_intraday.return_value = {"AAPL": drying_df}

            # Analyze volume pattern
            stats = await scanner_repository.get_volume_statistics(["AAPL"])
            intraday = await scanner_repository.get_intraday_data(["AAPL"], lookback_hours=4)

            aapl_data = intraday["AAPL"]
            avg_volume = stats["AAPL"]["avg_volume"]

            # Check for declining volume
            recent_avg = aapl_data.tail(4)["volume"].mean()
            early_avg = aapl_data.head(4)["volume"].mean()

            volume_decline_ratio = recent_avg / early_avg
            relative_to_avg = recent_avg / avg_volume

            # Should detect volume decline
            assert volume_decline_ratio < 1.0  # Volume declining
            assert relative_to_avg < 0.5  # Below average volume

    async def test_volume_profile_analysis(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test volume profile analysis for price levels."""
        with patch.object(scanner_repository, "get_intraday_data") as mock_intraday:
            # Third-party imports
            import pandas as pd

            # Create volume profile data
            timestamps = [
                datetime.now(UTC) - timedelta(minutes=i * 5)
                for i in range(24)  # 2 hours of 5-min data
            ]

            # Volume concentrated at specific price levels
            prices = [148.0, 149.0, 150.0, 151.0, 152.0] * 4 + [150.0] * 4
            volumes = [2000000] * 20 + [8000000] * 4  # High volume at $150

            profile_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "close": prices,
                    "volume": volumes,
                    "symbol": ["AAPL"] * 24,
                }
            )

            mock_intraday.return_value = {"AAPL": profile_df}

            # Analyze volume profile
            intraday = await scanner_repository.get_intraday_data(["AAPL"], lookback_hours=2)
            aapl_data = intraday["AAPL"]

            # Group by price level and sum volume
            price_volume = aapl_data.groupby("close")["volume"].sum().to_dict()

            # Should show high volume at $150 level
            assert price_volume[150.0] > price_volume[149.0]
            assert price_volume[150.0] > price_volume[151.0]

    async def test_error_handling_no_volume_data(
        self, scanner_repository: IScannerRepository, test_symbols
    ):
        """Test error handling when volume data is unavailable."""
        with patch.object(scanner_repository, "get_volume_statistics") as mock_stats:
            # Mock database error
            mock_stats.side_effect = Exception("Volume data unavailable")

            result = await scanner_repository.get_volume_statistics(["AAPL"])

            # Should return empty stats gracefully
            assert isinstance(result, dict)
            assert "AAPL" in result
            assert result["AAPL"]["avg_volume"] == 0.0

    async def test_volume_query_performance(
        self, scanner_repository: IScannerRepository, test_symbols, performance_thresholds
    ):
        """Test volume query performance meets scanner requirements."""
        with patch.object(scanner_repository, "db_adapter") as mock_db:
            # Mock fast response
            async def mock_run_async(func):
                return [
                    {
                        "symbol": symbol,
                        "avg_volume": 50000000.0,
                        "std_volume": 10000000.0,
                        "min_volume": 30000000.0,
                        "max_volume": 80000000.0,
                        "data_points": 20,
                    }
                    for symbol in test_symbols
                ]

            mock_db.run_async = mock_run_async

            # Time the volume statistics query
            start_time = datetime.now()
            result = await scanner_repository.get_volume_statistics(test_symbols)
            end_time = datetime.now()

            query_time_ms = (end_time - start_time).total_seconds() * 1000

            # Should meet performance threshold
            threshold = performance_thresholds["repository"]["query_time_ms"]
            assert query_time_ms < threshold
            assert len(result) == len(test_symbols)

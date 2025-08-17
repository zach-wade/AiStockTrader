"""
Integration tests for the corporate actions ETL pipeline.

Tests the complete flow from fetching corporate actions to storing them in the database
and applying adjustments to market data.
"""

# Standard library imports
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

# Third-party imports
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pytest

# Local imports
from main.data_pipeline.processing.manager import ProcessingManager
from main.data_pipeline.processing.transformer import DataTransformer
from main.data_pipeline.storage.repositories.dividends_repository import DividendsRepository
from main.data_pipeline.types import DataPipelineResult, RawDataRecord
from main.interfaces.database import IAsyncDatabase


class TestCorporateActionsETL:
    """Test the complete corporate actions ETL pipeline."""

    @pytest.fixture
    def mock_db_adapter(self):
        """Create a mock database adapter."""
        adapter = AsyncMock(spec=IAsyncDatabase)
        adapter.execute_query = AsyncMock()
        adapter.fetch_all = AsyncMock(return_value=[])
        adapter.fetch_one = AsyncMock(return_value=None)
        return adapter

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DictConfig(
            {
                "processing": {
                    "batch_size": 100,
                    "parallel_workers": 2,
                    "timeout_seconds": 60,
                    "transformer": {
                        "field_mappings": {
                            "polygon": {
                                "o": "open",
                                "h": "high",
                                "l": "low",
                                "c": "close",
                                "v": "volume",
                            }
                        },
                        "missing_value_strategy": "ffill",
                        "outlier_method": "iqr",
                    },
                }
            }
        )

    @pytest.fixture
    def processing_manager(self, config, mock_db_adapter):
        """Create ProcessingManager instance."""
        return ProcessingManager(config=config, db_adapter=mock_db_adapter)

    @pytest.fixture
    def sample_corporate_actions_data(self):
        """Create sample corporate actions data."""
        return {
            "data": [
                {
                    "ticker": "AAPL",
                    "ex_dividend_date": "2024-11-08",
                    "cash_amount": 0.25,
                    "pay_date": "2024-11-14",
                    "record_date": "2024-11-11",
                    "declaration_date": "2024-10-31",
                    "dividend_type": "regular",
                },
                {"ticker": "AAPL", "execution_date": "2024-06-10", "split_from": 1, "split_to": 4},
            ]
        }

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data DataFrame."""
        dates = pd.date_range(start="2024-06-01", end="2024-11-30", freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.secure_uniform(170, 180, len(dates)),
                "high": np.secure_uniform(175, 185, len(dates)),
                "low": np.secure_uniform(165, 175, len(dates)),
                "close": np.secure_uniform(170, 180, len(dates)),
                "volume": np.secure_randint(50000000, 100000000, len(dates)),
            }
        ).set_index("timestamp")

    @pytest.mark.asyncio
    async def test_corporate_actions_etl_flow(
        self, processing_manager, mock_db_adapter, sample_corporate_actions_data
    ):
        """Test the complete ETL flow for corporate actions."""
        # Mock the archive query to return corporate actions
        with patch("main.data_pipeline.processing.manager.get_archive") as mock_get_archive:
            mock_archive = AsyncMock()
            mock_archive.query_raw_records = AsyncMock(
                return_value=[
                    RawDataRecord(
                        source="polygon",
                        data_type="corporate_actions",
                        symbol="AAPL",
                        timestamp=datetime.now(UTC),
                        data=sample_corporate_actions_data,
                        metadata={"action_count": 2},
                    )
                ]
            )
            mock_get_archive.return_value = mock_archive

            # Mock the repository add_dividends_batch to return success
            mock_db_adapter.execute_query = AsyncMock(return_value=2)  # 2 records inserted

            # Process corporate actions
            result = await processing_manager.process_corporate_actions(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            # Verify results
            assert isinstance(result, DataPipelineResult)
            assert result.status == "success"
            assert result.records_processed == 2
            assert result.records_failed == 0

            # Verify archive was queried
            mock_archive.query_raw_records.assert_called_once()

            # Verify database operations were called
            assert mock_db_adapter.execute_query.call_count > 0

    @pytest.mark.asyncio
    async def test_corporate_action_adjustments(self, config, mock_db_adapter, sample_market_data):
        """Test that corporate actions properly adjust market data."""
        # Create transformer with mock corporate actions in database
        transformer = DataTransformer(config, mock_db_adapter)

        # Mock the dividends repository to return corporate actions
        mock_corporate_actions = [
            {
                "symbol": "AAPL",
                "action_type": "split",
                "ex_date": datetime(2024, 6, 10).date(),
                "ratio": "4:1",
                "amount": None,
            },
            {
                "symbol": "AAPL",
                "action_type": "dividend",
                "ex_date": datetime(2024, 11, 8).date(),
                "amount": 0.25,
                "ratio": None,
            },
        ]

        with patch.object(
            transformer.dividends_repo,
            "get_corporate_actions_for_period",
            return_value=mock_corporate_actions,
        ):
            # Transform market data with corporate action adjustments
            adjusted_data = await transformer.transform_market_data(
                sample_market_data.copy(), source="polygon", symbol="AAPL"
            )

            # Verify split adjustment (prices before 2024-06-10 should be divided by 4)
            pre_split_data = adjusted_data[adjusted_data.index < "2024-06-10"]
            post_split_data = adjusted_data[adjusted_data.index >= "2024-06-10"]

            # Check that pre-split prices are lower (divided by 4)
            if len(pre_split_data) > 0 and len(post_split_data) > 0:
                avg_pre_split_close = pre_split_data["close"].mean()
                avg_post_split_close = post_split_data["close"].mean()
                assert avg_pre_split_close < avg_post_split_close * 0.3  # Should be roughly 1/4

            # Verify dividend adjustment (prices before 2024-11-08 should be slightly lower)
            pre_div_data = adjusted_data[
                (adjusted_data.index < "2024-11-08") & (adjusted_data.index >= "2024-06-10")
            ]
            post_div_data = adjusted_data[adjusted_data.index >= "2024-11-08"]

            if len(pre_div_data) > 0 and len(post_div_data) > 0:
                # Dividend adjustment should be small
                avg_pre_div_close = pre_div_data["close"].mean()
                avg_post_div_close = post_div_data["close"].mean()
                # Pre-dividend prices should be slightly lower due to adjustment
                assert avg_pre_div_close < avg_post_div_close

    @pytest.mark.asyncio
    async def test_dividends_repository_integration(self, mock_db_adapter):
        """Test the DividendsRepository integration with corporate actions."""
        repo = DividendsRepository(mock_db_adapter)

        # Mock database responses
        mock_db_adapter.fetch_all = AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "action_type": "dividend",
                    "ex_date": datetime(2024, 11, 8).date(),
                    "amount": 0.25,
                    "ratio": None,
                    "pay_date": datetime(2024, 11, 14).date(),
                },
                {
                    "symbol": "AAPL",
                    "action_type": "split",
                    "ex_date": datetime(2024, 6, 10).date(),
                    "amount": None,
                    "ratio": "4:1",
                    "pay_date": None,
                },
            ]
        )

        # Test get_corporate_actions_for_period
        actions = await repo.get_corporate_actions_for_period(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
        )

        assert len(actions) == 2
        assert actions[0]["action_type"] == "dividend"
        assert actions[0]["amount"] == 0.25
        assert actions[1]["action_type"] == "split"
        assert actions[1]["ratio"] == "4:1"

    @pytest.mark.asyncio
    async def test_error_handling_in_etl(self, processing_manager):
        """Test error handling in the ETL pipeline."""
        # Mock archive to raise an error
        with patch("main.data_pipeline.processing.manager.get_archive") as mock_get_archive:
            mock_archive = AsyncMock()
            mock_archive.query_raw_records = AsyncMock(
                side_effect=Exception("Archive connection failed")
            )
            mock_get_archive.return_value = mock_archive

            # Process should handle error gracefully
            result = await processing_manager.process_corporate_actions(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            assert result.status == "failed"
            assert result.records_processed == 0
            assert len(result.errors) > 0
            assert "Archive connection failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_transform_corporate_action_formats(self, processing_manager):
        """Test transformation of different corporate action formats."""
        # Test dividend transformation
        dividend_raw = {
            "ticker": "MSFT",
            "ex_dividend_date": "2024-11-14",
            "cash_amount": 0.75,
            "dividend_type": "regular",
            "frequency": "quarterly",
        }

        transformed = processing_manager._transform_corporate_action(dividend_raw, "MSFT")

        assert transformed["symbol"] == "MSFT"
        assert transformed["action_type"] == "dividend"
        assert transformed["ex_date"] == "2024-11-14"
        assert transformed["amount"] == 0.75

        # Test split transformation
        split_raw = {
            "ticker": "NVDA",
            "execution_date": "2024-06-07",
            "split_from": 1,
            "split_to": 10,
        }

        transformed = processing_manager._transform_corporate_action(split_raw, "NVDA")

        assert transformed["symbol"] == "NVDA"
        assert transformed["action_type"] == "split"
        assert transformed["ex_date"] == "2024-06-07"
        assert transformed["ratio"] == "10:1"

        # Test unknown format
        unknown_raw = {"ticker": "UNKNOWN", "some_field": "value"}

        transformed = processing_manager._transform_corporate_action(unknown_raw, "UNKNOWN")
        assert transformed is None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

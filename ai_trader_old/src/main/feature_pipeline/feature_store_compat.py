# File: src/main/feature_pipeline/feature_store_compat.py
"""
Compatibility wrapper for legacy FeatureStore imports.

This module provides backward compatibility for code that expects the old FeatureStore class.
It wraps the new dual-store architecture (PostgreSQL + HDF5) behind a simple interface.
"""

# Standard library imports
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.config.config_manager import get_config
from main.feature_pipeline.feature_store import FeatureStoreV2

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Legacy compatibility wrapper that provides the old FeatureStore interface.

    This class wraps FeatureStoreV2 (HDF5) for training/backtesting data.
    """

    def __init__(self, feature_path: str | None = None, config: dict | None = None):
        """
        Initialize FeatureStore compatibility wrapper.

        Args:
            feature_path: Path to feature storage (for HDF5 files)
            config: Configuration dictionary
        """
        self.config = config or get_config()

        # Initialize HDF5 store for training/backtesting
        if feature_path:
            self.features_path = Path(feature_path)
        else:
            # Always use the ai_trader/data_lake location
            ai_trader_root = Path(__file__).parent.parent.parent.parent
            data_lake_path = ai_trader_root / "data_lake"

            # If it doesn't exist, check if we're running from parent directory
            if not data_lake_path.exists():
                parent_data_lake = Path.cwd() / "ai_trader" / "data_lake"
                if parent_data_lake.exists():
                    data_lake_path = parent_data_lake
                else:
                    # Fall back to environment/config
                    env_path = os.environ.get("DATA_LAKE_PATH")
                    if env_path:
                        data_lake_path = Path(env_path)
                    else:
                        # Last resort - use relative path
                        data_lake_path = Path("data_lake")

            self.features_path = Path(data_lake_path) / "features"

        # Pass the features directory path, not the parent
        self.feature_store = FeatureStoreV2(str(self.features_path), self.config)

        logger.info(
            f"FeatureStore compatibility wrapper initialized with path: {self.features_path}"
        )

    @property
    def feature_repo(self):
        """Lazy initialization of PostgreSQL feature repository."""
        if self._feature_repo is None:
            db_factory = DatabaseFactory()
            self._db_adapter = db_factory.create_async_database(self.config)
            self._feature_repo = FeatureStoreRepository(self._db_adapter)
        return self._feature_repo

    def save_features(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        feature_type: str = "technical_indicators",
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Save features to both stores (HDF5 and PostgreSQL).

        Args:
            symbol: Trading symbol
            features_df: DataFrame with features
            feature_type: Type of features
            timestamp: Timestamp for features (uses current if not provided)

        Returns:
            True if successful
        """
        try:
            # Use current timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now()

            # Extract year and month for HDF5 storage
            year = timestamp.year
            month = timestamp.month

            # Save to HDF5 (versioned storage for training)
            # FeatureStoreV2 has store_features method, not save_features
            self.feature_store.store_features(symbol=symbol, features_df=features_df)
            logger.debug(f"Saved features to HDF5 for {symbol}")

            return True

        except Exception as e:
            logger.error(f"Error saving features for {symbol}: {e}")
            return False

    def load_features(
        self,
        symbol: str,
        feature_type: str = "technical_indicators",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Load features from HDF5 store.

        Args:
            symbol: Trading symbol
            feature_type: Type of features to load
            start_date: Start date for features
            end_date: End date for features

        Returns:
            DataFrame with features or None
        """
        try:
            # Load from HDF5
            # FeatureStoreV2 has get_features method, not load_features
            if start_date and end_date:
                features_df = self.feature_store.get_features(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )
            else:
                # Get all available features
                features_df = self.feature_store.get_latest_features(symbol)

            if features_df is not None:
                logger.debug(f"Loaded features from HDF5 for {symbol}")

            return features_df

        except Exception as e:
            logger.error(f"Error loading features for {symbol}: {e}")
            raise RuntimeError(f"Failed to load features for {symbol}: {e}") from e

    def get_latest_features(
        self, symbol: str, feature_type: str = "technical_indicators"
    ) -> pd.DataFrame | None:
        """
        Get the most recent features for a symbol from HDF5.

        Args:
            symbol: Trading symbol
            feature_type: Type of features

        Returns:
            DataFrame with latest features or None
        """
        try:
            # Get from HDF5
            features_df = self.feature_store.get_latest_features(symbol)

            if features_df is not None and not features_df.empty:
                # Return as DataFrame if it's a Series
                if isinstance(features_df, pd.Series):
                    return pd.DataFrame([features_df])
                return features_df

            return None  # This is a valid case - no features found

        except Exception as e:
            logger.error(f"Error getting latest features for {symbol}: {e}")
            raise RuntimeError(f"Failed to get latest features for {symbol}: {e}") from e

    def list_available_features(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        List available features from HDF5 store.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of feature information
        """
        try:
            # Get features from HDF5
            if symbol:
                symbols = [symbol]
            else:
                symbols = self.feature_store.list_symbols()

            all_features = []
            for sym in symbols:
                feature_names = self.feature_store.get_feature_names(sym)
                if feature_names:
                    all_features.append(
                        {"symbol": sym, "features": feature_names, "storage": "HDF5"}
                    )

            return all_features

        except Exception as e:
            logger.error(f"Error listing features: {e}")
            return []

    async def get_features(self, request: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """
        Get features based on request parameters for training/backtesting.

        Args:
            request: Dictionary with keys:
                - feature_sets: List of feature sets to retrieve
                - symbols: List of symbols
                - lookback_days: Number of days to look back

        Returns:
            Dictionary mapping feature_set -> DataFrame
        """
        try:
            feature_sets = request.get("feature_sets", ["all"])
            symbols = request.get("symbols", [])
            lookback_days = request.get("lookback_days", 252)

            logger.info(f"Loading features for symbols: {symbols}, lookback_days: {lookback_days}")

            results = {}

            # Load features with targets from HDF5 files
            for symbol in symbols:
                try:
                    # Check if we have pre-computed features with targets
                    targets_dir = self.features_path.parent / "targets"
                    combined_file = targets_dir / f"{symbol}_features_with_targets.csv"

                    if combined_file.exists():
                        logger.info(
                            f"Loading pre-computed features with targets from {combined_file}"
                        )
                        df = pd.read_csv(combined_file, index_col=0, parse_dates=True)
                        logger.info(
                            f"Loaded {len(df)} rows with {len(df.columns)} columns for {symbol}"
                        )

                        # Apply lookback filter
                        if lookback_days > 0:
                            df = df.tail(lookback_days)
                            logger.info(f"Applied lookback filter: {len(df)} rows remaining")

                        # Add symbol column for multi-symbol training
                        df["symbol"] = symbol

                        # Store in results
                        for feature_set in feature_sets:
                            if feature_set == "all" or feature_set in [
                                "technical",
                                "features",
                                "combined",
                            ]:
                                if feature_set not in results:
                                    results[feature_set] = df.copy()
                                else:
                                    results[feature_set] = pd.concat(
                                        [results[feature_set], df], ignore_index=False
                                    )

                    else:
                        # Fall back to loading from HDF5 and generating targets on-the-fly
                        logger.info(f"Loading features from HDF5 for {symbol}")
                        hdf5_path = self.features_path / "features" / f"{symbol}_features.h5"

                        if hdf5_path.exists():
                            # Use our target generator to load and create targets
                            from .target_generator import TargetGenerator

                            generator = TargetGenerator()

                            features_df, targets_df = generator.generate_targets_from_hdf5(
                                str(hdf5_path), lookback_days=lookback_days, save_targets=True
                            )

                            if not features_df.empty:
                                # Combine features and targets
                                combined_df = pd.concat([features_df, targets_df], axis=1)
                                combined_df["symbol"] = symbol

                                # Store in results
                                for feature_set in feature_sets:
                                    if feature_set == "all" or feature_set in [
                                        "technical",
                                        "features",
                                        "combined",
                                    ]:
                                        if feature_set not in results:
                                            results[feature_set] = combined_df.copy()
                                        else:
                                            results[feature_set] = pd.concat(
                                                [results[feature_set], combined_df],
                                                ignore_index=False,
                                            )

                                logger.info(
                                    f"Generated features and targets for {symbol}: {len(combined_df)} rows"
                                )
                            else:
                                logger.warning(f"No features loaded for {symbol} from HDF5")
                        else:
                            logger.warning(
                                f"No HDF5 features file found for {symbol} at {hdf5_path}"
                            )

                except Exception as e:
                    logger.error(f"Error loading features for {symbol}: {e}")
                    continue

            # Log final results
            for feature_set, df in results.items():
                logger.info(
                    f"Final result for '{feature_set}': {len(df)} rows, {len(df.columns)} columns"
                )
                if "symbol" in df.columns:
                    symbols_in_df = df["symbol"].unique()
                    logger.info(f"Symbols in '{feature_set}': {symbols_in_df}")

            return results

        except Exception as e:
            logger.error(f"Error in get_features: {e}", exc_info=True)
            # Return empty results on error
            return {}

    async def _get_available_feature_sets(self) -> list[str]:
        """Get available feature sets."""
        # Return default feature sets
        return ["technical_indicators", "sentiment", "fundamental"]

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1day",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical market data for backtesting.

        This method provides compatibility with BacktestEngine expectations.

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (only '1day' supported currently)
            adjusted: Whether to use adjusted prices

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Use the resolved data_lake path
            data_lake_path = self.features_path.parent
            market_path = (
                data_lake_path
                / "processed"
                / "market_data"
                / f"symbol={symbol}"
                / f"interval={timeframe}"
            )

            if not market_path.exists():
                logger.warning(f"No data found for {symbol} at {market_path}")
                return pd.DataFrame()

            all_data = []
            # Read all parquet files in date folders
            for date_folder in sorted(market_path.glob("date=*")):
                for parquet_file in date_folder.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(parquet_file)
                        all_data.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read {parquet_file}: {e}")

            if not all_data:
                logger.warning(f"No data files found for {symbol}")
                return pd.DataFrame()

            # Combine all data
            df = pd.concat(all_data, ignore_index=False)
            df = df.sort_index()

            # Filter by date range
            if df.index.tz is not None and start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=UTC)
                end_date = end_date.replace(tzinfo=UTC)

            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Ensure we have required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                logger.error(
                    f"Missing required columns for {symbol}. Available: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            # Add timestamp column from index
            df["timestamp"] = df.index

            # Return with required columns
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def close(self):
        """Clean up resources."""
        self.feature_store.close()

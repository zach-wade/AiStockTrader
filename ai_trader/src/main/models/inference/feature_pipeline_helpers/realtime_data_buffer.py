# File: src/ai_trader/models/inference/feature_pipeline_helpers/realtime_data_buffer.py

import logging
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class RealtimeDataBuffer:
    """
    Manages in-memory buffers for streaming price and volume data.
    Provides methods to update buffers and retrieve them as Pandas DataFrames
    for lookback calculations.
    """

    def __init__(self, max_buffer_size: int = 500):
        """
        Initializes the RealtimeDataBuffer.

        Args:
            max_buffer_size: The maximum number of data points to keep in the buffers.
        """
        self._price_buffer: List[Tuple[datetime, float]] = [] # (timestamp, price)
        self._volume_buffer: List[Tuple[datetime, int]] = []  # (timestamp, volume)
        self._max_buffer_size = max_buffer_size
        logger.debug(f"RealtimeDataBuffer initialized with max_buffer_size: {max_buffer_size}")

    def update_buffers(self, timestamp: datetime, price: float, volume: int):
        """
        Updates the internal price and volume buffers with a new data point.
        Maintains the buffer size by dropping the oldest entries.

        Args:
            timestamp: The timestamp of the new data point.
            price: The closing price for the new data point.
            volume: The volume for the new data point.
        """
        # Ensure timestamp is timezone-aware UTC for consistency
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        self._price_buffer.append((timestamp, price))
        self._volume_buffer.append((timestamp, volume))
        
        # Trim buffers to maintain max_buffer_size
        if len(self._price_buffer) > self._max_buffer_size:
            self._price_buffer = self._price_buffer[-self._max_buffer_size:]
        if len(self._volume_buffer) > self._max_buffer_size:
            self._volume_buffer = self._volume_buffer[-self._max_buffer_size:]
        
        logger.debug(f"Buffers updated. Current size: {len(self._price_buffer)}.")

    def get_buffered_dataframe(self, columns: List[str] = ['close', 'volume']) -> Optional[pd.DataFrame]:
        """
        Retrieves the buffered data as a Pandas DataFrame.
        The DataFrame will have a DatetimeIndex and specified columns.

        Args:
            columns: List of columns to include in the DataFrame (e.g., 'close', 'volume').

        Returns:
            A Pandas DataFrame, or None if buffers are empty.
        """
        if not self._price_buffer:
            return None
        
        # Create a unified list of records (timestamp, close, volume) for DataFrame creation
        # Assuming price buffer's timestamp is primary and volume buffer aligns
        combined_data = []
        for i in range(len(self._price_buffer)):
            ts, price_val = self._price_buffer[i]
            # Find corresponding volume, assuming order by timestamp.
            # For simplicity, we just use the volume at the same index.
            # For robust real-time streams, you'd ensure timestamps align.
            volume_val = self._volume_buffer[i][1] if i < len(self._volume_buffer) else 0 
            combined_data.append({
                'timestamp': ts,
                'close': price_val,
                'volume': volume_val
            })
        
        df = pd.DataFrame(combined_data).set_index('timestamp').sort_index()
        
        # Ensure only requested columns are present
        cols_to_return = [col for col in columns if col in df.columns]
        if not cols_to_return:
            logger.warning("No requested columns found in buffered DataFrame.")
            return None

        # Filter out NaN/None values from the index which can happen if timestamp is bad
        df = df[df.index.notna()]
        
        return df[cols_to_return]

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Returns the latest data point from the buffers as a dictionary.
        """
        if self._price_buffer:
            ts, price = self._price_buffer[-1]
            volume = self._volume_buffer[-1][1] if self._volume_buffer else 0
            return {'timestamp': ts, 'close': price, 'volume': volume}
        return None

    def get_buffer_size(self) -> int:
        """Returns the current number of data points in the buffer."""
        return len(self._price_buffer)
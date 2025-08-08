"""
Outcome data fetcher for model training.

This module fetches and prepares outcome data for training classification models:
- Historical price data retrieval
- Trade execution data
- Market event alignment
- Outcome period calculations
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class OutcomeRequest:
    """Request for outcome data."""
    symbol: str
    start_date: datetime
    end_date: datetime
    outcome_horizons: List[int]  # Hours to look forward
    include_intraday: bool = True
    include_overnight: bool = True
    min_price_change: float = 0.01  # Minimum price change to consider


@dataclass
class OutcomeData:
    """Outcome data result."""
    symbol: str
    timestamp: datetime
    entry_price: float
    
    # Outcome horizons (hours -> outcome)
    price_outcomes: Dict[int, float] = field(default_factory=dict)
    return_outcomes: Dict[int, float] = field(default_factory=dict)
    
    # Classification labels
    direction_outcomes: Dict[int, str] = field(default_factory=dict)  # 'up', 'down', 'flat'
    magnitude_outcomes: Dict[int, str] = field(default_factory=dict)  # 'small', 'medium', 'large'
    
    # Market context
    volume: Optional[float] = None
    volatility: Optional[float] = None
    market_hours: bool = True
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutcomeDataFetcher(ErrorHandlingMixin):
    """
    Fetches outcome data for training classification models.
    
    Features:
    - Multi-horizon outcome calculation
    - Market hours awareness
    - Volume and volatility context
    - Flexible labeling criteria
    - Efficient batch processing
    """
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize outcome data fetcher."""
        super().__init__()
        self.db_pool = db_pool
        
        # Default classification thresholds
        self._direction_thresholds = {
            'flat_threshold': 0.005,  # +/- 0.5%
        }
        
        self._magnitude_thresholds = {
            'small': 0.01,   # 1%
            'medium': 0.03,  # 3%
            'large': 0.05    # 5%
        }
    
    @timer
    async def fetch_outcome_data(
        self,
        request: OutcomeRequest
    ) -> List[OutcomeData]:
        """
        Fetch outcome data for the given request.
        
        Args:
            request: Outcome data request
            
        Returns:
            List of outcome data points
        """
        with self._handle_error("fetching outcome data"):
            # Fetch base price data
            price_data = await self._fetch_price_data(
                request.symbol,
                request.start_date,
                request.end_date
            )
            
            if price_data.empty:
                logger.warning(f"No price data found for {request.symbol}")
                return []
            
            # Calculate outcomes for each timestamp
            outcome_data = []
            
            for idx, row in price_data.iterrows():
                timestamp = row['timestamp']
                entry_price = row['close']
                
                # Skip if too close to end date
                max_horizon = max(request.outcome_horizons)
                if timestamp + timedelta(hours=max_horizon) > request.end_date:
                    continue
                
                # Calculate outcomes for each horizon
                outcomes = await self._calculate_outcomes(
                    request.symbol,
                    timestamp,
                    entry_price,
                    request.outcome_horizons,
                    request.min_price_change
                )
                
                if outcomes:
                    # Add market context
                    context = await self._get_market_context(
                        request.symbol,
                        timestamp
                    )
                    
                    outcome_data_point = OutcomeData(
                        symbol=request.symbol,
                        timestamp=timestamp,
                        entry_price=entry_price,
                        price_outcomes=outcomes['prices'],
                        return_outcomes=outcomes['returns'],
                        direction_outcomes=outcomes['directions'],
                        magnitude_outcomes=outcomes['magnitudes'],
                        volume=context.get('volume'),
                        volatility=context.get('volatility'),
                        market_hours=context.get('market_hours', True)
                    )
                    
                    outcome_data.append(outcome_data_point)
            
            logger.info(
                f"Fetched {len(outcome_data)} outcome data points for "
                f"{request.symbol} from {request.start_date} to {request.end_date}"
            )
            
            record_metric(
                'outcome_data_fetcher.data_points_fetched',
                len(outcome_data),
                tags={
                    'symbol': request.symbol,
                    'horizons': len(request.outcome_horizons)
                }
            )
            
            return outcome_data
    
    async def fetch_batch_outcome_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        outcome_horizons: List[int],
        max_concurrent: int = 10
    ) -> Dict[str, List[OutcomeData]]:
        """
        Fetch outcome data for multiple symbols in batch.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            outcome_horizons: Outcome horizons in hours
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping symbol to outcome data
        """
        with self._handle_error("fetching batch outcome data"):
            # Create requests
            requests = [
                OutcomeRequest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    outcome_horizons=outcome_horizons
                )
                for symbol in symbols
            ]
            
            # Process in batches
            results = {}
            
            for i in range(0, len(requests), max_concurrent):
                batch = requests[i:i + max_concurrent]
                
                # Process batch concurrently
                batch_results = await asyncio.gather(
                    *[self.fetch_outcome_data(req) for req in batch],
                    return_exceptions=True
                )
                
                # Collect results
                for request, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error fetching data for {request.symbol}: {result}"
                        )
                        results[request.symbol] = []
                    else:
                        results[request.symbol] = result
            
            total_points = sum(len(data) for data in results.values())
            
            logger.info(
                f"Batch fetched {total_points} total outcome data points "
                f"for {len(symbols)} symbols"
            )
            
            return results
    
    async def _fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch price data for the given period."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM market_data
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            
            rows = await conn.fetch(query, symbol, start_date, end_date)
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                for row in rows
            ])
            
            df.set_index('timestamp', inplace=True)
            
            return df
    
    async def _calculate_outcomes(
        self,
        symbol: str,
        entry_timestamp: datetime,
        entry_price: float,
        horizons: List[int],
        min_price_change: float
    ) -> Optional[Dict[str, Dict[int, Any]]]:
        """Calculate outcomes for all horizons."""
        outcomes = {
            'prices': {},
            'returns': {},
            'directions': {},
            'magnitudes': {}
        }
        
        for horizon in horizons:
            # Calculate target timestamp
            target_timestamp = entry_timestamp + timedelta(hours=horizon)
            
            # Get price at target time
            exit_price = await self._get_price_at_time(
                symbol, target_timestamp
            )
            
            if exit_price is None:
                continue
            
            # Calculate return
            price_return = (exit_price - entry_price) / entry_price
            
            # Skip if change too small
            if abs(price_return) < min_price_change:
                continue
            
            # Store outcomes
            outcomes['prices'][horizon] = exit_price
            outcomes['returns'][horizon] = price_return
            
            # Classify direction
            outcomes['directions'][horizon] = self._classify_direction(
                price_return
            )
            
            # Classify magnitude
            outcomes['magnitudes'][horizon] = self._classify_magnitude(
                abs(price_return)
            )
        
        # Return None if no valid outcomes
        if not outcomes['prices']:
            return None
        
        return outcomes
    
    async def _get_price_at_time(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[float]:
        """Get price at specific timestamp (or closest available)."""
        async with self.db_pool.acquire() as conn:
            # Try exact match first
            query = """
                SELECT close
                FROM market_data
                WHERE symbol = $1 AND timestamp = $2
            """
            
            price = await conn.fetchval(query, symbol, timestamp)
            
            if price is not None:
                return float(price)
            
            # Find closest timestamp within 1 hour
            query = """
                SELECT close, timestamp
                FROM market_data
                WHERE symbol = $1
                AND timestamp BETWEEN $2 AND $3
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $2)))
                LIMIT 1
            """
            
            hour_before = timestamp - timedelta(hours=1)
            hour_after = timestamp + timedelta(hours=1)
            
            row = await conn.fetchrow(query, symbol, hour_before, hour_after)
            
            if row:
                return float(row['close'])
            
            return None
    
    async def _get_market_context(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Get market context for the given timestamp."""
        context = {}
        
        async with self.db_pool.acquire() as conn:
            # Get volume and calculate volatility
            query = """
                SELECT 
                    volume,
                    close,
                    LAG(close) OVER (ORDER BY timestamp) as prev_close
                FROM market_data
                WHERE symbol = $1
                AND timestamp <= $2
                ORDER BY timestamp DESC
                LIMIT 20
            """
            
            rows = await conn.fetch(query, symbol, timestamp)
            
            if rows:
                # Current volume
                context['volume'] = float(rows[0]['volume']) if rows[0]['volume'] else None
                
                # Calculate recent volatility
                returns = []
                for row in rows:
                    if row['prev_close']:
                        ret = (row['close'] - row['prev_close']) / row['prev_close']
                        returns.append(ret)
                
                if len(returns) > 1:
                    context['volatility'] = float(np.std(returns))
                
                # Check if market hours (simplified)
                hour = timestamp.hour
                context['market_hours'] = 9 <= hour <= 16  # 9 AM to 4 PM
        
        return context
    
    def _classify_direction(
        self,
        price_return: float
    ) -> str:
        """Classify price movement direction."""
        threshold = self._direction_thresholds['flat_threshold']
        
        if price_return > threshold:
            return 'up'
        elif price_return < -threshold:
            return 'down'
        else:
            return 'flat'
    
    def _classify_magnitude(
        self,
        abs_return: float
    ) -> str:
        """Classify price movement magnitude."""
        if abs_return >= self._magnitude_thresholds['large']:
            return 'large'
        elif abs_return >= self._magnitude_thresholds['medium']:
            return 'medium'
        elif abs_return >= self._magnitude_thresholds['small']:
            return 'small'
        else:
            return 'minimal'
    
    def set_classification_thresholds(
        self,
        direction_thresholds: Optional[Dict[str, float]] = None,
        magnitude_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        """Update classification thresholds."""
        if direction_thresholds:
            self._direction_thresholds.update(direction_thresholds)
        
        if magnitude_thresholds:
            self._magnitude_thresholds.update(magnitude_thresholds)
        
        logger.info("Updated classification thresholds")
    
    async def get_outcome_statistics(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        horizon: int
    ) -> Dict[str, Any]:
        """Get statistics about outcomes for a symbol and horizon."""
        request = OutcomeRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            outcome_horizons=[horizon]
        )
        
        outcome_data = await self.fetch_outcome_data(request)
        
        if not outcome_data:
            return {}
        
        # Extract returns for the horizon
        returns = [
            data.return_outcomes.get(horizon)
            for data in outcome_data
            if horizon in data.return_outcomes
        ]
        
        if not returns:
            return {}
        
        # Extract directions and magnitudes
        directions = [
            data.direction_outcomes.get(horizon)
            for data in outcome_data
            if horizon in data.direction_outcomes
        ]
        
        magnitudes = [
            data.magnitude_outcomes.get(horizon)
            for data in outcome_data
            if horizon in data.magnitude_outcomes
        ]
        
        # Calculate statistics
        returns_array = np.array(returns)
        
        stats = {
            'total_samples': len(returns),
            'mean_return': float(np.mean(returns_array)),
            'std_return': float(np.std(returns_array)),
            'min_return': float(np.min(returns_array)),
            'max_return': float(np.max(returns_array)),
            'positive_returns': int(np.sum(returns_array > 0)),
            'negative_returns': int(np.sum(returns_array < 0)),
            'win_rate': float(np.mean(returns_array > 0)),
            
            # Direction distribution
            'direction_distribution': {
                direction: directions.count(direction)
                for direction in set(directions)
            },
            
            # Magnitude distribution
            'magnitude_distribution': {
                magnitude: magnitudes.count(magnitude)
                for magnitude in set(magnitudes)
            }
        }
        
        return stats
    
    async def export_outcome_data_to_csv(
        self,
        outcome_data: List[OutcomeData],
        file_path: str
    ) -> None:
        """Export outcome data to CSV file."""
        if not outcome_data:
            logger.warning("No outcome data to export")
            return
        
        # Convert to DataFrame
        rows = []
        
        for data in outcome_data:
            base_row = {
                'symbol': data.symbol,
                'timestamp': data.timestamp,
                'entry_price': data.entry_price,
                'volume': data.volume,
                'volatility': data.volatility,
                'market_hours': data.market_hours
            }
            
            # Add horizon-specific data
            for horizon in data.return_outcomes.keys():
                row = base_row.copy()
                row['horizon'] = horizon
                row['exit_price'] = data.price_outcomes.get(horizon)
                row['return'] = data.return_outcomes.get(horizon)
                row['direction'] = data.direction_outcomes.get(horizon)
                row['magnitude'] = data.magnitude_outcomes.get(horizon)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        logger.info(f"Exported {len(rows)} outcome records to {file_path}")
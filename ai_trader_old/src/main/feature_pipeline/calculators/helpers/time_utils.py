"""
Time and Market-Aware Utilities for Feature Calculators

Provides utilities for handling market hours, time decay calculations,
temporal feature engineering, and market session alignment.
"""

# Standard library imports
from datetime import time, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz

# Local imports
from main.utils.core import (
    ensure_utc,
    get_last_us_trading_day,
    get_logger,
    get_next_trading_day,
    is_trading_day,
)

logger = get_logger(__name__)

# Market timezone
MARKET_TZ = pytz.timezone("America/New_York")

# Trading session times (Eastern Time)
PREMARKET_START = time(4, 0)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
AFTERMARKET_END = time(20, 0)


def get_market_sessions(
    date: pd.Timestamp, market: str = "NYSE"
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get market session times for a given date.

    Args:
        date: Date to get sessions for
        market: Market calendar to use

    Returns:
        Dictionary of session names to (start, end) timestamps
    """
    try:
        # Ensure date is timezone-aware
        if date.tz is None:
            date = date.tz_localize("UTC")

        # Convert to market timezone
        market_date = date.astimezone(MARKET_TZ)
        date_only = market_date.date()

        sessions = {
            "premarket": (
                pd.Timestamp.combine(date_only, PREMARKET_START).tz_localize(MARKET_TZ),
                pd.Timestamp.combine(date_only, MARKET_OPEN).tz_localize(MARKET_TZ),
            ),
            "regular": (
                pd.Timestamp.combine(date_only, MARKET_OPEN).tz_localize(MARKET_TZ),
                pd.Timestamp.combine(date_only, MARKET_CLOSE).tz_localize(MARKET_TZ),
            ),
            "aftermarket": (
                pd.Timestamp.combine(date_only, MARKET_CLOSE).tz_localize(MARKET_TZ),
                pd.Timestamp.combine(date_only, AFTERMARKET_END).tz_localize(MARKET_TZ),
            ),
        }

        # Convert back to UTC
        for session_name, (start, end) in sessions.items():
            sessions[session_name] = (start.tz_convert("UTC"), end.tz_convert("UTC"))

        return sessions

    except Exception as e:
        logger.error(f"Error getting market sessions: {e}")
        return {}


def align_to_market_time(
    timestamp: pd.Timestamp, session: str = "regular", direction: str = "forward"
) -> pd.Timestamp:
    """
    Align timestamp to market session boundaries.

    Args:
        timestamp: Timestamp to align
        session: Market session to align to
        direction: Direction to align ('forward' or 'backward')

    Returns:
        Aligned timestamp
    """
    try:
        timestamp = ensure_utc(timestamp)

        # Get market sessions for this date
        sessions = get_market_sessions(timestamp)

        if session not in sessions:
            return timestamp

        session_start, session_end = sessions[session]

        # Check if already within session
        if session_start <= timestamp <= session_end:
            return timestamp

        # Align based on direction
        if direction == "forward":
            if timestamp < session_start:
                return session_start
            else:
                # Move to next trading day's session
                next_day = get_next_trading_day(timestamp)
                next_sessions = get_market_sessions(next_day)
                return next_sessions[session][0]
        elif timestamp > session_end:
            return session_end
        else:
            # Move to previous trading day's session
            prev_day = get_last_us_trading_day(timestamp)
            prev_sessions = get_market_sessions(prev_day)
            return prev_sessions[session][1]

    except Exception as e:
        logger.error(f"Error aligning to market time: {e}")
        return timestamp


def calculate_time_decay(
    base_value: float,
    elapsed_time: timedelta | float,
    half_life: timedelta | float,
    decay_type: str = "exponential",
) -> float:
    """
    Calculate time decay factor.

    Args:
        base_value: Initial value
        elapsed_time: Time elapsed (timedelta or hours)
        half_life: Half-life period (timedelta or hours)
        decay_type: Type of decay ('exponential', 'linear', 'power')

    Returns:
        Decayed value
    """
    try:
        # Convert to hours if needed
        if isinstance(elapsed_time, timedelta):
            elapsed_hours = elapsed_time.total_seconds() / 3600
        else:
            elapsed_hours = float(elapsed_time)

        if isinstance(half_life, timedelta):
            half_life_hours = half_life.total_seconds() / 3600
        else:
            half_life_hours = float(half_life)

        if half_life_hours <= 0:
            return base_value

        # Calculate decay factor
        if decay_type == "exponential":
            # Exponential decay: value * exp(-ln(2) * t / half_life)
            decay_factor = np.exp(-np.log(2) * elapsed_hours / half_life_hours)
        elif decay_type == "linear":
            # Linear decay: max(0, 1 - t / (2 * half_life))
            decay_factor = max(0, 1 - elapsed_hours / (2 * half_life_hours))
        elif decay_type == "power":
            # Power decay: (half_life / (half_life + t))^2
            decay_factor = (half_life_hours / (half_life_hours + elapsed_hours)) ** 2
        else:
            decay_factor = 1.0

        return base_value * decay_factor

    except Exception as e:
        logger.error(f"Error calculating time decay: {e}")
        return base_value


def create_time_windows(
    end_time: pd.Timestamp, windows: list[timedelta | int], window_type: str = "rolling"
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create time windows for analysis.

    Args:
        end_time: End time for windows
        windows: List of window sizes (timedelta or days)
        window_type: Type of windows ('rolling', 'expanding')

    Returns:
        List of (start, end) timestamp tuples
    """
    time_windows = []
    end_time = ensure_utc(end_time)

    for window in windows:
        # Convert to timedelta if needed
        if isinstance(window, int):
            window_delta = timedelta(days=window)
        else:
            window_delta = window

        if window_type == "rolling":
            # Fixed-size rolling window
            start_time = end_time - window_delta
            time_windows.append((start_time, end_time))
        elif window_type == "expanding":
            # Expanding window from beginning
            start_time = end_time - window_delta
            for i in range(len(time_windows) + 1):
                if i == 0:
                    time_windows.append((start_time, end_time))
                else:
                    # Expand previous window
                    prev_start, _ = time_windows[i - 1]
                    time_windows[i - 1] = (min(prev_start, start_time), end_time)

    return time_windows


def get_trading_days(
    start_date: pd.Timestamp, end_date: pd.Timestamp, market: str = "NYSE"
) -> pd.DatetimeIndex:
    """
    Get trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        market: Market calendar to use

    Returns:
        DatetimeIndex of trading days
    """
    try:
        # Get market calendar
        calendar = mcal.get_calendar(market)

        # Get valid trading days
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)

        # Extract dates
        trading_days = pd.DatetimeIndex(schedule.index).normalize()

        # Ensure UTC
        if trading_days.tz is None:
            trading_days = trading_days.tz_localize("UTC")
        else:
            trading_days = trading_days.tz_convert("UTC")

        return trading_days

    except Exception as e:
        logger.error(f"Error getting trading days: {e}")
        # Fallback to business days
        return pd.bdate_range(start=start_date, end=end_date, tz="UTC")


def calculate_business_days(
    start_date: pd.Timestamp, end_date: pd.Timestamp, market: str = "NYSE"
) -> int:
    """
    Calculate number of business days between dates.

    Args:
        start_date: Start date
        end_date: End date
        market: Market calendar to use

    Returns:
        Number of business days
    """
    trading_days = get_trading_days(start_date, end_date, market)
    return len(trading_days)


def is_market_hours(
    timestamp: pd.Timestamp, session: str = "regular", market: str = "NYSE"
) -> bool:
    """
    Check if timestamp is during market hours.

    Args:
        timestamp: Timestamp to check
        session: Market session to check
        market: Market calendar to use

    Returns:
        True if during specified market session
    """
    try:
        timestamp = ensure_utc(timestamp)

        # Check if trading day
        if not is_trading_day(timestamp):
            return False

        # Get market sessions
        sessions = get_market_sessions(timestamp, market)

        if session not in sessions:
            return False

        session_start, session_end = sessions[session]

        return session_start <= timestamp <= session_end

    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False


def get_time_of_day_weights(
    timestamps: pd.Series, weight_type: str = "linear", peak_hour: int = 10
) -> pd.Series:
    """
    Calculate time-of-day weights for features.

    Args:
        timestamps: Series of timestamps
        weight_type: Type of weighting ('linear', 'gaussian', 'uniform')
        peak_hour: Peak hour for gaussian weighting

    Returns:
        Series of weights
    """
    try:
        # Convert to market timezone
        market_times = timestamps.dt.tz_convert(MARKET_TZ)

        # Extract hour of day
        hours = market_times.dt.hour + market_times.dt.minute / 60

        if weight_type == "linear":
            # Linear decay from market open
            market_open_hour = MARKET_OPEN.hour + MARKET_OPEN.minute / 60
            market_close_hour = MARKET_CLOSE.hour + MARKET_CLOSE.minute / 60

            # Normalize to [0, 1] based on distance from open
            weights = 1 - np.abs(hours - market_open_hour) / (market_close_hour - market_open_hour)
            weights = np.clip(weights, 0, 1)

        elif weight_type == "gaussian":
            # Gaussian centered at peak hour
            sigma = 2.0  # Standard deviation in hours
            weights = np.exp(-0.5 * ((hours - peak_hour) / sigma) ** 2)

        else:  # uniform
            weights = pd.Series(1.0, index=timestamps.index)

        return weights

    except Exception as e:
        logger.error(f"Error calculating time-of-day weights: {e}")
        return pd.Series(1.0, index=timestamps.index)


def calculate_time_weighted_average(
    values: pd.Series,
    timestamps: pd.Series,
    decay_type: str = "exponential",
    half_life_hours: float = 24.0,
) -> float:
    """
    Calculate time-weighted average with decay.

    Args:
        values: Series of values
        timestamps: Series of timestamps
        decay_type: Type of time decay
        half_life_hours: Half-life for decay in hours

    Returns:
        Time-weighted average
    """
    try:
        if len(values) == 0 or len(timestamps) == 0:
            return np.nan

        # Ensure timestamps are sorted
        sorted_idx = timestamps.argsort()
        values = values.iloc[sorted_idx]
        timestamps = timestamps.iloc[sorted_idx]

        # Calculate time since last observation
        current_time = timestamps.iloc[-1]
        time_diffs = (current_time - timestamps).dt.total_seconds() / 3600  # Hours

        # Calculate weights based on time decay
        weights = pd.Series(index=values.index, dtype=float)

        for i, elapsed in enumerate(time_diffs):
            weights.iloc[i] = calculate_time_decay(1.0, elapsed, half_life_hours, decay_type)

        # Calculate weighted average
        weighted_sum = (values * weights).sum()
        weight_sum = weights.sum()

        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return np.nan

    except Exception as e:
        logger.error(f"Error calculating time-weighted average: {e}")
        return np.nan


def create_temporal_features(
    timestamps: pd.Series, reference_time: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Create temporal features from timestamps.

    Args:
        timestamps: Series of timestamps
        reference_time: Reference time for relative features

    Returns:
        DataFrame with temporal features
    """
    features = pd.DataFrame(index=timestamps.index)

    try:
        # Ensure timezone
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("UTC")

        # Convert to market timezone for feature extraction
        market_times = timestamps.dt.tz_convert(MARKET_TZ)

        # Basic temporal features
        features["hour"] = market_times.dt.hour
        features["minute"] = market_times.dt.minute
        features["day_of_week"] = market_times.dt.dayofweek
        features["day_of_month"] = market_times.dt.day
        features["week_of_year"] = market_times.dt.isocalendar().week
        features["month"] = market_times.dt.month
        features["quarter"] = market_times.dt.quarter

        # Market session features
        features["is_premarket"] = False
        features["is_regular_hours"] = False
        features["is_aftermarket"] = False

        for idx, ts in enumerate(timestamps):
            sessions = get_market_sessions(ts)

            if "premarket" in sessions:
                pm_start, pm_end = sessions["premarket"]
                if pm_start <= ts <= pm_end:
                    features.loc[idx, "is_premarket"] = True

            if "regular" in sessions:
                reg_start, reg_end = sessions["regular"]
                if reg_start <= ts <= reg_end:
                    features.loc[idx, "is_regular_hours"] = True

            if "aftermarket" in sessions:
                am_start, am_end = sessions["aftermarket"]
                if am_start <= ts <= am_end:
                    features.loc[idx, "is_aftermarket"] = True

        # Time since market open
        features["minutes_since_open"] = 0
        for idx, ts in enumerate(market_times):
            if features.loc[idx, "is_regular_hours"]:
                open_time = pd.Timestamp.combine(ts.date(), MARKET_OPEN).tz_localize(MARKET_TZ)
                features.loc[idx, "minutes_since_open"] = (ts - open_time).total_seconds() / 60

        # Time until market close
        features["minutes_until_close"] = 0
        for idx, ts in enumerate(market_times):
            if features.loc[idx, "is_regular_hours"]:
                close_time = pd.Timestamp.combine(ts.date(), MARKET_CLOSE).tz_localize(MARKET_TZ)
                features.loc[idx, "minutes_until_close"] = (close_time - ts).total_seconds() / 60

        # Relative features if reference time provided
        if reference_time is not None:
            reference_time = ensure_utc(reference_time)
            features["hours_since_reference"] = (
                timestamps - reference_time
            ).dt.total_seconds() / 3600
            features["days_since_reference"] = (
                timestamps - reference_time
            ).dt.total_seconds() / 86400

        # Cyclical encoding for periodic features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        return features

    except Exception as e:
        logger.error(f"Error creating temporal features: {e}")
        return features

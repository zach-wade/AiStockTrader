# File: ai_trader/utils/time_helpers.py

from datetime import date, timedelta, datetime, timezone
from typing import Tuple, List, Optional
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from pytz import timezone as ZoneInfo  # Fallback for older Python
import pandas as pd
import pandas_market_calendars as mcal


def ensure_utc(dt: Optional[datetime]) -> datetime:
    """
    Ensure datetime is UTC timezone-aware.
    
    Args:
        dt: Datetime to convert, or None to get current UTC time
        
    Returns:
        UTC timezone-aware datetime
    """
    if dt is None:
        return datetime.now(timezone.utc)
    
    # Handle date objects (convert to datetime at midnight UTC)
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())
        return dt.replace(tzinfo=timezone.utc)
    
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


def get_last_us_trading_day(check_date: date) -> date:
    """
    Finds the most recent day the NYSE was open, accounting for weekends AND holidays.
    
    :param check_date: The date to check from (usually today).
    :return: The date of the last valid trading session.
    """
    nyse = mcal.get_calendar('NYSE')
    # Get the schedule for the past 10 calendar days to be safe
    schedule = nyse.schedule(start_date=check_date - timedelta(days=10), end_date=check_date)
    
    # The last valid trading day is the last date in the schedule's index
    if not schedule.empty:
        return schedule.index[-1].date()
    
    # Fallback in case of an issue (e.g., long market closure)
    return check_date


def is_market_open(tz: str = "America/New_York") -> bool:
    """
    Checks if the US stock market is currently open.
    (This is a simplified example).
    """
    tz_obj = ZoneInfo(tz)
    now = datetime.now(timezone.utc).astimezone(tz_obj)
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


def get_market_hours(date: datetime, market: str = 'NYSE') -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get market open and close times in UTC for a specific date.
    
    Args:
        date: Date to check
        market: Market calendar to use (default: NYSE)
        
    Returns:
        Tuple of (open_time, close_time) in UTC, or (None, None) if market closed
    """
    cal = mcal.get_calendar(market)
    schedule = cal.schedule(start_date=date, end_date=date)
    
    if schedule.empty:
        return None, None  # Market closed
    
    # mcal returns times in market timezone, convert to UTC
    open_time = ensure_utc(pd.Timestamp(schedule.iloc[0]['market_open']))
    close_time = ensure_utc(pd.Timestamp(schedule.iloc[0]['market_close']))
    return open_time, close_time


def get_trading_days_between(start: datetime, end: datetime, market: str = 'NYSE') -> List[datetime]:
    """
    Get all trading days between two dates as UTC datetime objects.
    
    Args:
        start: Start date
        end: End date  
        market: Market calendar to use
        
    Returns:
        List of trading days as UTC datetime objects
    """
    cal = mcal.get_calendar(market)
    schedule = cal.schedule(start_date=start, end_date=end)
    
    # Return as UTC datetime objects (midnight UTC)
    trading_days = []
    for ts in schedule.index:
        # Convert to UTC midnight
        day = pd.Timestamp(ts).normalize()  # Set to midnight
        trading_days.append(ensure_utc(day.to_pydatetime()))
    
    return trading_days


def is_trading_day(date: datetime, market: str = 'NYSE') -> bool:
    """
    Check if a specific date is a trading day.
    
    Args:
        date: Date to check
        market: Market calendar to use
        
    Returns:
        True if it's a trading day
    """
    cal = mcal.get_calendar(market)
    schedule = cal.schedule(start_date=date, end_date=date)
    return not schedule.empty


def get_next_trading_day(date: datetime, market: str = 'NYSE') -> datetime:
    """
    Get the next trading day after the given date.
    
    Args:
        date: Starting date
        market: Market calendar to use
        
    Returns:
        Next trading day as UTC datetime
    """
    cal = mcal.get_calendar(market)
    # Look ahead up to 10 days to handle long weekends/holidays
    end_date = date + timedelta(days=10)
    schedule = cal.schedule(start_date=date + timedelta(days=1), end_date=end_date)
    
    if not schedule.empty:
        next_day = pd.Timestamp(schedule.index[0]).normalize()
        return ensure_utc(next_day.to_pydatetime())
    
    # Fallback - should rarely happen
    return ensure_utc((date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
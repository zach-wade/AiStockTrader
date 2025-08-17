"""
Financial Data Repository Interfaces

Interfaces for financial statements, guidance, ratings, and dividends.
"""

# Standard library imports
from abc import abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

from .base import IRepository, OperationResult


class IFinancialsRepository(IRepository):
    """Interface for financial statements repository."""

    @abstractmethod
    async def get_financial_statements(
        self, symbol: str, statement_type: str, period: str = "quarterly", limit: int = 8
    ) -> pd.DataFrame:
        """
        Get financial statements for a symbol.

        Args:
            symbol: Stock symbol
            statement_type: Type (income, balance, cashflow)
            period: Period type (quarterly, annual)
            limit: Number of periods to retrieve

        Returns:
            DataFrame with financial data
        """
        pass

    @abstractmethod
    async def store_financial_statement(
        self, symbol: str, statement_type: str, period_date: datetime, data: dict[str, Any]
    ) -> OperationResult:
        """
        Store a financial statement.

        Args:
            symbol: Stock symbol
            statement_type: Type of statement
            period_date: Period end date
            data: Statement data

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_latest_financials(self, symbol: str) -> dict[str, Any] | None:
        """
        Get latest financial data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest financial data or None
        """
        pass


class IGuidanceRepository(IRepository):
    """Interface for company guidance repository."""

    @abstractmethod
    async def get_guidance(
        self, symbol: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """
        Get company guidance data.

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with guidance data
        """
        pass

    @abstractmethod
    async def store_guidance(
        self, symbol: str, guidance_date: datetime, guidance_type: str, data: dict[str, Any]
    ) -> OperationResult:
        """
        Store company guidance.

        Args:
            symbol: Stock symbol
            guidance_date: Guidance issue date
            guidance_type: Type of guidance
            data: Guidance data

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_latest_guidance(self, symbol: str) -> dict[str, Any] | None:
        """
        Get latest guidance for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest guidance or None
        """
        pass


class IRatingsRepository(IRepository):
    """Interface for analyst ratings repository."""

    @abstractmethod
    async def get_ratings(
        self, symbol: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """
        Get analyst ratings.

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with ratings
        """
        pass

    @abstractmethod
    async def store_rating(
        self,
        symbol: str,
        rating_date: datetime,
        analyst: str,
        rating: str,
        price_target: float | None = None,
    ) -> OperationResult:
        """
        Store an analyst rating.

        Args:
            symbol: Stock symbol
            rating_date: Rating issue date
            analyst: Analyst or firm name
            rating: Rating value
            price_target: Optional price target

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_consensus_rating(self, symbol: str) -> dict[str, Any] | None:
        """
        Get consensus rating for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Consensus rating data or None
        """
        pass

    @abstractmethod
    async def get_price_targets(self, symbol: str, active_only: bool = True) -> dict[str, float]:
        """
        Get price target summary.

        Args:
            symbol: Stock symbol
            active_only: Only include recent ratings

        Returns:
            Dict with high, low, mean targets
        """
        pass


class IDividendsRepository(IRepository):
    """Interface for dividends and corporate actions repository."""

    @abstractmethod
    async def get_dividends(
        self, symbol: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """
        Get dividend history.

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with dividend data
        """
        pass

    @abstractmethod
    async def store_dividend(
        self, symbol: str, ex_date: datetime, amount: float, payment_date: datetime | None = None
    ) -> OperationResult:
        """
        Store dividend information.

        Args:
            symbol: Stock symbol
            ex_date: Ex-dividend date
            amount: Dividend amount
            payment_date: Optional payment date

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_corporate_actions(
        self,
        symbol: str,
        action_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Get corporate actions.

        Args:
            symbol: Stock symbol
            action_type: Optional action type filter
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with corporate actions
        """
        pass

    @abstractmethod
    async def get_dividend_yield(
        self, symbol: str, as_of_date: datetime | None = None
    ) -> float | None:
        """
        Calculate dividend yield.

        Args:
            symbol: Stock symbol
            as_of_date: Optional calculation date

        Returns:
            Dividend yield or None
        """
        pass

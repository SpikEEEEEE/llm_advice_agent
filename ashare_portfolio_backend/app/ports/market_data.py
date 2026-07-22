from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from app.domain.models import SymbolMarketSnapshot


class MarketDataProvider(Protocol):
    name: str

    def latest_completed_session(self, as_of: datetime) -> date:
        """Return the latest market session whose closing data should be available."""

    def next_session(self, after: date) -> date:
        """Return the first trading session strictly after ``after``."""

    def load_symbol(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        """Load and validate all market inputs needed for one symbol."""


class MarketDataError(RuntimeError):
    pass


class ProviderConfigurationError(MarketDataError):
    pass


class DataUnavailableError(MarketDataError):
    pass


class StaleDataError(MarketDataError):
    pass

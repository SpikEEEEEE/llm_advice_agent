from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Protocol

import pandas as pd

from app.adapters.tushare import TushareMarketDataProvider
from app.domain.models import SymbolMarketSnapshot
from app.ports.market_data import DataUnavailableError


class HistoricalDataFeed(Protocol):
    name: str

    def prepare(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
    ) -> tuple[date, ...]:
        """Load the calendar and point-in-time-safe price histories."""

    def snapshot(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        """Return only information observable by the historical cutoff."""

    def price(
        self,
        symbol: str,
        session: date,
        field: str,
    ) -> Decimal | None:
        """Return an exact-session total-return price for simulation."""


class TushareHistoricalDataFeed:
    """Historical feed using Tushare data and a start-normalized return price."""

    name = "tushare"

    def __init__(self, provider: TushareMarketDataProvider) -> None:
        self.provider = provider
        self._histories: dict[str, pd.DataFrame] = {}
        self._sessions: tuple[date, ...] = ()

    @staticmethod
    def _total_return_history(
        bars: pd.DataFrame,
        first_session: date,
    ) -> pd.DataFrame:
        output = bars.copy().sort_values("date").reset_index(drop=True)
        factors = pd.to_numeric(
            output.get(
                "adj_factor",
                pd.Series(1.0, index=output.index),
            ),
            errors="coerce",
        )
        first_rows = output[output["date"] <= first_session]
        first_index = (
            first_rows.index[-1]
            if not first_rows.empty
            else output.index[0]
        )
        base_factor = factors.loc[first_index]
        if pd.isna(base_factor) or float(base_factor) <= 0:
            factors = pd.Series(1.0, index=output.index)
            base_factor = 1.0
        factors = factors.ffill().bfill().fillna(float(base_factor))
        scales = factors / float(base_factor)
        for column in ("open", "high", "low", "close", "vwap"):
            if column in output.columns:
                output[column] = (
                    pd.to_numeric(output[column], errors="coerce") * scales
                )
        output["adjusted_close"] = output["close"]
        return output

    def prepare(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
    ) -> tuple[date, ...]:
        sessions = self.provider.sessions_between(start, end)
        if len(sessions) < 2:
            raise DataUnavailableError(
                "Backtest requires at least two trading sessions"
            )
        history_start = start - timedelta(
            days=self.provider.settings.market_history_days
        )
        histories: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        for symbol in symbols:
            try:
                raw = self.provider.load_history(
                    symbol,
                    history_start,
                    sessions[-1],
                )
                histories[symbol] = self._total_return_history(
                    raw,
                    sessions[0],
                )
            except Exception as exc:
                failures[symbol] = str(exc)
        if not histories:
            raise DataUnavailableError(
                f"No symbol history could be loaded: {failures}"
            )
        self._histories = histories
        self._sessions = sessions
        return sessions

    def snapshot(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        history = self._histories.get(symbol)
        if history is None:
            raise DataUnavailableError(
                f"No prepared historical prices for {symbol}"
            )
        eligible = history[history["date"] <= data_date].copy()
        exact = eligible[eligible["date"] == data_date]
        if exact.empty:
            raise DataUnavailableError(
                f"No exact historical price for {symbol} on {data_date}"
            )
        raw_snapshot = self.provider.load_symbol(
            symbol,
            data_date,
            as_of,
        )
        return replace(
            raw_snapshot,
            reference_price=Decimal(str(exact.iloc[-1]["close"])),
            bars=eligible.reset_index(drop=True),
        )

    def price(
        self,
        symbol: str,
        session: date,
        field: str,
    ) -> Decimal | None:
        if field not in {"open", "close"}:
            raise ValueError("Backtest price field must be open or close")
        history = self._histories.get(symbol)
        if history is None:
            return None
        exact = history[history["date"] == session]
        if exact.empty:
            return None
        value = pd.to_numeric(exact.iloc[-1].get(field), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            return None
        return Decimal(str(float(value)))

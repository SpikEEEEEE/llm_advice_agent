from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int
    available_shares: int
    average_cost: Decimal
    holding_days: int | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    portfolio_id: str
    version: int
    name: str
    cash: Decimal
    positions: tuple[Position, ...]

    def position_map(self) -> dict[str, Position]:
        return {position.symbol: position for position in self.positions}


@dataclass(frozen=True)
class SymbolMarketSnapshot:
    symbol: str
    data_date: date
    reference_price: Decimal
    bars: Any
    news: tuple[dict[str, Any], ...] = ()
    fundamentals: dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime | None = None
    data_quality_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionInput:
    run_id: str
    portfolio: PortfolioSnapshot
    mode: str
    as_of: datetime
    data_date: date
    valid_for_session: date
    universe_version: str
    symbols: tuple[str, ...]
    market: dict[str, SymbolMarketSnapshot]
    unavailable_symbols: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RawDecisionBundle:
    decisions: dict[str, dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

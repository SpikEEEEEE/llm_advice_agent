from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import SYMBOL_PATTERN


def _normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("symbol must look like 600519.SH, 000001.SZ, or 430047.BJ")
    return symbol


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PositionPayload(StrictModel):
    symbol: str
    shares: int = Field(gt=0, le=2_000_000_000)
    available_shares: int | None = Field(
        default=None, ge=0, le=2_000_000_000
    )
    average_cost: Decimal = Field(ge=0, le=Decimal("1000000000"))
    holding_days: int | None = Field(default=None, ge=0, le=36500)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="after")
    def validate_available_shares(self) -> "PositionPayload":
        if self.available_shares is None:
            self.available_shares = self.shares
        if self.available_shares > self.shares:
            raise ValueError("available_shares cannot exceed shares")
        return self


class PortfolioWriteRequest(StrictModel):
    name: str = Field(default="Current portfolio", min_length=1, max_length=100)
    cash: Decimal = Field(ge=0, le=Decimal("1000000000000000"))
    positions: list[PositionPayload] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def reject_duplicate_symbols(self) -> "PortfolioWriteRequest":
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("positions contain duplicate symbols")
        return self


class PortfolioResponse(StrictModel):
    id: str
    name: str
    cash: Decimal
    positions: list[PositionPayload]
    version: int
    created_at: datetime
    updated_at: datetime


class DecisionRunCreateRequest(StrictModel):
    portfolio_id: str = Field(min_length=1, max_length=64)
    mode: Literal["holdings_only", "rebalance"] = "rebalance"
    as_of: datetime | None = None
    universe: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional point-in-time universe for this run. When omitted, the "
            "server-configured universe is used."
        ),
    )

    @field_validator("universe")
    @classmethod
    def normalize_universe(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        symbols = [_normalize_symbol(symbol) for symbol in value]
        if len(symbols) != len(set(symbols)):
            raise ValueError("universe contains duplicate symbols")
        return symbols


class DecisionRunResponse(StrictModel):
    id: str
    portfolio_id: str
    portfolio_version: int
    status: Literal[
        "pending",
        "fetching_data",
        "building_features",
        "calling_llm",
        "validating",
        "completed",
        "degraded",
        "failed",
    ]
    mode: Literal["holdings_only", "rebalance"]
    as_of: datetime
    universe_version: str
    universe: list[str]
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class UniverseResponse(StrictModel):
    version: str
    symbols: list[str]


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: str

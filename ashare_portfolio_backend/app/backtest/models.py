from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal


RebalanceFrequency = Literal["daily", "weekly", "monthly"]


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    initial_cash: Decimal = Decimal("1000000")
    rebalance_frequency: RebalanceFrequency = "monthly"
    initial_rebalance: bool = True
    max_decisions: int = 24
    commission_rate: Decimal = Decimal("0.0003")
    minimum_commission: Decimal = Decimal("5")
    sell_stamp_duty_before_cutover: Decimal = Decimal("0.001")
    sell_stamp_duty_after_cutover: Decimal = Decimal("0.0005")
    stamp_duty_cutover: date = date(2023, 8, 28)
    slippage_bps: Decimal = Decimal("5")
    reuse_sale_proceeds: bool = False

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Backtest end must be later than start")
        if self.initial_cash <= 0:
            raise ValueError("Backtest initial cash must be positive")
        if self.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError(
                "Rebalance frequency must be daily, weekly, or monthly"
            )
        if self.max_decisions < 1:
            raise ValueError("Backtest max decisions must be at least 1")
        non_negative = (
            self.commission_rate,
            self.minimum_commission,
            self.sell_stamp_duty_before_cutover,
            self.sell_stamp_duty_after_cutover,
            self.slippage_bps,
        )
        if any(value < 0 or not value.is_finite() for value in non_negative):
            raise ValueError(
                "Backtest cost assumptions must be finite and non-negative"
            )
        if self.slippage_bps > Decimal("10000"):
            raise ValueError("Backtest slippage cannot exceed 10000 basis points")

    def stamp_duty(self, session: date) -> Decimal:
        return (
            self.sell_stamp_duty_after_cutover
            if session >= self.stamp_duty_cutover
            else self.sell_stamp_duty_before_cutover
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: (
                value.isoformat()
                if isinstance(value, date)
                else float(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in payload.items()
        }


@dataclass
class BacktestHolding:
    symbol: str
    shares: int
    average_cost: Decimal
    acquired_on: date


@dataclass
class BacktestTrade:
    session: date
    decision_session: date
    symbol: str
    side: Literal["buy", "sell"]
    requested_shares: int
    executed_shares: int
    reference_open: Decimal
    execution_price: Decimal
    notional: Decimal
    commission: Decimal
    stamp_duty: Decimal
    cash_after: Decimal
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: (
                value.isoformat()
                if isinstance(value, date)
                else float(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in payload.items()
        }


@dataclass
class BacktestResult:
    run_id: str
    config: BacktestConfig
    universe: tuple[str, ...]
    universe_version: str
    data_source: str
    decision_engine: str
    metrics: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    trades: list[BacktestTrade]
    decisions: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "universe": list(self.universe),
            "universe_version": self.universe_version,
            "data_source": self.data_source,
            "decision_engine": self.decision_engine,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }

    def write(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                self.summary(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_csv(output_dir / "equity_curve.csv", self.equity_curve)
        self._write_csv(
            output_dir / "trades.csv",
            [trade.to_dict() for trade in self.trades],
        )
        (output_dir / "decisions.json").write_text(
            json.dumps(
                self.decisions,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_dir

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def finite_metric(value: float | Decimal | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None

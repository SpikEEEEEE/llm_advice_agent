from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from app.backtest.cache import BacktestDecisionCache
from app.backtest.engine import PortfolioBacktester
from app.backtest.models import BacktestConfig
from app.domain.models import RawDecisionBundle, SymbolMarketSnapshot
from app.domain.risk import AShareRiskPolicy

from .helpers import make_settings


SYMBOL = "600519.SH"
SESSIONS = (
    date(2025, 1, 2),
    date(2025, 1, 3),
    date(2025, 1, 31),
    date(2025, 2, 3),
    date(2025, 2, 28),
    date(2025, 3, 3),
)


class FakeHistoricalFeed:
    name = "fake_history"

    def __init__(self) -> None:
        self.snapshot_dates: list[date] = []
        self.prices = {
            session: {
                "open": Decimal(str(10 + index * 0.4)),
                "close": Decimal(str(10.2 + index * 0.4)),
            }
            for index, session in enumerate(SESSIONS)
        }

    def prepare(self, symbols, start, end):
        assert symbols == (SYMBOL,)
        assert start == SESSIONS[0]
        assert end == SESSIONS[-1]
        return SESSIONS

    def snapshot(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        assert symbol == SYMBOL
        assert as_of.date() == data_date
        self.snapshot_dates.append(data_date)
        eligible = [
            session for session in SESSIONS if session <= data_date
        ]
        bars = pd.DataFrame(
            {
                "date": eligible,
                "open": [
                    float(self.prices[item]["open"])
                    for item in eligible
                ],
                "high": [
                    float(self.prices[item]["close"]) + 0.1
                    for item in eligible
                ],
                "low": [
                    float(self.prices[item]["open"]) - 0.1
                    for item in eligible
                ],
                "close": [
                    float(self.prices[item]["close"])
                    for item in eligible
                ],
                "volume": [100000.0] * len(eligible),
                "vwap": [
                    float(self.prices[item]["close"])
                    for item in eligible
                ],
            }
        )
        return SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=self.prices[data_date]["close"],
            bars=bars,
            fundamentals={"pe_ttm": 20.0},
            news=(),
            data_quality_warnings=(),
        )

    def price(
        self,
        symbol: str,
        session: date,
        field: str,
    ) -> Decimal | None:
        assert symbol == SYMBOL
        return self.prices.get(session, {}).get(field)


class FiftyPercentDecisionEngine:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, decision_input, on_stage=None) -> RawDecisionBundle:
        self.calls += 1
        snapshot = decision_input.market[SYMBOL]
        position = decision_input.portfolio.position_map().get(SYMBOL)
        current = float(snapshot.reference_price) * (
            position.shares if position else 0
        )
        total_assets = float(decision_input.portfolio.cash) + current
        target = total_assets * 0.5
        if abs(target - current) <= 0.01:
            action = "hold"
        elif target > current:
            action = "increase"
        elif target <= 0:
            action = "close"
        else:
            action = "decrease"
        return RawDecisionBundle(
            decisions={
                SYMBOL: {
                    "action": action,
                    "target_cash_amount": target,
                    "confidence": 0.9,
                    "reasons": ["scripted fifty-percent allocation"],
                }
            },
            meta={"calls": 1, "engine": "scripted"},
        )


def test_backtest_uses_next_open_and_writes_auditable_results(tmp_path):
    settings = make_settings(tmp_path)
    feed = FakeHistoricalFeed()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=feed,
        decision_engine=FiftyPercentDecisionEngine(),
        risk_policy=AShareRiskPolicy(settings),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        rebalance_frequency="monthly",
    )

    result = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert feed.snapshot_dates == [
        SESSIONS[0],
        SESSIONS[2],
        SESSIONS[4],
    ]
    assert result.trades
    assert result.trades[0].decision_session == SESSIONS[0]
    assert result.trades[0].session == SESSIONS[1]
    assert result.trades[0].side == "buy"
    assert result.trades[0].execution_price > result.trades[0].reference_open
    assert result.metrics["decision_count"] == 3
    assert result.metrics["llm_provider_calls"] == 3
    assert result.metrics["total_return"] > 0
    assert result.metrics["equal_weight_universe_return"] > 0
    assert result.metrics["total_fees"] > 0

    output = result.write(tmp_path / "result")
    assert (output / "summary.json").exists()
    assert (output / "equity_curve.csv").exists()
    assert (output / "trades.csv").exists()
    assert (output / "decisions.json").exists()


def test_rebalance_schedule_never_decides_without_a_next_session():
    monthly = PortfolioBacktester.decision_sessions(
        SESSIONS,
        "monthly",
        True,
    )
    daily = PortfolioBacktester.decision_sessions(
        SESSIONS,
        "daily",
        False,
    )

    assert monthly == {SESSIONS[0], SESSIONS[2], SESSIONS[4]}
    assert daily == set(SESSIONS[:-1])
    assert SESSIONS[-1] not in monthly
    assert SESSIONS[-1] not in daily


def test_backtest_decision_cache_replays_without_new_model_calls(tmp_path):
    settings = make_settings(tmp_path)
    scripted_engine = FiftyPercentDecisionEngine()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=scripted_engine,
        decision_cache=BacktestDecisionCache(
            tmp_path / "decision_cache",
            settings,
        ),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        rebalance_frequency="monthly",
    )

    first = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )
    second = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert scripted_engine.calls == 3
    assert first.metrics["decision_cache_hits"] == 0
    assert second.metrics["decision_cache_hits"] == 3
    assert second.metrics["llm_provider_calls"] == 0
    assert second.metrics["final_equity"] == first.metrics["final_equity"]


def test_backtest_decision_budget_blocks_accidental_expensive_run(tmp_path):
    settings = make_settings(tmp_path)
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=FiftyPercentDecisionEngine(),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        rebalance_frequency="daily",
        max_decisions=2,
    )

    with pytest.raises(ValueError, match="exceeding"):
        backtester.run(
            config=config,
            universe=(SYMBOL,),
            universe_version="test_v1",
        )

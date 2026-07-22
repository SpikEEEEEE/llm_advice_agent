from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    RawDecisionBundle,
    SymbolMarketSnapshot,
)
from app.domain.risk import AShareRiskPolicy, _confidence
from .helpers import make_settings


def _market(
    symbol: str,
    price: str,
    warnings: tuple[str, ...] = (),
) -> SymbolMarketSnapshot:
    return SymbolMarketSnapshot(
        symbol=symbol,
        data_date=date(2026, 7, 21),
        reference_price=Decimal(price),
        bars=pd.DataFrame({"date": [date(2026, 7, 21)], "close": [float(price)]}),
        data_quality_warnings=warnings,
    )


def test_risk_policy_applies_board_lot_and_t_plus_one(tmp_path):
    settings = make_settings(tmp_path)
    portfolio = PortfolioSnapshot(
        portfolio_id="pf_test",
        version=1,
        name="test",
        cash=Decimal("5000"),
        positions=(
            Position(
                symbol="600519.SH",
                shares=200,
                available_shares=100,
                average_cost=Decimal("9"),
            ),
        ),
    )
    market = {
        "600519.SH": _market("600519.SH", "10"),
        "300750.SZ": _market("300750.SZ", "12"),
    }
    decision_input = DecisionInput(
        run_id="run_test",
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_date=date(2026, 7, 21),
        valid_for_session=date(2026, 7, 22),
        universe_version="test_v1",
        symbols=("600519.SH", "300750.SZ"),
        market=market,
    )
    bundle = RawDecisionBundle(
        decisions={
            "600519.SH": {
                "action": "close",
                "target_cash_amount": 0,
                "confidence": 0.7,
                "reasons": ["exit"],
            },
            "300750.SZ": {
                "action": "increase",
                "target_cash_amount": 2500,
                "confidence": 0.9,
                "reasons": ["enter"],
            },
        },
        meta={"calls": 1},
    )

    result = AShareRiskPolicy(settings).apply(decision_input, bundle)
    decisions = {item["symbol"]: item for item in result["decisions"]}

    assert decisions["600519.SH"]["target_shares"] == 100
    assert decisions["600519.SH"]["action"] == "decrease"
    assert "SELL_CAPPED_BY_AVAILABLE_SHARES" in decisions["600519.SH"]["risk_flags"]
    assert decisions["300750.SZ"]["target_shares"] == 200
    assert decisions["300750.SZ"]["delta_shares"] % 100 == 0


def test_missing_price_forces_safe_hold(tmp_path):
    settings = make_settings(tmp_path)
    portfolio = PortfolioSnapshot(
        portfolio_id="pf_test",
        version=1,
        name="test",
        cash=Decimal("1000"),
        positions=(Position("600519.SH", 100, 100, Decimal("10")),),
    )
    decision_input = DecisionInput(
        run_id="run_test",
        portfolio=portfolio,
        mode="holdings_only",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
        data_date=date(2026, 7, 21),
        valid_for_session=date(2026, 7, 22),
        universe_version="test",
        symbols=("600519.SH",),
        market={},
        unavailable_symbols={"600519.SH": "stale price"},
    )

    result = AShareRiskPolicy(settings).apply(
        decision_input,
        RawDecisionBundle(
            decisions={"600519.SH": {"action": "close", "target_cash_amount": 0}}
        ),
    )

    decision = result["decisions"][0]
    assert decision["action"] == "hold"
    assert decision["target_shares"] == 100
    assert decision["risk_flags"] == ["DATA_UNAVAILABLE"]


def test_unpriced_existing_holding_freezes_all_new_buys(tmp_path):
    settings = make_settings(tmp_path)
    portfolio = PortfolioSnapshot(
        portfolio_id="pf_test",
        version=1,
        name="test",
        cash=Decimal("10000"),
        positions=(Position("600519.SH", 100, 100, Decimal("10")),),
    )
    decision_input = DecisionInput(
        run_id="run_test",
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
        data_date=date(2026, 7, 21),
        valid_for_session=date(2026, 7, 22),
        universe_version="test",
        symbols=("600519.SH", "300750.SZ"),
        market={
            "300750.SZ": _market(
                "300750.SZ",
                "10",
                warnings=("NEWS_PROVIDER_UNAVAILABLE",),
            )
        },
        unavailable_symbols={"600519.SH": "stale price"},
    )
    bundle = RawDecisionBundle(
        decisions={
            "300750.SZ": {
                "action": "increase",
                "target_cash_amount": 5000,
                "confidence": 0.9,
            }
        },
        meta={"calls": 1},
    )

    result = AShareRiskPolicy(settings).apply(decision_input, bundle)
    decisions = {item["symbol"]: item for item in result["decisions"]}

    assert decisions["300750.SZ"]["target_shares"] == 0
    assert (
        "NEW_BUYS_FROZEN_MISSING_HOLDING_PRICE"
        in decisions["300750.SZ"]["risk_flags"]
    )
    assert result["portfolio_summary"]["valuation_complete"] is False


def test_data_quality_warning_blocks_increase_but_keeps_raw_decision(tmp_path):
    settings = make_settings(tmp_path)
    portfolio = PortfolioSnapshot(
        portfolio_id="pf_test",
        version=1,
        name="test",
        cash=Decimal("10000"),
        positions=(),
    )
    decision_input = DecisionInput(
        run_id="run_test",
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
        data_date=date(2026, 7, 21),
        valid_for_session=date(2026, 7, 22),
        universe_version="test",
        symbols=("300750.SZ",),
        market={
            "300750.SZ": _market(
                "300750.SZ",
                "10",
                warnings=("NEWS_PROVIDER_UNAVAILABLE",),
            )
        },
    )
    raw = {
        "action": "increase",
        "target_cash_amount": 5000,
        "confidence": 0.9,
        "reasons": ["model wanted to buy"],
    }
    bundle = RawDecisionBundle(
        decisions={"300750.SZ": raw},
        meta={"calls": 1},
    )

    result = AShareRiskPolicy(settings).apply(decision_input, bundle)
    decision = result["decisions"][0]

    assert decision["action"] == "hold"
    assert decision["target_shares"] == 0
    assert decision["raw_decision"] == raw
    assert "INCREASE_BLOCKED_BY_DATA_QUALITY" in decision["risk_flags"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_is_never_prioritized(value):
    assert _confidence(value) == 0.0

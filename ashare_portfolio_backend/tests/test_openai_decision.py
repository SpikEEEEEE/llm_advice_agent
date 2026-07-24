from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.adapters.openai_decision import DecisionOutputError, OpenAIDecisionEngine
from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    SymbolMarketSnapshot,
)
from .helpers import make_settings


def _input(symbols: tuple[str, ...] = ("600519.SH",)) -> DecisionInput:
    data_date = date(2026, 7, 21)
    dates = [data_date - timedelta(days=offset) for offset in range(39, -1, -1)]
    market = {}
    for symbol in symbols:
        bars = pd.DataFrame(
            {
                "date": dates,
                "open": [9.5] * 40,
                "high": [10.5] * 40,
                "low": [9.0] * 40,
                "close": [10.0] * 40,
                "volume": [100000.0] * 40,
                "vwap": [10.0] * 40,
            }
        )
        market[symbol] = SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=Decimal("10"),
            bars=bars,
            news=(
                {
                    "title": "Known item",
                    "description": "Point-in-time news",
                    "published_utc": "2026-07-21T07:00:00+00:00",
                    "api_source": "tushare",
                },
            ),
            fundamentals={"pe_ttm": 20.0, "roe": 12.0},
        )
    return DecisionInput(
        run_id="run_adapter",
        portfolio=PortfolioSnapshot(
            portfolio_id="pf_adapter",
            version=1,
            name="adapter",
            cash=Decimal("10000"),
            positions=(
                Position(
                    symbol="600519.SH",
                    shares=100,
                    available_shares=80,
                    average_cost=Decimal("8"),
                    holding_days=30,
                ),
            ),
        ),
        mode="holdings_only",
        as_of=datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_date=data_date,
        valid_for_session=date(2026, 7, 22),
        universe_version="test",
        symbols=symbols,
        market=market,
    )


def _response(decisions):
    content = json.dumps({"decisions": decisions})
    return SimpleNamespace(
        id="chat_test",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 123}),
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def test_direct_llm_adapter_builds_its_own_point_in_time_features(tmp_path):
    client = FakeClient(
        [
            _response(
                [
                    {
                        "symbol": "600519.SH",
                        "action": "hold",
                        "target_position_value": 1000,
                        "confidence": 0.8,
                        "reasons": ["valuation is balanced"],
                    }
                ]
            )
        ]
    )
    engine = OpenAIDecisionEngine(make_settings(tmp_path), client=client)

    bundle = engine.decide(_input())

    request = client.completions.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    context = json.loads(request["messages"][1]["content"])
    symbol = context["symbols"][0]
    assert symbol["market"]["close_7d"][-1] == 10.0
    assert symbol["position"]["average_cost"] == 8.0
    assert symbol["position"]["available_shares"] == 80
    assert symbol["position"]["unrealized_pnl_pct"] == 0.25
    assert symbol["fundamentals"]["pe_ttm"] == 20.0
    assert bundle.decisions["600519.SH"]["action"] == "hold"
    assert bundle.meta["calls"] == 1
    assert bundle.meta["response_format"] == "json_schema"


def test_json_schema_unsupported_falls_back_to_strict_json_object(tmp_path):
    error = RuntimeError("response_format json_schema is unsupported")
    error.status_code = 400  # type: ignore[attr-defined]
    client = FakeClient(
        [
            error,
            _response(
                [
                    {
                        "symbol": "600519.SH",
                        "action": "close",
                        "target_position_value": 0,
                        "confidence": 0.6,
                        "reasons": ["risk reduction"],
                    }
                ]
            ),
        ]
    )

    bundle = OpenAIDecisionEngine(make_settings(tmp_path), client=client).decide(
        _input()
    )

    assert len(client.completions.requests) == 2
    assert client.completions.requests[1]["response_format"] == {
        "type": "json_object"
    }
    assert "target_position_value" in (
        client.completions.requests[1]["messages"][0]["content"]
    )
    assert bundle.meta["calls"] == 2
    assert bundle.meta["response_format"] == "json_object"
    assert bundle.warnings == (
        "LLM_JSON_SCHEMA_UNSUPPORTED_USED_JSON_OBJECT",
    )


def test_no_market_context_is_an_explicit_failed_safe_hold(tmp_path):
    client = FakeClient([])
    bundle = OpenAIDecisionEngine(
        make_settings(tmp_path),
        client=client,
    ).decide(replace(_input(), market={}))

    assert bundle.decisions == {}
    assert bundle.meta["engine"] == "failed_safe_hold"
    assert bundle.meta["decision_quality"] == "failed"
    assert bundle.meta["calls"] == 0
    assert client.completions.requests == []


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"decisions": []}),
        json.dumps(
            {
                "decisions": [
                    {
                        "symbol": "000001.SZ",
                        "action": "hold",
                        "target_position_value": 0,
                        "confidence": 0.5,
                        "reasons": ["wrong symbol"],
                    }
                ]
            }
        ),
        json.dumps(
            {
                "decisions": [
                    {
                        "symbol": "600519.SH",
                        "action": "hold",
                        "target_position_value": "1000",
                        "confidence": 0.5,
                        "reasons": ["numeric string is forbidden"],
                    }
                ]
            }
        ),
    ],
)
def test_invalid_or_mismatched_model_output_is_rejected(tmp_path, content):
    response = SimpleNamespace(
        id="bad",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )
    engine = OpenAIDecisionEngine(make_settings(tmp_path), client=FakeClient([response]))

    with pytest.raises(DecisionOutputError):
        engine.decide(_input())

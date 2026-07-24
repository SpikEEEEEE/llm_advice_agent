from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.adapters.decision_engine_factory import build_decision_engine
from app.adapters.openai_decision import OpenAIDecisionEngine
from app.adapters.portfolio_multi_agent import (
    OpenAIStructuredAgentClient,
    PortfolioMultiAgentDecisionEngine,
)
from app.agents.portfolio_graph import PortfolioAgentGraph
from app.agents.portfolio_schemas import PoolAnalystReport
from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    SymbolMarketSnapshot,
)

from .helpers import make_settings


SYMBOLS = ("600519.SH", "300750.SZ")


def _decision_input() -> DecisionInput:
    data_date = date(2026, 7, 21)
    dates = [
        data_date - timedelta(days=offset)
        for offset in range(39, -1, -1)
    ]
    market: dict[str, SymbolMarketSnapshot] = {}
    for symbol in SYMBOLS:
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
                    "title": f"Point-in-time item for {symbol}",
                    "description": "Known before the cutoff",
                    "published_utc": "2026-07-21T07:00:00+00:00",
                    "api_source": "test",
                },
            ),
            fundamentals={"pe_ttm": 20.0, "roe": 0.12},
        )
    return DecisionInput(
        run_id="run_multi_agent",
        portfolio=PortfolioSnapshot(
            portfolio_id="pf_multi_agent",
            version=1,
            name="multi-agent",
            cash=Decimal("10000"),
            positions=(
                Position(
                    symbol="600519.SH",
                    shares=100,
                    available_shares=100,
                    average_cost=Decimal("8"),
                    holding_days=30,
                ),
            ),
        ),
        mode="rebalance",
        as_of=datetime(
            2026,
            7,
            21,
            16,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ),
        data_date=data_date,
        valid_for_session=date(2026, 7, 22),
        universe_version="test",
        symbols=SYMBOLS,
        market=market,
    )


class ScriptedPortfolioAgents:
    def __init__(self, *, invalid_first_trader: bool = False) -> None:
        self.invalid_first_trader = invalid_first_trader
        self.calls: list[str] = []
        self.payloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def invoke(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[Any],
    ) -> Any:
        del system_prompt
        with self._lock:
            self.calls.append(agent_name)
            self.payloads[agent_name] = payload

        role = agent_name.removesuffix("_repair")
        if role.endswith("_analyst"):
            scores = {
                "technical_analyst": (30.0, 70.0),
                "fundamental_analyst": (60.0, 40.0),
                "news_analyst": (10.0, 80.0),
            }[role]
            output = {
                "assessments": [
                    {
                        "symbol": symbol,
                        "score": scores[index],
                        "confidence": 0.8,
                        "stance": "positive",
                        "evidence": [f"{role} evidence"],
                        "risks": [],
                    }
                    for index, symbol in enumerate(SYMBOLS)
                ],
                "summary": f"{role} compared the complete pool",
            }
        elif role in {"bull_researcher", "bear_researcher"}:
            conviction = 60.0 if role == "bull_researcher" else -30.0
            output = {
                "views": [
                    {
                        "symbol": symbol,
                        "conviction": conviction,
                        "reasons": [f"{role} reason"],
                        "risks": [f"{role} risk"],
                    }
                    for symbol in SYMBOLS
                ],
                "portfolio_argument": f"{role} cross-sectional case",
                "preferred_cash_ratio": 0.5,
            }
        elif role == "research_manager":
            output = {
                "candidates": [
                    {
                        "symbol": "300750.SZ",
                        "rating": "buy",
                        "conviction": 0.8,
                        "priority": 1,
                        "thesis": ["stronger relative opportunity"],
                        "risks": ["volatility"],
                    },
                    {
                        "symbol": "600519.SH",
                        "rating": "underweight",
                        "conviction": 0.7,
                        "priority": 2,
                        "thesis": ["lower relative opportunity"],
                        "risks": ["opportunity cost"],
                    },
                ],
                "preferred_cash_ratio": 0.7,
                "portfolio_thesis": "Prefer the stronger candidate with ample cash",
            }
        elif role == "portfolio_trader":
            cash_weight = (
                0.5
                if self.invalid_first_trader
                and agent_name == "portfolio_trader"
                else 0.7
            )
            output = {
                "targets": [
                    {
                        "symbol": "600519.SH",
                        "target_weight": 0.1,
                        "confidence": 0.7,
                        "reasons": ["reduce the lower-ranked holding"],
                    },
                    {
                        "symbol": "300750.SZ",
                        "target_weight": 0.2,
                        "confidence": 0.8,
                        "reasons": ["fund the higher-ranked opportunity"],
                    },
                ],
                "cash_weight": cash_weight,
                "rationale": "One shared target-weight vector",
            }
        elif role.endswith("_risk_reviewer"):
            stance = role.removesuffix("_risk_reviewer")
            output = {
                "stance": stance,
                "approve": True,
                "suggested_cash_weight": 0.7,
                "adjustments": [],
                "portfolio_risk": f"{stance} risk view",
            }
        elif role == "portfolio_manager":
            output = {
                "targets": [
                    {
                        "symbol": "600519.SH",
                        "target_weight": 0.05,
                        "confidence": 0.75,
                        "reasons": ["trim in favor of a better relative idea"],
                    },
                    {
                        "symbol": "300750.SZ",
                        "target_weight": 0.2,
                        "confidence": 0.85,
                        "reasons": ["highest cross-sectional conviction"],
                    },
                ],
                "cash_weight": 0.75,
                "rationale": "Final portfolio after three risk perspectives",
            }
        else:
            raise AssertionError(f"Unexpected agent: {agent_name}")
        return response_model.model_validate(output)

    def meta(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "warnings": [],
            "trace": [{"agent": name} for name in self.calls],
        }


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _analyst_response(response_id: str) -> Any:
    content = (
        '{"assessments":[{"symbol":"600519.SH","score":1.0,'
        '"confidence":0.8,"stance":"positive","evidence":["evidence"],'
        '"risks":[]}],"summary":"summary"}'
    )
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        usage=None,
    )


def test_pool_graph_produces_one_coherent_allocation(tmp_path):
    agents = ScriptedPortfolioAgents()
    settings = replace(
        make_settings(tmp_path),
        decision_engine_mode="portfolio_multi_agent",
    )
    stages: list[str] = []
    engine = PortfolioMultiAgentDecisionEngine(
        settings,
        agent_client=agents,
    )

    bundle = engine.decide(_decision_input(), on_stage=stages.append)

    assert stages == ["building_features", "calling_llm"]
    assert set(bundle.decisions) == set(SYMBOLS)
    assert bundle.decisions["600519.SH"]["action"] == "decrease"
    assert bundle.decisions["600519.SH"]["target_cash_amount"] == 550.0
    assert bundle.decisions["300750.SZ"]["action"] == "increase"
    assert bundle.decisions["300750.SZ"]["target_cash_amount"] == 2200.0
    assert bundle.meta["calls"] == 11
    assert bundle.meta["agent_artifacts"]["shortlist"] == list(SYMBOLS)

    for role in ("technical", "fundamental", "news"):
        symbols = {
            item["symbol"]
            for item in agents.payloads[f"{role}_analyst"]["symbols"]
        }
        assert symbols == set(SYMBOLS)


def test_semantically_invalid_weights_are_repaired(tmp_path):
    agents = ScriptedPortfolioAgents(invalid_first_trader=True)
    settings = replace(
        make_settings(tmp_path),
        decision_engine_mode="portfolio_multi_agent",
        multi_agent_semantic_retries=1,
    )

    bundle = PortfolioMultiAgentDecisionEngine(
        settings,
        agent_client=agents,
    ).decide(_decision_input())

    assert "portfolio_trader" in agents.calls
    assert "portfolio_trader_repair" in agents.calls
    assert bundle.meta["calls"] == 12
    assert bundle.meta["agent_artifacts"]["trader_proposal"]["cash_weight"] == 0.7


def test_shortlist_never_drops_an_existing_holding(tmp_path):
    agents = ScriptedPortfolioAgents()
    settings = replace(
        make_settings(tmp_path),
        multi_agent_shortlist_size=1,
    )
    graph = PortfolioAgentGraph(settings, agents)
    state = graph.prepare(_decision_input())
    state.combined_scores = {
        "600519.SH": {"score": -80.0, "confidence": 0.2},
        "300750.SZ": {"score": 90.0, "confidence": 0.9},
    }

    graph._select_shortlist(state)

    assert state.shortlist == ["600519.SH"]


def test_decision_engine_factory_preserves_original_engine_by_default(tmp_path):
    default_engine = build_decision_engine(make_settings(tmp_path))
    multi_agent_engine = build_decision_engine(
        replace(
            make_settings(tmp_path),
            decision_engine_mode="portfolio_multi_agent",
        )
    )

    assert isinstance(default_engine, OpenAIDecisionEngine)
    assert isinstance(multi_agent_engine, PortfolioMultiAgentDecisionEngine)


def test_provider_call_budget_and_trace_are_scoped_to_each_run(tmp_path):
    completions = FakeCompletions(
        [_analyst_response("first"), _analyst_response("second")]
    )
    provider = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    settings = replace(
        make_settings(tmp_path),
        multi_agent_max_calls=1,
    )
    template = OpenAIStructuredAgentClient(settings, client=provider)

    for _ in range(2):
        session = template.new_run_session()
        report = session.invoke(
            agent_name="technical_analyst",
            system_prompt="Return the contract",
            payload={"symbols": [{"symbol": "600519.SH"}]},
            response_model=PoolAnalystReport,
        )
        assert report.assessments[0].symbol == "600519.SH"
        assert session.meta()["calls"] == 1

    assert len(completions.requests) == 2
    assert completions.requests[0]["response_format"]["type"] == "json_schema"
    assert completions.requests[0]["response_format"]["json_schema"]["strict"] is True

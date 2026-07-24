from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.adapters.decision_engine_factory import build_decision_engine
from app.adapters.openai_decision import OpenAIDecisionEngine
from app.adapters.portfolio_multi_agent import (
    OpenAIStructuredAgentClient,
    PortfolioAgentOutputError,
    PortfolioMultiAgentDecisionEngine,
)
from app.agents.portfolio_allocator import DeterministicPortfolioAllocator
from app.agents.portfolio_graph import (
    PortfolioAgentGraph,
    PortfolioAgentGraphError,
)
from app.agents.portfolio_schemas import PoolAnalystReport, PortfolioProposal
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
    def __init__(
        self,
        *,
        invalid_first_trader: bool = False,
        failed_analysts: set[str] | None = None,
        failed_risk_reviewers: set[str] | None = None,
    ) -> None:
        self.invalid_first_trader = invalid_first_trader
        self.failed_analysts = failed_analysts or set()
        self.failed_risk_reviewers = failed_risk_reviewers or set()
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
        if role in self.failed_analysts or role in self.failed_risk_reviewers:
            raise RuntimeError(f"scripted failure for {role}")
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
                        "key_signal": f"{role} signal",
                        "risk_flags": [],
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
        '"confidence":0.8,"stance":"positive","key_signal":"signal",'
        '"risk_flags":[]}],"summary":"summary"}'
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


def test_empty_market_context_returns_failed_safe_hold(tmp_path):
    decision_input = replace(_decision_input(), market={})
    bundle = PortfolioMultiAgentDecisionEngine(
        make_settings(tmp_path),
        agent_client=ScriptedPortfolioAgents(),
    ).decide(decision_input)

    assert bundle.decisions == {}
    assert bundle.meta["engine"] == "failed_safe_hold"
    assert bundle.meta["decision_quality"] == "failed"
    assert bundle.meta["provider_attempts"] == 0
    assert bundle.meta["stage_health"]["preparation"]["status"] == "failed"


def test_arithmetic_inconsistency_is_normalized_without_llm_repair(tmp_path):
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
    assert "portfolio_trader_repair" not in agents.calls
    assert bundle.meta["calls"] == 11
    assert bundle.meta["agent_artifacts"]["raw_trader_proposal"]["cash_weight"] == 0.5
    assert bundle.meta["agent_artifacts"]["trader_proposal"]["cash_weight"] == 0.7
    assert (
        bundle.meta["agent_artifacts"]["allocation_normalization"]["trader"][
            "normalized_cash_weight"
        ]
        == 0.7
    )


def test_single_analyst_cannot_proceed(tmp_path):
    agents = ScriptedPortfolioAgents(
        failed_analysts={
            "fundamental_analyst",
            "news_analyst",
        }
    )
    graph = PortfolioAgentGraph(make_settings(tmp_path), agents)
    state = graph.prepare(_decision_input())

    with pytest.raises(
        PortfolioAgentGraphError,
        match=r"analyst quorum not met \(1/2\)",
    ):
        graph.run_prepared(state)

    assert "bull_researcher" not in agents.calls
    assert state.analysis_coverage == 0.35
    assert state.stage_health["analysts"]["analysis_coverage"] == 0.35


def test_two_of_three_analysts_use_fixed_weights_and_are_reduce_only(tmp_path):
    agents = ScriptedPortfolioAgents(
        failed_analysts={"news_analyst"},
    )
    state = PortfolioAgentGraph(make_settings(tmp_path), agents).run(
        _decision_input()
    )

    assert state.decision_quality == "degraded"
    assert state.analysis_coverage == 0.75
    assert state.combined_scores["600519.SH"]["score"] == 27.6
    assert state.combined_scores["600519.SH"]["confidence"] == 0.6
    assert state.stage_health["analysts"]["successful_count"] == 2
    assert state.final_allocation is not None
    targets = {
        target.symbol: target.target_weight
        for target in state.final_allocation.targets
    }
    assert targets["600519.SH"] <= state.current_weights["600519.SH"]
    assert targets["300750.SZ"] == 0.0
    assert state.artifacts()["allocation_normalization"]["final"][
        "reduce_only"
    ] is True

    all_cash_input = replace(
        _decision_input(),
        portfolio=PortfolioSnapshot(
            portfolio_id="pf_all_cash",
            version=1,
            name="all-cash",
            cash=Decimal("11000"),
            positions=(),
        ),
    )
    all_cash_state = PortfolioAgentGraph(
        make_settings(tmp_path),
        ScriptedPortfolioAgents(failed_analysts={"news_analyst"}),
    ).run(all_cash_input)
    assert all_cash_state.final_allocation is not None
    assert all(
        target.target_weight == 0.0
        for target in all_cash_state.final_allocation.targets
    )
    assert all_cash_state.final_allocation.cash_weight == 1.0


def test_incomplete_pool_preparation_is_reduce_only(tmp_path):
    decision_input = _decision_input()
    incomplete_portfolio = replace(
        decision_input.portfolio,
        positions=(
            *decision_input.portfolio.positions,
            Position(
                symbol="000001.SZ",
                shares=100,
                available_shares=100,
                average_cost=Decimal("10"),
                holding_days=10,
            ),
        ),
    )
    incomplete_input = replace(
        decision_input,
        portfolio=incomplete_portfolio,
        symbols=(*decision_input.symbols, "000001.SZ"),
        unavailable_symbols={"000001.SZ": "missing historical price"},
    )

    state = PortfolioAgentGraph(
        make_settings(tmp_path),
        ScriptedPortfolioAgents(),
    ).run(incomplete_input)

    assert state.stage_health["preparation"]["status"] == "degraded"
    assert state.decision_quality == "degraded"
    assert state.final_allocation is not None
    targets = {
        target.symbol: target.target_weight
        for target in state.final_allocation.targets
    }
    assert targets["300750.SZ"] == 0.0
    assert any(
        "Pool preparation was incomplete" in warning
        for warning in state.warnings
    )


def test_malformed_symbol_snapshot_degrades_pool_without_crashing(tmp_path):
    decision_input = _decision_input()
    malformed_symbol = "000001.SZ"
    template = decision_input.market[SYMBOLS[0]]
    malformed_snapshot = replace(
        template,
        symbol=malformed_symbol,
        bars=pd.DataFrame(
            {
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.0],
            }
        ),
    )
    malformed_input = replace(
        decision_input,
        symbols=(*decision_input.symbols, malformed_symbol),
        market={
            **decision_input.market,
            malformed_symbol: malformed_snapshot,
        },
    )

    bundle = PortfolioMultiAgentDecisionEngine(
        make_settings(tmp_path),
        agent_client=ScriptedPortfolioAgents(),
    ).decide(malformed_input)

    preparation = bundle.meta["stage_health"]["preparation"]
    assert bundle.meta["decision_quality"] == "degraded"
    assert preparation["status"] == "degraded"
    assert preparation["failed_symbols"] == [malformed_symbol]
    assert preparation["failure_categories"][malformed_symbol] == "KeyError"
    assert bundle.decisions[malformed_symbol]["action"] == "hold"
    assert bundle.decisions[malformed_symbol]["target_cash_amount"] == 0.0


def test_unexpected_preparation_failure_returns_sanitized_safe_hold(
    tmp_path,
    monkeypatch,
):
    secret = "TOP_SECRET_PREPARATION_DETAIL"

    def fail_prepare(self, decision_input):
        del self, decision_input
        raise RuntimeError(secret)

    monkeypatch.setattr(PortfolioAgentGraph, "prepare", fail_prepare)
    bundle = PortfolioMultiAgentDecisionEngine(
        make_settings(tmp_path),
        agent_client=ScriptedPortfolioAgents(),
    ).decide(_decision_input())

    assert bundle.decisions == {}
    assert bundle.meta["engine"] == "failed_safe_hold"
    assert bundle.meta["decision_quality"] == "failed"
    assert bundle.meta["calls"] == 0
    assert bundle.meta["failure_category"] == "RuntimeError"
    assert secret not in " ".join(bundle.warnings)


def test_concurrent_agent_failures_do_not_persist_exception_text(tmp_path):
    secret = "TOP_SECRET_PROVIDER_RESPONSE_MUST_NOT_BE_PERSISTED"

    class SensitiveFailureAgents(ScriptedPortfolioAgents):
        def invoke(
            self,
            *,
            agent_name: str,
            system_prompt: str,
            payload: dict[str, Any],
            response_model: type[Any],
        ) -> Any:
            role = agent_name.removesuffix("_repair")
            if role == "news_analyst":
                raise RuntimeError(secret)
            if role == "aggressive_risk_reviewer":
                raise PortfolioAgentOutputError(
                    "invalid provider output",
                    category="schema_validation",
                    response_model="PortfolioRiskReview",
                    validation_errors=[
                        {
                            "loc": ["adjustments", 0, "symbol"],
                            "msg": secret,
                        }
                    ],
                )
            return super().invoke(
                agent_name=agent_name,
                system_prompt=system_prompt,
                payload=payload,
                response_model=response_model,
            )

    state = PortfolioAgentGraph(
        make_settings(tmp_path),
        SensitiveFailureAgents(),
    ).run(_decision_input())
    persisted = " ".join(state.warnings)

    assert secret not in persisted
    assert "RuntimeError" in persisted
    assert "schema_validation" in persisted
    assert "validation_error_count=1" in persisted


def test_custom_agent_meta_cannot_persist_arbitrary_text(tmp_path):
    secret = "TOP_SECRET_CUSTOM_AGENT_META"

    class MaliciousMetaAgents(ScriptedPortfolioAgents):
        def meta(self) -> dict[str, Any]:
            base = super().meta()
            return {
                **base,
                "warnings": [secret],
                "trace": [
                    {
                        "agent": secret,
                        "status": secret,
                        "error": secret,
                    }
                ],
            }

    bundle = PortfolioMultiAgentDecisionEngine(
        make_settings(tmp_path),
        agent_client=MaliciousMetaAgents(),
    ).decide(_decision_input())
    persisted = json.dumps(
        {
            "meta": bundle.meta,
            "warnings": bundle.warnings,
        },
        ensure_ascii=False,
    )

    assert secret not in persisted
    assert bundle.meta["calls"] == 11
    assert bundle.meta["agent_trace"] == []


def test_risk_quorum_failure_cannot_authorize_buys(tmp_path):
    agents = ScriptedPortfolioAgents(
        failed_risk_reviewers={
            "neutral_risk_reviewer",
            "conservative_risk_reviewer",
        }
    )
    state = PortfolioAgentGraph(make_settings(tmp_path), agents).run(
        _decision_input()
    )

    assert state.risk_quorum_met is False
    assert state.decision_quality == "degraded"
    assert "portfolio_manager" not in agents.calls
    assert state.stage_health["portfolio_manager"]["status"] == "safe_fallback"
    assert state.final_allocation is not None
    targets = {
        target.symbol: target.target_weight
        for target in state.final_allocation.targets
    }
    assert targets["600519.SH"] <= state.current_weights["600519.SH"]
    assert targets["300750.SZ"] == 0.0


def test_risk_quorum_allows_healthy_allocation_with_two_reviews(tmp_path):
    agents = ScriptedPortfolioAgents(
        failed_risk_reviewers={"aggressive_risk_reviewer"},
    )
    state = PortfolioAgentGraph(make_settings(tmp_path), agents).run(
        _decision_input()
    )

    assert state.risk_quorum_met is True
    assert state.decision_quality == "healthy"
    assert state.stage_health["risk_team"]["status"] == "degraded"
    assert "portfolio_manager" in agents.calls
    assert state.final_allocation is not None
    targets = {
        target.symbol: target.target_weight
        for target in state.final_allocation.targets
    }
    assert targets["300750.SZ"] == 0.2


def test_deterministic_allocator_enforces_all_hard_constraints():
    allocator = DeterministicPortfolioAllocator(
        minimum_cash_ratio=0.2,
        maximum_position_ratio=0.3,
        maximum_positions=2,
    )

    def proposal(order: tuple[str, ...]) -> PortfolioProposal:
        confidence = {"000001.SZ": 0.8, "000002.SZ": 0.9, "000003.SZ": 0.7}
        return PortfolioProposal.model_validate(
            {
                "targets": [
                    {
                        "symbol": symbol,
                        "target_weight": 0.9,
                        "confidence": confidence[symbol],
                        "reasons": ["ranked intent"],
                    }
                    for symbol in order
                ],
                "cash_weight": 0.1,
                "rationale": "Deliberately violates hard constraints",
            }
        )

    first, first_audit = allocator.normalize(
        proposal(("000003.SZ", "000001.SZ", "000002.SZ")),
        expected_symbols=["000003.SZ", "000001.SZ", "000002.SZ"],
        current_weights={},
        reduce_only=False,
        output_model=PortfolioProposal,
    )
    second, second_audit = allocator.normalize(
        proposal(("000002.SZ", "000003.SZ", "000001.SZ")),
        expected_symbols=["000002.SZ", "000003.SZ", "000001.SZ"],
        current_weights={},
        reduce_only=False,
        output_model=PortfolioProposal,
    )

    assert first.model_dump() == second.model_dump()
    assert first_audit.as_dict() == second_audit.as_dict()
    assert [target.symbol for target in first.targets] == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ]
    assert sum(target.target_weight > 0 for target in first.targets) == 2
    assert all(target.target_weight <= 0.3 for target in first.targets)
    assert first.cash_weight == 0.4
    assert sum(target.target_weight for target in first.targets) + first.cash_weight == 1.0
    assert first_audit.position_limit_drops == ("000003.SZ",)


def test_complete_twenty_symbol_pool_reaches_one_healthy_allocation(tmp_path):
    symbols = tuple(f"{600000 + index:06d}.SH" for index in range(20))
    base_input = _decision_input()
    template_snapshot = next(iter(base_input.market.values()))
    market = {
        symbol: replace(
            template_snapshot,
            symbol=symbol,
            news=(
                {
                    "title": f"Point-in-time item for {symbol}",
                    "description": "Known before the cutoff",
                    "published_utc": "2026-07-21T07:00:00+00:00",
                    "api_source": "test",
                },
            ),
        )
        for symbol in symbols
    }
    decision_input = replace(
        base_input,
        symbols=symbols,
        market=market,
        portfolio=replace(
            base_input.portfolio,
            positions=(
                Position(
                    symbol=symbols[0],
                    shares=100,
                    available_shares=100,
                    average_cost=Decimal("8"),
                    holding_days=30,
                ),
            ),
        ),
    )

    class FullPoolAgents:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @staticmethod
        def _symbols(payload: dict[str, Any], role: str) -> list[str]:
            if role.endswith("_analyst"):
                return [str(item["symbol"]) for item in payload["symbols"]]
            if role == "bull_researcher":
                rows = payload["shortlist"]
            elif role == "bear_researcher":
                rows = payload["consensus"]["shortlist"]
            elif role == "research_manager":
                rows = payload["consensus"]["shortlist"]
            elif role == "portfolio_trader":
                rows = payload["research_plan"]["candidates"]
            elif role.endswith("_risk_reviewer"):
                rows = payload["proposal"]["targets"]
            else:
                rows = payload["trader_proposal"]["targets"]
            return [str(item["symbol"]) for item in rows]

        def invoke(
            self,
            *,
            agent_name: str,
            system_prompt: str,
            payload: dict[str, Any],
            response_model: type[Any],
        ) -> Any:
            del system_prompt
            self.calls.append(agent_name)
            role = agent_name.removesuffix("_repair")
            selected = self._symbols(payload, role)
            if role.endswith("_analyst"):
                output = {
                    "assessments": [
                        {
                            "symbol": symbol,
                            "score": float(50 - index),
                            "confidence": 0.8,
                            "stance": "positive",
                            "key_signal": f"compact signal {index}",
                            "risk_flags": ["volatility"],
                        }
                        for index, symbol in enumerate(selected)
                    ],
                    "summary": "Compact comparison of the complete pool",
                }
            elif role in {"bull_researcher", "bear_researcher"}:
                output = {
                    "views": [
                        {
                            "symbol": symbol,
                            "conviction": (
                                50.0 if role == "bull_researcher" else -20.0
                            ),
                            "reasons": ["relative evidence"],
                            "risks": ["uncertainty"],
                        }
                        for symbol in selected
                    ],
                    "portfolio_argument": "Cross-sectional portfolio case",
                    "preferred_cash_ratio": 0.6,
                }
            elif role == "research_manager":
                output = {
                    "candidates": [
                        {
                            "symbol": symbol,
                            "rating": "overweight",
                            "conviction": 0.7,
                            "priority": index + 1,
                            "thesis": ["ranked opportunity"],
                            "risks": ["uncertainty"],
                        }
                        for index, symbol in enumerate(selected)
                    ],
                    "preferred_cash_ratio": 0.6,
                    "portfolio_thesis": "Allocate across the bounded shortlist",
                }
            elif role in {"portfolio_trader", "portfolio_manager"}:
                output = {
                    "targets": [
                        {
                            "symbol": symbol,
                            "target_weight": 0.05,
                            "confidence": 0.7,
                            "reasons": ["bounded allocation intent"],
                        }
                        for symbol in selected
                    ],
                    "cash_weight": 0.6,
                    "rationale": "Deterministic allocator enforces hard limits",
                }
            else:
                output = {
                    "stance": role.removesuffix("_risk_reviewer"),
                    "approve": True,
                    "suggested_cash_weight": 0.6,
                    "adjustments": [],
                    "portfolio_risk": "Risk is within configured bounds",
                }
            return response_model.model_validate(output)

        def meta(self) -> dict[str, Any]:
            return {
                "calls": len(self.calls),
                "provider_attempts": len(self.calls),
                "validated_outputs": len(self.calls),
                "warnings": [],
                "trace": [{"agent": name} for name in self.calls],
            }

    agents = FullPoolAgents()
    bundle = PortfolioMultiAgentDecisionEngine(
        replace(
            make_settings(tmp_path),
            decision_engine_mode="portfolio_multi_agent",
        ),
        agent_client=agents,
    ).decide(decision_input)

    artifacts = bundle.meta["agent_artifacts"]
    assert bundle.meta["decision_quality"] == "healthy"
    assert bundle.meta["analysis_coverage"] == 1.0
    assert bundle.meta["calls"] == 11
    assert all(
        len(report["assessments"]) == 20
        for report in artifacts["analyst_reports"].values()
    )
    assert len(artifacts["shortlist"]) == 8
    assert len(artifacts["final_allocation"]["targets"]) == 8
    assert set(bundle.decisions) == set(symbols)


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

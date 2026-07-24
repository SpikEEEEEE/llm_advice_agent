from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from app.domain.models import RawDecisionBundle
from app.domain.risk import AShareRiskPolicy
from app.repositories.sqlite import SQLiteRepository
from app.services.decision_service import DecisionService
from .helpers import FakeDecisionEngine, FakeMarketData, make_settings


def test_decision_service_uses_fixed_universe_and_held_symbols(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    portfolio = repository.create_portfolio(
        {
            "name": "test",
            "cash": "10000",
            "positions": [
                {
                    "symbol": "000001.SZ",
                    "shares": 100,
                    "available_shares": 100,
                    "average_cost": "9",
                }
            ],
        }
    )
    engine = FakeDecisionEngine()
    service = DecisionService(
        repository,
        FakeMarketData(),
        engine,
        AShareRiskPolicy(settings),
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test_v1",
        universe=["600519.SH", "300750.SZ"],
        idempotency_key=None,
        request_fingerprint="decision-service-test",
    )

    service.run(run["id"])

    completed = repository.get_decision_run(run["id"])
    assert completed["status"] == "completed"
    assert engine.last_input.symbols == ("600519.SH", "300750.SZ", "000001.SZ")
    assert len(completed["result"]["decisions"]) == 3


def test_non_finite_llm_payload_is_persisted_as_strict_json(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    portfolio = repository.create_portfolio(
        {
            "name": "test",
            "cash": "10000",
            "positions": [
                {
                    "symbol": "600519.SH",
                    "shares": 100,
                    "available_shares": 100,
                    "average_cost": "9",
                }
            ],
        }
    )

    class NonFiniteEngine:
        def decide(self, _decision_input, on_stage=None):
            return RawDecisionBundle(
                decisions={
                    "600519.SH": {
                        "action": "hold",
                        "target_cash_amount": float("inf"),
                        "confidence": float("nan"),
                        "tech_score": float("-inf"),
                    }
                },
                meta={"calls": 1, "latency_ms": float("nan")},
            )

    service = DecisionService(
        repository,
        FakeMarketData(),
        NonFiniteEngine(),  # type: ignore[arg-type]
        AShareRiskPolicy(settings),
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="holdings_only",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test_v1",
        universe=[],
        idempotency_key=None,
        request_fingerprint="non-finite-test",
    )

    service.run(run["id"])

    completed = repository.get_decision_run(run["id"])
    assert completed is not None
    decision = completed["result"]["decisions"][0]
    assert decision["raw_decision"]["target_cash_amount"] is None
    assert decision["raw_decision"]["confidence"] is None
    assert decision["raw_decision"]["tech_score"] is None
    assert completed["result"]["llm_meta"]["latency_ms"] is None
    json.dumps(completed["result"], allow_nan=False)


def test_decision_engine_failure_is_sanitized_and_marked_failed_quality(
    tmp_path,
):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    portfolio = repository.create_portfolio(
        {
            "name": "test",
            "cash": "10000",
            "positions": [
                {
                    "symbol": "600519.SH",
                    "shares": 100,
                    "available_shares": 100,
                    "average_cost": "9",
                }
            ],
        }
    )
    secret = "TOP_SECRET_PROVIDER_RESPONSE"

    class FailingEngine:
        def decide(self, _decision_input, on_stage=None):
            del on_stage
            raise RuntimeError(secret)

    service = DecisionService(
        repository,
        FakeMarketData(),
        FailingEngine(),  # type: ignore[arg-type]
        AShareRiskPolicy(settings),
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="holdings_only",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test_v1",
        universe=[],
        idempotency_key=None,
        request_fingerprint="failed-engine-test",
    )

    service.run(run["id"])

    completed = repository.get_decision_run(run["id"])
    assert completed is not None
    persisted = json.dumps(completed, ensure_ascii=False)
    assert completed["status"] == "degraded"
    assert completed["result"]["decision_quality"] == "failed"
    assert (
        completed["result"]["llm_meta"]["engine"]
        == "failed_safe_hold"
    )
    assert secret not in persisted

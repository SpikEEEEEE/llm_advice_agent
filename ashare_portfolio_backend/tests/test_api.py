from __future__ import annotations

from fastapi.testclient import TestClient

from app.container import AppContainer
from app.domain.risk import AShareRiskPolicy
from app.main import create_app
from app.repositories.sqlite import SQLiteRepository
from app.services.decision_service import DecisionService
from app.services.task_runner import DecisionTaskRunner
from app.services.task_runner import TaskQueueFullError
from .helpers import FakeDecisionEngine, FakeMarketData, make_settings


def test_portfolio_and_decision_run_api(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    market = FakeMarketData()
    engine = FakeDecisionEngine()
    risk = AShareRiskPolicy(settings)
    service = DecisionService(repository, market, engine, risk)
    runner = DecisionTaskRunner(service, mode="inline", max_workers=1)
    container = AppContainer(
        settings=settings,
        repository=repository,
        market_data=market,  # type: ignore[arg-type]
        decision_engine=engine,  # type: ignore[arg-type]
        risk_policy=risk,
        decision_service=service,
        task_runner=runner,
    )

    with TestClient(create_app(container)) as client:
        portfolio_response = client.post(
            "/api/v1/portfolios",
            json={
                "name": "My A-share portfolio",
                "cash": "10000",
                "positions": [
                    {
                        "symbol": "600519.SH",
                        "shares": 100,
                        "available_shares": 100,
                        "average_cost": "9.5",
                    }
                ],
            },
        )
        assert portfolio_response.status_code == 201
        portfolio_id = portfolio_response.json()["id"]

        run_response = client.post(
            "/api/v1/decision-runs",
            headers={"Idempotency-Key": "api-test-1"},
            json={"portfolio_id": portfolio_id, "mode": "rebalance"},
        )
        assert run_response.status_code == 202
        body = run_response.json()
        assert body["status"] == "completed"

        duplicate = client.post(
            "/api/v1/decision-runs",
            headers={"Idempotency-Key": "api-test-1"},
            json={"portfolio_id": portfolio_id, "mode": "rebalance"},
        )
        assert duplicate.status_code == 202
        assert duplicate.json()["id"] == body["id"]

        conflict = client.post(
            "/api/v1/decision-runs",
            headers={"Idempotency-Key": "api-test-1"},
            json={"portfolio_id": portfolio_id, "mode": "holdings_only"},
        )
        assert conflict.status_code == 409

        fetched = client.get(f"/api/v1/decision-runs/{body['id']}")
        assert fetched.status_code == 200
        assert len(fetched.json()["result"]["decisions"]) == 2

        historical = client.post(
            "/api/v1/decision-runs",
            json={
                "portfolio_id": portfolio_id,
                "mode": "rebalance",
                "as_of": "2020-01-01T16:00:00+08:00",
            },
        )
        assert historical.status_code == 422

        class FullRunner:
            def submit(self, _run_id):
                raise TaskQueueFullError("full")

            def shutdown(self):
                pass

        container.task_runner = FullRunner()  # type: ignore[assignment]
        queue_full = client.post(
            "/api/v1/decision-runs",
            headers={"Idempotency-Key": "queue-full-test"},
            json={"portfolio_id": portfolio_id, "mode": "rebalance"},
        )
        assert queue_full.status_code == 503
        failed_run_id = queue_full.json()["detail"]["run_id"]
        failed_run = client.get(f"/api/v1/decision-runs/{failed_run_id}")
        assert failed_run.status_code == 200
        assert failed_run.json()["status"] == "failed"
        assert failed_run.json()["error_code"] == "TASK_QUEUE_FULL"

        duplicate_failed = client.post(
            "/api/v1/decision-runs",
            headers={"Idempotency-Key": "queue-full-test"},
            json={"portfolio_id": portfolio_id, "mode": "rebalance"},
        )
        assert duplicate_failed.status_code == 202
        assert duplicate_failed.json()["id"] == failed_run_id

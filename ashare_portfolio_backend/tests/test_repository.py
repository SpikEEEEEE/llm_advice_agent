from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.repositories.sqlite import SQLiteRepository
from .helpers import make_settings


def _run(repository: SQLiteRepository):
    portfolio = repository.create_portfolio(
        {"name": "test", "cash": "1000", "positions": []}
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test",
        universe=["600519.SH"],
        idempotency_key=None,
        request_fingerprint="repository-terminal-test",
    )
    return run


def test_terminal_run_cannot_be_resurrected_by_an_orphan_worker(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run = _run(repository)
    repository.fail_decision_run(run["id"], "PROCESS_RESTARTED", "restart")

    repository.update_run_status(run["id"], "calling_llm")
    repository.complete_decision_run(run["id"], {"unsafe": True}, degraded=False)

    persisted = repository.get_decision_run(run["id"])
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["error_code"] == "PROCESS_RESTARTED"
    assert persisted["result"] is None


def test_timeout_cannot_overwrite_a_completed_run(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run = _run(repository)
    repository.complete_decision_run(run["id"], {"ok": True}, degraded=False)

    repository.fail_decision_run(run["id"], "TASK_TIMEOUT", "late watchdog")

    persisted = repository.get_decision_run(run["id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result"] == {"ok": True}

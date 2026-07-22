from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.services.task_runner import DecisionTaskRunner, TaskQueueFullError
from app.repositories.sqlite import SQLiteRepository
from .helpers import make_settings


class BlockingService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, run_id: str) -> None:
        self.started.set()
        self.release.wait(timeout=2)


def test_thread_runner_rejects_work_above_bounded_capacity():
    service = BlockingService()
    runner = DecisionTaskRunner(
        service,  # type: ignore[arg-type]
        mode="thread",
        max_workers=1,
        max_queue_size=0,
    )
    try:
        first = runner.submit("run_1")
        assert service.started.wait(timeout=1)
        with pytest.raises(TaskQueueFullError):
            runner.submit("run_2")
        service.release.set()
        assert first is not None
        first.result(timeout=2)
    finally:
        service.release.set()
        runner.shutdown()


def test_process_runner_executes_a_persisted_run(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    settings = replace(
        make_settings(tmp_path),
        project_root=backend_root,
        execution_mode="process",
    )
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    portfolio = repository.create_portfolio(
        {"name": "empty", "cash": "1000", "positions": []}
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="holdings_only",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test",
        universe=[],
        idempotency_key=None,
        request_fingerprint="process-runner-test",
    )
    runner = DecisionTaskRunner(
        BlockingService(),  # type: ignore[arg-type]
        mode="process",
        max_workers=1,
        max_queue_size=0,
        settings=settings,
        repository=repository,
        task_timeout_seconds=30,
    )
    try:
        future = runner.submit(run["id"])
        assert future is not None
        future.result(timeout=15)
    finally:
        runner.shutdown()

    completed = repository.get_decision_run(run["id"])
    assert completed is not None
    assert completed["status"] == "failed"
    assert completed["error_code"] == "VALUEERROR"

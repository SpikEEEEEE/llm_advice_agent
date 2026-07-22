from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from app.core.config import Settings
from app.repositories.sqlite import SQLiteRepository, TERMINAL_STATUSES
from app.services.decision_service import DecisionService


class TaskQueueFullError(RuntimeError):
    """Raised when a new decision cannot be admitted without unbounded growth."""


class DecisionTaskRunner:
    """Bounded runner for decision jobs.

    ``process`` (the default) launches each job in a disposable child process,
    so the parent can terminate a provider or LLM call that exceeds the task
    deadline. ``thread`` is useful for lightweight development but cannot
    forcibly stop a stuck Python call. ``inline`` is deterministic for tests.
    """

    def __init__(
        self,
        service: DecisionService,
        mode: str,
        max_workers: int,
        *,
        max_queue_size: int = 10,
        settings: Settings | None = None,
        repository: SQLiteRepository | None = None,
        task_timeout_seconds: int = 600,
    ) -> None:
        if mode not in {"process", "thread", "inline"}:
            raise ValueError("Unsupported decision execution mode")
        if mode == "process" and (settings is None or repository is None):
            raise ValueError("process mode requires settings and repository")

        self.service = service
        self.mode = mode
        self.settings = settings
        self.repository = repository
        self.task_timeout_seconds = task_timeout_seconds
        self._capacity = threading.BoundedSemaphore(max_workers + max_queue_size)
        self._process_lock = threading.Lock()
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._executor = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="decision")
            if mode != "inline"
            else None
        )

    def submit(self, run_id: str) -> Future | None:
        if self._executor is None:
            self.service.run(run_id)
            return None
        if not self._capacity.acquire(blocking=False):
            raise TaskQueueFullError("Decision task queue is full")

        try:
            future = self._executor.submit(self._execute, run_id)
        except Exception:
            self._capacity.release()
            raise
        future.add_done_callback(lambda _future: self._capacity.release())
        return future

    def _execute(self, run_id: str) -> None:
        if self.mode == "thread":
            self.service.run(run_id)
            return
        try:
            self._execute_in_child_process(run_id)
        except Exception as exc:
            assert self.repository is not None
            current = self.repository.get_decision_run(run_id)
            if current and current["status"] not in TERMINAL_STATUSES:
                self.repository.fail_decision_run(
                    run_id,
                    code="WORKER_LAUNCH_FAILED",
                    message=str(exc),
                )

    def _execute_in_child_process(self, run_id: str) -> None:
        assert self.settings is not None
        assert self.repository is not None
        command = [
            sys.executable,
            "-m",
            "app.worker",
            run_id,
            "--project-root",
            str(self.settings.project_root),
            "--database-path",
            str(self.settings.database_path),
        ]
        worker_environment = os.environ.copy()
        worker_environment.update(
            {
                "APP_ENV": self.settings.app_env,
                "BACKEND_API_KEY": self.settings.backend_api_key or "",
                "DATABASE_PATH": str(self.settings.database_path),
                "UNIVERSE_PATH": str(self.settings.universe_path),
                "CACHE_PATH": str(self.settings.cache_path),
                "DECISION_EXECUTION_MODE": self.settings.execution_mode,
                "DECISION_WORKERS": str(self.settings.decision_workers),
                "DECISION_QUEUE_CAPACITY": str(
                    self.settings.decision_queue_capacity
                ),
                "DECISION_TASK_TIMEOUT_SECONDS": str(
                    self.settings.decision_task_timeout_seconds
                ),
                "MAX_AS_OF_SKEW_MINUTES": str(
                    self.settings.max_as_of_skew_minutes
                ),
                "DATA_MODE": self.settings.data_mode,
                "TUSHARE_TOKEN": self.settings.tushare_token or "",
                "TUSHARE_TIMEOUT_SECONDS": str(
                    self.settings.tushare_timeout_seconds
                ),
                "MARKET_HISTORY_DAYS": str(self.settings.market_history_days),
                "NEWS_LOOKBACK_DAYS": str(self.settings.news_lookback_days),
                "NEWS_TOP_K": str(self.settings.news_top_k),
                "LLM_API_KEY": self.settings.llm_api_key or "",
                "LLM_BASE_URL": self.settings.llm_base_url,
                "LLM_MODEL": self.settings.llm_model,
                "LLM_TEMPERATURE": str(self.settings.llm_temperature),
                "LLM_MAX_TOKENS": str(self.settings.llm_max_tokens),
                "LLM_TIMEOUT_SECONDS": str(self.settings.llm_timeout_seconds),
                "LLM_MAX_RETRIES": str(self.settings.llm_max_retries),
                "MIN_CASH_RATIO": str(self.settings.min_cash_ratio),
                "MAX_POSITION_RATIO": str(self.settings.max_position_ratio),
                "MAX_POSITIONS": str(self.settings.max_positions),
                "BUY_FEE_BUFFER_RATIO": str(
                    self.settings.buy_fee_buffer_ratio
                ),
                "MARKET_CLOSE_HOUR": str(self.settings.market_close_hour),
                "MARKET_CLOSE_MINUTE": str(
                    self.settings.market_close_minute
                ),
            }
        )
        process = subprocess.Popen(
            command,
            cwd=self.settings.project_root,
            env=worker_environment,
        )
        with self._process_lock:
            self._active_processes[run_id] = process
        try:
            try:
                return_code = process.wait(timeout=self.task_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self.repository.fail_decision_run(
                    run_id,
                    code="TASK_TIMEOUT",
                    message=(
                        "Decision task exceeded "
                        f"{self.task_timeout_seconds} seconds and was terminated"
                    ),
                )
                return

            if return_code != 0:
                current = self.repository.get_decision_run(run_id)
                if current and current["status"] not in TERMINAL_STATUSES:
                    self.repository.fail_decision_run(
                        run_id,
                        code="WORKER_PROCESS_FAILED",
                        message=f"Decision worker exited with status {return_code}",
                    )
        finally:
            with self._process_lock:
                self._active_processes.pop(run_id, None)

    def shutdown(self) -> None:
        with self._process_lock:
            processes = list(self._active_processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

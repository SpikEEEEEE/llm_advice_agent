from __future__ import annotations

from dataclasses import dataclass

from app.adapters.decision_engine_factory import build_decision_engine
from app.adapters.tushare import TushareMarketDataProvider
from app.core.config import Settings
from app.domain.risk import AShareRiskPolicy
from app.ports.decision_engine import DecisionEngine
from app.repositories.sqlite import SQLiteRepository
from app.services.decision_service import DecisionService
from app.services.task_runner import DecisionTaskRunner


@dataclass
class AppContainer:
    settings: Settings
    repository: SQLiteRepository
    market_data: TushareMarketDataProvider
    decision_engine: DecisionEngine
    risk_policy: AShareRiskPolicy
    decision_service: DecisionService
    task_runner: DecisionTaskRunner

    @classmethod
    def build(cls, settings: Settings | None = None) -> "AppContainer":
        effective_settings = settings or Settings.from_env()
        repository = SQLiteRepository(effective_settings.database_path)
        market_data = TushareMarketDataProvider(effective_settings)
        decision_engine = build_decision_engine(effective_settings)
        risk_policy = AShareRiskPolicy(effective_settings)
        decision_service = DecisionService(
            repository,
            market_data,
            decision_engine,
            risk_policy,
        )
        task_runner = DecisionTaskRunner(
            decision_service,
            mode=effective_settings.execution_mode,
            max_workers=effective_settings.decision_workers,
            max_queue_size=effective_settings.decision_queue_capacity,
            settings=effective_settings,
            repository=repository,
            task_timeout_seconds=effective_settings.decision_task_timeout_seconds,
        )
        return cls(
            settings=effective_settings,
            repository=repository,
            market_data=market_data,
            decision_engine=decision_engine,
            risk_policy=risk_policy,
            decision_service=decision_service,
            task_runner=task_runner,
        )

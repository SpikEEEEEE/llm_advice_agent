from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.core.config import Settings
from app.domain.models import RawDecisionBundle, SymbolMarketSnapshot


def make_settings(tmp_path: Path, *, execution_mode: str = "inline") -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    universe_path = config_dir / "universe.yaml"
    universe_path.write_text(
        "version: test_v1\nsymbols:\n  - 600519.SH\n  - 300750.SZ\n",
        encoding="utf-8",
    )
    return Settings(
        project_root=tmp_path,
        app_env="test",
        backend_api_key=None,
        database_path=tmp_path / "backend.db",
        universe_path=universe_path,
        cache_path=tmp_path / "cache",
        execution_mode=execution_mode,
        decision_workers=1,
        decision_queue_capacity=10,
        decision_task_timeout_seconds=60,
        max_as_of_skew_minutes=10,
        data_mode="offline_only",
        tushare_token=None,
        tushare_timeout_seconds=5,
        market_history_days=400,
        news_lookback_days=3,
        news_top_k=5,
        llm_api_key=None,
        llm_base_url="https://llm.invalid/v1",
        llm_model="mock-model",
        llm_temperature=0.0,
        llm_max_tokens=1000,
        llm_timeout_seconds=10,
        llm_max_retries=0,
        min_cash_ratio=Decimal("0"),
        max_position_ratio=Decimal("1"),
        max_positions=10,
        buy_fee_buffer_ratio=Decimal("0"),
        market_close_hour=15,
        market_close_minute=15,
    )


class FakeMarketData:
    name = "fake"

    def latest_completed_session(self, as_of: datetime) -> date:
        return date(2026, 7, 21)

    def next_session(self, after: date) -> date:
        return date(2026, 7, 22)

    def load_symbol(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        dates = [data_date - timedelta(days=offset) for offset in range(9, -1, -1)]
        bars = pd.DataFrame(
            {
                "date": dates,
                "open": [10.0] * 10,
                "high": [11.0] * 10,
                "low": [9.0] * 10,
                "close": [10.0] * 10,
                "volume": [1000.0] * 10,
                "vwap": [10.0] * 10,
            }
        )
        return SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=Decimal("10"),
            bars=bars,
            news=(),
            retrieved_at=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
        )


class FakeDecisionEngine:
    def __init__(self, decisions: dict[str, dict[str, Any]] | None = None) -> None:
        self.decisions = decisions
        self.last_input = None

    def decide(self, decision_input, on_stage=None) -> RawDecisionBundle:
        self.last_input = decision_input
        if on_stage:
            on_stage("building_features")
            on_stage("calling_llm")
        decisions = self.decisions
        if decisions is None:
            decisions = {
                symbol: {
                    "action": "hold",
                    "target_cash_amount": float(
                        decision_input.market[symbol].reference_price
                        * (
                            decision_input.portfolio.position_map().get(symbol).shares
                            if decision_input.portfolio.position_map().get(symbol)
                            else 0
                        )
                    ),
                    "confidence": 0.8,
                    "reasons": ["mock decision"],
                }
                for symbol in decision_input.market
            }
        return RawDecisionBundle(decisions=decisions, meta={"calls": 1})

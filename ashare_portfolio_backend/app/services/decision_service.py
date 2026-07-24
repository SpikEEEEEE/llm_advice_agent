from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    RawDecisionBundle,
)
from app.domain.risk import AShareRiskPolicy
from app.core.json_safety import sanitize_json_value
from app.ports.decision_engine import DecisionEngine
from app.ports.market_data import MarketDataProvider
from app.repositories.sqlite import SQLiteRepository


logger = logging.getLogger(__name__)
SAFE_EXCEPTION_TYPES = {
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "TimeoutError",
    "ConnectionError",
    "DataUnavailableError",
    "StaleDataError",
    "ProviderConfigurationError",
    "PortfolioAgentGraphError",
    "PortfolioAgentOutputError",
    "DecisionOutputError",
}


def _safe_exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    return name if name in SAFE_EXCEPTION_TYPES else "Exception"


class DecisionService:
    def __init__(
        self,
        repository: SQLiteRepository,
        market_data: MarketDataProvider,
        decision_engine: DecisionEngine,
        risk_policy: AShareRiskPolicy,
    ) -> None:
        self.repository = repository
        self.market_data = market_data
        self.decision_engine = decision_engine
        self.risk_policy = risk_policy

    @staticmethod
    def _portfolio(payload: dict[str, Any]) -> PortfolioSnapshot:
        positions = tuple(
            Position(
                symbol=str(item["symbol"]),
                shares=int(item["shares"]),
                available_shares=int(item.get("available_shares", item["shares"])),
                average_cost=Decimal(str(item["average_cost"])),
                holding_days=(
                    int(item["holding_days"])
                    if item.get("holding_days") is not None
                    else None
                ),
            )
            for item in payload.get("positions", [])
        )
        return PortfolioSnapshot(
            portfolio_id=payload["id"],
            version=int(payload["version"]),
            name=str(payload["name"]),
            cash=Decimal(str(payload["cash"])),
            positions=positions,
        )

    @staticmethod
    def _as_of(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        timezone = ZoneInfo("Asia/Shanghai")
        return (
            parsed.replace(tzinfo=timezone)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone)
        )

    def run(self, run_id: str) -> None:
        run = self.repository.get_decision_run(run_id)
        if not run:
            return
        try:
            portfolio = self._portfolio(run["input"])
            as_of = self._as_of(run["as_of"])
            held_symbols = [position.symbol for position in portfolio.positions]
            if run["mode"] == "holdings_only":
                symbols = list(dict.fromkeys(held_symbols))
                if not symbols:
                    raise ValueError("holdings_only mode requires at least one position")
            else:
                symbols = list(dict.fromkeys([*run["universe"], *held_symbols]))

            self.repository.update_run_status(run_id, "fetching_data")
            data_date = self.market_data.latest_completed_session(as_of)
            valid_for_session = self.market_data.next_session(data_date)
            market = {}
            unavailable: dict[str, str] = {}
            data_quality_warnings: list[str] = []
            for symbol in symbols:
                try:
                    snapshot = self.market_data.load_symbol(
                        symbol,
                        data_date,
                        as_of,
                    )
                    market[symbol] = snapshot
                    data_quality_warnings.extend(
                        f"{symbol}: {warning}"
                        for warning in snapshot.data_quality_warnings
                    )
                except Exception as exc:
                    failure_type = _safe_exception_type(exc)
                    unavailable[symbol] = (
                        f"{failure_type}: market data unavailable"
                    )
                    logger.warning(
                        "Market data unavailable for %s (%s)",
                        symbol,
                        failure_type,
                    )

            decision_input = DecisionInput(
                run_id=run_id,
                portfolio=portfolio,
                mode=run["mode"],
                as_of=as_of,
                data_date=data_date,
                valid_for_session=valid_for_session,
                universe_version=run["universe_version"],
                symbols=tuple(symbols),
                market=market,
                unavailable_symbols=unavailable,
            )

            try:
                bundle = self.decision_engine.decide(
                    decision_input,
                    on_stage=lambda stage: self.repository.update_run_status(
                        run_id, stage
                    ),
                )
            except Exception as exc:
                failure_type = _safe_exception_type(exc)
                logger.error(
                    "LLM decision failed for %s (%s)",
                    run_id,
                    failure_type,
                )
                bundle = RawDecisionBundle(
                    decisions={},
                    meta={
                        "calls": 0,
                        "provider_attempts": 0,
                        "engine": "failed_safe_hold",
                        "decision_quality": "failed",
                        "analysis_coverage": 0.0,
                        "stage_health": {
                            "decision_engine": {
                                "status": "failed",
                                "failure_category": failure_type,
                            }
                        },
                    },
                    warnings=(
                        "LLM decision failed; safe-hold fallback applied "
                        f"({failure_type})",
                    ),
                )

            safe_decisions = sanitize_json_value(bundle.decisions)
            safe_meta = sanitize_json_value(bundle.meta)
            bundle = RawDecisionBundle(
                decisions=safe_decisions if isinstance(safe_decisions, dict) else {},
                meta=safe_meta if isinstance(safe_meta, dict) else {},
                warnings=bundle.warnings,
            )

            if data_quality_warnings:
                bundle = RawDecisionBundle(
                    decisions=bundle.decisions,
                    meta=bundle.meta,
                    warnings=tuple([*bundle.warnings, *data_quality_warnings]),
                )

            self.repository.update_run_status(run_id, "validating")
            result = self.risk_policy.apply(decision_input, bundle)
            market_snapshot: dict[str, dict[str, Any]] = {}
            for symbol, snapshot in market.items():
                try:
                    close_series = snapshot.bars.get("close")
                    recent_closes = (
                        [
                            float(value)
                            for value in close_series.tail(7).tolist()
                        ]
                        if close_series is not None
                        else []
                    )
                except Exception:
                    recent_closes = []
                market_snapshot[symbol] = {
                    "data_date": snapshot.data_date.isoformat(),
                    "reference_price": float(snapshot.reference_price),
                    "retrieved_at": snapshot.retrieved_at.isoformat()
                    if snapshot.retrieved_at
                    else None,
                    "recent_closes": recent_closes,
                    "news": [
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "published_utc": item.get("published_utc"),
                            "api_source": item.get("api_source"),
                            "description": item.get("description"),
                        }
                        for item in snapshot.news
                    ],
                    "fundamentals": snapshot.fundamentals,
                    "data_quality_warnings": list(snapshot.data_quality_warnings),
                }
            result["market_snapshot"] = market_snapshot
            degraded = bool(
                unavailable
                or result.get("warnings")
                or result.get("decision_quality") != "healthy"
            )
            if market and int(bundle.meta.get("calls", 0) or 0) == 0:
                degraded = True
                result.setdefault("warnings", []).append(
                    "No successful decision-agent call was recorded"
                )
            self.repository.complete_decision_run(run_id, result, degraded=degraded)
        except Exception as exc:
            failure_type = _safe_exception_type(exc)
            logger.error(
                "Decision run %s failed (%s)",
                run_id,
                failure_type,
            )
            self.repository.fail_decision_run(
                run_id,
                code=failure_type.upper(),
                message=f"Decision run failed safely ({failure_type})",
            )

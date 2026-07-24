from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.json_safety import sanitize_json_value
from app.domain.features import build_symbol_context
from app.domain.models import DecisionInput, RawDecisionBundle


CACHE_ENVELOPE_VERSION = 2
CACHE_KEY_SCHEMA_VERSION = 3

# These explicit tokens are part of the content-addressed key. Bump the
# corresponding token whenever the graph, prompts, output contracts, or
# decision-quality rules change in a way that can alter a historical decision.
PORTFOLIO_GRAPH_VERSION = "pool_graph_quality_gate_v2"
AGENT_PROMPT_VERSION = "compact_full_pool_prompts_v2"
AGENT_OUTPUT_SCHEMA_VERSION = "pool_agent_contracts_v2"
DECISION_QUALITY_POLICY_VERSION = "analyst_and_risk_quorum_v2"


def decision_quality(bundle: RawDecisionBundle) -> str:
    """Return a stable quality label, including legacy engine output."""

    if not bundle.decisions:
        return "failed"
    raw = bundle.meta.get("decision_quality")
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"healthy", "degraded", "failed"}:
            return normalized
        return "failed"
    if raw is not None:
        return "failed"
    engine = str(bundle.meta.get("engine") or "").strip().lower()
    if engine in {
        "failed_safe_hold",
        "failed-safe-hold",
        "failed_safe",
        "failed-safe",
    }:
        return "failed"
    # Scripted and single-LLM engines created before quality metadata existed
    # remain cache-compatible unless they explicitly identify as failed-safe.
    return "healthy"


def _non_negative_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _validated_outputs(meta: dict[str, Any]) -> int:
    explicit = meta.get("validated_outputs")
    if explicit is None:
        explicit = meta.get("provider_successes")
    if explicit is not None:
        return _non_negative_int(explicit)
    trace = meta.get("agent_trace")
    if not isinstance(trace, list):
        trace = meta.get("trace")
    return len(trace) if isinstance(trace, list) else 0


class BacktestDecisionCache:
    """Content-addressed cache for costly historical LLM decisions."""

    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root
        self.settings = settings

    def key(self, decision_input: DecisionInput) -> str:
        positions = [
            {
                "symbol": item.symbol,
                "shares": item.shares,
                "available_shares": item.available_shares,
                "average_cost": str(item.average_cost),
                "holding_days": item.holding_days,
            }
            for item in decision_input.portfolio.positions
        ]
        contexts: list[dict[str, Any]] = []
        context_failures: list[str] = []
        for symbol in decision_input.symbols:
            snapshot = decision_input.market.get(symbol)
            if snapshot is None:
                continue
            try:
                contexts.append(
                    build_symbol_context(
                        decision_input,
                        snapshot,
                    )
                )
            except Exception:
                # Failed/degraded bundles are never written, but key creation
                # must still allow the graph to perform its safe degradation.
                context_failures.append(symbol)
        payload = {
            "cache_key_schema_version": CACHE_KEY_SCHEMA_VERSION,
            "implementation_versions": {
                "graph": PORTFOLIO_GRAPH_VERSION,
                "prompts": AGENT_PROMPT_VERSION,
                "output_schema": AGENT_OUTPUT_SCHEMA_VERSION,
                "quality_policy": DECISION_QUALITY_POLICY_VERSION,
            },
            "engine": self.settings.decision_engine_mode,
            "provider": {
                "base_url": self.settings.llm_base_url,
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "max_retries": self.settings.llm_max_retries,
                "structured_output_mode": getattr(
                    self.settings,
                    "llm_structured_output_mode",
                    "auto",
                ),
            },
            "constraints": {
                "min_cash_ratio": str(self.settings.min_cash_ratio),
                "max_position_ratio": str(
                    self.settings.max_position_ratio
                ),
                "max_positions": self.settings.max_positions,
                "shortlist_size": self.settings.multi_agent_shortlist_size,
                "max_calls": self.settings.multi_agent_max_calls,
                "semantic_retries": (
                    self.settings.multi_agent_semantic_retries
                ),
                "output_retries": getattr(
                    self.settings,
                    "multi_agent_output_retries",
                    1,
                ),
                "minimum_analysts": getattr(
                    self.settings,
                    "multi_agent_min_analysts",
                    2,
                ),
                "minimum_risk_reviews": getattr(
                    self.settings,
                    "multi_agent_min_risk_reviews",
                    2,
                ),
                "analyst_weights": [
                    self.settings.multi_agent_technical_weight,
                    self.settings.multi_agent_fundamental_weight,
                    self.settings.multi_agent_news_weight,
                ],
            },
            "as_of": decision_input.as_of.isoformat(),
            "mode": decision_input.mode,
            "data_date": decision_input.data_date.isoformat(),
            "valid_for_session": (
                decision_input.valid_for_session.isoformat()
            ),
            "universe_version": decision_input.universe_version,
            "symbols": list(decision_input.symbols),
            "portfolio": {
                "cash": str(decision_input.portfolio.cash),
                "positions": positions,
            },
            "market_contexts": contexts,
            "market_context_failures": context_failures,
            "unavailable_symbols": decision_input.unavailable_symbols,
        }
        safe = sanitize_json_value(payload)
        encoded = json.dumps(
            safe,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def read(self, key: str) -> RawDecisionBundle | None:
        path = self.root / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_envelope_version") != CACHE_ENVELOPE_VERSION:
            return None
        cached_quality = payload.get("decision_quality")
        if cached_quality != "healthy":
            return None
        raw_bundle = payload.get("bundle")
        if not isinstance(raw_bundle, dict):
            return None
        decisions = raw_bundle.get("decisions")
        meta = raw_bundle.get("meta")
        warnings = raw_bundle.get("warnings")
        if not isinstance(decisions, dict) or not isinstance(meta, dict):
            return None
        original_calls = _non_negative_int(meta.get("calls"))
        original_attempts = _non_negative_int(
            meta.get("provider_attempts"),
            original_calls,
        )
        original_validated_outputs = _validated_outputs(meta)
        original_repair_attempts = _non_negative_int(
            meta.get("output_repair_attempts")
        )
        cached_meta = {
            **meta,
            "calls": 0,
            "provider_attempts": 0,
            "validated_outputs": 0,
            "output_repair_attempts": 0,
            "backtest_cache_hit": True,
            "cached_original_calls": original_calls,
            "cached_original_provider_attempts": original_attempts,
            "cached_original_validated_outputs": (
                original_validated_outputs
            ),
            "cached_original_output_repair_attempts": (
                original_repair_attempts
            ),
        }
        bundle = RawDecisionBundle(
            decisions=decisions,
            meta=cached_meta,
            warnings=tuple(
                str(item)
                for item in (
                    warnings if isinstance(warnings, list) else []
                )
            ),
        )
        return bundle if decision_quality(bundle) == "healthy" else None

    def write(self, key: str, bundle: RawDecisionBundle) -> bool:
        quality = decision_quality(bundle)
        if quality != "healthy":
            return False
        safe = sanitize_json_value(
            {
                "cache_envelope_version": CACHE_ENVELOPE_VERSION,
                "decision_quality": quality,
                "bundle": {
                    "decisions": bundle.decisions,
                    "meta": bundle.meta,
                    "warnings": list(bundle.warnings),
                },
            }
        )
        encoded = json.dumps(
            safe,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{key}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        destination = self.root / f"{key}.json"
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def get_or_none(
        self,
        decision_input: DecisionInput,
    ) -> tuple[str, RawDecisionBundle | None]:
        key = self.key(decision_input)
        return key, self.read(key)

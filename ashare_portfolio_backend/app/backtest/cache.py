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
        contexts = [
            build_symbol_context(
                decision_input,
                decision_input.market[symbol],
            )
            for symbol in decision_input.symbols
            if symbol in decision_input.market
        ]
        payload = {
            "schema_version": 1,
            "engine": self.settings.decision_engine_mode,
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "constraints": {
                "min_cash_ratio": str(self.settings.min_cash_ratio),
                "max_position_ratio": str(
                    self.settings.max_position_ratio
                ),
                "max_positions": self.settings.max_positions,
                "shortlist_size": self.settings.multi_agent_shortlist_size,
                "analyst_weights": [
                    self.settings.multi_agent_technical_weight,
                    self.settings.multi_agent_fundamental_weight,
                    self.settings.multi_agent_news_weight,
                ],
            },
            "as_of": decision_input.as_of.isoformat(),
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
        decisions = payload.get("decisions")
        meta = payload.get("meta")
        warnings = payload.get("warnings")
        if not isinstance(decisions, dict) or not isinstance(meta, dict):
            return None
        original_calls = meta.get("calls", 0)
        cached_meta = {
            **meta,
            "calls": 0,
            "backtest_cache_hit": True,
            "cached_original_calls": (
                original_calls if isinstance(original_calls, int) else 0
            ),
        }
        return RawDecisionBundle(
            decisions=decisions,
            meta=cached_meta,
            warnings=tuple(
                str(item)
                for item in (
                    warnings if isinstance(warnings, list) else []
                )
            ),
        )

    def write(self, key: str, bundle: RawDecisionBundle) -> None:
        safe = sanitize_json_value(
            {
                "decisions": bundle.decisions,
                "meta": bundle.meta,
                "warnings": list(bundle.warnings),
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

    def get_or_none(
        self,
        decision_input: DecisionInput,
    ) -> tuple[str, RawDecisionBundle | None]:
        key = self.key(decision_input)
        return key, self.read(key)

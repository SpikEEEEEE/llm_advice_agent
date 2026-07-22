from __future__ import annotations

import json
import math
import time
from typing import Any

from app.core.config import Settings
from app.domain.features import build_decision_context
from app.domain.models import DecisionInput, RawDecisionBundle
from app.ports.decision_engine import StageCallback


SYSTEM_PROMPT = """You are an A-share portfolio research assistant.
Use only the point-in-time market context supplied by the user. Produce one
advisory target position value in CNY for every supplied symbol. Never claim to
place orders. Treat missing or low-quality inputs conservatively. The output
must match the requested JSON schema exactly.

Actions:
- increase: target value is above the current position value
- hold: maintain the current position
- decrease: target value is below the current value but above zero
- close: target value is zero
"""


class DecisionOutputError(ValueError):
    """The model returned syntactically or semantically invalid output."""


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    class_name = type(exc).__name__
    if class_name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
    }:
        return True
    cause = getattr(exc, "__cause__", None)
    return (
        isinstance(exc, (ConnectionError, TimeoutError))
        or isinstance(cause, (ConnectionError, TimeoutError))
        or "timeout" in str(exc).lower()
    )


def _schema_unsupported(exc: Exception) -> bool:
    if _status_code(exc) != 400:
        return False
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message


def _response_value(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


class OpenAIDecisionEngine:
    """Direct OpenAI-compatible Chat Completions decision adapter."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._injected_client = client
        self._client_instance: Any | None = None

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._client_instance is None:
            if not self.settings.llm_api_key:
                raise RuntimeError("LLM_API_KEY or OPENAI_API_KEY is not configured")
            from openai import OpenAI

            self._client_instance = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
        return self._client_instance

    @staticmethod
    def _schema(symbol_count: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": symbol_count,
                    "maxItems": symbol_count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "symbol": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["increase", "hold", "decrease", "close"],
                            },
                            "target_position_value": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "reasons": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 5,
                                "items": {"type": "string", "maxLength": 500},
                            },
                        },
                        "required": [
                            "symbol",
                            "action",
                            "target_position_value",
                            "confidence",
                            "reasons",
                        ],
                    },
                }
            },
            "required": ["decisions"],
        }

    def _request(
        self,
        context: dict[str, Any],
        response_format: dict[str, Any],
    ) -> Any:
        system_prompt = SYSTEM_PROMPT
        if response_format.get("type") == "json_object":
            system_prompt += (
                "\nExact fallback JSON contract (no additional fields):\n"
                + json.dumps(
                    self._schema(len(context["symbols"])),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return self._client().chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False, allow_nan=False),
                },
            ],
            response_format=response_format,
        )

    def _invoke(self, context: dict[str, Any]) -> tuple[Any, int, str, list[str]]:
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "portfolio_decisions",
                "strict": True,
                "schema": self._schema(len(context["symbols"])),
            },
        }
        response_format = schema_format
        format_name = "json_schema"
        warnings: list[str] = []
        calls = 0
        retry_index = 0

        while True:
            try:
                calls += 1
                return (
                    self._request(context, response_format),
                    calls,
                    format_name,
                    warnings,
                )
            except Exception as exc:
                if format_name == "json_schema" and _schema_unsupported(exc):
                    response_format = {"type": "json_object"}
                    format_name = "json_object"
                    warnings.append("LLM_JSON_SCHEMA_UNSUPPORTED_USED_JSON_OBJECT")
                    retry_index = 0
                    continue
                if _retryable(exc) and retry_index < self.settings.llm_max_retries:
                    time.sleep(min(2.0, 0.25 * (2**retry_index)))
                    retry_index += 1
                    continue
                raise

    @staticmethod
    def _content(response: Any) -> str:
        choices = _response_value(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise DecisionOutputError("LLM response did not contain choices")
        choice = choices[0]
        message = _response_value(choice, "message")
        content = _response_value(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise DecisionOutputError("LLM response content was empty")
        return content

    @staticmethod
    def _parse(
        content: str,
        expected_symbols: set[str],
        current_values: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        def reject_constant(value: str) -> None:
            raise DecisionOutputError(f"Non-finite JSON constant is forbidden: {value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, value in pairs:
                if key in output:
                    raise DecisionOutputError(f"Duplicate JSON key is forbidden: {key}")
                output[key] = value
            return output

        try:
            payload = json.loads(
                content,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (json.JSONDecodeError, DecisionOutputError) as exc:
            raise DecisionOutputError("LLM response was not strict JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"decisions"}:
            raise DecisionOutputError("LLM response root must contain only decisions")
        items = payload["decisions"]
        if not isinstance(items, list):
            raise DecisionOutputError("decisions must be an array")

        required = {
            "symbol",
            "action",
            "target_position_value",
            "confidence",
            "reasons",
        }
        parsed: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != required:
                raise DecisionOutputError("Each decision must match the exact schema")
            if not isinstance(item["symbol"], str) or not isinstance(
                item["action"], str
            ):
                raise DecisionOutputError("symbol and action must be strings")
            symbol = item["symbol"].strip().upper()
            if symbol in parsed:
                raise DecisionOutputError(f"Duplicate decision for {symbol}")
            action = item["action"].strip().lower()
            if action not in {"increase", "hold", "decrease", "close"}:
                raise DecisionOutputError(f"Invalid action for {symbol}")
            raw_target = item["target_position_value"]
            raw_confidence = item["confidence"]
            if (
                isinstance(raw_target, bool)
                or not isinstance(raw_target, (int, float))
                or isinstance(raw_confidence, bool)
                or not isinstance(raw_confidence, (int, float))
            ):
                raise DecisionOutputError(f"Numeric fields must be JSON numbers for {symbol}")
            target = float(raw_target)
            confidence = float(raw_confidence)
            if not math.isfinite(target) or target < 0:
                raise DecisionOutputError(f"Invalid target value for {symbol}")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise DecisionOutputError(f"Invalid confidence for {symbol}")
            reasons = item["reasons"]
            if (
                not isinstance(reasons, list)
                or not 1 <= len(reasons) <= 5
                or any(
                    not isinstance(reason, str)
                    or not reason.strip()
                    or len(reason) > 500
                    for reason in reasons
                )
            ):
                raise DecisionOutputError(f"Invalid reasons for {symbol}")
            current = float(current_values.get(symbol, 0.0) or 0.0)
            tolerance = max(0.01, current * 0.005)
            if action == "increase" and target <= current:
                raise DecisionOutputError(f"Increase target is not above current value for {symbol}")
            if action == "decrease" and not 0 < target < current:
                raise DecisionOutputError(f"Decrease target is not below current value for {symbol}")
            if action == "close" and target != 0:
                raise DecisionOutputError(f"Close target is not zero for {symbol}")
            if action == "hold" and abs(target - current) > tolerance:
                raise DecisionOutputError(f"Hold target differs from current value for {symbol}")
            parsed[symbol] = {
                "action": action,
                "target_cash_amount": target,
                "confidence": confidence,
                "reasons": [reason.strip() for reason in reasons],
            }

        if set(parsed) != expected_symbols:
            missing = sorted(expected_symbols - set(parsed))
            extra = sorted(set(parsed) - expected_symbols)
            raise DecisionOutputError(
                f"Decision symbols did not match input; missing={missing}, extra={extra}"
            )
        return parsed

    def decide(
        self,
        decision_input: DecisionInput,
        on_stage: StageCallback | None = None,
    ) -> RawDecisionBundle:
        if on_stage:
            on_stage("building_features")
        context = build_decision_context(decision_input)
        expected_symbols = {
            item["symbol"] for item in context["symbols"] if item.get("symbol")
        }
        if not expected_symbols:
            return RawDecisionBundle(
                decisions={},
                meta={"calls": 0, "model": self.settings.llm_model},
                warnings=("No valid market context was available for the LLM",),
            )
        if on_stage:
            on_stage("calling_llm")
        response, calls, format_name, warnings = self._invoke(context)
        current_values = {
            str(item["symbol"]): float(
                item.get("position", {}).get("current_position_value") or 0
            )
            for item in context["symbols"]
        }
        decisions = self._parse(
            self._content(response),
            expected_symbols,
            current_values,
        )
        usage = _response_value(response, "usage")
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            usage = None
        return RawDecisionBundle(
            decisions=decisions,
            meta={
                "calls": calls,
                "model": self.settings.llm_model,
                "response_format": format_name,
                "response_id": _response_value(response, "id"),
                "usage": usage,
            },
            warnings=tuple(warnings),
        )

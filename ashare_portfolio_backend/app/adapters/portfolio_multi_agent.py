from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from typing import Any, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from app.agents.portfolio_graph import (
    PortfolioAgentGraph,
    PortfolioAgentState,
    StructuredAgentClient,
)
from app.core.config import Settings
from app.domain.models import DecisionInput, RawDecisionBundle
from app.ports.decision_engine import StageCallback


ModelT = TypeVar("ModelT", bound=BaseModel)
MAX_REPAIR_OUTPUT_CHARS = 16_000
SAFE_FINISH_REASONS = {
    "stop",
    "length",
    "content_filter",
    "tool_calls",
    "function_call",
}
SAFE_OUTPUT_ERROR_CATEGORIES = {
    "invalid_output",
    "response_content",
    "strict_json",
    "json_decode",
    "schema_validation",
    "truncated_output",
}
SAFE_EXCEPTION_TYPES = {
    "PortfolioAgentGraphError",
    "PortfolioAgentOutputError",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "TimeoutError",
    "ConnectionError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "RateLimitError",
    "BadRequestError",
}


def _safe_exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    return name if name in SAFE_EXCEPTION_TYPES else "Exception"


def _safe_output_category(value: Any) -> str:
    return (
        value
        if value in SAFE_OUTPUT_ERROR_CATEGORIES
        else "invalid_output"
    )


def _safe_finish_reason_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"max_tokens", "max_output_tokens"}:
        normalized = "length"
    return (
        normalized
        if normalized in SAFE_FINISH_REASONS
        else "other"
    )


class PortfolioAgentOutputError(ValueError):
    """Safe, structured diagnostics for an invalid agent response."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "invalid_output",
        response_model: str | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
        finish_reason: str | None = None,
        truncated: bool = False,
        output_attempts: int = 1,
    ) -> None:
        self.message = message
        self.category = category
        self.response_model = response_model
        self.validation_errors = tuple(validation_errors or ())
        self.finish_reason = finish_reason
        self.truncated = truncated
        self.output_attempts = output_attempts
        super().__init__(self._render())

    def _render(self) -> str:
        details = [f"category={_safe_output_category(self.category)}"]
        safe_finish_reason = _safe_finish_reason_value(
            self.finish_reason
        )
        if safe_finish_reason:
            details.append(f"finish_reason={safe_finish_reason}")
        if self.truncated:
            details.append("truncated=true")
        if self.validation_errors:
            details.append(
                f"validation_error_count={len(self.validation_errors)}"
            )
        return f"Agent response was invalid ({', '.join(details)})"

    def set_output_attempts(self, attempts: int) -> None:
        self.output_attempts = attempts
        self.args = (self._render(),)

    def diagnostics(self) -> dict[str, Any]:
        validation_errors: list[dict[str, Any]] = []
        for item in self.validation_errors[:20]:
            raw_location = item.get("loc")
            safe_location: list[str | int] = []
            if isinstance(raw_location, (list, tuple)):
                for part in raw_location:
                    if isinstance(part, int) and not isinstance(part, bool):
                        safe_location.append(part)
                    elif part == "$":
                        safe_location.append("$")
                    else:
                        safe_location.append("?")
            validation_errors.append({"loc": safe_location})
        return {
            "category": _safe_output_category(self.category),
            "validation_errors": validation_errors,
            "finish_reason": _safe_finish_reason_value(
                self.finish_reason
            ),
            "truncated": self.truncated is True,
            "output_attempts": (
                self.output_attempts
                if isinstance(self.output_attempts, int)
                and self.output_attempts >= 0
                else 0
            ),
        }


class _StructuredFormatCapability:
    """Provider capability memo shared by sessions using the same transport."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.json_schema_unsupported = False


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


class OpenAIStructuredAgentClient(StructuredAgentClient):
    """Shared OpenAI-compatible structured client for all pool-agent roles."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        _format_capability: _StructuredFormatCapability | None = None,
    ) -> None:
        self.settings = settings
        self._injected_client = client
        self._client_instance: Any | None = None
        self._format_capability = (
            _format_capability or _StructuredFormatCapability()
        )
        self._lock = threading.RLock()
        self._calls = 0
        self._validated_outputs = 0
        self._output_repair_attempts = 0
        self._warnings: list[str] = []
        self._trace: list[dict[str, Any]] = []

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._lock:
            if self._client_instance is None:
                if not self.settings.llm_api_key:
                    raise RuntimeError(
                        "LLM_API_KEY or OPENAI_API_KEY is not configured"
                    )
                from openai import OpenAI

                self._client_instance = OpenAI(
                    api_key=self.settings.llm_api_key,
                    base_url=self.settings.llm_base_url,
                    timeout=self.settings.llm_timeout_seconds,
                    max_retries=0,
                )
            return self._client_instance

    def _reserve_call(self) -> None:
        with self._lock:
            if self._calls >= self.settings.multi_agent_max_calls:
                raise RuntimeError(
                    "Multi-agent LLM call budget exhausted "
                    f"({self.settings.multi_agent_max_calls})"
                )
            self._calls += 1

    @staticmethod
    def _schema_name(agent_name: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_name)
        return f"{normalized[:48]}_output"

    def _auto_prefers_json_object(self) -> bool:
        base_url = self.settings.llm_base_url.strip()
        parsed = urlsplit(
            base_url if "://" in base_url else f"https://{base_url}"
        )
        return (parsed.hostname or "").lower() == "api.deepseek.com"

    def _selected_format_name(self) -> str:
        configured = self.settings.llm_structured_output_mode
        if configured == "json_object":
            return "json_object"
        if configured == "auto" and self._auto_prefers_json_object():
            return "json_object"
        with self._format_capability.lock:
            if self._format_capability.json_schema_unsupported:
                return "json_object"
        return "json_schema"

    def _response_format(
        self,
        *,
        agent_name: str,
        schema: dict[str, Any],
        format_name: str,
    ) -> dict[str, Any]:
        if format_name == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(agent_name),
                "strict": True,
                "schema": schema,
            },
        }

    def _append_trace(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._trace.append(entry)

    def _request(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_format: dict[str, Any],
        schema: dict[str, Any],
    ) -> Any:
        effective_prompt = system_prompt
        if response_format.get("type") == "json_object":
            effective_prompt += (
                "\nJSON contract; return exactly this schema with no extra fields:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        return self._client().chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": effective_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            response_format=response_format,
        )

    def _invoke_provider(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        output_attempt: int,
        repair: bool,
    ) -> tuple[Any, int, str]:
        format_name = self._selected_format_name()
        response_format = self._response_format(
            agent_name=agent_name,
            schema=schema,
            format_name=format_name,
        )
        retry_index = 0
        calls = 0
        while True:
            self._reserve_call()
            calls += 1
            try:
                response = self._request(
                    agent_name=agent_name,
                    system_prompt=system_prompt,
                    payload=payload,
                    response_format=response_format,
                    schema=schema,
                )
                return response, calls, format_name
            except Exception as exc:
                if (
                    self.settings.llm_structured_output_mode == "auto"
                    and format_name == "json_schema"
                    and _schema_unsupported(exc)
                ):
                    with self._format_capability.lock:
                        self._format_capability.json_schema_unsupported = True
                    self._append_trace(
                        {
                            "agent": agent_name,
                            "status": "response_format_fallback",
                            "output_attempt": output_attempt,
                            "repair": repair,
                            "calls": calls,
                            "response_format": "json_schema",
                            "error": {
                                "category": "response_format_unsupported",
                                "status_code": _status_code(exc),
                            },
                        }
                    )
                    response_format = {"type": "json_object"}
                    format_name = "json_object"
                    with self._lock:
                        self._warnings.append(
                            f"{agent_name}: "
                            "LLM_JSON_SCHEMA_UNSUPPORTED_USED_JSON_OBJECT"
                        )
                    retry_index = 0
                    continue
                if _retryable(exc) and retry_index < self.settings.llm_max_retries:
                    self._append_trace(
                        {
                            "agent": agent_name,
                            "status": "provider_retry",
                            "output_attempt": output_attempt,
                            "repair": repair,
                            "calls": calls,
                            "response_format": format_name,
                            "error": {
                                "category": "retryable_provider_error",
                                "status_code": _status_code(exc),
                                "exception_type": "provider_exception",
                            },
                        }
                    )
                    time.sleep(min(2.0, 0.25 * (2**retry_index)))
                    retry_index += 1
                    continue
                self._append_trace(
                    {
                        "agent": agent_name,
                        "status": "provider_error",
                        "output_attempt": output_attempt,
                        "repair": repair,
                        "calls": calls,
                        "response_format": format_name,
                        "error": {
                            "category": "provider_error",
                            "status_code": _status_code(exc),
                            "exception_type": "provider_exception",
                        },
                    }
                )
                raise

    @staticmethod
    def _choice(response: Any) -> Any:
        choices = _response_value(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise PortfolioAgentOutputError(
                "LLM response did not contain choices",
                category="response_content",
                validation_errors=[
                    {"loc": ["choices"], "msg": "At least one choice is required"}
                ],
            )
        return choices[0]

    @staticmethod
    def _content(response: Any) -> str:
        message = _response_value(
            OpenAIStructuredAgentClient._choice(response),
            "message",
        )
        content = _response_value(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise PortfolioAgentOutputError(
                "LLM response content was empty",
                category="response_content",
                validation_errors=[
                    {"loc": ["choices", 0, "message", "content"], "msg": "Empty content"}
                ],
            )
        return content

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        try:
            value = _response_value(
                OpenAIStructuredAgentClient._choice(response),
                "finish_reason",
            )
        except PortfolioAgentOutputError:
            return None
        if value is None:
            return None
        return _safe_finish_reason_value(value)

    @staticmethod
    def _usage(response: Any) -> dict[str, Any] | None:
        usage = _response_value(response, "usage")
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return None
        safe: dict[str, int | float] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ):
            value = usage.get(key)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 1_000_000_000_000
            ):
                safe[key] = value
            elif (
                isinstance(value, float)
                and math.isfinite(value)
                and 0 <= value <= 1_000_000_000_000
            ):
                safe[key] = value
        return safe or None

    @staticmethod
    def _response_id_fingerprint(response: Any) -> str | None:
        value = _response_value(response, "id")
        if value is None:
            return None
        encoded = str(value).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _schema_property_names(schema: dict[str, Any]) -> set[str]:
        names: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict):
                    names.update(
                        key
                        for key in properties
                        if isinstance(key, str)
                    )
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(schema)
        return names

    @classmethod
    def _persisted_output_error(
        cls,
        error: PortfolioAgentOutputError,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        category = (
            error.category
            if error.category in SAFE_OUTPUT_ERROR_CATEGORIES
            else "invalid_output"
        )
        allowed_fields = cls._schema_property_names(schema)
        validation_errors: list[dict[str, Any]] = []
        for item in error.validation_errors[:20]:
            raw_location = item.get("loc")
            safe_location: list[str | int] = []
            if isinstance(raw_location, (list, tuple)):
                for part in raw_location:
                    if isinstance(part, int) and not isinstance(part, bool):
                        safe_location.append(part)
                    elif part == "$":
                        safe_location.append("$")
                    elif isinstance(part, str) and part in allowed_fields:
                        safe_location.append(part)
                    else:
                        safe_location.append("?")
            safe_error: dict[str, Any] = {"loc": safe_location}
            for numeric_key in ("line", "column"):
                numeric_value = item.get(numeric_key)
                if (
                    isinstance(numeric_value, int)
                    and not isinstance(numeric_value, bool)
                    and numeric_value >= 0
                ):
                    safe_error[numeric_key] = numeric_value
            validation_errors.append(safe_error)
        return {
            "category": category,
            "validation_errors": validation_errors,
            "truncated": error.truncated is True,
            "output_attempts": (
                error.output_attempts
                if isinstance(error.output_attempts, int)
                and error.output_attempts >= 0
                else 0
            ),
        }

    @staticmethod
    def _parse(content: str, response_model: type[ModelT]) -> ModelT:
        def reject_constant(value: str) -> None:
            raise PortfolioAgentOutputError(
                "Agent response contained a forbidden non-finite JSON constant",
                category="strict_json",
                response_model=response_model.__name__,
                validation_errors=[
                    {"loc": ["$"], "msg": f"Non-finite constant {value} is forbidden"}
                ],
            )

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, value in pairs:
                if key in output:
                    raise PortfolioAgentOutputError(
                        "Agent response contained a duplicate JSON key",
                        category="strict_json",
                        response_model=response_model.__name__,
                        validation_errors=[
                            {"loc": [key], "msg": "Duplicate JSON key is forbidden"}
                        ],
                    )
                output[key] = value
            return output

        try:
            payload = json.loads(
                content,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except json.JSONDecodeError as exc:
            raise PortfolioAgentOutputError(
                f"Agent response did not match {response_model.__name__}",
                category="json_decode",
                response_model=response_model.__name__,
                validation_errors=[
                    {
                        "loc": ["$"],
                        "msg": exc.msg,
                        "line": exc.lineno,
                        "column": exc.colno,
                    }
                ],
            ) from exc
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            errors = []
            for item in exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[:20]:
                errors.append(
                    {
                        "loc": list(item.get("loc") or ()),
                        "msg": str(item.get("msg") or "")[:300],
                        "type": str(item.get("type") or ""),
                    }
                )
            raise PortfolioAgentOutputError(
                f"Agent response did not match {response_model.__name__}",
                category="schema_validation",
                response_model=response_model.__name__,
                validation_errors=errors,
            ) from exc

    @staticmethod
    def _bounded_invalid_output(content: str) -> tuple[str, bool]:
        if len(content) <= MAX_REPAIR_OUTPUT_CHARS:
            return content, False
        marker = (
            "\n...[invalid output truncated for repair; "
            f"original_chars={len(content)}]...\n"
        )
        available = MAX_REPAIR_OUTPUT_CHARS - len(marker)
        head_length = available // 2
        tail_length = available - head_length
        return (
            content[:head_length] + marker + content[-tail_length:],
            True,
        )

    @staticmethod
    def _repair_payload(
        *,
        agent_name: str,
        original_system_prompt: str,
        original_payload: dict[str, Any],
        invalid_output: str,
        error: PortfolioAgentOutputError,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        bounded_output, output_was_truncated = (
            OpenAIStructuredAgentClient._bounded_invalid_output(invalid_output)
        )
        return {
            "instruction": (
                "Repair only the output format and schema violations. Preserve the "
                "original analysis where it is compatible with the contract. Return "
                "only the corrected JSON object."
            ),
            "original_task": {
                "agent_name": agent_name,
                "system_prompt": original_system_prompt,
                "payload": original_payload,
            },
            "previous_invalid_output": bounded_output,
            "previous_invalid_output_truncated": output_was_truncated,
            "previous_invalid_output_original_chars": len(invalid_output),
            "validation_errors": [
                dict(item) for item in error.validation_errors[:12]
            ],
            "required_schema": schema,
        }

    def invoke(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT:
        schema = response_model.model_json_schema()
        original_system_prompt = system_prompt
        original_payload = payload
        invalid_output = ""
        last_error: PortfolioAgentOutputError | None = None
        total_call_count = 0

        for retry_index in range(self.settings.multi_agent_output_retries + 1):
            output_attempt = retry_index + 1
            repair = retry_index > 0
            request_system_prompt = system_prompt
            request_payload = payload
            if repair:
                assert last_error is not None
                with self._lock:
                    self._output_repair_attempts += 1
                request_system_prompt = (
                    "You repair invalid structured agent output. Use the original "
                    "task and previous output below. Return only one JSON object "
                    "matching the required schema, with no markdown or commentary."
                )
                request_payload = self._repair_payload(
                    agent_name=agent_name,
                    original_system_prompt=original_system_prompt,
                    original_payload=original_payload,
                    invalid_output=invalid_output,
                    error=last_error,
                    schema=schema,
                )

            response, call_count, format_name = self._invoke_provider(
                agent_name=agent_name,
                system_prompt=request_system_prompt,
                payload=request_payload,
                schema=schema,
                output_attempt=output_attempt,
                repair=repair,
            )
            total_call_count += call_count
            finish_reason = self._finish_reason(response)
            invalid_output = ""
            try:
                invalid_output = self._content(response)
                if finish_reason == "length":
                    raise PortfolioAgentOutputError(
                        f"Agent response did not match {response_model.__name__}",
                        category="truncated_output",
                        response_model=response_model.__name__,
                        validation_errors=[
                            {
                                "loc": ["$"],
                                "msg": (
                                    "Model output was truncated because "
                                    "finish_reason=length"
                                ),
                            }
                        ],
                        finish_reason=finish_reason,
                        truncated=True,
                    )
                result = self._parse(invalid_output, response_model)
            except PortfolioAgentOutputError as exc:
                if exc.finish_reason is None:
                    exc.finish_reason = finish_reason
                    exc.args = (exc._render(),)
                exc.set_output_attempts(output_attempt)
                last_error = exc
                self._append_trace(
                    {
                        "agent": agent_name,
                        "status": "invalid_output",
                        "output_attempt": output_attempt,
                        "repair": repair,
                        "calls": total_call_count,
                        "provider_calls": call_count,
                        "response_format": format_name,
                        "response_id_sha256": (
                            self._response_id_fingerprint(response)
                        ),
                        "finish_reason": finish_reason,
                        "truncated": exc.truncated,
                        "content_length": len(invalid_output),
                        "usage": self._usage(response),
                        "error": self._persisted_output_error(
                            exc,
                            schema,
                        ),
                    }
                )
                if retry_index >= self.settings.multi_agent_output_retries:
                    raise
                continue

            with self._lock:
                self._validated_outputs += 1
            self._append_trace(
                {
                    "agent": agent_name,
                    "status": "success",
                    "output_attempt": output_attempt,
                    "repair": repair,
                    "calls": total_call_count,
                    "provider_calls": call_count,
                    "response_format": format_name,
                    "response_id_sha256": (
                        self._response_id_fingerprint(response)
                    ),
                    "finish_reason": finish_reason,
                    "truncated": False,
                    "usage": self._usage(response),
                }
            )
            return result

        raise AssertionError("Structured output retry loop exited unexpectedly")

    def meta(self) -> dict[str, Any]:
        resolved_response_format = self._selected_format_name()
        with self._lock:
            return {
                "calls": self._calls,
                "provider_attempts": self._calls,
                "validated_outputs": self._validated_outputs,
                "output_repair_attempts": self._output_repair_attempts,
                "configured_response_format": (
                    self.settings.llm_structured_output_mode
                ),
                "resolved_response_format": resolved_response_format,
                "warnings": list(dict.fromkeys(self._warnings)),
                "trace": list(self._trace),
            }

    def new_run_session(self) -> "OpenAIStructuredAgentClient":
        """Share the provider transport but reset budget and trace per decision run."""

        return OpenAIStructuredAgentClient(
            self.settings,
            client=self._client(),
            _format_capability=self._format_capability,
        )


class PortfolioMultiAgentDecisionEngine:
    """Convert a complete A-share pool snapshot into one coherent allocation."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        agent_client: StructuredAgentClient | None = None,
    ) -> None:
        self.settings = settings
        self._injected_agent_client = agent_client
        self._client_template = OpenAIStructuredAgentClient(
            settings,
            client=client,
        )

    @staticmethod
    def _action(current: float, target: float) -> tuple[str, float]:
        tolerance = max(0.01, current * 0.005)
        if abs(target - current) <= tolerance:
            return "hold", current
        if target > current:
            return "increase", target
        if target <= tolerance and current > 0:
            return "close", 0.0
        return "decrease", target

    def _raw_decisions(
        self,
        decision_input: DecisionInput,
        state: PortfolioAgentState,
    ) -> dict[str, dict[str, Any]]:
        assert state.final_allocation is not None
        targets = {
            target.symbol: target for target in state.final_allocation.targets
        }
        decision_quality = str(
            getattr(state, "decision_quality", "healthy") or "healthy"
        )
        reduce_only = decision_quality != "healthy"
        positions = decision_input.portfolio.position_map()
        decisions: dict[str, dict[str, Any]] = {}
        for symbol in decision_input.symbols:
            snapshot = decision_input.market.get(symbol)
            if snapshot is None:
                continue
            try:
                reference_price = float(snapshot.reference_price)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(reference_price) or reference_price <= 0:
                continue
            position = positions.get(symbol)
            current = reference_price * (
                position.shares if position else 0
            )
            allocation = targets.get(symbol)
            if allocation is not None:
                target_value = max(
                    0.0,
                    float(state.total_assets) * float(allocation.target_weight),
                )
                confidence = float(allocation.confidence)
                reasons = list(allocation.reasons)
            elif current > 0:
                target_value = current
                confidence = 0.0
                reasons = [
                    "Safe hold: an existing holding was outside the final allocation"
                ]
            else:
                target_value = 0.0
                confidence = float(
                    state.combined_scores.get(symbol, {}).get("confidence", 0.0)
                )
                reasons = [
                    "Not selected for the bounded multi-agent research shortlist"
                ]
            if reduce_only and target_value > current:
                target_value = current
                reasons = [
                    (
                        f"Reduce-only safety gate: decision quality is "
                        f"{decision_quality}"
                    ),
                    *reasons,
                ]
            action, normalized_target = self._action(current, target_value)
            decisions[symbol] = {
                "action": action,
                "target_cash_amount": normalized_target,
                "confidence": max(0.0, min(1.0, confidence)),
                "reasons": reasons[:5],
            }
        return decisions

    @staticmethod
    def _client_meta(agent_client: StructuredAgentClient) -> dict[str, Any]:
        meta_method = getattr(agent_client, "meta", None)
        if callable(meta_method):
            try:
                meta = meta_method()
            except Exception:
                meta = None
            if isinstance(meta, dict):
                if type(agent_client) is OpenAIStructuredAgentClient:
                    return meta
                # A custom client can report counters, but arbitrary text,
                # warnings, and trace records are not a trusted persistence
                # boundary.
                return {
                    "calls": meta.get("calls", 0),
                    "provider_attempts": meta.get(
                        "provider_attempts",
                        meta.get("calls", 0),
                    ),
                    "validated_outputs": meta.get(
                        "validated_outputs",
                        0,
                    ),
                    "output_repair_attempts": meta.get(
                        "output_repair_attempts",
                        0,
                    ),
                    "warnings": [],
                    "trace": [],
                }
        raw_calls = getattr(agent_client, "calls", 0)
        calls = raw_calls if isinstance(raw_calls, int) else 0
        return {
            "calls": calls,
            "provider_attempts": calls,
            "validated_outputs": 0,
            "output_repair_attempts": 0,
            "warnings": [],
            "trace": [],
        }

    def decide(
        self,
        decision_input: DecisionInput,
        on_stage: StageCallback | None = None,
    ) -> RawDecisionBundle:
        if on_stage:
            on_stage("building_features")
        if not decision_input.market:
            return RawDecisionBundle(
                decisions={},
                meta={
                    "engine": "failed_safe_hold",
                    "model": self.settings.llm_model,
                    "calls": 0,
                    "provider_attempts": 0,
                    "validated_outputs": 0,
                    "output_repair_attempts": 0,
                    "configured_response_format": (
                        self.settings.llm_structured_output_mode
                    ),
                    "resolved_response_format": None,
                    "decision_quality": "failed",
                    "analysis_coverage": 0.0,
                    "stage_health": {
                        "preparation": {
                            "status": "failed",
                            "available_symbol_count": 0,
                            "requested_symbol_count": len(
                                decision_input.symbols
                            ),
                        }
                    },
                },
                warnings=("No valid market context was available for the agent graph",),
            )
        preparation_client = (
            self._injected_agent_client or self._client_template
        )
        try:
            state = PortfolioAgentGraph(
                self.settings,
                preparation_client,
            ).prepare(decision_input)
        except Exception as exc:
            failure_type = _safe_exception_type(exc)
            return RawDecisionBundle(
                decisions={},
                meta={
                    "engine": "failed_safe_hold",
                    "model": self.settings.llm_model,
                    "calls": 0,
                    "provider_attempts": 0,
                    "validated_outputs": 0,
                    "output_repair_attempts": 0,
                    "configured_response_format": (
                        self.settings.llm_structured_output_mode
                    ),
                    "resolved_response_format": None,
                    "decision_quality": "failed",
                    "analysis_coverage": 0.0,
                    "stage_health": {
                        "preparation": {
                            "status": "failed",
                            "available_symbol_count": 0,
                            "requested_symbol_count": len(
                                decision_input.symbols
                            ),
                        }
                    },
                    "failure_category": failure_type,
                },
                warnings=(
                    "Multi-agent preparation failed safely; no target changes "
                    f"were emitted ({failure_type})",
                ),
            )
        if on_stage:
            on_stage("calling_llm")
        agent_client = (
            self._injected_agent_client
            or self._client_template.new_run_session()
        )
        graph = PortfolioAgentGraph(self.settings, agent_client)
        run_error: Exception | None = None
        try:
            state = graph.run_prepared(state)
        except Exception as exc:
            run_error = exc
            state.decision_quality = "failed"
        client_meta = self._client_meta(agent_client)
        raw_client_warnings = client_meta.get("warnings", [])
        client_warnings = (
            raw_client_warnings
            if isinstance(raw_client_warnings, (list, tuple))
            else []
        )
        raw_trace = client_meta.get("trace", [])
        trace = raw_trace if isinstance(raw_trace, list) else []
        raw_calls = client_meta.get("calls", 0)
        calls = raw_calls if isinstance(raw_calls, int) else 0
        raw_provider_attempts = client_meta.get("provider_attempts", calls)
        provider_attempts = (
            raw_provider_attempts
            if isinstance(raw_provider_attempts, int)
            else calls
        )
        raw_validated_outputs = client_meta.get("validated_outputs", 0)
        validated_outputs = (
            raw_validated_outputs
            if isinstance(raw_validated_outputs, int)
            else 0
        )
        raw_output_repair_attempts = client_meta.get(
            "output_repair_attempts", 0
        )
        output_repair_attempts = (
            raw_output_repair_attempts
            if isinstance(raw_output_repair_attempts, int)
            else 0
        )
        configured_response_format = client_meta.get(
            "configured_response_format",
            self.settings.llm_structured_output_mode,
        )
        resolved_response_format = client_meta.get(
            "resolved_response_format",
            self.settings.llm_structured_output_mode,
        )
        decision_quality = str(
            getattr(state, "decision_quality", "healthy") or "healthy"
        )
        raw_analysis_coverage = getattr(state, "analysis_coverage", 1.0)
        analysis_coverage = (
            float(raw_analysis_coverage)
            if isinstance(raw_analysis_coverage, (int, float))
            else 1.0
        )
        raw_stage_health = getattr(state, "stage_health", {})
        stage_health = (
            dict(raw_stage_health)
            if isinstance(raw_stage_health, dict)
            else {}
        )
        warnings = [
            *state.warnings,
            *[str(item) for item in client_warnings],
        ]
        if run_error is not None:
            try:
                artifacts = state.artifacts()
            except Exception:
                artifacts = {}
            failure_type = _safe_exception_type(run_error)
            failure_warning = (
                "Multi-agent decision failed safely; no target changes were "
                f"emitted ({failure_type})"
            )
            return RawDecisionBundle(
                decisions={},
                meta={
                    "engine": "failed_safe_hold",
                    "model": self.settings.llm_model,
                    "calls": calls,
                    "provider_attempts": provider_attempts,
                    "validated_outputs": validated_outputs,
                    "output_repair_attempts": output_repair_attempts,
                    "configured_response_format": (
                        configured_response_format
                    ),
                    "resolved_response_format": resolved_response_format,
                    "agent_trace": trace,
                    "agent_artifacts": artifacts,
                    "valuation_complete": state.valuation_complete,
                    "decision_quality": "failed",
                    "analysis_coverage": analysis_coverage,
                    "stage_health": stage_health,
                    "failure_category": failure_type,
                },
                warnings=tuple(
                    dict.fromkeys(
                        [
                            failure_warning,
                            *[str(item) for item in state.warnings],
                            *[str(item) for item in client_warnings],
                        ]
                    )
                ),
            )
        return RawDecisionBundle(
            decisions=self._raw_decisions(decision_input, state),
            meta={
                "engine": "portfolio_multi_agent",
                "model": self.settings.llm_model,
                "calls": calls,
                "provider_attempts": provider_attempts,
                "validated_outputs": validated_outputs,
                "output_repair_attempts": output_repair_attempts,
                "configured_response_format": configured_response_format,
                "resolved_response_format": resolved_response_format,
                "agent_trace": trace,
                "agent_artifacts": state.artifacts(),
                "valuation_complete": state.valuation_complete,
                "decision_quality": decision_quality,
                "analysis_coverage": analysis_coverage,
                "stage_health": stage_health,
            },
            warnings=tuple(dict.fromkeys(warnings)),
        )

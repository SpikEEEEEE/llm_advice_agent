from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, TypeVar

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


class PortfolioAgentOutputError(ValueError):
    """An agent response was not strict JSON matching its declared contract."""


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

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._injected_client = client
        self._client_instance: Any | None = None
        self._lock = threading.RLock()
        self._calls = 0
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
                "\nFallback JSON contract; return exactly this schema with no extra fields:\n"
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
    ) -> tuple[Any, int, str]:
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(agent_name),
                "strict": True,
                "schema": schema,
            },
        }
        format_name = "json_schema"
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
                if format_name == "json_schema" and _schema_unsupported(exc):
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
                    time.sleep(min(2.0, 0.25 * (2**retry_index)))
                    retry_index += 1
                    continue
                raise

    @staticmethod
    def _content(response: Any) -> str:
        choices = _response_value(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise PortfolioAgentOutputError("LLM response did not contain choices")
        message = _response_value(choices[0], "message")
        content = _response_value(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise PortfolioAgentOutputError("LLM response content was empty")
        return content

    @staticmethod
    def _parse(content: str, response_model: type[ModelT]) -> ModelT:
        def reject_constant(value: str) -> None:
            raise PortfolioAgentOutputError(
                f"Non-finite JSON constant is forbidden: {value}"
            )

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, value in pairs:
                if key in output:
                    raise PortfolioAgentOutputError(
                        f"Duplicate JSON key is forbidden: {key}"
                    )
                output[key] = value
            return output

        try:
            payload = json.loads(
                content,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
            return response_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, PortfolioAgentOutputError) as exc:
            raise PortfolioAgentOutputError(
                f"Agent response did not match {response_model.__name__}"
            ) from exc

    def invoke(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT:
        schema = response_model.model_json_schema()
        response, call_count, format_name = self._invoke_provider(
            agent_name=agent_name,
            system_prompt=system_prompt,
            payload=payload,
            schema=schema,
        )
        result = self._parse(self._content(response), response_model)
        usage = _response_value(response, "usage")
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            usage = None
        with self._lock:
            self._trace.append(
                {
                    "agent": agent_name,
                    "calls": call_count,
                    "response_format": format_name,
                    "response_id": _response_value(response, "id"),
                    "usage": usage,
                }
            )
        return result

    def meta(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "warnings": list(dict.fromkeys(self._warnings)),
                "trace": list(self._trace),
            }

    def new_run_session(self) -> "OpenAIStructuredAgentClient":
        """Share the provider transport but reset budget and trace per decision run."""

        return OpenAIStructuredAgentClient(
            self.settings,
            client=self._client(),
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
        positions = decision_input.portfolio.position_map()
        decisions: dict[str, dict[str, Any]] = {}
        for symbol in decision_input.symbols:
            snapshot = decision_input.market.get(symbol)
            if snapshot is None:
                continue
            position = positions.get(symbol)
            current = float(snapshot.reference_price) * (
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
            meta = meta_method()
            if isinstance(meta, dict):
                return meta
        raw_calls = getattr(agent_client, "calls", 0)
        calls = raw_calls if isinstance(raw_calls, int) else 0
        return {"calls": calls, "warnings": [], "trace": []}

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
                    "engine": "portfolio_multi_agent",
                    "model": self.settings.llm_model,
                    "calls": 0,
                },
                warnings=("No valid market context was available for the agent graph",),
            )
        preparation_client = (
            self._injected_agent_client or self._client_template
        )
        state = PortfolioAgentGraph(
            self.settings,
            preparation_client,
        ).prepare(decision_input)
        if on_stage:
            on_stage("calling_llm")
        agent_client = (
            self._injected_agent_client
            or self._client_template.new_run_session()
        )
        graph = PortfolioAgentGraph(self.settings, agent_client)
        state = graph.run_prepared(state)
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
        warnings = [
            *state.warnings,
            *[str(item) for item in client_warnings],
        ]
        return RawDecisionBundle(
            decisions=self._raw_decisions(decision_input, state),
            meta={
                "engine": "portfolio_multi_agent",
                "model": self.settings.llm_model,
                "calls": calls,
                "agent_trace": trace,
                "agent_artifacts": state.artifacts(),
                "valuation_complete": state.valuation_complete,
            },
            warnings=tuple(dict.fromkeys(warnings)),
        )

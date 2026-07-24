from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ConfigDict

from app.adapters.portfolio_multi_agent import (
    MAX_REPAIR_OUTPUT_CHARS,
    OpenAIStructuredAgentClient,
    PortfolioAgentOutputError,
    PortfolioMultiAgentDecisionEngine,
)
from app.agents.portfolio_graph import PortfolioAgentGraph, PortfolioAgentState
from app.agents.portfolio_schemas import FinalPortfolioAllocation
from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    SymbolMarketSnapshot,
)

from .helpers import make_settings


class ExampleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class UnsupportedSchemaError(RuntimeError):
    status_code = 400


class FakeCompletions:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _response(
    content: str,
    *,
    response_id: str = "response",
    finish_reason: str = "stop",
) -> Any:
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        ),
    )


def _provider(results: list[Any]) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(results)
    return (
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        completions,
    )


def _invoke(client: OpenAIStructuredAgentClient) -> ExampleOutput:
    return client.invoke(
        agent_name="test_agent",
        system_prompt="Analyze the supplied point-in-time task.",
        payload={"symbol": "600519.SH", "feature": 1.0},
        response_model=ExampleOutput,
    )


def _decision_input() -> DecisionInput:
    data_date = date(2026, 1, 5)
    return DecisionInput(
        run_id="failed-run",
        portfolio=PortfolioSnapshot(
            portfolio_id="portfolio",
            version=1,
            name="portfolio",
            cash=Decimal("100000"),
            positions=(),
        ),
        mode="rebalance",
        as_of=datetime(
            2026,
            1,
            5,
            16,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ),
        data_date=data_date,
        valid_for_session=date(2026, 1, 6),
        universe_version="test",
        symbols=("600519.SH",),
        market={
            "600519.SH": SymbolMarketSnapshot(
                symbol="600519.SH",
                data_date=data_date,
                reference_price=Decimal("1000"),
                bars=None,
            )
        },
    )


def test_deepseek_auto_uses_json_object_without_schema_probe(tmp_path):
    provider, completions = _provider([_response('{"value":1}')])
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        llm_structured_output_mode="auto",
        multi_agent_output_retries=0,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    result = _invoke(client)

    assert result.value == 1
    assert len(completions.requests) == 1
    assert completions.requests[0]["response_format"] == {
        "type": "json_object"
    }
    assert "JSON contract" in completions.requests[0]["messages"][0][
        "content"
    ]
    assert client.meta()["resolved_response_format"] == "json_object"
    assert client.meta()["provider_attempts"] == 1


def test_schema_fallback_is_memoized_across_sessions_for_same_client(tmp_path):
    provider, completions = _provider(
        [
            UnsupportedSchemaError(
                "response_format json_schema is unsupported"
            ),
            _response('{"value":1}', response_id="fallback"),
            _response('{"value":2}', response_id="memoized"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://compatible-provider.invalid/v1",
        llm_structured_output_mode="auto",
        multi_agent_output_retries=0,
    )
    template = OpenAIStructuredAgentClient(settings, client=provider)

    first_session = template.new_run_session()
    second_session = template.new_run_session()
    assert _invoke(first_session).value == 1
    assert _invoke(second_session).value == 2

    assert [
        request["response_format"]["type"]
        for request in completions.requests
    ] == ["json_schema", "json_object", "json_object"]
    assert first_session.meta()["provider_attempts"] == 2
    assert second_session.meta()["provider_attempts"] == 1
    assert second_session.meta()["resolved_response_format"] == "json_object"
    assert first_session.meta()["trace"][0]["status"] == (
        "response_format_fallback"
    )


def test_explicit_json_schema_does_not_silently_fallback(tmp_path):
    provider, completions = _provider(
        [
            UnsupportedSchemaError(
                "response_format json_schema is unsupported"
            )
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_structured_output_mode="json_schema",
        multi_agent_output_retries=0,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    with pytest.raises(UnsupportedSchemaError):
        _invoke(client)

    assert len(completions.requests) == 1
    assert completions.requests[0]["response_format"]["type"] == "json_schema"
    assert client.meta()["resolved_response_format"] == "json_schema"
    assert client.meta()["trace"][0]["status"] == "provider_error"


def test_json_decode_failure_is_repaired_with_original_task_and_schema(tmp_path):
    invalid = '{"value":'
    provider, completions = _provider(
        [
            _response(invalid, response_id="invalid"),
            _response('{"value":7}', response_id="repaired"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    result = _invoke(client)

    assert result.value == 7
    repair_request = json.loads(
        completions.requests[1]["messages"][1]["content"]
    )
    assert repair_request["previous_invalid_output"] == invalid
    assert repair_request["original_task"]["agent_name"] == "test_agent"
    assert repair_request["original_task"]["payload"]["symbol"] == "600519.SH"
    assert repair_request["validation_errors"][0]["loc"] == ["$"]
    assert repair_request["required_schema"]["properties"]["value"][
        "type"
    ] == "integer"

    meta = client.meta()
    assert meta["calls"] == 2
    assert meta["provider_attempts"] == 2
    assert meta["validated_outputs"] == 1
    assert meta["output_repair_attempts"] == 1
    assert [item["status"] for item in meta["trace"]] == [
        "invalid_output",
        "success",
    ]
    assert meta["trace"][0]["error"]["category"] == "json_decode"
    assert meta["trace"][1]["repair"] is True
    assert meta["trace"][1]["calls"] == 2


def test_pydantic_validation_errors_are_safe_and_repairable(tmp_path):
    secret_marker = "TOP_SECRET_SHOULD_NOT_APPEAR_IN_DIAGNOSTICS"
    provider, _ = _provider(
        [
            _response(
                json.dumps({"value": secret_marker}),
                response_id="invalid",
            ),
            _response('{"value":9}', response_id="repaired"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    assert _invoke(client).value == 9

    invalid_trace = client.meta()["trace"][0]
    assert invalid_trace["error"]["category"] == "schema_validation"
    assert invalid_trace["error"]["validation_errors"][0]["loc"] == ["value"]
    assert secret_marker not in json.dumps(
        invalid_trace,
        ensure_ascii=False,
    )


def test_persisted_trace_whitelists_provider_controlled_metadata(tmp_path):
    secret_marker = "TOP_SECRET_VENDOR_TOKEN_sk-live-ABC123"
    response = _response(
        '{"value":4}',
        response_id=secret_marker,
        finish_reason=secret_marker,
    )
    response.usage = SimpleNamespace(
        model_dump=lambda: {
            "prompt_tokens": 10**10000,
            "total_tokens": 7,
            "vendor_note": secret_marker,
        }
    )
    provider, _ = _provider([response])
    client = OpenAIStructuredAgentClient(
        replace(
            make_settings(tmp_path),
            llm_base_url="https://api.deepseek.com",
        ),
        client=provider,
    )

    assert _invoke(client).value == 4
    trace = client.meta()["trace"][0]
    serialized = json.dumps(trace, ensure_ascii=False)

    assert secret_marker not in serialized
    assert trace["finish_reason"] == "other"
    assert trace["usage"] == {"total_tokens": 7}
    assert len(trace["response_id_sha256"]) == 16


def test_output_error_string_and_diagnostics_are_value_whitelisted():
    secret_marker = "TOP_SECRET_VENDOR_TOKEN_sk-live-ABC123"
    error = PortfolioAgentOutputError(
        secret_marker,
        category=secret_marker,
        response_model=secret_marker,
        validation_errors=[
            {
                "loc": [secret_marker],
                "msg": secret_marker,
            }
        ],
        finish_reason=secret_marker,
    )

    persisted = json.dumps(
        {
            "text": str(error),
            "diagnostics": error.diagnostics(),
        },
        ensure_ascii=False,
    )

    assert secret_marker not in persisted
    assert error.diagnostics()["category"] == "invalid_output"
    assert error.diagnostics()["finish_reason"] == "other"
    assert error.diagnostics()["validation_errors"][0]["loc"] == ["?"]


def test_persisted_validation_location_is_schema_whitelisted(tmp_path):
    secret_key = "TOP_SECRET_VENDOR_TOKEN"
    # json.dumps cannot preserve duplicate keys, so construct it explicitly.
    invalid = f'{{"{secret_key}":1,"{secret_key}":2}}'
    provider, _ = _provider(
        [
            _response(invalid),
            _response('{"value":5}'),
        ]
    )
    client = OpenAIStructuredAgentClient(
        replace(
            make_settings(tmp_path),
            llm_base_url="https://api.deepseek.com",
            multi_agent_output_retries=1,
        ),
        client=provider,
    )

    assert _invoke(client).value == 5
    trace = client.meta()["trace"][0]

    assert secret_key not in json.dumps(trace, ensure_ascii=False)
    assert trace["error"]["validation_errors"][0]["loc"] == ["?"]

    malicious_type = "sk_live_abc123"
    error = PortfolioAgentOutputError(
        "invalid",
        validation_errors=[
            {
                "loc": ["value"],
                "type": malicious_type,
            }
        ],
    )
    persisted = client._persisted_output_error(
        error,
        ExampleOutput.model_json_schema(),
    )
    assert malicious_type not in json.dumps(persisted)


def test_repair_payload_bounds_invalid_output_and_preserves_both_ends(tmp_path):
    invalid = '{"prefix":"' + ("x" * 20_000) + '","suffix":"end"'
    provider, completions = _provider(
        [
            _response(invalid, response_id="oversized"),
            _response('{"value":3}', response_id="repaired"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    assert _invoke(client).value == 3

    repair_request = json.loads(
        completions.requests[1]["messages"][1]["content"]
    )
    bounded = repair_request["previous_invalid_output"]
    assert len(bounded) <= MAX_REPAIR_OUTPUT_CHARS
    assert bounded.startswith('{"prefix":"')
    assert bounded.endswith('","suffix":"end"')
    assert "invalid output truncated for repair" in bounded
    assert repair_request["previous_invalid_output_truncated"] is True
    assert repair_request["previous_invalid_output_original_chars"] == len(
        invalid
    )


def test_exhausted_output_repairs_raise_detailed_safe_error(tmp_path):
    provider, _ = _provider(
        [
            _response('{"value":"wrong"}', response_id="invalid-1"),
            _response('{"value":"still-wrong"}', response_id="invalid-2"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    with pytest.raises(PortfolioAgentOutputError) as caught:
        _invoke(client)

    error = caught.value
    assert error.category == "schema_validation"
    assert error.output_attempts == 2
    assert error.validation_errors[0]["loc"] == ["value"]
    assert "schema_validation" in str(error)
    assert "still-wrong" not in str(error)
    meta = client.meta()
    assert meta["provider_attempts"] == 2
    assert meta["validated_outputs"] == 0
    assert meta["output_repair_attempts"] == 1
    assert [item["status"] for item in meta["trace"]] == [
        "invalid_output",
        "invalid_output",
    ]


def test_finish_reason_length_is_traced_as_truncation_and_repaired(tmp_path):
    provider, _ = _provider(
        [
            _response(
                '{"value":',
                response_id="truncated",
                finish_reason="length",
            ),
            _response('{"value":11}', response_id="repaired"),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    assert _invoke(client).value == 11

    first_attempt = client.meta()["trace"][0]
    assert first_attempt["finish_reason"] == "length"
    assert first_attempt["truncated"] is True
    assert first_attempt["error"]["category"] == "truncated_output"
    assert first_attempt["error"]["truncated"] is True


def test_output_repair_cannot_exceed_provider_call_budget(tmp_path):
    provider, completions = _provider(
        [
            _response('{"value":"wrong"}'),
            _response('{"value":1}'),
        ]
    )
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_max_calls=1,
        multi_agent_output_retries=1,
    )
    client = OpenAIStructuredAgentClient(settings, client=provider)

    with pytest.raises(RuntimeError, match="call budget exhausted"):
        _invoke(client)

    assert len(completions.requests) == 1
    assert client.meta()["provider_attempts"] == 1
    assert client.meta()["output_repair_attempts"] == 1


def test_graph_failure_returns_safe_hold_with_actual_provider_cost(
    tmp_path,
    monkeypatch,
):
    provider, _ = _provider([_response('{"value":1}')])
    settings = replace(
        make_settings(tmp_path),
        llm_base_url="https://api.deepseek.com",
        multi_agent_output_retries=0,
    )
    agent_client = OpenAIStructuredAgentClient(settings, client=provider)
    decision_input = _decision_input()
    prepared = PortfolioAgentState(
        decision_input=decision_input,
        symbol_contexts={},
        current_values={},
        current_weights={},
        total_assets=100000.0,
        valuation_complete=True,
    )

    monkeypatch.setattr(
        PortfolioAgentGraph,
        "prepare",
        lambda self, value: prepared,
    )

    def fail_after_paid_call(self, state):
        self.agent_client.invoke(
            agent_name="technical_analyst",
            system_prompt="Return the contract",
            payload={"symbol": "600519.SH"},
            response_model=ExampleOutput,
        )
        state.analysis_coverage = 0.35
        state.stage_health = {"analysts": {"status": "failed"}}
        raise RuntimeError("SENSITIVE_PROVIDER_DETAIL")

    monkeypatch.setattr(
        PortfolioAgentGraph,
        "run_prepared",
        fail_after_paid_call,
    )

    bundle = PortfolioMultiAgentDecisionEngine(
        settings,
        agent_client=agent_client,
    ).decide(decision_input)

    assert bundle.decisions == {}
    assert bundle.meta["engine"] == "failed_safe_hold"
    assert bundle.meta["decision_quality"] == "failed"
    assert bundle.meta["calls"] == 1
    assert bundle.meta["provider_attempts"] == 1
    assert bundle.meta["validated_outputs"] == 1
    assert bundle.meta["output_repair_attempts"] == 0
    assert bundle.meta["configured_response_format"] == "auto"
    assert bundle.meta["resolved_response_format"] == "json_object"
    assert bundle.meta["analysis_coverage"] == 0.35
    assert bundle.meta["stage_health"]["analysts"]["status"] == "failed"
    assert bundle.meta["agent_trace"][0]["status"] == "success"
    assert "SENSITIVE_PROVIDER_DETAIL" not in " ".join(bundle.warnings)


def test_degraded_all_cash_portfolio_cannot_open_a_position(tmp_path):
    decision_input = _decision_input()
    state = PortfolioAgentState(
        decision_input=decision_input,
        symbol_contexts={},
        current_values={"600519.SH": 0.0},
        current_weights={"600519.SH": 0.0},
        total_assets=100000.0,
        valuation_complete=True,
        decision_quality="degraded",
        combined_scores={
            "600519.SH": {"score": 80.0, "confidence": 0.9}
        },
        final_allocation=FinalPortfolioAllocation.model_validate(
            {
                "targets": [
                    {
                        "symbol": "600519.SH",
                        "target_weight": 0.5,
                        "confidence": 0.9,
                        "reasons": ["high conviction"],
                    }
                ],
                "cash_weight": 0.5,
                "rationale": "Would buy without the safety gate",
            }
        ),
    )
    engine = PortfolioMultiAgentDecisionEngine(make_settings(tmp_path))

    decisions = engine._raw_decisions(decision_input, state)

    assert decisions["600519.SH"]["action"] == "hold"
    assert decisions["600519.SH"]["target_cash_amount"] == 0.0
    assert decisions["600519.SH"]["reasons"][0].startswith(
        "Reduce-only safety gate"
    )

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from app.core.config import Settings, has_configured_secret
from .helpers import make_settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_cash_ratio", Decimal("-0.01")),
        ("max_position_ratio", Decimal("1.01")),
        ("buy_fee_buffer_ratio", Decimal("-1")),
        ("llm_temperature", 2.1),
        ("market_history_days", 29),
        ("max_positions", 0),
        ("max_as_of_skew_minutes", 0),
        ("multi_agent_shortlist_size", 0),
        ("multi_agent_parallelism", 9),
        ("multi_agent_max_calls", 0),
        ("multi_agent_output_retries", -1),
        ("multi_agent_output_retries", 6),
        ("multi_agent_semantic_retries", -1),
        ("multi_agent_min_analysts", 1),
        ("multi_agent_min_analysts", 4),
        ("multi_agent_min_risk_reviews", 1),
        ("multi_agent_min_risk_reviews", 4),
        ("multi_agent_technical_weight", -0.1),
        ("decision_engine_mode", "unknown"),
        ("llm_structured_output_mode", "yaml"),
    ],
)
def test_risk_and_cache_settings_fail_fast(tmp_path, field, value):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError):
        replace(settings, **{field: value})


def test_placeholder_secrets_are_not_treated_as_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("BACKEND_API_KEY", "replace-me")
    assert has_configured_secret("BACKEND_API_KEY") is False
    with pytest.raises(ValueError, match="BACKEND_API_KEY"):
        Settings.from_env(tmp_path)


def test_multi_agent_settings_are_loaded_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_ENGINE", "portfolio_multi_agent")
    monkeypatch.setenv("MULTI_AGENT_SHORTLIST_SIZE", "12")
    monkeypatch.setenv("MULTI_AGENT_MAX_CALLS", "40")
    monkeypatch.setenv("MULTI_AGENT_OUTPUT_RETRIES", "2")
    monkeypatch.setenv("MULTI_AGENT_MIN_ANALYSTS", "3")
    monkeypatch.setenv("MULTI_AGENT_MIN_RISK_REVIEWS", "2")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_MODE", "json_object")

    settings = Settings.from_env(tmp_path)

    assert settings.decision_engine_mode == "portfolio_multi_agent"
    assert settings.multi_agent_shortlist_size == 12
    assert settings.multi_agent_max_calls == 40
    assert settings.multi_agent_output_retries == 2
    assert settings.multi_agent_min_analysts == 3
    assert settings.multi_agent_min_risk_reviews == 2
    assert settings.llm_structured_output_mode == "json_object"


def test_multi_agent_requires_at_least_one_positive_analyst_weight(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        replace(
            make_settings(tmp_path),
            multi_agent_technical_weight=0,
            multi_agent_fundamental_weight=0,
            multi_agent_news_weight=0,
        )

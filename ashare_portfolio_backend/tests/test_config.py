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

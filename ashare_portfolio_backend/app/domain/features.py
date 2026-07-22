from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.core.json_safety import sanitize_json_value
from app.domain.models import DecisionInput, SymbolMarketSnapshot


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _last(series: pd.Series) -> float | None:
    return _finite(series.iloc[-1]) if not series.empty else None


def _return(closes: pd.Series, periods: int) -> float | None:
    if len(closes) <= periods:
        return None
    start = _finite(closes.iloc[-periods - 1])
    end = _finite(closes.iloc[-1])
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1


def _rsi(closes: pd.Series, periods: int = 14) -> float | None:
    if len(closes) <= periods:
        return None
    changes = closes.diff().tail(periods)
    gains = changes.clip(lower=0).mean()
    losses = -changes.clip(upper=0).mean()
    if not math.isfinite(float(gains)) or not math.isfinite(float(losses)):
        return None
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return float(100 - 100 / (1 + gains / losses))


def _volatility(closes: pd.Series, periods: int = 20) -> float | None:
    returns = closes.pct_change(fill_method=None).dropna().tail(periods)
    if len(returns) < 2:
        return None
    value = float(returns.std(ddof=1) * math.sqrt(252))
    return value if math.isfinite(value) else None


def build_symbol_context(
    decision_input: DecisionInput,
    snapshot: SymbolMarketSnapshot,
) -> dict[str, Any]:
    """Build deterministic, point-in-time features from normalized data."""

    bars = snapshot.bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.date
    bars = bars[bars["date"] <= decision_input.data_date].sort_values("date")
    close_column = "adjusted_close" if "adjusted_close" in bars.columns else "close"
    bars[close_column] = pd.to_numeric(bars[close_column], errors="coerce")
    bars["volume"] = pd.to_numeric(
        bars.get("volume", pd.Series(index=bars.index, dtype=float)),
        errors="coerce",
    )
    valid = bars[close_column].map(
        lambda value: pd.notna(value)
        and math.isfinite(float(value))
        and float(value) > 0
    )
    bars = bars[valid]
    closes = bars[close_column]
    volumes = bars["volume"]
    recent_252 = closes.tail(252)
    latest = _last(closes)

    ma_5 = _finite(closes.tail(5).mean()) if len(closes) >= 5 else None
    ma_20 = _finite(closes.tail(20).mean()) if len(closes) >= 20 else None
    high_52w = _finite(recent_252.max()) if not recent_252.empty else None
    low_52w = _finite(recent_252.min()) if not recent_252.empty else None
    distance_from_high = (
        latest / high_52w - 1
        if latest is not None and high_52w is not None and high_52w > 0
        else None
    )
    recent_volume = _finite(volumes.tail(5).mean()) if len(volumes) >= 5 else None
    baseline_volume = _finite(volumes.tail(20).mean()) if len(volumes) >= 20 else None
    volume_ratio = (
        recent_volume / baseline_volume
        if recent_volume is not None
        and baseline_volume is not None
        and baseline_volume > 0
        else None
    )

    position = decision_input.portfolio.position_map().get(snapshot.symbol)
    current_shares = position.shares if position else 0
    average_cost = float(position.average_cost) if position else None
    unrealized_pnl_pct = (
        latest / average_cost - 1
        if latest is not None and average_cost is not None and average_cost > 0
        else None
    )

    payload = {
        "symbol": snapshot.symbol,
        "data_date": snapshot.data_date.isoformat(),
        "market": {
            "open": _last(pd.to_numeric(bars["open"], errors="coerce")),
            "high": _last(pd.to_numeric(bars["high"], errors="coerce")),
            "low": _last(pd.to_numeric(bars["low"], errors="coerce")),
            "close": latest,
            "close_7d": [_finite(value) for value in closes.tail(7).tolist()],
            "return_5d": _return(closes, 5),
            "return_20d": _return(closes, 20),
            "ma_5": ma_5,
            "ma_20": ma_20,
            "annualized_volatility_20d": _volatility(closes, 20),
            "rsi_14": _rsi(closes, 14),
            "adjusted_close_high_52w": high_52w,
            "adjusted_close_low_52w": low_52w,
            "distance_from_52w_high": distance_from_high,
            "volume_ratio_5d_to_20d": volume_ratio,
            "technical_price_basis": close_column,
        },
        "fundamentals": snapshot.fundamentals,
        "news": [
            {
                "title": item.get("title"),
                "summary": item.get("description"),
                "published_utc": item.get("published_utc"),
                "source": item.get("api_source"),
            }
            for item in snapshot.news
        ],
        "position": {
            "shares": current_shares,
            "available_shares": position.available_shares if position else 0,
            "average_cost": average_cost,
            "holding_days": position.holding_days if position else None,
            "current_position_value": latest * current_shares if latest else None,
            "unrealized_pnl_pct": unrealized_pnl_pct,
        },
        "data_quality_warnings": list(snapshot.data_quality_warnings),
    }
    sanitized = sanitize_json_value(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def build_decision_context(decision_input: DecisionInput) -> dict[str, Any]:
    return {
        "run_id": decision_input.run_id,
        "as_of": decision_input.as_of.isoformat(),
        "data_date": decision_input.data_date.isoformat(),
        "valid_for_session": decision_input.valid_for_session.isoformat(),
        "mode": decision_input.mode,
        "portfolio": {
            "cash": float(decision_input.portfolio.cash),
            "position_count": len(decision_input.portfolio.positions),
        },
        "symbols": [
            build_symbol_context(decision_input, decision_input.market[symbol])
            for symbol in decision_input.symbols
            if symbol in decision_input.market
        ],
    }

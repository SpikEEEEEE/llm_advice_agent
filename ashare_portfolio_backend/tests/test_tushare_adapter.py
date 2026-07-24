from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.adapters.tushare import LocalMarketCache, TushareMarketDataProvider
from app.ports.market_data import DataUnavailableError
from .helpers import make_settings


DATA_DATE = date(2026, 7, 21)


def _daily(count: int = 40) -> pd.DataFrame:
    days = [DATA_DATE - timedelta(days=offset) for offset in range(count - 1, -1, -1)]
    return pd.DataFrame(
        {
            "trade_date": [item.strftime("%Y%m%d") for item in reversed(days)],
            "open": [9.5] * count,
            "high": [10.5] * count,
            "low": [9.0] * count,
            "close": [10.0] * count,
            "vol": [1000.0] * count,
            "amount": [1000.0] * count,
        }
    )


class FakePro:
    def __init__(self) -> None:
        self.calendar_calls = 0
        self.news_failure = False
        self.daily_frame = _daily()

    def trade_cal(self, **_kwargs):
        self.calendar_calls += 1
        return pd.DataFrame({"cal_date": ["20260721"], "is_open": [1]})

    def daily(self, **_kwargs):
        return self.daily_frame.copy()

    def adj_factor(self, **_kwargs):
        return pd.DataFrame(
            {
                "trade_date": self.daily_frame["trade_date"].tolist(),
                "adj_factor": [2.0] * len(self.daily_frame),
            }
        )

    def daily_basic(self, **_kwargs):
        return pd.DataFrame(
            {
                "pe_ttm": [20.0],
                "pb": [3.0],
                "turnover_rate": [1.2],
                "volume_ratio": [0.9],
                "ps_ttm": [5.0],
                "total_mv": [1000000.0],
                "circ_mv": [800000.0],
            }
        )

    def fina_indicator(self, **_kwargs):
        return pd.DataFrame(
            {
                "ann_date": ["20260722", "20260430"],
                "end_date": ["20260630", "20260331"],
                "roe": [99.0, 12.0],
                "grossprofit_margin": [88.0, 35.0],
                "debt_to_assets": [1.0, 42.0],
                "netprofit_yoy": [100.0, 8.0],
                "or_yoy": [100.0, 6.0],
            }
        )

    def stock_basic(self, **_kwargs):
        return pd.DataFrame({"ts_code": ["600519.SH"], "name": ["贵州茅台"]})

    def major_news(self, **_kwargs):
        if self.news_failure:
            raise PermissionError("not entitled")
        return pd.DataFrame(
            {
                "title": ["贵州茅台发布公告", "贵州茅台盘后消息"],
                "content": ["已公开信息", "未来信息"],
                "pub_time": ["2026-07-21 15:00:00", "2026-07-21 17:00:00"],
                "src": ["test", "test"],
            }
        )


def _online_settings(tmp_path):
    return replace(make_settings(tmp_path), data_mode="auto", tushare_token="test")


def test_offline_calendar_requires_project_local_cache(tmp_path):
    provider = TushareMarketDataProvider(make_settings(tmp_path))
    with pytest.raises(DataUnavailableError, match="cached A-share"):
        provider._is_trading_day(DATA_DATE)


def test_online_calendar_result_is_cached_locally(tmp_path):
    client = FakePro()
    settings = _online_settings(tmp_path)
    provider = TushareMarketDataProvider(settings, client=client)

    assert provider._is_trading_day(DATA_DATE) is True
    assert provider._is_trading_day(DATA_DATE) is True
    assert client.calendar_calls == 1
    assert (settings.cache_path / "calendar" / "2026-07-21.json").exists()


def test_session_resolution_uses_after_close_semantics(tmp_path, monkeypatch):
    provider = TushareMarketDataProvider(make_settings(tmp_path))
    monkeypatch.setattr(provider, "_is_trading_day", lambda value: value.weekday() < 5)

    before = datetime(2026, 7, 22, 14, tzinfo=ZoneInfo("Asia/Shanghai"))
    after = datetime(2026, 7, 22, 16, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert provider.latest_completed_session(before) == date(2026, 7, 21)
    assert provider.latest_completed_session(after) == date(2026, 7, 22)
    assert provider.next_session(date(2026, 7, 22)) == date(2026, 7, 23)


def test_direct_provider_normalizes_units_and_enforces_point_in_time(tmp_path):
    provider = TushareMarketDataProvider(
        _online_settings(tmp_path),
        client=FakePro(),
    )
    as_of = datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai"))

    snapshot = provider.load_symbol("600519.SH", DATA_DATE, as_of)

    assert snapshot.reference_price == Decimal("10.0")
    assert snapshot.bars.iloc[-1]["date"] == DATA_DATE
    assert snapshot.bars.iloc[-1]["volume"] == 100000.0
    assert snapshot.fundamentals["roe_ratio"] == 0.12
    assert snapshot.fundamentals["announcement_date"] == "20260430"
    assert snapshot.fundamentals["turnover_ratio"] == 0.012
    assert snapshot.fundamentals["total_market_cap_cny"] == 10_000_000_000.0
    assert [item["title"] for item in snapshot.news] == ["贵州茅台发布公告"]
    assert snapshot.data_quality_warnings == ()


def test_provider_failure_keeps_price_but_marks_input_quality(tmp_path):
    client = FakePro()
    client.news_failure = True
    provider = TushareMarketDataProvider(_online_settings(tmp_path), client=client)

    snapshot = provider.load_symbol(
        "600519.SH",
        DATA_DATE,
        datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert snapshot.reference_price == Decimal("10.0")
    assert snapshot.news == ()
    assert "NEWS_PROVIDER_UNAVAILABLE_OR_UNAUTHORIZED" in (
        snapshot.data_quality_warnings
    )


def test_invalid_latest_open_falls_back_to_close(tmp_path):
    client = FakePro()
    client.daily_frame.loc[0, "open"] = float("inf")
    provider = TushareMarketDataProvider(_online_settings(tmp_path), client=client)

    snapshot = provider.load_symbol(
        "600519.SH",
        DATA_DATE,
        datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert snapshot.bars.iloc[-1]["open"] == snapshot.bars.iloc[-1]["close"]
    assert "INVALID_LATEST_OPEN_FALLBACK_TO_CLOSE" in (
        snapshot.data_quality_warnings
    )


def test_offline_mode_reuses_only_its_own_cache(tmp_path):
    settings = _online_settings(tmp_path)
    online = TushareMarketDataProvider(settings, client=FakePro())
    as_of = datetime(2026, 7, 21, 16, tzinfo=ZoneInfo("Asia/Shanghai"))
    online.load_symbol("600519.SH", DATA_DATE, as_of)

    offline = TushareMarketDataProvider(
        replace(settings, data_mode="offline_only", tushare_token=None),
        cache=LocalMarketCache(settings.cache_path),
    )
    snapshot = offline.load_symbol("600519.SH", DATA_DATE, as_of)

    assert snapshot.reference_price == Decimal("10.0")
    assert snapshot.fundamentals["pe_ttm"] == 20.0
    assert [item["title"] for item in snapshot.news] == ["贵州茅台发布公告"]


def test_historical_date_uses_eligible_rows_from_a_newer_cache(tmp_path):
    cache = LocalMarketCache(tmp_path / "cache")
    future_date = DATA_DATE + timedelta(days=1)
    cached = pd.DataFrame(
        {
            "date": [DATA_DATE - timedelta(days=1), DATA_DATE, future_date],
            "open": [9.0, 10.0, 11.0],
            "high": [9.5, 10.5, 11.5],
            "low": [8.5, 9.5, 10.5],
            "close": [9.0, 10.0, 11.0],
            "volume": [100.0, 100.0, 100.0],
            "amount_cny": [900.0, 1000.0, 1100.0],
            "vwap": [9.0, 10.0, 11.0],
            "adj_factor": [1.0, 1.0, 2.0],
        }
    )
    cache.write_bars("600519.SH", cached)
    provider = TushareMarketDataProvider(
        replace(make_settings(tmp_path), cache_path=cache.root),
        cache=cache,
    )

    bars, warnings = provider._bars("600519.SH", DATA_DATE)

    assert bars.iloc[-1]["date"] == DATA_DATE
    assert future_date not in set(bars["date"])
    assert bars.iloc[-1]["adjusted_close"] == 10.0
    assert warnings == []


def test_backtest_history_and_calendar_are_loaded_in_bounded_ranges(tmp_path):
    client = FakePro()
    provider = TushareMarketDataProvider(
        _online_settings(tmp_path),
        client=client,
    )
    history_start = DATA_DATE - timedelta(days=39)

    sessions = provider.sessions_between(DATA_DATE, DATA_DATE)
    history = provider.load_history(
        "600519.SH",
        history_start,
        DATA_DATE,
    )

    assert sessions == (DATA_DATE,)
    assert history.iloc[0]["date"] == history_start
    assert history.iloc[-1]["date"] == DATA_DATE
    assert history.iloc[-1]["adj_factor"] == 2.0

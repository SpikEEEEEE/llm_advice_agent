from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.core.config import Settings
from app.domain.models import SymbolMarketSnapshot
from app.ports.market_data import (
    DataUnavailableError,
    ProviderConfigurationError,
    StaleDataError,
)


CHINA_TZ = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def _plain(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


class LocalMarketCache:
    """Small project-local cache; it never reads another repository's storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()

    def _json_path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def read_json(self, kind: str, key: str) -> Any | None:
        path = self._json_path(kind, key)
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return None

    def write_json(self, kind: str, key: str, payload: Any) -> None:
        path = self._json_path(kind, key)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._atomic_text(path, encoded)

    def read_bars(self, symbol: str) -> pd.DataFrame | None:
        path = self.root / "bars" / f"{_safe_symbol(symbol)}.csv"
        with self._lock:
            try:
                return pd.read_csv(path)
            except (FileNotFoundError, OSError, pd.errors.ParserError):
                return None

    def write_bars(self, symbol: str, bars: pd.DataFrame) -> None:
        path = self.root / "bars" / f"{_safe_symbol(symbol)}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                bars.to_csv(temporary, index=False)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


class TushareMarketDataProvider:
    name = "tushare"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        cache: LocalMarketCache | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache or LocalMarketCache(settings.cache_path)
        self._injected_client = client
        self._client_instance: Any | None = None

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._client_instance is None:
            if not self.settings.tushare_token:
                raise ProviderConfigurationError("TUSHARE_TOKEN is not configured")
            import tushare as ts

            self._client_instance = ts.pro_api(self.settings.tushare_token)
            if hasattr(self._client_instance, "timeout"):
                self._client_instance.timeout = self.settings.tushare_timeout_seconds
        return self._client_instance

    def _online_allowed(self) -> bool:
        return self.settings.data_mode == "auto"

    def _is_trading_day(self, value: date) -> bool:
        key = value.isoformat()
        cached = self.cache.read_json("calendar", key)
        if isinstance(cached, bool):
            return cached
        if not self._online_allowed():
            raise DataUnavailableError(
                f"No cached A-share trading-calendar value for {key}"
            )
        try:
            frame = self._client().trade_cal(
                exchange="",
                start_date=value.strftime("%Y%m%d"),
                end_date=value.strftime("%Y%m%d"),
                fields="cal_date,is_open",
            )
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise DataUnavailableError(f"Tushare returned no calendar row for {key}")
            result = bool(int(frame.iloc[-1]["is_open"]))
        except DataUnavailableError:
            raise
        except Exception as exc:
            raise DataUnavailableError(f"Tushare calendar failed for {key}: {exc}") from exc
        self.cache.write_json("calendar", key, result)
        return result

    def latest_completed_session(self, as_of: datetime) -> date:
        local = as_of.astimezone(CHINA_TZ) if as_of.tzinfo else as_of.replace(tzinfo=CHINA_TZ)
        close_time = local.replace(
            hour=self.settings.market_close_hour,
            minute=self.settings.market_close_minute,
            second=0,
            microsecond=0,
        )
        candidate = local.date() if local >= close_time else local.date() - timedelta(days=1)
        for _ in range(40):
            if self._is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise DataUnavailableError("Could not resolve a completed A-share session")

    def next_session(self, after: date) -> date:
        candidate = after + timedelta(days=1)
        for _ in range(40):
            if self._is_trading_day(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise DataUnavailableError("Could not resolve the next A-share session")

    @staticmethod
    def _normalize_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame(), []
        output = frame.copy()
        if "trade_date" in output.columns:
            output["date"] = pd.to_datetime(
                output["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
            ).dt.date
        else:
            output["date"] = pd.to_datetime(output.get("date"), errors="coerce").dt.date

        warnings: list[str] = []
        for column in ("open", "high", "low", "close"):
            output[column] = pd.to_numeric(output.get(column), errors="coerce")
        if "volume" in output.columns:
            output["volume"] = pd.to_numeric(output["volume"], errors="coerce")
        else:
            output["volume"] = pd.to_numeric(output.get("vol"), errors="coerce") * 100
        if "amount_cny" in output.columns:
            output["amount_cny"] = pd.to_numeric(
                output["amount_cny"], errors="coerce"
            )
        elif "amount" in output.columns:
            output["amount_cny"] = pd.to_numeric(
                output["amount"], errors="coerce"
            ) * 1000
        else:
            output["amount_cny"] = float("nan")
        if "vwap" in output.columns:
            output["vwap"] = pd.to_numeric(output["vwap"], errors="coerce")
        else:
            output["vwap"] = output["amount_cny"] / output["volume"].replace(0, pd.NA)

        before = len(output)
        valid_close = output["close"].map(
            lambda value: pd.notna(value) and math.isfinite(float(value)) and value > 0
        )
        output = output[output["date"].notna() & valid_close]
        if len(output) < before:
            warnings.append("INVALID_BAR_ROWS_DROPPED")
        columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount_cny",
            "vwap",
        ]
        if "adj_factor" in output.columns:
            output["adj_factor"] = pd.to_numeric(
                output["adj_factor"], errors="coerce"
            )
            columns.append("adj_factor")
        output = (
            output[columns]
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        return output, warnings

    @staticmethod
    def _has_fresh_price(bars: pd.DataFrame, data_date: date) -> bool:
        if bars.empty or bars.iloc[-1]["date"] != data_date:
            return False
        recent = bars["close"].tail(min(7, len(bars)))
        return bool(
            len(recent)
            and all(math.isfinite(float(value)) and float(value) > 0 for value in recent)
        )

    def _fetch_bars(
        self,
        symbol: str,
        data_date: date,
        *,
        history_start: date | None = None,
        require_end_price: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        start = history_start or (
            data_date - timedelta(days=self.settings.market_history_days)
        )
        frame = self._client().daily(
            ts_code=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=data_date.strftime("%Y%m%d"),
        )
        normalized, warnings = self._normalize_bars(frame)
        if normalized.empty or (
            require_end_price
            and not self._has_fresh_price(normalized, data_date)
        ):
            raise StaleDataError(
                f"Tushare did not return a valid close for {symbol} on {data_date}"
            )
        try:
            factors = self._client().adj_factor(
                ts_code=symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=data_date.strftime("%Y%m%d"),
            )
            if not isinstance(factors, pd.DataFrame) or factors.empty:
                raise DataUnavailableError("No adjustment factors returned")
            normalized_factors = factors.copy()
            normalized_factors["date"] = pd.to_datetime(
                normalized_factors["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            ).dt.date
            normalized_factors["adj_factor"] = pd.to_numeric(
                normalized_factors["adj_factor"], errors="coerce"
            )
            normalized = normalized.merge(
                normalized_factors[["date", "adj_factor"]],
                on="date",
                how="left",
            )
        except Exception:
            warnings.append("ADJUSTMENT_FACTOR_UNAVAILABLE")
        return normalized, warnings

    def sessions_between(self, start: date, end: date) -> tuple[date, ...]:
        """Return open A-share sessions in an inclusive historical range."""

        if end < start:
            raise ValueError("Historical session range end precedes start")
        if self._online_allowed():
            try:
                frame = self._client().trade_cal(
                    exchange="",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    fields="cal_date,is_open",
                )
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    raise DataUnavailableError(
                        "Tushare returned no historical calendar rows"
                    )
                normalized: list[date] = []
                for _, row in frame.iterrows():
                    parsed = pd.to_datetime(
                        str(row.get("cal_date") or ""),
                        format="%Y%m%d",
                        errors="coerce",
                    )
                    if pd.isna(parsed):
                        continue
                    session = parsed.date()
                    is_open = bool(int(row.get("is_open") or 0))
                    self.cache.write_json(
                        "calendar",
                        session.isoformat(),
                        is_open,
                    )
                    if is_open:
                        normalized.append(session)
                sessions = tuple(sorted(set(normalized)))
                if sessions:
                    return sessions
            except DataUnavailableError:
                raise
            except Exception as exc:
                raise DataUnavailableError(
                    f"Tushare historical calendar failed: {exc}"
                ) from exc

        sessions: list[date] = []
        candidate = start
        while candidate <= end:
            if self._is_trading_day(candidate):
                sessions.append(candidate)
            candidate += timedelta(days=1)
        if not sessions:
            raise DataUnavailableError(
                "No cached A-share sessions were available in the range"
            )
        return tuple(sessions)

    def load_history(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Load normalized raw bars and adjustment factors for backtesting."""

        if end < start:
            raise ValueError("Historical price range end precedes start")
        cached_raw = self.cache.read_bars(symbol)
        cached, cached_warnings = (
            self._normalize_bars(cached_raw)
            if cached_raw is not None
            else (pd.DataFrame(), [])
        )
        eligible = (
            cached[
                (cached["date"] >= start)
                & (cached["date"] <= end)
            ].copy()
            if not cached.empty
            else cached
        )
        has_end = self._has_fresh_price(
            cached[cached["date"] <= end].copy()
            if not cached.empty
            else cached,
            end,
        )
        has_start_coverage = (
            not eligible.empty
            and eligible.iloc[0]["date"] <= start + timedelta(days=10)
        )
        if has_end and has_start_coverage:
            return eligible.reset_index(drop=True)
        if not self._online_allowed():
            if not eligible.empty:
                return eligible.reset_index(drop=True)
            raise DataUnavailableError(
                f"No complete cached history for {symbol} from {start} to {end}"
            )

        try:
            fetched, fetch_warnings = self._fetch_bars(
                symbol,
                end,
                history_start=start,
                require_end_price=False,
            )
            merged = pd.concat([cached, fetched], ignore_index=True)
            merged = (
                merged.drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            warnings = list(
                dict.fromkeys([*cached_warnings, *fetch_warnings])
            )
            self.cache.write_bars(symbol, merged)
            self.cache.write_json(
                "bar_quality",
                _safe_symbol(symbol),
                {
                    "warnings": warnings,
                    "retrieved_for": end.isoformat(),
                },
            )
            eligible = merged[
                (merged["date"] >= start)
                & (merged["date"] <= end)
            ].copy()
            if eligible.empty:
                raise StaleDataError(
                    f"No historical prices for {symbol} in the requested range"
                )
            return eligible.reset_index(drop=True)
        except (DataUnavailableError, StaleDataError):
            raise
        except Exception as exc:
            raise DataUnavailableError(
                f"Historical price data failed for {symbol}: {exc}"
            ) from exc

    @staticmethod
    def _with_point_in_time_adjustment(
        bars: pd.DataFrame,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        output = bars.copy()
        if "adj_factor" not in output.columns:
            output["adjusted_close"] = output["close"]
            warnings.append("ADJUSTMENT_FACTOR_UNAVAILABLE")
            return output, list(dict.fromkeys(warnings))
        factors = pd.to_numeric(output["adj_factor"], errors="coerce")
        latest_factor = factors.iloc[-1] if not factors.empty else float("nan")
        valid = factors.map(
            lambda value: pd.notna(value)
            and math.isfinite(float(value))
            and float(value) > 0
        )
        if not valid.all() or not pd.notna(latest_factor) or latest_factor <= 0:
            output["adjusted_close"] = output["close"]
            warnings.append("ADJUSTMENT_FACTOR_INCOMPLETE")
        else:
            output["adjusted_close"] = (
                output["close"] * factors / float(latest_factor)
            )
        return output, list(dict.fromkeys(warnings))

    def _bars(self, symbol: str, data_date: date) -> tuple[pd.DataFrame, list[str]]:
        cached_raw = self.cache.read_bars(symbol)
        cached, cached_warnings = (
            self._normalize_bars(cached_raw)
            if cached_raw is not None
            else (pd.DataFrame(), [])
        )
        quality = self.cache.read_json("bar_quality", _safe_symbol(symbol))
        if isinstance(quality, dict) and isinstance(quality.get("warnings"), list):
            cached_warnings.extend(str(item) for item in quality["warnings"])
        eligible = (
            cached[cached["date"] <= data_date].copy()
            if not cached.empty
            else cached
        )
        if self._has_fresh_price(eligible, data_date):
            return self._with_point_in_time_adjustment(
                eligible,
                list(dict.fromkeys(cached_warnings)),
            )
        if not self._online_allowed():
            raise DataUnavailableError(
                f"No fresh cached price for {symbol} on {data_date}"
            )
        try:
            fetched, fetch_warnings = self._fetch_bars(symbol, data_date)
            merged = pd.concat([cached, fetched], ignore_index=True)
            merged = (
                merged.drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            self.cache.write_bars(symbol, merged)
            self.cache.write_json(
                "bar_quality",
                _safe_symbol(symbol),
                {
                    "warnings": list(dict.fromkeys(fetch_warnings)),
                    "retrieved_for": data_date.isoformat(),
                },
            )
            eligible = merged[merged["date"] <= data_date].copy()
            return self._with_point_in_time_adjustment(
                eligible,
                list(dict.fromkeys(fetch_warnings)),
            )
        except Exception as exc:
            raise DataUnavailableError(f"Price data failed for {symbol}: {exc}") from exc

    @staticmethod
    def _number(value: Any) -> float | None:
        normalized = _plain(value)
        if normalized is None:
            return None
        try:
            number = float(normalized)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _ratio(cls, value: Any) -> float | None:
        number = cls._number(value)
        return number / 100 if number is not None else None

    def _fetch_fundamentals(self, symbol: str, data_date: date) -> tuple[dict[str, Any], list[str]]:
        cache_key = f"{_safe_symbol(symbol)}_{data_date.isoformat()}"
        cached = self.cache.read_json("fundamentals", cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("data"), dict):
            cached_data = cached["data"]
            cached_warnings = [str(item) for item in cached.get("warnings", [])]
            cached_complete = bool(cached.get("complete"))
        elif isinstance(cached, dict):
            cached_data = cached
            cached_warnings = []
            cached_complete = True
        else:
            cached_data = {}
            cached_warnings = []
            cached_complete = False
        if not self._online_allowed():
            if cached_data:
                return cached_data, cached_warnings
            return {}, ["FUNDAMENTAL_CACHE_UNAVAILABLE"]

        payload: dict[str, Any] = {"data_source": "tushare"}
        warnings: list[str] = []
        try:
            daily = self._client().daily_basic(
                ts_code=symbol,
                trade_date=data_date.strftime("%Y%m%d"),
                fields=(
                    "ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,pb,"
                    "ps_ttm,total_mv,circ_mv"
                ),
            )
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                row = daily.iloc[0]
                total_mv = self._number(row.get("total_mv"))
                circ_mv = self._number(row.get("circ_mv"))
                payload.update(
                    {
                        "turnover_ratio": self._ratio(row.get("turnover_rate")),
                        "volume_ratio": self._number(row.get("volume_ratio")),
                        "pe_ttm": self._number(row.get("pe_ttm")),
                        "pb": self._number(row.get("pb")),
                        "ps_ttm": self._number(row.get("ps_ttm")),
                        "total_market_cap_cny": (
                            total_mv * 10_000 if total_mv is not None else None
                        ),
                        "circulating_market_cap_cny": (
                            circ_mv * 10_000 if circ_mv is not None else None
                        ),
                    }
                )
            else:
                warnings.append("DAILY_BASIC_UNAVAILABLE")
        except Exception:
            warnings.append("DAILY_BASIC_PROVIDER_UNAVAILABLE_OR_UNAUTHORIZED")

        try:
            start = data_date - timedelta(days=800)
            financial = self._client().fina_indicator(
                ts_code=symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=data_date.strftime("%Y%m%d"),
            )
            if isinstance(financial, pd.DataFrame) and not financial.empty:
                visible = financial.copy()
                if "ann_date" in visible.columns:
                    visible = visible[
                        visible["ann_date"].astype(str) <= data_date.strftime("%Y%m%d")
                    ].sort_values("ann_date", ascending=False)
                if not visible.empty:
                    row = visible.iloc[0]
                    payload.update(
                        {
                            "roe_ratio": self._ratio(row.get("roe")),
                            "gross_margin_ratio": self._ratio(
                                row.get("grossprofit_margin")
                            ),
                            "debt_to_assets_ratio": self._ratio(
                                row.get("debt_to_assets")
                            ),
                            "net_profit_yoy_ratio": self._ratio(
                                row.get("netprofit_yoy")
                            ),
                            "revenue_yoy_ratio": self._ratio(row.get("or_yoy")),
                        }
                    )
                    payload["report_period"] = _plain(row.get("end_date"))
                    payload["announcement_date"] = _plain(row.get("ann_date"))
                else:
                    warnings.append("POINT_IN_TIME_FINANCIALS_UNAVAILABLE")
            else:
                warnings.append("FINANCIAL_INDICATORS_UNAVAILABLE")
        except Exception:
            warnings.append("FINANCIAL_PROVIDER_UNAVAILABLE_OR_UNAUTHORIZED")

        envelope = {
            "schema_version": 1,
            "data": payload,
            "warnings": warnings,
            "complete": not warnings,
            "retrieved_for": data_date.isoformat(),
        }
        if warnings and cached_data and cached_complete:
            return cached_data, list(
                dict.fromkeys(
                    [*warnings, "FUNDAMENTAL_PROVIDER_DEGRADED_USING_CACHE"]
                )
            )
        self.cache.write_json("fundamentals", cache_key, envelope)
        return payload, warnings

    def _stock_name(self, symbol: str) -> str | None:
        key = _safe_symbol(symbol)
        cached = self.cache.read_json("symbols", key)
        if isinstance(cached, dict) and isinstance(cached.get("name"), str):
            return cached["name"]
        try:
            frame = self._client().stock_basic(ts_code=symbol, fields="ts_code,name")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                name = str(frame.iloc[0]["name"]).strip()
                self.cache.write_json("symbols", key, {"name": name})
                return name
        except Exception:
            return None
        return None

    @staticmethod
    def _published_utc(row: pd.Series) -> datetime | None:
        raw = next(
            (
                row.get(name)
                for name in ("pub_time", "datetime", "published_at", "published_utc")
                if row.get(name) is not None
            ),
            None,
        )
        if raw is None:
            return None
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(CHINA_TZ)
        return timestamp.tz_convert(UTC).to_pydatetime()

    def _normalize_news(
        self,
        frame: pd.DataFrame,
        symbol: str,
        company_name: str | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        code = symbol.split(".", 1)[0]
        items: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            row_symbol = str(row.get("ts_code") or "").upper()
            title = str(row.get("title") or "").strip()
            description = str(row.get("content") or row.get("summary") or "").strip()
            searchable = f"{title}\n{description}"
            if row_symbol and row_symbol != symbol:
                continue
            if not row_symbol and code not in searchable and not (
                company_name and company_name in searchable
            ):
                continue
            published = self._published_utc(row)
            if published is None or not title:
                continue
            source = str(row.get("src") or row.get("source") or "tushare")
            digest = hashlib.sha256(
                f"{published.isoformat()}|{source}|{title}".encode("utf-8")
            ).hexdigest()[:24]
            items.append(
                {
                    "id": digest,
                    "title": title[:500],
                    "description": description[:2000],
                    "published_utc": published.isoformat(),
                    "api_source": source,
                }
            )
        return items

    def _news(
        self,
        symbol: str,
        as_of: datetime,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        cache_key = _safe_symbol(symbol)
        cached = self.cache.read_json("news", cache_key)
        cached_items = cached if isinstance(cached, list) else []
        warnings: list[str] = []
        combined = list(cached_items)

        if self._online_allowed():
            try:
                local_as_of = as_of.astimezone(CHINA_TZ)
                start = local_as_of - timedelta(days=self.settings.news_lookback_days)
                frame = self._client().major_news(
                    src="",
                    start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=local_as_of.strftime("%Y-%m-%d %H:%M:%S"),
                )
                direct = self._normalize_news(
                    frame,
                    symbol,
                    self._stock_name(symbol),
                )
                combined.extend(direct)
            except Exception:
                warnings.append("NEWS_PROVIDER_UNAVAILABLE_OR_UNAUTHORIZED")
        elif not cached_items:
            warnings.append("NEWS_CACHE_UNAVAILABLE")

        cutoff = as_of.astimezone(UTC)
        lower_bound = cutoff - timedelta(days=self.settings.news_lookback_days)
        unique: dict[str, dict[str, Any]] = {}
        for item in combined:
            try:
                published = datetime.fromisoformat(str(item["published_utc"]))
                if published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)
            except (KeyError, TypeError, ValueError):
                continue
            if lower_bound <= published <= cutoff:
                unique[str(item.get("id") or published.isoformat())] = item
        ordered = sorted(
            unique.values(),
            key=lambda item: str(item.get("published_utc") or ""),
            reverse=True,
        )
        if self._online_allowed() and not warnings:
            self.cache.write_json("news", cache_key, ordered)
        return ordered[: self.settings.news_top_k], warnings

    def load_symbol(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        bars, bar_warnings = self._bars(symbol, data_date)
        bars = bars[bars["date"] <= data_date].copy()
        if not self._has_fresh_price(bars, data_date):
            raise StaleDataError(f"No exact close for {symbol} on {data_date}")

        warnings = list(bar_warnings)
        latest_index = bars.index[-1]
        latest_open = bars.loc[latest_index, "open"]
        if pd.isna(latest_open) or not math.isfinite(float(latest_open)) or latest_open <= 0:
            bars.loc[latest_index, "open"] = bars.loc[latest_index, "close"]
            warnings.append("INVALID_LATEST_OPEN_FALLBACK_TO_CLOSE")
        if len(bars) < 20:
            warnings.append("SHORT_PRICE_HISTORY")

        fundamentals, fundamental_warnings = self._fetch_fundamentals(symbol, data_date)
        news, news_warnings = self._news(symbol, as_of)
        warnings.extend(fundamental_warnings)
        warnings.extend(news_warnings)
        reference_price = Decimal(str(bars.iloc[-1]["close"]))
        return SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=reference_price,
            bars=bars,
            news=tuple(news),
            fundamentals=fundamentals,
            retrieved_at=datetime.now(tz=CHINA_TZ),
            data_quality_warnings=tuple(dict.fromkeys(warnings)),
        )

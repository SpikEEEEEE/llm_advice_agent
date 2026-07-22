from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml
from dotenv import dotenv_values


SYMBOL_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
PLACEHOLDER_SECRETS = {"replace-me", "changeme", "your-key", "..."}


def _secret_value(*names: str) -> str | None:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value and value.lower() not in PLACEHOLDER_SECRETS:
            return value
    return None


def has_configured_secret(name: str) -> bool:
    return _secret_value(name) is not None


@dataclass(frozen=True)
class Settings:
    """Validated application settings with no source-tree assumptions."""

    project_root: Path
    app_env: str
    backend_api_key: str | None
    database_path: Path
    universe_path: Path
    cache_path: Path
    execution_mode: str
    decision_workers: int
    decision_queue_capacity: int
    decision_task_timeout_seconds: int
    max_as_of_skew_minutes: int
    data_mode: str
    tushare_token: str | None
    tushare_timeout_seconds: int
    market_history_days: int
    news_lookback_days: int
    news_top_k: int
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int
    llm_max_retries: int
    min_cash_ratio: Decimal
    max_position_ratio: Decimal
    max_positions: int
    buy_fee_buffer_ratio: Decimal
    market_close_hour: int
    market_close_minute: int

    def __post_init__(self) -> None:
        if self.app_env.lower() == "production" and not self.backend_api_key:
            raise ValueError("BACKEND_API_KEY is required when APP_ENV=production")
        if self.execution_mode not in {"process", "thread", "inline"}:
            raise ValueError(
                "DECISION_EXECUTION_MODE must be 'process', 'thread', or 'inline'"
            )
        if self.data_mode not in {"auto", "offline_only"}:
            raise ValueError("DATA_MODE must be 'auto' or 'offline_only'")
        if self.decision_workers < 1:
            raise ValueError("DECISION_WORKERS must be at least 1")
        if self.decision_queue_capacity < 0:
            raise ValueError("DECISION_QUEUE_CAPACITY cannot be negative")
        if self.decision_task_timeout_seconds < 30:
            raise ValueError("DECISION_TASK_TIMEOUT_SECONDS must be at least 30")
        if self.max_as_of_skew_minutes < 1:
            raise ValueError("MAX_AS_OF_SKEW_MINUTES must be at least 1")
        if self.tushare_timeout_seconds < 1:
            raise ValueError("TUSHARE_TIMEOUT_SECONDS must be positive")
        if self.market_history_days < 30:
            raise ValueError("MARKET_HISTORY_DAYS must be at least 30")
        if self.news_lookback_days < 1 or self.news_top_k < 1:
            raise ValueError("NEWS_LOOKBACK_DAYS and NEWS_TOP_K must be positive")
        if not self.llm_base_url.strip() or not self.llm_model.strip():
            raise ValueError("LLM_BASE_URL and LLM_MODEL cannot be empty")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm_max_tokens < 1:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        if self.llm_timeout_seconds < 1:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative")
        if not Decimal("0") <= self.min_cash_ratio <= Decimal("1"):
            raise ValueError("MIN_CASH_RATIO must be between 0 and 1")
        if not Decimal("0") <= self.max_position_ratio <= Decimal("1"):
            raise ValueError("MAX_POSITION_RATIO must be between 0 and 1")
        if not Decimal("0") <= self.buy_fee_buffer_ratio <= Decimal("0.1"):
            raise ValueError("BUY_FEE_BUFFER_RATIO must be between 0 and 0.1")
        if self.max_positions < 1:
            raise ValueError("MAX_POSITIONS must be at least 1")
        if not 0 <= self.market_close_hour <= 23:
            raise ValueError("MARKET_CLOSE_HOUR must be between 0 and 23")
        if not 0 <= self.market_close_minute <= 59:
            raise ValueError("MARKET_CLOSE_MINUTE must be between 0 and 59")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        source_candidate = Path(__file__).resolve().parents[2]
        root = (
            project_root
            or (
                source_candidate
                if (source_candidate / "pyproject.toml").exists()
                else Path.cwd()
            )
        ).resolve()
        file_environment = {
            key: value
            for key, value in dotenv_values(root / ".env").items()
            if value is not None
        }
        environment = {**file_environment, **os.environ}

        def env(name: str, default: str = "") -> str:
            return str(environment.get(name, default))

        def configured_secret(*names: str) -> str | None:
            for name in names:
                value = env(name).strip()
                if value and value.lower() not in PLACEHOLDER_SECRETS:
                    return value
            return None

        def resolved_path(name: str, default: str) -> Path:
            value = Path(env(name, default))
            return (value if value.is_absolute() else root / value).resolve()

        configured_universe = env("UNIVERSE_PATH").strip()
        source_universe = root / "config" / "universe.yaml"
        packaged_universe = Path(__file__).resolve().parents[1] / "resources" / "universe.yaml"
        if configured_universe:
            universe_path = Path(configured_universe)
            if not universe_path.is_absolute():
                universe_path = root / universe_path
        elif source_universe.exists():
            universe_path = source_universe
        else:
            universe_path = packaged_universe

        app_env = env("APP_ENV", "development").strip().lower()
        backend_api_key = configured_secret("BACKEND_API_KEY")
        if app_env == "production" and backend_api_key is None:
            raise ValueError("BACKEND_API_KEY is required when APP_ENV=production")

        return cls(
            project_root=root,
            app_env=app_env,
            backend_api_key=backend_api_key,
            database_path=resolved_path("DATABASE_PATH", "./data/ashare_advisor.db"),
            universe_path=universe_path.resolve(),
            cache_path=resolved_path("CACHE_PATH", "./data/cache"),
            execution_mode=env("DECISION_EXECUTION_MODE", "process")
            .strip()
            .lower(),
            decision_workers=int(env("DECISION_WORKERS", "1")),
            decision_queue_capacity=int(env("DECISION_QUEUE_CAPACITY", "10")),
            decision_task_timeout_seconds=int(
                env("DECISION_TASK_TIMEOUT_SECONDS", "600")
            ),
            max_as_of_skew_minutes=int(env("MAX_AS_OF_SKEW_MINUTES", "10")),
            data_mode=env("DATA_MODE", "auto").strip().lower(),
            tushare_token=configured_secret("TUSHARE_TOKEN"),
            tushare_timeout_seconds=int(env("TUSHARE_TIMEOUT_SECONDS", "20")),
            market_history_days=int(env("MARKET_HISTORY_DAYS", "400")),
            news_lookback_days=int(env("NEWS_LOOKBACK_DAYS", "3")),
            news_top_k=int(env("NEWS_TOP_K", "5")),
            llm_api_key=configured_secret("LLM_API_KEY", "OPENAI_API_KEY"),
            llm_base_url=env("LLM_BASE_URL", "https://api.openai.com/v1").rstrip(
                "/"
            ),
            llm_model=env("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(env("LLM_TEMPERATURE", "0")),
            llm_max_tokens=int(env("LLM_MAX_TOKENS", "4000")),
            llm_timeout_seconds=int(env("LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=int(env("LLM_MAX_RETRIES", "1")),
            min_cash_ratio=Decimal(env("MIN_CASH_RATIO", "0.05")),
            max_position_ratio=Decimal(env("MAX_POSITION_RATIO", "0.30")),
            max_positions=int(env("MAX_POSITIONS", "10")),
            buy_fee_buffer_ratio=Decimal(env("BUY_FEE_BUFFER_RATIO", "0.001")),
            market_close_hour=int(env("MARKET_CLOSE_HOUR", "15")),
            market_close_minute=int(env("MARKET_CLOSE_MINUTE", "15")),
        )

    def load_universe(self) -> tuple[str, list[str]]:
        with self.universe_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        version = str(payload.get("version") or "unversioned")
        symbols = [str(item).strip().upper() for item in payload.get("symbols", [])]
        invalid = [symbol for symbol in symbols if not SYMBOL_PATTERN.fullmatch(symbol)]
        if invalid:
            raise ValueError(f"Invalid symbols in universe: {invalid}")
        if len(symbols) != len(set(symbols)):
            raise ValueError("Universe contains duplicate symbols")
        if not symbols:
            raise ValueError("Universe cannot be empty")
        return version, symbols

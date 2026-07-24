from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from app.adapters.decision_engine_factory import build_decision_engine
from app.adapters.tushare import TushareMarketDataProvider
from app.core.config import Settings, SYMBOL_PATTERN
from app.domain.risk import AShareRiskPolicy

from .cache import BacktestDecisionCache
from .data import TushareHistoricalDataFeed
from .engine import PortfolioBacktester
from .models import BacktestConfig


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD"
        ) from exc


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Expected a decimal number"
        ) from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("Number must be finite")
    return parsed


def _symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(
        dict.fromkeys(
            item.strip().upper()
            for item in value.split(",")
            if item.strip()
        )
    )
    invalid = [
        symbol
        for symbol in symbols
        if not SYMBOL_PATTERN.fullmatch(symbol)
    ]
    if invalid or not symbols:
        raise argparse.ArgumentTypeError(
            f"Invalid A-share symbols: {invalid or symbols}"
        )
    return symbols


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the A-share portfolio decision engine with "
            "point-in-time data and next-session-open execution"
        )
    )
    parser.add_argument("--start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument(
        "--initial-cash",
        type=_decimal,
        default=Decimal("1000000"),
    )
    parser.add_argument(
        "--rebalance",
        choices=("daily", "weekly", "monthly"),
        default="monthly",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=24,
        help="Hard guard against unexpectedly expensive LLM backtests",
    )
    parser.add_argument(
        "--engine",
        choices=("single_llm", "portfolio_multi_agent"),
        default=None,
        help="Defaults to DECISION_ENGINE from the environment",
    )
    parser.add_argument(
        "--symbols",
        type=_symbols,
        default=None,
        help="Comma-separated symbols; defaults to the configured universe",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/backtests/<run_id>",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use only already cached Tushare inputs",
    )
    parser.add_argument(
        "--no-decision-cache",
        action="store_true",
    )
    parser.add_argument(
        "--no-initial-rebalance",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-sale-proceeds",
        action="store_true",
        help="Allow same-open sale proceeds to fund buys",
    )
    parser.add_argument(
        "--commission-rate",
        type=_decimal,
        default=Decimal("0.0003"),
    )
    parser.add_argument(
        "--minimum-commission",
        type=_decimal,
        default=Decimal("5"),
    )
    parser.add_argument(
        "--slippage-bps",
        type=_decimal,
        default=Decimal("5"),
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else None
    )
    settings = Settings.from_env(project_root)
    effective_engine = args.engine or settings.decision_engine_mode
    settings = replace(
        settings,
        decision_engine_mode=effective_engine,
        data_mode=(
            "offline_only" if args.offline_only else settings.data_mode
        ),
    )
    if settings.data_mode == "auto" and not settings.tushare_token:
        parser.error(
            "TUSHARE_TOKEN is required unless --offline-only has complete cache"
        )
    if not settings.llm_api_key:
        parser.error(
            "LLM_API_KEY or OPENAI_API_KEY is required for historical decisions"
        )

    if args.symbols is None:
        universe_version, loaded_symbols = settings.load_universe()
        universe = tuple(loaded_symbols)
    else:
        universe_version = "cli_symbols_v1"
        universe = args.symbols

    config = BacktestConfig(
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        rebalance_frequency=args.rebalance,
        initial_rebalance=not args.no_initial_rebalance,
        max_decisions=args.max_decisions,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        slippage_bps=args.slippage_bps,
        reuse_sale_proceeds=args.reuse_sale_proceeds,
    )
    provider = TushareMarketDataProvider(settings)
    feed = TushareHistoricalDataFeed(provider)
    decision_cache = (
        None
        if args.no_decision_cache
        else BacktestDecisionCache(
            settings.cache_path / "backtest_decisions",
            settings,
        )
    )
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=feed,
        decision_engine=build_decision_engine(settings),
        risk_policy=AShareRiskPolicy(settings),
        decision_cache=decision_cache,
    )
    result = backtester.run(
        config=config,
        universe=universe,
        universe_version=universe_version,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else settings.project_root / "data" / "backtests" / result.run_id
    )
    result.write(output_dir)
    print(
        json.dumps(
            {
                **result.summary(),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

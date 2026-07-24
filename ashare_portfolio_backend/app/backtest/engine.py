from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.core.config import Settings
from app.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    RawDecisionBundle,
)
from app.domain.risk import AShareRiskPolicy
from app.ports.decision_engine import DecisionEngine

from .cache import BacktestDecisionCache, decision_quality
from .data import HistoricalDataFeed
from .models import (
    BacktestConfig,
    BacktestHolding,
    BacktestResult,
    BacktestTrade,
    finite_metric,
)


CHINA_TZ = ZoneInfo("Asia/Shanghai")
ONE_HUNDRED = Decimal("100")
TEN_THOUSAND = Decimal("10000")
SAFE_EXCEPTION_TYPES = {
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "TimeoutError",
    "ConnectionError",
    "DataUnavailableError",
    "StaleDataError",
    "ProviderConfigurationError",
    "PortfolioAgentGraphError",
    "PortfolioAgentOutputError",
    "DecisionOutputError",
}


def _safe_exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    return name if name in SAFE_EXCEPTION_TYPES else "Exception"


def _non_negative_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _analysis_coverage(meta: dict[str, Any], quality: str) -> float:
    raw = meta.get("analysis_coverage")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        numeric = float(raw)
        if math.isfinite(numeric):
            return max(0.0, min(1.0, numeric))
    return 0.0 if quality == "failed" else 1.0


def _validated_outputs(meta: dict[str, Any]) -> int:
    explicit = meta.get("validated_outputs")
    if explicit is None:
        explicit = meta.get("provider_successes")
    if explicit is not None:
        return _non_negative_int(explicit)
    trace = meta.get("agent_trace")
    if not isinstance(trace, list):
        trace = meta.get("trace")
    return len(trace) if isinstance(trace, list) else 0


def _bundle_audit(bundle: RawDecisionBundle) -> dict[str, Any]:
    meta = bundle.meta
    quality = decision_quality(bundle)
    raw_stage_health = meta.get("stage_health")
    stage_health = (
        raw_stage_health if isinstance(raw_stage_health, dict) else {}
    )
    fresh_calls = _non_negative_int(meta.get("calls"))
    fresh_attempts = _non_negative_int(
        meta.get("provider_attempts"),
        fresh_calls,
    )
    cached_original_calls = _non_negative_int(
        meta.get("cached_original_calls")
    )
    cached_original_attempts = _non_negative_int(
        meta.get("cached_original_provider_attempts"),
        cached_original_calls,
    )
    return {
        "decision_quality": quality,
        "analysis_coverage": _analysis_coverage(meta, quality),
        "stage_health": stage_health,
        "fresh_provider_calls": fresh_calls,
        "provider_attempts": fresh_attempts,
        "cached_original_calls": cached_original_calls,
        "cached_original_provider_attempts": (
            cached_original_attempts
        ),
        "validated_outputs": _validated_outputs(meta),
        "cached_original_validated_outputs": _non_negative_int(
            meta.get("cached_original_validated_outputs")
        ),
        "output_repair_attempts": _non_negative_int(
            meta.get("output_repair_attempts")
        ),
        "cached_original_output_repair_attempts": _non_negative_int(
            meta.get("cached_original_output_repair_attempts")
        ),
    }


@dataclass
class BacktestAccount:
    cash: Decimal
    holdings: dict[str, BacktestHolding] = field(default_factory=dict)


class PortfolioBacktester:
    """Event-driven close-decision/next-open-execution portfolio backtester."""

    def __init__(
        self,
        *,
        settings: Settings,
        data_feed: HistoricalDataFeed,
        decision_engine: DecisionEngine,
        risk_policy: AShareRiskPolicy | None = None,
        decision_cache: BacktestDecisionCache | None = None,
    ) -> None:
        self.settings = settings
        self.data_feed = data_feed
        self.decision_engine = decision_engine
        self.risk_policy = risk_policy or AShareRiskPolicy(settings)
        self.decision_cache = decision_cache

    @staticmethod
    def decision_sessions(
        sessions: tuple[date, ...],
        frequency: str,
        initial_rebalance: bool,
    ) -> set[date]:
        if len(sessions) < 2:
            return set()
        selected: set[date] = set()
        if initial_rebalance:
            selected.add(sessions[0])
        if frequency == "daily":
            selected.update(sessions[:-1])
        else:
            grouped: dict[Any, date] = {}
            for session in sessions:
                key = (
                    (session.year, session.month)
                    if frequency == "monthly"
                    else session.isocalendar()[:2]
                )
                grouped[key] = session
            selected.update(
                session
                for session in grouped.values()
                if session != sessions[-1]
            )
        selected.discard(sessions[-1])
        return selected

    def _portfolio(
        self,
        account: BacktestAccount,
        data_date: date,
    ) -> PortfolioSnapshot:
        positions = tuple(
            Position(
                symbol=holding.symbol,
                shares=holding.shares,
                available_shares=holding.shares,
                average_cost=holding.average_cost,
                holding_days=max(0, (data_date - holding.acquired_on).days),
            )
            for holding in sorted(
                account.holdings.values(),
                key=lambda item: item.symbol,
            )
            if holding.shares > 0
        )
        return PortfolioSnapshot(
            portfolio_id="backtest",
            version=1,
            name="Historical backtest account",
            cash=account.cash,
            positions=positions,
        )

    def _decision_input(
        self,
        *,
        run_id: str,
        account: BacktestAccount,
        universe: tuple[str, ...],
        universe_version: str,
        data_date: date,
        execution_session: date,
    ) -> DecisionInput:
        as_of = datetime(
            data_date.year,
            data_date.month,
            data_date.day,
            16,
            0,
            tzinfo=CHINA_TZ,
        )
        symbols = tuple(
            dict.fromkeys([*universe, *account.holdings.keys()])
        )
        market = {}
        unavailable: dict[str, str] = {}
        for symbol in symbols:
            try:
                market[symbol] = self.data_feed.snapshot(
                    symbol,
                    data_date,
                    as_of,
                )
            except Exception as exc:
                unavailable[symbol] = (
                    f"{_safe_exception_type(exc)}: market snapshot unavailable"
                )
        return DecisionInput(
            run_id=run_id,
            portfolio=self._portfolio(account, data_date),
            mode="rebalance",
            as_of=as_of,
            data_date=data_date,
            valid_for_session=execution_session,
            universe_version=universe_version,
            symbols=symbols,
            market=market,
            unavailable_symbols=unavailable,
        )

    def _decide(
        self,
        decision_input: DecisionInput,
    ) -> tuple[RawDecisionBundle, bool]:
        cache_key: str | None = None
        if self.decision_cache is not None:
            cache_key, cached = self.decision_cache.get_or_none(
                decision_input
            )
            if cached is not None:
                return cached, True
        bundle = self.decision_engine.decide(decision_input)
        if self.decision_cache is not None and cache_key is not None:
            self.decision_cache.write(cache_key, bundle)
        return bundle, False

    @staticmethod
    def _commission(
        notional: Decimal,
        config: BacktestConfig,
    ) -> Decimal:
        if notional <= 0:
            return Decimal("0")
        return max(
            config.minimum_commission,
            notional * config.commission_rate,
        )

    @staticmethod
    def _execution_price(
        reference: Decimal,
        side: str,
        config: BacktestConfig,
    ) -> Decimal:
        slippage = config.slippage_bps / TEN_THOUSAND
        multiplier = (
            Decimal("1") + slippage
            if side == "buy"
            else Decimal("1") - slippage
        )
        return reference * multiplier

    def _sell(
        self,
        *,
        account: BacktestAccount,
        session: date,
        decision_session: date,
        symbol: str,
        requested: int,
        reference_open: Decimal,
        config: BacktestConfig,
    ) -> BacktestTrade | None:
        holding = account.holdings.get(symbol)
        if holding is None or holding.shares <= 0 or requested <= 0:
            return None
        executed = min(requested, holding.shares)
        execution_price = self._execution_price(
            reference_open,
            "sell",
            config,
        )
        notional = execution_price * Decimal(executed)
        commission = self._commission(notional, config)
        stamp_duty = notional * config.stamp_duty(session)
        net_proceeds = notional - commission - stamp_duty
        if net_proceeds <= 0:
            return None
        account.cash += net_proceeds
        holding.shares -= executed
        if holding.shares == 0:
            del account.holdings[symbol]
        return BacktestTrade(
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="sell",
            requested_shares=requested,
            executed_shares=executed,
            reference_open=reference_open,
            execution_price=execution_price,
            notional=notional,
            commission=commission,
            stamp_duty=stamp_duty,
            cash_after=account.cash,
        )

    def _affordable_buy_shares(
        self,
        *,
        requested: int,
        execution_price: Decimal,
        budget: Decimal,
        config: BacktestConfig,
    ) -> int:
        lots = max(0, requested // 100)
        while lots > 0:
            shares = lots * 100
            notional = execution_price * Decimal(shares)
            total = notional + self._commission(notional, config)
            if total <= budget:
                return shares
            lots -= 1
        return 0

    def _buy(
        self,
        *,
        account: BacktestAccount,
        session: date,
        decision_session: date,
        symbol: str,
        requested: int,
        reference_open: Decimal,
        budget: Decimal,
        config: BacktestConfig,
    ) -> tuple[BacktestTrade | None, Decimal]:
        if requested < 100 or budget <= 0:
            return None, budget
        execution_price = self._execution_price(
            reference_open,
            "buy",
            config,
        )
        executed = self._affordable_buy_shares(
            requested=requested,
            execution_price=execution_price,
            budget=min(budget, account.cash),
            config=config,
        )
        if executed <= 0:
            return None, budget
        notional = execution_price * Decimal(executed)
        commission = self._commission(notional, config)
        total_cost = notional + commission
        account.cash -= total_cost
        existing = account.holdings.get(symbol)
        if existing is None:
            account.holdings[symbol] = BacktestHolding(
                symbol=symbol,
                shares=executed,
                average_cost=total_cost / Decimal(executed),
                acquired_on=session,
            )
        else:
            previous_cost = existing.average_cost * Decimal(
                existing.shares
            )
            existing.shares += executed
            existing.average_cost = (
                previous_cost + total_cost
            ) / Decimal(existing.shares)
        trade = BacktestTrade(
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="buy",
            requested_shares=requested,
            executed_shares=executed,
            reference_open=reference_open,
            execution_price=execution_price,
            notional=notional,
            commission=commission,
            stamp_duty=Decimal("0"),
            cash_after=account.cash,
            note=(
                "cash_or_board_lot_cap"
                if executed < requested
                else ""
            ),
        )
        return trade, budget - total_cost

    def _open_equity(
        self,
        account: BacktestAccount,
        session: date,
    ) -> Decimal:
        value = account.cash
        for symbol, holding in account.holdings.items():
            price = self.data_feed.price(symbol, session, "open")
            if price is not None:
                value += price * Decimal(holding.shares)
        return value

    def _execute(
        self,
        *,
        account: BacktestAccount,
        plan: dict[str, Any],
        session: date,
        decision_session: date,
        config: BacktestConfig,
    ) -> list[BacktestTrade]:
        rows = plan.get("decisions")
        if not isinstance(rows, list):
            return []
        current_targets = {
            str(row.get("symbol")): max(
                0,
                int(row.get("target_shares") or 0),
            )
            for row in rows
            if isinstance(row, dict) and row.get("symbol")
        }
        opening_cash = account.cash
        opening_equity = self._open_equity(account, session)
        cash_floor = opening_equity * self.settings.min_cash_ratio
        trades: list[BacktestTrade] = []

        for symbol in sorted(account.holdings):
            holding = account.holdings.get(symbol)
            if holding is None:
                continue
            target = current_targets.get(symbol, holding.shares)
            requested = max(0, holding.shares - target)
            reference = self.data_feed.price(symbol, session, "open")
            if reference is None:
                continue
            trade = self._sell(
                account=account,
                session=session,
                decision_session=decision_session,
                symbol=symbol,
                requested=requested,
                reference_open=reference,
                config=config,
            )
            if trade is not None:
                trades.append(trade)

        budget_cash = (
            account.cash
            if config.reuse_sale_proceeds
            else opening_cash
        )
        buy_budget = max(Decimal("0"), budget_cash - cash_floor)
        buy_rows = sorted(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("symbol") in current_targets
            ),
            key=lambda row: (
                -float(row.get("confidence") or 0),
                str(row.get("symbol") or ""),
            ),
        )
        for row in buy_rows:
            symbol = str(row["symbol"])
            current = (
                account.holdings[symbol].shares
                if symbol in account.holdings
                else 0
            )
            requested = max(0, current_targets[symbol] - current)
            reference = self.data_feed.price(symbol, session, "open")
            if reference is None:
                continue
            trade, buy_budget = self._buy(
                account=account,
                session=session,
                decision_session=decision_session,
                symbol=symbol,
                requested=requested,
                reference_open=reference,
                budget=buy_budget,
                config=config,
            )
            if trade is not None:
                trades.append(trade)
        return trades

    def _mark_to_market(
        self,
        account: BacktestAccount,
        session: date,
        last_prices: dict[str, Decimal],
    ) -> tuple[Decimal, Decimal, list[str]]:
        position_value = Decimal("0")
        warnings: list[str] = []
        for symbol, holding in account.holdings.items():
            exact = self.data_feed.price(symbol, session, "close")
            if exact is not None:
                last_prices[symbol] = exact
            price = exact or last_prices.get(symbol)
            if price is None:
                warnings.append(
                    f"{session}: no mark price for held symbol {symbol}"
                )
                continue
            if exact is None:
                warnings.append(
                    f"{session}: used last price for suspended symbol {symbol}"
                )
            position_value += price * Decimal(holding.shares)
        return account.cash + position_value, position_value, warnings

    def _benchmark_return(
        self,
        universe: tuple[str, ...],
        sessions: tuple[date, ...],
        entry_session: date | None,
    ) -> tuple[float | None, int]:
        if len(sessions) < 2 or entry_session is None:
            return None, 0
        exit_session = sessions[-1]
        returns: list[float] = []
        for symbol in universe:
            start_price = self.data_feed.price(
                symbol,
                entry_session,
                "open",
            )
            end_price = self.data_feed.price(
                symbol,
                exit_session,
                "close",
            )
            if end_price is None:
                for candidate in reversed(sessions):
                    if candidate < entry_session:
                        break
                    end_price = self.data_feed.price(
                        symbol,
                        candidate,
                        "close",
                    )
                    if end_price is not None:
                        break
            if (
                start_price is None
                or end_price is None
                or start_price <= 0
            ):
                continue
            returns.append(float(end_price / start_price - Decimal("1")))
        if not returns:
            return None, 0
        return sum(returns) / len(returns), len(returns)

    @staticmethod
    def _metrics(
        *,
        config: BacktestConfig,
        equity_curve: list[dict[str, Any]],
        trades: list[BacktestTrade],
        decisions: list[dict[str, Any]],
        benchmark_return: float | None,
        benchmark_symbols: int,
    ) -> dict[str, Any]:
        equities = [float(item["equity"]) for item in equity_curve]
        final_equity = equities[-1]
        initial = float(config.initial_cash)
        total_return = final_equity / initial - 1
        daily_returns = [
            equities[index] / equities[index - 1] - 1
            for index in range(1, len(equities))
            if equities[index - 1] > 0
        ]
        periods = max(1, len(daily_returns))
        try:
            annualized_return = (
                (final_equity / initial) ** (252 / periods) - 1
                if final_equity > 0 and initial > 0
                else None
            )
        except (OverflowError, ValueError):
            annualized_return = None
        if len(daily_returns) >= 2:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum(
                (value - mean_return) ** 2
                for value in daily_returns
            ) / (len(daily_returns) - 1)
            daily_volatility = math.sqrt(max(0.0, variance))
            annualized_volatility = daily_volatility * math.sqrt(252)
            sharpe = (
                mean_return / daily_volatility * math.sqrt(252)
                if daily_volatility > 0
                else None
            )
        else:
            annualized_volatility = None
            sharpe = None

        peak = equities[0]
        max_drawdown = 0.0
        for value in equities:
            peak = max(peak, value)
            drawdown = value / peak - 1 if peak > 0 else 0.0
            max_drawdown = min(max_drawdown, drawdown)
        total_notional = sum(float(trade.notional) for trade in trades)
        total_fees = sum(
            float(trade.commission + trade.stamp_duty)
            for trade in trades
        )
        average_equity = sum(equities) / len(equities)
        fresh_provider_calls = sum(
            _non_negative_int(item.get("fresh_provider_calls"))
            for item in decisions
        )
        provider_attempts = sum(
            _non_negative_int(item.get("provider_attempts"))
            for item in decisions
        )
        cached_original_calls = sum(
            _non_negative_int(item.get("cached_original_calls"))
            for item in decisions
        )
        cached_original_provider_attempts = sum(
            _non_negative_int(
                item.get("cached_original_provider_attempts")
            )
            for item in decisions
        )
        validated_outputs = sum(
            _non_negative_int(item.get("validated_outputs"))
            for item in decisions
        )
        cached_original_validated_outputs = sum(
            _non_negative_int(
                item.get("cached_original_validated_outputs")
            )
            for item in decisions
        )
        output_repair_attempts = sum(
            _non_negative_int(item.get("output_repair_attempts"))
            for item in decisions
        )
        cached_original_output_repair_attempts = sum(
            _non_negative_int(
                item.get("cached_original_output_repair_attempts")
            )
            for item in decisions
        )
        cache_hits = sum(bool(item.get("cache_hit")) for item in decisions)
        decision_count = len(decisions)
        quality_counts = {
            quality: sum(
                item.get("decision_quality") == quality
                for item in decisions
            )
            for quality in ("healthy", "degraded", "failed")
        }
        quality_rates = {
            quality: (
                count / decision_count if decision_count else None
            )
            for quality, count in quality_counts.items()
        }
        result_quality_status = (
            "not_evaluated"
            if not decisions
            else (
                "valid"
                if quality_counts["healthy"] == decision_count
                else "invalid"
            )
        )
        return {
            "initial_cash": initial,
            "final_equity": final_equity,
            "total_return": finite_metric(total_return),
            "annualized_return": finite_metric(annualized_return),
            "annualized_volatility": finite_metric(
                annualized_volatility
            ),
            "sharpe_ratio_zero_rate": finite_metric(sharpe),
            "max_drawdown": finite_metric(max_drawdown),
            "trade_count": len(trades),
            "buy_count": sum(trade.side == "buy" for trade in trades),
            "sell_count": sum(trade.side == "sell" for trade in trades),
            "total_trade_notional": total_notional,
            "total_fees": total_fees,
            "turnover_on_average_equity": (
                total_notional / average_equity
                if average_equity > 0
                else None
            ),
            "decision_count": decision_count,
            "healthy_decision_count": quality_counts["healthy"],
            "degraded_decision_count": quality_counts["degraded"],
            "failed_decision_count": quality_counts["failed"],
            "healthy_decision_rate": quality_rates["healthy"],
            "degraded_decision_rate": quality_rates["degraded"],
            "failed_decision_rate": quality_rates["failed"],
            "result_quality_status": result_quality_status,
            # Backward-compatible name: calls are fresh provider attempts in
            # the current run, never attempts replayed from a cache entry.
            "llm_provider_calls": fresh_provider_calls,
            "llm_fresh_provider_calls": fresh_provider_calls,
            "llm_cached_original_calls": cached_original_calls,
            "llm_provider_attempts": provider_attempts,
            "llm_cached_original_provider_attempts": (
                cached_original_provider_attempts
            ),
            "llm_provider_attempts_represented_total": (
                provider_attempts + cached_original_provider_attempts
            ),
            "llm_validated_outputs": validated_outputs,
            "llm_cached_original_validated_outputs": (
                cached_original_validated_outputs
            ),
            "llm_output_repair_attempts": output_repair_attempts,
            "llm_cached_original_output_repair_attempts": (
                cached_original_output_repair_attempts
            ),
            "decision_cache_hits": cache_hits,
            "equal_weight_universe_return": finite_metric(
                benchmark_return
            ),
            "benchmark_symbol_count": benchmark_symbols,
            "excess_return_vs_equal_weight": (
                finite_metric(total_return - benchmark_return)
                if benchmark_return is not None
                else None
            ),
        }

    def run(
        self,
        *,
        config: BacktestConfig,
        universe: tuple[str, ...],
        universe_version: str,
    ) -> BacktestResult:
        if not universe:
            raise ValueError("Backtest universe cannot be empty")
        sessions = self.data_feed.prepare(
            universe,
            config.start,
            config.end,
        )
        decision_sessions = self.decision_sessions(
            sessions,
            config.rebalance_frequency,
            config.initial_rebalance,
        )
        if len(decision_sessions) > config.max_decisions:
            raise ValueError(
                "Backtest would create "
                f"{len(decision_sessions)} decisions, exceeding the configured "
                f"maximum of {config.max_decisions}"
            )
        next_session = {
            sessions[index]: sessions[index + 1]
            for index in range(len(sessions) - 1)
        }
        account = BacktestAccount(cash=config.initial_cash)
        pending: dict[date, tuple[date, dict[str, Any]]] = {}
        trades: list[BacktestTrade] = []
        decisions: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        warnings: list[str] = [
            (
                "The configured fixed universe is reused historically; "
                "results may contain survivorship and selection bias"
            ),
            (
                "Corporate actions are represented by a start-normalized "
                "adjustment-factor total-return price proxy"
            ),
            (
                "Limit-up/limit-down queueing, intraday liquidity, market "
                "impact, dividends in cash, and delistings are not modeled"
            ),
        ]
        last_prices: dict[str, Decimal] = {}
        run_id = (
            f"bt_{config.start:%Y%m%d}_{config.end:%Y%m%d}_"
            f"{uuid4().hex[:8]}"
        )

        for session in sessions:
            scheduled = pending.pop(session, None)
            if scheduled is not None:
                decision_session, plan = scheduled
                trades.extend(
                    self._execute(
                        account=account,
                        plan=plan,
                        session=session,
                        decision_session=decision_session,
                        config=config,
                    )
                )

            equity, position_value, mark_warnings = self._mark_to_market(
                account,
                session,
                last_prices,
            )
            warnings.extend(mark_warnings)
            equity_curve.append(
                {
                    "session": session.isoformat(),
                    "equity": float(equity),
                    "cash": float(account.cash),
                    "position_value": float(position_value),
                    "cash_weight": (
                        float(account.cash / equity)
                        if equity > 0
                        else None
                    ),
                    "position_count": len(account.holdings),
                }
            )

            if session not in decision_sessions:
                continue
            execution_session = next_session[session]
            decision_input = self._decision_input(
                run_id=f"{run_id}_{session:%Y%m%d}",
                account=account,
                universe=universe,
                universe_version=universe_version,
                data_date=session,
                execution_session=execution_session,
            )
            cache_hit = False
            try:
                bundle, cache_hit = self._decide(decision_input)
            except Exception as exc:
                bundle = RawDecisionBundle(
                    decisions={},
                    meta={
                        "calls": 0,
                        "provider_attempts": 0,
                        "engine": "failed_safe_hold",
                        "decision_quality": "failed",
                        "analysis_coverage": 0.0,
                        "stage_health": {
                            "decision_engine": "failed",
                        },
                    },
                    warnings=(
                        "Decision engine failed; safe hold used "
                        f"({_safe_exception_type(exc)})",
                    ),
                )
            risk_result = self.risk_policy.apply(
                decision_input,
                bundle,
            )
            pending[execution_session] = (session, risk_result)
            audit = _bundle_audit(bundle)
            decision_warnings = [
                str(item)
                for item in risk_result.get("warnings", [])
            ]
            warnings.extend(
                f"{session}: {item}" for item in decision_warnings
            )
            decisions.append(
                {
                    "decision_session": session.isoformat(),
                    "execution_session": execution_session.isoformat(),
                    "cache_hit": cache_hit,
                    "llm_calls": audit["fresh_provider_calls"],
                    **audit,
                    "unavailable_symbols": (
                        decision_input.unavailable_symbols
                    ),
                    "warnings": decision_warnings,
                    "portfolio_before": {
                        "cash": float(account.cash),
                        "equity": float(equity),
                        "positions": {
                            symbol: holding.shares
                            for symbol, holding in account.holdings.items()
                        },
                    },
                    "llm_meta": bundle.meta,
                    "risk_result": risk_result,
                }
            )

        first_decision_session = (
            min(decision_sessions) if decision_sessions else None
        )
        benchmark_entry_session = (
            next_session[first_decision_session]
            if first_decision_session is not None
            else None
        )
        benchmark_return, benchmark_symbols = self._benchmark_return(
            universe,
            sessions,
            benchmark_entry_session,
        )
        metrics = self._metrics(
            config=config,
            equity_curve=equity_curve,
            trades=trades,
            decisions=decisions,
            benchmark_return=benchmark_return,
            benchmark_symbols=benchmark_symbols,
        )
        if metrics["result_quality_status"] == "invalid":
            invalid_count = (
                metrics["degraded_decision_count"]
                + metrics["failed_decision_count"]
            )
            warnings.append(
                "BACKTEST_RESULT_INVALID: "
                f"{invalid_count} of {metrics['decision_count']} decisions "
                "were degraded or failed; performance metrics include "
                "fallback behavior and must not be interpreted as a valid "
                "evaluation of the complete strategy"
            )
        elif metrics["result_quality_status"] == "not_evaluated":
            warnings.append(
                "BACKTEST_RESULT_NOT_EVALUATED: no decision was generated "
                "during the requested period"
            )
        if not trades:
            warnings.append(
                "No trades were executed; inspect decision and data-quality warnings"
            )
        return BacktestResult(
            run_id=run_id,
            config=config,
            universe=universe,
            universe_version=universe_version,
            data_source=self.data_feed.name,
            decision_engine=self.settings.decision_engine_mode,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            decisions=decisions,
            warnings=list(dict.fromkeys(warnings)),
        )

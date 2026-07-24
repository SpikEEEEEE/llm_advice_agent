from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.config import Settings
from app.domain.models import DecisionInput, RawDecisionBundle


VALID_ACTIONS = {"increase", "hold", "decrease", "close"}


def _decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        output = Decimal(str(value))
        return output if output.is_finite() else default
    except (InvalidOperation, TypeError, ValueError):
        return default


def _confidence(value: Any) -> float:
    try:
        number = float(value)
        if not math.isfinite(number):
            return 0.0
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return 0.0


class AShareRiskPolicy:
    """Convert model targets into safe, advisory A-share quantities.

    This layer never submits orders. It applies deterministic constraints and
    preserves both the raw model decision and every adjustment for auditability.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def apply(
        self,
        decision_input: DecisionInput,
        bundle: RawDecisionBundle,
    ) -> dict[str, Any]:
        positions = decision_input.portfolio.position_map()
        market = decision_input.market
        warnings = list(bundle.warnings)
        raw_decision_quality = (
            bundle.meta.get("decision_quality")
            if isinstance(bundle.meta, dict)
            else None
        )
        if raw_decision_quality is None:
            # Legacy/single-agent engines predate the quality marker. Preserve
            # their existing behavior unless they explicitly provide a marker.
            decision_quality = "healthy"
        else:
            candidate_quality = str(raw_decision_quality).strip().lower()
            if candidate_quality in {"healthy", "degraded", "failed"}:
                decision_quality = candidate_quality
            else:
                # An explicit but unrecognized marker is malformed safety
                # metadata, so fail closed instead of allowing new exposure.
                decision_quality = "failed"
                warnings.append(
                    "Invalid decision_quality metadata was treated as failed"
                )
        quality_reduce_only = decision_quality in {"degraded", "failed"}
        if quality_reduce_only:
            warnings.append(
                f"Decision quality {decision_quality} enforces reduce-only risk controls"
            )
        raw_increase_blocks = (
            bundle.meta.get("increase_blocked_symbols", {})
            if isinstance(bundle.meta, dict)
            else {}
        )
        increase_blocked_symbols = (
            dict(raw_increase_blocks)
            if isinstance(raw_increase_blocks, dict)
            else {}
        )
        valid_prices: dict[str, Decimal] = {}
        for symbol, snapshot in market.items():
            price = _decimal(snapshot.reference_price)
            if price > 0:
                valid_prices[symbol] = price
            else:
                increase_blocked_symbols.setdefault(
                    symbol,
                    "INVALID_REFERENCE_PRICE",
                )
            if snapshot.data_quality_warnings:
                increase_blocked_symbols.setdefault(
                    symbol,
                    "INPUT_DATA_QUALITY_WARNING",
                )

        unpriced_held_symbols = sorted(
            symbol
            for symbol, position in positions.items()
            if position.shares > 0 and symbol not in valid_prices
        )
        valuation_complete = not unpriced_held_symbols

        known_position_value = sum(
            Decimal(position.shares) * valid_prices[symbol]
            for symbol, position in positions.items()
            if symbol in valid_prices
        )
        total_assets = decision_input.portfolio.cash + known_position_value
        reserve_cash = total_assets * self.settings.min_cash_ratio
        spendable_cash = (
            max(Decimal("0"), decision_input.portfolio.cash - reserve_cash)
            if valuation_complete
            else Decimal("0")
        )
        if quality_reduce_only:
            # This is an independent cash-level guard in addition to the
            # per-symbol target clamp below.
            spendable_cash = Decimal("0")
        max_position_value = total_assets * self.settings.max_position_ratio

        if not valuation_complete:
            warnings.append(
                "New buys were frozen because one or more existing holdings could not be valued"
            )

        plans: dict[str, dict[str, Any]] = {}
        for symbol in decision_input.symbols:
            snapshot = market.get(symbol)
            position = positions.get(symbol)
            current_shares = position.shares if position else 0
            available_shares = position.available_shares if position else 0
            raw = bundle.decisions.get(symbol, {})
            adjustments: list[dict[str, Any]] = []
            flags: list[str] = []

            if snapshot is None or symbol not in valid_prices:
                default_reason = (
                    "Market data was unavailable"
                    if snapshot is None
                    else "Reference price was invalid"
                )
                reason = decision_input.unavailable_symbols.get(
                    symbol, default_reason
                )
                plans[symbol] = {
                    "symbol": symbol,
                    "raw_decision": raw,
                    "raw_action": str(raw.get("action") or "hold").lower(),
                    "raw_target_position_value": None,
                    "action": "hold",
                    "current_shares": current_shares,
                    "available_shares": available_shares,
                    "target_shares": current_shares,
                    "delta_shares": 0,
                    "reference_price": None,
                    "current_position_value": None,
                    "target_position_value": None,
                    "confidence": 0.0,
                    "reasons": [f"Safe hold: {reason}"],
                    "risk_flags": ["DATA_UNAVAILABLE"],
                    "adjustments": [
                        {
                            "rule": "MISSING_OR_STALE_PRICE",
                            "message": "Trading recommendation suppressed",
                        }
                    ],
                }
                continue

            price = valid_prices[symbol]
            current_value = Decimal(current_shares) * price
            raw_action = str(raw.get("action") or "hold").strip().lower()
            if not raw:
                adjustments.append(
                    {
                        "rule": "MISSING_LLM_DECISION",
                        "message": "No model decision was returned; defaulted to hold",
                    }
                )
                raw_action = "hold"
            elif raw_action not in VALID_ACTIONS:
                adjustments.append(
                    {
                        "rule": "INVALID_ACTION",
                        "before": raw_action,
                        "after": "hold",
                    }
                )
                raw_action = "hold"

            quality_increase_requested = (
                quality_reduce_only and raw_action == "increase"
            )
            increase_block_reason = increase_blocked_symbols.get(symbol)
            if raw_action == "increase":
                if increase_block_reason:
                    adjustments.append(
                        {
                            "rule": "INCOMPLETE_INPUT_BLOCKS_INCREASE",
                            "reason": str(increase_block_reason),
                            "after": "hold",
                        }
                    )
                    flags.append("INCREASE_BLOCKED_BY_DATA_QUALITY")

                # Treat this independently from per-symbol data quality. A
                # recommendation can be blocked by both conditions, and both
                # reasons must remain visible in the audit result.
                if not valuation_complete:
                    adjustments.append(
                        {
                            "rule": "INCOMPLETE_PORTFOLIO_VALUATION",
                            "before": "increase",
                            "after": "hold",
                        }
                    )
                    flags.append("NEW_BUYS_FROZEN_MISSING_HOLDING_PRICE")

                if increase_block_reason or not valuation_complete:
                    raw_action = "hold"

            raw_target = _decimal(raw.get("target_cash_amount"), current_value)
            if raw_target < 0:
                adjustments.append(
                    {
                        "rule": "NON_NEGATIVE_TARGET",
                        "before": float(raw_target),
                        "after": 0.0,
                    }
                )
                raw_target = Decimal("0")

            if raw_action == "hold":
                intended_target = current_value
            elif raw_action == "close":
                intended_target = Decimal("0")
            elif raw_action == "increase":
                intended_target = max(current_value, raw_target)
                if intended_target > max_position_value:
                    capped = max(current_value, max_position_value)
                    adjustments.append(
                        {
                            "rule": "MAX_POSITION_RATIO",
                            "before": float(intended_target),
                            "after": float(capped),
                        }
                    )
                    intended_target = capped
                    flags.append("POSITION_CAP_APPLIED")
            else:  # decrease
                intended_target = min(current_value, raw_target)

            theoretical_target = max(0, int(intended_target / price))
            if theoretical_target > current_shares:
                buy_delta = ((theoretical_target - current_shares) // 100) * 100
                target_shares = current_shares + buy_delta
                if buy_delta != theoretical_target - current_shares:
                    adjustments.append(
                        {
                            "rule": "A_SHARE_BUY_BOARD_LOT",
                            "before_delta": theoretical_target - current_shares,
                            "after_delta": buy_delta,
                        }
                    )
            elif theoretical_target < current_shares:
                requested_sell = current_shares - theoretical_target
                allowed_sell = min(requested_sell, available_shares)
                target_shares = current_shares - allowed_sell
                if allowed_sell < requested_sell:
                    adjustments.append(
                        {
                            "rule": "T_PLUS_ONE_AVAILABLE_SHARES",
                            "before_sell": requested_sell,
                            "after_sell": allowed_sell,
                        }
                    )
                    flags.append("SELL_CAPPED_BY_AVAILABLE_SHARES")
            else:
                target_shares = current_shares

            if quality_reduce_only and (
                quality_increase_requested or target_shares > current_shares
            ):
                adjustments.append(
                    {
                        "rule": "DECISION_QUALITY_REDUCE_ONLY",
                        "decision_quality": decision_quality,
                        "before": target_shares,
                        "after": current_shares,
                    }
                )
                flags.append("INCREASE_BLOCKED_BY_DECISION_QUALITY")
                target_shares = min(target_shares, current_shares)

            reasons = raw.get("reasons") if isinstance(raw.get("reasons"), list) else []
            if increase_block_reason:
                reasons = [
                    *reasons,
                    f"Risk control: {increase_block_reason}",
                ]
            if quality_increase_requested:
                reasons = [
                    *reasons,
                    f"Risk control: decision quality {decision_quality} is reduce-only",
                ]
            plans[symbol] = {
                "symbol": symbol,
                "raw_decision": raw,
                "raw_action": str(raw.get("action") or "hold").lower(),
                "raw_target_position_value": float(raw_target),
                "current_shares": current_shares,
                "available_shares": available_shares,
                "target_shares": target_shares,
                "reference_price": float(price),
                "current_position_value": float(current_value),
                "confidence": _confidence(raw.get("confidence")),
                "reasons": [str(reason) for reason in reasons[:10]],
                "risk_flags": flags,
                "adjustments": adjustments,
            }

        # Allocate conservative cash to new/increased positions. Expected sale
        # proceeds are deliberately not reused because this service does not
        # execute orders and cannot guarantee those sales will fill.
        active_positions = {
            symbol
            for symbol, position in positions.items()
            if position.shares > 0
        }
        buy_plans = sorted(
            (
                plan
                for plan in plans.values()
                if plan["target_shares"] > plan["current_shares"]
            ),
            key=lambda plan: (-plan["confidence"], plan["symbol"]),
        )
        remaining_cash = spendable_cash
        fee_multiplier = Decimal("1") + self.settings.buy_fee_buffer_ratio

        for plan in buy_plans:
            if not valuation_complete:
                plan["adjustments"].append(
                    {
                        "rule": "INCOMPLETE_PORTFOLIO_VALUATION",
                        "before": plan["target_shares"],
                        "after": plan["current_shares"],
                    }
                )
                plan["risk_flags"].append("NEW_BUYS_FROZEN_MISSING_HOLDING_PRICE")
                plan["target_shares"] = plan["current_shares"]
                continue

            is_new_position = plan["current_shares"] == 0
            if is_new_position and len(active_positions) >= self.settings.max_positions:
                plan["adjustments"].append(
                    {
                        "rule": "MAX_POSITIONS",
                        "before": plan["target_shares"],
                        "after": 0,
                    }
                )
                plan["risk_flags"].append("MAX_POSITIONS_REACHED")
                plan["target_shares"] = 0
                continue

            price = _decimal(plan["reference_price"])
            requested_delta = plan["target_shares"] - plan["current_shares"]
            requested_cost = Decimal(requested_delta) * price * fee_multiplier
            if requested_cost > remaining_cash:
                affordable_lots = int(
                    remaining_cash / (Decimal(100) * price * fee_multiplier)
                )
                allowed_delta = max(0, affordable_lots * 100)
                plan["adjustments"].append(
                    {
                        "rule": "AVAILABLE_CASH",
                        "before_delta": requested_delta,
                        "after_delta": allowed_delta,
                    }
                )
                plan["risk_flags"].append("BUY_CAPPED_BY_CASH")
                plan["target_shares"] = plan["current_shares"] + allowed_delta
                requested_delta = allowed_delta
                requested_cost = Decimal(requested_delta) * price * fee_multiplier
            remaining_cash -= requested_cost
            if is_new_position and requested_delta > 0:
                active_positions.add(plan["symbol"])

        decisions = []
        for symbol in decision_input.symbols:
            plan = plans[symbol]
            delta = plan["target_shares"] - plan["current_shares"]
            if delta > 0:
                action = "increase"
            elif delta < 0 and plan["target_shares"] == 0:
                action = "close"
            elif delta < 0:
                action = "decrease"
            else:
                action = "hold"
            plan["action"] = action
            plan["delta_shares"] = delta
            if plan["reference_price"] is not None:
                plan["target_position_value"] = float(
                    Decimal(plan["target_shares"]) * _decimal(plan["reference_price"])
                )
            decisions.append(plan)

        if decision_input.unavailable_symbols:
            warnings.append(
                "Some symbols were forced to hold because current market data was unavailable"
            )

        return {
            "run_id": decision_input.run_id,
            "mode": decision_input.mode,
            "as_of": decision_input.as_of.isoformat(),
            "data_date": decision_input.data_date.isoformat(),
            "valid_for_session": decision_input.valid_for_session.isoformat(),
            "universe_version": decision_input.universe_version,
            "provider": "tushare",
            "model": self.settings.llm_model,
            "decision_quality": decision_quality,
            "portfolio_summary": {
                "cash": float(decision_input.portfolio.cash),
                "known_total_assets": float(total_assets),
                "valuation_complete": valuation_complete,
                "unpriced_held_symbols": unpriced_held_symbols,
                "minimum_cash_reserve": float(reserve_cash),
                "projected_cash_after_buys": float(
                    decision_input.portfolio.cash - (spendable_cash - remaining_cash)
                ),
            },
            "decisions": decisions,
            "warnings": warnings,
            "llm_meta": bundle.meta,
        }

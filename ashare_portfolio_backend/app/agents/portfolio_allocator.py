from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import TypeVar

from .portfolio_schemas import (
    FinalPortfolioAllocation,
    PortfolioProposal,
)


AllocationT = TypeVar(
    "AllocationT",
    PortfolioProposal,
    FinalPortfolioAllocation,
)
_ONE = Decimal("1")
_ZERO = Decimal("0")
_WEIGHT_QUANTUM = Decimal("0.000000000001")


@dataclass(frozen=True)
class AllocationNormalization:
    """Audit facts for deterministic conversion of LLM intent into hard weights."""

    reduce_only: bool
    requested_cash_weight: float
    normalized_cash_weight: float
    requested_invested_weight: float
    normalized_invested_weight: float
    scaled_to_cash_budget: bool
    position_limit_drops: tuple[str, ...]
    capped_symbols: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "reduce_only": self.reduce_only,
            "requested_cash_weight": self.requested_cash_weight,
            "normalized_cash_weight": self.normalized_cash_weight,
            "requested_invested_weight": self.requested_invested_weight,
            "normalized_invested_weight": self.normalized_invested_weight,
            "scaled_to_cash_budget": self.scaled_to_cash_budget,
            "position_limit_drops": list(self.position_limit_drops),
            "capped_symbols": list(self.capped_symbols),
        }


class DeterministicPortfolioAllocator:
    """Normalize allocation intent without inventing additional invested capital."""

    def __init__(
        self,
        *,
        minimum_cash_ratio: float,
        maximum_position_ratio: float,
        maximum_positions: int,
    ) -> None:
        self.minimum_cash = Decimal(str(minimum_cash_ratio))
        self.maximum_position = Decimal(str(maximum_position_ratio))
        self.maximum_positions = maximum_positions

    @staticmethod
    def _weight(value: float) -> Decimal:
        return max(_ZERO, Decimal(str(value)))

    def normalize(
        self,
        proposal: AllocationT,
        *,
        expected_symbols: list[str],
        current_weights: dict[str, float],
        reduce_only: bool,
        output_model: type[AllocationT],
    ) -> tuple[AllocationT, AllocationNormalization]:
        """Return a complete, bounded vector and an audit record.

        The LLM's target weights are treated as upper-bound intent. The allocator
        may cap or scale them down, but it never fills unused capacity by scaling
        positions up. Therefore every unused unit of risk budget remains cash.
        """

        if len(expected_symbols) != len(set(expected_symbols)):
            raise ValueError("Expected allocation symbols must be unique")
        proposal_symbols = [target.symbol for target in proposal.targets]
        if len(proposal_symbols) != len(set(proposal_symbols)):
            raise ValueError("Allocation intent contains duplicate symbols")
        if set(proposal_symbols) != set(expected_symbols):
            raise ValueError("Allocation intent symbol coverage mismatch")

        targets = {target.symbol: target for target in proposal.targets}
        requested: dict[str, Decimal] = {
            symbol: self._weight(targets[symbol].target_weight)
            for symbol in expected_symbols
        }
        bounded: dict[str, Decimal] = {}
        capped: set[str] = set()
        for symbol in expected_symbols:
            value = min(requested[symbol], self.maximum_position)
            if reduce_only:
                value = min(
                    value,
                    self._weight(current_weights.get(symbol, 0.0)),
                )
            if value != requested[symbol]:
                capped.add(symbol)
            bounded[symbol] = value

        active_rank = sorted(
            (symbol for symbol, value in bounded.items() if value > _ZERO),
            key=lambda symbol: (
                -bounded[symbol],
                -Decimal(str(targets[symbol].confidence)),
                symbol,
            ),
        )
        retained = set(active_rank[: self.maximum_positions])
        position_limit_drops = tuple(
            sorted(symbol for symbol in active_rank if symbol not in retained)
        )
        for symbol in position_limit_drops:
            bounded[symbol] = _ZERO

        requested_cash = max(
            self.minimum_cash,
            min(_ONE, self._weight(proposal.cash_weight)),
        )
        investable_budget = _ONE - requested_cash
        bounded_total = sum(bounded.values(), _ZERO)
        scaled = bounded_total > investable_budget and bounded_total > _ZERO
        if scaled:
            factor = investable_budget / bounded_total
            bounded = {
                symbol: value * factor
                for symbol, value in bounded.items()
            }

        normalized_weights = {
            symbol: value.quantize(_WEIGHT_QUANTUM, rounding=ROUND_DOWN)
            for symbol, value in bounded.items()
        }
        normalized_invested = sum(normalized_weights.values(), _ZERO)
        normalized_cash = _ONE - normalized_invested

        normalized_targets = [
            targets[symbol].model_copy(
                update={"target_weight": float(normalized_weights[symbol])}
            )
            for symbol in sorted(expected_symbols)
        ]
        normalized = output_model(
            targets=normalized_targets,
            cash_weight=float(normalized_cash),
            rationale=proposal.rationale,
        )
        audit = AllocationNormalization(
            reduce_only=reduce_only,
            requested_cash_weight=float(proposal.cash_weight),
            normalized_cash_weight=float(normalized_cash),
            requested_invested_weight=float(sum(requested.values(), _ZERO)),
            normalized_invested_weight=float(normalized_invested),
            scaled_to_cash_budget=scaled,
            position_limit_drops=position_limit_drops,
            capped_symbols=tuple(sorted(capped)),
        )
        return normalized, audit

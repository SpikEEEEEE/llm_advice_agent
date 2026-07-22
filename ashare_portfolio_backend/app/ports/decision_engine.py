from __future__ import annotations

from typing import Callable, Protocol

from app.domain.models import DecisionInput, RawDecisionBundle


StageCallback = Callable[[str], None]


class DecisionEngine(Protocol):
    def decide(
        self,
        decision_input: DecisionInput,
        on_stage: StageCallback | None = None,
    ) -> RawDecisionBundle:
        """Generate raw model decisions without applying trading rules."""


from __future__ import annotations

from app.core.config import Settings
from app.ports.decision_engine import DecisionEngine

from .openai_decision import OpenAIDecisionEngine
from .portfolio_multi_agent import PortfolioMultiAgentDecisionEngine


def build_decision_engine(settings: Settings) -> DecisionEngine:
    if settings.decision_engine_mode == "portfolio_multi_agent":
        return PortfolioMultiAgentDecisionEngine(settings)
    return OpenAIDecisionEngine(settings)


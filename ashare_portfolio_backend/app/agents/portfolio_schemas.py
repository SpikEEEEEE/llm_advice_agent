from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentPoint = Annotated[str, Field(min_length=1, max_length=500)]
CompactSignal = Annotated[str, Field(min_length=1, max_length=180)]
RiskFlag = Annotated[str, Field(min_length=1, max_length=48)]


class StrictAgentModel(BaseModel):
    """Base model for provider-enforced, audit-friendly agent output."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class SymbolAssessment(StrictAgentModel):
    symbol: str = Field(min_length=1, max_length=32)
    score: float = Field(ge=-100, le=100)
    confidence: float = Field(ge=0, le=1)
    stance: Literal[
        "strong_positive",
        "positive",
        "neutral",
        "negative",
        "strong_negative",
    ]
    key_signal: CompactSignal
    risk_flags: list[RiskFlag] = Field(max_length=3)


class PoolAnalystReport(StrictAgentModel):
    assessments: list[SymbolAssessment] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=800)


class DebateSymbolView(StrictAgentModel):
    symbol: str = Field(min_length=1, max_length=32)
    conviction: float = Field(ge=-100, le=100)
    reasons: list[AgentPoint] = Field(min_length=1, max_length=4)
    risks: list[AgentPoint] = Field(max_length=4)


class DebateCase(StrictAgentModel):
    views: list[DebateSymbolView] = Field(min_length=1)
    portfolio_argument: str = Field(min_length=1, max_length=4000)
    preferred_cash_ratio: float = Field(ge=0, le=1)


class ResearchCandidate(StrictAgentModel):
    symbol: str = Field(min_length=1, max_length=32)
    rating: Literal["buy", "overweight", "hold", "underweight", "sell"]
    conviction: float = Field(ge=0, le=1)
    priority: int = Field(ge=1)
    thesis: list[AgentPoint] = Field(min_length=1, max_length=4)
    risks: list[AgentPoint] = Field(max_length=4)


class PortfolioResearchPlan(StrictAgentModel):
    candidates: list[ResearchCandidate] = Field(min_length=1)
    preferred_cash_ratio: float = Field(ge=0, le=1)
    portfolio_thesis: str = Field(min_length=1, max_length=4000)


class AllocationTarget(StrictAgentModel):
    symbol: str = Field(min_length=1, max_length=32)
    target_weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reasons: list[AgentPoint] = Field(min_length=1, max_length=5)


class PortfolioProposal(StrictAgentModel):
    targets: list[AllocationTarget] = Field(min_length=1)
    cash_weight: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=4000)


class RiskAdjustment(StrictAgentModel):
    symbol: str = Field(min_length=1, max_length=32)
    suggested_weight: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1000)


class PortfolioRiskReview(StrictAgentModel):
    stance: Literal["aggressive", "neutral", "conservative"]
    approve: bool
    suggested_cash_weight: float = Field(ge=0, le=1)
    adjustments: list[RiskAdjustment]
    portfolio_risk: str = Field(min_length=1, max_length=3000)


class FinalPortfolioAllocation(StrictAgentModel):
    targets: list[AllocationTarget] = Field(min_length=1)
    cash_weight: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=4000)

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_complete_multi_agent_result_renders_without_errors():
    script = """
from ashare_portfolio_frontend.components import render_decision_run

allocation = {
    "targets": [{
        "symbol": "600519.SH",
        "target_weight": 0.2,
        "confidence": 0.8,
        "reasons": ["structured reason"],
    }],
    "cash_weight": 0.8,
    "rationale": "bounded allocation",
}
debate = {
    "views": [{
        "symbol": "600519.SH",
        "conviction": 50,
        "reasons": ["evidence"],
        "risks": ["valuation"],
    }],
    "portfolio_argument": "portfolio case",
    "preferred_cash_ratio": 0.8,
}
artifacts = {
    "analyst_reports": {
        "technical": {
            "summary": "complete pool comparison",
            "assessments": [{
                "symbol": "600519.SH",
                "score": 50,
                "confidence": 0.8,
                "stance": "positive",
                "key_signal": "trend",
                "risk_flags": [],
            }],
        }
    },
    "combined_scores": {
        "600519.SH": {
            "score": 40,
            "confidence": 0.8,
            "analysis_coverage": 1,
        }
    },
    "shortlist": ["600519.SH"],
    "bull_case": debate,
    "bear_case": debate,
    "research_plan": {
        "portfolio_thesis": "research conclusion",
        "preferred_cash_ratio": 0.8,
        "candidates": [{
            "priority": 1,
            "symbol": "600519.SH",
            "rating": "buy",
            "conviction": 0.8,
            "thesis": ["quality"],
            "risks": ["valuation"],
        }],
    },
    "raw_trader_proposal": allocation,
    "trader_proposal": allocation,
    "risk_reviews": {
        stance: {
            "approve": True,
            "suggested_cash_weight": 0.8,
            "adjustments": [],
            "portfolio_risk": f"{stance} review",
        }
        for stance in ["aggressive", "neutral", "conservative"]
    },
    "final_allocation": allocation,
    "allocation_normalization": {"final": {"scaled_to_cash_budget": False}},
}
run = {
    "id": "run_test",
    "status": "completed",
    "portfolio_version": 1,
    "universe_version": "test",
    "universe": ["600519.SH"],
    "result": {
        "decision_quality": "healthy",
        "warnings": [],
        "portfolio_summary": {"known_total_assets": 110000},
        "decisions": [{
            "symbol": "600519.SH",
            "action": "hold",
            "current_shares": 100,
            "target_shares": 100,
            "delta_shares": 0,
            "reference_price": 1000,
            "current_position_value": 100000,
            "target_position_value": 100000,
            "confidence": 0.8,
            "risk_flags": [],
            "reasons": ["hold"],
            "adjustments": [],
        }],
        "llm_meta": {
            "calls": 11,
            "provider_attempts": 11,
            "validated_outputs": 11,
            "output_repair_attempts": 0,
            "analysis_coverage": 1,
            "resolved_response_format": "json_schema",
            "stage_health": {"overall": {"status": "healthy"}},
            "agent_trace": [{"agent": "technical_analyst"}],
            "agent_artifacts": artifacts,
        },
        "market_snapshot": {
            "600519.SH": {
                "data_date": "2026-07-24",
                "reference_price": 1000,
                "fundamentals": {"pe_ttm": 20},
                "news": [],
                "data_quality_warnings": [],
            }
        },
    },
}
render_decision_run(run)
"""
    app = AppTest.from_string(script, default_timeout=20).run()

    assert not app.exception
    assert any(tab.label == "多空辩论" for tab in app.tabs)
    assert any(
        button.label == "下载完整决策 JSON"
        for button in app.get("download_button")
    )

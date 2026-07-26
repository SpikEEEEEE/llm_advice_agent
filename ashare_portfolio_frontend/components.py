from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


ACTION_LABELS = {
    "increase": "加仓",
    "hold": "持有",
    "decrease": "减仓",
    "close": "清仓",
}
STATUS_LABELS = {
    "pending": "等待执行",
    "fetching_data": "获取市场数据",
    "building_features": "构建特征",
    "calling_llm": "多 Agent 研究",
    "validating": "确定性风控",
    "completed": "已完成",
    "degraded": "降级完成",
    "failed": "失败",
}
QUALITY_LABELS = {
    "healthy": "健康",
    "degraded": "降级 / 只减仓",
    "failed": "失败 / 安全持有",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #10243a;
          --muted: #627386;
          --line: #dfe7ee;
          --panel: #ffffff;
          --accent: #0f8f83;
          --navy: #102d4f;
          --amber: #d79024;
        }
        .stApp {
          background:
            radial-gradient(circle at 92% 3%, rgba(15,143,131,.09), transparent 24rem),
            #f4f7fa;
        }
        [data-testid="stSidebar"] {
          background: #0e263f;
          border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #eef5f7; }
        [data-testid="stSidebar"] input {
          color: #10243a !important;
          background: #ffffff !important;
        }
        .hero {
          padding: 1.7rem 1.9rem 1.55rem;
          border-radius: 18px;
          background: linear-gradient(122deg, #102d4f 0%, #16466a 58%, #0f8f83 140%);
          box-shadow: 0 16px 42px rgba(16,45,79,.16);
          margin-bottom: 1.35rem;
          color: white;
        }
        .hero-kicker {
          font-size: .72rem;
          letter-spacing: .16em;
          text-transform: uppercase;
          color: #8fe1d9;
          font-weight: 700;
          margin-bottom: .45rem;
        }
        .hero h1 {
          margin: 0;
          padding: 0;
          color: #ffffff;
          font-size: 2rem;
          letter-spacing: -.03em;
        }
        .hero p {
          max-width: 48rem;
          margin: .55rem 0 0;
          color: #d9e8ef;
          line-height: 1.7;
        }
        .section-label {
          color: #0f8f83;
          font-size: .76rem;
          font-weight: 800;
          letter-spacing: .12em;
          text-transform: uppercase;
          margin: .3rem 0 .2rem;
        }
        div[data-testid="stMetric"] {
          background: rgba(255,255,255,.88);
          border: 1px solid var(--line);
          padding: .9rem 1rem;
          border-radius: 14px;
          box-shadow: 0 6px 20px rgba(22,50,75,.04);
        }
        div[data-testid="stForm"], div[data-testid="stExpander"] {
          background: rgba(255,255,255,.88);
          border-color: var(--line);
          border-radius: 14px;
        }
        .stButton > button, .stFormSubmitButton > button {
          border-radius: 10px;
          min-height: 2.65rem;
          font-weight: 700;
        }
        .stFormSubmitButton > button {
          background: #0f8f83;
          color: white;
          border: 1px solid #0f8f83;
        }
        .audit-note {
          border-left: 3px solid #0f8f83;
          background: #eaf6f4;
          color: #204b50;
          padding: .75rem .9rem;
          border-radius: 0 9px 9px 0;
          font-size: .88rem;
          line-height: 1.55;
          margin-bottom: .8rem;
        }
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--line);
          border-radius: 12px;
          overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">A-SHARE DECISION DESK</div>
          <h1>组合研究与投资建议台</h1>
          <p>
            将当前资金、持仓和本次股票池固化为决策快照，查看从三路分析、
            多空辩论、组合构建到 A 股确定性风控的完整证据链。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(value: str) -> None:
    st.markdown(
        f'<div class="section-label">{value}</div>',
        unsafe_allow_html=True,
    )


def render_run_progress(run: dict[str, Any]) -> None:
    status = str(run.get("status") or "pending")
    steps = [
        "pending",
        "fetching_data",
        "building_features",
        "calling_llm",
        "validating",
        "completed",
    ]
    if status in {"degraded", "failed"}:
        progress = 1.0
    else:
        try:
            progress = (steps.index(status) + 1) / len(steps)
        except ValueError:
            progress = 0.1
    st.progress(progress, text=STATUS_LABELS.get(status, status))
    st.caption(f"任务 {run.get('id', '—')} · 组合版本 {run.get('portfolio_version', '—')}")


def _decision_rows(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "股票": item.get("symbol"),
                "建议": ACTION_LABELS.get(
                    str(item.get("action")), item.get("action")
                ),
                "当前股数": item.get("current_shares"),
                "目标股数": item.get("target_shares"),
                "变化股数": item.get("delta_shares"),
                "参考价": item.get("reference_price"),
                "当前市值": item.get("current_position_value"),
                "目标市值": item.get("target_position_value"),
                "置信度": item.get("confidence"),
                "风险标记": " · ".join(item.get("risk_flags") or []),
                "核心理由": "；".join(str(x) for x in item.get("reasons") or []),
            }
            for item in decisions
        ]
    )


def _render_allocation(
    allocation: dict[str, Any] | None,
    *,
    empty_message: str,
) -> None:
    if not allocation:
        st.caption(empty_message)
        return
    targets = allocation.get("targets") or []
    if targets:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "股票": item.get("symbol"),
                        "目标权重": item.get("target_weight"),
                        "置信度": item.get("confidence"),
                        "理由": "；".join(item.get("reasons") or []),
                    }
                    for item in targets
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "目标权重": st.column_config.NumberColumn(format="%.2f"),
                "置信度": st.column_config.ProgressColumn(min_value=0, max_value=1),
            },
        )
    st.caption(
        f"现金权重：{float(allocation.get('cash_weight') or 0):.1%} · "
        f"{allocation.get('rationale') or '未提供组合说明'}"
    )


def _render_debate_case(case: dict[str, Any] | None, title: str) -> None:
    st.markdown(f"#### {title}")
    if not case:
        st.caption("本次任务没有生成该阶段产物。")
        return
    st.write(case.get("portfolio_argument") or "—")
    st.caption(
        f"偏好现金比例：{float(case.get('preferred_cash_ratio') or 0):.1%}"
    )
    for view in case.get("views") or []:
        with st.expander(
            f"{view.get('symbol', '—')} · conviction "
            f"{float(view.get('conviction') or 0):.0f}"
        ):
            reasons = view.get("reasons") or []
            risks = view.get("risks") or []
            if reasons:
                st.markdown("**论据**")
                for reason in reasons:
                    st.write(f"• {reason}")
            if risks:
                st.markdown("**风险**")
                for risk in risks:
                    st.write(f"• {risk}")


def _render_agent_research(artifacts: dict[str, Any]) -> None:
    reports = artifacts.get("analyst_reports") or {}
    if not reports:
        st.info("本次结果没有多 Agent 研究产物；可能使用的是单 LLM 引擎。")
        return

    for role, report in reports.items():
        role_name = {
            "technical": "技术分析师",
            "fundamental": "基本面分析师",
            "news": "新闻分析师",
        }.get(role, role)
        with st.expander(f"{role_name} · {report.get('summary', '')}"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "股票": item.get("symbol"),
                            "评分": item.get("score"),
                            "置信度": item.get("confidence"),
                            "观点": item.get("stance"),
                            "关键信号": item.get("key_signal"),
                            "风险": " · ".join(item.get("risk_flags") or []),
                        }
                        for item in report.get("assessments") or []
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    combined = artifacts.get("combined_scores") or {}
    if combined:
        st.markdown("#### 确定性聚合与候选名单")
        shortlist = set(artifacts.get("shortlist") or [])
        rows = [
            {
                "股票": symbol,
                "综合评分": values.get("score"),
                "综合置信度": values.get("confidence"),
                "分析覆盖率": values.get("analysis_coverage"),
                "进入 shortlist": "是" if symbol in shortlist else "否",
            }
            for symbol, values in combined.items()
        ]
        st.dataframe(
            pd.DataFrame(rows).sort_values("综合评分", ascending=False),
            width="stretch",
            hide_index=True,
        )

    plan = artifacts.get("research_plan") or {}
    if plan:
        st.markdown("#### Research Manager 研究结论")
        st.write(plan.get("portfolio_thesis") or "—")
        st.caption(
            f"偏好现金比例：{float(plan.get('preferred_cash_ratio') or 0):.1%}"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "优先级": item.get("priority"),
                        "股票": item.get("symbol"),
                        "评级": item.get("rating"),
                        "确信度": item.get("conviction"),
                        "投资逻辑": "；".join(item.get("thesis") or []),
                        "主要风险": "；".join(item.get("risks") or []),
                    }
                    for item in plan.get("candidates") or []
                ]
            ).sort_values("优先级"),
            width="stretch",
            hide_index=True,
        )


def _render_risk_workflow(artifacts: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Trader 原始意图")
        _render_allocation(
            artifacts.get("raw_trader_proposal"),
            empty_message="没有原始 Trader 产物。",
        )
    with right:
        st.markdown("#### 确定性归一化后")
        _render_allocation(
            artifacts.get("trader_proposal"),
            empty_message="没有归一化 Trader 产物。",
        )

    reviews = artifacts.get("risk_reviews") or {}
    if reviews:
        st.markdown("#### 三路风险审查")
        columns = st.columns(3)
        for column, stance in zip(
            columns,
            ["aggressive", "neutral", "conservative"],
            strict=False,
        ):
            review = reviews.get(stance)
            with column:
                stance_label = {
                    "aggressive": "激进",
                    "neutral": "中性",
                    "conservative": "保守",
                }[stance]
                st.markdown(f"**{stance_label}**")
                if not review:
                    st.caption("该审查没有成功返回。")
                    continue
                st.write(review.get("portfolio_risk") or "—")
                st.caption(
                    f"审批：{'通过' if review.get('approve') else '不通过'} · "
                    f"建议现金：{float(review.get('suggested_cash_weight') or 0):.1%}"
                )
                for adjustment in review.get("adjustments") or []:
                    st.write(
                        f"• {adjustment.get('symbol')} → "
                        f"{float(adjustment.get('suggested_weight') or 0):.1%}："
                        f"{adjustment.get('rationale')}"
                    )

    final_left, final_right = st.columns([1.5, 1])
    with final_left:
        st.markdown("#### Portfolio Manager 最终组合")
        _render_allocation(
            artifacts.get("final_allocation"),
            empty_message="没有最终组合产物。",
        )
    with final_right:
        st.markdown("#### 归一化审计")
        st.json(artifacts.get("allocation_normalization") or {})


def _render_market_snapshot(snapshot: dict[str, Any]) -> None:
    if not snapshot:
        st.caption("没有保存市场快照。")
        return
    for symbol, item in snapshot.items():
        with st.expander(
            f"{symbol} · {item.get('data_date', '—')} · "
            f"参考价 {item.get('reference_price', '—')}"
        ):
            warnings = item.get("data_quality_warnings") or []
            if warnings:
                st.warning("；".join(str(value) for value in warnings))
            st.markdown("**基本面**")
            st.json(item.get("fundamentals") or {})
            st.markdown("**新闻**")
            news = item.get("news") or []
            if not news:
                st.caption("本次快照没有新闻。")
            for article in news:
                st.write(
                    f"• {article.get('title') or '无标题'} "
                    f"({article.get('published_utc') or '时间未知'})"
                )
                if article.get("description"):
                    st.caption(str(article["description"]))


def render_decision_run(run: dict[str, Any]) -> None:
    result = run.get("result") or {}
    llm_meta = result.get("llm_meta") or {}
    artifacts = llm_meta.get("agent_artifacts") or {}
    quality = str(result.get("decision_quality") or llm_meta.get("decision_quality") or "—")
    coverage = llm_meta.get("analysis_coverage")
    portfolio_summary = result.get("portfolio_summary") or {}

    section_label("DECISION SNAPSHOT")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "任务状态",
        STATUS_LABELS.get(str(run.get("status")), str(run.get("status") or "—")),
    )
    metric_columns[1].metric("决策质量", QUALITY_LABELS.get(quality, quality))
    metric_columns[2].metric(
        "分析覆盖",
        f"{float(coverage):.0%}" if isinstance(coverage, (int, float)) else "—",
    )
    metric_columns[3].metric("LLM 调用", llm_meta.get("calls", 0))
    total_assets = portfolio_summary.get("known_total_assets")
    metric_columns[4].metric(
        "已知总资产",
        f"¥{float(total_assets):,.2f}" if total_assets is not None else "—",
    )

    if run.get("error_message"):
        st.error(
            f"{run.get('error_code') or 'TASK_FAILED'}：{run.get('error_message')}"
        )
    warnings = result.get("warnings") or []
    if warnings:
        with st.expander(f"风险与数据警告（{len(warnings)}）", expanded=True):
            for warning in warnings:
                st.write(f"• {warning}")

    tabs = st.tabs(
        [
            "投资建议",
            "研究全景",
            "多空辩论",
            "交易与风控",
            "调用审计",
            "市场快照",
            "完整 JSON",
        ]
    )
    with tabs[0]:
        decisions = result.get("decisions") or []
        if decisions:
            st.dataframe(
                _decision_rows(decisions),
                width="stretch",
                hide_index=True,
                column_config={
                    "置信度": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                    ),
                    "参考价": st.column_config.NumberColumn(format="%.2f"),
                    "当前市值": st.column_config.NumberColumn(format="%.2f"),
                    "目标市值": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            for decision in decisions:
                adjustments = decision.get("adjustments") or []
                if adjustments:
                    with st.expander(
                        f"{decision.get('symbol')} · 风控调整 {len(adjustments)} 项"
                    ):
                        st.json(adjustments)
        else:
            st.info("本次任务没有产生可执行的建议，系统保持安全持有。")
        st.markdown(
            '<div class="audit-note">界面展示的是投资建议，不会向券商提交订单。'
            "最终目标股数已经过现金、整手、T+1、仓位与数据质量约束。</div>",
            unsafe_allow_html=True,
        )

    with tabs[1]:
        _render_agent_research(artifacts)

    with tabs[2]:
        bull, bear = st.columns(2)
        with bull:
            _render_debate_case(artifacts.get("bull_case"), "Bull Researcher")
        with bear:
            _render_debate_case(artifacts.get("bear_case"), "Bear Researcher")

    with tabs[3]:
        _render_risk_workflow(artifacts)

    with tabs[4]:
        audit_columns = st.columns(4)
        audit_columns[0].metric(
            "供应商尝试",
            llm_meta.get("provider_attempts", 0),
        )
        audit_columns[1].metric(
            "结构化输出",
            llm_meta.get("validated_outputs", 0),
        )
        audit_columns[2].metric(
            "输出修复",
            llm_meta.get("output_repair_attempts", 0),
        )
        audit_columns[3].metric(
            "实际格式",
            llm_meta.get("resolved_response_format") or "—",
        )
        st.markdown("#### 阶段健康度")
        st.json(llm_meta.get("stage_health") or {})
        trace = llm_meta.get("agent_trace") or []
        st.markdown("#### Agent 调用轨迹")
        if trace:
            st.json(trace)
        else:
            st.caption("本次引擎没有记录多 Agent 调用轨迹。")

    with tabs[5]:
        _render_market_snapshot(result.get("market_snapshot") or {})

    with tabs[6]:
        st.caption(
            "这里保留后端返回的完整结构化输出，包括所有 Agent 产物、辩论、"
            "原始目标、风控调整和调用审计。"
        )
        st.download_button(
            "下载完整决策 JSON",
            data=json.dumps(run, ensure_ascii=False, indent=2),
            file_name=f"{run.get('id', 'decision')}.json",
            mime="application/json",
            width="stretch",
        )
        st.json(run)


def history_label(run: dict[str, Any]) -> str:
    created = str(run.get("created_at") or "")
    try:
        created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    status = STATUS_LABELS.get(str(run.get("status")), str(run.get("status")))
    return f"{created} · {status} · {run.get('id', '—')}"

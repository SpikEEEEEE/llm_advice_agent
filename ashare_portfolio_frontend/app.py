from __future__ import annotations

import os
import time
from decimal import Decimal, InvalidOperation
from uuid import uuid4

import pandas as pd
import streamlit as st

from ashare_portfolio_frontend.api_client import AdvisorApi, BackendError
from ashare_portfolio_frontend.components import (
    history_label,
    inject_styles,
    render_decision_run,
    render_hero,
    render_run_progress,
    section_label,
)
from ashare_portfolio_frontend.validation import (
    InputValidationError,
    parse_positions,
    parse_universe,
)


TERMINAL_STATUSES = {"completed", "degraded", "failed"}
DEFAULT_POSITIONS = pd.DataFrame(
    [
        {
            "symbol": "",
            "shares": None,
            "available_shares": None,
            "average_cost": None,
            "holding_days": None,
        }
    ]
)


st.set_page_config(
    page_title="A 股组合研究台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


def _state_default(name: str, value: object) -> None:
    if name not in st.session_state:
        st.session_state[name] = value


_state_default(
    "api_url",
    os.getenv("ADVISOR_API_URL", "http://127.0.0.1:8000"),
)
_state_default("api_key", os.getenv("BACKEND_API_KEY", ""))
_state_default("universe_text", "")
_state_default("universe_loaded", False)
_state_default("positions_table", DEFAULT_POSITIONS)
_state_default("active_run_id", None)
_state_default("latest_run", None)
_state_default("workspace", "创建投资建议")


def _api() -> AdvisorApi:
    return AdvisorApi(
        st.session_state.api_url,
        st.session_state.api_key,
    )


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 决策控制台")
        st.caption("连接现有 FastAPI 服务；凭据只保存在当前页面会话。")
        st.text_input("后端地址", key="api_url")
        st.text_input("API Key", type="password", key="api_key")
        if st.button("检测连接", width="stretch"):
            try:
                live = _api().live()
                st.success(f"已连接：{live.get('service', 'backend')}")
                try:
                    ready = _api().ready()
                    if ready.get("status") != "ready":
                        st.warning("服务在线，但部分行情或模型配置尚未就绪。")
                except BackendError:
                    st.warning("服务在线，但 readiness 检查未通过。")
            except BackendError as exc:
                st.error(str(exc))

        st.divider()
        st.radio(
            "工作区",
            ["创建投资建议", "决策档案"],
            key="workspace",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "安全边界\n\n"
            "• 仅生成建议，不连接券商\n\n"
            "• 数据异常时禁止加仓\n\n"
            "• 降级决策只允许减仓"
        )


def _load_default_universe(api: AdvisorApi, *, force: bool = False) -> None:
    if st.session_state.universe_loaded and not force:
        return
    try:
        payload = api.universe()
        st.session_state.universe_text = "\n".join(payload.get("symbols") or [])
        st.session_state.universe_loaded = True
        st.session_state.universe_version = payload.get("version")
    except BackendError as exc:
        st.session_state.universe_load_error = str(exc)


def _poll_active_run(api: AdvisorApi) -> None:
    run_id = st.session_state.active_run_id
    if not run_id:
        return
    section_label("LIVE DECISION RUN")
    try:
        run = api.decision_run(run_id)
    except BackendError as exc:
        st.error(str(exc))
        if st.button("重新查询任务"):
            st.rerun()
        return

    render_run_progress(run)
    if run.get("status") in TERMINAL_STATUSES:
        st.session_state.latest_run = run
        st.session_state.active_run_id = None
        if run.get("status") == "completed":
            st.success("组合研究和确定性风控已经完成。")
        elif run.get("status") == "degraded":
            st.warning("任务已降级完成，请重点查看数据质量和 reduce-only 警告。")
        else:
            st.error("任务安全失败，没有产生新增风险建议。")
        return

    st.caption("页面会自动刷新任务状态，通常耗时取决于股票数量和模型服务。")
    time.sleep(1.5)
    st.rerun()


def _create_workspace(api: AdvisorApi) -> None:
    _load_default_universe(api)
    _poll_active_run(api)

    section_label("NEW PORTFOLIO SNAPSHOT")
    title_column, action_column = st.columns([4, 1])
    with title_column:
        st.subheader("输入本次决策依据")
        st.caption(
            "提交后，现金、持仓和股票池会作为不可变快照保存，后续修改不会影响本次任务。"
        )
    with action_column:
        if st.button("载入服务器股票池", width="stretch"):
            _load_default_universe(api, force=True)
            st.rerun()

    if st.session_state.get("universe_load_error") and not st.session_state.universe_text:
        st.warning(st.session_state.universe_load_error)

    with st.form("portfolio_advice_form", clear_on_submit=False):
        top_left, top_right = st.columns([1.3, 1])
        with top_left:
            portfolio_name = st.text_input(
                "组合名称",
                value="当前投资组合",
                max_chars=100,
            )
            cash = st.number_input(
                "可用现金（元）",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                format="%.2f",
            )
        with top_right:
            mode_label = st.selectbox(
                "决策范围",
                ["股票池再平衡", "仅分析当前持仓"],
                help=(
                    "股票池再平衡会在候选股票和当前持仓的并集中寻找机会；"
                    "仅分析当前持仓不会提出新股票。"
                ),
            )
            version = st.session_state.get("universe_version")
            if version:
                st.caption(f"当前服务器股票池版本：{version}")

        universe_text = st.text_area(
            "本次股票池",
            key="universe_text",
            height=160,
            placeholder="600519.SH\n300750.SZ\n000001.SZ",
            help="每行一个，也支持逗号或空格分隔；股票池最多 200 只。",
        )

        st.markdown("#### 当前持仓")
        st.caption(
            "可卖股数用于表达 A 股 T+1 限制；留空时默认等于持有股数。"
        )
        positions_table = st.data_editor(
            st.session_state.positions_table,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn(
                    "股票代码",
                    help="例如 600519.SH",
                ),
                "shares": st.column_config.NumberColumn(
                    "持有股数",
                    min_value=1,
                    step=1,
                    format="%d",
                ),
                "available_shares": st.column_config.NumberColumn(
                    "可卖股数",
                    min_value=0,
                    step=1,
                    format="%d",
                ),
                "average_cost": st.column_config.NumberColumn(
                    "平均成本",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                ),
                "holding_days": st.column_config.NumberColumn(
                    "持有天数",
                    min_value=0,
                    step=1,
                    format="%d",
                ),
            },
            key="positions_editor",
        )
        acknowledge = st.checkbox(
            "我理解结果仅为投资建议，不会自动下单。",
        )
        submitted = st.form_submit_button(
            "生成投资建议",
            width="stretch",
            disabled=bool(st.session_state.active_run_id),
        )

    st.session_state.positions_table = positions_table
    if submitted:
        try:
            if not acknowledge:
                raise InputValidationError("请先确认投资建议的使用边界。")
            positions = parse_positions(positions_table.to_dict("records"))
            mode = (
                "rebalance"
                if mode_label == "股票池再平衡"
                else "holdings_only"
            )
            if mode == "holdings_only" and not positions:
                raise InputValidationError("仅分析当前持仓模式至少需要一条持仓。")
            universe = parse_universe(universe_text) if mode == "rebalance" else None
            try:
                cash_decimal = Decimal(str(cash))
            except InvalidOperation as exc:
                raise InputValidationError("现金必须是有效数字。") from exc

            portfolio = api.create_portfolio(
                {
                    "name": portfolio_name,
                    "cash": str(cash_decimal),
                    "positions": positions,
                }
            )
            decision_payload = {
                "portfolio_id": portfolio["id"],
                "mode": mode,
            }
            if universe is not None:
                decision_payload["universe"] = universe
            run = api.create_decision_run(
                decision_payload,
                idempotency_key=f"streamlit-{uuid4().hex}",
            )
            st.session_state.latest_run = None
            if run.get("status") in TERMINAL_STATUSES:
                st.session_state.latest_run = run
                st.session_state.active_run_id = None
            else:
                st.session_state.active_run_id = run["id"]
            st.rerun()
        except (BackendError, InputValidationError) as exc:
            st.error(str(exc))

    latest_run = st.session_state.latest_run
    if latest_run:
        st.divider()
        st.subheader("本次投资建议与完整研究记录")
        render_decision_run(latest_run)


def _history_workspace(api: AdvisorApi) -> None:
    section_label("DECISION ARCHIVE")
    st.subheader("历史决策档案")
    st.caption(
        "这些记录直接来自后端 SQLite；每条记录保留当时的股票池版本、"
        "完整结构化 Agent 产物和最终风控结果。"
    )
    history_limit = st.slider("读取最近记录", 10, 100, 30, 10)
    try:
        runs = api.decision_runs(limit=history_limit)
    except BackendError as exc:
        st.error(str(exc))
        return
    if not runs:
        st.info("还没有历史决策。创建第一条建议后，它会出现在这里。")
        return

    selected_id = st.selectbox(
        "选择决策任务",
        options=[run["id"] for run in runs],
        format_func=lambda run_id: history_label(
            next(run for run in runs if run["id"] == run_id)
        ),
    )
    selected = next(run for run in runs if run["id"] == selected_id)
    universe = selected.get("universe") or []
    st.caption(
        f"股票池：{len(universe)} 只 · 版本 {selected.get('universe_version', '—')} · "
        f"组合版本 {selected.get('portfolio_version', '—')}"
    )
    render_decision_run(selected)


_sidebar()
render_hero()
try:
    api = _api()
except ValueError as exc:
    st.error(str(exc))
else:
    if st.session_state.workspace == "决策档案":
        _history_workspace(api)
    else:
        _create_workspace(api)

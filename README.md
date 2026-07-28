# A-share Portfolio Advisor Backend

一个完全独立的 FastAPI 项目：接收用户手工维护的 A 股持仓和现金，直接读取
Tushare 数据，调用 OpenAI-compatible LLM 生成目标仓位，再经过确定性的 A 股风控
输出组合建议。

本服务只提供分析建议，不连接券商，也不会创建或提交订单。

## 独立性边界

- 只依赖 `pyproject.toml` 中声明的公开第三方 Python 包；
- 不动态修改 `sys.path`，不查找相邻源码目录；
- 行情、交易日历、复权因子、基本面和新闻均直接通过 Tushare SDK 获取；
- 技术指标、Prompt、模型调用、严格 JSON 校验和 A 股风控均由本项目实现；
- SQLite 数据库与文件缓存只写入本项目自己的 `data/` 目录；
- 可以单独复制、构建 wheel、安装和启动。

## 主要能力

- 创建、查询和更新手工持仓组合；
- `holdings_only` 与固定股票池再平衡两种模式；
- 固定股票池自动与当前持仓取并集；
- SQLite WAL 持久化；
- 有界异步任务队列、幂等键和子进程硬超时；
- 公开 Tushare SDK + 项目本地 cache-first 数据层；
- 交易日、行情、复权因子、估值、财务指标和公司相关新闻；
- 本地计算收益率、均线、波动率、RSI、成交量比例和 52 周位置；
- OpenAI-compatible Chat Completions；
- 可按供应商能力选择 strict JSON Schema 或严格 `json_object`，DeepSeek
  在 `auto` 模式下直接使用 `json_object`，不会为每个 Agent 先做一次失败探测；
- 可选的股票池原生多 Agent 决策图：全池横向研究、多空辩论、组合构建、三路风险审查；
- 多 Agent 不按股票循环调用，而是共享同一个股票池状态并最终只输出一个组合权重向量；
- 模型输出必须覆盖所有有效股票，并通过严格字段、数值和 action/target 语义校验；
- A 股买入 100 股整手、T+1 可卖数量、现金、最低现金比例、单股仓位和最大持仓数限制；
- 缺价时强制 `hold`，持仓估值不完整时冻结全部新买入；
- 数据质量异常时禁止新增或加仓，但仍允许确定性风控后的减仓；
- 单只股票特征构建失败时只隔离该股票并把整次组合决策降级为 reduce-only，不会让
  一只畸形快照击穿完整股票池任务；
- 原始模型结论、风控调整和最终结论分别保存，便于审计；
- 可选 `X-API-Key` 认证、健康检查和 OpenAPI 文档。

## 目录结构

```text
ashare_portfolio_backend/
├── app/
│   ├── adapters/
│   │   ├── openai_decision.py  # 原有的单次 LLM 决策引擎
│   │   ├── portfolio_multi_agent.py # 多 Agent 的 LLM 适配与最终输出转换
│   │   ├── decision_engine_factory.py # 按配置选择决策引擎
│   │   └── tushare.py          # 直接调用 Tushare + 本地缓存
│   ├── agents/
│   │   ├── portfolio_graph.py  # 股票池研究、辩论、交易、风控状态图
│   │   └── portfolio_schemas.py # 各节点严格结构化输出契约
│   ├── api/                     # FastAPI 路由与认证
│   ├── core/                    # 配置与 JSON 安全处理
│   ├── domain/                  # 领域模型、特征和 A 股风控
│   ├── ports/                   # 可替换的数据与决策接口
│   ├── repositories/            # SQLite 持久化
│   ├── schemas/                 # HTTP Schema
│   ├── services/                # 决策编排和任务运行器
│   ├── container.py             # 依赖组装
│   ├── main.py                  # FastAPI 入口
│   └── worker.py                # 单次决策子进程
├── config/universe.yaml
├── tests/
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## 安装

```bash
cd /Users/test/Documents/Personal/ashare_portfolio_backend
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
```

编辑 `.env`，至少配置：

```dotenv
TUSHARE_TOKEN=your-token
OPENAI_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_STRUCTURED_OUTPUT_MODE=auto
```

也可以用 `LLM_API_KEY` 替代 `OPENAI_API_KEY`，并将 `LLM_BASE_URL` 指向兼容
Chat Completions 的服务。

默认仍使用原来的单次 LLM 引擎。启用股票池多 Agent 图：

```dotenv
DECISION_ENGINE=portfolio_multi_agent
MULTI_AGENT_SHORTLIST_SIZE=8
MULTI_AGENT_PARALLELISM=3
MULTI_AGENT_MAX_CALLS=32
MULTI_AGENT_OUTPUT_RETRIES=1
MULTI_AGENT_SEMANTIC_RETRIES=1
MULTI_AGENT_MIN_ANALYSTS=2
MULTI_AGENT_MIN_RISK_REVIEWS=2
```

`MULTI_AGENT_SHORTLIST_SIZE` 只限制进入深度辩论的非持仓候选；已有持仓一定进入
shortlist，因此不会因为预筛选而失去减仓或清仓判断。三个分析权重可以通过
`MULTI_AGENT_TECHNICAL_WEIGHT`、`MULTI_AGENT_FUNDAMENTAL_WEIGHT` 和
`MULTI_AGENT_NEWS_WEIGHT` 调整。

`LLM_STRUCTURED_OUTPUT_MODE` 可设为：

- `auto`：`api.deepseek.com` 直接使用 `json_object`，其他地址先使用
  `json_schema`，只有供应商明确不支持时才降级；
- `json_object`：适用于 DeepSeek 等支持 JSON Mode、但不实现 strict JSON
  Schema 的 OpenAI-compatible 服务；
- `json_schema`：只使用 strict JSON Schema，不进行格式能力降级。

例如使用 DeepSeek 时可配置：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-deepseek-key
LLM_MODEL=your-enabled-deepseek-model
LLM_STRUCTURED_OUTPUT_MODE=json_object
```

`MULTI_AGENT_OUTPUT_RETRIES` 用于 JSON 解码或 Pydantic 字段校验失败后的定向修复；
`MULTI_AGENT_SEMANTIC_RETRIES` 用于股票覆盖、priority 等跨字段语义校验失败后的
修复。修复请求会携带有限长度的原始输出、具体字段路径和校验错误，而不是只重复
原 Prompt。两类重试都计入 `MULTI_AGENT_MAX_CALLS`。

## 股票池多 Agent 决策图

启用 `portfolio_multi_agent` 后，一次决策任务的状态流为：

```text
完整股票池快照
  ├─ Technical Analyst ─┐
  ├─ Fundamental Analyst ├─> 确定性加权排名 -> shortlist
  └─ News Analyst ──────┘
                                |
                         Bull Researcher
                                |
                         Bear Researcher
                                |
                        Research Manager
                                |
                        Portfolio Trader
                                |
        ┌─ Aggressive Risk ─────┼─ Neutral Risk ─┐
        └─ Conservative Risk ───┴────────────────┘
                                |
                       Portfolio Manager
                                |
                    全组合目标权重 + 现金权重
                                |
                       现有 A 股确定性风控
```

三个 Analyst 每次都看到完整有效股票池并做横向比较；全池阶段每只股票只返回紧凑
的分数、置信度、方向、短结论和标准化风险标签，详细辩论只针对 shortlist，避免
20 股输出因自然语言过长而被截断。后续角色共享前序结构化产物，而不是让每只股票
各自形成互不知情的结论。

分析阶段不会把失败 Agent 的权重重新归一化给剩余 Agent，而是保留原始权重并输出
`analysis_coverage`：

- 股票池行情或持仓估值不完整时，准备阶段直接标记为 `degraded`，整个组合只能
  减仓或保持；
- 3/3 Analyst 成功时为 `healthy`，允许在后续风控约束内新增风险；
- 达到 `MULTI_AGENT_MIN_ANALYSTS`、但不足 3 个时为 `degraded`，整个组合只能
  减仓或保持，不能新开仓或加仓；
- 未达到 Analyst 法定人数时为 `failed`，决策失败并进入安全持有；
- 风险审查未达到 `MULTI_AGENT_MIN_RISK_REVIEWS` 时，即使研究完成也只能使用
  确定性的 reduce-only 安全组合。

Trader 和 Portfolio Manager 提供相对偏好与目标意图；最终精确权重由确定性分配器
执行归一化、最低现金、单股上限、最大持仓数和 reduce-only 约束。LLM 不再独自承担
“股票权重加现金必须等于 100%”的算术正确性。`AShareRiskPolicy` 还会在生成最终
股数时再次读取 `decision_quality`：只要是 `degraded`、`failed` 或非法质量标记，
即使上游错误地产生了加仓目标，也会在最后一道风控中被强制压回当前持仓。

正常路径基础调用数为 11 次：3 个 Analyst、Bull、Bear、Research Manager、
Portfolio Trader、3 个 Risk Reviewer 和 Portfolio Manager。最终结果的
`llm_meta.agent_artifacts` 会保留各阶段结构化结论，`agent_trace` 会保留调用和
token usage，便于审计。格式修复、语义修复和网络重试会增加供应商尝试次数，因此
应同时查看 `provider_attempts`、`validated_outputs` 和
`output_repair_attempts`；`configured_response_format` 与
`resolved_response_format` 会分别记录配置模式和本次实际采用的格式。它们是当前
任务内的显式状态，不会自动读取上一次交易任务；跨任务记忆仍需另行设计绩效反馈
或交易日志输入。

## 历史回测

回测是独立 CLI，不开放历史 HTTP 接口，也不会写入生产 Portfolio 或 Decision Run。
它复用相同的决策引擎、股票池输入和 `AShareRiskPolicy`，在历史收盘后生成建议，并
在下一交易日开盘执行模拟成交：

```bash
PYTHONPATH=ashare_portfolio_backend \
.venv/bin/python -m app.backtest.cli \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --engine portfolio_multi_agent \
  --rebalance monthly \
  --max-decisions 24 \
  --initial-cash 1000000
```

也可以只测试部分股票：

```text
--symbols 600519.SH,300750.SZ,601318.SH
```

默认行为：

- 首个交易日收盘生成初始组合，第二个交易日开盘成交；
- 随后在每月最后一个交易日收盘调仓；
- 买卖滑点均为 5 bps；
- 佣金按成交金额 0.03%、单笔最低 5 元模拟，这只是可修改的券商费用假设；
- 卖出印花税在 2023-08-28 前按 0.1%、此后按 0.05%模拟；
- 默认不使用同一开盘卖出所得继续买入，与在线确定性风控的保守现金语义一致；
- 使用复权因子构造以回测起点归一化的总回报价格，避免分红除权造成虚假亏损；
- 只有 `decision_quality=healthy` 的历史决策会保存到
  `data/cache/backtest_decisions/`；`degraded`、`failed` 或安全持有不会写入
  长期缓存；空决策和非法质量标记同样不可缓存，避免把不完整研究反复当作正常策略
  结果；
- 缓存 key 包含图、Prompt、输出 Schema、质量策略的显式版本，以及结构化输出模式、
  重试次数、Analyst/Risk 法定人数、运行模式和组合约束。上述逻辑或配置变化会使
  旧决策自动失效；旧格式缓存也会被拒绝读取；
- 默认最多允许 24 个决策时点；更长或更高频的回测必须显式提高
  `--max-decisions`，防止意外产生大量模型费用。

印花税切换日期依据财政部、税务总局自 2023-08-28 起减半征收的
[官方公告](https://shanxi.chinatax.gov.cn/web/detail/sx-11400-545-1780448)；
复权价格使用 Tushare 的
[复权因子](https://tushare.pro/document/2?doc_id=28)。

默认结果目录为 `data/backtests/<run_id>/`：

```text
summary.json       # 收益、风险、费用、调用审计和回测结果质量
equity_curve.csv   # 每个交易日的净值、现金和持仓市值
trades.csv         # 决策日、成交日、成交价、股数、滑点和费用
decisions.json     # 每次质量、覆盖率、阶段健康度、原始 Agent 元数据和风控结果
```

`summary.json` 中的 `result_quality_status` 只有在全部决策均为 `healthy` 时才是
`valid`。只要出现一次 `degraded` 或 `failed`，状态就是 `invalid`，并产生
`BACKTEST_RESULT_INVALID` 强告警；这时收益率包含降级或安全持有行为，不能解释为
完整多 Agent 策略的有效表现。调用统计会区分：

- `llm_provider_attempts`：本次实际发出的供应商尝试；
- `llm_cached_original_provider_attempts`：缓存决策最初生成时的供应商尝试；
- `llm_validated_outputs` 与 `llm_output_repair_attempts`：本次成功结构化节点数和
  输出修复次数；
- `healthy/degraded/failed_decision_count`：三类决策数量与对应比例。

完整 20 股股票池做一年月频多 Agent 回测，正常路径大约产生 12 次决策、
132 次模型调用；建议先用 2–3 只股票和 2–3 个月验证凭据、Tushare 权限和成本。

必须正确理解以下偏差：

- `config/universe.yaml` 是当前固定股票池，回测过去会有幸存者偏差和选股偏差；
- 新闻、财务指标必须在历史 `as_of` 前可见；取不到时会触发现有数据质量风控，
  可能导致不交易；
- 当前未模拟涨跌停排队、停牌后的可成交性、成交量冲击、退市和现金分红明细；
- 总回报复权价格是绩效代理，不是逐笔券商对账单；
- LLM 即使温度为零也不保证所有供应商长期完全确定，因此应保留 decision cache、
  模型名和 `decisions.json`。

## 启动

```bash
cd /Users/test/Documents/Personal/ashare_portfolio_backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：

- Swagger UI：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>
- Liveness：<http://127.0.0.1:8000/health/live>
- Readiness：<http://127.0.0.1:8000/health/ready>

## API

### 创建组合

```bash
curl -X POST http://127.0.0.1:8000/api/v1/portfolios \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "我的A股组合",
    "cash": "30000",
    "positions": [
      {
        "symbol": "600519.SH",
        "shares": 100,
        "available_shares": 100,
        "average_cost": "1480.50",
        "holding_days": 120
      }
    ]
  }'
```

`available_shares` 是当日实际可卖数量；不填写时默认为全部持仓可卖。

### 查询或更新组合

```text
GET /api/v1/portfolios/{portfolio_id}
PUT /api/v1/portfolios/{portfolio_id}
```

### 创建决策任务

```bash
curl -X POST http://127.0.0.1:8000/api/v1/decision-runs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: portfolio-20260722-after-close' \
  -d '{
    "portfolio_id": "pf_xxx",
    "mode": "rebalance"
  }'
```

接口返回 HTTP 202 和 `run_id`，随后查询：

```text
GET /api/v1/decision-runs/{run_id}
```

状态包括：

```text
pending
fetching_data
building_features
calling_llm
validating
completed
degraded
failed
```

模式说明：

- `holdings_only`：只分析已有持仓；
- `rebalance`：分析 `config/universe.yaml` 与已有持仓的并集。

如果配置了 `BACKEND_API_KEY`，所有 `/api/v1/*` 请求都需要：

```text
X-API-Key: your-backend-key
```

## 数据与时点语义

服务使用“最近已经完成的 A 股交易日数据，为下一个交易日生成建议”。当请求发生在
配置的收盘数据可用时间之前，使用前一个已完成交易日。

- 估值和最终股数使用 `data_date` 当日未复权收盘价；
- 技术收益和均线使用截至 `data_date` 的复权因子，归一到该日，不读取未来因子；
- Tushare `daily.vol` 从手转换为股；`amount` 从千元转换为人民币元；
- Tushare 市值从万元转换为人民币元；百分数字段转换为 0–1 ratio；
- 财务指标严格过滤 `ann_date <= data_date`；
- 新闻严格限制在 `[as_of - lookback, as_of]`；
- 任意价格、新闻或基本面质量警告都会保留在结果中，并阻止模型建议新增或加仓。

显式 `as_of` 只允许与当前服务时间相差 `MAX_AS_OF_SKEW_MINUTES`，此接口不是历史回测
入口。

## 缓存与离线模式

缓存目录由 `CACHE_PATH` 控制，默认是 `./data/cache`。缓存采用原子文件替换，并按
股票、日期和数据类型隔离。

`DATA_MODE=offline_only` 时只读取本项目缓存：缺少精确交易日价格或交易日历会让相应
股票安全降级或任务失败，不会猜测节假日，也不会使用任意默认价格。

## 测试

```bash
cd /Users/test/Documents/Personal/ashare_portfolio_backend
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
```

测试不调用真实 Tushare 或 LLM，覆盖：

- Portfolio、Universe 和 Decision Run API；
- SQLite、幂等冲突、任务队列和子进程 Worker；
- 固定股票池与已有持仓合并；
- 100 股整手、T+1、现金和仓位风控；
- 交易日历、收盘时点、本地缓存和历史 point-in-time 读取；
- Tushare 行情单位、复权因子、基本面公告日期和新闻 cutoff；
- 特征上下文、strict JSON Schema、兼容降级和非法模型输出拒绝；
- 历史收盘决策、下一交易日开盘成交、交易费用、净值和调仓计划；
- 从隔离工作目录导入应用，以及依赖元数据独立性。

## 运行限制与安全

- 当前是单租户 MVP；`BACKEND_API_KEY` 是服务级密钥，不是用户身份系统；
- 默认使用一个任务 Worker。多实例生产部署应改用 Redis + RQ/Celery/Arq；
- Tushare 接口权限不足会显示为数据质量告警；
- 子进程硬超时可以终止整次任务，但不能保证第三方请求已经在上游取消；
- 正式环境需要 HTTPS、用户认证、数据库加密、日志脱敏、访问审计和密钥管理；
- 不要把 `.env`、Tushare Token、LLM Key、完整 Prompt 或持仓数据提交到 Git；
- 结果是研究建议，不构成自动成交指令。



export PYTHONPATH="$PWD/ashare_portfolio_backend"

python -m app.backtest.cli \
  --start 2025-01-01 \
  --end 2025-03-31 \
  --engine portfolio_multi_agent \
  --rebalance monthly \
  --max-decisions 6 \
  --initial-cash 1000000 \
  --symbols 600519.SH,300750.SZ,601318.SH


python -m app.backtest.cli `
  --project-root . `
  --start 2023-01-01 `
  --end 2023-12-30 `
  --engine portfolio_multi_agent `
  --rebalance weekly `
  --max-decisions 200 `
  --initial-cash 1000000


$env:PYTHONPATH = (Resolve-Path .\ashare_portfolio_backend).Path
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000


python -m streamlit run ashare_portfolio_frontend\app.py
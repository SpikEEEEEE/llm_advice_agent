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
- 优先使用 strict JSON Schema，不支持时显式降级到严格 `json_object`；
- 可选的股票池原生多 Agent 决策图：全池横向研究、多空辩论、组合构建、三路风险审查；
- 多 Agent 不按股票循环调用，而是共享同一个股票池状态并最终只输出一个组合权重向量；
- 模型输出必须覆盖所有有效股票，并通过严格字段、数值和 action/target 语义校验；
- A 股买入 100 股整手、T+1 可卖数量、现金、最低现金比例、单股仓位和最大持仓数限制；
- 缺价时强制 `hold`，持仓估值不完整时冻结全部新买入；
- 数据质量异常时禁止新增或加仓，但仍允许确定性风控后的减仓；
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
```

也可以用 `LLM_API_KEY` 替代 `OPENAI_API_KEY`，并将 `LLM_BASE_URL` 指向兼容
Chat Completions 的服务。

默认仍使用原来的单次 LLM 引擎。启用股票池多 Agent 图：

```dotenv
DECISION_ENGINE=portfolio_multi_agent
MULTI_AGENT_SHORTLIST_SIZE=8
MULTI_AGENT_PARALLELISM=3
MULTI_AGENT_MAX_CALLS=32
MULTI_AGENT_SEMANTIC_RETRIES=1
```

`MULTI_AGENT_SHORTLIST_SIZE` 只限制进入深度辩论的非持仓候选；已有持仓一定进入
shortlist，因此不会因为预筛选而失去减仓或清仓判断。三个分析权重可以通过
`MULTI_AGENT_TECHNICAL_WEIGHT`、`MULTI_AGENT_FUNDAMENTAL_WEIGHT` 和
`MULTI_AGENT_NEWS_WEIGHT` 调整。

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

三个 Analyst 每次都看到完整有效股票池并做横向比较；后续角色共享前序结构化产物，
而不是让每只股票各自形成互不知情的结论。Portfolio Manager 的输出必须覆盖全部
shortlist，股票权重与现金权重之和必须为 100%，并满足最低现金、单股上限和最大
持仓数。语义校验失败时会携带校验错误进行有限次数修复；总模型请求数还受
`MULTI_AGENT_MAX_CALLS` 硬限制。

正常路径基础调用数为 11 次：3 个 Analyst、Bull、Bear、Research Manager、
Portfolio Trader、3 个 Risk Reviewer 和 Portfolio Manager。最终结果的
`llm_meta.agent_artifacts` 会保留各阶段结构化结论，`agent_trace` 会保留调用和
token usage，便于审计。它们是当前任务内的显式状态，不会自动读取上一次交易任务；
跨任务记忆仍需另行设计绩效反馈或交易日志输入。

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
- 从隔离工作目录导入应用，以及依赖元数据独立性。

## 运行限制与安全

- 当前是单租户 MVP；`BACKEND_API_KEY` 是服务级密钥，不是用户身份系统；
- 默认使用一个任务 Worker。多实例生产部署应改用 Redis + RQ/Celery/Arq；
- Tushare 接口权限不足会显示为数据质量告警；
- 子进程硬超时可以终止整次任务，但不能保证第三方请求已经在上游取消；
- 正式环境需要 HTTPS、用户认证、数据库加密、日志脱敏、访问审计和密钥管理；
- 不要把 `.env`、Tushare Token、LLM Key、完整 Prompt 或持仓数据提交到 Git；
- 结果是研究建议，不构成自动成交指令。

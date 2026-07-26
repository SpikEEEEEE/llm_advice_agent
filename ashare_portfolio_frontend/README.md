# A-share Portfolio Advisor Streamlit Frontend

这是现有 FastAPI 后端的独立 Streamlit 操作台，支持：

- 输入本次股票池、现金和当前持仓；
- 提交异步投资建议任务并自动轮询；
- 展示最终目标股数和每条确定性风控调整；
- 查看三路 Analyst、Shortlist、Bull/Bear 辩论、Research Manager；
- 查看 Trader、三路 Risk Reviewer 和 Portfolio Manager；
- 查看调用轨迹、阶段健康度、市场快照及完整 JSON；
- 从后端 SQLite 读取历史决策档案。

## 启动

先在仓库根目录启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir ashare_portfolio_backend --host 127.0.0.1 --port 8000
```

安装前端依赖并启动 Streamlit：

```powershell
.\.venv\Scripts\pip.exe install -r ashare_portfolio_frontend\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run ashare_portfolio_frontend\app.py
```

默认连接 `http://127.0.0.1:8000`。也可以设置：

```dotenv
ADVISOR_API_URL=http://127.0.0.1:8000
BACKEND_API_KEY=
```

API Key 只保存在当前 Streamlit 页面会话中。前端不会连接券商，也不会提交订单。

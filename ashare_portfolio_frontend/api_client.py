from __future__ import annotations

from typing import Any

import httpx


class BackendError(RuntimeError):
    """A safe, user-facing backend communication error."""


class AdvisorApi:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 15.0,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("Backend URL cannot be empty")
        self.base_url = normalized
        self.api_key = api_key.strip() if api_key else None
        self.timeout_seconds = timeout_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_headers = {**self._headers, **(headers or {})}
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                params=params,
                headers=request_headers,
                timeout=self.timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise BackendError(
                "无法连接投资建议后端，请确认 FastAPI 服务地址和运行状态。"
            ) from exc

        if response.is_success:
            if response.status_code == 204:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise BackendError("后端返回了无法解析的响应。") from exc

        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except ValueError:
            detail = response.text.strip() or response.reason_phrase
        raise BackendError(f"后端请求失败（HTTP {response.status_code}）：{detail}")

    def live(self) -> dict[str, Any]:
        return self._request("GET", "/health/live")

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/health/ready")

    def universe(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/universe")

    def create_portfolio(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/portfolios", json=payload)

    def create_decision_run(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/decision-runs",
            json=payload,
            headers={"Idempotency-Key": idempotency_key},
        )

    def decision_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/decision-runs/{run_id}")

    def decision_runs(
        self,
        *,
        limit: int = 50,
        portfolio_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if portfolio_id:
            params["portfolio_id"] = portfolio_id
        payload = self._request("GET", "/api/v1/decision-runs", params=params)
        return payload if isinstance(payload, list) else []

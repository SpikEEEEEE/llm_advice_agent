from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse

from app.api.dependencies import get_container, require_api_key
from app.container import AppContainer
from app.repositories.sqlite import IdempotencyConflictError
from app.schemas.api import (
    DecisionRunCreateRequest,
    DecisionRunResponse,
    HealthResponse,
    PortfolioResponse,
    PortfolioWriteRequest,
    UniverseResponse,
)
from app.services.task_runner import TaskQueueFullError


router = APIRouter()
protected = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
Container = Annotated[AppContainer, Depends(get_container)]


def _run_response(payload: dict[str, Any]) -> DecisionRunResponse:
    return DecisionRunResponse.model_validate(
        {
            key: payload.get(key)
            for key in (
                "id",
                "portfolio_id",
                "portfolio_version",
                "status",
                "mode",
                "as_of",
                "universe_version",
                "universe",
                "result",
                "error_code",
                "error_message",
                "created_at",
                "started_at",
                "completed_at",
            )
        }
    )


def _request_fingerprint(
    *,
    portfolio: dict[str, Any],
    mode: str,
    requested_as_of: str | None,
    universe_version: str,
    universe: list[str],
) -> str:
    logical_request = {
        "portfolio_id": portfolio["id"],
        "portfolio_version": portfolio["version"],
        "mode": mode,
        # A server-generated current timestamp is intentionally excluded so a
        # retry of the same request remains idempotent.
        "requested_as_of": requested_as_of,
        "universe_version": universe_version,
        "universe": universe,
    }
    encoded = json.dumps(
        logical_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@router.get("/health/live", response_model=HealthResponse)
def health_live() -> HealthResponse:
    return HealthResponse(service="ashare-portfolio-advisor")


@router.get("/health/ready")
def health_ready(container: Container) -> Response:
    checks: dict[str, bool] = {
        "database": container.repository.ping(),
        "universe": False,
        "tushare_credentials": bool(container.settings.tushare_token)
        or container.settings.data_mode == "offline_only",
        "llm_credentials": bool(container.settings.llm_api_key),
    }
    try:
        container.settings.load_universe()
        checks["universe"] = True
    except Exception:
        pass
    ready = all(checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@protected.get("/universe", response_model=UniverseResponse)
def get_universe(container: Container) -> UniverseResponse:
    version, symbols = container.settings.load_universe()
    return UniverseResponse(version=version, symbols=symbols)


@protected.post(
    "/portfolios",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_portfolio(
    request: PortfolioWriteRequest,
    container: Container,
) -> PortfolioResponse:
    payload = request.model_dump(mode="json")
    portfolio = container.repository.create_portfolio(payload)
    return PortfolioResponse.model_validate(portfolio)


@protected.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(portfolio_id: str, container: Container) -> PortfolioResponse:
    portfolio = container.repository.get_portfolio(portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return PortfolioResponse.model_validate(portfolio)


@protected.put("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def update_portfolio(
    portfolio_id: str,
    request: PortfolioWriteRequest,
    container: Container,
) -> PortfolioResponse:
    portfolio = container.repository.update_portfolio(
        portfolio_id,
        request.model_dump(mode="json"),
    )
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return PortfolioResponse.model_validate(portfolio)


@protected.post(
    "/decision-runs",
    response_model=DecisionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_decision_run(
    request: DecisionRunCreateRequest,
    container: Container,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DecisionRunResponse:
    if idempotency_key and len(idempotency_key) > 200:
        raise HTTPException(status_code=400, detail="Idempotency-Key is too long")
    portfolio = container.repository.get_portfolio(request.portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if request.universe is None:
        version, universe = container.settings.load_universe()
    else:
        universe = request.universe
        universe_digest = hashlib.sha256(
            ",".join(universe).encode("utf-8")
        ).hexdigest()[:12]
        version = f"custom-{universe_digest}"
    timezone = ZoneInfo("Asia/Shanghai")
    as_of = request.as_of or datetime.now(timezone)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone)
    else:
        as_of = as_of.astimezone(timezone)
    now = datetime.now(timezone)
    max_skew = timedelta(minutes=container.settings.max_as_of_skew_minutes)
    if request.as_of is not None and abs(now - as_of) > max_skew:
        raise HTTPException(
            status_code=422,
            detail=(
                "as_of must describe the current online decision time; historical replay "
                "is not supported by this endpoint"
            ),
        )
    requested_as_of = as_of.isoformat() if request.as_of is not None else None
    try:
        run, created = container.repository.create_decision_run(
            portfolio=portfolio,
            mode=request.mode,
            as_of=as_of.isoformat(),
            universe_version=version,
            universe=universe,
            idempotency_key=idempotency_key,
            request_fingerprint=_request_fingerprint(
                portfolio=portfolio,
                mode=request.mode,
                requested_as_of=requested_as_of,
                universe_version=version,
                universe=universe,
            ),
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        try:
            container.task_runner.submit(run["id"])
        except TaskQueueFullError as exc:
            container.repository.fail_decision_run(
                run["id"],
                code="TASK_QUEUE_FULL",
                message="Decision task queue was full at admission time",
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Decision task queue is full; retry with a new key later",
                    "run_id": run["id"],
                },
            ) from exc
        except Exception as exc:
            container.repository.fail_decision_run(
                run["id"],
                code="TASK_SUBMISSION_FAILED",
                message=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Decision task could not be submitted",
            ) from exc
        run = container.repository.get_decision_run(run["id"]) or run
    return _run_response(run)


@protected.get("/decision-runs", response_model=list[DecisionRunResponse])
def list_decision_runs(
    container: Container,
    portfolio_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[DecisionRunResponse]:
    return [
        _run_response(run)
        for run in container.repository.list_decision_runs(
            portfolio_id=portfolio_id,
            limit=limit,
        )
    ]


@protected.get("/decision-runs/{run_id}", response_model=DecisionRunResponse)
def get_decision_run(run_id: str, container: Container) -> DecisionRunResponse:
    run = container.repository.get_decision_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Decision run not found")
    return _run_response(run)


router.include_router(protected)

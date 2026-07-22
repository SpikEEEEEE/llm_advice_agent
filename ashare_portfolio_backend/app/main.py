from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.container import AppContainer


def create_app(container: AppContainer | None = None) -> FastAPI:
    effective_container = container or AppContainer.build()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        effective_container.repository.initialize()
        app.state.container = effective_container
        yield
        effective_container.task_runner.shutdown()

    application = FastAPI(
        title="Independent A-share Portfolio Advisor",
        version="0.2.0",
        description=(
            "Manual portfolio ingestion and asynchronous AI-assisted decisions. "
            "The service returns advisory results and never submits broker orders."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = create_app()

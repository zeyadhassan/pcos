"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from percos import __version__
from percos.config import get_settings
from percos.logging import setup_logging
from percos.stores.database import get_engine
from percos.stores.tables import Base


async def _maintenance_loop(interval_minutes: int) -> None:
    """GAP-M5: Background coroutine that runs maintenance at a fixed interval."""
    from percos.api.deps import get_brain_standalone
    from percos.logging import get_logger

    log = get_logger("maintenance_scheduler")
    interval_seconds = interval_minutes * 60
    log.info("maintenance_scheduler_started", interval_minutes=interval_minutes)

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            async with get_brain_standalone() as brain:
                result = await brain.run_maintenance()
                log.info("scheduled_maintenance_complete",
                         stale=result.get("stale_detected", 0),
                         contradictions=len(result.get("contradictions", [])))
        except asyncio.CancelledError:
            log.info("maintenance_scheduler_stopped")
            break
        except Exception as exc:
            log.error("scheduled_maintenance_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings.log_level.value)

    # Create tables
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # GAP-M5: Start periodic maintenance if configured
    maintenance_task = None
    if settings.maintenance_interval_minutes > 0:
        maintenance_task = asyncio.create_task(
            _maintenance_loop(settings.maintenance_interval_minutes)
        )

    yield

    # Shutdown
    if maintenance_task is not None:
        maintenance_task.cancel()
        try:
            await maintenance_task
        except asyncio.CancelledError:
            pass

    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Ontology-Governed Knowledge Base Framework",
        description="Domain-Agnostic, Adaptive, Self-Developing Knowledge Base Framework",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Bearer token auth (§11)
    from percos.api.auth import BearerAuthMiddleware
    app.add_middleware(BearerAuthMiddleware)

    # Routes
    from percos.api.routes import router
    app.include_router(router, prefix="/api/v1")

    # Memory Control Panel UI (§12)
    from percos.api.panel import panel_router
    app.include_router(panel_router)

    return app


app = create_app()

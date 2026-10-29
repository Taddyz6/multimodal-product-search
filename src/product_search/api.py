"""FastAPI admission and serialization; all inference stays in SearchService."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import logging
import os
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator
from .errors import QueryProcessor, ServiceError, Unavailable
from .dispatch import Dispatcher, QueueConfig


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=2048)
    top_k: StrictInt = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def text(cls, v):
        return QueryProcessor().validate(v, 10)


class ProductResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str
    product_locale: str
    title: str | None
    image_url: str | None
    score: float = Field(allow_inf_nan=False)
    rank: int


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    top_k: int
    candidate_budget: int
    mode: str
    timings_ms: dict[str, float | None]
    results: list[ProductResult]


async def with_disconnect(request, operation):
    task = asyncio.create_task(operation)

    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        raise HTTPException(499, "Client closed the request")
    finally:
        if not task.done():
            task.cancel()
        watcher.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


class ServerTiming:
    """Pure ASGI middleware preserves request receive/disconnect semantics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()

        async def timed_send(message):
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": list(message.get("headers", []))
                    + [(b"x-server-ms", f"{(time.perf_counter() - start) * 1000:.6f}".encode())],
                }
            await send(message)

        await self.app(scope, receive, timed_send)


def create_app(service_factory, *, queue_config=None):
    config = queue_config or QueueConfig()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        began = time.perf_counter()
        try:
            app.state.service = service_factory()
        except Exception:
            logging.exception("Search assets could not be loaded")
        app.state.startup_seconds = time.perf_counter() - began
        try:
            async with Dispatcher(config) as dispatcher:
                app.state.dispatcher = dispatcher
                yield
            # Dispatcher drains the active CPU job before the encoders are closed.
        finally:
            if app.state.service is not None:
                app.state.service.close()
            app.state.service = None

    app = FastAPI(title="Product Search", version="1.0", lifespan=lifespan)
    app.add_middleware(ServerTiming)

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        headers = {"Retry-After": "1"} if exc.http_status == 429 else None
        return JSONResponse(
            {"error": {"code": exc.code, "message": str(exc)}},
            status_code=exc.http_status,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return JSONResponse(
            {"error": {"code": "invalid_input", "message": "Check the query and top_k fields."}},
            status_code=422,
        )

    def service():
        if app.state.service is None:
            raise Unavailable("Search assets are unavailable; inspect the startup logs.")
        return app.state.service

    def health():
        import psutil

        current = app.state.service
        state = (
            current.status()
            if current
            else {"ready": False, "components": {}, "reason": "startup_failed"}
        )
        dispatcher = app.state.dispatcher.snapshot()
        ready = state["ready"] and dispatcher["accepting"]
        return {
            **state,
            "ready": ready,
            "status": "ready" if ready else "unavailable",
            "dispatch": dispatcher,
            "queue_config": asdict(config),
            "startup_seconds": app.state.startup_seconds,
            "rss_bytes": psutil.Process().memory_info().rss,
        }

    @app.get("/health")
    async def liveness():
        return health()

    @app.get("/ready")
    async def readiness():
        result = health()
        return JSONResponse(result, status_code=200 if result["ready"] else 503)

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        current = service()
        if not current.status()["ready"]:
            raise Unavailable(current.status()["reason"])
        try:
            result = await with_disconnect(
                request,
                app.state.dispatcher.submit(
                    ("search", body.query, body.top_k),
                    lambda: current.search(body.query, body.top_k),
                ),
            )
            began = time.perf_counter()
            result.payload["timings_ms"].update(
                queue_wait=result.queue_wait_ms,
                scheduler_total=result.scheduler_ms,
                shared_job_queue_wait=result.shared_job_queue_ms,
            )
            payload = SearchResponse.model_validate(result.payload).model_dump(mode="json")
            content = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
            return Response(
                content,
                media_type="application/json",
                headers={
                    "X-Serialization-Ms": f"{(time.perf_counter() - began) * 1000:.6f}",
                    "X-Search-Mode": payload["mode"],
                    "X-Request-Coalesced": str(result.coalesced).lower(),
                    "X-Queue-Wait-Ms": f"{result.queue_wait_ms:.6f}",
                    "X-Scheduler-Ms": f"{result.scheduler_ms:.6f}",
                },
            )
        except (ServiceError, HTTPException):
            raise
        except Exception:
            logging.exception("Search failed")
            return JSONResponse(
                {
                    "error": {
                        "code": "inference_failed",
                        "message": "Search failed; inspect the service logs.",
                    }
                },
                status_code=500,
            )

    return app


def application():
    # One Uvicorn worker owns one core. Multiple workers duplicate the catalog.
    from .service import load_service

    config = os.environ.get("SEARCH_CONFIG", "configs/service.json")
    return create_app(lambda: load_service(config))

"""REST API (Starlette). Ishga tushirish:  uvicorn lexrag.api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import json
import logging
from functools import partial

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .pipeline import Engine
from .service import Service, err

log = logging.getLogger("lexrag.api")


class UJSON(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def create_app(service: Service | None = None) -> Starlette:
    state: dict = {"svc": service}

    def svc() -> Service:
        if state["svc"] is None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            state["svc"] = Service(Engine())
        return state["svc"]

    async def _body(request: Request):
        raw = await request.body()
        if not raw:
            return None, None
        try:
            return json.loads(raw.decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return None, UJSON(err("INVALID_JSON", f"JSON o'qilmadi: {e}"), status_code=400)

    async def _guard(request: Request, rate: bool = True):
        s = svc()
        a = s.auth(request.headers.get("authorization"))
        if a:
            return UJSON(a[1], status_code=a[0])
        if rate:
            lim = s.limit(request.headers.get("authorization"))
            if lim:
                return UJSON(lim[1], status_code=lim[0], headers=lim[2])
        return None

    async def post_batch(request: Request):
        g = await _guard(request)
        if g:
            return g
        body, e = await _body(request)
        if e:
            return e
        async_mode = request.query_params.get("mode") == "async" or request.query_params.get("async") in ("1", "true")
        code, payload = await run_in_threadpool(partial(svc().post_batch, body, async_mode))
        return UJSON(payload, status_code=code)

    async def get_batch(request: Request):
        g = await _guard(request, rate=False)
        if g:
            return g
        code, payload = await run_in_threadpool(svc().get_batch, request.path_params["batch_id"])
        return UJSON(payload, status_code=code)

    async def post_query(request: Request):
        g = await _guard(request)
        if g:
            return g
        body, e = await _body(request)
        if e:
            return e
        code, payload = await run_in_threadpool(svc().query, body)
        return UJSON(payload, status_code=code)

    async def meta(request: Request):
        code, payload = svc().meta()
        return UJSON(payload, status_code=code)

    async def health(request: Request):
        code, payload = svc().health()
        return UJSON(payload, status_code=code)

    async def root(request: Request):
        return UJSON({"service": "LexRAG (lex.uz RAG API)", "endpoints": ["POST /v1/rag/batch", "GET /v1/rag/batch/{batch_id}", "POST /v1/rag/query", "GET /v1/meta", "GET /v1/health"]})

    async def not_found(request: Request, exc):
        return UJSON(err("NOT_FOUND", f"{request.method} {request.url.path} topilmadi"), status_code=404)

    async def method_not_allowed(request: Request, exc):
        return UJSON(err("METHOD_NOT_ALLOWED", f"{request.method} ruxsat etilmagan"), status_code=405)

    async def server_error(request: Request, exc):
        log.exception("server xatosi")
        return UJSON(err("INTERNAL", "Ichki xato", True), status_code=500)

    routes = [
        Route("/", root),
        Route("/v1/rag/batch", post_batch, methods=["POST"]),
        Route("/v1/rag/batch/{batch_id:str}", get_batch, methods=["GET"]),
        Route("/v1/rag/query", post_query, methods=["POST"]),
        Route("/v1/meta", meta, methods=["GET"]),
        Route("/v1/health", health, methods=["GET"]),
    ]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])],
        exception_handlers={404: not_found, 405: method_not_allowed, 500: server_error},
    )
    app.state.get_service = svc
    return app


app = create_app()

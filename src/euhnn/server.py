"""Authenticated loopback API and bundled no-CDN document workbench.

The server deliberately has no remote URL-fetch or arbitrary local-path endpoint.
Only explicitly uploaded bytes are indexed. Authentication and same-origin/Host
checks apply before a request body is consumed. This is a single-user local tool,
not a hardened multi-tenant public document hosting service.
"""

from __future__ import annotations

from hmac import compare_digest
from pathlib import Path
import asyncio
import re
import secrets
import sqlite3
import time
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .documents import MAX_FILE_BYTES
from .hologram import MAX_HOLOGRAM_BYTES, preview
from .index import HolographicIndex
from .sample import load_demo
from .schema import EUHNNError, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchRequest(StrictModel):
    query: str = Field(max_length=4096)
    top_k: int = Field(default=5, ge=1, le=100)
    mode: str = "hybrid"
    source: str | None = Field(default=None, max_length=4096)
    phrase: bool = False


class TextRequest(StrictModel):
    text: str = Field(min_length=1, max_length=1_048_576)
    source: str = Field(min_length=1, max_length=1024)
    title: str | None = Field(default=None, max_length=4096)


class TeachingRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4096)
    chunk_id: str = Field(min_length=1, max_length=64)


class QueryRequest(StrictModel):
    query: str = Field(max_length=4096)


class LocalBoundary:
    """An ASGI boundary with bounded bodies and an absolute input deadline."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers: dict[bytes, list[bytes]] = {}
        for key, value in scope.get("headers", []):
            headers.setdefault(key.lower(), []).append(value)

        async def reject(status: int, error: str) -> None:
            response = JSONResponse({"error": error}, status_code=status, headers={"Cache-Control": "no-store"})
            await response(scope, receive, send)

        hosts = headers.get(b"host", [])
        if len(hosts) != 1:
            await reject(403, "invalid_host")
            return
        try:
            authority = hosts[0].decode("ascii").lower()
            host = urlsplit("http://" + authority)
            if (
                host.hostname not in {"127.0.0.1", "localhost", "::1"}
                or host.username is not None
                or host.password is not None
                or host.path
                or host.query
                or host.fragment
                or not 1 <= (host.port or 80) <= 65535
            ):
                raise ValueError("Invalid authority")
        except (ValueError, UnicodeError):
            await reject(403, "invalid_host")
            return
        origins = headers.get(b"origin", [])
        expected = (scope.get("scheme", "http") + "://" + authority).encode()
        if len(origins) > 1 or (origins and origins[0].lower() != expected):
            await reject(403, "cross_origin_denied")
            return
        if headers.get(b"sec-fetch-site") == [b"cross-site"]:
            await reject(403, "cross_site_denied")
            return
        path = scope["path"]
        api = path.startswith("/api/")
        if api:
            auth = headers.get(b"authorization", [])
            if len(auth) != 1 or not compare_digest(auth[0], ("Bearer " + self.token).encode()):
                await reject(401, "unauthorized")
                return
        elif scope["method"] not in ("GET", "HEAD"):
            await reject(405, "method_not_allowed")
            return
        body = bytearray()
        if api and scope["method"] in ("POST", "PUT", "PATCH"):
            limit = (
                MAX_HOLOGRAM_BYTES
                if path == "/api/memory/import"
                else (MAX_FILE_BYTES if path.startswith("/api/upload/") else 2 * 1024 * 1024)
            )
            lengths = headers.get(b"content-length", [])
            if len(lengths) > 1 or headers.get(b"content-encoding"):
                await reject(400, "unsupported_framing")
                return
            if lengths:
                try:
                    if not lengths[0].isdigit() or int(lengths[0]) > limit:
                        raise ValueError("Invalid length")
                except ValueError:
                    await reject(413, "request_too_large")
                    return
            try:
                async with asyncio.timeout(60):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > limit:
                            await reject(413, "request_too_large")
                            return
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await reject(408, "request_timeout")
                return
            delivered = False

            async def replay() -> dict:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            input_receive = replay
        else:
            input_receive = receive

        async def protected_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"content-security-policy",
                            b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                        ),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
            await send(message)

        await self.app(scope, input_receive, protected_send)


def create_app(index: HolographicIndex, *, token: str | None = None) -> FastAPI:
    """Build the local API; the caller retains responsibility for index.close()."""
    token = token or secrets.token_urlsafe(32)
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
        raise ValidationError("The API token must be 32-256 randomly generated URL-safe characters.")
    app = FastAPI(title="EUHNN local workbench", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalBoundary, token=token)
    app.state.connection_token = token
    app.state.index = index
    static = Path(__file__).with_name("web")

    @app.exception_handler(EUHNNError)
    async def application_error(request: Request, error: EUHNNError) -> JSONResponse:
        return JSONResponse({"error": type(error).__name__, "message": str(error)}, status_code=400)

    @app.exception_handler(sqlite3.DatabaseError)
    async def storage_error(request: Request, error: sqlite3.DatabaseError) -> JSONResponse:
        return JSONResponse(
            {
                "error": "storage_error",
                "message": "The local index could not complete this operation. Verify its integrity and free disk space.",
            },
            status_code=500,
        )

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(static / "index.html", media_type="text/html")

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(static / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def stylesheet() -> FileResponse:
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/status")
    def status() -> dict:
        return {"version": __version__, **index.stats()}

    @app.get("/api/documents")
    def documents(limit: int = 100, offset: int = 0) -> dict:
        return {"documents": index.documents(limit=limit, offset=offset)}

    @app.post("/api/text")
    def text(payload: TextRequest) -> dict:
        return index.ingest_text(payload.text, source=payload.source, title=payload.title)

    @app.put("/api/upload/{filename}")
    async def upload(filename: str, request: Request) -> dict:
        if (
            filename != Path(filename).name
            or "/" in filename
            or "\\" in filename
            or not filename.strip()
            or len(filename) > 255
        ):
            raise ValidationError("Upload a file with a simple, non-empty filename.")
        data = await request.body()
        return await run_in_threadpool(index.ingest_bytes, data, filename=filename, source="upload/" + filename)

    @app.delete("/api/documents/{document_id}")
    def delete(document_id: str) -> dict:
        return {"deleted": index.delete_document(document_id)}

    @app.post("/api/search")
    def search(payload: SearchRequest) -> dict:
        started = time.perf_counter()
        hits = index.search(**payload.model_dump())
        return {
            "hits": [hit.to_dict() for hit in hits],
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "answer_type": "retrieved source passages; no generated factual answer",
        }

    @app.post("/api/learn")
    def learn(payload: TeachingRequest) -> dict:
        return index.learn(payload.query, payload.chunk_id)

    @app.post("/api/scene")
    def scene(payload: QueryRequest) -> dict:
        return index.encoder.scene(payload.query)

    @app.post("/api/demo")
    def demo() -> dict:
        return {"documents": load_demo(index)}

    @app.get("/api/memory/export")
    def export() -> Response:
        return Response(
            index.export_hologram(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="euhnn-memory.holo"'},
        )

    @app.put("/api/memory/import")
    async def import_memory(request: Request) -> dict:
        return await run_in_threadpool(index.import_hologram, await request.body())

    @app.get("/api/memory/preview")
    def memory_preview() -> dict:
        return preview(index.export_hologram())

    @app.get("/api/verify")
    def verify() -> dict:
        return index.verify()

    return app

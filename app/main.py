"""FastAPI service.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.container import Container, build_container
from app.logging_setup import request_id_var, setup_logging
from app.rag.loaders import SUPPORTED_EXT
from app.rag.pipeline import PipelineError
from app.schemas import ChatRequest, ChatResponse, DocumentInfo, IngestJob
from app.security import RateLimiter, make_guard

log = logging.getLogger("rag.api")


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    s = settings or (container.settings if container else get_settings())
    setup_logging(s.log_level, s.log_json)
    state: dict[str, Container] = {}
    if container:
        state["c"] = container

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if "c" not in state:  # heavy models (embeddings, FlashRank) stay lazy -> instant start-up
            state["c"] = await asyncio.to_thread(build_container, s)
        log.info("API ready (env=%s)", s.app_env)
        yield

    app = FastAPI(title="Enterprise RAG API", version="1.0.0", lifespan=lifespan,
                  docs_url=None if s.is_prod else "/docs", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_list, allow_methods=["*"], allow_headers=["*"])

    def c() -> Container:
        return state["c"]

    auth = make_guard(s, RateLimiter(s.rate_limit_per_minute))
    protected = [Depends(auth)]

    # ── middleware: request id, timing, security headers ────────────────────────────────────────
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path not in ("/health", "/ready"):
            log.info("%s %s -> %s (%d ms)", request.method, request.url.path, response.status_code,
                     (time.perf_counter() - t) * 1000)
        return response

    @app.exception_handler(PipelineError)
    async def pipeline_error(_: Request, exc: PipelineError):
        return JSONResponse({"detail": str(exc), "providers": exc.detail}, status_code=exc.status)

    # ── health ──────────────────────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready():
        if "c" not in state:
            raise HTTPException(503, "starting")
        ok = await asyncio.to_thread(c().kb.healthy)
        if not ok:
            raise HTTPException(503, "vector store unreachable")
        return {"status": "ready", "providers": len(c().gateway.providers)}

    @app.get("/v1/providers", dependencies=protected, tags=["ops"])
    async def providers():
        cont = c()
        return {
            "llm": cont.gateway.status(),
            "reranker": "active" if cont.reranker.active else ("fallback" if cont.reranker._failed else "idle (lazy)"),
            "web_search": cont.pipeline.web.available,
            "cache": {"enabled": cont.cache.enabled, "hits": cont.cache.hits, "misses": cont.cache.misses},
            "embedding": s.embedding_provider + ":" + s.embedding_model,
        }

    # ── chat ────────────────────────────────────────────────────────────────────────────────────
    @app.post("/v1/chat", response_model=ChatResponse, dependencies=protected, tags=["chat"])
    async def chat(req: ChatRequest) -> ChatResponse:
        return await c().pipeline.run(req)

    @app.post("/v1/chat/stream", dependencies=protected, tags=["chat"])
    async def chat_stream(req: ChatRequest, request: Request):
        rid = request_id_var.get()

        async def gen() -> AsyncIterator[str]:
            request_id_var.set(rid)
            async for ev in c().pipeline.stream(req):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"
                if await request.is_disconnected():
                    break

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/threads", dependencies=protected, tags=["chat"])
    async def threads():
        return await asyncio.to_thread(c().memory.threads)

    @app.get("/v1/threads/{thread_id}", dependencies=protected, tags=["chat"])
    async def thread_messages(thread_id: str):
        return await asyncio.to_thread(c().memory.messages, thread_id)

    @app.delete("/v1/threads/{thread_id}", dependencies=protected, tags=["chat"])
    async def delete_thread(thread_id: str):
        return {"deleted": await asyncio.to_thread(c().memory.delete, thread_id)}

    # ── knowledge base ──────────────────────────────────────────────────────────────────────────
    async def _run_job(job: IngestJob, root: Path, force: bool) -> None:
        cont = c()
        try:
            await asyncio.to_thread(cont.ingestor.run, root, force, job)
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest job failed")
            job.status, job.errors = "failed", job.errors + [str(exc)[:200]]
        finally:
            await asyncio.to_thread(cont.cache.clear)  # knowledge changed -> cached answers are stale

    @app.post("/v1/ingest/directory", response_model=IngestJob, dependencies=protected, tags=["knowledge"])
    async def ingest_directory(bg: BackgroundTasks, subpath: str = "", force: bool = False):
        root = (s.data_dir / subpath).resolve()
        if not str(root).startswith(str(s.data_dir.resolve())) or not root.exists():
            raise HTTPException(400, "subpath must exist inside DATA_DIR")
        job = c().ingestor.new_job()
        bg.add_task(_run_job, job, root, force)
        return job

    @app.post("/v1/ingest/upload", response_model=IngestJob, dependencies=protected, tags=["knowledge"])
    async def ingest_upload(bg: BackgroundTasks, files: list[UploadFile] = File(...), force: bool = False):
        up_dir = (s.data_dir / "uploads").resolve()
        up_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for f in files:
            name = re.sub(r"[^A-Za-z0-9._\- ]", "_", Path(f.filename or "file").name)[:120]
            if Path(name).suffix.lower() not in SUPPORTED_EXT:
                raise HTTPException(415, f"Unsupported file type: {name}")
            dest = up_dir / name
            size = 0
            with dest.open("wb") as out:
                while chunk := await f.read(1 << 20):
                    size += len(chunk)
                    if size > s.max_upload_mb * 1024 * 1024:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"{name} exceeds {s.max_upload_mb} MB")
                    out.write(chunk)
            saved.append(dest)
        job = c().ingestor.new_job()

        async def run_all() -> None:
            cont = c()
            job.status, job.files_total = "running", len(saved)
            for p in saved:
                try:
                    n, skipped = await asyncio.to_thread(cont.ingestor.ingest_file, p, force)
                    job.chunks_added += n
                    job.files_skipped += int(skipped)
                except Exception as exc:  # noqa: BLE001
                    job.errors.append(f"{p.name}: {str(exc)[:200]}")
                job.files_done += 1
            job.status = "failed" if len(job.errors) == len(saved) else "done"
            await asyncio.to_thread(cont.cache.clear)

        bg.add_task(run_all)
        return job

    @app.get("/v1/ingest/jobs/{job_id}", response_model=IngestJob, dependencies=protected, tags=["knowledge"])
    async def job_status(job_id: str):
        job = c().ingestor.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/v1/documents", response_model=list[DocumentInfo], dependencies=protected, tags=["knowledge"])
    async def documents():
        return await asyncio.to_thread(c().kb.list_documents)

    @app.delete("/v1/documents", dependencies=protected, tags=["knowledge"])
    async def delete_document(source: str = Query(..., min_length=1)):
        await asyncio.to_thread(c().kb.delete_source, source)
        await asyncio.to_thread(c().cache.clear)
        return {"deleted": source}

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()

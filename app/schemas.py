"""Pydantic request/response models shared by the API and the UI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = Field(None, description="Conversation id. Omit to start a new thread.")
    use_web_fallback: bool | None = None
    use_cache: bool | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    top_n: int | None = Field(None, ge=1, le=15)


class Source(BaseModel):
    id: int
    title: str
    origin: Literal["document", "web"] = "document"
    location: str = ""  # file path / url
    page: int | None = None
    score: float = 0.0
    snippet: str = ""
    cited: bool = False


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    intent: str
    standalone_query: str
    sources: list[Source] = []
    provider: str | None = None
    model: str | None = None
    cached: bool = False
    grounded: bool = True
    blocked: bool = False
    latency_ms: dict[str, int] = {}
    request_id: str = ""


class IngestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    files_total: int = 0
    files_done: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    errors: list[str] = []


class DocumentInfo(BaseModel):
    source: str
    chunks: int

"""Ingestion: load -> split -> embed -> upsert into Qdrant (idempotent, hash-based change detection)."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.rag.loaders import discover, load_file
from app.rag.vectorstore import KnowledgeBase
from app.schemas import IngestJob

log = logging.getLogger("rag.ingest")

MIN_CHUNK_CHARS = 40
ADD_BATCH = 64


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Ingestor:
    def __init__(self, s: Settings, kb: KnowledgeBase):
        self.s, self.kb = s, kb
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self.jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def source_name(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.s.data_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def ingest_file(self, path: Path, force: bool = False) -> tuple[int, bool]:
        """Returns (chunks_added, skipped)."""
        source = self.source_name(path)
        digest = file_hash(path)
        if not force and self.kb.source_hash(source) == digest:
            log.info("skip unchanged: %s", source)
            return 0, True

        pages = load_file(path)
        if not pages:
            raise ValueError("no extractable text")
        title = path.stem.replace("_", " ").replace("-", " ").strip()
        chunks: list[Document] = []
        for page_doc in pages:
            for piece in self.splitter.split_text(page_doc.page_content):
                if len(piece.strip()) < MIN_CHUNK_CHARS:
                    continue
                meta = {
                    **page_doc.metadata,
                    "source": source,
                    "title": page_doc.metadata.get("html_title") or title,
                    "file_type": path.suffix.lower().lstrip("."),
                    "content_hash": digest,
                    "chunk_index": len(chunks),
                    "ingested_at": int(time.time()),
                }
                chunks.append(Document(page_content=piece.strip(), metadata=meta))
        if not chunks:
            raise ValueError("document produced no usable chunks")

        # Re-ingest = replace. IDs are derived from (source, content-hash, index): the new version is fully
        # written first, then stale points of older versions are removed. If embedding fails half-way the
        # previous version stays queryable and the next run cleans up.
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{digest}:{c.metadata['chunk_index']}")) for c in chunks]
        for i in range(0, len(chunks), ADD_BATCH):
            self.kb.add(chunks[i : i + ADD_BATCH], ids[i : i + ADD_BATCH])
        self._drop_stale(source, keep_ids=set(ids))
        log.info("ingested %s -> %d chunks", source, len(chunks))
        return len(chunks), False

    def _drop_stale(self, source: str, keep_ids: set[str]) -> None:
        """Remove points of this source that belong to an older version of the file."""
        from qdrant_client import models

        stale: list = []
        offset = None
        while True:
            pts, offset = self.kb.client.scroll(
                self.kb.collection,
                scroll_filter=self.kb._source_filter(source),
                limit=512,
                offset=offset,
                with_payload=False,
            )
            stale += [p.id for p in pts if str(p.id) not in keep_ids]
            if offset is None:
                break
        if stale:
            self.kb.client.delete(self.kb.collection, points_selector=models.PointIdsList(points=stale), wait=True)

    # ── jobs ─────────────────────────────────────────────────────────────────────────────────────
    def run(self, root: Path, force: bool = False, job: IngestJob | None = None,
            on_update: Callable[[IngestJob], None] | None = None) -> IngestJob:
        job = job or IngestJob(job_id=uuid.uuid4().hex[:12], status="running")
        job.status = "running"
        files = discover(root)
        job.files_total = len(files)
        self.jobs[job.job_id] = job
        for f in files:
            try:
                n, skipped = self.ingest_file(f, force=force)
                job.chunks_added += n
                job.files_skipped += int(skipped)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                log.exception("failed to ingest %s", f)
                job.errors.append(f"{self.source_name(f)}: {str(exc)[:200]}")
            job.files_done += 1
            if on_update:
                on_update(job)
        job.status = "failed" if job.errors and job.files_done == len(job.errors) and files else "done"
        return job

    def new_job(self) -> IngestJob:
        job = IngestJob(job_id=uuid.uuid4().hex[:12], status="queued")
        with self._lock:
            self.jobs[job.job_id] = job
        return job

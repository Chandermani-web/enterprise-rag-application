"""Stage-2 cross-encoder reranking with FlashRank (ONNX, CPU). Lazy-loaded; falls back to vector scores."""
from __future__ import annotations

import logging
import threading

from langchain_core.documents import Document

from app.config import Settings

log = logging.getLogger("rag.rerank")


class Reranker:
    def __init__(self, s: Settings):
        self.s = s
        self._ranker = None
        self._failed = False
        self._lock = threading.Lock()

    def _get(self):
        if self._ranker is None and not self._failed:
            with self._lock:
                if self._ranker is None and not self._failed:
                    try:
                        from flashrank import Ranker

                        cache = str(self.s.state_dir / "flashrank")
                        self._ranker = Ranker(model_name=self.s.rerank_model, cache_dir=cache)
                        log.info("FlashRank model %s loaded", self.s.rerank_model)
                    except Exception as exc:  # noqa: BLE001
                        self._failed = True
                        log.warning("FlashRank unavailable (%s) -> using vector scores only", str(exc)[:200])
        return self._ranker

    @property
    def active(self) -> bool:
        return self._ranker is not None

    def rerank(self, query: str, hits: list[tuple[Document, float]], top_n: int) -> tuple[list[tuple[Document, float]], bool]:
        """Returns (ranked hits, reranked?). Never raises."""
        if not hits:
            return [], False
        if not self.s.rerank_enabled or (ranker := self._get()) is None:
            return hits[:top_n], False
        try:
            from flashrank import RerankRequest

            passages = [{"id": i, "text": d.page_content} for i, (d, _) in enumerate(hits)]
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
            out = [(hits[int(r["id"])][0], float(r["score"])) for r in ranked[:top_n]]
            return out, True
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed (%s) -> vector order", str(exc)[:200])
            return hits[:top_n], False

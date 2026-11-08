"""Tavily web search used as a fallback when the knowledge base cannot ground the answer."""
from __future__ import annotations

import logging

import httpx

from app.config import Settings

log = logging.getLogger("rag.web")


class WebSearch:
    URL = "https://api.tavily.com/search"

    def __init__(self, s: Settings):
        self.s = s
        self.key = s.secret("tavily_api_key")

    @property
    def available(self) -> bool:
        return bool(self.key) and self.s.web_fallback_enabled

    async def search(self, query: str) -> list[dict]:
        if not self.key:
            return []
        payload = {"query": query[:380], "max_results": self.s.web_max_results, "search_depth": "basic"}
        headers = {"Authorization": f"Bearer {self.key}"}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=12) as client:
                    r = await client.post(self.URL, json=payload, headers=headers)
                    r.raise_for_status()
                return [
                    {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", ""), "score": x.get("score", 0.0)}
                    for x in r.json().get("results", [])
                    if x.get("content")
                ]
            except Exception as exc:  # noqa: BLE001
                log.warning("tavily attempt %d failed: %s", attempt + 1, type(exc).__name__)
        return []

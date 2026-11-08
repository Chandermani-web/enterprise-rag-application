"""Document loaders: PDF, DOCX, HTML, Markdown/text/CSV/JSON. Pure-python, no heavy cloud dependencies."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document

log = logging.getLogger("rag.loaders")

TEXT_EXT = {".md", ".markdown", ".txt", ".rst", ".log", ".csv", ".json", ".yaml", ".yml", ".tf", ".toml", ".ini"}
SUPPORTED_EXT = TEXT_EXT | {".pdf", ".docx", ".html", ".htm"}


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            raise ValueError("PDF is password-protected")
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("%s p.%d unreadable: %s", path.name, i, exc)
            continue
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))
    if not docs:
        log.warning("%s has no extractable text (scanned PDF? run OCR first)", path.name)
    return docs


def _load_docx(path: Path) -> list[Document]:
    import docx  # python-docx

    d = docx.Document(str(path))
    parts: list[str] = []
    for p in d.paragraphs:
        if not p.text.strip():
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading"):
            level = int(re.sub(r"\D", "", style) or 1)
            parts.append("\n" + "#" * min(level, 4) + " " + p.text.strip())
        else:
            parts.append(p.text.strip())
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = _clean("\n\n".join(parts))
    return [Document(page_content=text)] if text else []


def _load_html(path: Path) -> list[Document]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = _clean(soup.get_text("\n"))
    meta = {"html_title": title} if title else {}
    return [Document(page_content=text, metadata=meta)] if text else []


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = _load_pdf(path)
    elif ext == ".docx":
        docs = _load_docx(path)
    elif ext in {".html", ".htm"}:
        docs = _load_html(path)
    elif ext in TEXT_EXT:
        text = _clean(_read_text(path))
        docs = [Document(page_content=text)] if text else []
    else:
        raise ValueError(f"unsupported file type: {ext}")
    return docs


def discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXT else []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT and not any(part.startswith(".") for part in p.parts)
    )

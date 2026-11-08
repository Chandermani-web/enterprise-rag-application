"""Streamlit front-end.  Run:  streamlit run ui/streamlit_app.py
Talks to the FastAPI backend only (never touches Qdrant/LLMs directly)."""
from __future__ import annotations

import json
import os
import time
import uuid

import httpx
import streamlit as st

DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise RAG", page_icon="🧠", layout="wide")


# ── API helpers ─────────────────────────────────────────────────────────────────────────────────
def headers() -> dict:
    key = st.session_state.get("api_key") or os.getenv("API_AUTH_KEY", "")
    return {"X-API-Key": key} if key else {}


def api(method: str, path: str, **kw):
    base = st.session_state.api_url.rstrip("/")
    try:
        r = httpx.request(method, base + path, headers=headers(), timeout=kw.pop("timeout", 30), **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.sidebar.error(f"{e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"API unreachable: {type(e).__name__}")
    return None


def stream_chat(payload: dict):
    """Yield (event, data) tuples from the SSE endpoint."""
    base = st.session_state.api_url.rstrip("/")
    with httpx.stream("POST", base + "/v1/chat/stream", json=payload, headers=headers(),
                      timeout=httpx.Timeout(120, connect=10)) as r:
        if r.status_code != 200:
            r.read()
            yield "error", {"message": f"HTTP {r.status_code}: {r.text[:300]}"}
            return
        event = "message"
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, json.loads(line[5:].strip())


# ── state ───────────────────────────────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("api_url", DEFAULT_API)
ss.setdefault("thread_id", uuid.uuid4().hex)
ss.setdefault("messages", [])
ss.setdefault("loaded_thread", None)


def load_thread(tid: str) -> None:
    data = api("GET", f"/v1/threads/{tid}") or []
    ss.thread_id, ss.loaded_thread = tid, tid
    ss.messages = [{"role": m["role"], "content": m["content"], "meta": m.get("meta", {})} for m in data]


def new_chat() -> None:
    ss.thread_id, ss.messages, ss.loaded_thread = uuid.uuid4().hex, [], None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            icon = "🌐" if s["origin"] == "web" else "📄"
            cited = " ✅ cited" if s.get("cited") else ""
            page = f" · p.{s['page']}" if s.get("page") else ""
            loc = f"[{s['location']}]({s['location']})" if s["origin"] == "web" else f"`{s['location']}`"
            st.markdown(f"**[{s['id']}] {icon} {s['title']}**{page} — {loc} · score {s['score']:.2f}{cited}")
            st.caption(s["snippet"] + "…")


def render_meta(meta: dict) -> None:
    bits = []
    if meta.get("cached"):
        bits.append("⚡ semantic cache")
    elif meta.get("provider"):
        bits.append(f"🤖 {meta['provider']} · {meta.get('model', '')}")
    if meta.get("grounded") is False:
        bits.append("⚠️ low-confidence: weak match in knowledge base")
    lat = meta.get("latency_ms") or {}
    if lat.get("total"):
        bits.append(f"⏱ {lat['total'] / 1000:.1f}s")
    if bits:
        st.caption("  ·  ".join(bits))


# ── sidebar ─────────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Enterprise RAG")
    with st.expander("Connection", expanded=False):
        ss.api_url = st.text_input("API URL", ss.api_url)
        ss.api_key = st.text_input("API key (X-API-Key)", type="password", value=ss.get("api_key", ""))

    c1, c2 = st.columns(2)
    c1.button("➕ New chat", on_click=new_chat, use_container_width=True)
    if c2.button("🗑 Delete", use_container_width=True) and ss.messages:
        api("DELETE", f"/v1/threads/{ss.thread_id}")
        new_chat()
        st.rerun()

    threads = api("GET", "/v1/threads") or []
    if threads:
        labels = {t["thread_id"]: f"{t['title'] or 'Untitled'} ({t['messages']})" for t in threads}
        ids = list(labels)
        current = ss.thread_id if ss.thread_id in ids else None
        choice = st.selectbox("History", ids, index=ids.index(current) if current else None,
                              format_func=lambda i: labels[i], placeholder="Past conversations…")
        if choice and choice != ss.loaded_thread and choice != ss.thread_id:
            load_thread(choice)
            st.rerun()

    st.subheader("Retrieval")
    use_web = st.toggle("Web fallback (Tavily)", value=True, help="Used only when the knowledge base can't ground the answer")
    use_cache = st.toggle("Semantic cache", value=True)
    top_k = st.slider("Candidates (vector search)", 5, 40, 15)
    top_n = st.slider("Context chunks (after rerank)", 1, 10, 5)

    st.subheader("Knowledge base")
    docs = api("GET", "/v1/documents") or []
    st.caption(f"{len(docs)} documents · {sum(d['chunks'] for d in docs)} chunks")
    up = st.file_uploader("Upload documents", accept_multiple_files=True,
                          type=["pdf", "docx", "md", "txt", "html", "htm", "csv", "json", "yaml", "yml"])
    colA, colB = st.columns(2)
    force = st.checkbox("Force re-index", value=False)
    job = None
    if colA.button("⬆️ Ingest", disabled=not up, use_container_width=True):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in up]
        job = api("POST", "/v1/ingest/upload", files=files, params={"force": force}, timeout=120)
    if colB.button("📁 Scan data/", use_container_width=True):
        job = api("POST", "/v1/ingest/directory", params={"force": force})
    if job:
        bar, txt = st.progress(0.0), st.empty()
        while True:
            j = api("GET", f"/v1/ingest/jobs/{job['job_id']}") or {}
            total = max(j.get("files_total", 0), 1)
            bar.progress(min(j.get("files_done", 0) / total, 1.0))
            txt.caption(f"{j.get('status')} · {j.get('files_done', 0)}/{j.get('files_total', 0)} files · "
                        f"{j.get('chunks_added', 0)} chunks · {j.get('files_skipped', 0)} unchanged")
            if j.get("status") in ("done", "failed") or not j:
                break
            time.sleep(1.0)
        for e in j.get("errors", []):
            st.error(e)
        if j.get("status") == "done":
            st.success("Ingestion finished")
            st.rerun()
    if docs:
        with st.expander("Indexed documents"):
            for d in docs:
                cA, cB = st.columns([5, 1])
                cA.caption(f"{d['source']} · {d['chunks']}")
                if cB.button("✕", key="del" + d["source"]):
                    api("DELETE", "/v1/documents", params={"source": d["source"]})
                    st.rerun()

    with st.expander("System status"):
        prov = api("GET", "/v1/providers")
        if prov:
            for p in prov["llm"]:
                dot = "🟢" if p["state"] == "closed" else f"🔴 retry in {p['retry_in_s']}s"
                st.caption(f"{dot} **{p['provider']}** · {p['model']} · {p['calls']} calls / {p['errors']} errors")
            st.caption(f"Reranker: {prov['reranker']} · Web: {'on' if prov['web_search'] else 'off'} · "
                       f"Cache: {prov['cache']['hits']} hits / {prov['cache']['misses']} misses")
            st.caption(f"Embeddings: {prov['embedding']}")

# ── chat ────────────────────────────────────────────────────────────────────────────────────────
if not ss.messages:
    st.markdown("### Ask anything about your documents")
    st.caption("Answers are grounded in your knowledge base and cite their sources.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources((m.get("meta") or {}).get("sources", []))
            render_meta(m.get("meta") or {})

if prompt := st.chat_input("Ask a question…"):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status, box = st.status("Thinking…", expanded=False), st.empty()
        text, final, error = "", None, None
        payload = {"message": prompt, "thread_id": ss.thread_id, "use_web_fallback": use_web,
                   "use_cache": use_cache, "top_k": top_k, "top_n": top_n}
        try:
            for ev, data in stream_chat(payload):
                if ev == "status":
                    labels = {"planning": "Understanding question…", "retrieving": f"Searching: {data.get('query', '')}",
                              "web_search": "Searching the web…", "generating": "Writing answer…"}
                    status.update(label=labels.get(data["stage"], data["stage"]))
                elif ev == "token":
                    text += data["text"]
                    box.markdown(text + "▌")
                elif ev == "reset":
                    text = ""
                    status.update(label=f"Provider {data.get('failed_provider')} failed — retrying on next provider…")
                    box.empty()
                elif ev == "final":
                    final = data
                elif ev == "error":
                    error = data
        except Exception as e:  # noqa: BLE001
            error = {"message": f"Connection problem: {type(e).__name__}"}
        if error:
            status.update(label="Failed", state="error")
            box.error(error["message"])
            for d in error.get("detail", []):
                st.caption(d)
        elif final:
            status.update(label="Done", state="complete")
            box.markdown(final["answer"])
            render_sources(final["sources"])
            render_meta(final)
            ss.messages.append({"role": "assistant", "content": final["answer"],
                                "meta": {"sources": final["sources"], **{k: final[k] for k in ("provider", "model", "cached", "grounded", "latency_ms")}}})

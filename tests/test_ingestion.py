import pytest

from tests.conftest import build, provider


def test_ingest_and_skip_unchanged(settings, docs_dir):
    c = build(settings, [provider("x")])
    job = c.ingestor.run(docs_dir)
    assert job.status == "done" and job.chunks_added > 0 and not job.errors
    n = c.kb.count()
    job2 = c.ingestor.run(docs_dir)
    assert job2.files_skipped == job2.files_total and c.kb.count() == n


def test_changed_file_replaces_old_chunks(settings, docs_dir):
    c = build(settings, [provider("x")])
    c.ingestor.run(docs_dir)
    (docs_dir / "redis.txt").write_text("Completely new content about vector databases and Qdrant collections " * 3)
    c.ingestor.run(docs_dir)
    docs = {d["source"]: d["chunks"] for d in c.kb.list_documents()}
    assert docs["redis.txt"] == 1
    hits = c.kb.search("Redis semantic caching embeddings", 5)
    assert not any("semantic caching" in d.page_content for d, _ in hits if d.metadata["source"] == "redis.txt")


def test_bad_file_does_not_stop_batch(settings, docs_dir):
    (docs_dir / "broken.pdf").write_bytes(b"not a pdf")
    c = build(settings, [provider("x")])
    job = c.ingestor.run(docs_dir)
    assert any("broken.pdf" in e for e in job.errors)
    assert job.chunks_added > 0


def test_delete_source(container):
    container.kb.delete_source("redis.txt")
    assert "redis.txt" not in {d["source"] for d in container.kb.list_documents()}


def test_html_docx_loaders(settings, docs_dir):
    import docx

    (docs_dir / "page.html").write_text("<html><title>T</title><script>evil()</script><body><p>" + "Visible paragraph text. " * 5 + "</p></body></html>")
    d = docx.Document(); d.add_heading("Heading One", 1); d.add_paragraph("Body paragraph with enough words to be a chunk, " * 3); d.save(docs_dir / "w.docx")
    c = build(settings, [provider("x")])
    job = c.ingestor.run(docs_dir)
    assert not job.errors
    srcs = {x["source"] for x in c.kb.list_documents()}
    assert {"page.html", "w.docx"} <= srcs
    assert not any("evil()" in d.page_content for d, _ in c.kb.search("evil", 20))

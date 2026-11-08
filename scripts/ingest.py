"""CLI ingestion (no API server needed):  python scripts/ingest.py [path-inside-DATA_DIR] [--force]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    ap.add_argument("path", nargs="?", default="", help="sub-folder or file inside DATA_DIR (default: everything)")
    ap.add_argument("--force", action="store_true", help="re-index even if the file hash is unchanged")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    c = build_container(s)
    root = (s.data_dir / args.path).resolve()
    if not root.exists() or not str(root).startswith(str(s.data_dir.resolve())):
        print(f"Path must exist inside {s.data_dir}", file=sys.stderr)
        return 2
    job = c.ingestor.run(root, force=args.force)
    c.cache.clear()
    print(f"{job.status}: {job.files_done}/{job.files_total} files, {job.chunks_added} chunks, "
          f"{job.files_skipped} unchanged, {len(job.errors)} errors")
    for e in job.errors:
        print("  ✗", e)
    return 1 if job.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

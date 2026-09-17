"""Complete command-line entry points and a one-command local workbench."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser

from . import __version__
from .hologram import MAX_HOLOGRAM_BYTES, decode_memory
from .index import HolographicIndex
from .optics import Backend
from .sample import load_demo
from .schema import EUHNNError, IndexConfig, ValidationError


def write_new(path: str | Path, content: bytes) -> None:
    """Write a complete new artifact without overwriting an existing target.

    The temporary file is flushed first. Linking it into place is exclusive on
    supported local filesystems. On unsupported filesystems fail explicitly; do
    not fall back to a potentially destructive overwrite.
    """
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".euhnn-export-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="euhnn", description="Local ray-traced holographic document retrieval.")
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument(
        "--index", default=os.environ.get("EUHNN_INDEX", "euhnn-library.sqlite"), help="Persistent local SQLite index"
    )
    result.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    commands = result.add_subparsers(dest="command", required=True)
    create = commands.add_parser("init", help="Create or verify a local index; never overwrite another format")
    create.add_argument("--chunk-words", type=int, default=192)
    create.add_argument("--overlap-words", type=int, default=32)
    ingest = commands.add_parser("ingest", help="Index selected files; no implicit recursion or network crawling")
    ingest.add_argument("files", nargs="+")
    search = commands.add_parser("search", help="Print exact matching passages and their provenance")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=("hybrid", "lexical", "optical"), default="hybrid")
    search.add_argument("--phrase", action="store_true")
    search.add_argument("--source")
    search.add_argument("--json", action="store_true")
    context = commands.add_parser("context", help="Build a citation-bearing context for an external RAG model")
    context.add_argument("query")
    context.add_argument("--top-k", type=int, default=5)
    context.add_argument("--max-characters", type=int, default=12000)
    teach = commands.add_parser("learn", help="Train an explicit query-to-passage association")
    teach.add_argument("query")
    teach.add_argument("chunk_id")
    remove = commands.add_parser("delete", help="Delete one explicitly identified source document")
    remove.add_argument("document_id")
    commands.add_parser("list", help="List documents in this index")
    commands.add_parser("status", help="Report real runtime/backend and index statistics")
    commands.add_parser("verify", help="Verify storage and source/vector checksums")
    commands.add_parser("demo", help="Index bundled original examples and run a cited search")
    commands.add_parser("doctor", help="Inspect dependency, SQLite and CUDA availability")
    export = commands.add_parser("export", help="Export exact source memory as an RGB Fourier hologram")
    export.add_argument("output")
    restore = commands.add_parser(
        "import", help="Import a checked hologram; existing sources with matching names are replaced"
    )
    restore.add_argument("input")
    serve = commands.add_parser("serve", help="Start the authenticated loopback workbench")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="Open the authenticated local page in the default browser")
    serve.add_argument("--demo", action="store_true", help="Add the bundled examples before serving")
    return result


def _output(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def _open_when_ready(port: int, token: str) -> None:
    """Open only our ready loopback server, without sending the token in a URL query."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/status", headers={"Authorization": "Bearer " + token}
            )
            with opener.open(request, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(f"http://127.0.0.1:{port}/#token={token}")
                    return
        except OSError:
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    """Run a complete operation and report failures without inventing success."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            runtime = Backend.select(args.backend)
            with sqlite3.connect(":memory:") as db:
                db.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
            versions = {}
            for name in ("numpy", "cupy-cuda13x", "cupy-cuda12x", "fastapi", "uvicorn", "pypdf", "python-docx"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            _output(
                {
                    "version": __version__,
                    "python": sys.version.split()[0],
                    "sqlite": sqlite3.sqlite_version,
                    "fts5": True,
                    "packages": versions,
                    **runtime.info(),
                }
            )
            return 0
        create = args.command in ("init", "ingest", "demo", "serve", "import")
        config = (
            IndexConfig(chunk_words=args.chunk_words, overlap_words=args.overlap_words)
            if args.command == "init"
            else None
        )
        import_blob = None
        if args.command == "import":
            with open(args.input, "rb") as handle:
                import_blob = handle.read(MAX_HOLOGRAM_BYTES + 1)
            if not Path(args.index).exists():
                memory = decode_memory(import_blob)
                if not isinstance(memory, dict) or memory.get("kind") != "euhnn-document-memory":
                    raise ValidationError("The file does not contain a document memory.")
                config = IndexConfig.from_dict(memory["config"])
        with HolographicIndex(args.index, create=create, config=config, backend=args.backend) as index:
            command = args.command
            if command in ("init", "status"):
                _output(index.stats())
            elif command == "ingest":
                for path in args.files:
                    _output(index.ingest_file(path))
            elif command == "search":
                hits = index.search(
                    args.query, top_k=args.top_k, mode=args.mode, source=args.source, phrase=args.phrase
                )
                if args.json:
                    _output([hit.to_dict() for hit in hits])
                elif not hits:
                    print("No matching source passage. No answer was generated.")
                else:
                    for number, hit in enumerate(hits, 1):
                        print(f"\n[{number}] {hit.citation}\nScore: {hit.score:.6f}\n{hit.text}\n")
            elif command == "context":
                print(index.context(args.query, top_k=args.top_k, max_characters=args.max_characters))
            elif command == "learn":
                _output(index.learn(args.query, args.chunk_id))
            elif command == "delete":
                _output({"deleted": index.delete_document(args.document_id)})
            elif command == "list":
                for offset in range(0, index.stats()["documents"], 1000):
                    _output(index.documents(limit=1000, offset=offset))
            elif command == "verify":
                _output(index.verify())
            elif command == "demo":
                _output(
                    {"ingested": load_demo(index), "results": [h.to_dict() for h in index.search("blue wavelength")]}
                )
            elif command == "export":
                blob = index.export_hologram()
                write_new(args.output, blob)
                _output(
                    {
                        "output": str(Path(args.output).resolve()),
                        "bytes": len(blob),
                        "sha256": sha256(blob).hexdigest(),
                        "encrypted": False,
                    }
                )
            elif command == "import":
                assert import_blob is not None
                _output(index.import_hologram(import_blob))
            elif command == "serve":
                from .server import create_app
                import uvicorn

                if not 0 <= args.port <= 65535:
                    raise ValidationError("port must be between 0 and 65535.")
                token = secrets.token_urlsafe(32)
                app = create_app(index, token=token)
                if args.demo:
                    load_demo(index)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.bind(("127.0.0.1", args.port))
                    port = sock.getsockname()[1]
                    print(f"EUHNN {__version__} | {index.encoder.backend.name}", flush=True)
                    print(f"Local workbench: http://127.0.0.1:{port}/#token={token}", flush=True)
                    print(
                        "Keep this window open. Press Ctrl+C to stop. The token grants access to this local library.",
                        flush=True,
                    )
                    if args.open:
                        threading.Thread(target=_open_when_ready, args=(port, token), daemon=True).start()
                    server = uvicorn.Server(
                        uvicorn.Config(
                            app, log_level="warning", access_log=False, limit_concurrency=8, timeout_keep_alive=5
                        )
                    )
                    server.run(sockets=[sock])
                finally:
                    sock.close()
        return 0
    except (EUHNNError, OSError, sqlite3.DatabaseError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ImportError:
        print(
            "A required optional dependency is missing. Install the workbench extra for the web interface or documents extra for PDF/DOCX.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

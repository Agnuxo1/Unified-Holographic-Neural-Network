"""Desktop entry point for the self-contained Windows CPU distribution."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile

from euhnn import HolographicIndex, __version__
from euhnn.cli import main as command_line


def installation_self_test() -> int:
    """Verify packaged data files, readers, optical memory and local API schemas."""
    from docx import Document
    from pypdf import PdfWriter
    from euhnn.sample import load_demo
    from euhnn.server import create_app

    static = Path(__file__).with_name("web")
    assert all((static / name).is_file() for name in ("index.html", "app.js", "style.css"))
    document = Document()
    document.add_paragraph("Packaged DOCX template check")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with tempfile.TemporaryDirectory(prefix="euhnn-desktop-check-") as temporary:
        with HolographicIndex(Path(temporary) / "check.sqlite", create=True, backend="cpu") as index:
            load_demo(index)
            assert index.search("blue wavelength")[0].source == "demo/optical-memory.md"
            memory = index.export_hologram()
            assert index.import_hologram(memory)["documents"] == 3
            assert index.verify()["ok"]
            assert create_app(index).openapi()["info"]["version"] == __version__
    print(
        json.dumps(
            {
                "ok": True,
                "version": __version__,
                "edition": "Windows CPU portable",
                "packaged_browser_assets": True,
                "document_reader_resources": True,
                "retrieval_and_memory_round_trip": True,
            }
        )
    )
    return 0


def main() -> int:
    """Open the local workbench without requiring an installed Python interpreter."""
    options = argparse.ArgumentParser(description="EUHNN local holographic document workbench (CPU edition)")
    options.add_argument("--self-test", action="store_true")
    options.add_argument("--no-open", action="store_true")
    options.add_argument("--index", type=Path)
    options.add_argument("--port", type=int, default=0)
    args = options.parse_args()
    if args.self_test:
        return installation_self_test()
    home = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EUHNN"
    path = args.index or home / "library.sqlite"
    command = ["--index", str(path), "--backend", "cpu", "serve", "--port", str(args.port)]
    if not args.no_open:
        command.append("--open")
    return command_line(command)


if __name__ == "__main__":
    raise SystemExit(main())

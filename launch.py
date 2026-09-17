"""One-command EUHNN installer/launcher, isolated from the user's Python packages.

Run Start-EUHNN.cmd on Windows or ./start-euhnn.sh on Linux/macOS. Python 3.12+
with venv support is required. First installation downloads declared packages
from the configured Python package index. No administrator privilege is requested.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv

VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    """Detect source changes so an old installation is never silently reused."""
    digest = sha256()
    for path in sorted([ROOT / "pyproject.toml", *(ROOT / "src" / "euhnn").rglob("*")]):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cuda_package() -> str | None:
    """Choose one CuPy family using the installed driver's advertised capability."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        status = subprocess.run([executable], capture_output=True, text=True, timeout=10, check=False)
        match = re.search(r"CUDA Version:\s*(\d+)\.", status.stdout)
        if status.returncode or not match or int(match.group(1)) < 12:
            return None
        family = 13 if int(match.group(1)) >= 13 else 12
        # Component wheels supply the runtime on machines without a toolkit.
        toolkit = os.environ.get("CUDA_PATH")
        extra = "" if toolkit and Path(toolkit).is_dir() else "[ctk]"
        return f"cupy-cuda{family}x{extra}==14.2.0"
    except (OSError, subprocess.TimeoutExpired):
        return None


def run(command: list[str]) -> None:
    """Execute one explicit setup command and preserve its nonzero failure."""
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Setup command failed with exit code {completed.returncode}.")


def main(argv: list[str] | None = None) -> int:
    options = argparse.ArgumentParser(description="Prepare and open the local EUHNN document workbench.")
    options.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    options.add_argument("--environment", type=Path, default=ROOT / ".euhnn-runtime")
    options.add_argument("--index", type=Path, default=ROOT / ".local-state" / "library.sqlite")
    options.add_argument("--prepare-only", action="store_true")
    options.add_argument("--self-test", action="store_true")
    options.add_argument("--no-open", action="store_true")
    options.add_argument("--port", type=int, default=0)
    args = options.parse_args(argv)
    if sys.version_info < (3, 12):
        print(
            "Python 3.12 or later is required. The launcher did not change your Python installation.", file=sys.stderr
        )
        return 2
    environment = args.environment.expanduser().resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / "euhnn-installation.json"
    fingerprint = source_fingerprint()
    try:
        if not python.is_file():
            if environment.exists() and any(environment.iterdir()):
                raise RuntimeError("The selected environment folder contains unrelated files; it was not modified.")
            print("Preparing an isolated Python environment...", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        installed = {}
        if marker.is_file():
            try:
                installed = json.loads(marker.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                installed = {}
        check = subprocess.run(
            [str(python), "-c", f"import euhnn,fastapi,uvicorn,pypdf,docx; assert euhnn.__version__=={VERSION!r}"],
            capture_output=True,
            check=False,
        )
        if installed.get("source_sha256") != fingerprint or check.returncode:
            print("Installing the versioned workbench and document readers...", flush=True)
            run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "install",
                    "--no-input",
                    str(ROOT) + "[workbench]",
                ]
            )
            installed = {"version": VERSION, "source_sha256": fingerprint, "cuda_package": None}
        if args.backend != "cpu" and not installed.get("cuda_checked"):
            package = cuda_package()
            if package:
                print("Preparing the matching NVIDIA CUDA backend...", flush=True)
                result = subprocess.run(
                    [str(python), "-m", "pip", "--disable-pip-version-check", "install", "--no-input", package],
                    cwd=ROOT,
                )
                if result.returncode and args.backend == "cuda":
                    raise RuntimeError(
                        "CUDA installation failed; rerun with --backend cpu to use the reference backend."
                    )
                installed["cuda_package"] = package if result.returncode == 0 else None
            elif args.backend == "cuda":
                raise RuntimeError(
                    "No compatible NVIDIA driver was detected. Install a supported driver or select --backend cpu."
                )
            installed["cuda_checked"] = True
        run([str(python), "-c", f"import euhnn,fastapi,uvicorn,pypdf,docx; assert euhnn.__version__=={VERSION!r}"])
        marker.write_text(json.dumps(installed, indent=2) + "\n", encoding="utf-8")
        if args.self_test:
            print("Checking a temporary local document library...", flush=True)
            code = (
                "import tempfile; from pathlib import Path; from euhnn import HolographicIndex; "
                "from euhnn.sample import load_demo; "
                "temporary=tempfile.TemporaryDirectory(); "
                "index=HolographicIndex(Path(temporary.name)/'test.sqlite',create=True,backend='cpu'); "
                "load_demo(index); assert index.search('blue wavelength'); assert index.verify()['ok']; "
                "index.close(); temporary.cleanup(); print('EUHNN installation self-test passed.')"
            )
            run([str(python), "-c", code])
            return 0
        if args.prepare_only:
            print(f"EUHNN {VERSION} is installed. Run the launcher again to open the workbench.")
            return 0
        command = [
            str(python),
            "-m",
            "euhnn",
            "--index",
            str(args.index.expanduser().resolve()),
            "--backend",
            args.backend,
            "serve",
            "--port",
            str(args.port),
        ]
        if not args.no_open:
            command.append("--open")
        return subprocess.call(command, cwd=ROOT)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("EUHNN could not start: " + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

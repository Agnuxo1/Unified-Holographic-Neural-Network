# Setup, operation and troubleshooting

## Source launcher

Use Python 3.12 or later. On Windows run `Start-EUHNN.cmd`; on Linux/macOS run
`sh start-euhnn.sh`. First use downloads the declared packages into `.euhnn-runtime`.
No system Python packages are changed and no administrator privilege is requested.
The default index is `.local-state/library.sqlite`. Do not delete that file to
troubleshoot an installation: it contains your documents and learned associations.

```bash
python launch.py --self-test --backend cpu
python launch.py --prepare-only --backend cpu
python launch.py --backend cpu
python launch.py --backend cuda
```

The launcher fingerprints the active source and reinstalls its package after changes.
It opens an unused loopback port by default. To choose a fixed port:

```bash
python launch.py --port 8765
```

When the page is refreshed, use the original launch link again or enter the local
connection token using **Connection**. The token is kept only in browser memory,
removed from the address-bar fragment and never written into page HTML. Do not
share a live token: it grants access to the entire local library.

## Manual package installation

```bash
python -m venv .venv
# Activate the environment using the normal command for your operating system.
python -m pip install ".[workbench]"
python -m euhnn --index library.sqlite init
python -m euhnn --index library.sqlite ingest manual.pdf notes.md
python -m euhnn --index library.sqlite search "calibration procedure" --json
python -m euhnn --index library.sqlite verify
python -m euhnn --index library.sqlite serve --port 0 --open
```

Global `--index` and `--backend` options precede the subcommand. Each input path is
explicit; the program does not crawl your disk or the internet. Importing the same
source and content twice is idempotent. Updating a source replaces its old passages
and teaching examples transactionally. An unreadable later page rolls back the update.

## CUDA

`cpu` is the NumPy reference. `cuda` must initialize a real NVIDIA device and fails
if unavailable. `auto` attempts CUDA and reports a fallback reason when it must use CPU.
The launcher detects the driver's CUDA family and installs one compatible CuPy family.
It does not upgrade your graphics driver. Never install both CuPy CUDA families in
one environment.

With a system CUDA toolkit, select one manual extra:

```bash
python -m pip install ".[cuda13]"
python -m euhnn --backend cuda doctor
```

For CUDA 12 use `.[cuda12]` instead. On a driver-only installation, CuPy's component
wheel option can supply runtime dependencies, for example
`python -m pip install "cupy-cuda13x[ctk]==14.2.0"` for an appropriate CUDA 13 driver.
Official compatibility reference: https://docs.cupy.dev/en/stable/install.html

The ray stage benefits from parallelism; small end-to-end document searches are not
necessarily faster on GPU. See the measured CPU and CUDA results rather than assuming
that GPU detection proves an overall speedup.

## Memory backup

```bash
python -m euhnn --index library.sqlite export backup.holo
python -m euhnn --index restored.sqlite import backup.holo
python -m euhnn --index restored.sqlite verify
```

CLI export refuses to overwrite an existing target. A restored new index adopts the
export's feature configuration; an existing incompatible index fails explicitly.
Import replaces sources with matching names, while unrelated documents remain.
The file contains source text and is not encrypted. For larger collections exceeding
the hologram export limit, stop the application and preserve the complete SQLite
files together; do not copy only the main file while another process is writing WAL.

## Supported input and limits

UTF-8 text, or UTF-16 with a byte-order mark; UTF-8 BOM is accepted. HTML scripts and
styles are not executed. PDFs must be unencrypted and contain extractable text.
Scanned image-only PDFs need OCR in a separate trusted workflow. DOCX paragraph/table
order is retained; headers, drawings, text boxes and page layout are not promised.

One file: 64 MiB. Extracted text: 64 Mi characters and 5,000 pages per document.
One index: 100,000 documents / 1,000,000 passages. Hologram export: 32 MiB before
compression. Queries: 4,096 characters, at most 100 returned results. Teaching:
256 examples. These are enforced limits, not claims of validated capacity at every maximum.

Unsupported format, corrupt memory, wrong configuration, missing CUDA, occupied port
or disk errors are reported; they do not count as successful indexing or retrieval.

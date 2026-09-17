"""Local document loaders with explicit type, encoding and resource boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import io
import zipfile

from .schema import TextPage, ValidationError, bounded_text

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 64 * 1024 * 1024
SUPPORTED_SUFFIXES = frozenset(
    {".txt", ".md", ".rst", ".log", ".csv", ".json", ".py", ".js", ".ts", ".html", ".htm", ".pdf", ".docx"}
)


class _VisibleHTML(HTMLParser):
    """Extract visible text; never execute scripts, fetch links or load resources."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        elif not self.hidden and tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in ("p", "div", "li", "h1", "h2", "h3", "tr", "section"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def decode_text(data: bytes) -> str:
    """Accept UTF-8 (optional BOM) or BOM-marked UTF-16; do not guess silently."""
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16")
        return data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValidationError("Text must use UTF-8 or BOM-marked UTF-16 encoding.") from exc


def iter_document(data: bytes, filename: str) -> Iterator[TextPage]:
    """Yield extracted pages. PDF offsets are relative to extracted page text.

    No OCR is performed. An image-only/encrypted PDF fails explicitly. DOCX page
    numbers are deliberately omitted because page layout is not preserved by XML.
    """
    bounded_text(filename, "filename", 1024)
    if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES:
        raise ValidationError("Each document must be at most 64 MiB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValidationError("Unsupported document type: " + (suffix or "no extension"))
    extracted = 0
    found = False
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValidationError("PDF support requires the euhnn[documents] extra.") from exc
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ValidationError("Encrypted PDFs are not accepted. Supply a decrypted copy.")
            if len(reader.pages) > 5000:
                raise ValidationError("PDF files are limited to 5000 pages.")
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                extracted += len(text)
                if extracted > MAX_EXTRACTED_CHARACTERS:
                    raise ValidationError("Extracted document text exceeds 64 Mi characters.")
                if text.strip():
                    bounded_text(text, "page text", MAX_EXTRACTED_CHARACTERS)
                    found = True
                    yield TextPage(text, number)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError("The PDF could not be parsed as a valid text-bearing document.") from exc
        if not found:
            raise ValidationError(
                "No extractable text was found. Image-only PDFs require OCR outside this application."
            )
        return
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ValidationError("DOCX support requires the euhnn[documents] extra.") from exc
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                items = archive.infolist()
                if (
                    len(items) > 10000
                    or sum(i.file_size for i in items) > 128 * 1024 * 1024
                    or any(i.flag_bits & 1 for i in items)
                ):
                    raise ValidationError("DOCX expanded contents exceed the resource limit.")
            document = Document(io.BytesIO(data))
            parts = []
            # Iteration preserves the interleaving of body paragraphs and tables.
            from docx.text.paragraph import Paragraph
            from docx.table import Table

            for item in document.iter_inner_content():
                if isinstance(item, Paragraph):
                    parts.append(item.text)
                elif isinstance(item, Table):
                    for row in item.rows:
                        parts.append("\t".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError("The DOCX could not be parsed.") from exc
    else:
        text = decode_text(data)
        if suffix in (".html", ".htm"):
            parser = _VisibleHTML()
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
    bounded_text(text, "document text", MAX_EXTRACTED_CHARACTERS)
    yield TextPage(text)


def read_local(path: str | Path) -> tuple[bytes, str]:
    """Read one explicitly selected regular local file without recursive crawling."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValidationError("Document path is not a regular readable file.")
    with file.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValidationError("Each document must be at most 64 MiB.")
    return data, str(file.resolve())


def fingerprint(data: bytes) -> str:
    """Content digest for idempotent imports; not an ownership/authenticity claim."""
    return sha256(data).hexdigest()

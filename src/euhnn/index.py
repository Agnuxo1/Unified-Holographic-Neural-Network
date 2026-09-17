"""Persistent source-grounded retrieval with transactional SQLite/FTS5 storage.

The optical projection augments, rather than impersonates, semantic embeddings.
Default hybrid retrieval reranks a bounded BM25 candidate pool. Explicit optical
mode scans every signature in batches and has O(ND) query cost. Original source
text and precise provenance remain stored independently of the lossy signatures.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from hashlib import sha256
import heapq
import json
import os
from pathlib import Path
import sqlite3
import threading
import tempfile
import time
from typing import Any

import numpy as np

from .documents import fingerprint, iter_document, read_local
from .hologram import decode_memory, encode_memory, MAX_RAW_BYTES
from .optics import OpticalEncoder, OpticalReadout
from .schema import (
    EUHNNError,
    IndexConfig,
    IndexFormatError,
    SearchHit,
    TextChunk,
    TextPage,
    ValidationError,
    bounded_int,
    bounded_text,
)
from .text import fts_query, iter_chunks

APP_ID = 0x45554832
SCHEMA_VERSION = 2
MAX_DOCUMENTS = 100000
MAX_CHUNKS = 1000000
MAX_CHUNK_CHARACTERS = 65536


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError("Metadata must be finite JSON with valid Unicode.") from exc


def _metadata(value: dict[str, Any] | None) -> str:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValidationError("metadata must be a JSON object.")
    text = _json(value)
    bounded_text(text, "metadata", 16384)
    return text


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


class HolographicIndex:
    """A persistent local index whose runtime is independent of any LLM.

    Use create=True only to initialize a missing index. Existing databases are
    validated, never silently reset. Re-indexing the same source replaces its old
    passages and teaching examples in one transaction; malformed input rolls back.
    """

    def __init__(
        self, path: str | Path, *, create: bool = False, config: IndexConfig | None = None, backend: str = "auto"
    ) -> None:
        if type(create) is not bool:
            raise ValidationError("create must be a boolean.")
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.RLock()
        self._closed = False
        self._readout: OpticalReadout | None = None
        self._readout_generation = -1
        existed = self.path.exists()
        if not existed and not create:
            raise ValidationError("Index does not exist. Create it before searching.")
        try:
            if not existed:
                # Build the complete database under a unique temporary name before
                # making it visible. A failed encoder or schema creation leaves no
                # half-created target; an existing target is never overwritten.
                self.config = config or IndexConfig()
                self.encoder = OpticalEncoder(self.config, backend=backend)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".euhnn-create-", dir=self.path.parent)
                os.close(fd)
                try:
                    self._db = self._connect(Path(temporary))
                    self._create()
                    self._db.close()
                    with open(temporary, "r+b") as handle:
                        os.fsync(handle.fileno())
                    try:
                        os.link(temporary, self.path)
                    except FileExistsError:
                        # Another cooperating creator won the race. Validate its
                        # complete index below instead of replacing it.
                        pass
                finally:
                    if hasattr(self, "_db"):
                        self._db.close()
                    Path(temporary).unlink(missing_ok=True)
            self._db = self._connect(self.path)
            if self._db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
                raise IndexFormatError("This file is not an EUHNN index; it was not modified.")
            if self._db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise IndexFormatError("Unsupported index version; an explicit migration is required.")
            try:
                meta = dict(self._db.execute("SELECT key,value FROM metadata"))
                self.config = IndexConfig.from_dict(json.loads(meta["config"]))
                model = bytes(meta["optical_model"])
                if _digest(model) != meta["model_sha256"]:
                    raise IndexFormatError("Optical model checksum mismatch.")
            except (KeyError, TypeError, ValueError) as exc:
                raise IndexFormatError("The stored index configuration is invalid.") from exc
            if config is not None and config != self.config:
                raise ValidationError("Requested configuration differs from the existing index.")
            self.encoder = OpticalEncoder(self.config, backend=backend, model=model)
            if self._db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise IndexFormatError("SQLite integrity check failed.")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
        except Exception:
            if hasattr(self, "_db"):
                self._db.close()
            self._closed = True
            raise

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        """Open only an existing file with bounded waits and explicit transactions."""
        connection = sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, timeout=30, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _create(self) -> None:
        statements = [
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value BLOB NOT NULL)",
            """CREATE TABLE documents(id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL,
                 title TEXT NOT NULL, fingerprint TEXT NOT NULL, metadata_json TEXT NOT NULL)""",
            """CREATE TABLE chunks(id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,
                 document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 text TEXT NOT NULL, page INTEGER, start INTEGER NOT NULL, end INTEGER NOT NULL,
                 line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                 vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL)""",
            "CREATE INDEX chunks_document ON chunks(document_id)",
            "CREATE VIRTUAL TABLE passages USING fts5(text, tokenize='unicode61 remove_diacritics 2')",
            """CREATE TABLE teaching(query_hash TEXT PRIMARY KEY, query TEXT NOT NULL,
                 chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                 vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL)""",
        ]
        with self._transaction():
            for statement in statements:
                self._db.execute(statement)
            model = self.encoder.dump()
            self._db.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                [
                    ("config", _json(self.config.to_dict())),
                    ("optical_model", model),
                    ("model_sha256", _digest(model)),
                    ("generation", "0"),
                ],
            )
            self._db.execute(f"PRAGMA application_id={APP_ID}")
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self, *, write: bool = True) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise EUHNNError("The index is closed.")
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def _bump_generation(self) -> None:
        self._db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='generation'")
        self._readout_generation = -1

    def close(self) -> None:
        """Release the connection. Repeated close calls are safe."""
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def __enter__(self) -> "HolographicIndex":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _vector(self, blob: bytes, checksum: str) -> np.ndarray:
        if not isinstance(blob, bytes) or len(blob) != self.config.dimensions * 4 or _digest(blob) != checksum:
            raise IndexFormatError("Stored optical signature checksum or length mismatch.")
        vector = np.frombuffer(blob, dtype="<f4")
        if not np.isfinite(vector).all():
            raise IndexFormatError("Stored signature contains non-finite values.")
        return vector

    def ingest_text(
        self, text: str, *, source: str = "manual", title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Index exact Unicode source text without any remote service calls."""
        bounded_text(text, "document text", 64 * 1024 * 1024)
        return self.ingest_pages(
            [TextPage(text)],
            source=source,
            title=title,
            content_fingerprint=fingerprint(text.encode()),
            metadata=metadata,
        )

    def ingest_file(
        self, path: str | Path, *, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data, resolved = read_local(path)
        return self.ingest_bytes(data, filename=Path(path).name, source=source or resolved, metadata=metadata)

    def ingest_bytes(
        self, data: bytes, *, filename: str, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.ingest_pages(
            iter_document(data, filename),
            source=source or filename,
            title=filename,
            content_fingerprint=fingerprint(data),
            metadata=metadata,
        )

    def _remove(self, document_id: str) -> None:
        self._db.execute(
            "DELETE FROM passages WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (document_id,)
        )
        self._db.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def _insert_chunks(self, document_id: str, digest: str, chunks: Iterable[TextChunk], *, total_existing: int) -> int:
        batch: list[TextChunk] = []
        count = 0

        def flush() -> None:
            nonlocal count
            if not batch:
                return
            vectors = self.encoder.encode([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors):
                chunk_id = _digest(f"{document_id}:{digest}:{count}".encode())[:32]
                binary = np.asarray(vector, dtype="<f4").tobytes()
                cursor = self._db.execute(
                    """INSERT INTO chunks(chunk_id,document_id,text,page,start,end,
                    line_start,line_end,ordinal,vector,vector_sha256,text_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chunk_id,
                        document_id,
                        chunk.text,
                        chunk.page,
                        chunk.start,
                        chunk.end,
                        chunk.line_start,
                        chunk.line_end,
                        count,
                        binary,
                        _digest(binary),
                        _digest(chunk.text.encode()),
                    ),
                )
                self._db.execute("INSERT INTO passages(rowid,text) VALUES(?,?)", (cursor.lastrowid, chunk.text))
                count += 1
            batch.clear()

        for chunk in chunks:
            bounded_text(chunk.text, "chunk text", MAX_CHUNK_CHARACTERS)
            if total_existing + count + len(batch) >= MAX_CHUNKS:
                raise ValidationError("This index is limited to one million passages.")
            batch.append(chunk)
            if len(batch) >= 32:
                flush()
        flush()
        return count

    def ingest_pages(
        self,
        pages: Iterable[TextPage],
        *,
        source: str,
        title: str | None = None,
        content_fingerprint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Consume pages in bounded batches and replace a source atomically."""
        bounded_text(source, "source", 4096)
        title = source if title is None else title
        bounded_text(title, "title", 4096)
        encoded_metadata = _metadata(metadata)
        if content_fingerprint is not None:
            if (
                not isinstance(content_fingerprint, str)
                or len(content_fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in content_fingerprint)
            ):
                raise ValidationError("content_fingerprint must be a SHA-256 hex digest.")
        digest = content_fingerprint or _digest(os.urandom(32))
        document_id = _digest(source.encode())[:32]
        started = time.perf_counter()
        with self._transaction():
            old = self._db.execute("SELECT * FROM documents WHERE source=?", (source,)).fetchone()
            if old and content_fingerprint and old["fingerprint"] == digest:
                self._db.execute(
                    "UPDATE documents SET title=?,metadata_json=? WHERE id=?", (title, encoded_metadata, document_id)
                )
                count = self._db.execute("SELECT count(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()[
                    0
                ]
                return {
                    "document_id": document_id,
                    "source": source,
                    "chunks": count,
                    "unchanged": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    **self.encoder.backend.info(),
                }
            if not old and self._db.execute("SELECT count(*) FROM documents").fetchone()[0] >= MAX_DOCUMENTS:
                raise ValidationError("This index is limited to 100000 documents.")
            self._remove(document_id)
            self._db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?)", (document_id, source, title, digest, encoded_metadata)
            )
            total = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]

            def chunks() -> Iterator[TextChunk]:
                size = 0
                count_pages = 0
                for page in pages:
                    if not isinstance(page, TextPage):
                        raise ValidationError("All pages must be TextPage instances.")
                    bounded_text(page.text, "page text", 64 * 1024 * 1024, empty=True)
                    size += len(page.text)
                    count_pages += 1
                    if size > 64 * 1024 * 1024 or count_pages > 5000:
                        raise ValidationError("A document exceeds the 64 Mi character / 5000 page limit.")
                    yield from iter_chunks(page, self.config.chunk_words, self.config.overlap_words)

            count = self._insert_chunks(document_id, digest, chunks(), total_existing=total)
            if count == 0:
                raise ValidationError("The document contains no indexable text.")
            self._bump_generation()
        return {
            "document_id": document_id,
            "source": source,
            "chunks": count,
            "unchanged": False,
            "elapsed_seconds": time.perf_counter() - started,
            **self.encoder.backend.info(),
        }

    def delete_document(self, document_id: str) -> bool:
        """Remove a document, its lexical postings, signatures and teaching data."""
        bounded_text(document_id, "document_id", 64)
        with self._transaction():
            exists = self._db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone() is not None
            if exists:
                self._remove(document_id)
                self._bump_generation()
            return exists

    def documents(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        bounded_int(limit, "limit", 1, 1000)
        bounded_int(offset, "offset", 0, MAX_DOCUMENTS)
        with self._transaction(write=False):
            rows = self._db.execute(
                """SELECT d.*,count(c.id) AS chunks FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.source LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            return [
                {
                    "document_id": r["id"],
                    "source": r["source"],
                    "title": r["title"],
                    "chunks": r["chunks"],
                    "metadata": json.loads(r["metadata_json"]),
                }
                for r in rows
            ]

    def stats(self) -> dict[str, Any]:
        with self._transaction(write=False):
            result = {
                name: self._db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                for name in ("documents", "chunks", "teaching")
            }
            result.update(
                {
                    "format_version": SCHEMA_VERSION,
                    "dimensions": self.config.dimensions,
                    "config": self.config.to_dict(),
                    **self.encoder.backend.info(),
                }
            )
            return result

    def _load_readout(self) -> OpticalReadout | None:
        generation = int(self._db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])
        if self._readout_generation != generation:
            rows = self._db.execute("SELECT chunk_id,vector,vector_sha256 FROM teaching ORDER BY query_hash").fetchall()
            self._readout = (
                OpticalReadout(
                    np.stack([self._vector(r["vector"], r["vector_sha256"]) for r in rows]),
                    [r["chunk_id"] for r in rows],
                )
                if rows
                else None
            )
            self._readout_generation = generation
        return self._readout

    def learn(self, query: str, chunk_id: str) -> dict[str, Any]:
        """Teach one explicit query-to-existing-passage association persistently."""
        bounded_text(query, "teaching query", 4096)
        bounded_text(chunk_id, "chunk_id", 64)
        vector = self.encoder.encode([query])[0]
        if not np.any(vector):
            raise ValidationError("A teaching query must contain non-stopword terms.")
        binary = np.asarray(vector, dtype="<f4").tobytes()
        key = _digest(query.encode())
        with self._transaction():
            if not self._db.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone():
                raise ValidationError("The selected teaching passage does not exist.")
            if (
                not self._db.execute("SELECT 1 FROM teaching WHERE query_hash=?", (key,)).fetchone()
                and self._db.execute("SELECT count(*) FROM teaching").fetchone()[0] >= 256
            ):
                raise ValidationError("The ridge readout supports at most 256 teaching examples.")
            self._db.execute(
                "INSERT OR REPLACE INTO teaching VALUES(?,?,?,?,?)", (key, query, chunk_id, binary, _digest(binary))
            )
            self._bump_generation()
            readout = self._load_readout()
            score = readout.predict(vector).get(chunk_id, 0) if readout else 0
        return {"chunk_id": chunk_id, "training_score": score, "model": "supervised optical ridge readout"}

    def _hit(
        self, row: sqlite3.Row, *, score: float, lexical: float = 0, optical: float = 0, learned: float = 0
    ) -> SearchHit:
        if _digest(row["text"].encode()) != row["text_sha256"]:
            raise IndexFormatError("Stored passage text checksum mismatch.")
        return SearchHit(
            row["chunk_id"],
            row["document_id"],
            row["source"],
            row["title"],
            row["text"],
            row["page"],
            row["start"],
            row["end"],
            row["line_start"],
            row["line_end"],
            float(score),
            float(lexical),
            float(optical),
            float(learned),
            json.loads(row["metadata_json"]),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: str = "hybrid",
        source: str | None = None,
        phrase: bool = False,
        candidate_limit: int = 256,
    ) -> list[SearchHit]:
        """Retrieve exact passages. Scores rank results; they are not confidence probabilities."""
        bounded_text(query, "query", 4096, empty=True)
        bounded_int(top_k, "top_k", 1, 100)
        bounded_int(candidate_limit, "candidate_limit", top_k, 4096)
        if mode not in ("hybrid", "lexical", "optical"):
            raise ValidationError("mode must be hybrid, lexical or optical.")
        if source is not None:
            bounded_text(source, "source", 4096)
        expression = fts_query(query, phrase=phrase)
        if not expression:
            return []
        if phrase and mode == "optical":
            raise ValidationError("Phrase filtering requires lexical or hybrid mode.")
        query_vector = self.encoder.encode([query])[0] if mode != "lexical" else None
        with self._transaction(write=False):
            return self._search(query_vector, expression, top_k, mode, source, phrase, candidate_limit)

    def _search(
        self,
        query_vector: np.ndarray | None,
        expression: str,
        top_k: int,
        mode: str,
        source: str | None,
        phrase: bool,
        candidate_limit: int,
    ) -> list[SearchHit]:
        select = "SELECT c.*,d.source,d.title,d.metadata_json FROM chunks c JOIN documents d ON d.id=c.document_id"
        filter_sql = " AND d.source=?" if source is not None else ""
        filter_args = (source,) if source is not None else ()
        lexical_rows: list[sqlite3.Row] = []
        if mode != "optical":
            lexical_rows = self._db.execute(
                """SELECT c.*,d.source,d.title,d.metadata_json,bm25(passages) AS bm25
                FROM passages JOIN chunks c ON c.id=passages.rowid JOIN documents d ON d.id=c.document_id
                WHERE passages MATCH ?"""
                + filter_sql
                + " ORDER BY bm25(passages),c.chunk_id LIMIT ?",
                (expression, *filter_args, candidate_limit),
            ).fetchall()
        if mode == "lexical":
            return [
                self._hit(row, score=1 / (i + 1), lexical=-row["bm25"]) for i, row in enumerate(lexical_rows[:top_k])
            ]
        assert query_vector is not None
        if not np.any(query_vector):
            # A quoted stopword-only phrase can have exact lexical evidence even
            # though its optical illumination is zero. Preserve that evidence.
            return [
                self._hit(row, score=1 / (i + 1), lexical=-row["bm25"]) for i, row in enumerate(lexical_rows[:top_k])
            ]
        readout = self._load_readout() if not phrase else None
        learned = readout.predict(query_vector) if readout else {}
        candidates = {row["chunk_id"]: row for row in lexical_rows}
        for key in learned:
            row = self._db.execute(select + " WHERE c.chunk_id=?" + filter_sql, (key, *filter_args)).fetchone()
            if row is not None:
                candidates[key] = row
        optical_scores: dict[str, float] = {}
        if mode == "optical":
            cursor = self._db.execute(select + " WHERE 1=1" + filter_sql, filter_args)
            best: list[tuple[float, str, sqlite3.Row]] = []
            xp = self.encoder.backend.xp
            device_query = xp.asarray(query_vector)
            while True:
                rows = cursor.fetchmany(256)
                if not rows:
                    break
                matrix = np.stack([self._vector(r["vector"], r["vector_sha256"]) for r in rows])
                scores = self.encoder.backend.host(xp.asarray(matrix) @ device_query)
                for row, value in zip(rows, scores):
                    score = float(np.clip(value, -1, 1))
                    if score <= 0:
                        continue
                    entry = (score, row["chunk_id"], row)
                    if len(best) < candidate_limit:
                        heapq.heappush(best, entry)
                    elif entry[:2] > best[0][:2]:
                        heapq.heapreplace(best, entry)
            for score, key, row in best:
                candidates[key] = row
                optical_scores[key] = score
        if not candidates:
            return []
        remaining = [(key, row) for key, row in candidates.items() if key not in optical_scores]
        if remaining:
            matrix = np.stack([self._vector(r["vector"], r["vector_sha256"]) for _, r in remaining])
            xp = self.encoder.backend.xp
            scores = self.encoder.backend.host(xp.asarray(matrix) @ xp.asarray(query_vector))
            optical_scores.update({key: float(np.clip(value, -1, 1)) for (key, _), value in zip(remaining, scores)})
        optical_order = sorted(candidates, key=lambda key: (-optical_scores[key], key))
        optical_ranks = {key: rank + 1 for rank, key in enumerate(optical_order)}
        # Preserve BM25's relevance margin. Reciprocal rank alone made a strong
        # rare-identifier match almost indistinguishable from the next common-word
        # hit, allowing a lossy optical projection to displace exact evidence.
        lexical_strength = {row["chunk_id"]: max(0.0, -float(row["bm25"])) for row in lexical_rows}
        strongest_lexical = max(lexical_strength.values(), default=0.0)
        ranked = []
        for key, row in candidates.items():
            lex = lexical_strength.get(key, 0.0) / strongest_lexical if strongest_lexical else 0.0
            opt = optical_scores[key]
            base = opt if mode == "optical" else 0.65 * lex + 0.35 * (61 / (60 + optical_ranks[key]))
            score = base + 0.8 * learned.get(key, 0.0)
            ranked.append(self._hit(row, score=score, lexical=lex, optical=opt, learned=learned.get(key, 0)))
        ranked.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return ranked[:top_k]

    def context(self, query: str, *, top_k: int = 5, max_characters: int = 12000, mode: str = "hybrid") -> str:
        """Format retrieved quotations for a caller-owned RAG model, with citations."""
        bounded_int(max_characters, "max_characters", 256, 100000)
        blocks: list[str] = []
        used = 0
        for number, hit in enumerate(self.search(query, top_k=top_k, mode=mode), 1):
            prefix = f"[{number}] {hit.citation}\n"
            room = max_characters - used - len(prefix) - 2
            if room <= 0:
                break
            blocks.append(prefix + hit.text[:room])
            used += len(blocks[-1]) + 2
        return "\n\n".join(blocks)

    def export_hologram(self) -> bytes:
        """Export exact passages/metadata, not lossy learned signatures.

        The export includes source configuration; import recomputes signatures.
        A 32 MiB export limit is explicit even for larger on-disk indexes.
        """
        with self._transaction(write=False):
            docs = []
            used = 0
            for doc in self._db.execute("SELECT * FROM documents ORDER BY source"):
                chunks = []
                for row in self._db.execute("SELECT * FROM chunks WHERE document_id=? ORDER BY ordinal", (doc["id"],)):
                    if _digest(row["text"].encode()) != row["text_sha256"]:
                        raise IndexFormatError("Cannot export a corrupted passage.")
                    chunk = {k: row[k] for k in ("text", "page", "start", "end", "line_start", "line_end", "ordinal")}
                    used += len(_json(chunk).encode())
                    if used > MAX_RAW_BYTES:
                        raise ValidationError(
                            "The holographic export exceeds 32 MiB; export a smaller document collection."
                        )
                    chunks.append(chunk)
                docs.append(
                    {
                        "source": doc["source"],
                        "title": doc["title"],
                        "fingerprint": doc["fingerprint"],
                        "metadata": json.loads(doc["metadata_json"]),
                        "chunks": chunks,
                    }
                )
            teaching = [
                {"query": row["query"], "chunk_id": row["chunk_id"]}
                for row in self._db.execute("SELECT query,chunk_id FROM teaching ORDER BY query_hash")
            ]
            payload = {
                "format": 2,
                "kind": "euhnn-document-memory",
                "config": self.config.to_dict(),
                "documents": docs,
                "teaching": teaching,
            }
        return encode_memory(payload)

    def import_hologram(self, blob: bytes) -> dict[str, int]:
        """Import a fully validated memory atomically, retaining exact source spans."""
        payload = decode_memory(blob)
        expected = {"format", "kind", "config", "documents", "teaching"}
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload["format"] != 2
            or payload["kind"] != "euhnn-document-memory"
        ):
            raise ValidationError("The hologram is not an EUHNN document memory.")
        imported_config = IndexConfig.from_dict(payload["config"])
        if imported_config != self.config:
            raise ValidationError("Imported memory configuration differs from this index.")
        documents, teaching = payload["documents"], payload["teaching"]
        if (
            not isinstance(documents, list)
            or len(documents) > MAX_DOCUMENTS
            or not isinstance(teaching, list)
            or len(teaching) > 256
        ):
            raise ValidationError("Invalid memory collection size.")
        seen: set[str] = set()
        validated = []
        for doc in documents:
            if not isinstance(doc, dict) or set(doc) != {"source", "title", "fingerprint", "metadata", "chunks"}:
                raise ValidationError("Invalid imported document schema.")
            source = bounded_text(doc["source"], "source", 4096)
            title = bounded_text(doc["title"], "title", 4096)
            digest = doc["fingerprint"]
            if (
                source in seen
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValidationError("Duplicate source or invalid document digest.")
            seen.add(source)
            meta = _metadata(doc["metadata"])
            if not isinstance(doc["chunks"], list) or not doc["chunks"]:
                raise ValidationError("An imported document must contain passages.")
            chunks = []
            for ordinal, c in enumerate(doc["chunks"]):
                if not isinstance(c, dict) or set(c) != {
                    "text",
                    "page",
                    "start",
                    "end",
                    "line_start",
                    "line_end",
                    "ordinal",
                }:
                    raise ValidationError("Invalid imported passage schema.")
                text = bounded_text(c["text"], "chunk text", MAX_CHUNK_CHARACTERS)
                for key in ("start", "end", "ordinal"):
                    bounded_int(c[key], key, 0, 64 * 1024 * 1024)
                for key in ("line_start", "line_end"):
                    bounded_int(c[key], key, 1, 64 * 1024 * 1024)
                if c["page"] is not None:
                    bounded_int(c["page"], "page", 1, 5000)
                if (
                    c["end"] - c["start"] != len(text)
                    or c["ordinal"] != ordinal
                    or c["line_end"] != c["line_start"] + text.count("\n")
                ):
                    raise ValidationError("Imported passage offsets do not match its text.")
                chunks.append(TextChunk(**c))
            validated.append((source, title, digest, meta, chunks))
        for item in teaching:
            if not isinstance(item, dict) or set(item) != {"query", "chunk_id"}:
                raise ValidationError("Invalid imported teaching schema.")
            bounded_text(item["query"], "teaching query", 4096)
            bounded_text(item["chunk_id"], "chunk_id", 64)
        count = 0
        with self._transaction():
            for source, title, digest, meta, chunks in validated:
                doc_id = _digest(source.encode())[:32]
                self._remove(doc_id)
                if self._db.execute("SELECT count(*) FROM documents").fetchone()[0] >= MAX_DOCUMENTS:
                    raise ValidationError("Document limit exceeded.")
                self._db.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (doc_id, source, title, digest, meta))
                total = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]
                count += self._insert_chunks(doc_id, digest, chunks, total_existing=total)
            for item in teaching:
                vector = self.encoder.encode([item["query"]])[0]
                if (
                    not np.any(vector)
                    or not self._db.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (item["chunk_id"],)).fetchone()
                ):
                    raise ValidationError("Imported teaching data references invalid text or a missing passage.")
                key = _digest(item["query"].encode())
                binary = np.asarray(vector, dtype="<f4").tobytes()
                self._db.execute(
                    "INSERT OR REPLACE INTO teaching VALUES(?,?,?,?,?)",
                    (key, item["query"], item["chunk_id"], binary, _digest(binary)),
                )
            if self._db.execute("SELECT count(*) FROM teaching").fetchone()[0] > 256:
                raise ValidationError("Teaching limit exceeded.")
            self._bump_generation()
        return {"documents": len(validated), "chunks": count, "teaching": len(teaching)}

    def verify(self) -> dict[str, Any]:
        """Check database, source/vector checksums, foreign keys and lexical alignment."""
        with self._transaction():
            self._db.execute("INSERT INTO passages(passages) VALUES('integrity-check')")
            if (
                self._db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or self._db.execute("PRAGMA foreign_key_check").fetchone()
            ):
                raise IndexFormatError("Database integrity or foreign-key check failed.")
            chunks = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if self._db.execute("SELECT count(*) FROM passages").fetchone()[0] != chunks:
                raise IndexFormatError("Lexical posting count does not match stored passages.")
            bad = self._db.execute(
                "SELECT 1 FROM chunks c LEFT JOIN passages p ON p.rowid=c.id WHERE p.rowid IS NULL OR p.text != c.text LIMIT 1"
            ).fetchone()
            if bad:
                raise IndexFormatError("Lexical postings do not match source text.")
            for row in self._db.execute("SELECT text,text_sha256,vector,vector_sha256 FROM chunks"):
                if _digest(row["text"].encode()) != row["text_sha256"]:
                    raise IndexFormatError("Passage checksum mismatch.")
                self._vector(row["vector"], row["vector_sha256"])
            self._load_readout()
            return {
                "ok": True,
                "verified_passages": chunks,
                "format_version": SCHEMA_VERSION,
                "integrity_scope": "local corruption checks; not authenticated tamper resistance",
            }

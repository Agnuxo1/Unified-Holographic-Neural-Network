# Changelog

## 2.0.0 - 2026-09-17

Added a complete installable Python package and local document workbench. The new
core provides deterministic content-dependent RGB illumination, actual CPU/CUDA
segment-sphere tracing, coherent propagation, nonlinear detector signatures and a
persistent supervised ridge readout. It does not use an external embedding/model API.

Added exact-source document retrieval with SQLite FTS5/BM25, bounded hybrid reranking,
explicit optical scanning, stable source/page/line/character citations, transactional
replacement and deletion, idempotent ingestion, corruption checks, Unicode chunking
that retains document tails, and text PDF/DOCX/HTML/source-code readers.

Added reversible RGB-phase/Fourier memory export and import with bounded decompression,
strict schemas, checksums and exact reconstruction. Added a complete CLI, automatic
isolated source launcher, authenticated loopback server and bundled responsive UI.

Added native integrations for ten frameworks, including real retrievers and serialized
pipelines. Added numerical, persistence, rollback, input-boundary, native-execution,
real Chromium and mandatory-CUDA checks. Added a reproducible 2,000-page fixture,
explicit BM25 baseline and synchronized CPU/CUDA timing evidence.

The long-document fixture revealed that rank-only fusion displaced strong exact
identifier matches. Hybrid ranking now preserves normalized BM25 relevance margins;
a regression test and both the initial and corrected results are retained.

Preserved the original regular source files in a verified version archive and retained
legacy demo folders, author attribution and original public contest-history material.
New documentation separates measured behavior from historical experimental claims.

# Contributing

Start with a reproducible problem or a scoped implementation change. Include tests,
source-provenance checks and documentation for any changed numerical/storage behavior.
Keep real credentials, private documents, local databases and generated environments
out of commits. Use synthetic fixtures. Do not reinterpret scores as probabilities.

Run `python scripts/run-tests.py tests -q` in an environment containing the relevant
extras. Tests must execute native operations for an integration claim. CUDA evidence
must require actual hardware rather than silently skipping. Benchmark changes must
retain the BM25 baseline, method, dataset checksum, all relevant modes and limitations.
A screenshot must show the running application, not a fabricated interface.

Preserve historical versions and source attribution. Do not modify legacy demos merely
to make them appear to have passed v2 tests. Storage-format changes require explicit
migration behavior; never silently reset a user's index.

Communication must remain technical and relevant. No mass promotional comments,
repeated bumps, unsolicited cross-repository issues or requests for artificial stars.
Respect a maintainer's refusal and do not reopen or recontact declined/closed threads.

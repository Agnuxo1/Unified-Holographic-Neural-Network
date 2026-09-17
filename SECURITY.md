# Security and data handling

EUHNN 2.0 is a single-user local document tool. It is not a public, multi-tenant
hosting platform or an isolation boundary for arbitrary hostile PDF parsers.
Keep it on loopback and process documents you intentionally choose to trust.

API requests require a random token; Host and Origin are checked. Authentication
precedes bounded body reads. The browser cannot ask the API to fetch remote URLs
or read arbitrary local paths. Static assets are bundled. Source text is rendered
as text rather than executable HTML. No external language model is called by the core.

The SQLite library and `.holo` exports contain recoverable source text. Holographic
encoding and SHA-256 checks are not encryption, digital signatures or proof of
ownership. Protect their filesystem permissions and use trusted backups. The
connection token authorizes reading, adding, deleting, teaching and exporting data.
Anyone who obtains it has access to that local library while the server is running.

Native adapters intentionally return passages to the calling application. Do not
send confidential passages to a remote model without authorization. Treat document
instructions as untrusted data, not as permission to execute code or change tools.
Native graph checkpoints and caller logs may retain source text. Restore serialized
pipeline configurations only from trusted operators because they select index paths.

Resource limits reduce accidental overuse; they do not make complex PDF/DOCX parsers
invulnerable to all malicious inputs. No OCR, automatic internet crawling, system
command execution from documents, or P2P sharing is enabled.

For a vulnerability, use the repository's private vulnerability-reporting facility
when available. Do not post live tokens, private documents or exploit-bearing files
in a public issue. When private reporting is unavailable, open a minimal issue asking
for a private contact route without disclosing the vulnerability or sensitive data.

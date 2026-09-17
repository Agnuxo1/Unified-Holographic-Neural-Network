# Legacy source, attribution and migration

The source at commit `ca2cf440280f5994dfcc1f0729f0721f0c897c47` was preserved before
implementing v2. Its regular files were compared byte-for-byte against their Git blobs.

- [Original source archive](../versions/legacy-2024/source-ca2cf44.tar.gz)
- [Archive manifest, SHA-256 and Gitlink references](../versions/legacy-2024/MANIFEST.json)
- The original `README.md` and its 2024 images/links are preserved inside the verified source archive.

The old JavaScript/TypeScript and Python demo folders remain in the repository.
The legacy `python_version` Gitlink records another commit reference, not embedded
source. The archive does not invent missing submodule contents. No private sibling
repository or private award file was copied into this public release.

The historical README contains broad experimental descriptions, placeholder setup
URLs and claims not established by the v2 tests. It is preserved as historical
material, not used as evidence of v2 RT-core execution, quantum hardware, reduced
power consumption, P2P production readiness or a fresh NVIDIA award/certification.
The author attribution and the published 2024 project/contest history are retained.

Public historical record should be read carefully: NVIDIA's developer forum preserves the
initial prize-notification discussion and the later disqualification notice. The current official
contest winners page does not list EUHNN. V2 therefore describes this as contest history, not as
a current NVIDIA endorsement or certification.

- NVIDIA forum history: https://forums.developer.nvidia.com/t/winner-nvidia-and-llamaindex-developers-2024/317943
- Current contest page: https://developer.nvidia.com/llamaindex-developer-contest/

## Upgrade path

V2 is a new, documented storage format and installable application. Keep the old
checkout and exports. Reimport original documents into v2 and verify their source
quotations. Do not reinterpret an old experimental memory array as a v2 `.holo`
container; only checked v2 containers are imported. The legacy provider/P2P demos
remain separate, rather than starting network sharing as a side effect of an upgrade.

The current supported launch path is `Start-EUHNN.cmd`, `start-euhnn.sh` or
`python -m euhnn`. Root `requirements.txt` now points at the actual v2 workbench;
the previous requirements are retained in the legacy archive.

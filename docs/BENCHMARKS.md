# EUHNN 2.0 benchmark report

This report documents reproducible measurements from `scripts/benchmark.py`.
It is a synthetic functional and scaling fixture, not an independent semantic-search benchmark.
No private documents, remote language models, provider APIs, or paid services are used.

## Test system

The recorded final run used Windows, Python 3.12.12, NumPy 2.5.3 and an NVIDIA GeForce RTX 3090.
The CUDA path executes the project's own ray/sphere intersection and RGB phase kernel through CuPy.
It does not use NVIDIA OptiX or RT cores, and the report records `rt_cores_used: false`.

## Corpus

The fixture builds one synthetic 2,000-page maintenance manual containing 424,000 words,
3,946,105 UTF-8 bytes and 4,000 indexed passages. Twenty-four pages receive independently
addressable `UNITxxxxxx` identifiers and exact pressure-setting sentences used as retrieval targets.
The complete generated corpus is hashed before evaluation.

| Property | Recorded value |
| --- | ---: |
| Pages | 2,000 |
| Words | 424,000 |
| Passages | 4,000 |
| Questions | 24 |
| Index size | 33,611,776 bytes |
| CPU ingestion | 9.025 s |

The questions intentionally contain unique identifiers. They test long-document integrity,
source localization and retrieval plumbing; they do not establish general semantic understanding.
## Retrieval results

| Backend | Mode | Top-1 | Top-5 | Median query | p95 query |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | BM25 lexical | 24/24 | 24/24 | 24.452 ms | 25.953 ms |
| CPU | Hybrid BM25 + optical rerank | 24/24 | 24/24 | 32.200 ms | 35.801 ms |
| CPU | Experimental full optical scan | 4/24 | 5/24 | 126.548 ms | 137.694 ms |
| CUDA | Hybrid BM25 + optical rerank | 24/24 | 24/24 | 35.253 ms | 39.510 ms |
| CUDA | Experimental full optical scan | 4/24 | 5/24 | 149.947 ms | 161.019 ms |

Every returned evaluation passage was checked against the original synthetic page text;
120 exact quotations were verified for each five-result retrieval run.
The BM25 baseline remains visible because hiding it would overstate the value of the optical layer.
For this identifier-heavy fixture, BM25 is both accurate and faster end-to-end.

The optical-only result is deliberately retained. It demonstrates that deterministic
lexical/subword optical signatures are not a substitute for a pretrained semantic embedding model.
The default hybrid mode therefore uses FTS5/BM25 for bounded candidate generation and an optical
signature as a secondary rank signal.

An earlier recorded run, `audit/benchmark-initial.json`, exposed a fusion regression:
hybrid retrieval achieved only 5/24 Top-1 and 11/24 Top-5 while BM25 was 24/24.
The ranking fusion was corrected before release, and the repeated final fixture in
`audit/benchmark.json` recovered all 24 targets at rank one. The failed run is preserved
rather than deleted so the improvement remains auditable.
## Ray-kernel measurements

The 96-source, 128-detector, 18-sphere scene produces 36,864 source/detector segments per trace before RGB channel expansion. Thirty completed traces were timed.

| Backend | Median | p95 | CPU/CUDA output difference |
| --- | ---: | ---: | ---: |
| NumPy CPU | 9.912 ms | 10.133 ms | reference |
| CUDA | 0.459 ms | 0.612 ms | max complex error 0.0 in this run |

This acceleration applies to the ray calculation itself. It does not imply that a complete SQLite document search is faster on GPU. In the measured small-candidate hybrid queries, CUDA was slightly slower end-to-end because tokenization, SQLite work and host/device transfers remain part of the request.

## Reversible holographic export

The synthetic manual's 3,946,105 source bytes were encoded into an RGB phase/Fourier hologram of 323,221 bytes in the recorded run. Encoding took 0.0329 s and decoding 0.0162 s, and the reconstructed bytes matched exactly.

The unusually small exported size is caused by the highly repetitive synthetic fixture and zlib compression before phase encoding. It must not be generalized into a compression claim for arbitrary documents. The export is also not encryption or authenticated provenance.
## Reproduce

Install the workbench, test and matching CUDA extra, then run `python scripts/benchmark.py --pages 2000 --cuda`.
Use the `cuda12` or `cuda13` extra appropriate for the installed NVIDIA stack; the source launcher can choose a compatible CuPy family automatically.

Full raw measurements, individual query latencies, corpus checksum and hardware metadata are stored in `audit/benchmark.json`. The pre-correction run remains in `audit/benchmark-initial.json`.

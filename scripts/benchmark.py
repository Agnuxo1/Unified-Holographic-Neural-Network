"""Reproducible synthetic long-document retrieval and actual CPU/CUDA measurements.

This is a functional/scaling fixture, not an independent semantic retrieval
benchmark and not evidence of superiority over pretrained embedding models.
No private documents, service credentials, network downloads or model APIs are used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
from euhnn import HolographicIndex, IndexConfig, TextPage, __version__
from euhnn.hologram import encode_bytes, decode_bytes
from euhnn.optics import Backend, Geometry, trace_numpy, trace_cuda


def timing(function, repeats=10):
    """Time completed operations; CUDA functions return host arrays and synchronize."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)), "samples": repeats}


def corpus(pages):
    """Produce one long manual and questions with independently located exact answers."""
    selected = sorted(set(np.linspace(0, pages - 1, 24, dtype=int).tolist()))
    questions = []
    texts = []
    filler = (
        "The maintenance handbook records component inspection, alignment, calibration, "
        "pressure monitoring, documentation revision, optical sensor stability and quality control. "
    )
    for page in range(pages):
        code = f"UNIT{page:06d}"
        pressure = 17 + (page * 37) % 211
        needle = f"The certified pressure setting for {code} is {pressure} kilopascals."
        chunks = [filler for _ in range(11)]
        chunks.insert(page % 12, needle)
        text = f"# Service manual page {page + 1}\n" + "\n".join(chunks) + "\n"
        texts.append(text)
        if page in selected:
            questions.append({"query": f"certified pressure setting {code}", "answer": needle, "page": page + 1})
    return texts, questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=2000)
    parser.add_argument("--cuda", action="store_true", help="Require real CUDA measurements; fail if unavailable")
    parser.add_argument("--output", type=Path, default=Path("audit/benchmark.json"))
    args = parser.parse_args()
    if not 24 <= args.pages <= 5000:
        parser.error("pages must be from 24 to 5000")
    report = {
        "version": __version__,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic functional/scaling fixture, not independent semantic quality evaluation",
        "hardware": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "remote_model_calls": 0,
        "private_documents_used": False,
        "rt_cores_used": False,
    }
    config = IndexConfig()
    geometry = Geometry.create(config)
    trace_numpy(geometry, config.wavelengths)
    report["ray_tracing"] = {
        "cpu": timing(lambda: trace_numpy(geometry, config.wavelengths), 30),
        "segments_per_trace": 3 * config.source_count * config.detector_count,
        "spheres": config.sphere_count,
    }
    runtime = None
    if args.cuda:
        runtime = Backend.select("cuda")
        actual = trace_cuda(geometry, config.wavelengths, runtime)
        reference = trace_numpy(geometry, config.wavelengths)
        error = float(np.max(np.abs(actual - reference)))
        np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=2e-5)
        report["hardware"].update(runtime.info())
        report["ray_tracing"].update(
            {
                "cuda": timing(lambda: trace_cuda(geometry, config.wavelengths, runtime), 30),
                "max_cpu_cuda_complex_error": error,
            }
        )
    texts, questions = corpus(args.pages)
    raw = "".join(texts).encode()
    report["corpus"] = {
        "documents": 1,
        "pages": args.pages,
        "utf8_bytes": len(raw),
        "words": sum(len(t.split()) for t in texts),
        "sha256": sha256(raw).hexdigest(),
        "queries": len(questions),
        "config": config.to_dict(),
    }
    with tempfile.TemporaryDirectory(prefix="euhnn-benchmark-") as temporary:
        path = Path(temporary) / "manual.sqlite"
        started = time.perf_counter()
        with HolographicIndex(path, create=True, config=config, backend="cpu") as index:
            result = index.ingest_pages(
                (TextPage(t, i + 1) for i, t in enumerate(texts)),
                source="synthetic-service-manual.md",
                content_fingerprint=sha256(raw).hexdigest(),
            )
            report["corpus"]["passages"] = result["chunks"]
            report["ingestion"] = {"backend": "cpu", "elapsed_seconds": time.perf_counter() - started}
            assert index.verify()["ok"]
        report["index_bytes"] = path.stat().st_size
        report["retrieval"] = []
        for backend in ["cpu", "cuda"] if args.cuda else ["cpu"]:
            with HolographicIndex(path, backend=backend) as index:
                for mode in ("lexical", "hybrid", "optical"):
                    if backend == "cuda" and mode == "lexical":
                        continue
                    index.search(questions[0]["query"], mode=mode)
                    measurements = []
                    top1 = 0
                    top5 = 0
                    exact = 0
                    for question in questions:
                        start = time.perf_counter()
                        hits = index.search(question["query"], mode=mode, top_k=5)
                        measurements.append((time.perf_counter() - start) * 1000)
                        def relevant(hit):
                            return hit.page == question["page"] and question["answer"] in hit.text

                        top1 += int(bool(hits) and relevant(hits[0]))
                        top5 += int(any(relevant(hit) for hit in hits))
                        for hit in hits:
                            assert hit.text == texts[hit.page - 1][hit.start : hit.end]
                            exact += 1
                    report["retrieval"].append(
                        {
                            "backend": backend,
                            "mode": mode,
                            "top1_correct": top1,
                            "top5_contains_answer": top5,
                            "query_count": len(questions),
                            "median_ms": statistics.median(measurements),
                            "p95_ms": float(np.percentile(measurements, 95)),
                            "verified_exact_quotations": exact,
                            "all_query_latencies_ms": measurements,
                        }
                    )
        started = time.perf_counter()
        hologram = encode_bytes(raw)
        report["hologram"] = {
            "raw_bytes": len(raw),
            "hologram_bytes": len(hologram),
            "encoding_seconds": time.perf_counter() - started,
        }
        started = time.perf_counter()
        restored = decode_bytes(hologram)
        assert restored == raw
        report["hologram"].update({"decoding_seconds": time.perf_counter() - started, "exact_round_trip": True})
    report["interpretation"] = [
        "BM25 is an explicit baseline, not removed from reporting.",
        "Optical signatures are lexical/subword reservoir features, not pretrained semantic embeddings.",
        "Optical full scan is O(ND); hybrid limits optical work to BM25 candidates.",
        "End-to-end CUDA timings include signature transfer and CPU text/token/SQLite work.",
        "These synthetic questions deliberately contain unique unit identifiers. They do not establish general semantic accuracy.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "corpus": report["corpus"],
                "ingestion": report["ingestion"],
                "retrieval": [
                    {k: v for k, v in row.items() if k != "all_query_latencies_ms"} for row in report["retrieval"]
                ],
                "ray_tracing": report["ray_tracing"],
                "hologram": report["hologram"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

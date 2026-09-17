"""A bundled, original demonstration corpus; no private documents or APIs."""

DEMO_DOCUMENTS = {
    "optical-memory.md": """# Optical memory handbook
This handbook describes a classical simulated optical reservoir.
The red channel has wavelength 0.63, the green channel 0.53, and the blue channel 0.46 simulation units.
A ray intersects spherical phase inclusions. Refractive index changes optical path length while absorption reduces amplitude.
Detector fields combine complex amplitudes. The intensity is the squared magnitude of the combined field.
EUHNN stores exact source passages separately from lossy optical retrieval signatures.
The holographic export encodes compressed source bytes in three phase channels and stores their Fourier spectrum.
A SHA-256 checksum verifies the reconstructed byte sequence. It is not encryption or a digital signature.
""",
    "retrieval-guide.md": """# Document retrieval guide
The default hybrid search selects a bounded pool of passages using SQLite FTS5 and BM25.
An optical signature reranks that pool; it is not a pretrained language embedding.
The explicit optical mode scans every stored vector in batches and has linear collection-size cost.
Each result contains the original quotation, source name, page if available, line range and character offsets.
A teaching example links an explicit query to an existing passage. A regularized ridge readout learns these associations.
Re-importing identical source bytes is idempotent. Replacing a changed source updates passages and lexical postings in one transaction.
""",
    "railway-manual.md": """# Railway maintenance notes
The locomotive inspection interval is 120 operating hours.
The brake reservoir must be inspected before departure and after a pressure alarm.
A cracked wheel must be removed from service. Cooling fans are checked during scheduled maintenance.
The depot inventory code for the replacement coupling is RAIL-742.
""",
}


def load_demo(index):
    """Index bundled examples without deleting any user documents."""
    return [index.ingest_text(text, source="demo/" + name, title=name) for name, text in DEMO_DOCUMENTS.items()]

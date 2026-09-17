# Architecture and numerical contract

EUHNN 2.0 separates three different jobs: a **lossy optical retrieval signature**,
**exact document storage**, and a **reversible holographic export**. They are not
interchangeable representations and are not described as physical optical hardware.

## 1. Content-dependent illumination

The tokenizer keeps Unicode source text unchanged for citations. For retrieval it
case-folds, removes combining accents and extracts words, low-weight adjacent word
bigrams, and character trigrams. Stable BLAKE2b projections map these features into
two complex source locations for each of three color channels. Source illumination
is normalized. Stopword-only illumination is zero. These are lexical/subword
features; there is no pretrained semantic embedding model hidden behind the interface.

## 2. Actual straight-ray geometry

The stored scene contains source points, detector points and spherical phase
inclusions with RGB refractive indices and absorption coefficients. For source `s`,
detector `d`, channel `c`, segment length `L`, and in-sphere chord lengths `ell_j`:

```text
OPL_c = L + sum_j ((n_jc - 1) * ell_j)
tau_c = sum_j (absorption_jc * ell_j)
T_cds = exp(-tau_c) / (1 + L^2) * exp(i * 2*pi*OPL_c / wavelength_c)
```

Only the part of a sphere chord inside the finite source-to-detector segment is
counted. A NumPy implementation and a CUDA RawKernel calculate these quantities.
The matrix is column-normalized as a reservoir feature transform. This normalization
and the attenuation law are modeling choices, not a calibrated energy-conserving
physical instrument. Overlapping inclusions contribute additively in this approximation.

Ray directions remain straight: no Snell-law bending, surface reflections, multiple
scattering, polarization transport or Maxwell boundary-value solution is claimed.
The CUDA kernel does not invoke OptiX, RT cores or a hardware ray-tracing API.

Detector fields are `E_cd = sum_s T_cds * illumination_cs`. The real and imaginary
parts plus centered detector intensities form the signature. Field components have
combined squared weight 0.8 and centered intensities 0.2, followed by normalization.
With 128 detectors the stored signature has 1,152 float32 dimensions.

## 3. Persistent source retrieval

The database stores source names, titles, metadata, original text slices, Unicode
character offsets, extracted page numbers, line spans, signatures and checksums.
No source quotation is reconstructed from its lossy signature. Chunking uses bounded
look-ahead and overlaps without dropping separators or the final tail.

`lexical` mode queries FTS5/BM25. Application-generated quoted terms prevent user
input from becoming FTS operators or SQL. `hybrid` mode uses a bounded BM25 candidate
pool (256 by default) and computes optical scores only for those candidates plus
applicable explicitly taught passages.

The hybrid score is:

```text
lexical_strength = max(0, -BM25_score) / strongest_candidate_BM25_strength
optical_rank_score = 61 / (60 + optical_rank)
score = 0.65*lexical_strength + 0.35*optical_rank_score + 0.8*learned_affinity
```

Preserving BM25's relevance margin fixes a measured failure of the initial
rank-only fusion on rare identifiers. These ranking values are not confidence
probabilities and are not calibrated across independent indexes.

`optical` mode scans all signatures in batches of 256. It has **O(ND)** query cost,
not constant-time or logarithmic search. The empirical weaknesses of this mode
are reported alongside the default and baseline results.

## 4. Actual supervised learning

A teaching example binds a query to an existing passage ID. Its optical signature
and label are persisted. For training examples `X`, one-hot labels `Y`, and positive
regularization `lambda`, the dual ridge readout solves:

```text
A = solve(X*X.T + lambda*I, Y)
prediction(q) = (X*q).T * A
```

The implementation caps teaching at 256 examples, defaults to regularization 0.05,
and requires similarity at least 0.75 to an example for that label before using its
prediction. This guards against remote extrapolation, not against all false matches.
Changing/deleting a source removes stale labels. Cross-connection generation numbers
invalidate cached readouts. No random teaching response is presented as learning.

## 5. Reversible RGB/Fourier memory

Canonical source JSON is compressed with zlib. Each compressed byte is represented
as a phase on a 256-level unit circle, laid out in three padded RGB planes. An
orthonormal 2D FFT is stored as little-endian complex64. Inversion recovers phases,
rounds to bytes, decompresses with a bound and verifies the exact original SHA-256.
The numeric spectrum itself also has a checksum.

The export is not encryption, a signature, error-correcting storage or a universal
compression scheme. A complex spectrum uses eight bytes per padded phase byte;
compression gains on repetitive text come from zlib, not a holographic information
capacity claim. Maximum uncompressed export size is 32 MiB. Large index collections
can exceed that export limit and fail explicitly rather than truncate their source.

## 6. Runtime boundaries

NumPy is the only mandatory computational dependency. CUDA is optional and lazy.
The HTML/JS/CSS workbench is bundled and uses no CDN. Its server listens on loopback,
checks the Host and Origin, authenticates API calls before consuming bounded request
bodies and never accepts arbitrary local paths or remote URLs through the browser.

The package exposes local retrieval and context construction. It does not silently
send documents to an LLM, execute document instructions, or start a P2P service.
Native adapters make any onward model/data decision the calling application's responsibility.

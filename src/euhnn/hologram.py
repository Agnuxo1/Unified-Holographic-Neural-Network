"""Reversible RGB phase/Fourier storage for exact text and structured memories.

This numerical hologram is a classical encoding, not encryption or a claim of
optical hardware. Its float32 complex spectrum is larger than compressed text.
SHA-256 detects accidental corruption; it is not an authenticated signature.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import struct
import zlib
from typing import Any

import numpy as np

from .optics import Backend
from .schema import ValidationError

MAGIC = b"EUHNNH2\n"
MAX_RAW_BYTES = 32 * 1024 * 1024
MAX_HOLOGRAM_BYTES = 272 * 1024 * 1024
MAX_HEADER_BYTES = 4096


def _canonical(value: Any) -> bytes:
    """Serialize JSON deterministically, rejecting non-finite floating values."""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError("Memory must be JSON-serializable and contain valid Unicode.") from exc


def encode_bytes(data: bytes, *, backend: str = "cpu") -> bytes:
    """Store bounded bytes as three phase channels and a complex Fourier plane."""
    if not isinstance(data, bytes) or len(data) > MAX_RAW_BYTES:
        raise ValidationError("A holographic memory accepts at most 32 MiB of bytes.")
    compressed = zlib.compress(data, level=6)
    side = max(1, math.ceil(math.sqrt(len(compressed) / 3)))
    phase_bytes = np.zeros(3 * side * side, dtype=np.uint8)
    phase_bytes[: len(compressed)] = np.frombuffer(compressed, dtype=np.uint8)
    phase = phase_bytes.astype(np.float64).reshape(3, side, side) * (2 * np.pi / 256)
    runtime = Backend.select(backend)
    xp = runtime.xp
    spectrum = runtime.host(xp.fft.fft2(xp.exp(1j * xp.asarray(phase)), axes=(-2, -1), norm="ortho"))
    spectrum = np.ascontiguousarray(spectrum, dtype="<c8")
    body = spectrum.tobytes()
    header = {
        "format": 2,
        "side": side,
        "channels": 3,
        "dtype": "complex64-le",
        "compressed_bytes": len(compressed),
        "raw_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
        "spectrum_sha256": sha256(body).hexdigest(),
    }
    encoded_header = _canonical(header)
    return MAGIC + struct.pack("<I", len(encoded_header)) + encoded_header + body


def _parse(blob: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """Validate all lengths and types before allocating an FFT-sized array."""
    if not isinstance(blob, bytes) or len(blob) > MAX_HOLOGRAM_BYTES or not blob.startswith(MAGIC):
        raise ValidationError("Invalid or oversized EUHNN hologram.")
    if len(blob) < len(MAGIC) + 4:
        raise ValidationError("Truncated hologram header.")
    length = struct.unpack_from("<I", blob, len(MAGIC))[0]
    if not 1 <= length <= MAX_HEADER_BYTES:
        raise ValidationError("Invalid hologram header length.")
    start = len(MAGIC) + 4
    try:
        header = json.loads(blob[start : start + length])
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("Malformed hologram header.") from exc
    expected = {"format", "side", "channels", "dtype", "compressed_bytes", "raw_bytes", "sha256", "spectrum_sha256"}
    if not isinstance(header, dict) or set(header) != expected:
        raise ValidationError("Unsupported hologram schema.")
    for key in ("format", "side", "channels", "compressed_bytes", "raw_bytes"):
        if type(header[key]) is not int:
            raise ValidationError("Hologram dimensions must be integers.")
    if header["format"] != 2 or header["channels"] != 3 or header["dtype"] != "complex64-le":
        raise ValidationError("Unsupported hologram version or numeric representation.")
    side = header["side"]
    size = 3 * side * side
    if not 1 <= side <= math.ceil(math.sqrt((MAX_RAW_BYTES + 65536) / 3)):
        raise ValidationError("Hologram dimensions exceed the resource limit.")
    if not 0 <= header["raw_bytes"] <= MAX_RAW_BYTES or not 1 <= header["compressed_bytes"] <= size:
        raise ValidationError("Invalid hologram payload lengths.")
    for key in ("sha256", "spectrum_sha256"):
        value = header[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValidationError("Invalid hologram checksum.")
    body = memoryview(blob)[start + length :]
    if len(body) != size * 8 or sha256(body).hexdigest() != header["spectrum_sha256"]:
        raise ValidationError("Hologram spectrum checksum or length mismatch.")
    spectrum = np.frombuffer(body, dtype="<c8").reshape(3, side, side)
    if not np.isfinite(spectrum).all():
        raise ValidationError("Hologram spectrum contains non-finite numbers.")
    return header, spectrum


def decode_bytes(blob: bytes, *, backend: str = "cpu") -> bytes:
    """Invert the stored spectrum and verify the exact original byte sequence."""
    header, spectrum = _parse(blob)
    runtime = Backend.select(backend)
    xp = runtime.xp
    field = runtime.host(xp.fft.ifft2(xp.asarray(spectrum), axes=(-2, -1), norm="ortho"))
    phase = np.mod(np.angle(field), 2 * np.pi)
    recovered = (np.rint(phase * (256 / (2 * np.pi))).astype(np.uint16) % 256).astype(np.uint8)
    compressed = recovered.ravel()[: header["compressed_bytes"]].tobytes()
    try:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, header["raw_bytes"] + 1)
        if len(raw) != header["raw_bytes"] or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("Expanded payload length mismatch")
    except (zlib.error, ValueError) as exc:
        raise ValidationError("The phase-decoded payload is corrupted or exceeds its declared size.") from exc
    if sha256(raw).hexdigest() != header["sha256"]:
        raise ValidationError("The recovered memory does not match its checksum.")
    return raw


def encode_memory(value: Any, *, backend: str = "cpu") -> bytes:
    """Store a validated JSON-compatible memory in the RGB Fourier format."""
    return encode_bytes(_canonical(value), backend=backend)


def decode_memory(blob: bytes, *, backend: str = "cpu") -> Any:
    """Read JSON only, never executable pickles or object deserialization hooks."""
    try:
        return json.loads(
            decode_bytes(blob, backend=backend), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("Decoded memory is not valid finite JSON.") from exc


def preview(blob: bytes, resolution: int = 64) -> dict[str, Any]:
    """Produce a bounded RGB visualization of the stored spectral amplitudes."""
    if type(resolution) is not int or not 1 <= resolution <= 256:
        raise ValidationError("Preview resolution must be from 1 to 256.")
    header, spectrum = _parse(blob)
    side = header["side"]
    sample = np.linspace(0, side - 1, min(resolution, side), dtype=np.int64)
    image = np.log1p(np.abs(spectrum[:, sample[:, None], sample[None, :]]))
    image /= np.maximum(image.max(axis=(1, 2), keepdims=True), 1e-12)
    return {
        "width": len(sample),
        "height": len(sample),
        "rgb": np.moveaxis(image, 0, -1).tolist(),
        "stored_side": side,
        "raw_bytes": header["raw_bytes"],
        "sha256": header["sha256"],
    }

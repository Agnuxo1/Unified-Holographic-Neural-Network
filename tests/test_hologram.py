"""Reversible phase/Fourier storage, malformed headers and corruption boundaries."""

import hashlib
import pytest
from euhnn.hologram import encode_bytes, decode_bytes, encode_memory, decode_memory, preview
from euhnn.schema import ValidationError


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"x",
        b"abc",
        bytes(range(256)),
        bytes(range(256)) * 100,
        "Exact \u00f3ptica \u6f22\u5b57 \U0001f308\n".encode(),
    ],
    ids=["empty", "one-byte", "three-bytes", "all-byte-values", "repeated-byte-values", "unicode"],
)
def test_exact_bytes_round_trip(data):
    stored = encode_bytes(data)
    assert decode_bytes(stored) == data
    assert len(stored) > len(data) or len(data) > 1000


def test_memory_preserves_nested_structures_and_colors():
    value = {"words": ["\u00f3ptica", "memory"], "rgb": [0.1, 0.2, 0.3], "nested": {"a": None, "b": True}}
    assert decode_memory(encode_memory(value)) == value


@pytest.mark.parametrize("part", [0, 4, 9, 20, -1, -20])
def test_corrupt_holograms_fail_closed(part):
    blob = bytearray(encode_bytes(b"test memory" * 100))
    blob[part] ^= 255
    with pytest.raises(ValidationError):
        decode_bytes(bytes(blob))


@pytest.mark.parametrize("length", [0, 1, 8, 10, 12, 100])
def test_truncated_holograms(length):
    with pytest.raises(ValidationError):
        decode_bytes(encode_bytes(b"source")[:length])


def test_preview_is_bounded_and_three_channel():
    result = preview(encode_bytes(bytes(range(256)) * 500), resolution=32)
    assert result["width"] <= 32 and result["height"] <= 32
    assert all(len(pixel) == 3 for row in result["rgb"] for pixel in row)
    assert result["sha256"] == hashlib.sha256(bytes(range(256)) * 500).hexdigest()


def test_non_finite_json_is_rejected():
    with pytest.raises(ValidationError):
        encode_memory({"x": float("nan")})

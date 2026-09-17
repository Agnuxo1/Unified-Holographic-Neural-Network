"""GPU tests execute real CUDA ray intersections, propagation and Fourier storage."""

import os
import numpy as np
import pytest
from euhnn import HolographicIndex
from euhnn.optics import Backend, CUDAUnavailable, Geometry, OpticalEncoder, trace_cuda, trace_numpy
from euhnn.hologram import encode_bytes, decode_bytes

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def cuda():
    try:
        return Backend.select("cuda")
    except CUDAUnavailable:
        if os.environ.get("EUHNN_REQUIRE_CUDA") == "1":
            raise
        pytest.skip("CUDA device/runtime not available; set EUHNN_REQUIRE_CUDA=1 for mandatory hardware validation")


def test_cuda_ray_sphere_intersections_match_reference(cuda, config):
    geometry = Geometry.create(config)
    actual = trace_cuda(geometry, config.wavelengths, cuda)
    expected = trace_numpy(geometry, config.wavelengths)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-5)
    assert cuda.name == "cuda" and cuda.device and not cuda.info()["rt_cores_used"]


def test_cuda_feature_propagation_matches_cpu(cuda, config):
    cpu = OpticalEncoder(config, backend="cpu")
    gpu = OpticalEncoder(config, backend="cuda")
    texts = [
        "RGB phase holographic memory",
        "railway locomotive maintenance",
        "\u00f3ptica azul memoria",
        "the and of",
        "",
    ]
    np.testing.assert_allclose(cpu.transfer, gpu.transfer, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(cpu.encode(texts), gpu.encode(texts), atol=2e-5, rtol=2e-4)
    assert gpu.backend.name == "cuda"


def test_cuda_hologram_round_trip_and_cpu_interchange(cuda):
    payload = bytes(range(256)) * 1024 + b"exact GPU Fourier payload"
    assert decode_bytes(encode_bytes(payload, backend="cuda"), backend="cpu") == payload
    assert decode_bytes(encode_bytes(payload, backend="cpu"), backend="cuda") == payload


def test_cuda_persistent_index_reopens_and_searches_on_cpu(cuda, tmp_path, config):
    path = tmp_path / "gpu.sqlite"
    with HolographicIndex(path, create=True, config=config, backend="cuda") as gpu:
        gpu.ingest_text(
            "The blue optical wavelength is 0.46. RGB phase memory retains source citations.", source="optics"
        )
        gpu.ingest_text("Steam railway locomotive brake reservoir maintenance procedure.", source="railway")
        assert gpu.encoder.backend.name == "cuda"
        found = gpu.search("blue optical wavelength", mode="optical", top_k=1)[0]
        assert found.source == "optics"
    with HolographicIndex(path, backend="cpu") as cpu:
        match = cpu.search("blue optical wavelength", mode="optical", top_k=1)[0]
        assert match.chunk_id == found.chunk_id
        assert abs(match.score - found.score) < 2e-5
        assert cpu.verify()["ok"]

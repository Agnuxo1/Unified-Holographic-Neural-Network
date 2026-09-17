"""Numerical reference, actual geometry and persisted-model regression tests."""

import numpy as np
import pytest
from euhnn.optics import Geometry, OpticalEncoder, OpticalReadout, Backend, trace_numpy
from euhnn.schema import ValidationError, IndexFormatError


def test_exact_straight_ray_chord_matches_analytic_solution():
    scene = Geometry(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 6.0]]),
        np.array([[0.0, 0.0, 3.0, 0.5]]),
        np.array([[1.5, 1.5, 1.5]]),
        np.array([[0.1, 0.1, 0.1]]),
    )
    result = trace_numpy(scene, (0.63, 0.53, 0.46))[:, 0, 0]
    expected = np.exp(-0.1) / 37 * np.exp(1j * 2 * np.pi * 6.5 / np.array([0.63, 0.53, 0.46]))
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_no_intersection_keeps_vacuum_path():
    scene = Geometry(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 6.0]]),
        np.array([[5.0, 0.0, 3.0, 0.5]]),
        np.array([[1.5, 1.5, 1.5]]),
        np.array([[0.1, 0.1, 0.1]]),
    )
    expected = np.exp(1j * 2 * np.pi * 6 / np.array([0.63, 0.53, 0.46])) / 37
    np.testing.assert_allclose(trace_numpy(scene, (0.63, 0.53, 0.46))[:, 0, 0], expected, atol=1e-8)


def test_real_signatures_are_content_dependent_deterministic_and_finite(config):
    encoder = OpticalEncoder(config, backend="cpu")
    texts = ["holographic memory", "holographic memory", "railway locomotive", "the and", ""]
    vectors = encoder.encode(texts)
    assert vectors.shape == (5, config.dimensions)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert np.dot(vectors[0], vectors[2]) < 0.9
    np.testing.assert_allclose(np.linalg.norm(vectors[:3], axis=1), 1, atol=1e-6)
    assert not vectors[3:].any()
    assert np.isfinite(vectors).all()
    assert encoder.encode([]).shape == (0, config.dimensions)


def test_model_reopen_reproduces_exact_cpu_features(config):
    first = OpticalEncoder(config, backend="cpu")
    second = OpticalEncoder(config, backend="cpu", model=first.dump())
    np.testing.assert_array_equal(first.encode(["memory diffraction"]), second.encode(["memory diffraction"]))
    assert first.model_sha256 == second.model_sha256


def test_same_length_inputs_no_longer_have_identical_patterns(config):
    encoder = OpticalEncoder(config, backend="cpu")
    assert not np.array_equal(encoder.encode(["cat"]), encoder.encode(["dog"]))


def test_model_rejects_non_model_bytes(config):
    with pytest.raises(IndexFormatError):
        OpticalEncoder(config, backend="cpu", model=b"not an array archive")


def test_readout_is_trained_not_a_random_response(config):
    encoder = OpticalEncoder(config, backend="cpu")
    examples = encoder.encode(["railway replacement coupling", "blue wavelength memory"])
    readout = OpticalReadout(examples, ["railway", "optics"])
    scores = readout.predict(examples[0])
    assert scores["railway"] > 0.9
    assert scores.get("optics", 0) < 0.2


def test_scene_uses_actual_geometry_and_detector_fields(config):
    encoder = OpticalEncoder(config, backend="cpu")
    data = encoder.scene("blue memory")
    assert len(data["rgb"]) == config.detector_count
    assert len(data["sources"]) == config.source_count
    assert data["rt_cores_used"] is False
    assert data["backend"] == "cpu"


def test_explicit_backend_validation():
    with pytest.raises(ValidationError):
        Backend.select("imaginary-gpu")

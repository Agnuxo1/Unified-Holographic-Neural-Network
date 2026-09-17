"""Classical RGB ray-traced reservoir with a NumPy reference and CUDA kernels.

Each source/detector segment intersects spherical phase inclusions. Its optical
path is geometric length plus (n(lambda)-1) times the chord in each inclusion;
absorption attenuates amplitude. This straight-ray/eikonal approximation does not
bend rays or solve Maxwell's equations. CUDA executes real intersection/phase
calculations; this implementation does not invoke NVIDIA OptiX or RT cores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b, sha256
import io
import math
import zipfile
from typing import Any

import numpy as np

from .schema import EUHNNError, IndexConfig, IndexFormatError, ValidationError
from .text import feature_counts


class CUDAUnavailable(EUHNNError):
    """An explicitly requested CUDA backend cannot execute on this host."""


@dataclass
class Backend:
    """Runtime execution backend; importing EUHNN never initializes the GPU."""

    name: str
    xp: Any
    device: str | None = None
    fallback_reason: str | None = None

    @classmethod
    def select(cls, requested: str) -> "Backend":
        if requested not in ("cpu", "cuda", "auto"):
            raise ValidationError("backend must be cpu, cuda or auto.")
        if requested == "cpu":
            return cls("cpu", np)
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device")
            with cp.cuda.Device(0):
                cp.zeros(1, dtype=cp.float32).sum().item()
                prop = cp.cuda.runtime.getDeviceProperties(0)
                name = prop["name"]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
            return cls("cuda", cp, str(name))
        except Exception as exc:
            reason = type(exc).__name__
            if requested == "cuda":
                raise CUDAUnavailable(
                    "CUDA initialization failed. Check the NVIDIA driver, CUDA toolkit "
                    "and the matching cupy-cuda12x/cupy-cuda13x package. "
                    f"Failure type: {reason}. Use --backend cpu for the reference backend."
                ) from exc
            return cls("cpu", np, fallback_reason=f"CUDA unavailable ({reason}); using NumPy.")

    def host(self, value: Any) -> np.ndarray:
        """Copy a device result to a NumPy array, synchronizing GPU work."""
        return self.xp.asnumpy(value) if self.name == "cuda" else np.asarray(value)

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": self.device,
            "fallback_reason": self.fallback_reason,
            "rt_cores_used": False,
        }


@dataclass(frozen=True)
class Geometry:
    """Explicit scene geometry; rows of spheres are x, y, z, radius."""

    sources: np.ndarray
    detectors: np.ndarray
    spheres: np.ndarray
    refractive: np.ndarray
    absorption: np.ndarray

    @classmethod
    def create(cls, config: IndexConfig) -> "Geometry":
        rng = np.random.Generator(np.random.PCG64(config.seed))
        sources = np.column_stack((rng.uniform(-1.8, 1.8, (config.source_count, 2)), np.zeros(config.source_count)))
        detectors = np.column_stack(
            (rng.uniform(-2, 2, (config.detector_count, 2)), np.full(config.detector_count, 6.0))
        )
        spheres = np.column_stack(
            (
                rng.uniform(-1.5, 1.5, (config.sphere_count, 2)),
                rng.uniform(0.7, 5.3, config.sphere_count),
                rng.uniform(0.18, 0.45, config.sphere_count),
            )
        )
        refractive = rng.uniform(1.35, 1.6, (config.sphere_count, 1)) + np.array([[0, 0.035, 0.08]])
        absorption = rng.uniform(0.015, 0.09, (config.sphere_count, 3))
        geometry = cls(sources, detectors, spheres, refractive, absorption)
        geometry.validate(config)
        return geometry

    def validate(self, config: IndexConfig) -> None:
        expected = [
            (config.source_count, 3),
            (config.detector_count, 3),
            (config.sphere_count, 4),
            (config.sphere_count, 3),
            (config.sphere_count, 3),
        ]
        for array, shape in zip(self.arrays(), expected):
            if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
                raise IndexFormatError("Invalid optical scene dimensions or non-finite values.")
        if np.any(self.spheres[:, 3] <= 0) or np.any(self.absorption < 0) or np.any(self.refractive < 1):
            raise IndexFormatError("Invalid optical material parameters.")
        distance = np.linalg.norm(self.detectors[:, None] - self.sources[None], axis=2)
        if np.any(distance < 1e-8):
            raise IndexFormatError("Optical sources and detectors must be distinct.")

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.sources, self.detectors, self.spheres, self.refractive, self.absorption


def trace_numpy(geometry: Geometry, wavelengths: Sequence[float]) -> np.ndarray:
    """Reference all-segment sphere intersections and coherent transfer amplitudes."""
    delta = geometry.detectors[:, None, :] - geometry.sources[None, :, :]
    length = np.linalg.norm(delta, axis=2)
    direction = delta / length[:, :, None]
    optical_length = np.broadcast_to(length, (3, *length.shape)).copy()
    optical_depth = np.zeros_like(optical_length)
    for sphere, refractive, absorption in zip(geometry.spheres, geometry.refractive, geometry.absorption):
        offset = geometry.sources - sphere[:3]
        b = np.sum(direction * offset[None, :, :], axis=2)
        c = np.sum(offset * offset, axis=1) - sphere[3] ** 2
        discriminant = b * b - c[None, :]
        root = np.sqrt(np.maximum(discriminant, 0))
        entry, exit_ = -b - root, -b + root
        chord = np.maximum(0, np.minimum(exit_, length) - np.maximum(entry, 0))
        chord *= discriminant >= 0
        optical_length += (refractive - 1)[:, None, None] * chord
        optical_depth += absorption[:, None, None] * chord
    phase = (2 * np.pi / np.asarray(wavelengths))[:, None, None] * optical_length
    amplitude = np.exp(-optical_depth) / (1 + length[None] ** 2)
    transfer = amplitude * (np.cos(phase) + 1j * np.sin(phase))
    return transfer.astype(np.complex64)


_CUDA_RAYS = r"""
extern "C" __global__ void optical_segments(
    const double* src, const double* dst, const double* spheres,
    const double* refractive, const double* absorption, const double* wavelength,
    const int ns, const int nd, const int nm, float* result) {
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= 3 * ns * nd) return;
    const int channel = k / (ns * nd);
    const int detector = (k / ns) % nd;
    const int source = k % ns;
    const double sx=src[3*source], sy=src[3*source+1], sz=src[3*source+2];
    const double dx=dst[3*detector]-sx, dy=dst[3*detector+1]-sy, dz=dst[3*detector+2]-sz;
    const double length=sqrt(dx*dx+dy*dy+dz*dz);
    const double ux=dx/length, uy=dy/length, uz=dz/length;
    double optical=length, depth=0;
    for (int m=0; m<nm; ++m) {
        const double ox=sx-spheres[4*m], oy=sy-spheres[4*m+1], oz=sz-spheres[4*m+2];
        const double radius=spheres[4*m+3];
        const double b=ux*ox+uy*oy+uz*oz;
        const double disc=b*b-(ox*ox+oy*oy+oz*oz-radius*radius);
        if (disc < 0) continue;
        const double root=sqrt(disc);
        const double chord=fmax(0.0, fmin(-b+root,length)-fmax(-b-root,0.0));
        optical += (refractive[3*m+channel]-1.0)*chord;
        depth += absorption[3*m+channel]*chord;
    }
    const double phase=6.2831853071795864769*optical/wavelength[channel];
    const double amplitude=exp(-depth)/(1.0+length*length);
    result[2*k]=(float)(amplitude*cos(phase));
    result[2*k+1]=(float)(amplitude*sin(phase));
}
"""


def trace_cuda(geometry: Geometry, wavelengths: Sequence[float], backend: Backend) -> np.ndarray:
    """Execute the actual CUDA intersection/phase kernel and return its output."""
    if backend.name != "cuda":
        raise CUDAUnavailable("The CUDA ray tracer requires a CUDA backend.")
    cp = backend.xp
    ns, nd, nm = len(geometry.sources), len(geometry.detectors), len(geometry.spheres)
    with cp.cuda.Device(0):
        arrays = [cp.asarray(a) for a in geometry.arrays()]
        wave = cp.asarray(wavelengths, dtype=cp.float64)
        output = cp.empty((3, nd, ns), dtype=cp.complex64)
        kernel = cp.RawKernel(_CUDA_RAYS, "optical_segments", options=("--std=c++11",))
        kernel(((3 * ns * nd + 127) // 128,), (128,), (*arrays, wave, np.int32(ns), np.int32(nd), np.int32(nm), output))
        return cp.asnumpy(output)


class OpticalEncoder:
    """A fixed ray-traced reservoir with coherent fields and nonlinear detectors.

    Text is encoded as stable, content-dependent complex RGB source illumination.
    The ray transfer matrix produces a complex detector field. Real/imaginary
    field components and centered intensities form the normalized signature.
    Optional supervised learning is provided by OpticalReadout, not by claiming
    that random illumination or an animation is a trained language model.
    """

    def __init__(self, config: IndexConfig | None = None, *, backend: str = "auto", model: bytes | None = None) -> None:
        self.config = config or IndexConfig()
        self.backend = Backend.select(backend)
        if model is None:
            self.geometry = Geometry.create(self.config)
            if self.backend.name == "cuda":
                try:
                    transfer = trace_cuda(self.geometry, self.config.wavelengths, self.backend)
                except Exception as exc:
                    if backend != "auto":
                        raise CUDAUnavailable("The CUDA ray kernel could not compile or execute.") from exc
                    self.backend = Backend("cpu", np, fallback_reason="CUDA ray kernel failed; using NumPy.")
                    transfer = trace_numpy(self.geometry, self.config.wavelengths)
            else:
                transfer = trace_numpy(self.geometry, self.config.wavelengths)
            norm = np.sqrt(np.sum(np.abs(transfer) ** 2, axis=1, keepdims=True))
            self.transfer = np.asarray(transfer / np.maximum(norm, 1e-12), dtype=np.complex64)
        else:
            self.geometry, self.transfer = self._load(model)
        self._device_transfer = self.backend.xp.asarray(self.transfer)
        self.model_sha256 = sha256(self.dump()).hexdigest()

    def dump(self) -> bytes:
        """Serialize numeric arrays only. Loading never enables NumPy pickles."""
        stream = io.BytesIO()
        np.savez(
            stream,
            sources=self.geometry.sources,
            detectors=self.geometry.detectors,
            spheres=self.geometry.spheres,
            refractive=self.geometry.refractive,
            absorption=self.geometry.absorption,
            transfer=self.transfer,
        )
        return stream.getvalue()

    def _load(self, model: bytes) -> tuple[Geometry, np.ndarray]:
        if len(model) > 32 * 1024 * 1024:
            raise IndexFormatError("Optical model exceeds the 32 MiB limit.")
        names = {"sources", "detectors", "spheres", "refractive", "absorption", "transfer"}
        try:
            with zipfile.ZipFile(io.BytesIO(model)) as archive:
                entries = archive.infolist()
                if len(entries) != 6 or {i.filename for i in entries} != {n + ".npy" for n in names}:
                    raise ValueError("Unexpected model entries")
                if sum(i.file_size for i in entries) > 32 * 1024 * 1024:
                    raise ValueError("Expanded model exceeds limit")
            with np.load(io.BytesIO(model), allow_pickle=False) as stored:
                geometry = Geometry(
                    *(
                        np.array(stored[k], copy=True)
                        for k in ("sources", "detectors", "spheres", "refractive", "absorption")
                    )
                )
                geometry.validate(self.config)
                transfer = np.array(stored["transfer"], copy=True)
            shape = (3, self.config.detector_count, self.config.source_count)
            if transfer.shape != shape or transfer.dtype != np.complex64 or not np.isfinite(transfer).all():
                raise ValueError("Invalid transfer matrix")
            if not np.allclose(np.sum(np.abs(transfer) ** 2, axis=1), 1, atol=2e-5):
                raise ValueError("Transfer matrix is not column normalized")
            return geometry, transfer
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            raise IndexFormatError("The optical model is damaged or has an unsupported format.") from exc

    def illumination(self, texts: Sequence[str]) -> np.ndarray:
        """Produce deterministic complex RGB fields from word/subword content."""
        if isinstance(texts, str) or len(texts) > 256:
            raise ValidationError("encode expects a sequence of at most 256 strings.")
        output = np.zeros((len(texts), 3, self.config.source_count), dtype=np.complex64)
        for row, text in enumerate(texts):
            if not isinstance(text, str) or len(text) > 1_048_576:
                raise ValidationError("Each encoded text must contain at most 1 Mi characters.")
            features = feature_counts(text)
            for feature in sorted(features):
                hashed = blake2b(feature.encode("utf-8"), digest_size=24, person=b"EUHNN-v2-light").digest()
                weight = features[feature] / math.sqrt(6)
                for channel in range(3):
                    for replica in range(2):
                        offset = (2 * channel + replica) * 4
                        value = int.from_bytes(hashed[offset : offset + 4], "little")
                        slot = (value & 65535) % self.config.source_count
                        phase = (value >> 16) * (2 * math.pi / 65536)
                        output[row, channel, slot] += weight * complex(math.cos(phase), math.sin(phase))
        norm = np.linalg.norm(output.reshape(len(texts), -1), axis=1) if len(texts) else np.empty(0)
        output /= np.maximum(norm, 1e-12)[:, None, None]
        return output

    def fields(self, texts: Sequence[str]) -> np.ndarray:
        """Propagate illumination through the stored ray-derived transfer matrix."""
        light = self.illumination(texts)
        if not len(texts):
            return np.empty((0, 3, self.config.detector_count), dtype=np.complex64)
        xp = self.backend.xp
        field = xp.einsum("cds,bcs->bcd", self._device_transfer, xp.asarray(light))
        return self.backend.host(field).astype(np.complex64, copy=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized signatures; zero-content input gives an all-zero row."""
        field = self.fields(texts)
        if not len(texts):
            return np.empty((0, self.config.dimensions), dtype=np.float32)
        flat = field.reshape(len(texts), -1)
        norm = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        intensity = np.abs(flat) ** 2
        intensity -= np.mean(intensity, axis=1, keepdims=True)
        inorm = np.maximum(np.linalg.norm(intensity, axis=1, keepdims=True), 1e-12)
        result = np.concatenate(
            (math.sqrt(0.8) * flat.real / norm, math.sqrt(0.8) * flat.imag / norm, math.sqrt(0.2) * intensity / inorm),
            axis=1,
        )
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        return np.asarray(result, dtype=np.float32)

    def scene(self, text: str) -> dict[str, Any]:
        """Expose real model geometry and RGB detector activity for visualization."""
        field = self.fields([text])[0]
        intensity = np.abs(field) ** 2
        intensity /= max(float(intensity.max()), 1e-12)
        light = np.abs(self.illumination([text])[0]) ** 2
        strongest = np.argsort(light.sum(axis=0))[-8:]
        rays = []
        for source in strongest:
            strengths = np.sum(np.abs(self.transfer[:, :, source]) ** 2, axis=0)
            for detector in np.argsort(strengths)[-12:]:
                rays.append([int(source), int(detector)])
        return {
            "sources": self.geometry.sources.tolist(),
            "detectors": self.geometry.detectors.tolist(),
            "spheres": self.geometry.spheres.tolist(),
            "rgb": intensity.T.tolist(),
            "phase": np.angle(field).T.tolist(),
            "source_rgb": light.T.tolist(),
            "rays": rays,
            "model": "straight-ray RGB phase-delay reservoir",
            **self.backend.info(),
        }


class OpticalReadout:
    """Supervised ridge readout learned from explicit query-to-passage examples.

    A dual solve keeps training bounded by the number of teaching examples, not
    feature dimension. Outputs are regression scores, never probabilities.
    """

    def __init__(self, vectors: np.ndarray, labels: Sequence[str], regularization: float = 0.05) -> None:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or not 1 <= len(matrix) <= 256 or len(labels) != len(matrix):
            raise ValidationError("Readout training requires 1-256 labeled vectors.")
        if not np.isfinite(matrix).all() or not math.isfinite(regularization) or regularization <= 0:
            raise ValidationError("Readout parameters must be finite; regularization must be positive.")
        self.labels = sorted(set(labels))
        self.examples = matrix
        targets = np.zeros((len(matrix), len(self.labels)))
        for i, label in enumerate(labels):
            targets[i, self.labels.index(label)] = 1
        gram = matrix @ matrix.T + regularization * np.eye(len(matrix))
        self.coefficients = np.linalg.solve(gram, targets)
        self.example_labels = list(labels)

    def predict(self, vector: np.ndarray) -> dict[str, float]:
        """Predict passage affinity, guarded against remote extrapolation."""
        query = np.asarray(vector, dtype=np.float64)
        if query.shape != (self.examples.shape[1],) or not np.isfinite(query).all():
            raise ValidationError("Invalid readout query vector.")
        similarity = self.examples @ query
        scores = similarity @ self.coefficients
        return {
            label: float(np.clip(scores[i], 0, 1))
            for i, label in enumerate(self.labels)
            if max(similarity[j] for j, key in enumerate(self.example_labels) if key == label) >= 0.75
        }

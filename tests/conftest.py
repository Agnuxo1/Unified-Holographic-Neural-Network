"""Small deterministic fixtures; all data is synthetic and locally owned."""

import pytest
from euhnn import HolographicIndex, IndexConfig


@pytest.fixture
def config():
    return IndexConfig(source_count=24, detector_count=32, sphere_count=8, chunk_words=32, overlap_words=6)


@pytest.fixture
def index(tmp_path, config):
    with HolographicIndex(tmp_path / "index.sqlite", create=True, config=config, backend="cpu") as store:
        yield store
